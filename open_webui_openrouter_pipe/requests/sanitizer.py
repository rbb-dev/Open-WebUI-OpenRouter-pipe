"""Request input sanitization.

This module handles cleaning and normalizing request input before sending
to the provider API. It removes non-replayable artifacts and normalizes
tool call items to ensure consistent format.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..api.transforms import _filter_replayable_input_items
from ..core.context_budget import (
    BudgetOutcome,
    apply_replay_tool_output_budget,
    effective_chars_per_token,
)
from ..core.url_scheme import loggable_link
from ..core.utils import (
    TOOL_CALL_STATUSES,
    _clean_str,
    is_picture_output,
    is_text_part_output,
    opens_a_turn,
    strip_hidden_marker_lines,
    tool_output_text_and_pictures,
)
from ..integrations.anthropic import _is_anthropic_model_id

_ORPHAN_STUB_OUTPUT = (
    "[Tool output unavailable -- not recorded in conversation history.]"
)

if TYPE_CHECKING:
    from ..api.transforms import ResponsesBody
    from ..pipe import Pipe


def _without_hidden_marker_lines(output: Any) -> Any:
    if isinstance(output, str):
        return strip_hidden_marker_lines(output)
    if is_picture_output(output):
        return [
            {**part, "text": strip_hidden_marker_lines(part["text"])}
            if part.get("type") == "input_text" and isinstance(part.get("text"), str) else part
            for part in output
        ]
    return output


def _normalise_tool_call_id(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _reasoning_item_unsigned(item: dict[str, Any]) -> bool:
    """True for a /responses reasoning item that is plaintext thinking with no
    signature and no encrypted payload -- unreplayable to Anthropic."""
    if _clean_str(item.get("signature")) or _clean_str(item.get("encrypted_content")):
        return False
    content = item.get("content")
    if not isinstance(content, list):
        return False
    has_text = False
    for part in content:
        if not isinstance(part, dict):
            continue
        if _clean_str(part.get("signature")) or _clean_str(part.get("encrypted_content")):
            return False
        if part.get("type") == "reasoning_text" and isinstance(part.get("text"), str) and part["text"].strip():
            has_text = True
    return has_text


def _detail_unsigned_text(detail: Any) -> bool:
    """True for a reasoning.text detail that has text but no signature."""
    if not isinstance(detail, dict) or detail.get("type") != "reasoning.text":
        return False
    if _clean_str(detail.get("signature")):
        return False
    text = detail.get("text")
    return isinstance(text, str) and bool(text.strip())


def _strip_unreplayable_anthropic_reasoning(items: list[Any]) -> list[Any]:
    """Drop thinking that is unreplayable to Anthropic (plaintext with no signature
    and no encrypted payload); caller must gate on an Anthropic target. The provider
    requires a turn's whole reasoning sequence to be replayed intact and rejects a
    partially modified one, so removal is all-or-nothing per turn: /responses reasoning
    items are grouped into turn spans -- delimited only by USER messages (the real turn
    boundary, matching the reinterleave/tool-pairing logic), since within one assistant
    turn the reasoning is split across tool items AND assistant text-chunk messages --
    and every reasoning item in a span is dropped when ANY item in that span is
    unreplayable. A message's reasoning_details is likewise dropped whole when ANY entry
    is an unsigned reasoning.text."""
    drop_idx: set[int] = set()
    span: list[int] = []
    tainted = False
    for idx, item in enumerate(items):
        if isinstance(item, dict) and item.get("type") == "reasoning":
            span.append(idx)
            if _reasoning_item_unsigned(item):
                tainted = True
        elif opens_a_turn(items, idx):
            if tainted:
                drop_idx.update(span)
            span = []
            tainted = False
    if tainted:
        drop_idx.update(span)

    out: list[Any] = []
    changed = bool(drop_idx)
    for idx, item in enumerate(items):
        if idx in drop_idx:
            continue
        if (
            isinstance(item, dict)
            and isinstance(item.get("reasoning_details"), list)
            and any(_detail_unsigned_text(d) for d in item["reasoning_details"])
        ):
            changed = True
            item = {k: v for k, v in item.items() if k != "reasoning_details"}
        out.append(item)
    return out if changed else items


def budget_model_id(body: Any) -> str:
    api_model = getattr(body, "api_model", None)
    if isinstance(api_model, str) and api_model.strip():
        return api_model
    return str(getattr(body, "model", "") or "")


def _request_overhead_chars(body: Any) -> int:
    try:
        rest = body.model_dump(exclude_none=True, exclude={"input"})
        return len(json.dumps(rest, ensure_ascii=False, default=str))
    except (TypeError, ValueError, AttributeError):
        return 0


_pending_status_emissions: set[asyncio.Task[Any]] = set()


def _inline_head_is_forwardable(url: str) -> bool:
    from .transformer import _is_forwardable_image_type

    text = url.strip()
    if not text.lower().startswith("data:"):
        return True
    return _is_forwardable_image_type(text[5:].split(",", 1)[0])


def _gate_tool_pictures(
    item: dict[str, Any], logger: logging.Logger, *,
    max_inline_bytes: int, allow_insecure: Callable[[str], bool],
    verdicts: dict[str, bool | None] | None = None,
    refusals: list[tuple[str, str, str]] | None = None,
) -> tuple[dict[str, Any], bool]:
    from .transformer import _tool_picture_gate, _tool_picture_verdict_gate

    if item.get("type") != "function_call_output":
        return item, False
    output = item.get("output")
    parts = output if isinstance(output, list) else []
    if not is_picture_output(parts):
        return item, False
    urls = [
        str(part.get("image_url"))
        for part in parts
        if isinstance(part, dict) and part.get("type") == "input_image" and part.get("image_url")
    ]
    kept, refused = _tool_picture_gate(
        urls, max_inline_bytes=max_inline_bytes, allow_insecure=allow_insecure,
    )
    kept, unfetchable = _tool_picture_verdict_gate(kept, verdicts)
    refused = [*refused, *unfetchable]
    unforwardable = [url for url in kept if not _inline_head_is_forwardable(url)]
    if unforwardable:
        refused.extend(
            (url, "not identifiable as an image", "inline_untyped") for url in unforwardable
        )
        kept = [url for url in kept if url not in set(unforwardable)]
    if not refused:
        return item, False
    if refusals is not None:
        refusals.extend(refused)
    for url, reason, cause in refused:
        logger.warning(
            "Not forwarding a tool's picture (%s): %s [cause=%s]",
            loggable_link(url), reason, cause,
        )
    survivors = set(kept)
    rebuilt = [
        part
        for part in parts
        if not (
            isinstance(part, dict)
            and part.get("type") == "input_image"
            and part.get("image_url") not in survivors
        )
    ]
    if rebuilt == parts:
        return item, False
    return {**item, "output": rebuilt}, True


def _sanitize_request_input(
    pipe: Pipe, body: ResponsesBody, *,
    verdicts: dict[str, bool | None] | None = None,
    event_emitter: Any = None,
) -> BudgetOutcome | None:
    """Remove non-replayable artifacts that may have snuck into body.input."""
    picture_refusals: list[tuple[str, str, str]] = []
    items = getattr(body, "input", None)
    if not isinstance(items, list):
        return None
    original_items = items
    target_model = budget_model_id(body)
    if _is_anthropic_model_id(target_model):
        items = _strip_unreplayable_anthropic_reasoning(items)
    sanitized = _filter_replayable_input_items(items, logger=pipe.logger)
    removed = len(items) - len(sanitized)

    def _strip_tool_item_extras(item: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Return a minimal, portable /responses input shape for tool items."""
        changed = False
        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id")
            if not (isinstance(call_id, str) and call_id.strip()):
                candidate = item.get("id")
                if isinstance(candidate, str) and candidate.strip():
                    call_id = candidate.strip()
                    changed = True
            name = item.get("name")
            if not (isinstance(name, str) and name.strip()):
                return item, False
            args = item.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args or {}, ensure_ascii=False)
                changed = True
            minimal: dict[str, Any] = {
                "type": "function_call",
                "call_id": _normalise_tool_call_id(call_id),
                "name": name.strip(),
                "arguments": args,
            }
            if set(item.keys()) != set(minimal.keys()):
                changed = True
            return minimal, changed
        if item_type == "function_call_output":
            item, gated = _gate_tool_pictures(
                item, pipe.logger,
                max_inline_bytes=pipe.valves.BASE64_MAX_SIZE_MB * 1024 * 1024,
                allow_insecure=pipe._multimodal_handler._is_insecure_http_allowed,
                verdicts=verdicts,
                refusals=picture_refusals,
            )
            changed = gated
            call_id = item.get("call_id")
            if not (isinstance(call_id, str) and call_id.strip()):
                return item, gated
            output = item.get("output")
            if isinstance(output, list) and not output:
                output = ""
                changed = True
            elif is_text_part_output(output):
                output = tool_output_text_and_pictures(output)[0]
                changed = True
            elif not isinstance(output, str) and not is_picture_output(output):
                output = json.dumps(output, ensure_ascii=False)
                changed = True
            cleaned = _without_hidden_marker_lines(output)
            if cleaned != output:
                output = cleaned
                changed = True
            minimal: dict[str, Any] = {
                "type": "function_call_output",
                "call_id": _normalise_tool_call_id(call_id),
                "output": output,
            }
            reported_status = item.get("status")
            if reported_status in TOOL_CALL_STATUSES:
                minimal["status"] = reported_status
            if set(item.keys()) != set(minimal.keys()):
                changed = True
            return minimal, changed
        return item, False

    stripped_any = False
    normalized: list[dict[str, Any]] = []
    for entry in sanitized:
        if not isinstance(entry, dict):
            normalized.append(entry)
            continue
        stripped, changed = _strip_tool_item_extras(entry)
        if changed:
            stripped_any = True
        normalized.append(stripped)

    model_for_budget = budget_model_id(body)
    budget = apply_replay_tool_output_budget(
        normalized,
        model_id=model_for_budget,
        logger=pipe.logger,
        referenced_sizes=getattr(body, "input_file_sizes", None),
        reserved_output_tokens=getattr(body, "max_output_tokens", None),
        fixed_overhead_chars=_request_overhead_chars(body),
        chars_per_token=effective_chars_per_token(
            getattr(body, "budget_chars_per_token", None), model_for_budget
        ),
    )
    omitted_call_ids = budget.omitted_call_ids

    validated = _validate_tool_call_pairs(normalized, logger=pipe.logger)
    pairs_changed = validated is not normalized

    if removed or stripped_any or omitted_call_ids or pairs_changed or (items is not original_items) or (sanitized is not items):
        if items is not original_items:
            pipe.logger.debug("Sanitized provider input: stripped unreplayable reasoning (Anthropic).")
        if removed:
            pipe.logger.debug(
                "Sanitized provider input: removed %d non-replayable artifact(s).",
                removed,
            )
        if stripped_any:
            pipe.logger.debug("Sanitized provider input: stripped extra tool item fields.")
        if omitted_call_ids:
            pipe.logger.debug(
                "Sanitized provider input: omitted %d replayed tool output(s) by context budget.",
                len(omitted_call_ids),
            )
        body.input = validated
    _report_refused_tool_pictures(pipe, picture_refusals, event_emitter)
    return budget


def _report_refused_tool_pictures(
    pipe: Pipe,
    refusals: list[tuple[str, str, str]],
    event_emitter: Any,
) -> None:
    from .transformer import _tool_picture_notice

    if not refusals or event_emitter is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(
        pipe._event_emitter_handler._emit_status(
            event_emitter, _tool_picture_notice(refusals), done=False,
        )
    )
    _pending_status_emissions.add(task)
    task.add_done_callback(_pending_status_emissions.discard)


def _validate_tool_call_pairs(
    items: list[Any],
    *,
    logger: logging.Logger,
) -> list[Any]:
    """Ensure function_call / function_call_output items are properly paired.

    * Orphaned function_call_output (no matching function_call): dropped.
    * Orphaned function_call (no matching function_call_output): a stub output
      is synthesised immediately after the call -- but only for *interior*
      orphans.  A function_call is considered "interior" (historical) when a
      user message appears after it in the input array, meaning a new
      conversation turn started and the call should have had an output.
      Frontier function_call items (no user message after them) are left
      alone because they represent pending tool executions.
    """
    call_counts: Counter[str] = Counter()
    output_counts: Counter[str] = Counter()
    positions: dict[str, list[int]] = {}

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        call_id = item.get("call_id")
        if not (isinstance(call_id, str) and call_id.strip()):
            continue
        cid = call_id
        item_type = item.get("type")
        if item_type == "function_call":
            call_counts[cid] += 1
            positions.setdefault(cid, []).append(i)
        elif item_type == "function_call_output":
            output_counts[cid] += 1

    orphaned_outputs = {cid for cid in output_counts if not call_counts[cid]}
    orphaned_calls = {cid for cid in call_counts if not output_counts[cid]}
    surplus_outputs = {
        cid: output_counts[cid] - call_counts[cid]
        for cid in output_counts
        if call_counts[cid] and output_counts[cid] > call_counts[cid]
    }

    last_user_pos = -1
    for i, item in enumerate(items):
        if (
            isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role") == "user"
        ):
            last_user_pos = i

    starved_occurrences: set[tuple[int, str]] = set()
    seen_calls: Counter[str] = Counter()
    for i, item in enumerate(items):
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        cid = item.get("call_id")
        if not (isinstance(cid, str) and cid.strip()):
            continue
        seen_calls[cid] += 1
        if last_user_pos >= 0 and i < last_user_pos and (
            cid in orphaned_calls or output_counts[cid] < seen_calls[cid]
        ):
            starved_occurrences.add((i, cid))

    interior_orphaned_calls: set[str] = {cid for _i, cid in starved_occurrences}

    if not orphaned_outputs and not orphaned_calls and not surplus_outputs and not starved_occurrences:
        return items

    stub_anchors: dict[str, int] = {}
    seen_outputs: Counter[str] = Counter()
    for i, item in enumerate(items):
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        cid = item.get("call_id")
        if not (isinstance(cid, str) and cid.strip()):
            continue
        if cid in orphaned_outputs:
            continue
        if cid in surplus_outputs and seen_outputs[cid] >= call_counts[cid]:
            continue
        seen_outputs[cid] += 1
        pos = positions.get(cid)
        stub_anchors[cid] = pos[-1] if pos and pos[-1] > i else i

    by_anchor: dict[int, list[str]] = {}
    for starved_index, starved_cid in sorted(starved_occurrences):
        by_anchor.setdefault(stub_anchors.get(starved_cid, starved_index), []).append(starved_cid)

    if orphaned_outputs:
        logger.warning(
            "Dropping %d orphaned function_call_output item(s) with no matching function_call: call_ids=%s",
            len(orphaned_outputs),
            sorted(orphaned_outputs),
        )
    if surplus_outputs:
        logger.warning(
            "Dropping %d surplus function_call_output item(s) beyond the calls that carry their id: %s",
            sum(surplus_outputs.values()),
            sorted(surplus_outputs),
        )
    if interior_orphaned_calls:
        logger.warning(
            "Synthesising stub function_call_output for %d orphaned function_call item(s): call_ids=%s",
            len(interior_orphaned_calls),
            sorted(interior_orphaned_calls),
        )

    result: list[Any] = []
    emitted: Counter[str] = Counter()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            result.append(item)
            continue
        item_type = item.get("type")
        raw_cid = item.get("call_id")
        cid = raw_cid if isinstance(raw_cid, str) else ""

        if item_type == "function_call_output" and cid in orphaned_outputs:
            continue

        if item_type == "function_call_output" and cid in surplus_outputs:
            if emitted[cid] >= call_counts[cid]:
                continue
            emitted[cid] += 1

        result.append(item)

        for starved_cid in by_anchor.pop(i, ()):
            result.append({
                "type": "function_call_output",
                "call_id": starved_cid,
                "output": _ORPHAN_STUB_OUTPUT,
                "status": "incomplete",
            })

    return result
