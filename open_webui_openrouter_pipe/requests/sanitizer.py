"""Request input sanitization.

This module handles cleaning and normalizing request input before sending
to the provider API. It removes non-replayable artifacts and normalizes
tool call items to ensure consistent format.
"""

import json
import logging
from typing import Any, TYPE_CHECKING

from ..api.transforms import _filter_replayable_input_items
from ..core.context_budget import apply_replay_tool_output_budget
from ..core.utils import _clean_str
from ..integrations.anthropic import _is_anthropic_model_id

_ORPHAN_STUB_OUTPUT = (
    "[Tool output unavailable -- not recorded in conversation history.]"
)

if TYPE_CHECKING:
    from ..pipe import Pipe
    from ..api.transforms import ResponsesBody


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
        elif isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "user":
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
        if isinstance(item, dict) and isinstance(item.get("reasoning_details"), list):
            if any(_detail_unsigned_text(d) for d in item["reasoning_details"]):
                changed = True
                item = {k: v for k, v in item.items() if k != "reasoning_details"}
        out.append(item)
    return out if changed else items


def _sanitize_request_input(pipe: "Pipe", body: "ResponsesBody") -> None:
    """Remove non-replayable artifacts that may have snuck into body.input."""
    items = getattr(body, "input", None)
    if not isinstance(items, list):
        return
    original_items = items
    target_model = getattr(body, "api_model", None)
    if not (isinstance(target_model, str) and target_model.strip()):
        target_model = str(getattr(body, "model", "") or "")
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
            minimal = {
                "type": "function_call",
                "call_id": call_id,
                "name": name.strip(),
                "arguments": args,
            }
            if set(item.keys()) != set(minimal.keys()):
                changed = True
            return minimal, changed
        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if not (isinstance(call_id, str) and call_id.strip()):
                return item, False
            output = item.get("output")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
                changed = True
            minimal = {
                "type": "function_call_output",
                "call_id": call_id.strip(),
                "output": output,
            }
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

    api_model = getattr(body, "api_model", None)
    model_for_budget = api_model if isinstance(api_model, str) and api_model.strip() else str(getattr(body, "model", "") or "")
    omitted_call_ids = apply_replay_tool_output_budget(
        normalized,
        model_id=model_for_budget,
        logger=pipe.logger,
    )

    validated = _validate_tool_call_pairs(normalized, logger=pipe.logger)
    pairs_changed = validated is not normalized

    if removed or stripped_any or omitted_call_ids or pairs_changed or (items is not original_items) or (sanitized is not items):
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
    call_ids: set[str] = set()
    output_ids: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        call_id = item.get("call_id")
        if not (isinstance(call_id, str) and call_id.strip()):
            continue
        cid = call_id.strip()
        item_type = item.get("type")
        if item_type == "function_call":
            call_ids.add(cid)
        elif item_type == "function_call_output":
            output_ids.add(cid)

    orphaned_outputs = output_ids - call_ids
    orphaned_calls = call_ids - output_ids

    if not orphaned_outputs and not orphaned_calls:
        return items

    # An orphaned function_call is "interior" (historical) when a user message
    # appears after it -- meaning the conversation moved on past this tool call.
    # Frontier calls (no user message after them) are pending executions.
    interior_orphaned_calls: set[str] = set()
    if orphaned_calls:
        # Find the position of the last user message.
        last_user_pos = -1
        for i, item in enumerate(items):
            if (
                isinstance(item, dict)
                and item.get("type") == "message"
                and item.get("role") == "user"
            ):
                last_user_pos = i
        # Any orphaned function_call BEFORE the last user message is interior.
        if last_user_pos >= 0:
            for i, item in enumerate(items):
                if not isinstance(item, dict) or item.get("type") != "function_call":
                    continue
                cid = item.get("call_id")
                if not (isinstance(cid, str) and cid.strip()):
                    continue
                if cid.strip() in orphaned_calls and i < last_user_pos:
                    interior_orphaned_calls.add(cid.strip())

    if orphaned_outputs:
        logger.warning(
            "Dropping %d orphaned function_call_output item(s) with no matching function_call: call_ids=%s",
            len(orphaned_outputs),
            sorted(orphaned_outputs),
        )
    if interior_orphaned_calls:
        logger.warning(
            "Synthesising stub function_call_output for %d orphaned function_call item(s): call_ids=%s",
            len(interior_orphaned_calls),
            sorted(interior_orphaned_calls),
        )

    result: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            result.append(item)
            continue
        item_type = item.get("type")
        raw_cid = item.get("call_id")
        cid = raw_cid.strip() if isinstance(raw_cid, str) else ""

        if item_type == "function_call_output" and cid in orphaned_outputs:
            continue

        result.append(item)

        if item_type == "function_call" and cid in interior_orphaned_calls:
            result.append({
                "type": "function_call_output",
                "call_id": cid,
                "output": _ORPHAN_STUB_OUTPUT,
            })

    return result
