"""Context budget estimation and dynamic tool output omission helpers."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..models.registry import ModelFamily
from .utils import TOOL_CALL_STATUSES, _coerce_positive_int

logger = logging.getLogger(__name__)

_FALLBACK_PROMPT_LIMIT_TOKENS = 128_000
_CHARS_PER_TOKEN_HEURISTIC = 4
_MAX_DEFAULT_OUTPUT_SHARE_DIVISOR = 2
_MIN_MEASURED_INPUT_TOKENS = 1_000
_MIN_MEASURED_CHARS_PER_TOKEN = 0.25
_AUDIO_BYTES_PER_TOKEN = 500
_VIDEO_BYTES_PER_TOKEN = 380
_DOCUMENT_BYTES_PER_TOKEN = 500
_PAYLOAD_HEAD_CHARS = 512
_TEXTUAL_MEDIA_TYPES = frozenset(
    {
        "application/json",
        "application/xml",
        "application/x-ndjson",
        "application/csv",
        "application/javascript",
        "application/yaml",
        "application/x-yaml",
        "application/toml",
        "application/sql",
        "application/graphql",
    }
)


_UNTYPED_DECLARATIONS = frozenset(
    {
        "application/octet-stream",
        "binary/octet-stream",
        "application/binary",
        "application/download",
        "application/force-download",
        "application/x-binary",
        "application/unknown",
        "*/*",
    }
)

_TEXTUAL_SUFFIXES = (
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xml",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log", ".sql", ".py", ".js", ".ts",
    ".html", ".htm", ".css", ".rst", ".srt", ".vtt",
)

_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
_BINARY_DOCUMENT_MAGICS = (b"%PDF-",)
_CONTROL_RATIO_CAP = 20
_ALLOWED_CONTROL = "\t\n\r\x1b"


def _decode_window(window: str, *, at_tail: bool = False) -> bytes | None:
    if not window:
        return None
    cleaned = "".join(window.split())
    if at_tail:
        cleaned += "=" * (-len(cleaned) % 4)
    else:
        cleaned = cleaned[: len(cleaned) // 4 * 4]
    try:
        decoded = base64.b64decode(cleaned)
    except (binascii.Error, ValueError):
        return None
    return decoded or None


def _window_reads_as_text(window: str, *, at_tail: bool) -> bool | None:
    decoded = _decode_window(window, at_tail=at_tail)
    if decoded is None:
        return None
    if b"\x00" in decoded and not at_tail:
        return False
    for trim in range(4):
        chunk = decoded[trim:] if at_tail else decoded[: len(decoded) - trim]
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not text:
            return None
        control = sum(1 for ch in text if ord(ch) < 32 and ch not in _ALLOWED_CONTROL)
        return control * _CONTROL_RATIO_CAP < len(text)
    return False if b"\x00" in decoded else None


_WINDOW_SCAN_CHARS = _PAYLOAD_HEAD_CHARS * 4
_DATA_URL_PREFIX_CHARS = 1024


def _payload_windows(raw: str) -> tuple[str, ...]:
    probe = raw[:_WINDOW_SCAN_CHARS]
    if probe == "".join(probe.split()):
        cleaned = "".join(raw.split())
        head = cleaned[:_PAYLOAD_HEAD_CHARS]
        start = max(0, len(cleaned) - _PAYLOAD_HEAD_CHARS)
        start -= start % 4
        tail = cleaned[start:]
    else:
        cleaned_len = len(raw) - sum(raw.count(char) for char in " \t\n\r\v\f")
        head = "".join(probe.split())[:_PAYLOAD_HEAD_CHARS]
        start = max(0, cleaned_len - _PAYLOAD_HEAD_CHARS)
        start -= start % 4
        take = cleaned_len - start
        tail = "".join(raw[-_WINDOW_SCAN_CHARS:].split())[-take:] if take else ""
    if not tail or tail == head:
        return (head,)
    return (head, tail)


def _reads_as_text(
    media_type: str, windows: tuple[str, ...] | None, filename: str = ""
) -> bool:
    if media_type.startswith("text/"):
        return True
    if media_type in _TEXTUAL_MEDIA_TYPES or media_type.endswith(("+json", "+xml")):
        return True
    if media_type in _UNTYPED_DECLARATIONS or not media_type:
        if windows:
            first = _decode_window(windows[0])
            if first is not None and first.startswith(_BINARY_DOCUMENT_MAGICS):
                return False
            if first is not None and first.startswith(_UTF16_BOMS):
                return True
            head_verdict = _window_reads_as_text(windows[0], at_tail=False)
            tail_verdicts = [
                verdict
                for window in windows[1:]
                if (verdict := _window_reads_as_text(window, at_tail=True)) is not None
            ]
            if head_verdict is False or False in tail_verdicts:
                return False
            if head_verdict is True:
                return True
        return filename.strip().lower().endswith(_TEXTUAL_SUFFIXES)
    return False


def _document_tokens(
    n: int, media_type: str, windows: tuple[str, ...] | None = None, filename: str = ""
) -> int:
    if _reads_as_text(media_type, windows, filename):
        return n // _CHARS_PER_TOKEN_HEURISTIC
    return n // _DOCUMENT_BYTES_PER_TOKEN


_OPAQUE_BLOCK_PAYLOADS: dict[str, tuple[tuple[str, ...], Callable[..., int]]] = {
    "input_image": (("image_url",), lambda _n, _t, _h=None, _f="": 1_700),
    "image_url": (("image_url",), lambda _n, _t, _h=None, _f="": 1_700),
    "input_audio": (
        ("input_audio",),
        lambda n, _t, _h=None, _f="": n // _AUDIO_BYTES_PER_TOKEN,
    ),
    "input_file": (("file_id", "file_data", "file_url"), _document_tokens),
    "video_url": (
        ("video_url",),
        lambda n, _t, _h=None, _f="": n // _VIDEO_BYTES_PER_TOKEN,
    ),
}
_LIVE_OMISSION_PREFIX = "[Tool result omitted due to context budget."
_REPLAY_OMISSION_PREFIX = "[Replayed tool result omitted due to context budget."


def _finite_positive(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def measure_chars_per_token(
    *, metered_chars: int, usage: Any, model_id: str | None = None
) -> float | None:
    if not isinstance(usage, Mapping) or metered_chars <= 0:
        return None
    input_tokens = _finite_positive(usage.get("input_tokens"))
    if input_tokens is None or input_tokens < _MIN_MEASURED_INPUT_TOKENS:
        return None
    window = model_context_length(model_id) if model_id else None
    if window is not None and input_tokens > window:
        return None
    output_tokens = _finite_positive(usage.get("output_tokens"))
    total_tokens = _finite_positive(usage.get("total_tokens"))
    if (
        output_tokens is not None
        and total_tokens is not None
        and abs(input_tokens + output_tokens - total_tokens) > 1.0
    ):
        return None
    ratio = metered_chars / input_tokens
    return ratio if ratio >= _MIN_MEASURED_CHARS_PER_TOKEN else None


def record_chars_per_token(store: Any, model_id: str, ratio: float) -> None:
    if not isinstance(store, dict):
        return
    key = model_id or ""
    previous = store.get(key)
    store[key] = ratio if previous is None else min(previous, ratio)


def effective_chars_per_token(store: Any, model_id: str) -> float | None:
    if not isinstance(store, Mapping) or not store:
        return None
    ratio = store.get(model_id or "")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio <= 0:
        return None
    return min(float(ratio), float(_CHARS_PER_TOKEN_HEURISTIC))


def model_context_length(model_id: str) -> int | None:
    spec = ModelFamily._lookup_spec(model_id)
    if not isinstance(spec, dict):
        return None
    full_model = spec.get("full_model")
    full = full_model if isinstance(full_model, dict) else {}
    top_provider = full.get("top_provider")
    provider = top_provider if isinstance(top_provider, dict) else {}
    for source in (full, spec, provider):
        value = _coerce_positive_int(source.get("context_length"))
        if value is not None:
            return value
    return None


def default_output_reservation(model_id: str) -> int | None:
    declared = _coerce_positive_int(ModelFamily.max_completion_tokens(model_id))
    if declared is None:
        return None
    window = model_context_length(model_id)
    if window is None:
        return declared
    return max(min(declared, window // _MAX_DEFAULT_OUTPUT_SHARE_DIVISOR), 1)


def compute_prompt_limit_tokens(
    model_id: str, *, reserved_output_tokens: int | None = None
) -> int:
    """Estimate max prompt tokens using model metadata with safe fallbacks.

    """
    spec = ModelFamily._lookup_spec(model_id)
    if not isinstance(spec, dict):
        return _FALLBACK_PROMPT_LIMIT_TOKENS

    full_model = spec.get("full_model")
    full = full_model if isinstance(full_model, dict) else {}

    max_prompt = _coerce_positive_int(full.get("max_prompt_tokens"))
    if max_prompt is None:
        max_prompt = _coerce_positive_int(spec.get("max_prompt_tokens"))
    if max_prompt is not None:
        return max_prompt

    context_length = model_context_length(model_id)
    if context_length is None:
        return _FALLBACK_PROMPT_LIMIT_TOKENS

    reserved = _coerce_positive_int(reserved_output_tokens)
    if reserved is not None and reserved < context_length:
        return context_length - reserved

    return context_length


_NAMES_CONTENT: tuple[int, str, tuple[str, ...] | None] = (0, "", None)
_LOCATOR_RE = re.compile(r"^(?!data:)[A-Za-z][A-Za-z0-9+.\-]*:")


def _payload_bytes(value: Any) -> tuple[int, str, tuple[str, ...] | None] | None:
    if isinstance(value, str):
        media_type = ""
        payload_head: tuple[str, ...] | None = None
        if ";base64," in value[:_DATA_URL_PREFIX_CHARS]:
            prefix, raw = value.split(";base64,", 1)
            if prefix.startswith("data:"):
                media_type = prefix[len("data:") :].split(";", 1)[0].strip().lower()
                payload_head = _payload_windows(raw)
        elif _LOCATOR_RE.match(value):
            return _NAMES_CONTENT
        else:
            raw = value
        size = len(raw) * 3 // 4
        return (size, media_type, payload_head) if size else None
    if isinstance(value, dict):
        for sub_key in ("url", "data"):
            if sub_key in value:
                return _payload_bytes(value[sub_key])
    return None


def _referenced_bytes(
    value: Any, referenced_sizes: Mapping[str, tuple[int, str, str]] | None
) -> tuple[int, str, tuple[str, ...] | None] | None:
    if not referenced_sizes or not isinstance(value, str):
        return None
    entry = referenced_sizes.get(value)
    if entry is None:
        return None
    size, media_type, _name = entry
    return (size, media_type, None) if size > 0 else None


def _referenced_filename(
    value: Any, referenced_sizes: Mapping[str, tuple[int, str, str]] | None
) -> str:
    if not referenced_sizes or not isinstance(value, str):
        return ""
    entry = referenced_sizes.get(value)
    return entry[2] if entry is not None else ""


def _budget_shape(
    value: Any,
    referenced_sizes: Mapping[str, tuple[int, str, str]] | None = None,
    charges: list[int] | None = None,
) -> Any:
    if isinstance(value, dict):
        block_type = value.get("type")
        spec = _OPAQUE_BLOCK_PAYLOADS.get(block_type) if isinstance(block_type, str) else None
        if spec is not None:
            payload_keys, rate = spec
            sizes: dict[str, tuple[int, str, tuple[str, ...] | None]] = {}
            for key in payload_keys:
                if key not in value:
                    continue
                measured = _referenced_bytes(
                    value[key], referenced_sizes
                ) or _payload_bytes(value[key])
                if measured is not None:
                    sizes[key] = measured
            resolved_first = payload_keys[0]
            charged_key = (
                resolved_first
                if resolved_first in sizes
                else max(sizes, key=lambda k: sizes[k][0])
                if sizes
                else None
            )
            shaped: dict[str, Any] = {}
            for key, item in value.items():
                if key == charged_key:
                    declared = value.get("filename")
                    filename = (
                        declared
                        if isinstance(declared, str) and declared.strip()
                        else _referenced_filename(item, referenced_sizes)
                    )
                    tokens = rate(*sizes[key], filename)
                    charged_chars = tokens * _CHARS_PER_TOKEN_HEURISTIC
                    names_content = sizes[key] is _NAMES_CONTENT
                    if charges is None:
                        shaped[key] = ("x" * charged_chars) + (item if names_content else "")
                    else:
                        charges.append(charged_chars)
                        shaped[key] = item if names_content else ""
                elif key in sizes:
                    shaped[key] = ""
                else:
                    shaped[key] = _budget_shape(item, referenced_sizes, charges)
            return shaped
        return {
            key: _budget_shape(item, referenced_sizes, charges)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_budget_shape(item, referenced_sizes, charges) for item in value]
    return value


def _baseline_without_tool_outputs(items: Any) -> Any:
    if not isinstance(items, list):
        return items
    baseline: list[Any] = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call":
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                arguments = item.get("arguments")
                baseline.append(
                    {
                        "type": "function_call",
                        "call_id": item.get("call_id") or item.get("id"),
                        "name": name.strip(),
                        "arguments": arguments
                        if isinstance(arguments, str)
                        else json.dumps(arguments or {}, ensure_ascii=False),
                    }
                )
            else:
                baseline.append(item)
        elif isinstance(item, dict) and item.get("type") == "function_call_output":
            shaped: dict[str, Any] = {
                "type": "function_call_output",
                "call_id": item.get("call_id"),
                "output": "",
            }
            status = item.get("status")
            if status in TOOL_CALL_STATUSES:
                shaped["status"] = status
            baseline.append(shaped)
        else:
            baseline.append(item)
    return baseline


def _output_text_of(item: Any) -> str:
    raw = item.get("output") if isinstance(item, dict) else None
    if isinstance(raw, str):
        return raw
    return "" if raw is None else str(raw)


def _wire_chars(text: str) -> int:
    try:
        return len(json.dumps(text, ensure_ascii=False)) - 2
    except (TypeError, ValueError):
        return len(text)


def _output_floor_chars(
    text: str, build_stub: Callable[..., str] | None = None
) -> int:
    if is_tool_omission_stub(text):
        return _wire_chars(text)
    builder = build_stub or build_replayed_tool_omission_stub
    stub = builder(result_chars=len(text), remaining_tokens=0)
    return min(_wire_chars(text), _wire_chars(stub))


def estimate_serialized_chars(
    value: Any,
    *,
    referenced_sizes: Mapping[str, tuple[int, str, str]] | None = None,
) -> int:
    """Estimate payload size by serialized character count."""
    charges: list[int] = []
    try:
        shaped = _budget_shape(value, referenced_sizes, charges)
    except RecursionError:
        shaped, charges = value, []
    try:
        return len(json.dumps(shaped, ensure_ascii=False)) + sum(charges)
    except (RecursionError, TypeError, ValueError):
        return len(str(shaped)) + sum(charges)


_OMISSION_STUB_RE = re.compile(
    r"\[Tool result omitted due to context budget\. "
    r"Result size: [\d,]+ chars \(~[\d,]+ tokens\)\. "
    r"Remaining prompt budget: ~[\d,]+ tokens\. "
    r"If needed, re-run the tool with a narrower query or explicit result limit\.\]"
    r"|"
    r"\[Replayed tool result omitted due to context budget\. "
    r"Result size: [\d,]+ chars \(~[\d,]+ tokens\)\. "
    r"Remaining prompt budget: ~[\d,]+ tokens\.\]"
)


def is_tool_omission_stub(text: str) -> bool:
    """Return True when text is already a tool omission stub."""
    if not isinstance(text, str):
        return False
    return _OMISSION_STUB_RE.fullmatch(text.strip()) is not None


def build_live_tool_omission_stub(result_chars: int, remaining_tokens: int) -> str:
    """Build a model-facing omission stub for live tool execution."""
    estimated_tokens = result_chars // _CHARS_PER_TOKEN_HEURISTIC
    return (
        "[Tool result omitted due to context budget. "
        f"Result size: {result_chars:,} chars (~{estimated_tokens:,} tokens). "
        f"Remaining prompt budget: ~{remaining_tokens:,} tokens. "
        "If needed, re-run the tool with a narrower query or explicit result limit.]"
    )


def build_replayed_tool_omission_stub(result_chars: int, remaining_tokens: int) -> str:
    """Build a model-facing omission stub for replayed tool artifacts."""
    estimated_tokens = result_chars // _CHARS_PER_TOKEN_HEURISTIC
    return (
        "[Replayed tool result omitted due to context budget. "
        f"Result size: {result_chars:,} chars (~{estimated_tokens:,} tokens). "
        f"Remaining prompt budget: ~{remaining_tokens:,} tokens.]"
    )


@dataclass(frozen=True)
class BudgetOutcome:
    omitted_call_ids: frozenset[str]
    futile: bool = False
    irreducible_chars: int = 0
    limit_chars: int = 0
    chars_per_token: float | None = None
    limit_tokens: int = 0


def omitted_tool_names(outcome: BudgetOutcome, items: Any) -> list[str]:
    named = {
        item.get("call_id"): str(item.get("name") or item.get("call_id"))
        for item in (items if isinstance(items, list) else [])
        if isinstance(item, dict) and item.get("type") == "function_call"
    }
    return sorted({named.get(call_id, call_id) for call_id in outcome.omitted_call_ids})


def _notice_tokens(outcome: BudgetOutcome, chars: int) -> int:
    if outcome.chars_per_token is None or outcome.limit_chars <= 0:
        return chars // _CHARS_PER_TOKEN_HEURISTIC
    return -(-chars * (outcome.limit_tokens or 1) // outcome.limit_chars)


def build_futility_notice(outcome: BudgetOutcome) -> str:
    return (
        "This conversation needs about "
        f"{_notice_tokens(outcome, outcome.irreducible_chars)} tokens even "
        "with every tool result reduced to a placeholder, more than this model's "
        f"{outcome.limit_tokens or outcome.limit_chars // _CHARS_PER_TOKEN_HEURISTIC}-token limit. "
        "Results were trimmed as far as they go and the request may still be too "
        "large. Shorten or remove the largest message or attachment, or start a "
        "new chat."
    )


_LIVE_FUTILE_MESSAGE = (
    "The live request needs ~%d chars even with every tool result reduced to a "
    "placeholder, against a ~%d char limit. Trimming proceeds anyway; the request may "
    "still be rejected."
)
_REPLAY_FUTILE_MESSAGE = (
    "The replay request needs ~%d chars even with every tool result reduced to a "
    "placeholder, against a ~%d char limit. Trimming proceeds anyway; the request may "
    "still be rejected."
)


def _apply_tool_output_budget(
    items: list[Any],
    *,
    model_id: str,
    live_from: int | None,
    futile_message: str,
    logger: logging.Logger = logger,
    referenced_sizes: Mapping[str, tuple[int, str, str]] | None = None,
    reserved_output_tokens: int | None = None,
    fixed_overhead_chars: int = 0,
    chars_per_token: float | None = None,
) -> BudgetOutcome:
    omitted_call_ids: set[str] = set()
    if not items:
        return BudgetOutcome(frozenset())

    prompt_limit_tokens = compute_prompt_limit_tokens(
        model_id, reserved_output_tokens=reserved_output_tokens
    )
    prompt_limit_chars = max(prompt_limit_tokens * _CHARS_PER_TOKEN_HEURISTIC, 0)
    measured_ratio = (
        chars_per_token
        if chars_per_token is not None
        and 0 < chars_per_token < _CHARS_PER_TOKEN_HEURISTIC
        else None
    )
    if measured_ratio is not None:
        prompt_limit_chars = max(int(prompt_limit_tokens * measured_ratio), 0)

    def _is_live(index: int) -> bool:
        return live_from is not None and index >= live_from

    def _builder(index: int) -> Callable[..., str]:
        if _is_live(index):
            return build_live_tool_omission_stub
        return build_replayed_tool_omission_stub

    fixed_chars = estimate_serialized_chars(
        _baseline_without_tool_outputs(items), referenced_sizes=referenced_sizes
    )
    irreducible_chars = fixed_chars + max(fixed_overhead_chars, 0) + sum(
        _output_floor_chars(_output_text_of(item), _builder(index))
        for index, item in enumerate(items)
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )
    futile = bool(prompt_limit_chars) and irreducible_chars >= prompt_limit_chars
    if futile:
        logger.warning(futile_message, irreducible_chars, prompt_limit_chars)
    remaining_chars = max(prompt_limit_chars - irreducible_chars, 0)

    for index, item in reversed(list(enumerate(items))):
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue

        raw_call_id = item.get("call_id") or item.get("id")
        call_id = raw_call_id.strip() if isinstance(raw_call_id, str) else ""

        output_text = _output_text_of(item)

        build_stub = _builder(index)
        result_chars = len(output_text)
        floor_chars = _output_floor_chars(output_text, build_stub)
        excess_chars = _wire_chars(output_text) - floor_chars

        if excess_chars <= remaining_chars:
            remaining_chars = max(remaining_chars - excess_chars, 0)
            continue

        stub = build_stub(
            result_chars=result_chars,
            remaining_tokens=(remaining_chars // _CHARS_PER_TOKEN_HEURISTIC),
        )
        if live_from is None or _is_live(index):
            item["output"] = stub
            if call_id:
                omitted_call_ids.add(call_id)
            logger.warning(
                "Omitted oversized tool result (call_id=%s): %d chars (remaining_budget=%d chars)",
                call_id,
                result_chars,
                remaining_chars,
            )
        remaining_chars = max(remaining_chars - (_wire_chars(stub) - floor_chars), 0)

    return BudgetOutcome(
        frozenset(omitted_call_ids),
        futile,
        irreducible_chars,
        prompt_limit_chars,
        measured_ratio,
        prompt_limit_tokens,
    )


def apply_live_tool_output_budget(
    outputs: list[dict[str, Any]],
    *,
    existing_input_items: Any,
    model_id: str,
    logger: logging.Logger = logger,
    referenced_sizes: Mapping[str, tuple[int, str, str]] | None = None,
    reserved_output_tokens: int | None = None,
    fixed_overhead_chars: int = 0,
    chars_per_token: float | None = None,
) -> BudgetOutcome:
    if not outputs:
        return BudgetOutcome(frozenset())
    existing = existing_input_items if isinstance(existing_input_items, list) else []
    return _apply_tool_output_budget(
        list(existing) + list(outputs),
        model_id=model_id,
        live_from=len(existing),
        futile_message=_LIVE_FUTILE_MESSAGE,
        logger=logger,
        referenced_sizes=referenced_sizes,
        reserved_output_tokens=reserved_output_tokens,
        fixed_overhead_chars=fixed_overhead_chars,
        chars_per_token=chars_per_token,
    )


def apply_replay_tool_output_budget(
    items: list[Any],
    *,
    model_id: str,
    logger: logging.Logger = logger,
    referenced_sizes: Mapping[str, tuple[int, str, str]] | None = None,
    reserved_output_tokens: int | None = None,
    fixed_overhead_chars: int = 0,
    chars_per_token: float | None = None,
) -> BudgetOutcome:
    return _apply_tool_output_budget(
        items,
        model_id=model_id,
        live_from=None,
        futile_message=_REPLAY_FUTILE_MESSAGE,
        logger=logger,
        referenced_sizes=referenced_sizes,
        reserved_output_tokens=reserved_output_tokens,
        fixed_overhead_chars=fixed_overhead_chars,
        chars_per_token=chars_per_token,
    )
