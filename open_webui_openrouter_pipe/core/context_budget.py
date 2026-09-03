"""Context budget estimation and dynamic tool output omission helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from ..models.registry import ModelFamily
from .utils import _coerce_positive_int

logger = logging.getLogger(__name__)

_FALLBACK_PROMPT_LIMIT_TOKENS = 128_000
_CHARS_PER_TOKEN_HEURISTIC = 4
_AUDIO_BYTES_PER_TOKEN = 500
_VIDEO_BYTES_PER_TOKEN = 380
_DOCUMENT_BYTES_PER_TOKEN = 500
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


def _reads_as_text(media_type: str, filename: str) -> bool:
    if media_type.startswith("text/"):
        return True
    if media_type in _TEXTUAL_MEDIA_TYPES or media_type.endswith(("+json", "+xml")):
        return True
    if media_type in _UNTYPED_DECLARATIONS or not media_type:
        return filename.strip().lower().endswith(_TEXTUAL_SUFFIXES)
    return False


def _document_tokens(n: int, media_type: str, filename: str = "") -> int:
    if _reads_as_text(media_type, filename):
        return n // _CHARS_PER_TOKEN_HEURISTIC
    return n // _DOCUMENT_BYTES_PER_TOKEN


_OPAQUE_BLOCK_PAYLOADS: dict[str, tuple[tuple[str, ...], Callable[..., int]]] = {
    "input_image": (("image_url",), lambda _n, _t, _f="": 1_700),
    "image_url": (("image_url",), lambda _n, _t, _f="": 1_700),
    "input_audio": (("input_audio",), lambda n, _t, _f="": n // _AUDIO_BYTES_PER_TOKEN),
    "input_file": (("file_data", "file_id", "file_url"), _document_tokens),
    "video_url": (("video_url",), lambda n, _t, _f="": n // _VIDEO_BYTES_PER_TOKEN),
}
_LIVE_OMISSION_PREFIX = "[Tool result omitted due to context budget."
_REPLAY_OMISSION_PREFIX = "[Replayed tool result omitted due to context budget."


def compute_prompt_limit_tokens(model_id: str) -> int:
    """Estimate max prompt tokens using model metadata with safe fallbacks.

    Precedence:
    1. max_prompt_tokens (top-level first)
    2. context_length/max_completion_tokens (top-level first, top_provider fallback)
    3. context_length
    4. conservative fallback
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

    top_provider = full.get("top_provider")
    provider = top_provider if isinstance(top_provider, dict) else {}

    context_length = _coerce_positive_int(full.get("context_length"))
    if context_length is None:
        context_length = _coerce_positive_int(spec.get("context_length"))
    if context_length is None:
        context_length = _coerce_positive_int(provider.get("context_length"))

    max_completion = _coerce_positive_int(full.get("max_completion_tokens"))
    if max_completion is None:
        max_completion = _coerce_positive_int(spec.get("max_completion_tokens"))
    if max_completion is None:
        max_completion = _coerce_positive_int(provider.get("max_completion_tokens"))

    if context_length is not None and max_completion is not None and 0 < max_completion < context_length:
        return context_length - max_completion

    if context_length is not None:
        return context_length

    return _FALLBACK_PROMPT_LIMIT_TOKENS


def _payload_bytes(value: Any) -> tuple[int, str] | None:
    if isinstance(value, str):
        head = value[:128]
        media_type = ""
        if ";base64," in head:
            prefix, raw = value.split(";base64,", 1)
            if prefix.startswith("data:"):
                media_type = prefix[len("data:") :].split(";", 1)[0].strip().lower()
        else:
            raw = value
        size = len(raw) * 3 // 4
        return (size, media_type) if size else None
    if isinstance(value, dict):
        for sub_key in ("url", "data"):
            if sub_key in value:
                return _payload_bytes(value[sub_key])
    return None


def _budget_shape(value: Any) -> Any:
    if isinstance(value, dict):
        block_type = value.get("type")
        spec = _OPAQUE_BLOCK_PAYLOADS.get(block_type) if isinstance(block_type, str) else None
        if spec is not None:
            payload_keys, rate = spec
            sizes: dict[str, tuple[int, str]] = {}
            for key in payload_keys:
                if key not in value:
                    continue
                measured = _payload_bytes(value[key])
                if measured is not None:
                    sizes[key] = measured
            charged_key = max(sizes, key=lambda k: sizes[k][0]) if sizes else None
            shaped: dict[str, Any] = {}
            for key, item in value.items():
                if key == charged_key:
                    filename = value.get("filename")
                    tokens = rate(
                        *sizes[key], filename if isinstance(filename, str) else ""
                    )
                    shaped[key] = "x" * (tokens * _CHARS_PER_TOKEN_HEURISTIC)
                elif key in sizes:
                    shaped[key] = ""
                else:
                    shaped[key] = _budget_shape(item)
            return shaped
        return {key: _budget_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_budget_shape(item) for item in value]
    return value


def _baseline_without_tool_outputs(items: Any) -> Any:
    if not isinstance(items, list):
        return items
    baseline: list[Any] = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            blanked = dict(item)
            blanked["output"] = ""
            baseline.append(blanked)
        else:
            baseline.append(item)
    return baseline


def _already_stubbed_chars(items: Any) -> int:
    if not isinstance(items, list):
        return 0
    return sum(
        len(item["output"])
        for item in items
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and isinstance(item.get("output"), str)
        and is_tool_omission_stub(item["output"])
    )


def estimate_serialized_chars(value: Any) -> int:
    """Estimate payload size by serialized character count."""
    try:
        shaped = _budget_shape(value)
    except RecursionError:
        shaped = value
    try:
        return len(json.dumps(shaped, ensure_ascii=False))
    except (RecursionError, TypeError, ValueError):
        return len(str(shaped))


def is_tool_omission_stub(text: str) -> bool:
    """Return True when text is already a tool omission stub."""
    if not isinstance(text, str):
        return False
    stripped = text.lstrip()
    return stripped.startswith((_LIVE_OMISSION_PREFIX, _REPLAY_OMISSION_PREFIX))


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


def apply_live_tool_output_budget(
    outputs: list[dict[str, Any]],
    *,
    existing_input_items: Any,
    model_id: str,
    logger: logging.Logger = logger,
) -> set[str]:
    """Mutate live tool outputs in-place when they exceed remaining context budget."""
    omitted_call_ids: set[str] = set()
    if not outputs:
        return omitted_call_ids

    prompt_limit_tokens = compute_prompt_limit_tokens(model_id)
    prompt_limit_chars = max(prompt_limit_tokens * _CHARS_PER_TOKEN_HEURISTIC, 0)
    fixed_chars = estimate_serialized_chars(_baseline_without_tool_outputs(existing_input_items))
    irreducible_chars = fixed_chars + _already_stubbed_chars(existing_input_items)
    if prompt_limit_chars and irreducible_chars >= prompt_limit_chars:
        logger.warning(
            "Skipping live tool-output budget: the request's irreducible content -- attachments "
            "plus results already omitted -- needs ~%d chars against a ~%d char limit, so "
            "omitting more cannot bring it under.",
            irreducible_chars,
            prompt_limit_chars,
        )
        return omitted_call_ids
    spent_chars = max(estimate_serialized_chars(existing_input_items) - fixed_chars, 0)
    remaining_chars = max(prompt_limit_chars - fixed_chars - spent_chars, 0)

    for output in outputs:
        if not isinstance(output, dict):
            continue

        raw_call_id = output.get("call_id") or output.get("id")
        call_id = raw_call_id.strip() if isinstance(raw_call_id, str) else ""

        raw_text = output.get("output")
        output_text = raw_text if isinstance(raw_text, str) else ("" if raw_text is None else str(raw_text))
        if not isinstance(raw_text, str):
            output["output"] = output_text

        if is_tool_omission_stub(output_text):
            if call_id:
                omitted_call_ids.add(call_id)
            remaining_chars = max(remaining_chars - len(output_text), 0)
            continue

        result_chars = len(output_text)

        if result_chars > remaining_chars:
            stub = build_live_tool_omission_stub(
                result_chars=result_chars,
                remaining_tokens=(remaining_chars // _CHARS_PER_TOKEN_HEURISTIC),
            )
            if len(stub) >= result_chars:
                remaining_chars = max(remaining_chars - result_chars, 0)
                continue
            output["output"] = stub
            if call_id:
                omitted_call_ids.add(call_id)
            logger.warning(
                "Omitted oversized live tool result (call_id=%s): %d chars (remaining_budget=%d chars)",
                call_id,
                result_chars,
                remaining_chars,
            )
            remaining_chars = max(remaining_chars - len(stub), 0)
        else:
            remaining_chars = max(remaining_chars - result_chars, 0)

    return omitted_call_ids


def apply_replay_tool_output_budget(
    items: list[Any],
    *,
    model_id: str,
    logger: logging.Logger = logger,
) -> set[str]:
    """Mutate replayed function_call_output entries that exceed remaining budget."""
    omitted_call_ids: set[str] = set()
    if not items:
        return omitted_call_ids

    prompt_limit_tokens = compute_prompt_limit_tokens(model_id)
    prompt_limit_chars = max(prompt_limit_tokens * _CHARS_PER_TOKEN_HEURISTIC, 0)

    fixed_chars = estimate_serialized_chars(_baseline_without_tool_outputs(items))
    irreducible_chars = fixed_chars + _already_stubbed_chars(items)
    if prompt_limit_chars and irreducible_chars >= prompt_limit_chars:
        logger.warning(
            "Skipping replay tool-output budget: the request's irreducible content -- "
            "attachments plus results already omitted -- needs ~%d chars against a ~%d char "
            "limit, so omitting more cannot bring it under.",
            irreducible_chars,
            prompt_limit_chars,
        )
        return omitted_call_ids
    remaining_chars = max(prompt_limit_chars - fixed_chars, 0)

    for item in items:
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue

        raw_call_id = item.get("call_id") or item.get("id")
        call_id = raw_call_id.strip() if isinstance(raw_call_id, str) else ""

        raw_text = item.get("output")
        output_text = raw_text if isinstance(raw_text, str) else ("" if raw_text is None else str(raw_text))
        if not isinstance(raw_text, str):
            item["output"] = output_text

        if is_tool_omission_stub(output_text):
            if call_id:
                omitted_call_ids.add(call_id)
            remaining_chars = max(remaining_chars - len(output_text), 0)
            continue

        result_chars = len(output_text)

        if result_chars > remaining_chars:
            stub = build_replayed_tool_omission_stub(
                result_chars=result_chars,
                remaining_tokens=(remaining_chars // _CHARS_PER_TOKEN_HEURISTIC),
            )
            if len(stub) >= result_chars:
                remaining_chars = max(remaining_chars - result_chars, 0)
                continue
            item["output"] = stub
            if call_id:
                omitted_call_ids.add(call_id)
            logger.warning(
                "Omitted oversized replayed tool result (call_id=%s): %d chars (remaining_budget=%d chars)",
                call_id,
                result_chars,
                remaining_chars,
            )
            remaining_chars = max(remaining_chars - len(stub), 0)
        else:
            remaining_chars = max(remaining_chars - result_chars, 0)

    return omitted_call_ids
