"""Context budget estimation and dynamic tool output omission helpers."""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import LOGGER
from .utils import _coerce_positive_int
from ..models.registry import ModelFamily

_FALLBACK_PROMPT_LIMIT_TOKENS = 128_000
_CHARS_PER_TOKEN_HEURISTIC = 4
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


def estimate_serialized_chars(value: Any) -> int:
    """Estimate payload size by serialized character count."""
    try:
        return len(json.dumps(value, ensure_ascii=False))
    except Exception:
        return len(str(value))


def estimate_serialized_tokens(value: Any) -> int:
    """Estimate token count using a simple char/token heuristic."""
    return estimate_serialized_chars(value) // _CHARS_PER_TOKEN_HEURISTIC


def is_tool_omission_stub(text: str) -> bool:
    """Return True when text is already a tool omission stub."""
    if not isinstance(text, str):
        return False
    stripped = text.lstrip()
    return stripped.startswith(_LIVE_OMISSION_PREFIX) or stripped.startswith(_REPLAY_OMISSION_PREFIX)


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
    logger: logging.Logger = LOGGER,
) -> set[str]:
    """Mutate live tool outputs in-place when they exceed remaining context budget."""
    omitted_call_ids: set[str] = set()
    if not outputs:
        return omitted_call_ids

    prompt_limit_tokens = compute_prompt_limit_tokens(model_id)
    prompt_limit_chars = max(prompt_limit_tokens * _CHARS_PER_TOKEN_HEURISTIC, 0)
    remaining_chars = max(prompt_limit_chars - estimate_serialized_chars(existing_input_items), 0)

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
    logger: logging.Logger = LOGGER,
) -> set[str]:
    """Mutate replayed function_call_output entries that exceed remaining budget."""
    omitted_call_ids: set[str] = set()
    if not items:
        return omitted_call_ids

    prompt_limit_tokens = compute_prompt_limit_tokens(model_id)
    prompt_limit_chars = max(prompt_limit_tokens * _CHARS_PER_TOKEN_HEURISTIC, 0)

    baseline_items: list[Any] = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            baseline = dict(item)
            baseline["output"] = ""
            baseline_items.append(baseline)
        else:
            baseline_items.append(item)

    remaining_chars = max(prompt_limit_chars - estimate_serialized_chars(baseline_items), 0)

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
