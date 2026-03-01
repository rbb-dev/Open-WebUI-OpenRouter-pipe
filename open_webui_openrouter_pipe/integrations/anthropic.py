"""Anthropic vendor integration module.

This module contains Anthropic-specific integration logic for prompt caching
and other Anthropic-specific features when routing through OpenRouter.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipe import Pipe


def _is_anthropic_model_id(model_id: Any) -> bool:
    """Check if a model ID belongs to Anthropic."""
    if not isinstance(model_id, str):
        return False
    normalized = model_id.strip()
    return normalized.startswith("anthropic/") or normalized.startswith("anthropic.")


def _maybe_apply_anthropic_prompt_caching(
    input_items: list[dict[str, Any]],
    *,
    model_id: str,
    valves: "Pipe.Valves",
    tools: list[dict[str, Any]] | None = None,
) -> None:
    """Apply Anthropic prompt caching to input items and optionally tools.

    When enabled via valves, this function adds cache_control markers to:
    - The last tool definition (when tools are present)
    - The last system/developer message
    - The last 2-3 user messages (3 without tools, 2 with tools)

    Anthropic allows a maximum of 4 cache_control breakpoints per request.
    The budget is allocated adaptively:
    - With tools:    1 tool + 1 system + 2 user = 4
    - Without tools: 1 system + 3 user = 4

    Args:
        input_items: List of input items to potentially modify in-place
        model_id: The model identifier to check if it's an Anthropic model
        valves: Valve configuration containing caching settings
        tools: Optional list of tool definitions to potentially modify in-place
    """
    if not valves.ENABLE_ANTHROPIC_PROMPT_CACHING:
        return
    if not _is_anthropic_model_id(model_id):
        return

    ttl = valves.ANTHROPIC_PROMPT_CACHE_TTL
    cache_control_payload: dict[str, Any] = {"type": "ephemeral"}
    if isinstance(ttl, str) and ttl:
        cache_control_payload["ttl"] = ttl

    # Mark the last tool definition with cache_control (tools are earliest
    # in Anthropic's cache prefix order: tools → system → messages).
    has_tools = isinstance(tools, list) and len(tools) > 0
    if has_tools:
        assert tools is not None  # help type narrowing
        for tool in reversed(tools):
            if not isinstance(tool, dict):
                continue
            existing_cc = tool.get("cache_control")
            if existing_cc is None:
                tool["cache_control"] = dict(cache_control_payload)
            elif isinstance(existing_cc, dict):
                if cache_control_payload.get("ttl") and "ttl" not in existing_cc:
                    existing_cc["ttl"] = cache_control_payload["ttl"]
            break

    system_message_indices: list[int] = []
    user_message_indices: list[int] = []
    for idx, item in enumerate(input_items):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        role = (item.get("role") or "").lower()
        if role in {"system", "developer"}:
            system_message_indices.append(idx)
        elif role == "user":
            user_message_indices.append(idx)

    # Adaptive user message breakpoint budget: 3 without tools, 2 with tools.
    max_user_breakpoints = 2 if has_tools else 3

    target_indices: list[int] = []
    if system_message_indices:
        target_indices.append(system_message_indices[-1])
    for i in range(min(max_user_breakpoints, len(user_message_indices))):
        target_indices.append(user_message_indices[-(i + 1)])

    seen: set[int] = set()
    for msg_idx in target_indices:
        if msg_idx in seen:
            continue
        seen.add(msg_idx)
        msg = input_items[msg_idx]
        content = msg.get("content")
        if not isinstance(content, list) or not content:
            continue
        for block in reversed(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") != "input_text":
                continue
            text = block.get("text")
            if not isinstance(text, str) or not text:
                continue
            existing_cc = block.get("cache_control")
            if existing_cc is None:
                block["cache_control"] = dict(cache_control_payload)
            elif isinstance(existing_cc, dict):
                if cache_control_payload.get("ttl") and "ttl" not in existing_cc:
                    existing_cc["ttl"] = cache_control_payload["ttl"]
            break
