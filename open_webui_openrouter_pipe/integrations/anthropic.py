"""Anthropic vendor integration module.

This module contains Anthropic-specific integration logic for prompt caching
and other Anthropic-specific features when routing through OpenRouter.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from ..core.timing_logger import timed

if TYPE_CHECKING:
    from ..pipe import Pipe


@timed
def _maybe_apply_anthropic_prompt_caching(
    self,
    input_items: list[dict[str, Any]],
    *,
    model_id: str,
    valves: "Pipe.Valves",
) -> None:
    """Apply Anthropic prompt caching to input items.

    When enabled via valves, this method adds cache_control markers to appropriate
    input_text blocks in the last few system and user messages to enable Anthropic's
    prompt caching feature.

    Args:
        input_items: List of input items to potentially modify in-place
        model_id: The model identifier to check if it's an Anthropic model
        valves: Valve configuration containing caching settings
    """
    if not getattr(valves, "ENABLE_ANTHROPIC_PROMPT_CACHING", False):
        return
    if not self._is_anthropic_model_id(model_id):
        return

    ttl = getattr(valves, "ANTHROPIC_PROMPT_CACHE_TTL", "5m")
    cache_control_payload: dict[str, Any] = {"type": "ephemeral"}
    if isinstance(ttl, str) and ttl:
        cache_control_payload["ttl"] = ttl

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

    target_indices: list[int] = []
    if system_message_indices:
        target_indices.append(system_message_indices[-1])
    if user_message_indices:
        target_indices.append(user_message_indices[-1])
    if len(user_message_indices) > 1:
        target_indices.append(user_message_indices[-2])
    if len(user_message_indices) > 2:
        target_indices.append(user_message_indices[-3])

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
