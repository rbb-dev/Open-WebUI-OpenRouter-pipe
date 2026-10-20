from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..pipe import Pipe

_ANTHROPIC_ID_RE = re.compile(r"~?anthropic[./]")


def _is_anthropic_model_id(model_id: Any) -> bool:
    if not isinstance(model_id, str):
        return False
    return bool(_ANTHROPIC_ID_RE.match(model_id.strip()))


def _maybe_apply_anthropic_prompt_caching(
    input_items: list[dict[str, Any]],
    *,
    model_id: str,
    valves: Pipe.Valves,
    tools: list[dict[str, Any]] | None = None,
) -> None:
    if not valves.ENABLE_ANTHROPIC_PROMPT_CACHING:
        return
    if not _is_anthropic_model_id(model_id):
        return

    ttl = valves.ANTHROPIC_PROMPT_CACHE_TTL
    cache_control_payload: dict[str, Any] = {"type": "ephemeral"}
    if isinstance(ttl, str) and ttl:
        cache_control_payload["ttl"] = ttl

    for item in input_items:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "input_text"
                and block.get("cache_control") == cache_control_payload
            ):
                del block["cache_control"]
    if isinstance(tools, list):
        for tool in tools:
            if (
                isinstance(tool, dict)
                and tool.get("cache_control") == cache_control_payload
            ):
                del tool["cache_control"]

    has_tools = isinstance(tools, list) and len(tools) > 0
    if has_tools:
        assert tools is not None
        for tool in reversed(tools):
            if not isinstance(tool, dict):
                continue
            existing_cc = tool.get("cache_control")
            if existing_cc is None:
                tool["cache_control"] = dict(cache_control_payload)
            elif (
                isinstance(existing_cc, dict)
                and cache_control_payload.get("ttl") and "ttl" not in existing_cc
            ):
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
            elif (
                isinstance(existing_cc, dict)
                and cache_control_payload.get("ttl") and "ttl" not in existing_cc
            ):
                existing_cc["ttl"] = cache_control_payload["ttl"]
            break


def _maybe_apply_responses_toplevel_cache_control(
    request_body: dict[str, Any],
    *,
    valves: Pipe.Valves,
) -> None:
    if not valves.ENABLE_ANTHROPIC_PROMPT_CACHING:
        return
    if not _is_anthropic_model_id(request_body.get("model")):
        return
    existing = request_body.get("cache_control")
    if isinstance(existing, dict) and existing:
        return
    payload: dict[str, Any] = {"type": "ephemeral"}
    if valves.ANTHROPIC_PROMPT_CACHE_TTL == "1h":
        payload["ttl"] = "1h"
    request_body["cache_control"] = payload
