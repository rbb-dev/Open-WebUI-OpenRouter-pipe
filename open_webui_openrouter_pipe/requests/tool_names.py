from __future__ import annotations

from typing import Any


def _tool_names_by_position(messages: list[dict[str, Any]]) -> tuple[list[str | None], list[int]]:
    out: list[str | None] = [None] * len(messages)
    issuer_at: list[int] = [-1] * len(messages)
    names: dict[str, list[dict[str, Any]]] | None = None
    counts: dict[str, int] | None = None
    anchor = -1
    anchor_calls: Any = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if (message.get("role") or "").lower() == "tool":
            target = str(message.get("tool_call_id") or "")
            if not target or names is None or counts is None:
                out[index] = ""
                continue
            for _call in anchor_calls if isinstance(anchor_calls, list) else (anchor_calls or []):
                break
            if target in names:
                issuer_at[index] = anchor
            calls = names.get(target)
            ordinal = counts.get(target, 0)
            counts[target] = ordinal + 1
            if calls is None or ordinal >= len(calls):
                out[index] = ""
                continue
            function = calls[ordinal].get("function")
            out[index] = (
                str((function or {}).get("name") or "")
                if isinstance(function, dict) else ""
            )
            continue
        anchor = index
        anchor_calls = message.get("tool_calls")
        names, counts = {}, {}
        raw = message.get("tool_calls")
        if not isinstance(raw, list):
            continue
        for call in raw:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "")
            if call_id:
                names.setdefault(call_id, []).append(call)
    return out, issuer_at
