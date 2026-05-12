"""Redacted logging payload for structured-output task-model calls.

Ports the pattern from `.external/seedream.py:137-162`.
"""
from __future__ import annotations

from typing import Any


def safe_log_payload(form_data: dict[str, Any]) -> dict[str, Any]:
    """Redact form_data for debug logs.

    Keeps: model, temperature, stream, response_format envelope (type/name/strict),
    message count + role list, metadata. Drops all message content (prompts, user
    text, etc.) and any auth headers.
    """
    response_format = form_data.get("response_format") or {}
    json_schema = response_format.get("json_schema") or {}
    messages = form_data.get("messages") or []
    roles: list[str] = []
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("role"), str):
                roles.append(msg["role"])
    metadata_raw = form_data.get("metadata") or {}
    redacted_metadata: dict[str, Any] = {}
    if isinstance(metadata_raw, dict):
        for k, v in metadata_raw.items():
            if k == "chat_id" and isinstance(v, str) and v:
                import hashlib
                redacted_metadata[k] = hashlib.sha256(v.encode("utf-8")).hexdigest()[:8]
            elif k in ("task", "session_id"):
                redacted_metadata[k] = v
    return {
        "model": form_data.get("model"),
        "temperature": form_data.get("temperature"),
        "stream": form_data.get("stream"),
        "response_format": {
            "type": response_format.get("type"),
            "name": json_schema.get("name"),
            "strict": json_schema.get("strict"),
        },
        "messages": {
            "count": len(messages) if isinstance(messages, list) else 0,
            "roles": roles,
        },
        "metadata": redacted_metadata,
    }
