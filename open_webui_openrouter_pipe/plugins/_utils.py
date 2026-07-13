"""Shared utilities for the plugin subsystem.

Provides :func:`extract_task_name` and :func:`extract_user_message` —
helpers for virtual-model plugins that intercept OWUI chat requests.
"""

from __future__ import annotations

from typing import Any


def extract_task_name(task: Any) -> str:
    """Extract task name from OWUI background task metadata.

    OWUI sends background tasks (title generation, tags, emoji, follow-ups)
    to all models including virtual ones.  The task parameter can be a string
    or a dict with ``type``, ``task``, or ``name`` keys.
    """
    if isinstance(task, str):
        return task.strip()
    if isinstance(task, dict):
        name = task.get("type") or task.get("task") or task.get("name")
        return name.strip() if isinstance(name, str) else ""
    return ""


def extract_user_message(body: dict[str, Any]) -> str:
    """Extract the last user message text from a chat completions request body.

    Handles both plain string content and multimodal content (list of parts).
    Returns empty string if no user message is found.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        if isinstance(text, str):
                            return text.strip()
    return ""
