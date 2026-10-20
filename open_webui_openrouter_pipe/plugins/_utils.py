from __future__ import annotations

from typing import Any


def extract_task_name(task: Any) -> str:
    if isinstance(task, str):
        return task.strip()
    if isinstance(task, dict):
        name = task.get("type") or task.get("task") or task.get("name")
        return name.strip() if isinstance(name, str) else ""
    return ""


def extract_user_message(body: dict[str, Any]) -> str:
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
