"""Response normalization and JSON parsing for structured-output task-model calls.

Ports patterns from `.external/seedream.py:325-399, 752-790`.
"""
from __future__ import annotations

import json
from typing import Any


def normalise_model_content(value: Any) -> str:
    """Best-effort conversion of model content fragments to string.

    Handles: plain strings, list-of-content-parts, dicts with text/content keys.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item.get("content", "")))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content"):
            if key in value and value[key]:
                return str(value[key])
    return str(value) if value is not None else ""


def consume_sse_line(raw_line: str, content_parts: list[str]) -> None:
    """Parse a single SSE data line and append its content if valid.

    Silently ignores malformed lines and the [DONE] sentinel.
    """
    line = (raw_line or "").strip()
    if not line or line == "data: [DONE]":
        return
    payload = line[5:].strip() if line.startswith("data:") else line
    if not payload:
        return
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return
    for choice in data.get("choices", []):
        delta = choice.get("delta") or choice.get("message")
        if not delta:
            continue
        content_value = delta.get("content") if isinstance(delta, dict) else delta
        piece = normalise_model_content(content_value)
        if piece:
            content_parts.append(piece)


async def read_model_response_content(response: Any) -> str:
    """Normalise streaming or non-streaming chat-completion responses to text.

    Handles `StreamingResponse` (SSE) and dict-shaped responses uniformly.
    """
    if hasattr(response, "body_iterator"):
        content_parts: list[str] = []
        buffer = ""
        async for chunk in response.body_iterator:
            if not chunk:
                continue
            try:
                chunk_str = chunk.decode("utf-8")
            except Exception:
                chunk_str = chunk.decode("utf-8", errors="ignore")
            buffer += chunk_str
            while "\n" in buffer:
                raw_line, buffer = buffer.split("\n", 1)
                consume_sse_line(raw_line, content_parts)
        if buffer:
            consume_sse_line(buffer, content_parts)
        return "".join(content_parts)
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            first_choice = choices[0]
            message = first_choice.get("message") or first_choice.get("delta")
            if message:
                content_value = (
                    message.get("content") if isinstance(message, dict) else message
                )
                return normalise_model_content(content_value)
    return str(response or "")


async def read_task_model_response_json(response: Any) -> dict[str, Any]:
    """Parse OWUI generate_chat_completion result into a JSON dict.

    Raises:
        RuntimeError("task_model_empty_response") on empty/whitespace content.
        RuntimeError(f"task_model_refusal: {text}") on explicit refusal field.
        RuntimeError("task_model_no_choices") on missing choices.
        TypeError on unexpected response shape.
        json.JSONDecodeError on unparseable JSON content.
    """
    if hasattr(response, "body_iterator"):
        content = await read_model_response_content(response)
        if not content:
            raise RuntimeError("task_model_empty_response")
        if len(content) > 256 * 1024:
            raise RuntimeError(f"task_model_response_too_large: {len(content)}")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"task_model_invalid_json: {exc}") from exc

    # Plain string from adapters that already extracted output_text (e.g.
    # this pipe's own TaskModelAdapter when OWUI's task model points back
    # at this pipe). Trim, size-cap, and json-parse with the same envelope.
    if isinstance(response, str):
        text = response.strip()
        if not text:
            raise RuntimeError("task_model_empty_response")
        if len(text) > 256 * 1024:
            raise RuntimeError(f"task_model_response_too_large: {len(text)}")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"task_model_invalid_json: {exc}") from exc

    if not isinstance(response, dict):
        raise TypeError(f"unexpected task model response type: {type(response).__name__}")

    output_items = response.get("output")
    if isinstance(output_items, list) and output_items:
        text_parts: list[str] = []
        for item in output_items:
            if not isinstance(item, dict):
                continue
            content_list = item.get("content")
            if not isinstance(content_list, list):
                continue
            for content in content_list:
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text_parts.append(str(content.get("text") or ""))
        if text_parts:
            joined = "\n".join(p for p in text_parts if p).strip()
            if joined:
                try:
                    return json.loads(joined)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"task_model_invalid_json: {exc}") from exc

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("task_model_no_choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise TypeError(f"unexpected choice type: {type(first_choice).__name__}")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        message = {}

    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        raise RuntimeError(f"task_model_refusal: {refusal.strip()}")

    content_value = message.get("content")
    if isinstance(content_value, dict):
        return content_value
    if isinstance(content_value, str):
        if not content_value.strip():
            raise RuntimeError("task_model_empty_response")
        try:
            return json.loads(content_value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"task_model_invalid_json: {exc}") from exc

    raise TypeError(f"unexpected task model content type: {type(content_value).__name__}")
