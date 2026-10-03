"""Response normalization and JSON parsing for structured-output task-model calls.

Ports patterns from `.external/seedream.py:325-399, 752-790`.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..core.utils import utf8_stream_decoder

_logger = logging.getLogger(__name__)

_TASK_RESPONSE_MAX_BYTES = 256 * 1024
_DEFAULT_MAX_HOLD_BYTES = 4 * _TASK_RESPONSE_MAX_BYTES
_ANSWER_PART_TYPES = frozenset({"text", "output_text", "input_text"})


def output_message_text(output: Any) -> str:
    if not isinstance(output, list):
        return ""
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        parts = item.get("content") or []
        if not isinstance(parts, list):
            continue
        text = "".join(
            str(part.get("text") or "")
            for part in parts
            if isinstance(part, dict) and part.get("type") == "output_text"
        )
        if text and not text.isspace():
            texts.append(text)
    return "\n".join(texts)


def _content_part_text(item: Any) -> str | None:
    if not isinstance(item, dict):
        return str(item)
    part_type = item.get("type")
    if isinstance(part_type, str) and part_type and part_type not in _ANSWER_PART_TYPES:
        return None
    if item.get("text") is not None:
        return str(item["text"])
    if "content" in item:
        return str(item["content"])
    return None


def _join_content_parts(value: list[Any]) -> str:
    parts: list[str] = []
    for item in value:
        piece = _content_part_text(item)
        if piece is not None:
            parts.append(piece)
    return "".join(parts)


class TaskModelFault(RuntimeError):
    def __init__(self, code: str, detail: str | None = None) -> None:
        super().__init__(code if detail is None else f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _schema_verdict(value: Any, schema_keys: frozenset[str] | None) -> None:
    if schema_keys is not None and isinstance(value, dict) and not schema_keys & value.keys():
        raise TaskModelFault("task_model_invalid_schema")


def normalise_model_content(value: Any) -> str:
    """Best-effort conversion of model content fragments to string.

    Handles: plain strings, list-of-content-parts, dicts with text/content keys.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return _join_content_parts(value)
    if isinstance(value, dict):
        part = _content_part_text(value)
        return "" if part is None else part
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
        _logger.debug("Structured-task chunk parse failed", exc_info=True)
        return
    if not isinstance(data, dict):
        return
    choices = data.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or choice.get("message")
        if not delta:
            continue
        content_value = delta.get("content") if isinstance(delta, dict) else delta
        piece = normalise_model_content(content_value)
        if piece:
            content_parts.append(piece)


async def read_model_response_content(
    response: Any, max_hold_bytes: int | None = _DEFAULT_MAX_HOLD_BYTES
) -> str:
    """Normalise streaming or non-streaming chat-completion responses to text.

    Handles `StreamingResponse` (SSE) and dict-shaped responses uniformly.
    """
    if hasattr(response, "body_iterator"):
        content_parts: list[str] = []
        content_bytes = 0
        buffer = ""
        _utf8 = utf8_stream_decoder()
        async for chunk in response.body_iterator:
            if not chunk:
                continue
            if isinstance(chunk, str):
                buffer += chunk
            elif isinstance(chunk, (bytes, bytearray, memoryview)):
                buffer += _utf8.decode(bytes(chunk))
            else:
                buffer += str(chunk)
            held = len(buffer) + content_bytes
            if max_hold_bytes is not None and held > max_hold_bytes:
                raise TaskModelFault("task_model_response_too_large", f"{held}")
            pos = 0
            while (nl := buffer.find("\n", pos)) != -1:
                before = len(content_parts)
                consume_sse_line(buffer[pos:nl], content_parts)
                for piece in content_parts[before:]:
                    content_bytes += len(piece)
                pos = nl + 1
                held = (len(buffer) - pos) + content_bytes
                if max_hold_bytes is not None and held > max_hold_bytes:
                    raise TaskModelFault("task_model_response_too_large", f"{held}")
            if pos:
                buffer = buffer[pos:]
        buffer += _utf8.decode(b"", True)
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


def _model_content_is_absent(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, list)):
        return not normalise_model_content(value).strip()
    return False


def _model_answer(message: dict[str, Any]) -> Any:
    content_value = message.get("content")
    if not _model_content_is_absent(content_value):
        return content_value
    for key in ("reasoning_content", "reasoning"):
        candidate = message.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return content_value


async def read_task_model_response_json(
    response: Any, *, schema_keys: frozenset[str] | None = None
) -> dict[str, Any]:
    """Parse OWUI generate_chat_completion result into a JSON dict.

    Raises:
        RuntimeError("task_model_empty_response") on empty/whitespace content.
        RuntimeError("task_model_no_choices") on missing choices.
        TypeError on unexpected response shape.
    """
    if hasattr(response, "body_iterator"):
        content = await read_model_response_content(response)
        if not content:
            raise TaskModelFault("task_model_empty_response")
        if len(content) > _TASK_RESPONSE_MAX_BYTES:
            raise TaskModelFault("task_model_response_too_large", f"{len(content)}")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise TaskModelFault("task_model_invalid_json", f"{exc}") from exc
        if not isinstance(parsed, dict):
            raise TaskModelFault("task_model_invalid_schema")
        _schema_verdict(parsed, schema_keys)
        return parsed

    if isinstance(response, str):
        text = response.strip()
        if not text:
            raise TaskModelFault("task_model_empty_response")
        if len(text) > _TASK_RESPONSE_MAX_BYTES:
            raise TaskModelFault("task_model_response_too_large", f"{len(text)}")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TaskModelFault("task_model_invalid_json", f"{exc}") from exc
        if not isinstance(parsed, dict):
            raise TaskModelFault("task_model_invalid_schema")
        _schema_verdict(parsed, schema_keys)
        return parsed

    if not isinstance(response, dict):
        raise TypeError(f"unexpected task model response type: {type(response).__name__}")

    output_items = response.get("output")
    if isinstance(output_items, list) and output_items:
        for item in output_items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            refusal = item.get("refusal")
            if isinstance(refusal, str) and refusal.strip():
                raise TaskModelFault("task_model_refusal")
            content_list = item.get("content")
            if not isinstance(content_list, list):
                continue
            for content in content_list:
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "refusal":
                    part_refusal = content.get("refusal")
                    if isinstance(part_refusal, str) and part_refusal.strip():
                        raise TaskModelFault("task_model_refusal")
        joined = output_message_text(output_items).strip()
        if len(joined) > _TASK_RESPONSE_MAX_BYTES:
            raise TaskModelFault("task_model_response_too_large", f"{len(joined)}")
        if joined:
            try:
                parsed = json.loads(joined)
            except json.JSONDecodeError as exc:
                raise TaskModelFault("task_model_invalid_json", f"{exc}") from exc
            if not isinstance(parsed, dict):
                raise TaskModelFault("task_model_invalid_schema")
            _schema_verdict(parsed, schema_keys)
            return parsed

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TaskModelFault("task_model_no_choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise TypeError(f"unexpected choice type: {type(first_choice).__name__}")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        message = {}

    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        raise TaskModelFault("task_model_refusal")

    content_value = _model_answer(message)
    if content_value is None:
        raise TaskModelFault("task_model_empty_response")
    if isinstance(content_value, list):
        content_value = normalise_model_content(content_value)
    part_type = content_value.get("type") if isinstance(content_value, dict) else None
    if part_type == "refusal" and isinstance(content_value, dict):
        part_refusal = content_value.get("refusal")
        if isinstance(part_refusal, str) and part_refusal.strip():
            raise TaskModelFault("task_model_refusal")
    if isinstance(content_value, dict) and (
        "text" in content_value
        or "content" in content_value
        or (isinstance(part_type, str) and part_type and part_type not in _ANSWER_PART_TYPES)
    ):
        content_value = _content_part_text(content_value) or ""
    if isinstance(content_value, dict):
        try:
            measured = len(json.dumps(content_value, ensure_ascii=False, default=str))
        except (ValueError, RecursionError):
            measured = _TASK_RESPONSE_MAX_BYTES + 1
        if measured > _TASK_RESPONSE_MAX_BYTES:
            raise TaskModelFault("task_model_response_too_large", f"{measured}")
        _schema_verdict(content_value, schema_keys)
        return content_value
    if isinstance(content_value, str):
        if not content_value.strip():
            raise TaskModelFault("task_model_empty_response")
        if len(content_value) > _TASK_RESPONSE_MAX_BYTES:
            raise TaskModelFault("task_model_response_too_large", f"{len(content_value)}")
        try:
            parsed = json.loads(content_value)
        except json.JSONDecodeError as exc:
            raise TaskModelFault("task_model_invalid_json", f"{exc}") from exc
        if not isinstance(parsed, dict):
            raise TaskModelFault("task_model_invalid_schema")
        _schema_verdict(parsed, schema_keys)
        return parsed

    raise TypeError(f"unexpected task model content type: {type(content_value).__name__}")
