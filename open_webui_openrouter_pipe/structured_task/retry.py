"""Candidate-loop retry harness for structured-output task-model calls.

Derives from `.external/seedream.py:600-635` candidate-loop body, generalized
into a reusable helper.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from itertools import islice
from typing import Any

from ..core.errors import OpenRouterAPIError
from ..core.logging_system import SessionLogger
from ..core.warn_latch import prune_expired, warn_level
from .client import (
    _ANSWER_PART_TYPES,
    TaskModelFault,
    _content_part_text,
    _model_answer,
    normalise_model_content,
    output_message_text,
    read_task_model_response_json,
)
from .logging import _fault_code, safe_log_payload

_MIN_CANDIDATE_SLICE_S = 0.05
_MIN_CANDIDATE_SHARE = 0.25
_warned_task_candidate: dict[str, float] = {}
_TASK_CANDIDATE_WARN_COOLDOWN_S = 3600.0
_TASK_CANDIDATE_PRUNE_AT = 256

_TASK_MODEL_FAULT_PREFIX = "task_model_"
_TAIL_NON_WHITESPACE = re.compile(r"\S")
_REPAIR_OUTPUT_CHARS = 200
_REPAIR_TEMPERATURE = 0.2
_NO_CHOICES_MARKER = "[the model returned no choices]"
_CORRECTABLE_FAULTS = frozenset(
    {
        "task_model_invalid_json",
        "task_model_empty_response",
        "task_model_refusal",
        "task_model_no_choices",
        "task_model_response_too_large",
        "task_model_invalid_schema",
    }
)


def _is_correctable(exc: BaseException | None) -> bool:
    if not isinstance(exc, RuntimeError) or isinstance(exc, OpenRouterAPIError):
        return False
    if str(exc).split(":", 1)[0].strip() in _CORRECTABLE_FAULTS:
        return True
    return str(exc).startswith(_TASK_MODEL_FAULT_PREFIX)


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                refusal = message.get("refusal")
                if isinstance(refusal, str) and refusal.strip():
                    return refusal
                content = _model_answer(message)
                if isinstance(content, str):
                    return content
                if isinstance(content, dict):
                    part = _content_part_text(content)
                    if part is not None:
                        return part
                    named = content.get("type")
                    if isinstance(named, str) and named and named not in _ANSWER_PART_TYPES:
                        return ""
                    return json.dumps(content, ensure_ascii=False, default=str)
                if content is not None:
                    return normalise_model_content(content)
        output = response.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                own = item.get("refusal")
                if isinstance(own, str) and own.strip():
                    return own
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "refusal":
                        part_refusal = part.get("refusal")
                        if isinstance(part_refusal, str) and part_refusal.strip():
                            return part_refusal
            output_text = output_message_text(output)
            if output_text:
                return output_text
        if not isinstance(choices, list) or not choices:
            return _no_choices_reason(response)
    return ""


def _repair_would_starve(
    slice_s: float, *, index: int, candidate_count: int, timeout_s: float
) -> bool:
    if index < candidate_count - 1:
        return slice_s < max(_MIN_CANDIDATE_SLICE_S, timeout_s * _MIN_CANDIDATE_SHARE)
    return slice_s <= _MIN_CANDIDATE_SLICE_S


def _no_choices_reason(response: Any) -> str:
    error = response.get("error") if isinstance(response, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message
        detail = error.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail
        code = error.get("code")
        if code not in (None, ""):
            return f"error code {code}"
    return _NO_CHOICES_MARKER


def _excerpt_past_the_bound(text: str, buf: list[str]) -> str:
    kept = 0
    for position, ch in enumerate(text):
        if ch != "\n" and not ch.isprintable():
            continue
        if kept == 0 and ch.isspace():
            continue
        kept += 1
        if kept <= _REPAIR_OUTPUT_CHARS:
            continue
        if not ch.isspace():
            return "".join(buf)[:_REPAIR_OUTPUT_CHARS]
        found = _TAIL_NON_WHITESPACE.search(text, position + 1)
        if found is None:
            return "".join(buf).rstrip()[:_REPAIR_OUTPUT_CHARS]
        if found.group().isprintable():
            return "".join(buf)[:_REPAIR_OUTPUT_CHARS]
        for later in islice(text, found.start(), None):
            if later != "\n" and not later.isprintable():
                continue
            if later.isspace():
                continue
            return "".join(buf)[:_REPAIR_OUTPUT_CHARS]
        return "".join(buf).rstrip()[:_REPAIR_OUTPUT_CHARS]
    return "".join(buf).rstrip()[:_REPAIR_OUTPUT_CHARS]


def _sanitised_excerpt(text: str) -> str:
    buf: list[str] = []
    for ch in text:
        if ch != "\n" and not ch.isprintable():
            continue
        if not buf and ch.isspace():
            continue
        buf.append(ch)
        if len(buf) > _REPAIR_OUTPUT_CHARS:
            if not ch.isspace():
                return "".join(buf)[:_REPAIR_OUTPUT_CHARS]
            return _excerpt_past_the_bound(text, buf)
    return "".join(buf).strip()[:_REPAIR_OUTPUT_CHARS]


def _with_repair(
    form_data: dict[str, Any], repair: list[dict[str, Any]] | None
) -> dict[str, Any]:
    repaired = dict(form_data)
    if repair:
        messages = list(form_data.get("messages") or [])
        messages.extend(repair)
        repaired["messages"] = messages
        if repaired.get("temperature") == 0:
            repaired["temperature"] = _REPAIR_TEMPERATURE
            options = repaired.get("options")
            if isinstance(options, dict) and options.get("temperature") == 0:
                repaired["options"] = {**options, "temperature": _REPAIR_TEMPERATURE}
    return repaired


async def call_with_candidates(
    *,
    candidates: list[str],
    build_form_data: Callable[[str], dict[str, Any]],
    invoke: Callable[[dict[str, Any]], Awaitable[Any]],
    timeout_s: float,
    logger: logging.Logger,
    log_redact: Callable[[dict[str, Any]], dict[str, Any]] = safe_log_payload,
    outcome: dict[str, Any] | None = None,
    attempts_per_candidate: int = 2,
    repair_messages: Callable[[str], list[dict[str, Any]]] | None = None,
    schema_keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Loop candidates calling the task model; return first success.

    Args:
        candidates: Ordered list of model IDs to try.
        build_form_data: Per-candidate factory that returns the chat-completion
            form_data dict (must include "model", "messages", optionally
            "response_format", "temperature", "stream").
        invoke: Async callable that takes form_data and returns the raw response
            (typically a closure over OWUI's `generate_chat_completion`).
        logger: For DEBUG payload logs and WARNING failure logs.
        log_redact: Redaction function for DEBUG payload logging.

    Returns:
        Parsed JSON dict from the first successful candidate.

    Raises:
        RuntimeError: when no candidates configured or all fail. The last
            exception is chained via `from`.
    """
    if not candidates:
        raise TaskModelFault("no_task_model_candidates")

    last_error: Exception | None = None
    deadline = time.monotonic() + timeout_s

    async def _attempt(fd: dict[str, Any]) -> Any:
        response = await invoke(fd)
        seen_output.append(_response_text(response))
        return await read_task_model_response_json(response, schema_keys=schema_keys)

    for index, model_id in enumerate(candidates):
        seen_output: list[str] = []
        try:
            form_data = build_form_data(model_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.log(
                warn_level(
                    _warned_task_candidate,
                    f"{model_id}:{_fault_code(exc)}",
                    cooldown_s=_TASK_CANDIDATE_WARN_COOLDOWN_S,
                ),
                "structured_task candidate '%s' failed: %s", model_id, _fault_code(exc),
            )
            if len(_warned_task_candidate) > _TASK_CANDIDATE_PRUNE_AT:
                prune_expired(
                    _warned_task_candidate, time.monotonic(), _TASK_CANDIDATE_WARN_COOLDOWN_S
                )
            continue
        for attempt in range(1, max(1, attempts_per_candidate) + 1):
            repair = None
            if attempt > 1 and repair_messages is not None and seen_output:
                repair = repair_messages(_sanitised_excerpt(seen_output[-1]))
            request = _with_repair(form_data, repair)
            if SessionLogger.debug_enabled(logger):
                try:
                    logger.debug(
                        "structured_task request payload: %s",
                        json.dumps(log_redact(request), ensure_ascii=False, default=str),
                    )
                except Exception:
                    logger.debug("structured_task payload could not be logged", exc_info=True)
            try:
                remaining = deadline - time.monotonic()
                reserved = (len(candidates) - index - 1) * max(
                    _MIN_CANDIDATE_SLICE_S, timeout_s * _MIN_CANDIDATE_SHARE
                )
                _slice = max(remaining - reserved, _MIN_CANDIDATE_SLICE_S)
                if attempt > 1 and _repair_would_starve(
                    _slice,
                    index=index,
                    candidate_count=len(candidates),
                    timeout_s=timeout_s,
                ):
                    break
                params = await asyncio.wait_for(_attempt(request), timeout=_slice)
                if outcome is not None:
                    outcome["index"] = index
                    outcome["model_id"] = model_id
                return params
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the candidate loop absorbs every fault and reports the last one
                attempt_error = exc
                logger.log(
                    warn_level(
                        _warned_task_candidate,
                        f"{model_id}:{_fault_code(exc)}",
                        cooldown_s=_TASK_CANDIDATE_WARN_COOLDOWN_S,
                    ),
                    "structured_task candidate '%s' failed: %s", model_id, _fault_code(exc),
                )
                if len(_warned_task_candidate) > _TASK_CANDIDATE_PRUNE_AT:
                    prune_expired(
                        _warned_task_candidate, time.monotonic(), _TASK_CANDIDATE_WARN_COOLDOWN_S
                    )
            if not _is_correctable(attempt_error):
                last_error = attempt_error
                break
            last_error = attempt_error

    raise RuntimeError(
        "task_model execution failed for all candidates; "
        f"last_error={_fault_code(last_error)}"
    ) from last_error
