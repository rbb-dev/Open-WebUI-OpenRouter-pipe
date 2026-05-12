"""Candidate-loop retry harness for structured-output task-model calls.

Derives from `.external/seedream.py:600-635` candidate-loop body, generalized
into a reusable helper.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from .client import read_task_model_response_json
from .logging import safe_log_payload


async def call_with_candidates(
    *,
    candidates: list[str],
    build_form_data: Callable[[str], dict[str, Any]],
    invoke: Callable[[dict[str, Any]], Awaitable[Any]],
    timeout_s: float,
    logger: logging.Logger,
    log_redact: Callable[[dict[str, Any]], dict[str, Any]] = safe_log_payload,
) -> dict[str, Any]:
    """Loop candidates calling the task model; return first success.

    Args:
        candidates: Ordered list of model IDs to try.
        build_form_data: Per-candidate factory that returns the chat-completion
            form_data dict (must include "model", "messages", optionally
            "response_format", "temperature", "stream").
        invoke: Async callable that takes form_data and returns the raw response
            (typically a closure over OWUI's `generate_chat_completion`).
        timeout_s: Per-candidate timeout in seconds (asyncio.wait_for wrapper).
        logger: For DEBUG payload logs and WARNING failure logs.
        log_redact: Redaction function for DEBUG payload logging.

    Returns:
        Parsed JSON dict from the first successful candidate.

    Raises:
        RuntimeError: when no candidates configured or all fail. The last
            exception is chained via `from`.
    """
    if not candidates:
        raise RuntimeError("no_task_model_candidates")

    last_error: Exception | None = None
    for model_id in candidates:
        form_data = build_form_data(model_id)
        if logger.isEnabledFor(logging.DEBUG):
            try:
                logger.debug(
                    "structured_task request payload: %s",
                    json.dumps(log_redact(form_data), ensure_ascii=False, default=str),
                )
            except Exception:
                pass
        try:
            response = await asyncio.wait_for(invoke(form_data), timeout=timeout_s)
            params = await read_task_model_response_json(response)
            if not isinstance(params, dict):
                raise RuntimeError("task_model_invalid_schema")
            return params
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "structured_task candidate '%s' failed: %s", model_id, exc
            )
            last_error = exc
            continue

    raise RuntimeError(
        f"task_model execution failed for all candidates; last_error={last_error}"
    ) from last_error
