"""Candidate-loop retry harness for structured-output task-model calls.

Derives from `.external/seedream.py:600-635` candidate-loop body, generalized
into a reusable helper.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ..core.logging_system import SessionLogger
from .client import TaskModelFault, read_task_model_response_json
from .logging import _fault_code, safe_log_payload


async def call_with_candidates(
    *,
    candidates: list[str],
    build_form_data: Callable[[str], dict[str, Any]],
    invoke: Callable[[dict[str, Any]], Awaitable[Any]],
    timeout_s: float,
    logger: logging.Logger,
    log_redact: Callable[[dict[str, Any]], dict[str, Any]] = safe_log_payload,
    outcome: dict[str, Any] | None = None,
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
        raise TaskModelFault("no_task_model_candidates")

    last_error: Exception | None = None
    for index, model_id in enumerate(candidates):
        form_data = build_form_data(model_id)
        if SessionLogger.debug_enabled(logger):
            try:
                logger.debug(
                    "structured_task request payload: %s",
                    json.dumps(log_redact(form_data), ensure_ascii=False, default=str),
                )
            except Exception:
                logger.debug("structured_task payload could not be logged", exc_info=True)
        async def _attempt(fd: dict[str, Any]) -> Any:
            response = await invoke(fd)
            params = await read_task_model_response_json(response)
            if not isinstance(params, dict):
                raise TaskModelFault("task_model_invalid_schema")
            return params
        try:
            params = await asyncio.wait_for(_attempt(form_data), timeout=timeout_s)
            if outcome is not None:
                outcome["index"] = index
                outcome["model_id"] = model_id
            return params
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the candidate loop absorbs every fault and reports the last one
            logger.warning(
                "structured_task candidate '%s' failed: %s", model_id, type(exc).__name__
            )
            last_error = exc
            continue

    raise RuntimeError(
        "task_model execution failed for all candidates; "
        f"last_error={_fault_code(last_error)}"
    ) from last_error
