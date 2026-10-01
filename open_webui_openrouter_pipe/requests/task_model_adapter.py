"""Task model request adapter for OpenRouter API.

This module handles housekeeping task requests (e.g., generating chat titles,
tags) via the Responses API and extracts plain text from responses.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal

import aiohttp

from ..api.transforms import (
    _apply_identifier_valves_to_payload,
    _drop_include_reasoning_for_unsupported_fallbacks,
    _filter_openrouter_request,
)
from ..core.costs import maybe_dump_costs_snapshot
from ..core.errors import (
    OpenRouterAPIError,
    _inline_span,
    _provider_log_subject,
    is_sign_in_failure,
)
from ..core.logging_system import SessionLogger
from ..core.timing_logger import timed
from ..core.utils import _render_error_template
from ..models.registry import OpenRouterModelRegistry
from ..storage.owui_files import is_temporary_chat
from ..structured_task.client import output_message_text

if TYPE_CHECKING:
    from ..pipe import Pipe


_TASK_FAILURE_NOTIFIED_WINDOW = 300

_warned_task_failure: OrderedDict[str, None] = OrderedDict()

_TASK_FAILURE_CARD_TEMPLATE = (
    "### ⚠️ Task model failed\n\n"
    "- **Task**: {task}\n"
    "- **Model**: {model}\n"
    "- **Attempts**: {attempts} attempt(s)\n"
    "- **Error**: {error_class}\n"
    "- **Error ID**: {error_id}\n"
)


def _task_failure_card(
    task_type: str,
    model_id: str,
    attempts: int,
    error_class: str,
    error_id: str,
) -> str:
    values = {
        "task": _inline_span(str(task_type or "")),
        "model": _inline_span(str(model_id or "")),
        "attempts": int(attempts),
        "error_class": _inline_span(str(error_class or "")),
        "error_id": _inline_span(str(error_id or "")),
    }
    stripped = {name: str(value).replace("{", "").replace("}", "") for name, value in values.items()}
    rendered = _render_error_template(_TASK_FAILURE_CARD_TEMPLATE, stripped)
    return rendered.replace("{", "").replace("}", "")


def _task_failure_latch_key(task_type: str, model_id: str, scope: str) -> str:
    if is_temporary_chat(scope):
        return ""
    return f"{model_id}\x1f{scope or '__no_chat_or_user__'}"


def _task_failure_was_notified(key: str) -> bool:
    return key in _warned_task_failure


def _task_failure_note_notified(key: str) -> None:
    _warned_task_failure[key] = None
    _warned_task_failure.move_to_end(key)
    while len(_warned_task_failure) > _TASK_FAILURE_NOTIFIED_WINDOW:
        _warned_task_failure.popitem(last=False)


def _task_failure_claim(key: str) -> bool:
    if key in _warned_task_failure:
        return False
    _task_failure_note_notified(key)
    return True


def _task_failure_release(key: str) -> None:
    _warned_task_failure.pop(key, None)


class TaskModelAdapter:
    """Adapter for housekeeping task model requests.

    Handles task model requests (e.g., chat titles, tags) via the Responses API
    and extracts plain text output from the response.
    """

    def __init__(self, pipe: Pipe, logger: logging.Logger):
        """Initialize TaskModelAdapter.

        Args:
            pipe: Reference to Pipe instance for accessing helper methods
            logger: Logger instance
        """
        self._pipe = pipe
        self.logger = logger

    @staticmethod
    def _extract_task_output_text(response: dict[str, Any]) -> str:
        """Normalize Responses API payloads into plain text string for task models."""
        if not isinstance(response, dict):
            return ""

        text_parts: list[str] = []
        text_parts.append(output_message_text(response.get("output")))

        joined = "\n".join(part for part in text_parts if part)
        if not joined.strip():
            fallback_text = response.get("output_text")
            if isinstance(fallback_text, str):
                joined = "\n".join(part for part in [*text_parts, fallback_text] if part)
        return joined

    @staticmethod
    def _task_name(task: Any) -> str:
        """Return a stable string task name from OWUI task metadata.

        Open WebUI passes `metadata.task` as a string (e.g. "tags_generation").
        Some callers may pass a dict-like task payload; handle both.
        """
        if isinstance(task, str):
            return task.strip()
        if isinstance(task, dict):
            name = task.get("type") or task.get("task") or task.get("name")
            return name.strip() if isinstance(name, str) else ""
        return ""

    @staticmethod
    def _uses_task_model_adapter(task: Any) -> bool:
        """Return whether the task should use the housekeeping task adapter path."""
        name = TaskModelAdapter._task_name(task)
        return bool(name) and name != "moa_response_generation"

    @timed
    async def _run_task_model_request(
        self,
        body: dict[str, Any],
        valves: Pipe.Valves,
        *,
        endpoint_override: Literal["responses", "chat_completions"] | None = None,
        session: aiohttp.ClientSession | None = None,
        task_context: Any = None,
        owui_metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
        user_obj: Any | None = None,
        pipe_id: str | None = None,
        snapshot_model_id: str | None = None,
        event_emitter: Any | None = None,
    ) -> str:
        task_body = dict(body or {})
        source_model_id = task_body.get("model", "")
        task_body["model"] = OpenRouterModelRegistry.api_model_id(source_model_id) or source_model_id
        task_body.setdefault("input", "")
        task_body["stream"] = False
        identifier_user_id = str(
            (user_id or "")
            or (
                (owui_metadata or {}).get("user_id")
                if isinstance(owui_metadata, dict)
                else ""
            )
            or ""
        )
        _apply_identifier_valves_to_payload(
            task_body,
            valves=valves,
            owui_metadata=owui_metadata or {},
            owui_user_id=identifier_user_id,
            owui_user=user_obj,
            logger=self.logger,
        )
        task_body = _filter_openrouter_request(task_body)
        _drop_include_reasoning_for_unsupported_fallbacks(task_body, self.logger)

        attempts = 2
        made_attempts = 0
        delay_seconds = 0.2
        last_error: Exception | None = None

        api_key_value, api_key_error = self._pipe._resolve_openrouter_api_key(valves)
        if api_key_error or api_key_value is None:
            reason = api_key_error or "the key gate returned no key."
            self.logger.warning(
                "OpenRouter API key is unusable for task %s: %s",
                TaskModelAdapter._task_name(task_context) or "model",
                reason,
            )
            return self._pipe._task_refusal_result(task_context, reason)

        if session is None:
            raise RuntimeError("HTTP session is required for task model requests")

        last_endpoint = endpoint_override
        for attempt in range(1, attempts + 1):
            made_attempts = attempt
            try:
                response: dict[str, Any] = {}
                async for event in self._pipe.send_openrouter_nonstreaming_request_as_events(
                    session,
                    task_body,
                    api_key=api_key_value,
                    base_url=valves.BASE_URL,
                    valves=valves,
                    endpoint_override=last_endpoint,
                    user=user_obj,
                    owui_chat_id=str((owui_metadata or {}).get("chat_id") or "") or None,
                    transient_retry=False,
                    task_request=True,
                ):
                    if event.get("type") == "openrouter_pipe.chat_fallback":
                        last_endpoint = "chat_completions"
                        continue
                    if event.get("type") == "response.completed":
                        completed = event.get("response")
                        if isinstance(completed, dict):
                            response = completed

                usage = response.get("usage") if isinstance(response, dict) else None
                if usage and isinstance(usage, dict) and user_id:
                    safe_model_id = snapshot_model_id or self._pipe._qualify_model_for_pipe(
                        pipe_id,
                        source_model_id,
                    )
                    if safe_model_id:
                        try:
                            await maybe_dump_costs_snapshot(
                                self._pipe,
                                valves,
                                user_id=user_id,
                                model_id=safe_model_id,
                                usage=usage,
                                user_obj=user_obj,
                                pipe_id=pipe_id,
                                chat_id=str((owui_metadata or {}).get("chat_id") or "") or None,
                                message_id=str((owui_metadata or {}).get("message_id") or "") or None,
                                kind="task",
                            )
                        except Exception as exc:  # pragma: no cover - guard against Redis-side issues
                            self.logger.debug(
                                "Task cost snapshot failed: %s", exc, exc_info=True
                            )

                message = self._extract_task_output_text(response).strip()
                if message:
                    await self._pipe._dispatch_generation_complete(
                        usage if isinstance(usage, dict) else None,
                        "ok",
                        request_id=SessionLogger.request_id.get() or "",
                        metadata=owui_metadata or {},
                        task=self._task_name(task_context) or "task",
                    )
                    return message

                raise ValueError(
                    "Task model returned no output_text content."
                )

            except Exception as exc:  # noqa: BLE001 - the provider raised; the retry loop decides
                last_error = exc
                is_auth_failure = isinstance(exc, OpenRouterAPIError) and is_sign_in_failure(exc)
                if is_auth_failure:
                    self._pipe._note_auth_failure()
                self.logger.warning(
                    "Task model attempt %d/%d failed: %s",
                    attempt,
                    attempts,
                    _provider_log_subject(exc),
                )
                if is_auth_failure:
                    break
                if attempt < attempts:
                    await asyncio.sleep(delay_seconds)
                    delay_seconds = min(delay_seconds * 2, 0.8)

        task_type = self._task_name(task_context) or "task"
        error_id, _context = self._pipe._ensure_error_formatter()._build_error_context()
        error_class = type(last_error).__name__ if last_error is not None else "NoneType"
        error_message = (
            f"Task model '{task_type}' failed after {made_attempts} attempt(s): "
            f"{_provider_log_subject(last_error)} "
            f"[model={source_model_id} error_class={error_class} "
            f"error_id={error_id} request_id={SessionLogger.request_id.get() or ''}]"
        )
        self.logger.error(error_message)
        await self._pipe._dispatch_generation_complete(
            None,
            "failed",
            request_id=SessionLogger.request_id.get() or "",
            metadata=owui_metadata or {},
            task=task_type,
        )
        card = _task_failure_card(task_type, source_model_id, made_attempts, error_class, error_id)
        latch_key = _task_failure_latch_key(
            task_type,
            str(source_model_id or ""),
            str((owui_metadata or {}).get("chat_id") or "") or str(identifier_user_id or ""),
        )
        log_key = latch_key or "<not retained>"
        if latch_key and not _task_failure_claim(latch_key):
            self.logger.debug("task-failure toast suppressed: %s task=%s", log_key, task_type)
        else:
            delivered = await self._pipe._event_emitter_handler._emit_notification(
                event_emitter, card, level="warning"
            )
            if delivered and latch_key:
                pass
            else:
                if latch_key:
                    _task_failure_release(latch_key)
                self.logger.debug(
                    "task-failure toast not delivered; latch left open: %s", log_key
                )
        return self._pipe._task_refusal_result(task_context, card)
