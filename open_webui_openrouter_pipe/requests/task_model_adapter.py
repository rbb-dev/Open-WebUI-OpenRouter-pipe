"""Task model request adapter for OpenRouter API.

This module handles task model requests (e.g., generating chat titles, tags)
via the Responses API and extracts plain text from responses.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import aiohttp

from ..core.config import EncryptedStr
from ..core.errors import OpenRouterAPIError
from ..api.transforms import _apply_identifier_valves_to_payload, _filter_openrouter_request
from ..models.registry import OpenRouterModelRegistry, ModelFamily
from ..core.timing_logger import timed

if TYPE_CHECKING:
    from ..pipe import Pipe


class TaskModelAdapter:
    """Adapter for task model requests.

    Handles task model requests (e.g., chat titles, tags) via the Responses API
    and extracts plain text output from the response.
    """

    @timed
    def __init__(self, pipe: "Pipe", logger: logging.Logger):
        """Initialize TaskModelAdapter.

        Args:
            pipe: Reference to Pipe instance for accessing helper methods
            logger: Logger instance
        """
        self._pipe = pipe
        self.logger = logger

    @staticmethod
    @timed
    def _extract_task_output_text(response: Dict[str, Any]) -> str:
        """Normalize Responses API payloads into plain text string for task models."""
        if not isinstance(response, dict):
            return ""

        text_parts: list[str] = []
        output_items = response.get("output", [])
        if isinstance(output_items, list):
            for item in output_items:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for content in item.get("content", []):
                    if not isinstance(content, dict):
                        continue
                    if content.get("type") == "output_text":
                        text_parts.append(content.get("text", "") or "")

        # Fallback: some providers return a collapsed output_text field
        fallback_text = response.get("output_text")
        if isinstance(fallback_text, str):
            text_parts.append(fallback_text)

        joined = "\n".join(part for part in text_parts if part)
        return joined

    @staticmethod
    @timed
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

    @timed
    async def _run_task_model_request(
        self,
        body: Dict[str, Any],
        valves: "Pipe.Valves",
        *,
        session: aiohttp.ClientSession | None = None,
        task_context: Any = None,
        owui_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        user_obj: Optional[Any] = None,
        pipe_id: Optional[str] = None,
        snapshot_model_id: Optional[str] = None,
    ) -> str:
        """Process a task model request via the Responses API.

        Task models (e.g. generating a chat title or tags) return their
        information as standard Responses output.  This helper performs a single
        non-streaming call and extracts the plain text from the response items.
        """
        task_body = dict(body or {})
        source_model_id = task_body.get("model", "")
        task_body["model"] = OpenRouterModelRegistry.api_model_id(source_model_id) or source_model_id
        task_body.setdefault("input", "")
        task_body["stream"] = False
        if valves.USE_MODEL_MAX_OUTPUT_TOKENS:
            if task_body.get("max_output_tokens") is None:
                default_max = ModelFamily.max_completion_tokens(source_model_id)
                if default_max:
                    task_body["max_output_tokens"] = default_max
        else:
            task_body.pop("max_output_tokens", None)

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
            logger=self.logger,
        )
        task_body = _filter_openrouter_request(task_body)

        attempts = 2
        delay_seconds = 0.2  # keep retries snappy; task models run in latency-sensitive contexts
        last_error: Optional[Exception] = None

        if session is None:
            raise RuntimeError("HTTP session is required for task model requests")

        for attempt in range(1, attempts + 1):
            try:
                response = await self._pipe.send_openai_responses_nonstreaming_request(
                    session,
                    task_body,
                    api_key=EncryptedStr.decrypt(valves.API_KEY),
                    base_url=valves.BASE_URL,
                    valves=valves,
                )

                usage = response.get("usage") if isinstance(response, dict) else None
                if usage and isinstance(usage, dict) and user_id:
                    safe_model_id = snapshot_model_id or self._pipe._qualify_model_for_pipe(
                        pipe_id,
                        source_model_id,
                    )
                    if safe_model_id:
                        try:
                            await self._pipe._maybe_dump_costs_snapshot(
                                valves,
                                user_id=user_id,
                                model_id=safe_model_id,
                                usage=usage,
                                user_obj=user_obj,
                                pipe_id=pipe_id,
                            )
                        except Exception as exc:  # pragma: no cover - guard against Redis-side issues
                            self.logger.debug("Task cost snapshot failed: %s", exc)

                message = self._extract_task_output_text(response).strip()
                if message:
                    return message

                raise ValueError(
                    "Task model returned no output_text content."
                )

            except Exception as exc:
                last_error = exc
                is_auth_failure = (
                    isinstance(exc, OpenRouterAPIError)
                    and getattr(exc, "status", None) in {401, 403}
                )
                if is_auth_failure:
                    self._pipe._note_auth_failure()
                self.logger.warning(
                    "Task model attempt %d/%d failed: %s",
                    attempt,
                    attempts,
                    exc,
                    exc_info=self.logger.isEnabledFor(logging.DEBUG) and (not is_auth_failure),
                )
                if is_auth_failure:
                    break
                if attempt < attempts:
                    await asyncio.sleep(delay_seconds)
                    delay_seconds = min(delay_seconds * 2, 0.8)

        task_type = self._task_name(task_context) or "task"
        error_message = (
            f"Task model '{task_type}' failed after {attempts} attempt(s): {last_error}"
        )
        self.logger.error(error_message, exc_info=self.logger.isEnabledFor(logging.DEBUG))
        return f"[Task error] Unable to generate {task_type}. Please retry later."
