"""Event emission and middleware stream handling.

Handles event emission to Open WebUI and middleware stream queue management.
"""

from __future__ import annotations

import asyncio
import logging
import datetime
import json
import secrets
import time
from time import perf_counter
from typing import Any, Optional, Awaitable, Callable, Dict, Literal, Protocol, TYPE_CHECKING
from ..core.timing_logger import timed

# Type hints for Open WebUI components
EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]

# Import helpers and SessionLogger
from ..core.utils import _render_error_template
from ..core.logging_system import SessionLogger

if TYPE_CHECKING:
    from ..pipe import _PipeJob

# Lazy loading for Open WebUI utilities to avoid triggering langchain at import time
_owui_template_cached: Optional[Callable[..., dict[str, Any]]] = None


@timed
def _stub_chat_chunk_template(
    model: str,
    content: Optional[str] = None,
    reasoning_content: Optional[str] = None,
    tool_calls: Optional[list[dict[str, Any]]] = None,
    usage: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Stub implementation mimicking OpenAI chat completion chunk format."""
    chunk: dict[str, Any] = {
        "id": f"chatcmpl-{secrets.token_hex(12)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": None,
        }],
    }

    if content:
        chunk["choices"][0]["delta"]["content"] = content
    if reasoning_content:
        chunk["choices"][0]["delta"]["reasoning_content"] = reasoning_content
    if tool_calls:
        chunk["choices"][0]["delta"]["tool_calls"] = tool_calls
    if usage:
        chunk["usage"] = usage

    return chunk


class _OpenAIChatChunkTemplate(Protocol):
    @timed
    def __call__(
        self,
        model: str,
        content: Optional[str] = None,
        reasoning_content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        usage: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        ...


@timed
def openai_chat_chunk_message_template(
    model: str,
    content: Optional[str] = None,
    reasoning_content: Optional[str] = None,
    tool_calls: Optional[list[dict[str, Any]]] = None,
    usage: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Wrapper that lazily loads the Open WebUI template or uses stub."""
    global _owui_template_cached
    if _owui_template_cached is None:
        try:
            from open_webui.utils.misc import openai_chat_chunk_message_template as _real_template
            _owui_template_cached = _real_template
        except ImportError:
            _owui_template_cached = _stub_chat_chunk_template
    return _owui_template_cached(model, content, reasoning_content, tool_calls, usage)

LOGGER = logging.getLogger(__name__)


class EventEmitterHandler:
    """Manages event emission and stream queues.

    This class encapsulates:
    - Status/error/citation/completion event emission
    - Safe event emitter wrapping
    - Middleware stream queue management
    - Event formatting and validation

    All event emission to Open WebUI is handled through this class, providing
    a consistent interface for UI updates, error reporting, and streaming output.

    Dependencies:
        - logger: Logger instance for diagnostic output
        - valves: Configuration valves for support contact info, etc.
        - pipe_instance: Reference to Pipe instance for SessionLogger access
    """

    # Constants for reasoning status emission
    _REASONING_STATUS_PUNCTUATION = (".", "!", "?", ":", "\n")
    _REASONING_STATUS_MAX_CHARS = 160
    _REASONING_STATUS_MIN_CHARS = 12
    _REASONING_STATUS_IDLE_SECONDS = 0.75

    @timed
    def __init__(
        self,
        logger: logging.Logger,
        valves: Any,
        pipe_instance: Any,  # Reference to Pipe instance for helper methods
        event_emitter: Optional[EventEmitter] = None,
    ):
        """Initialize EventEmitterHandler.
        
        Args:
            logger: Logger instance for diagnostic output
            valves: Configuration valves (Pipe.Valves instance)
            pipe_instance: Reference to parent Pipe instance for helper methods
            event_emitter: Optional event emitter from Open WebUI
        """
        self.logger = logger
        self.valves = valves
        self._pipe = pipe_instance
        self._event_emitter = event_emitter

    @timed
    async def _emit_status(
        self,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]],
        message: str,
        done: bool = False
    ):
        """Emit status updates to the Open WebUI client.

        Sends progress indicators to the UI during file/image processing operations.

        Args:
            event_emitter: Async callable for sending events to the client,
                          or None if no emitter available
            message: Status message to display (supports emoji for visual indicators)
            done: Whether this status represents completion (default: False)

        Status Message Conventions:
            - 📥 Download/upload in progress
            - ✅ Successful completion
            - ⚠️ Warning or non-critical error
            - 🔴 Critical error

        Note:
            - If event_emitter is None, this method is a no-op
            - Errors during emission are caught and logged
            - Does not interrupt processing flow

        Example:
            >>> await self._emit_status(
            ...     emitter,
            ...     "📥 Downloading remote image...",
            ...     done=False
            ... )
        """
        if event_emitter:
            try:
                await event_emitter({
                    "type": "status",
                    "data": {
                        "description": message,
                        "done": done
                    }
                })
            except Exception as exc:
                self.logger.error(f"Failed to emit status: {exc}")



    @timed
    async def _emit_error(
        self,
        event_emitter: EventEmitter | None,
        error_obj: Exception | str,
        *,
        show_error_message: bool = True,
        show_error_log_citation: bool = False,
        done: bool = False,
    ) -> None:
        """Log an error and optionally surface it to the UI.

        When ``show_error_log_citation`` is true the collected debug logs are
        dumped to the server log (instead of the UI) so developers can inspect
        what went wrong.
        """
        error_message = str(error_obj)  # If it's an exception, convert to string
        self.logger.error("Error: %s", error_message)

        if show_error_message and event_emitter:
            await event_emitter(
                {
                    "type": "chat:completion",
                    "data": {
                        "error": {"message": error_message},
                        "done": done,
                    },
                }
            )

        # 2) Optionally dump the collected logs to the backend logger
        if show_error_log_citation:
            request_id = SessionLogger.request_id.get()
            with SessionLogger._state_lock:
                logs = list(SessionLogger.logs.get(request_id or "", []))
            if logs:
                if self.logger.isEnabledFor(logging.DEBUG):
                    try:
                        rendered = "\n".join(SessionLogger.format_event_as_text(e) for e in logs if isinstance(e, dict))
                    except Exception:
                        rendered = ""
                    if rendered:
                        self.logger.debug("Error logs for request %s:\n%s", request_id, rendered)
            else:
                self.logger.warning("No debug logs found for request_id %s", request_id)



    @timed
    async def _emit_templated_error(
        self,
        event_emitter: EventEmitter | None,
        *,
        template: str,
        variables: dict[str, Any],
        log_message: str,
        log_level: int = logging.ERROR,
    ) -> None:
        """Render and emit an error using the template system.

        Automatically enriches variables with:
        - error_id: Unique identifier for support correlation
        - timestamp: ISO 8601 timestamp (UTC)
        - session_id: Current session identifier
        - user_id: Current user identifier
        - support_email: From valves
        - support_url: From valves

        Args:
            event_emitter: Event emitter for UI messages
            template: Markdown template with {{#if}} conditionals
            variables: Dictionary of template variables
            log_message: Technical message for operator logs
            log_level: Logging level (default: ERROR)
        """
        error_id, context_defaults = self._build_error_context()
        enriched_variables = {**context_defaults, **variables}

        # Log with error ID for correlation
        self.logger.log(
            log_level,
            f"[{error_id}] {log_message} (session={enriched_variables['session_id']}, user={enriched_variables['user_id']})"
        )

        if not event_emitter:
            return

        try:
            markdown = _render_error_template(template, enriched_variables)
        except Exception as e:
            self.logger.error(f"[{error_id}] Template rendering failed: {e}")
            markdown = (
                f"### ⚠️ Error\n\n"
                f"An error occurred, but we couldn't format the error message properly.\n\n"
                f"**Error ID:** `{error_id}` (share this with support)\n\n"
                f"Please contact your administrator."
            )

        try:
            await event_emitter({
                "type": "chat:message",
                "data": {"content": markdown}
            })
            await event_emitter({
                "type": "chat:completion",
                "data": {"done": True}
            })
        except Exception as e:
            self.logger.error(f"[{error_id}] Failed to emit error message: {e}")



    @timed
    def _build_error_context(self) -> tuple[str, dict[str, Any]]:
        """Return a unique error id plus contextual metadata for templates."""
        error_id = secrets.token_hex(8)
        context = {
            "error_id": error_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "session_id": SessionLogger.session_id.get() or "",
            "user_id": SessionLogger.user_id.get() or "",
            "support_email": self.valves.SUPPORT_EMAIL,
            "support_url": self.valves.SUPPORT_URL,
        }
        return error_id, context



    @timed
    async def _emit_citation(
        self,
        event_emitter: EventEmitter | None,
        citation: Dict[str, Any],
    ) -> None:
        """Send a normalized source block to the UI if an emitter is available."""
        if event_emitter is None or not isinstance(citation, dict):
            return

        documents_raw = citation.get("document")
        if isinstance(documents_raw, str):
            documents = [documents_raw] if documents_raw.strip() else []
        elif isinstance(documents_raw, list):
            documents = [str(doc).strip() for doc in documents_raw if str(doc).strip()]
        else:
            documents = []
        if not documents:
            documents = ["Citation"]

        metadata_raw = citation.get("metadata")
        if isinstance(metadata_raw, list):
            metadata = [m for m in metadata_raw if isinstance(m, dict)]
        else:
            metadata = []

        source_info = citation.get("source")
        if isinstance(source_info, dict):
            source_name = (source_info.get("name") or source_info.get("url") or "source").strip() or "source"
            if not source_info.get("name"):
                source_info = dict(source_info)
                source_info["name"] = source_name
        else:
            source_name = "source"
            source_info = {"name": source_name}

        if not metadata:
            metadata = [
                {
                    "date_accessed": datetime.datetime.now().isoformat(),
                    "source": source_info.get("url") or source_name,
                }
            ]

        await event_emitter(
            {
                "type": "source",
                "data": {
                    "document": documents,
                    "metadata": metadata,
                    "source": source_info,
                },
            }
        )


    @timed
    async def _emit_completion(
        self,
        event_emitter: EventEmitter | None,
        *,
        content: str | None = "",                       # always included (may be "").  UI will stall if you leave it out.
        title:   str | None = None,                     # optional title.
        usage:   dict[str, Any] | None = None,          # optional usage block
        done:    bool = True,                           # True -> final frame
    ) -> None:
        """Emit a ``chat:completion`` event if an emitter is present.

        The ``done`` flag indicates whether this is the final frame for the
        request.  When ``usage`` information is provided it is forwarded as part
        of the event data. Callers should pass the latest assistant snapshot so
        downstream emitters cannot overwrite the UI with an empty string.
        """
        if event_emitter is None:
            return

        # Note: Open WebUI emits a final "chat:completion" event after the stream ends, which overwrites any previously emitted completion events' content and title in the UI.
        await event_emitter(
            {
                "type": "chat:completion",
                "data": {
                    "done": done,
                    "content": content,
                    **({"title": title} if title is not None else {}),
                    **({"usage": usage} if usage is not None else {}),
                }
            }
        )


    @timed
    async def _emit_notification(
        self,
        event_emitter: EventEmitter | None,
        content: str,
        *,
        level: Literal["info", "success", "warning", "error"] = "info",
    ) -> None:
        """Emit a toast-style notification to the UI.

        The ``level`` argument controls the styling of the notification banner.
        """
        if event_emitter is None:
            return

        await event_emitter(
            {"type": "notification", "data": {"type": level, "content": content}}
        )


    @timed
    def _wrap_safe_event_emitter(
        self,
        emitter: EventEmitter | None,
    ) -> EventEmitter | None:
        """Return an emitter wrapper that swallows downstream transport errors."""

        if emitter is None:
            return None

        @timed
        async def _guarded(event: dict[str, Any]) -> None:
            try:
                await emitter(event)
            except Exception as exc:  # pragma: no cover - emitter transport errors
                evt_type = event.get("type") if isinstance(event, dict) else None
                suffix = f" ({evt_type})" if evt_type else ""
                self.logger.warning("Event emitter failure%s: %s", suffix, exc)

        return _guarded


    @timed
    def _try_put_middleware_stream_nowait(
        self,
        stream_queue: asyncio.Queue[dict[str, Any] | str | None],
        item: dict[str, Any] | str | None,
    ) -> None:
        try:
            stream_queue.put_nowait(item)
        except asyncio.QueueFull:
            return
        except Exception:
            return


    @timed
    async def _put_middleware_stream_item(
        self,
        job: _PipeJob,
        stream_queue: asyncio.Queue[dict[str, Any] | str | None],
        item: dict[str, Any] | str,
    ) -> None:
        if job.future.cancelled():
            raise asyncio.CancelledError()

        if job.valves.MIDDLEWARE_STREAM_QUEUE_MAXSIZE <= 0:
            await stream_queue.put(item)
            return

        timeout = job.valves.MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS
        if timeout <= 0:
            await stream_queue.put(item)
            return

        try:
            await asyncio.wait_for(stream_queue.put(item), timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.warning(
                "Middleware stream queue enqueue timed out (request_id=%s, maxsize=%s).",
                job.request_id,
                stream_queue.maxsize,
            )
            raise



    @timed
    def _make_middleware_stream_emitter(
        self,
        job: _PipeJob,
        stream_queue: asyncio.Queue[dict[str, Any] | str | None],
    ) -> EventEmitter:
        """Translate internal events into middleware-supported streaming output.

        Open WebUI's `process_chat_response` middleware consumes OpenAI-style
        streaming chunks (``choices[].delta``) and supports out-of-band events
        via a top-level ``event`` key. This adapter ensures:

        - assistant deltas become ``delta.content`` chunks
        - reasoning traces become ``delta.reasoning_content`` chunks
        - status/citation/notification/etc are forwarded via ``{"event": ...}``
        """

        model_id = ""
        metadata_model = job.metadata.get("model") if isinstance(job.metadata, dict) else None
        if isinstance(metadata_model, dict):
            model_id = str(metadata_model.get("id") or "")
        if not model_id:
            model_id = str(job.body.get("model") or "pipe")

        assistant_sent = ""
        answer_started = False
        reasoning_status_buffer = ""
        reasoning_status_last_emit: float | None = None

        thinking_mode = job.valves.THINKING_OUTPUT_MODE
        thinking_box_enabled = thinking_mode in {"open_webui", "both"}
        thinking_status_enabled = thinking_mode in {"status", "both"}

        # Get the original event emitter (if any) for test compatibility
        original_emitter = job.event_emitter

        @timed
        async def _emit(event: dict[str, Any]) -> None:
            nonlocal assistant_sent, answer_started
            if not isinstance(event, dict):
                return

            # Call the original emitter first (for tests and other consumers)
            if original_emitter:
                try:
                    await original_emitter(event)
                except Exception:
                    pass  # Don't let emitter failures break streaming

            etype = event.get("type")
            raw_data = event.get("data")
            data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}

            @timed
            async def _maybe_emit_reasoning_status(delta_text: str, *, force: bool = False) -> None:
                """Emit status updates for late-arriving reasoning without spamming the UI."""
                nonlocal reasoning_status_buffer, reasoning_status_last_emit
                if not isinstance(delta_text, str):
                    return
                reasoning_status_buffer += delta_text
                text = reasoning_status_buffer.strip()
                if not text:
                    return
                should_emit = force
                now = perf_counter()
                if not should_emit:
                    if delta_text.rstrip().endswith(self._REASONING_STATUS_PUNCTUATION):
                        should_emit = True
                    elif len(text) >= self._REASONING_STATUS_MAX_CHARS:
                        should_emit = True
                    else:
                        elapsed = None if reasoning_status_last_emit is None else (now - reasoning_status_last_emit)
                        if len(text) >= self._REASONING_STATUS_MIN_CHARS:
                            if elapsed is None or elapsed >= self._REASONING_STATUS_IDLE_SECONDS:
                                should_emit = True
                if not should_emit:
                    return
                await self._put_middleware_stream_item(
                    job,
                    stream_queue,
                    {
                        "event": {
                            "type": "status",
                            "data": {
                                "description": text,
                                "done": False,
                            },
                        }
                    }
                )
                reasoning_status_buffer = ""
                reasoning_status_last_emit = now

            if etype == "chat:message":
                delta = data.get("delta")
                content = data.get("content")
                delta_text: str | None = None

                if isinstance(delta, str) and delta:
                    delta_text = delta
                    if isinstance(content, str) and content.startswith(assistant_sent):
                        assistant_sent = content
                    else:
                        assistant_sent = assistant_sent + delta
                elif isinstance(content, str) and content:
                    if content.startswith(assistant_sent):
                        delta_text = content[len(assistant_sent) :]
                        assistant_sent = content
                if isinstance(delta_text, str) and delta_text:
                    answer_started = True
                    if thinking_status_enabled and reasoning_status_buffer:
                        await _maybe_emit_reasoning_status("", force=True)
                    await self._put_middleware_stream_item(
                        job,
                        stream_queue,
                        openai_chat_chunk_message_template(model_id, delta_text),
                    )
                return

            if etype == "chat:tool_calls":
                tool_calls = data.get("tool_calls")
                if not (isinstance(tool_calls, list) and tool_calls):
                    return
                try:
                    if self.logger.isEnabledFor(logging.DEBUG):
                        summaries: list[dict[str, Any]] = []
                        for call in tool_calls:
                            if not isinstance(call, dict):
                                continue
                            fn_raw = call.get("function")
                            fn = fn_raw if isinstance(fn_raw, dict) else {}
                            args = fn.get("arguments")
                            summaries.append(
                                {
                                    "index": call.get("index"),
                                    "id": call.get("id"),
                                    "type": call.get("type"),
                                    "name": fn.get("name"),
                                    "args_len": len(args) if isinstance(args, str) else None,
                                    "args_empty": (isinstance(args, str) and not args.strip()),
                                }
                            )
                        self.logger.debug(
                            "Emitting OWUI tool_calls chunk (request_id=%s): %s",
                            job.request_id,
                            json.dumps(summaries, ensure_ascii=False),
                        )
                    chunk = openai_chat_chunk_message_template(model_id, tool_calls=tool_calls)
                    await self._put_middleware_stream_item(job, stream_queue, chunk)
                except Exception:
                    self.logger.debug(
                        "Failed to emit tool_calls chunk (request_id=%s)",
                        job.request_id,
                        exc_info=self.logger.isEnabledFor(logging.DEBUG),
                    )
                return

            if etype == "reasoning:delta":
                delta = data.get("delta")
                if isinstance(delta, str) and delta:
                    if thinking_box_enabled:
                        await self._put_middleware_stream_item(
                            job,
                            stream_queue,
                            openai_chat_chunk_message_template(
                                model_id,
                                reasoning_content=delta,
                            ),
                        )
                    if thinking_status_enabled:
                        await _maybe_emit_reasoning_status(delta)
                return

            if etype == "reasoning:completed":
                if thinking_status_enabled and reasoning_status_buffer:
                    await _maybe_emit_reasoning_status("", force=True)
                return

            if etype == "chat:completion":
                if thinking_status_enabled and reasoning_status_buffer:
                    await _maybe_emit_reasoning_status("", force=True)
                error = data.get("error")
                if isinstance(error, dict) and error:
                    await self._put_middleware_stream_item(job, stream_queue, {"error": error})

                usage = data.get("usage")
                if isinstance(usage, dict) and usage:
                    await self._put_middleware_stream_item(job, stream_queue, {"usage": usage})
                return

            await self._put_middleware_stream_item(job, stream_queue, {"event": event})

        return _emit
