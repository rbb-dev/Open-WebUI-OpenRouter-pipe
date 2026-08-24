"""Event emission and middleware stream handling.

Handles event emission to Open WebUI and middleware stream queue management.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from ..core.logging_system import SessionLogger
from ..core.utils import (
    _render_error_template,
    citation_access_stamp,
    join_answer_and_card,
)

EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]

if TYPE_CHECKING:
    from ..pipe import _PipeJob

_owui_template_cached: Callable[..., dict[str, Any]] | None = None


def _stub_chat_chunk_template(
    model: str,
    content: str | None = None,
    _reasoning_unused: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
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
    if tool_calls:
        chunk["choices"][0]["delta"]["tool_calls"] = tool_calls
    if usage:
        chunk["usage"] = usage

    return chunk


def openai_chat_chunk_message_template(
    model: str,
    content: str | None = None,
    _reasoning_unused: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrapper that lazily loads the Open WebUI template or uses stub."""
    global _owui_template_cached
    if _owui_template_cached is None:
        try:
            from open_webui.utils.misc import (
                openai_chat_chunk_message_template as _real_template,
            )
            _owui_template_cached = _real_template
        except ImportError:
            _owui_template_cached = _stub_chat_chunk_template
        except Exception:
            logging.getLogger(__name__).warning(
                "open_webui.utils.misc failed to import for a reason other than absence; "
                "the features that depend on it are now disabled",
                exc_info=True,
            )
            _owui_template_cached = _stub_chat_chunk_template
    return _owui_template_cached(model, content, None, tool_calls, usage)


_UNGUARDED_ATTR = "_openrouter_unguarded_emitter"


def unguarded_emitter(emitter: EventEmitter) -> EventEmitter:
    """The emitter as handed to the pipe, before the wrapper that swallows its errors.

    ``_wrap_safe_event_emitter`` exists so an incidental status update cannot fail a
    request. A caller that must know whether the user was actually told needs the
    original, and gets it here; anything that was never wrapped is already original.
    """
    return getattr(emitter, _UNGUARDED_ATTR, emitter)


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

    def __init__(
        self,
        logger: logging.Logger,
        valves: Any,
        pipe_instance: Any,
        event_emitter: EventEmitter | None = None,
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

    async def _emit_status(
        self,
        event_emitter: Callable[[dict], Awaitable[None]] | None,
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
            except Exception:
                self.logger.exception("Failed to emit status")


    async def _emit_error_event(
        self,
        event_emitter: EventEmitter | None,
        error_obj: Exception | str,
        *,
        show_error_message: bool = True,
        show_error_log_citation: bool = False,
        done: bool = False,
    ) -> str:
        """Log an error and optionally surface it to the UI.

        When ``show_error_log_citation`` is true the collected debug logs are
        dumped to the server log (instead of the UI) so developers can inspect
        what went wrong.
        """
        error_message = str(error_obj)
        self.logger.error("Error: %s", error_message)
        shown = error_message if show_error_message else ""

        if show_error_message and event_emitter:
            try:
                await event_emitter(
                    {
                        "type": "chat:completion",
                        "data": {
                            "error": {"message": error_message},
                            "done": done,
                        },
                    }
                )
            except Exception:
                self.logger.exception("Failed to emit error event")

        if show_error_log_citation:
            request_id = SessionLogger.request_id.get()
            with SessionLogger._state_lock:
                logs = list(SessionLogger.logs.get(request_id or "", []))
            if logs:
                if self.logger.isEnabledFor(logging.DEBUG):
                    try:
                        rendered = "\n".join(SessionLogger.format_event_as_text(e) for e in logs if isinstance(e, dict))
                    except Exception:
                        self.logger.debug("Failed to render collected error logs", exc_info=True)
                        rendered = ""
                    if rendered:
                        self.logger.debug("Error logs for request %s:\n%s", request_id, rendered)
            else:
                self.logger.warning("No debug logs found for request_id %s", request_id)

        return shown


    async def _emit_templated_error_event(
        self,
        event_emitter: EventEmitter | None,
        *,
        template: str,
        variables: dict[str, Any],
        log_message: str,
        log_level: int = logging.ERROR,
        partial_answer: str = "",
    ) -> str:
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
        error_id, context_defaults = self._create_error_context()
        enriched_variables = {**context_defaults, **variables}

        # Log with error ID for correlation
        self.logger.log(
            log_level,
            f"[{error_id}] {log_message} (session={enriched_variables['session_id']}, user={enriched_variables['user_id']})"
        )

        try:
            markdown = _render_error_template(template, enriched_variables)
        except Exception:
            self.logger.exception("[%s] Template rendering failed", error_id)
            markdown = (
                f"### ⚠️ Error\n\n"
                f"An error occurred, but we couldn't format the error message properly.\n\n"
                f"**Error ID:** `{error_id}` (share this with support)\n\n"
                f"Please contact your administrator."
            )

        shown = join_answer_and_card(partial_answer, markdown)

        if not event_emitter:
            return shown

        try:
            await event_emitter({
                "type": "chat:message",
                "data": {"content": shown}
            })
            await event_emitter({
                "type": "chat:completion",
                "data": {"done": True}
            })
        except Exception:
            self.logger.exception("[%s] Failed to emit error message", error_id)

        return shown


    def _create_error_context(self) -> tuple[str, dict[str, Any]]:
        """Return a unique error id plus contextual metadata for templates."""
        error_id = secrets.token_hex(8)
        context = {
            "error_id": error_id,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
            "session_id": SessionLogger.session_id.get() or "",
            "user_id": SessionLogger.user_id.get() or "",
            "support_email": self.valves.SUPPORT_EMAIL,
            "support_url": self.valves.SUPPORT_URL,
        }
        return error_id, context


    async def _emit_citation(
        self,
        event_emitter: EventEmitter | None,
        citation: dict[str, Any],
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
                    "date_accessed": citation_access_stamp(),
                    "source": source_info.get("url") or source_name,
                }
            ]

        try:
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
        except Exception:
            self.logger.exception("Failed to emit citation")


    async def _emit_files(
        self,
        event_emitter: EventEmitter | None,
        files: list[dict[str, Any]],
    ) -> None:
        """Emit extracted files from tool results to the UI.

        Files typically contain images, audio, or other media extracted from
        tool execution results (e.g., image generation tools, MCP multipart results).

        Args:
            event_emitter: Event emitter for UI communication
            files: List of file dicts with 'type' and 'url' or 'content' keys
                   Example: [{"type": "image", "url": "/api/v1/files/..."}]

        Note:
            This method is called after tool execution when files are extracted
            from tool results via OpenWebUI's process_tool_result() function.
        """
        if event_emitter is None or not files:
            return

        try:
            await event_emitter({
                "type": "files",
                "data": {"files": files},
            })
        except Exception as exc:
            self.logger.debug("Failed to emit files event: %s", exc, exc_info=True)


    async def _emit_embeds(
        self,
        event_emitter: EventEmitter | None,
        embeds: list[str],
        *,
        replace: bool = False,
    ) -> None:
        """Emit embedded HTML content from tool results to the UI.

        Embeds are HTML snippets (typically iframes or interactive components)
        that should be displayed inline in the chat. They come from tools that
        return HTMLResponse with Content-Disposition: inline.

        Args:
            event_emitter: Event emitter for UI communication
            embeds: List of HTML strings to embed in the response
            replace: When True, OpenWebUI swaps the message's existing embeds for
                this set instead of appending. Omitted from the payload when False
                so existing callers stay byte-identical on the wire. No production
                caller currently sets this.

        Note:
            This method is called after tool execution when HTML embeds are
            extracted from tool results via OpenWebUI's process_tool_result() function.
        """
        if event_emitter is None or not embeds:
            return

        try:
            data: dict[str, object] = {"embeds": embeds}
            if replace:
                data["replace"] = True
            await event_emitter({
                "type": "embeds",
                "data": data,
            })
        except Exception as exc:
            self.logger.debug("Failed to emit embeds event: %s", exc, exc_info=True)


    async def _emit_completion(
        self,
        event_emitter: EventEmitter | None,
        *,
        content: str | None = "",
        title:   str | None = None,
        usage:   dict[str, Any] | None = None,
        done:    bool = True,
    ) -> None:
        """Emit a ``chat:completion`` event if an emitter is present.

        The ``done`` flag indicates whether this is the final frame for the
        request.  When ``usage`` information is provided it is forwarded as part
        of the event data. Callers should pass the latest assistant snapshot so
        downstream emitters cannot overwrite the UI with an empty string.
        """
        if event_emitter is None:
            return

        try:
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
        except Exception:
            self.logger.exception("Failed to emit completion")


    async def _emit_unstreamed_answer(
        self,
        event_emitter: EventEmitter | None,
        *,
        content: str,
        usage: dict[str, Any] | None = None,
    ) -> None:
        if event_emitter is None:
            return

        try:
            await event_emitter(
                {"type": "chat:message:delta", "data": {"content": content}}
            )
        except Exception:
            self.logger.exception("Failed to emit unstreamed answer")

        await self._emit_completion(
            event_emitter,
            content=content,
            done=True,
            usage=usage or None,
        )


    async def _emit_notification(
        self,
        event_emitter: EventEmitter | None,
        content: str,
        *,
        level: Literal["info", "success", "warning", "error"] = "info",
    ) -> bool:
        """Emit a toast-style notification to the UI, reporting whether it went out.

        The ``level`` argument controls the styling of the notification banner. The
        return value lets a caller that must not act unheard fail closed; callers that
        only inform ignore it. Emitted through :func:`unguarded_emitter` so a transport
        failure is observable here rather than swallowed one layer down, which would
        make every answer "delivered".
        """
        if event_emitter is None:
            return False

        try:
            await unguarded_emitter(event_emitter)(
                {"type": "notification", "data": {"type": level, "content": content}}
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("Failed to emit notification")
            return False
        return True


    def _wrap_safe_event_emitter(
        self,
        emitter: EventEmitter | None,
    ) -> EventEmitter | None:
        """Return an emitter wrapper that swallows downstream transport errors."""

        if emitter is None:
            return None

        async def _guarded(event: dict[str, Any]) -> None:
            try:
                await emitter(event)
            except Exception as exc:  # pragma: no cover - emitter transport errors
                evt_type = event.get("type") if isinstance(event, dict) else None
                suffix = f" ({evt_type})" if evt_type else ""
                self.logger.warning("Event emitter failure%s: %s", suffix, exc, exc_info=True)

        setattr(_guarded, _UNGUARDED_ATTR, emitter)
        return _guarded


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
            self.logger.debug("Dropped middleware stream item after unexpected enqueue error", exc_info=True)
            return


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
        except TimeoutError:
            self.logger.warning(
                "Middleware stream queue enqueue timed out, dropping item (request_id=%s, maxsize=%s).",
                job.request_id,
                stream_queue.maxsize,
            )


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
        - ``response.*`` events (native output items, incl. reasoning) pass
          through as raw SSE payloads for OWUI's serialize_output
        - status/citation/notification/etc are forwarded via ``{"event": ...}``
        """

        model_id = ""
        metadata_model = job.metadata.get("model") if isinstance(job.metadata, dict) else None
        if isinstance(metadata_model, dict):
            model_id = str(metadata_model.get("id") or "")
        if not model_id:
            model_id = str(job.body.get("model") or "pipe")

        assistant_sent = ""

        async def _emit(event: dict[str, Any]) -> None:
            nonlocal assistant_sent
            if not isinstance(event, dict):
                return

            etype = event.get("type")
            raw_data = event.get("data")
            data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}

            if isinstance(etype, str) and etype.startswith("response."):
                await self._put_middleware_stream_item(job, stream_queue, event)
                return

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
                    await self._put_middleware_stream_item(
                        job,
                        stream_queue,
                        openai_chat_chunk_message_template(model_id, delta_text),
                    )
                return

            if etype == "chat:message:delta":
                delta_text = data.get("content")
                if isinstance(delta_text, str) and delta_text:
                    assistant_sent = assistant_sent + delta_text
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
                        exc_info=True,
                    )
                return

            if etype == "chat:completion":
                completion_content = data.get("content")
                if isinstance(completion_content, str):
                    assistant_sent = completion_content
                    await self._put_middleware_stream_item(job, stream_queue, {"event": event})

                error = data.get("error")
                if isinstance(error, dict) and error:
                    await self._put_middleware_stream_item(job, stream_queue, {"error": error})

                usage = data.get("usage")
                if isinstance(usage, dict) and usage:
                    await self._put_middleware_stream_item(job, stream_queue, {"usage": usage})
                return

            await self._put_middleware_stream_item(job, stream_queue, {"event": event})

        return _emit
