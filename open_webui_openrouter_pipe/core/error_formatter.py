"""Error formatting, template selection, and emission.

This module handles OpenRouter error formatting and user-facing error messages,
including template selection based on HTTP status codes, SSE error event parsing,
and final status description formatting with usage metrics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional
from ..core.timing_logger import timed

# Use deferred import to avoid circular dependency
if TYPE_CHECKING:
    from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError
    from ..pipe import Pipe
    from ..streaming.event_emitter import EventEmitterHandler, EventEmitter

from .errors import _format_openrouter_error_markdown
from .utils import _pretty_json

# Default template constant
DEFAULT_OPENROUTER_ERROR_TEMPLATE = """**Provider Error**

The model provider returned an error:

```
{openrouter_message}
```

**Model**: {model_slug}
**Error ID**: {error_id}
"""


class ErrorFormatter:
    """Handles error formatting, template selection, and emission."""

    @timed
    def __init__(
        self,
        pipe: "Pipe",
        event_emitter_handler: "EventEmitterHandler",
        logger: logging.Logger,
    ):
        self._pipe = pipe
        self._event_emitter_handler = event_emitter_handler
        self.logger = logger
        self.valves = pipe.valves

    # ======================================================================
    # Error Emission Methods
    # ======================================================================

    @timed
    async def _emit_error(
        self,
        event_emitter: Optional["EventEmitter"],
        error_obj: Exception | str,
        *,
        show_error_message: bool = True,
        show_error_log_citation: bool = False,
        done: bool = False,
    ):
        """Delegate to EventEmitterHandler._emit_error."""
        if not self._event_emitter_handler:
            return
        await self._event_emitter_handler._emit_error(
            event_emitter,
            error_obj,
            show_error_message=show_error_message,
            show_error_log_citation=show_error_log_citation,
            done=done,
        )

    @timed
    async def _emit_templated_error(
        self,
        event_emitter: Optional["EventEmitter"],
        *,
        template: str,
        variables: dict[str, Any],
        log_message: str,
        log_level: int = logging.ERROR,
    ):
        """Delegate to EventEmitterHandler._emit_templated_error."""
        if not self._event_emitter_handler:
            return
        await self._event_emitter_handler._emit_templated_error(
            event_emitter,
            template=template,
            variables=variables,
            log_message=log_message,
            log_level=log_level,
        )

    @timed
    def _build_error_context(self) -> tuple[str, dict[str, Any]]:
        """Delegate to EventEmitterHandler._build_error_context."""
        if not self._event_emitter_handler:
            return "", {}
        return self._event_emitter_handler._build_error_context()

    # ======================================================================
    # Template Selection
    # ======================================================================

    @timed
    def _select_openrouter_template(self, status: Optional[int]) -> str:
        """Return the appropriate template based on the HTTP status."""
        if status == 401:
            return self.valves.AUTHENTICATION_ERROR_TEMPLATE
        if status == 402:
            return self.valves.INSUFFICIENT_CREDITS_TEMPLATE
        if status == 408:
            return self.valves.SERVER_TIMEOUT_TEMPLATE
        if status == 429:
            return self.valves.RATE_LIMIT_TEMPLATE
        return self.valves.OPENROUTER_ERROR_TEMPLATE

    # ======================================================================
    # Error Building and Extraction
    # ======================================================================

    @timed
    def _build_streaming_openrouter_error(
        self,
        event: dict[str, Any],
        *,
        requested_model: Optional[str],
    ) -> "OpenRouterAPIError":
        """Normalize SSE error events into an OpenRouterAPIError."""
        # Runtime import to avoid circular dependency
        from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError

        response_value = event.get("response")
        response_block: dict[str, Any] = response_value if isinstance(response_value, dict) else {}
        error_value = event.get("error")
        error_block = error_value if isinstance(error_value, dict) else None
        if not error_block and isinstance(response_block.get("error"), dict):
            error_block = response_block.get("error")
        message = ""
        if isinstance(error_block, dict):
            message = (error_block.get("message") or "").strip()
        if not message and isinstance(response_block.get("error"), dict):
            message = (response_block.get("error", {}).get("message") or "").strip()
        if not message:
            message = (event.get("message") or "").strip() or "Streaming error"
        code = error_block.get("code") if isinstance(error_block, dict) else None
        choices = event.get("choices") or response_block.get("choices")
        native_finish_reason = None
        if isinstance(choices, list) and choices:
            first_choice = choices[0] or {}
            native_finish_reason = first_choice.get("native_finish_reason") or first_choice.get("finish_reason")
        chunk_id = event.get("id") or response_block.get("id")
        chunk_created = event.get("created") or response_block.get("created")
        chunk_model = event.get("model") or response_block.get("model")
        chunk_provider = event.get("provider") or response_block.get("provider")
        metadata: dict[str, Any] = {
            "stream_event_type": event.get("type") or "",
            "raw": event,
        }
        if isinstance(response_block, dict):
            if response_block.get("status"):
                metadata["response_status"] = response_block.get("status")
            if response_block.get("error"):
                metadata["response_error"] = response_block.get("error")
            if response_block.get("id"):
                metadata.setdefault("request_id", response_block.get("id"))
        raw_body = _pretty_json(event)
        return OpenRouterAPIError(
            status=400,
            reason=message,
            provider=chunk_provider,
            openrouter_message=message,
            openrouter_code=code,
            upstream_message=message,
            upstream_type=(code or event.get("type") or "stream_error"),
            request_id=response_block.get("id") or event.get("response_id") or event.get("request_id"),
            raw_body=raw_body,
            metadata=metadata,
            moderation_reasons=None,
            flagged_input=None,
            model_slug=chunk_model,
            requested_model=requested_model,
            metadata_json=_pretty_json(metadata),
            provider_raw=event,
            provider_raw_json=raw_body,
            native_finish_reason=native_finish_reason,
            chunk_id=chunk_id,
            chunk_created=chunk_created,
            chunk_provider=chunk_provider,
            chunk_model=chunk_model,
            is_streaming_error=True,
        )

    @timed
    def _extract_streaming_error_event(
        self,
        event: dict[str, Any] | None,
        requested_model: Optional[str],
    ) -> Optional["OpenRouterAPIError"]:
        """Return an OpenRouterAPIError for SSE error payloads, if present."""
        if not isinstance(event, dict):
            return None
        event_data: dict[str, Any] = event
        event_type = (event_data.get("type") or "").strip()
        response_raw = event_data.get("response")
        response_block = response_raw if isinstance(response_raw, dict) else None
        error_raw = event_data.get("error")
        error_block = error_raw if isinstance(error_raw, dict) else None
        has_error = error_block is not None
        if isinstance(response_block, dict):
            if response_block.get("status") == "failed" or isinstance(response_block.get("error"), dict):
                has_error = True
        if event_type in {"response.failed", "response.error", "error"}:
            has_error = True
        if not has_error:
            return None
        return self._build_streaming_openrouter_error(event_data, requested_model=requested_model)

    # ======================================================================
    # Error Reporting
    # ======================================================================

    @timed
    async def _report_openrouter_error(
        self,
        exc: "OpenRouterAPIError",
        *,
        event_emitter: EventEmitter | None,
        normalized_model_id: Optional[str],
        api_model_id: Optional[str],
        usage: Optional[dict[str, Any]] = None,
        template: Optional[str] = None,
    ) -> None:
        """Emit a user-facing markdown message for OpenRouter 400 responses."""
        # Runtime import to avoid circular dependency
        from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError as _

        if getattr(exc, "status", None) in {401, 403}:
            self._pipe._note_auth_failure()
        error_id, context_defaults = self._build_error_context()
        template_to_use = template or self._select_openrouter_template(exc.status)
        retry_after_hint = (
            exc.metadata.get("retry_after_seconds")
            or exc.metadata.get("retry_after")
        )
        if retry_after_hint and not context_defaults.get("retry_after_seconds"):
            context_defaults["retry_after_seconds"] = retry_after_hint
        self.logger.warning("[%s] OpenRouter rejected the request: %s", error_id, exc)
        if event_emitter:
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "description": "Encountered a provider error. See details below.",
                        "done": True,
                    },
                }
            )
            content = _format_openrouter_error_markdown(
                exc,
                normalized_model_id=normalized_model_id,
                api_model_id=api_model_id,
                template=template_to_use or DEFAULT_OPENROUTER_ERROR_TEMPLATE,
                context=context_defaults,
            )
            await event_emitter({"type": "chat:message", "data": {"content": content}})
            await self._pipe._emit_completion(
                event_emitter,
                content="",
                usage=usage or None,
                done=True,
            )

    # ======================================================================
    # Status Formatting
    # ======================================================================

    @timed
    def _format_final_status_description(
        self,
        *,
        elapsed: float,
        total_usage: Dict[str, Any],
        valves: "Pipe.Valves",
        stream_duration: Optional[float] = None,
    ) -> str:
        """Return the final status line respecting valve + available metrics.

        ``stream_duration`` is expected to mirror provider dashboards (request
        start -> last output event, which includes first-token latency).
        """
        default_description = f"Thought for {elapsed:.1f} seconds"
        if not valves.SHOW_FINAL_USAGE_STATUS:
            return default_description

        usage = total_usage or {}
        time_segment = f"⧗ {elapsed:.2f}s"
        tokens_for_tps: Optional[int] = None
        segments: list[str] = []

        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and cost > 0:
            cost_str = f"{cost:.6f}".rstrip("0").rstrip(".")
            segments.append(f"${cost_str}")

        @timed
        def _to_int(value: Any) -> Optional[int]:
            """Best-effort conversion to ``int`` for usage counters."""
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            return None

        input_tokens = _to_int(usage.get("input_tokens"))
        output_tokens = _to_int(usage.get("output_tokens"))
        total_tokens = _to_int(usage.get("total_tokens"))
        if total_tokens is None:
            candidates = [v for v in (input_tokens, output_tokens) if v is not None]
            if candidates:
                total_tokens = sum(candidates)
        if output_tokens is not None:
            tokens_for_tps = output_tokens
        elif total_tokens is not None:
            tokens_for_tps = total_tokens

        cached_tokens = _to_int(
            (usage.get("input_tokens_details") or {}).get("cached_tokens")
        )
        reasoning_tokens = _to_int(
            (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        )

        token_details: list[str] = []
        if input_tokens is not None:
            token_details.append(f"▲ {input_tokens}")
        if output_tokens is not None:
            token_details.append(f"▼ {output_tokens}")
        if cached_tokens is not None and cached_tokens > 0:
            token_details.append(f"↺ {cached_tokens}")
        if reasoning_tokens is not None and reasoning_tokens > 0:
            token_details.append(f"▽ {reasoning_tokens}")

        if total_tokens is not None:
            token_segment = f"⇅ {total_tokens}"
            if token_details:
                token_segment += f" ({', '.join(token_details)})"
            segments.append(token_segment)
        elif token_details:
            segments.append("⇅ " + ", ".join(token_details))

        if (
            tokens_for_tps is not None
            and stream_duration is not None
            and stream_duration > 0
        ):
            tokens_per_second = tokens_for_tps / stream_duration
            if tokens_per_second > 0:
                time_segment = f"{time_segment} {tokens_per_second:.1f} tps"

        segments.insert(0, time_segment)

        description = " | ".join(segments)
        return description or default_description
