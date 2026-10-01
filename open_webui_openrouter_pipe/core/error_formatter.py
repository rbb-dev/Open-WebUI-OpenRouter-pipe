"""Error formatting, template selection, and emission.

This module handles OpenRouter error formatting and user-facing error messages,
including template selection based on HTTP status codes, SSE error event parsing,
and final status description formatting with usage metrics.
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Any

from starlette.responses import StreamingResponse

from ..core.timing_logger import timed

# Use deferred import to avoid circular dependency
if TYPE_CHECKING:
    from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError

    from ..pipe import Pipe
    from ..streaming.event_emitter import EventEmitter, EventEmitterHandler

from ..storage.owui_files import is_channel_chat
from .config import OWUI_CHAT_ID
from .errors import (
    _carries_a_content_decision,
    _provider_log_subject,
    _resolve_error_model_context,
    is_sign_in_failure,
)
from .utils import (
    CONTINUED_REPLY,
    _pretty_json,
    _resolve_retry_after_seconds,
    join_answer_and_card,
)

# Simple fallback template used when no valve template is available.
# The canonical DEFAULT_OPENROUTER_ERROR_TEMPLATE lives in core/config.py.
_FALLBACK_ERROR_TEMPLATE = """**Provider Error**

The model provider returned an error:

**Provider message**: `{openrouter_message}`

**Model**: {model_identifier}
**Error ID**: {error_id}
"""

_DEFAULT_USAGE_STATUS_ICONS: tuple[str, ...] = (
    "⧗",  # time
    "$",  # cost
    "⇅",  # total tokens
    "▲",  # input tokens
    "▼",  # output tokens
    "↺",  # cached tokens
    "▽",  # reasoning tokens
)
_USAGE_ICON_FIELDS = ("time", "cost", "total", "input", "output", "cached", "reasoning")


from .errors import (
    _IN_BAND_STATUS_BY_ERROR_TYPE,
    _IN_BAND_STATUS_BY_NATIVE_CODE,
    _resolved_error_status,
)


def _in_band_status(code: Any, error_type: str, *, default: int = 400) -> int:
    named_code = code.strip().lower() if isinstance(code, str) else ""
    for table, named in (
        (_IN_BAND_STATUS_BY_ERROR_TYPE, error_type.strip().lower()),
        (_IN_BAND_STATUS_BY_NATIVE_CODE, error_type.strip().lower()),
        (_IN_BAND_STATUS_BY_ERROR_TYPE, named_code),
        (_IN_BAND_STATUS_BY_NATIVE_CODE, named_code),
    ):
        status = table.get(named)
        if status is not None:
            return status
    if isinstance(code, int) and not isinstance(code, bool):
        numeric = code
    elif isinstance(code, str) and code.strip().isdigit():
        numeric = int(code.strip())
    else:
        numeric = None
    if numeric is not None and 400 <= numeric <= 599:
        return numeric
    return default


def _choice_error(event: dict[str, Any]) -> Any:
    choices = event.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    return first.get("error") if isinstance(first, dict) else None


def _error_is_present(value: Any) -> bool:
    return isinstance(value, dict) or bool(value)


def _as_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _request_path(request: Any) -> str:
    url = getattr(request, "url", None)
    return getattr(url, "path", "") or ""


_ANTHROPIC_MESSAGES_PATHS = ("/api/v1/messages", "/api/message")


def _is_anthropic_endpoint(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in _ANTHROPIC_MESSAGES_PATHS)


def _cost_segment(cost: Any, *, use_icons: bool, icon_cost: str) -> str | None:
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not cost > 0:
        return None
    cost_str = f"{cost:.6f}".rstrip("0").rstrip(".")
    if set(cost_str) <= {"0", ".", ""}:
        cost_str = f"{cost:.2e}"
    if use_icons and icon_cost:
        if icon_cost == "$":
            return f"{icon_cost}{cost_str}"
        return f"{icon_cost} {cost_str}"
    return f"Cost ${cost_str}"


def _admission_error_response(
    status: int, message: str, *, request: Any
) -> StreamingResponse | None:
    if _is_anthropic_endpoint(_request_path(request)):
        return None
    return StreamingResponse(
        iter([json.dumps({"error": {"message": message, "code": status}}).encode("utf-8")]),
        status_code=status,
        media_type="application/json",
    )


def _transported_failure_response(
    message: str, *, code: int, stream: bool, path: str, retry_after_seconds: float | None = None
) -> StreamingResponse | None:
    if stream or _is_anthropic_endpoint(path):
        return None
    error: dict[str, Any] = {"message": message, "code": code}
    headers: dict[str, str] = {}
    if retry_after_seconds is not None:
        error["retry_after_seconds"] = retry_after_seconds
        headers["Retry-After"] = str(int(retry_after_seconds))
    return StreamingResponse(
        iter([json.dumps({"error": error}).encode("utf-8")]),
        status_code=400,
        media_type="application/json",
        headers=headers,
    )


def _unreadable_body_failure_response(
    exc: Any, *, code: int, stream: bool, path: str
) -> StreamingResponse | None:
    if stream or _is_anthropic_endpoint(path):
        return None
    return _transported_failure_response(
        f"{exc.endpoint} answered with a body that is not an OpenRouter response "
        f"(Content-Type: {exc.content_type})",
        code=code,
        stream=stream,
        path=path,
    )


def _api_caller_error_response(
    exc: OpenRouterAPIError, *, stream: bool, path: str
) -> StreamingResponse | None:
    return _transported_failure_response(
        exc.openrouter_message or exc.reason,
        code=exc.status,
        stream=stream,
        path=path,
        retry_after_seconds=_resolve_retry_after_seconds(exc.metadata),
    )


class ErrorFormatter:
    """Handles error formatting, template selection, and emission."""

    def __init__(
        self,
        pipe: Pipe,
        event_emitter_handler: EventEmitterHandler,
        logger: logging.Logger,
    ):
        self._pipe = pipe
        self._event_emitter_handler = event_emitter_handler
        self.logger = logger

    @property
    def valves(self) -> Any:
        return self._pipe.valves

    # ======================================================================
    # Error Emission Methods
    # ======================================================================

    def _safe_detail(self, reason: str) -> str:
        return reason

    async def _emit_error(
        self,
        event_emitter: EventEmitter | None,
        error_obj: Exception | str,
        *,
        show_error_message: bool = True,
        show_error_log_citation: bool = False,
        done: bool = False,
        terminal: bool = True,
        partial_answer: str = "",
    ) -> str:
        if not self._event_emitter_handler:
            return ""
        return await self._event_emitter_handler._emit_error_event(
            event_emitter,
            error_obj,
            show_error_message=show_error_message,
            show_error_log_citation=show_error_log_citation,
            done=done,
            terminal=terminal,
            partial_answer=partial_answer,
        )

    async def _emit_templated_error(
        self,
        event_emitter: EventEmitter | None,
        *,
        template: str,
        variables: dict[str, Any],
        log_message: str,
        log_level: int = logging.ERROR,
        terminal: bool = True,
        partial_answer: str = "",
        fallback_template: str | None = None,
    ) -> str:
        if not self._event_emitter_handler:
            return ""
        return await self._event_emitter_handler._emit_templated_error_event(
            event_emitter,
            template=template,
            variables=variables,
            log_message=log_message,
            log_level=log_level,
            terminal=terminal,
            partial_answer=partial_answer,
            fallback_template=fallback_template,
        )

    def _build_error_context(self) -> tuple[str, dict[str, Any]]:
        if not self._event_emitter_handler:
            return "", {}
        return self._event_emitter_handler._create_error_context()

    # ======================================================================
    # Template Selection
    # ======================================================================

    def _select_openrouter_template(self, status: int | None, *, content_decision: bool = False) -> str:
        if content_decision:
            return self.valves.OPENROUTER_ERROR_TEMPLATE
        if status == 401:
            return self.valves.AUTHENTICATION_ERROR_TEMPLATE
        if status == 402:
            return self.valves.INSUFFICIENT_CREDITS_TEMPLATE
        if status == 408:
            return self.valves.SERVER_TIMEOUT_TEMPLATE
        if status == 413:
            return self.valves.PAYLOAD_TOO_LARGE_TEMPLATE
        if status == 429:
            return self.valves.RATE_LIMIT_TEMPLATE
        if status is not None and status >= 500:
            return self.valves.SERVICE_ERROR_TEMPLATE
        return self.valves.OPENROUTER_ERROR_TEMPLATE

    # ======================================================================
    # Error Building and Extraction
    # ======================================================================

    def _build_streaming_openrouter_error(
        self,
        event: dict[str, Any],
        *,
        requested_model: str | None,
    ) -> OpenRouterAPIError:
        """Normalize SSE error events into an OpenRouterAPIError."""
        # Runtime import to avoid circular dependency
        from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError

        response_value = event.get("response")
        response_block: dict[str, Any] = response_value if isinstance(response_value, dict) else {}
        error_value = event.get("error")
        error_block = error_value if isinstance(error_value, dict) else None
        if not error_block and isinstance(response_block.get("error"), dict):
            error_block = response_block.get("error")
        choice_error = _choice_error(event)
        if not error_block:
            error_block = choice_error if isinstance(choice_error, dict) else None
        message = ""
        for candidate in (error_value, response_block.get("error"), choice_error):
            if not _error_is_present(candidate):
                continue
            if isinstance(candidate, dict):
                text = (_as_text(candidate.get("message")) or "").strip()
            else:
                text = str(candidate).strip()
            if text:
                message = text
                break
        if not message and isinstance(response_block.get("error"), dict):
            message = (_as_text(response_block.get("error", {}).get("message")) or "").strip()
        if not message:
            message = (_as_text(event.get("message")) or "").strip() or "Streaming error"
        code = error_block.get("code") if isinstance(error_block, dict) else None
        choices = event.get("choices") or response_block.get("choices")
        native_finish_reason = None
        if isinstance(choices, list) and choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            native_finish_reason = first_choice.get("native_finish_reason") or first_choice.get("finish_reason")
        chunk_id = event.get("id") or response_block.get("id")
        chunk_created = event.get("created") or response_block.get("created")
        chunk_model = _as_text(event.get("model")) or _as_text(response_block.get("model"))
        chunk_provider = _as_text(event.get("provider")) or _as_text(response_block.get("provider"))
        error_metadata_value = error_block.get("metadata") if isinstance(error_block, dict) else None
        error_metadata: dict[str, Any] = error_metadata_value if isinstance(error_metadata_value, dict) else {}
        metadata: dict[str, Any] = {
            **error_metadata,
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
        error_type = str(error_metadata.get("error_type") or "")
        if not error_type and isinstance(error_block, dict):
            error_type = str(error_block.get("error_type") or "")
        if not error_type:
            error_type = str(event.get("error_type") or response_block.get("error_type") or "")
        reasons = error_metadata.get("reasons")
        raw_body = _pretty_json(event)
        return OpenRouterAPIError(
            status=_resolved_error_status(code, error_type, 400, from_wire=False),
            openrouter_error_type=error_type or None,
            reason=message,
            provider=chunk_provider or _as_text(error_metadata.get("provider_name")),
            openrouter_message=message,
            openrouter_code=code,
            upstream_message=message,
            upstream_type=(str(code) if code is not None else "") or _as_text(event.get("type")) or "stream_error",
            request_id=next(
                (
                    candidate
                    for candidate in (
                        response_block.get("id"),
                        event.get("response_id"),
                        event.get("request_id"),
                        chunk_id,
                    )
                    if isinstance(candidate, str) and candidate.strip()
                ),
                None,
            ),
            raw_body=raw_body,
            metadata=metadata,
            moderation_reasons=[str(reason) for reason in reasons if reason] if isinstance(reasons, list) else [],
            flagged_input=_as_text(error_metadata.get("flagged_input")),
            model_slug=chunk_model or _as_text(error_metadata.get("model_slug")),
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

    def _extract_streaming_error_event(
        self,
        event: dict[str, Any] | None,
        requested_model: str | None,
    ) -> OpenRouterAPIError | None:
        """Return an OpenRouterAPIError for SSE error payloads, if present."""
        if not isinstance(event, dict):
            return None
        event_data: dict[str, Any] = event
        event_type = (_as_text(event_data.get("type")) or "").strip()
        response_raw = event_data.get("response")
        response_block = response_raw if isinstance(response_raw, dict) else None
        error_raw = event_data.get("error")
        has_error = _error_is_present(error_raw) or _error_is_present(_choice_error(event_data))
        if isinstance(response_block, dict) and (
            response_block.get("status") == "failed"
            or _error_is_present(response_block.get("error"))
        ):
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
        exc: OpenRouterAPIError,
        *,
        event_emitter: EventEmitter | None,
        normalized_model_id: str | None,
        api_model_id: str | None,
        usage: dict[str, Any] | None = None,
        partial_answer: str = "",
        terminal: bool = True,
    ) -> str:
        """Emit a user-facing markdown message for OpenRouter 400 responses."""
        if is_sign_in_failure(exc):
            self._pipe._note_auth_failure()
        error_id, context_defaults = self._build_error_context()
        template_to_use = self._select_openrouter_template(
            exc.status,
            content_decision=_carries_a_content_decision(exc) and not is_sign_in_failure(exc),
        )
        retry_after_hint = _resolve_retry_after_seconds(exc.metadata)
        if retry_after_hint is not None and context_defaults.get("retry_after_seconds") is None:
            context_defaults["retry_after_seconds"] = retry_after_hint
        self.logger.warning(
            "[%s] OpenRouter rejected the request: %s",
            error_id,
            _provider_log_subject(exc),
        )
        model_display, diagnostics, metrics = _resolve_error_model_context(
            exc,
            normalized_model_id=normalized_model_id,
            api_model_id=api_model_id,
        )
        on_channel = is_channel_chat(OWUI_CHAT_ID.get()) if event_emitter else False
        content = exc.to_markdown(
            model_label=model_display,
            diagnostics=diagnostics or None,
            fallback_model=api_model_id or normalized_model_id,
            template=template_to_use or _FALLBACK_ERROR_TEMPLATE,
            metrics=metrics,
            normalized_model_id=normalized_model_id,
            api_model_id=api_model_id,
            context=context_defaults,
            shared=on_channel,
        )
        if not content.strip():
            content = exc.to_markdown(
                model_label=model_display,
                diagnostics=diagnostics or None,
                fallback_model=api_model_id or normalized_model_id,
                template=_FALLBACK_ERROR_TEMPLATE,
                metrics=metrics,
                normalized_model_id=normalized_model_id,
                api_model_id=api_model_id,
                context=context_defaults,
                shared=on_channel,
            )
        shown = join_answer_and_card(partial_answer, content)
        if not event_emitter:
            return shown
        try:
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "description": "Encountered a provider error. See details below.",
                        "done": True,
                    },
                }
            )
            continuing = CONTINUED_REPLY.get() is not None
            if not continuing:
                await event_emitter({"type": "chat:message", "data": {"content": shown}})
            if on_channel or continuing:
                await event_emitter({
                    "type": "chat:message:error",
                    "data": {"error": {"content": shown}, "done": True},
                })
            await self._pipe._event_emitter_handler._emit_completion(
                event_emitter,
                content=shown if on_channel else "",
                usage=usage or None,
                done=terminal,
            )
        except Exception:
            self.logger.exception(
                "[%s] Failed to emit OpenRouter error report", error_id
            )
        return shown

    # ======================================================================
    # Status Formatting
    # ======================================================================

    def _format_final_status_description(
        self,
        *,
        elapsed: float,
        total_usage: dict[str, Any],
        valves: Pipe.Valves,
        stream_duration: float | None = None,
    ) -> str:
        """Return the final status line respecting valve + available metrics.

        ``stream_duration`` is expected to mirror provider dashboards (request
        start -> last output event, which includes first-token latency).
        """
        default_description = f"Thought for {elapsed:.1f} seconds"
        if not valves.SHOW_FINAL_USAGE_STATUS:
            return default_description

        status_style = valves.FINAL_USAGE_STATUS_STYLE
        use_icons = status_style == "icons"
        icons: list[str] = list(_DEFAULT_USAGE_STATUS_ICONS)
        if use_icons:
            raw_icon_set = valves.USAGE_STATUS_ICON_SET
            if raw_icon_set:
                provided = [item.strip() for item in raw_icon_set.split(",")]
                for idx, icon in enumerate(provided[: len(icons)]):
                    if icon:
                        icons[idx] = icon

        icon_time, icon_cost, icon_total, icon_input, icon_output, icon_cached, icon_reasoning = icons

        usage = total_usage or {}
        if use_icons and icon_time:
            time_segment = f"{icon_time} {elapsed:.2f}s"
        else:
            time_segment = f"Time: {elapsed:.2f}s"
        tokens_for_tps: int | None = None
        segments: list[str] = []

        cost_segment = _cost_segment(usage.get("cost"), use_icons=use_icons, icon_cost=icon_cost)
        if cost_segment is not None:
            segments.append(cost_segment)

        def _to_int(value: Any) -> int | None:
            """Best-effort conversion to ``int`` for usage counters."""
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value) if math.isfinite(value) else None
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

        input_detail = usage.get("input_tokens_details")
        output_detail = usage.get("output_tokens_details")
        cached_tokens = _to_int(input_detail.get("cached_tokens")) if isinstance(input_detail, dict) else None
        reasoning_tokens = _to_int(output_detail.get("reasoning_tokens")) if isinstance(output_detail, dict) else None

        def _token_detail(label: str, icon: str, value: int) -> str:
            if use_icons and icon:
                return f"{icon} {value}"
            return f"{label}: {value}"

        token_details: list[str] = []
        if input_tokens is not None:
            token_details.append(_token_detail("Input", icon_input, input_tokens))
        if output_tokens is not None:
            token_details.append(_token_detail("Output", icon_output, output_tokens))
        if cached_tokens is not None and cached_tokens > 0:
            token_details.append(_token_detail("Cached", icon_cached, cached_tokens))
        if reasoning_tokens is not None and reasoning_tokens > 0:
            token_details.append(_token_detail("Reasoning", icon_reasoning, reasoning_tokens))

        if total_tokens is not None:
            if use_icons and icon_total:
                token_segment = f"{icon_total} {total_tokens}"
            else:
                token_segment = f"Total tokens: {total_tokens}"
            if token_details:
                token_segment += f" ({', '.join(token_details)})"
            segments.append(token_segment)
        elif token_details:
            if use_icons and icon_total:
                segments.append(f"{icon_total} " + ", ".join(token_details))
            else:
                segments.append("Tokens: " + ", ".join(token_details))

        if (
            tokens_for_tps is not None
            and stream_duration is not None
            and stream_duration > 0
        ):
            tokens_per_second = tokens_for_tps / stream_duration
            if tokens_per_second > 0:
                time_segment = f"{time_segment}  {tokens_per_second:.1f} tps"

        segments.insert(0, time_segment)

        description = " | ".join(segments)
        return description or default_description
