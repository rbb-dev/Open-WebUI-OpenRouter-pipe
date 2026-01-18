"""Error handling and user-facing error formatting.

This module handles all error-related functionality:
- OpenRouterAPIError: Rich error class with markdown formatting
- Error template rendering
- Error classification (retryable, rate limit, etc.)
- Error context building for user display
- HTTP error handling

Responsible for translating technical errors into user-friendly markdown messages.
"""

from __future__ import annotations

import asyncio
import datetime
import email.utils
import inspect
import logging
from typing import Any, Awaitable, Literal, Optional, TypeVar, cast

import httpx

from .timing_logger import timed
from .config import (
    DEFAULT_OPENROUTER_ERROR_TEMPLATE,
    DEFAULT_NETWORK_TIMEOUT_TEMPLATE,
    DEFAULT_CONNECTION_ERROR_TEMPLATE,
    DEFAULT_SERVICE_ERROR_TEMPLATE,
    DEFAULT_INTERNAL_ERROR_TEMPLATE,
    DEFAULT_ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE,
    DEFAULT_DIRECT_UPLOAD_FAILURE_TEMPLATE,
    DEFAULT_AUTHENTICATION_ERROR_TEMPLATE,
    DEFAULT_INSUFFICIENT_CREDITS_TEMPLATE,
    DEFAULT_RATE_LIMIT_TEMPLATE,
    DEFAULT_SERVER_TIMEOUT_TEMPLATE,
    _REMOTE_FILE_MAX_SIZE_MAX_MB,
)
from .utils import (
    _render_error_template,
    _template_value_present,
    _pretty_json,
    _safe_json_loads,
    _normalize_optional_str,
    _normalize_string_list,
    _coerce_positive_int,
    _coerce_bool,
    _get_open_webui_config_module,
    _unwrap_config_value,
    _retry_after_seconds,
)

LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Supporting Classes
# -----------------------------------------------------------------------------

class _RetryableHTTPStatusError(Exception):
    """Wrapper that marks an HTTPStatusError as retryable."""

    def __init__(self, original: httpx.HTTPStatusError, retry_after: Optional[float] = None):
        """Capture the original HTTP error plus optional Retry-After hint."""
        self.original = original
        self.retry_after = retry_after
        status_code = getattr(original.response, "status_code", "unknown")
        super().__init__(f"Retryable HTTP error ({status_code})")


class _RetryWait:
    """Custom Tenacity wait strategy honoring Retry-After headers."""

    def __init__(self, base_wait):
        """Store the wrapped Tenacity wait callable used as a baseline."""
        self._base_wait = base_wait

    def __call__(self, retry_state):
        """Return the greater of base delay or Retry-After header guidance."""
        base_delay = self._base_wait(retry_state) if self._base_wait else 0
        exc = None
        if retry_state.outcome is not None:
            try:
                exc = retry_state.outcome.exception()
            except Exception:
                exc = None
        if isinstance(exc, _RetryableHTTPStatusError):
            retry_after = exc.retry_after
            if isinstance(retry_after, (int, float)) and retry_after > 0:
                return max(base_delay, retry_after)
        return base_delay


_TWait = TypeVar("_TWait")


class StatusMessages:
    """Centralized status messages for multimodal processing."""

    # Image processing
    IMAGE_BASE64_SAVED = "📥 Saved base64 image to storage"
    IMAGE_REMOTE_SAVED = "📥 Downloaded and saved image from remote URL"

    # File processing
    FILE_BASE64_SAVED = "📥 Saved base64 file to storage"
    FILE_REMOTE_SAVED = "📥 Downloaded and saved file from remote URL"

    # Video processing
    VIDEO_BASE64 = "🎥 Processing base64 video input"
    VIDEO_YOUTUBE = "🎥 Processing YouTube video input"
    VIDEO_REMOTE = "🎥 Processing video input"

    # Audio processing
    AUDIO_BASE64_SAVED = "🎵 Saved base64 audio to storage"
    AUDIO_REMOTE_SAVED = "🎵 Downloaded and saved audio from remote URL"


# -----------------------------------------------------------------------------
# OpenRouterAPIError Class
# -----------------------------------------------------------------------------

class OpenRouterAPIError(RuntimeError):
    """User-facing error raised when OpenRouter rejects a request with status 400."""

    def __init__(
        self,
        *,
        status: int,
        reason: str,
        provider: Optional[str] = None,
        openrouter_message: Optional[str] = None,
        openrouter_code: Optional[Any] = None,
        upstream_message: Optional[str] = None,
        upstream_type: Optional[str] = None,
        request_id: Optional[str] = None,
        raw_body: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        moderation_reasons: Optional[list[str]] = None,
        flagged_input: Optional[str] = None,
        model_slug: Optional[str] = None,
        requested_model: Optional[str] = None,
        metadata_json: Optional[str] = None,
        provider_raw: Optional[Any] = None,
        provider_raw_json: Optional[str] = None,
        native_finish_reason: Optional[str] = None,
        chunk_id: Optional[str] = None,
        chunk_created: Optional[Any] = None,
        chunk_provider: Optional[str] = None,
        chunk_model: Optional[str] = None,
        is_streaming_error: bool = False,
    ) -> None:
        """Normalize raw OpenRouter metadata into convenient attributes."""
        self.status = status
        self.reason = reason
        self.provider = provider
        self.openrouter_message = (openrouter_message or "").strip() or None
        self.openrouter_code = openrouter_code
        self.upstream_message = (upstream_message or "").strip() or None
        self.upstream_type = (upstream_type or "").strip() or None
        self.request_id = (request_id or "").strip() or None
        self.raw_body = raw_body or ""
        self.metadata = metadata or {}
        self.moderation_reasons = moderation_reasons or []
        self.flagged_input = (flagged_input or "").strip() or None
        self.model_slug = (model_slug or "").strip() or None
        self.requested_model = (requested_model or "").strip() or None
        self.metadata_json = metadata_json or _pretty_json(self.metadata)
        self.provider_raw = provider_raw
        self.provider_raw_json = provider_raw_json or _pretty_json(provider_raw)
        self.native_finish_reason = native_finish_reason
        self.chunk_id = chunk_id
        self.chunk_created = chunk_created
        self.chunk_provider = chunk_provider
        self.chunk_model = chunk_model
        self.is_streaming_error = is_streaming_error
        summary = (
            self.upstream_message
            or self.openrouter_message
            or f"OpenRouter request failed ({self.status} {self.reason})"
        )
        super().__init__(summary)

    def to_markdown(
        self,
        *,
        model_label: Optional[str] = None,
        diagnostics: Optional[list[str]] = None,
        fallback_model: Optional[str] = None,
        template: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
        normalized_model_id: Optional[str] = None,
        api_model_id: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Return a user-friendly markdown block describing the failure."""
        provider_label = (self.provider or "").strip()
        effective_model = model_label or fallback_model or self.model_slug or self.requested_model
        if provider_label and effective_model:
            heading = f"{provider_label}: {effective_model}"
        elif effective_model:
            heading = effective_model
        elif provider_label:
            heading = provider_label
        else:
            heading = "OpenRouter"

        replacements = _build_error_template_values(
            self,
            heading=heading,
            diagnostics=diagnostics or [],
            metrics=metrics or {},
            model_identifier=self.model_slug or self.requested_model or fallback_model,
            normalized_model_id=normalized_model_id,
            api_model_id=api_model_id,
            context=context,
        )
        return _render_error_template(template or DEFAULT_OPENROUTER_ERROR_TEMPLATE, replacements)


# -----------------------------------------------------------------------------
# Error Helper Functions
# -----------------------------------------------------------------------------

def _classify_retryable_http_error(
    exc: httpx.HTTPStatusError,
) -> tuple[bool, Optional[float]]:
    """Return (is_retryable, retry_after_seconds) for an HTTPStatusError."""
    response = exc.response
    if response is None:
        return False, None
    status_code = response.status_code
    if status_code >= 500 or status_code in {408, 425, 429}:
        return True, _retry_after_seconds(response.headers.get("retry-after"))
    return False, None


def _extract_openrouter_error_details(body_text: Optional[str]) -> dict[str, Any]:
    """Normalize OpenRouter error payloads into structured metadata."""
    parsed = _safe_json_loads(body_text) if body_text else None
    error_section = parsed.get("error", {}) if isinstance(parsed, dict) else {}
    metadata = error_section.get("metadata", {}) if isinstance(error_section, dict) else {}
    metadata_dict = metadata if isinstance(metadata, dict) else {}

    raw_meta = metadata_dict.get("raw")
    if isinstance(raw_meta, str):
        raw_details = _safe_json_loads(raw_meta)
    elif isinstance(raw_meta, dict):
        raw_details = raw_meta
    else:
        raw_details = None

    upstream_error = raw_details.get("error", {}) if isinstance(raw_details, dict) else {}
    upstream_message = (
        upstream_error.get("message")
        if isinstance(upstream_error, dict)
        else None
    ) or (raw_details.get("message") if isinstance(raw_details, dict) else None)
    upstream_type = upstream_error.get("type") if isinstance(upstream_error, dict) else None

    request_id = (
        metadata.get("request_id")
        or (raw_details.get("request_id") if isinstance(raw_details, dict) else None)
        or (parsed.get("request_id") if isinstance(parsed, dict) else None)
    )

    return {
        "provider": metadata_dict.get("provider_name") or metadata_dict.get("provider"),
        "openrouter_message": error_section.get("message"),
        "openrouter_code": error_section.get("code"),
        "upstream_message": upstream_message,
        "upstream_type": upstream_type,
        "request_id": request_id,
        "raw_body": body_text or "",
        "metadata": metadata_dict,
        "metadata_json": _pretty_json(metadata_dict),
        "moderation_reasons": _normalize_string_list(metadata_dict.get("reasons")),
        "flagged_input": _normalize_optional_str(metadata_dict.get("flagged_input")),
        "model_slug": _normalize_optional_str(metadata_dict.get("model_slug")),
        "provider_raw": raw_details,
        "provider_raw_json": _pretty_json(raw_details),
    }


def _build_openrouter_api_error(
    status: int,
    reason: str,
    body_text: Optional[str],
    *,
    requested_model: Optional[str] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> "OpenRouterAPIError":
    """Create a structured error wrapper for OpenRouter 4xx responses."""
    details = _extract_openrouter_error_details(body_text)
    metadata_block = details.get("metadata") or {}
    if extra_metadata:
        metadata_block = {**metadata_block, **extra_metadata}
    return OpenRouterAPIError(
        status=status,
        reason=reason,
        provider=details.get("provider"),
        openrouter_message=details.get("openrouter_message"),
        openrouter_code=details.get("openrouter_code"),
        upstream_message=details.get("upstream_message"),
        upstream_type=details.get("upstream_type"),
        request_id=details.get("request_id"),
        raw_body=details.get("raw_body"),
        metadata=metadata_block,
        moderation_reasons=details.get("moderation_reasons") or [],
        flagged_input=details.get("flagged_input"),
        model_slug=details.get("model_slug"),
        requested_model=requested_model,
        metadata_json=details.get("metadata_json"),
        provider_raw=details.get("provider_raw"),
        provider_raw_json=details.get("provider_raw_json"),
    )


def _is_reasoning_effort_error(error_details: dict[str, Any]) -> bool:
    """Return True when the provider rejected reasoning.effort as unsupported."""
    provider_raw = error_details.get("provider_raw")
    if not isinstance(provider_raw, dict):
        return False
    upstream_error = provider_raw.get("error", {})
    if not isinstance(upstream_error, dict):
        return False
    return (
        upstream_error.get("param") == "reasoning.effort"
        and upstream_error.get("code") == "unsupported_value"
        and upstream_error.get("type") == "invalid_request_error"
    )


def _parse_supported_effort_values(error_message: str) -> list[str]:
    """Extract the list of supported reasoning efforts from an OpenRouter error."""
    import re

    match = re.search(r"Supported values are:\s*(.+?)\.", error_message, re.IGNORECASE)
    if not match:
        return []
    values_str = match.group(1)
    return re.findall(r"'([^']+)'", values_str)


def _select_best_effort_fallback(requested: str, supported: list[str]) -> Optional[str]:
    """Choose the closest supported effort to retry with."""
    ordering = ["none", "minimal", "low", "medium", "high", "xhigh"]
    if not supported:
        return None
    requested_lower = (requested or "").strip().lower()
    supported_lower = [value.strip().lower() for value in supported if value]
    if requested_lower in supported_lower:
        return requested_lower
    try:
        requested_idx = ordering.index(requested_lower)
    except ValueError:
        return supported_lower[0] if supported_lower else None
    indexed: list[tuple[int, str]] = []
    for value in supported_lower:
        try:
            indexed.append((ordering.index(value), value))
        except ValueError:
            continue
    if not indexed:
        return supported_lower[0] if supported_lower else None
    indexed.sort()
    min_idx, min_value = indexed[0]
    max_idx, max_value = indexed[-1]
    if requested_idx <= min_idx:
        return min_value
    if requested_idx >= max_idx:
        return max_value
    for idx, value in indexed:
        if idx > requested_idx:
            return value
    closest = None
    distance = float("inf")
    for idx, value in indexed:
        delta = abs(idx - requested_idx)
        if delta < distance:
            closest = value
            distance = delta
    return closest


def _build_error_template_values(
    error: "OpenRouterAPIError",
    *,
    heading: str,
    diagnostics: list[str],
    metrics: dict[str, Any],
    model_identifier: Optional[str],
    normalized_model_id: Optional[str],
    api_model_id: Optional[str],
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Prepare placeholder values for the customizable error template."""
    detail = (error.upstream_message or error.openrouter_message or str(error)).strip()
    sanitized_detail = detail.replace("`", "\\`")

    moderation_lines = [reason for reason in (error.moderation_reasons or []) if reason]
    flagged_excerpt = (error.flagged_input or "").strip()
    raw_body = (error.raw_body or "").strip()

    detail_lower = detail.lower()
    include_model_limits = "or use the \"middle-out\"" in detail_lower
    context_limit_value = metrics.get("context_limit") if include_model_limits else None
    max_output_tokens_value = metrics.get("max_output_tokens") if include_model_limits else None
    include_model_limits = include_model_limits and bool(
        (context_limit_value or max_output_tokens_value)
    )

    metadata_json = error.metadata_json or _pretty_json(error.metadata)
    provider_raw_json = error.provider_raw_json or _pretty_json(error.provider_raw)
    context = context or {}
    retry_after = (
        context.get("retry_after_seconds")
        or error.metadata.get("retry_after_seconds")
        or error.metadata.get("retry_after")
    )
    replacements: dict[str, Any] = {
        "heading": heading,
        "detail": detail,
        "sanitized_detail": sanitized_detail,
        "provider": (error.provider or "").strip(),
        "reason": str(error),
        "raw_body": raw_body,
        "model_identifier": model_identifier or "",
        "requested_model": error.requested_model or "",
        "openrouter_code": str(error.openrouter_code or ""),
        "upstream_type": error.upstream_type or "",
        "upstream_message": (error.upstream_message or "").strip(),
        "openrouter_message": (error.openrouter_message or "").strip(),
        "request_id": error.request_id or "",
        "request_id_reference": f"Request reference: `{error.request_id}`" if error.request_id else "",
        "moderation_reasons": "\n".join(f"- {reason}" for reason in moderation_lines),
        "flagged_excerpt": flagged_excerpt,
        "context_limit_tokens": f"{context_limit_value:,}" if context_limit_value else "",
        "max_output_tokens": f"{max_output_tokens_value:,}" if max_output_tokens_value else "",
        "include_model_limits": bool(include_model_limits),
        "api_model_id": api_model_id or "",
        "normalized_model_id": normalized_model_id or "",
        "metadata_json": metadata_json,
        "provider_raw_json": provider_raw_json,
        "native_finish_reason": error.native_finish_reason or "",
        "error_chunk_id": error.chunk_id or "",
        "error_chunk_created": error.chunk_created or "",
        "streaming_provider": error.chunk_provider or (error.provider or ""),
        "streaming_model": error.chunk_model or "",
        "is_streaming_error": bool(error.is_streaming_error),
        "error_id": context.get("error_id", ""),
        "timestamp": context.get("timestamp", ""),
        "session_id": context.get("session_id", ""),
        "user_id": context.get("user_id", ""),
        "support_email": context.get("support_email", ""),
        "support_url": context.get("support_url", ""),
        "retry_after_seconds": retry_after or "",
        "rate_limit_type": error.metadata.get("rate_limit_type") or "",
        "required_cost": error.metadata.get("required_cost") or "",
        "account_balance": error.metadata.get("account_balance") or "",
        "diagnostics": "\n".join(diagnostics),
    }
    return replacements


def _resolve_error_model_context(
    error: OpenRouterAPIError,
    *,
    normalized_model_id: Optional[str] = None,
    api_model_id: Optional[str] = None,
) -> tuple[Optional[str], list[str], dict[str, Any]]:
    """Return (display_label, diagnostics_lines, metrics) for the affected model."""
    # Import here to avoid circular dependency
    from ..models.registry import ModelFamily

    diagnostics: list[str] = []
    display_label: Optional[str] = None

    norm_id = ModelFamily.base_model(normalized_model_id or "") if normalized_model_id else None
    spec = ModelFamily._lookup_spec(norm_id or "")
    full_model = spec.get("full_model") or {}

    context_limit = spec.get("context_length") or full_model.get("context_length")
    max_output_tokens = spec.get("max_completion_tokens") or full_model.get("max_completion_tokens")

    if context_limit:
        diagnostics.append(f"- **Context window**: {context_limit:,} tokens")
    if max_output_tokens:
        diagnostics.append(f"- **Max output tokens**: {max_output_tokens:,} tokens")

    display_label = (
        full_model.get("name")
        or api_model_id
        or error.model_slug
        or error.requested_model
        or normalized_model_id
    )

    return display_label, diagnostics, {
        "context_limit": context_limit,
        "max_output_tokens": max_output_tokens,
    }


def _format_openrouter_error_markdown(
    error: OpenRouterAPIError,
    *,
    normalized_model_id: Optional[str],
    api_model_id: Optional[str],
    template: str,
    context: Optional[dict[str, Any]] = None,
) -> str:
    """Render a provider error into markdown after enriching with model context."""
    model_display, diagnostics, metrics = _resolve_error_model_context(
        error,
        normalized_model_id=normalized_model_id,
        api_model_id=api_model_id,
    )
    return error.to_markdown(
        model_label=model_display,
        diagnostics=diagnostics or None,
        fallback_model=api_model_id or normalized_model_id,
        template=template,
        metrics=metrics,
        normalized_model_id=normalized_model_id,
        api_model_id=api_model_id,
        context=context,
    )


# -----------------------------------------------------------------------------
# RAG File Constraints (using imported config helpers from core.utils)
# -----------------------------------------------------------------------------

def _read_rag_file_constraints() -> tuple[bool, Optional[int]]:
    """Return (rag_enabled, rag_file_size_mb) gleaned from Open WebUI config."""
    module = _get_open_webui_config_module()
    if module is None:
        return False, None

    bypass_raw = _unwrap_config_value(getattr(module, "BYPASS_EMBEDDING_AND_RETRIEVAL", None))
    bypass_bool = _coerce_bool(bypass_raw)
    rag_enabled = True if bypass_bool is None else not bypass_bool

    limit_mb: Optional[int] = None
    for attr_name in ("RAG_FILE_MAX_SIZE", "FILE_MAX_SIZE"):
        attr_value = _unwrap_config_value(getattr(module, attr_name, None))
        limit_mb = _coerce_positive_int(attr_value)
        if limit_mb is not None:
            break

    if limit_mb is not None:
        limit_mb = min(limit_mb, _REMOTE_FILE_MAX_SIZE_MAX_MB)

    return rag_enabled, limit_mb


# -----------------------------------------------------------------------------
# Async Helper
# -----------------------------------------------------------------------------

async def _wait_for(
    value: Awaitable[_TWait] | _TWait,
    *,
    timeout: Optional[float] = None,
) -> _TWait:
    """Return ``value`` immediately when it's synchronous, otherwise await it.

    Redis' asyncio client returns synchronous fallbacks (bool/str/list) when a
    pipeline is configured for immediate execution, which caused ``await`` to be
    applied to non-awaitables.  This helper centralizes the guard so call sites
    stay tidy and Pyright no longer reports "X is not awaitable" diagnostics.
    """
    if inspect.isawaitable(value):
        coroutine = cast(Awaitable[_TWait], value)
        if timeout is None:
            return cast(_TWait, await coroutine)
        return cast(_TWait, await asyncio.wait_for(coroutine, timeout=timeout))
    return cast(_TWait, value)

