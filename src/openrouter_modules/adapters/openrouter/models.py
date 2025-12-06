"""OpenRouter API models and error handling.

This module provides OpenRouter-specific data models and error handling:
- OpenRouterAPIError exception with structured metadata
- Error response parsing and normalization
- Debug helpers for request/response logging
- Utility functions for JSON parsing

Layer: adapters (OpenRouter-specific models)

Dependencies:
- domain.registry: ModelFamily for error context
- core.errors: render_error_template for markdown formatting

This module contains DTOs and error types specific to OpenRouter's API.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import aiohttp

if TYPE_CHECKING:
    from ...domain.registry import ModelFamily

# Module logger
LOGGER = logging.getLogger(__name__)


def _safe_json_loads(payload: Optional[str]) -> Any:
    """Return parsed JSON or None without raising.

    Args:
        payload: JSON string to parse

    Returns:
        Parsed JSON object or None if parsing fails

    Example:
        >>> result = _safe_json_loads('{"key": "value"}')
        >>> result["key"]
        'value'
        >>> _safe_json_loads("invalid json")
        None
    """
    if not payload:
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


def _normalize_optional_str(value: Any) -> Optional[str]:
    """Convert arbitrary input into a trimmed string or None.

    Args:
        value: Any value to normalize

    Returns:
        Trimmed string or None if empty

    Example:
        >>> _normalize_optional_str("  hello  ")
        'hello'
        >>> _normalize_optional_str("")
        None
        >>> _normalize_optional_str(None)
        None
    """
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def _normalize_string_list(value: Any) -> list[str]:
    """Return a list of trimmed strings.

    Args:
        value: List or other value to normalize

    Returns:
        List of non-empty trimmed strings

    Example:
        >>> _normalize_string_list(["  a  ", "", "b"])
        ['a', 'b']
        >>> _normalize_string_list(None)
        []
    """
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for entry in value:
        text = _normalize_optional_str(entry)
        if text:
            items.append(text)
    return items


def _extract_openrouter_error_details(body_text: Optional[str]) -> dict[str, Any]:
    """Normalize OpenRouter error payloads into structured metadata.

    Parses OpenRouter's error response format:
    ```json
    {
      "error": {
        "message": "...",
        "code": "...",
        "metadata": {
          "provider_name": "...",
          "raw": "..." or {...},
          "reasons": [...],
          ...
        }
      }
    }
    ```

    Args:
        body_text: Raw JSON response body from OpenRouter

    Returns:
        Dictionary with normalized error fields:
            - provider: Provider name (e.g., "OpenAI", "Anthropic")
            - openrouter_message: OpenRouter's wrapper message
            - openrouter_code: OpenRouter's error code
            - upstream_message: Provider's original error message
            - upstream_type: Provider's error type
            - request_id: Request ID for debugging
            - raw_body: Original response body
            - metadata: Full metadata dict
            - moderation_reasons: List of moderation flags
            - flagged_input: Flagged content snippet
            - model_slug: Affected model identifier

    Example:
        >>> details = _extract_openrouter_error_details('{"error": {"message": "Bad request"}}')
        >>> details["openrouter_message"]
        'Bad request'
    """
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
        "moderation_reasons": _normalize_string_list(metadata_dict.get("reasons")),
        "flagged_input": _normalize_optional_str(metadata_dict.get("flagged_input")),
        "model_slug": _normalize_optional_str(metadata_dict.get("model_slug")),
    }


def build_openrouter_api_error(
    status: int,
    reason: str,
    body_text: Optional[str],
    *,
    requested_model: Optional[str] = None,
) -> "OpenRouterAPIError":
    """Create a structured error wrapper for OpenRouter 4xx responses.

    Args:
        status: HTTP status code
        reason: HTTP reason phrase
        body_text: Raw response body JSON
        requested_model: Model ID from the original request

    Returns:
        OpenRouterAPIError with parsed metadata

    Example:
        >>> error = build_openrouter_api_error(
        ...     400,
        ...     "Bad Request",
        ...     '{"error": {"message": "Invalid parameter"}}',
        ...     requested_model="openai/gpt-4"
        ... )
        >>> str(error)
        'Invalid parameter'
    """
    details = _extract_openrouter_error_details(body_text)
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
        metadata=details.get("metadata") or {},
        moderation_reasons=details.get("moderation_reasons") or [],
        flagged_input=details.get("flagged_input"),
        model_slug=details.get("model_slug"),
        requested_model=requested_model,
    )


class OpenRouterAPIError(RuntimeError):
    """User-facing error raised when OpenRouter rejects a request with status 400.

    Attributes:
        status: HTTP status code
        reason: HTTP reason phrase
        provider: Provider name (e.g., "OpenAI")
        openrouter_message: OpenRouter's wrapper message
        openrouter_code: OpenRouter's error code
        upstream_message: Provider's original error message
        upstream_type: Provider's error type
        request_id: Request ID for debugging
        raw_body: Original response body
        metadata: Full metadata dictionary
        moderation_reasons: List of moderation flags
        flagged_input: Flagged content snippet
        model_slug: Affected model identifier
        requested_model: Model ID from original request

    Example:
        >>> try:
        ...     raise OpenRouterAPIError(
        ...         status=400,
        ...         reason="Bad Request",
        ...         openrouter_message="Invalid parameter: temperature must be between 0 and 2"
        ...     )
        ... except OpenRouterAPIError as e:
        ...     print(e.status, e.openrouter_message)
        400 Invalid parameter: temperature must be between 0 and 2
    """

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
    ) -> str:
        """Return a user-friendly markdown block describing the failure.

        Args:
            model_label: Display name for the model
            diagnostics: List of diagnostic lines (markdown bullet points)
            fallback_model: Model ID to use if label not available
            template: Custom error template string
            metrics: Model metrics (context_limit, max_output_tokens)
            normalized_model_id: Normalized model ID
            api_model_id: API-specific model ID

        Returns:
            Markdown-formatted error message

        Example:
            >>> error = OpenRouterAPIError(
            ...     status=400,
            ...     reason="Bad Request",
            ...     provider="OpenAI",
            ...     upstream_message="Invalid temperature"
            ... )
            >>> markdown = error.to_markdown(
            ...     model_label="GPT-4",
            ...     diagnostics=["- Context: 8192 tokens"]
            ... )
            >>> "GPT-4" in markdown
            True
        """
        from ...core.errors import render_error_template, DEFAULT_OPENROUTER_ERROR_TEMPLATE

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
        )
        return render_error_template(template or DEFAULT_OPENROUTER_ERROR_TEMPLATE, replacements)


def _build_error_template_values(
    error: OpenRouterAPIError,
    *,
    heading: str,
    diagnostics: List[str],
    metrics: Dict[str, Any],
    model_identifier: Optional[str] = None,
    normalized_model_id: Optional[str] = None,
    api_model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build template replacement values for error rendering.

    Args:
        error: OpenRouterAPIError instance
        heading: Display heading (provider + model)
        diagnostics: List of diagnostic lines
        metrics: Model metrics dict
        model_identifier: Model identifier string
        normalized_model_id: Normalized model ID
        api_model_id: API-specific model ID

    Returns:
        Dictionary of template replacement values
    """
    return {
        "heading": heading,
        "provider": error.provider or "",
        "model": model_identifier or "",
        "normalized_model_id": normalized_model_id or "",
        "api_model_id": api_model_id or "",
        "error_message": error.upstream_message or error.openrouter_message or "",
        "error_type": error.upstream_type or "",
        "error_code": str(error.openrouter_code) if error.openrouter_code else "",
        "request_id": error.request_id or "",
        "diagnostics": "\n".join(diagnostics) if diagnostics else "",
        "context_limit": metrics.get("context_limit", ""),
        "max_output_tokens": metrics.get("max_output_tokens", ""),
        "moderation_reasons": ", ".join(error.moderation_reasons) if error.moderation_reasons else "",
        "flagged_input": error.flagged_input or "",
    }


def _resolve_error_model_context(
    error: OpenRouterAPIError,
    *,
    normalized_model_id: Optional[str] = None,
    api_model_id: Optional[str] = None,
) -> tuple[Optional[str], list[str], dict[str, Any]]:
    """Return (display_label, diagnostics_lines, metrics) for the affected model.

    Args:
        error: OpenRouterAPIError instance
        normalized_model_id: Normalized model ID
        api_model_id: API-specific model ID

    Returns:
        Tuple of (display_label, diagnostics, metrics)
            - display_label: Human-readable model name
            - diagnostics: List of diagnostic markdown lines
            - metrics: Dict with context_limit and max_output_tokens

    Example:
        >>> error = OpenRouterAPIError(status=400, reason="Bad Request")
        >>> label, diag, metrics = _resolve_error_model_context(
        ...     error,
        ...     normalized_model_id="openai.gpt-4"
        ... )
        >>> isinstance(diag, list)
        True
    """
    from ...domain.registry import ModelFamily

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


def format_openrouter_error_markdown(
    error: OpenRouterAPIError,
    *,
    normalized_model_id: Optional[str],
    api_model_id: Optional[str],
    template: str,
) -> str:
    """Render a provider error into markdown after enriching with model context.

    Args:
        error: OpenRouterAPIError instance
        normalized_model_id: Normalized model ID
        api_model_id: API-specific model ID
        template: Error template string

    Returns:
        Rendered markdown error message

    Example:
        >>> error = OpenRouterAPIError(status=400, reason="Bad Request")
        >>> markdown = format_openrouter_error_markdown(
        ...     error,
        ...     normalized_model_id="openai.gpt-4",
        ...     api_model_id="openai/gpt-4",
        ...     template="Error: {error_message}"
        ... )
        >>> isinstance(markdown, str)
        True
    """
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
    )


def debug_print_request(headers: Dict[str, str], payload: Optional[Dict[str, Any]]) -> None:
    """Log sanitized request metadata when DEBUG logging is enabled.

    Redacts Authorization header to prevent credential leakage.

    Args:
        headers: HTTP headers dictionary
        payload: Request payload dictionary

    Example:
        >>> debug_print_request(
        ...     {"Authorization": "Bearer sk-1234567890abcdef"},
        ...     {"model": "gpt-4"}
        ... )
        # Logs with redacted auth
    """
    redacted_headers = dict(headers or {})
    if "Authorization" in redacted_headers:
        token = redacted_headers["Authorization"]
        redacted_headers["Authorization"] = f"{token[:10]}..." if len(token) > 10 else "***"
    LOGGER.debug("OpenRouter request headers: %s", json.dumps(redacted_headers, indent=2))
    if payload is not None:
        LOGGER.debug("OpenRouter request payload: %s", json.dumps(payload, indent=2))


async def debug_print_error_response(resp: aiohttp.ClientResponse) -> str:
    """Log the response payload and return the response body for debugging.

    Args:
        resp: aiohttp response object

    Returns:
        Response body text

    Example:
        >>> # In async context:
        >>> body = await debug_print_error_response(response)
        >>> isinstance(body, str)
        True
    """
    try:
        text = await resp.text()
    except Exception as exc:
        text = f"<<failed to read body: {exc}>>"
    payload = {
        "status": resp.status,
        "reason": resp.reason,
        "url": str(resp.url),
        "body": text,
    }
    LOGGER.debug("OpenRouter error response: %s", json.dumps(payload, indent=2))
    return text


__all__ = [
    "OpenRouterAPIError",
    "build_openrouter_api_error",
    "format_openrouter_error_markdown",
    "debug_print_request",
    "debug_print_error_response",
]
