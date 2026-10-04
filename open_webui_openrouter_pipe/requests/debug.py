"""Debug utilities for request/response logging.

These helpers are used for logging sanitized OpenRouter request/response data.

Important: callers must pass the per-request pipe logger (SessionLogger-backed)
so records are captured into per-message session archives. These helpers do not
fall back to any global logger.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..core.logging_system import SessionLogger, bounded_log_record_text


def _debug_print_request(
    headers: dict[str, str],
    payload: dict[str, Any] | None,
    *,
    logger: logging.Logger,
) -> None:
    """Log sanitized request metadata when DEBUG logging is enabled."""
    # Late import for test compatibility (allows monkeypatching)
    from ..core.config import _owui_forwarded_header_names
    from ..core.utils import _redact_payload_blobs

    if not SessionLogger.debug_enabled(logger):
        return

    try:
        redacted_headers = dict(headers or {})
        forwarded = _owui_forwarded_header_names()
        for key in list(redacted_headers):
            lowered = key.lower()
            if lowered == "authorization" or lowered.endswith("-authorization"):
                token = redacted_headers[key]
                redacted_headers[key] = f"{token[:10]}..." if len(token) > 10 else "***"
            elif lowered in forwarded:
                redacted_headers[key] = "***"
        logger.debug(
            "OpenRouter request headers: %s",
            bounded_log_record_text(json.dumps(redacted_headers, indent=2)),
        )
        if payload is not None:
            redacted_payload = _redact_payload_blobs(payload)
            logger.debug(
                "OpenRouter request payload: %s",
                bounded_log_record_text(json.dumps(redacted_payload, indent=2)),
            )
    except Exception:
        # Never allow debug logging helpers to break request handling.
        logger.debug("OpenRouter request debug logging failed", exc_info=True)


def _debug_print_response(payload: Any, *, logger: logging.Logger) -> None:
    """Log sanitized success response payload when DEBUG logging is enabled."""
    from ..core.utils import _redact_payload_blobs

    if not SessionLogger.debug_enabled(logger):
        return
    try:
        redacted = _redact_payload_blobs(payload)
        logger.debug(
            "OpenRouter response payload: %s",
            bounded_log_record_text(json.dumps(redacted, indent=2, ensure_ascii=False)),
        )
    except Exception:
        logger.debug("OpenRouter response debug logging failed", exc_info=True)


def _scrubbed_error_body(text: str) -> Any:
    from ..core.utils import _data_url_log_subject, _redact_payload_blobs

    def _subjects(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _subjects(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_subjects(v) for v in value]
        if isinstance(value, str):
            return _data_url_log_subject(value)
        return value

    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError):
        return _data_url_log_subject(_redact_payload_blobs(text))
    return _subjects(_redact_payload_blobs(parsed))


async def _debug_print_error_response(resp: Any, *, logger: logging.Logger) -> str:
    """Log the response payload and return the response body for debugging.

    Args:
        resp: aiohttp.ClientResponse object

    Returns:
        str: Response body text or error message
    """
    if not SessionLogger.debug_enabled(logger):
        try:
            return await resp.text()
        except Exception as exc:
            logger.warning(
                "Could not read the body of an OpenRouter error response; the failure "
                "detail is lost",
                exc_info=True,
            )
            return f"<<failed to read body: {exc}>>"

    try:
        try:
            text = await resp.text()
        except Exception as exc:
            logger.debug("could not read the error response body", exc_info=True)
            text = f"<<failed to read body: {exc}>>"
        payload = {
            "status": getattr(resp, "status", None),
            "reason": getattr(resp, "reason", None),
            "url": str(getattr(resp, "url", "")),
            "body": _scrubbed_error_body(text),
        }
        logger.debug(
            "OpenRouter error response: %s",
            bounded_log_record_text(json.dumps(payload, indent=2, ensure_ascii=False)),
        )
        return text
    except Exception:
        logger.debug("OpenRouter error response debug logging failed", exc_info=True)
        try:
            return await resp.text()
        except Exception as exc:
            logger.debug("could not read the error response body", exc_info=True)
            return f"<<failed to read body: {exc}>>"

