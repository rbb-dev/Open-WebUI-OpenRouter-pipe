"""Debug utilities for request/response logging.

These helpers are used for logging sanitized OpenRouter request/response data.

Important: callers must pass the per-request pipe logger (SessionLogger-backed)
so records are captured into per-message session archives. These helpers do not
fall back to any global logger.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from ..core.timing_logger import timed


def _debug_print_request(
    headers: Dict[str, str],
    payload: Optional[Dict[str, Any]],
    *,
    logger: logging.Logger,
) -> None:
    """Log sanitized request metadata when DEBUG logging is enabled."""
    # Late import for test compatibility (allows monkeypatching)
    from ..core.utils import _redact_payload_blobs

    if not logger.isEnabledFor(logging.DEBUG):
        return

    try:
        redacted_headers = dict(headers or {})
        if "Authorization" in redacted_headers:
            token = redacted_headers["Authorization"]
            redacted_headers["Authorization"] = f"{token[:10]}..." if len(token) > 10 else "***"
        logger.debug("OpenRouter request headers: %s", json.dumps(redacted_headers, indent=2))
        if payload is not None:
            redacted_payload = _redact_payload_blobs(payload)
            logger.debug("OpenRouter request payload: %s", json.dumps(redacted_payload, indent=2))
    except Exception:
        # Never allow debug logging helpers to break request handling.
        logger.debug("OpenRouter request debug logging failed", exc_info=True)


def _debug_print_response(payload: Any, *, logger: logging.Logger) -> None:
    """Log sanitized success response payload when DEBUG logging is enabled."""
    from ..core.utils import _redact_payload_blobs

    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        redacted = _redact_payload_blobs(payload) if isinstance(payload, dict) else payload
        logger.debug("OpenRouter response payload: %s", json.dumps(redacted, indent=2, ensure_ascii=False))
    except Exception:
        logger.debug("OpenRouter response debug logging failed", exc_info=True)


async def _debug_print_error_response(resp: Any, *, logger: logging.Logger) -> str:
    """Log the response payload and return the response body for debugging.

    Args:
        resp: aiohttp.ClientResponse object

    Returns:
        str: Response body text or error message
    """
    if not logger.isEnabledFor(logging.DEBUG):
        try:
            return await resp.text()
        except Exception as exc:
            return f"<<failed to read body: {exc}>>"

    try:
        try:
            text = await resp.text()
        except Exception as exc:
            text = f"<<failed to read body: {exc}>>"
        payload = {
            "status": getattr(resp, "status", None),
            "reason": getattr(resp, "reason", None),
            "url": str(getattr(resp, "url", "")),
            "body": text,
        }
        logger.debug("OpenRouter error response: %s", json.dumps(payload, indent=2))
        return text
    except Exception:
        logger.debug("OpenRouter error response debug logging failed", exc_info=True)
        try:
            return await resp.text()
        except Exception as exc:
            return f"<<failed to read body: {exc}>>"
