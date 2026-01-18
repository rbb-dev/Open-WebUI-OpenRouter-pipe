"""Debug utilities for request/response logging.

Provides helpers for logging sanitized request and response data
when DEBUG logging is enabled.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from ..core.timing_logger import timed


@timed
def _debug_print_request(headers: Dict[str, str], payload: Optional[Dict[str, Any]]) -> None:
    """Log sanitized request metadata when DEBUG logging is enabled."""
    # Late import for test compatibility (allows monkeypatching)
    from ..core.config import LOGGER
    from ..core.utils import _redact_payload_blobs

    redacted_headers = dict(headers or {})
    if "Authorization" in redacted_headers:
        token = redacted_headers["Authorization"]
        redacted_headers["Authorization"] = f"{token[:10]}..." if len(token) > 10 else "***"
    LOGGER.debug("OpenRouter request headers: %s", json.dumps(redacted_headers, indent=2))
    if payload is not None:
        redacted_payload = _redact_payload_blobs(payload)
        LOGGER.debug("OpenRouter request payload: %s", json.dumps(redacted_payload, indent=2))


@timed
async def _debug_print_error_response(resp: Any) -> str:
    """Log the response payload and return the response body for debugging.

    Args:
        resp: aiohttp.ClientResponse object

    Returns:
        str: Response body text or error message
    """
    # Late import for test compatibility (allows monkeypatching)
    from ..core.config import LOGGER

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
