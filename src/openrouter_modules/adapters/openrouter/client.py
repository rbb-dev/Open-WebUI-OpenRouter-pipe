"""OpenRouter API HTTP client adapter.

This module provides non-streaming HTTP client for OpenRouter's Responses API:
- Blocking JSON requests with retry logic
- Exponential backoff on transient failures
- Error handling and classification
- Request/response logging

Layer: adapters (integrates with external OpenRouter service)

Dependencies:
- aiohttp: Async HTTP client
- tenacity: Retry logic with exponential backoff
- .models: build_openrouter_api_error, debug_print_request, debug_print_error_response
"""

from __future__ import annotations

from typing import Any, Dict

import aiohttp
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .models import (
    build_openrouter_api_error,
    debug_print_request,
    debug_print_error_response,
)

# OpenRouter metadata
_OPENROUTER_TITLE = "OpenRouter Responses Pipe"


async def send_nonstreaming_request(
    session: aiohttp.ClientSession,
    request_params: dict[str, Any],
    api_key: str,
    base_url: str,
) -> Dict[str, Any]:
    """Send a blocking request to the Responses API and return the JSON payload.

    Args:
        session: aiohttp client session
        request_params: JSON payload for /responses endpoint
        api_key: OpenRouter API key
        base_url: OpenRouter base URL (e.g., https://openrouter.ai/api/v1)

    Returns:
        Parsed JSON response from OpenRouter

    Raises:
        OpenRouterAPIError: On HTTP 400 errors (user/provider issues)
        RuntimeError: On non-retryable HTTP errors (4xx except 400)
        aiohttp.ClientError: On network failures after retries
        asyncio.TimeoutError: On timeout after retries

    Retry Strategy:
        - 3 attempts with exponential backoff (0.5s, 1s, 2s, 4s max)
        - Retries on: aiohttp.ClientError, asyncio.TimeoutError
        - No retry on: HTTP 400 (raises OpenRouterAPIError immediately)
        - No retry on: HTTP 4xx except 400 (raises RuntimeError immediately)

    Example:
        >>> async with aiohttp.ClientSession() as session:
        ...     response = await send_nonstreaming_request(
        ...         session,
        ...         {"model": "openai/gpt-4", "input": "Hello", "stream": False},
        ...         api_key="sk-...",
        ...         base_url="https://openrouter.ai/api/v1",
        ...     )
        ...     print(response)
        {'id': '...', 'output': [...], 'usage': {...}}
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": _OPENROUTER_TITLE,
    }
    debug_print_request(headers, request_params)
    url = base_url.rstrip("/") + "/responses"

    retryer = AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((aiohttp.ClientError, Exception)),
        reraise=True,
    )

    async for attempt in retryer:
        with attempt:
            async with session.post(url, json=request_params, headers=headers) as resp:
                if resp.status >= 400:
                    error_body = await debug_print_error_response(resp)
                    if resp.status < 500:
                        if resp.status == 400:
                            raise build_openrouter_api_error(
                                resp.status,
                                resp.reason or "Bad Request",
                                error_body,
                                requested_model=request_params.get("model"),
                            )
                        raise RuntimeError(
                            f"OpenRouter request failed ({resp.status}): {resp.reason}"
                        )
                resp.raise_for_status()
                return await resp.json()

    # Should never reach here due to reraise=True, but for type checker
    raise RuntimeError("Retry logic failed unexpectedly")


__all__ = ["send_nonstreaming_request"]
