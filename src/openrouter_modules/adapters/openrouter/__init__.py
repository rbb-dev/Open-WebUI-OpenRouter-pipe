"""OpenRouter API adapters.

HTTP client, streaming, and error handling for OpenRouter's Responses API.
"""

from .models import (
    OpenRouterAPIError,
    build_openrouter_api_error,
    format_openrouter_error_markdown,
    debug_print_request,
    debug_print_error_response,
)

__all__ = [
    "OpenRouterAPIError",
    "build_openrouter_api_error",
    "format_openrouter_error_markdown",
    "debug_print_request",
    "debug_print_error_response",
]
