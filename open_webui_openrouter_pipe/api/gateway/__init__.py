"""OpenRouter API gateway adapters.

This module provides adapters for OpenRouter's API endpoints:
- ChatCompletionsAdapter: /api/v1/chat/completions
- ResponsesAdapter: /api/v1/responses
"""

from __future__ import annotations

from .chat_completions_adapter import ChatCompletionsAdapter
from .responses_adapter import ResponsesAdapter

__all__ = [
    "ChatCompletionsAdapter",
    "ResponsesAdapter",
]
