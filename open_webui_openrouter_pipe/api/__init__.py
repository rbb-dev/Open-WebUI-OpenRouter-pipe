"""API subsystem.

This module provides API integration with OpenRouter:
- Gateway adapters for ChatCompletions and Responses endpoints
- API format transforms (Responses <-> Chat)
"""

from __future__ import annotations

from .transforms import ResponsesBody, CompletionsBody

# Gateway adapters are accessed via api.gateway subpackage

__all__ = [
    "ResponsesBody",
    "CompletionsBody",
]
