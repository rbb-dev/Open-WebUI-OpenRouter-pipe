"""Request handling subsystem.

This module provides request adapters for OpenRouter API:
- NonStreamingAdapter: Handles non-streaming requests and converts to events
- TaskModelAdapter: Handles task model requests (e.g., chat titles, tags)
- RequestOrchestrator: Main request processing logic
- Debug utilities: Request/response logging helpers
"""

from __future__ import annotations

from .debug import _debug_print_error_response, _debug_print_request
from .nonstreaming_adapter import NonStreamingAdapter
from .orchestrator import RequestOrchestrator
from .task_model_adapter import TaskModelAdapter

__all__ = [
    "NonStreamingAdapter",
    "RequestOrchestrator",
    "TaskModelAdapter",
    "_debug_print_error_response",
    "_debug_print_request",
]
