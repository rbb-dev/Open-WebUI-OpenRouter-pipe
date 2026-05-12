"""Structured-output task model invocation via OWUI's generate_chat_completion.

Distinct from `requests.task_model_adapter.TaskModelAdapter` (which handles
OpenRouter Responses housekeeping like title/tag generation). This package
invokes OWUI's chat-completion path with JSON-schema strict response_format
for structured-output features (video intent classifier; future image
intent classification).

Pattern reference: .external/seedream.py (read-only).
"""
from __future__ import annotations

from .orchestrator import (
    TaskModelMode,
    TaskModelFallback,
    resolve_task_model_candidates,
)
from .schema import build_response_format, downgrade_strict_for_provider
from .client import (
    normalise_model_content,
    consume_sse_line,
    read_model_response_content,
    read_task_model_response_json,
)
from .logging import safe_log_payload
from .retry import call_with_candidates

__all__ = [
    "TaskModelMode",
    "TaskModelFallback",
    "resolve_task_model_candidates",
    "build_response_format",
    "downgrade_strict_for_provider",
    "normalise_model_content",
    "consume_sse_line",
    "read_model_response_content",
    "read_task_model_response_json",
    "safe_log_payload",
    "call_with_candidates",
]
