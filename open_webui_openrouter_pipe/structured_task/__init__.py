"""Structured-output task model invocation via OWUI's generate_chat_completion.

Distinct from `requests.task_model_adapter.TaskModelAdapter` (which handles
OpenRouter Responses housekeeping like title/tag generation). This package
invokes OWUI's chat-completion path with JSON-schema strict response_format
for structured-output features (video intent classifier; future image
intent classification).

Pattern reference: .external/seedream.py (read-only).
"""
from __future__ import annotations

from .client import (
    consume_sse_line,
    normalise_model_content,
    read_model_response_content,
    read_task_model_response_json,
)
from .logging import safe_log_payload
from .orchestrator import (
    TaskModelFallback,
    TaskModelMode,
    resolve_task_model_candidates,
)
from .retry import call_with_candidates
from .schema import build_response_format, downgrade_strict_for_provider

__all__ = [
    "TaskModelFallback",
    "TaskModelMode",
    "build_response_format",
    "call_with_candidates",
    "consume_sse_line",
    "downgrade_strict_for_provider",
    "normalise_model_content",
    "read_model_response_content",
    "read_task_model_response_json",
    "resolve_task_model_candidates",
    "safe_log_payload",
]
