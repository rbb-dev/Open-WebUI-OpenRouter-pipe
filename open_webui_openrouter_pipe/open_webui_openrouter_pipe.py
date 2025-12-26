"""
title: Open WebUI OpenRouter Responses Pipe
author: rbb-dev
author_url: https://github.com/rbb-dev
git_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: open_webui_openrouter_pipe
description: OpenRouter Responses API pipe for Open WebUI
required_open_webui_version: 0.6.28
version: 1.0.14
requirements: aiohttp, cryptography, fastapi, httpx, lz4, pydantic, pydantic_core, sqlalchemy, tenacity, pyzipper
license: MIT

- This work was based on excellent work by jrkropp (https://github.com/jrkropp/open-webui-developer-toolkit)
- It was adapted to be used exclusively with OpenRouter Responses API and extended with additional features.
- Auto-discovers and imports full OpenRouter Responses model catalog with capabilities and identifiers.
- Translates Completions to Responses API, persisting reasoning/tool artifacts per chat via scoped SQLAlchemy tables.
- Handles 100-500 concurrent users with per-request isolation, async queues, and global semaphores for overload protection (503 rejects).
- Non-blocking ops: Offloads sync DB to ThreadPool, async logging queue, per-request HTTP sessions with retries/breakers.
- Optional Redis cache (auto-detected via ENV/multi-worker): Write-behind with pub/sub/timed flushes, TTL for fast artifact reads.
- Secure artifact persistence: User-key encryption, LZ4 compression for large payloads, ULID markers for context replay.
- Tool execution: Per-request FIFO queues, parallel workers with semaphores/timeouts, per-user/type breakers, batching non-dependent calls.
- Streams SSE with producer-multi-consumer workers, configurable delta batching/zero-copy, inline citations, and usage metrics.
- Strictifies tool schemas (Open WebUI registry + Direct Tool Servers) for predictable function calling; deduplicates definitions.
- Auto-enables web search plugin if model-supported.
- Supports Open WebUI Direct Tool Servers (client-side OpenAPI tool servers executed via Socket.IO).
- Exposes valves for concurrency limits, logging levels, Redis/cache settings, tool timeouts, cleanup intervals, and more.
- OWUI-compatible: Uses internal sync DB, honors pipe IDs for tables, scales to multi-worker via Redis without assumptions.
"""

# TODO (Future Improvements):
# - Health endpoint: Expose metrics like active requests, error rates, queue sizes via a debug route.
# - DLQ: Implement dead-letter queues for unrecoverable errors (e.g., failed tool outputs) to debug later—evaluate if issues outweigh benefits.
# - ProcessPool Offload: For CPU-bound tools, add optional ProcessPoolExecutor integration.
# - Metrics: Add Prometheus export or simple stats (e.g., avg latency) for monitoring.
# - More...

from __future__ import annotations

_OPENROUTER_TITLE = "Open WebUI plugin for OpenRouter Responses API"
_OPENROUTER_REFERER = "https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/"
_DEFAULT_PIPE_ID = "open_webui_openrouter_pipe"
_FUNCTION_MODULE_PREFIX = "function_"
_MAX_OPENROUTER_ID_CHARS = 128
_MAX_OPENROUTER_METADATA_PAIRS = 16
_MAX_OPENROUTER_METADATA_KEY_CHARS = 64
_MAX_OPENROUTER_METADATA_VALUE_CHARS = 512

def _detect_runtime_pipe_id(default: str = _DEFAULT_PIPE_ID) -> str:
    """Infer the Open WebUI function id from the module name.

    Some loaders (Open WebUI hot reload, runpy, unit tests) execute this module via exec/run
    without setting a real __name__. Returning the default keeps imports working in those cases.
    """
    module_name = globals().get("__name__", "")
    if isinstance(module_name, str) and module_name.startswith(_FUNCTION_MODULE_PREFIX):
        candidate = module_name[len(_FUNCTION_MODULE_PREFIX) :].strip()
        if candidate:
            return candidate
    return default


_PIPE_RUNTIME_ID = _detect_runtime_pipe_id()

# Tool artifacts that shouldn't be replayed back to the provider on future turns.
# OpenAI's Responses docs only require reasoning/function-call outputs to be
# preserved. Re-sending large tool payloads (image generation blobs, web search
# telemetry, etc.) wastes context window budget and can trip provider limits.
_NON_REPLAYABLE_TOOL_ARTIFACTS = frozenset(
    {
        "image_generation_call",
        "web_search_call",
        "file_search_call",
        "local_shell_call",
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────────────────────
# Standard library, third-party, and Open WebUI imports
# Standard library imports
import asyncio
import datetime
import inspect
import json
import logging
import os
import queue
import re
import sys
import secrets
import random
import time
import hashlib
import base64
import binascii
import functools
import threading
from time import perf_counter
from collections import defaultdict, deque
from contextvars import ContextVar
import contextvars
from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Type, TypeVar, Union, TYPE_CHECKING, cast, no_type_check
from urllib.parse import urlparse
import ast
import email.utils

# Third-party imports
import aiohttp
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import Boolean, Column, DateTime, JSON, String, inspect as sa_inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

try:
    import lz4.frame as lz4frame
except ImportError:  # pragma: no cover - optional dependency, handled via requirements metadata
    lz4frame = None

try:
    import pyzipper  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - optional dependency
    pyzipper = None

try:  # optional Redis cache
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover - optional dependency
    aioredis = None

if TYPE_CHECKING:
    from redis.asyncio import Redis as _RedisClient
else:
    _RedisClient = Any

EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]
# Open WebUI internals
from open_webui.models.chats import Chats
from open_webui.models.models import ModelForm, Models
from open_webui.models.files import Files
from open_webui.models.users import Users
from open_webui.routers.files import upload_file_handler
from open_webui.utils.misc import openai_chat_chunk_message_template
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

# Additional imports for file/image handling
import io
import uuid
import httpx
from fastapi import UploadFile, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from starlette.datastructures import Headers


LOGGER = logging.getLogger(__name__)

_REASONING_STATUS_PUNCTUATION = (".", "!", "?", ":", "\n")
_REASONING_STATUS_MAX_CHARS = 160
_REASONING_STATUS_MIN_CHARS = 12
_REASONING_STATUS_IDLE_SECONDS = 0.75

_REMOTE_FILE_MAX_SIZE_DEFAULT_MB = 50
_REMOTE_FILE_MAX_SIZE_MAX_MB = 500
_INTERNAL_FILE_ID_PATTERN = re.compile(r"/files/([A-Za-z0-9-]+)/")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?P<url>[^)]+)\)")
_TEMPLATE_VAR_PATTERN = re.compile(r"{(\w+)}")
_TEMPLATE_IF_OPEN_RE = re.compile(r"\{\{\s*#if\s+(\w+)\s*\}\}")
_TEMPLATE_IF_CLOSE_RE = re.compile(r"\{\{\s*/if\s*\}\}")
_TEMPLATE_IF_TOKEN_RE = re.compile(r"\{\{\s*(#if\s+(\w+)|/if)\s*\}\}")
DEFAULT_OPENROUTER_ERROR_TEMPLATE = (
    "{{#if heading}}\n"
    "### 🚫 {heading} could not process your request.\n\n"
    "{{/if}}\n"
    "{{#if error_id}}\n"
    "- **Error ID**: `{error_id}`\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "- **Time**: {timestamp}\n"
    "{{/if}}\n"
    "{{#if session_id}}\n"
    "- **Session**: `{session_id}`\n"
    "{{/if}}\n"
    "{{#if user_id}}\n"
    "- **User**: `{user_id}`\n"
    "{{/if}}\n"
    "{{#if sanitized_detail}}\n"
    "### Error: `{sanitized_detail}`\n\n"
    "{{/if}}\n"
    "{{#if openrouter_message}}\n"
    "- **OpenRouter message**: `{openrouter_message}`\n"
    "{{/if}}\n"
    "{{#if upstream_message}}\n"
    "- **Provider message**: `{upstream_message}`\n"
    "{{/if}}\n"
    "{{#if model_identifier}}\n"
    "- **Model**: `{model_identifier}`\n"
    "{{/if}}\n"
    "{{#if provider}}\n"
    "- **Provider**: `{provider}`\n"
    "{{/if}}\n"
    "{{#if requested_model}}\n"
    "- **Requested model**: `{requested_model}`\n"
    "{{/if}}\n"
    "{{#if api_model_id}}\n"
    "- **API model id**: `{api_model_id}`\n"
    "{{/if}}\n"
    "{{#if normalized_model_id}}\n"
    "- **Normalized model id**: `{normalized_model_id}`\n"
    "{{/if}}\n"
    "{{#if openrouter_code}}\n"
    "- **OpenRouter code**: `{openrouter_code}`\n"
    "{{/if}}\n"
    "{{#if upstream_type}}\n"
    "- **Provider error**: `{upstream_type}`\n"
    "{{/if}}\n"
    "{{#if reason}}\n"
    "- **Reason**: `{reason}`\n"
    "{{/if}}\n"
    "{{#if request_id}}\n"
    "- **Request ID**: `{request_id}`\n"
    "{{/if}}\n"
    "{{#if native_finish_reason}}\n"
    "- **Finish reason**: `{native_finish_reason}`\n"
    "{{/if}}\n"
    "{{#if error_chunk_id}}\n"
    "- **Chunk ID**: `{error_chunk_id}`\n"
    "{{/if}}\n"
    "{{#if error_chunk_created}}\n"
    "- **Chunk time**: {error_chunk_created}\n"
    "{{/if}}\n"
    "{{#if streaming_provider}}\n"
    "- **Streaming provider**: `{streaming_provider}`\n"
    "{{/if}}\n"
    "{{#if streaming_model}}\n"
    "- **Streaming model**: `{streaming_model}`\n"
    "{{/if}}\n"
    "{{#if include_model_limits}}\n"
    "\n**Model limits:**\n"
    "{{#if context_limit_tokens}}Context window: {context_limit_tokens} tokens\n{{/if}}\n"
    "{{#if max_output_tokens}}Max output tokens: {max_output_tokens} tokens\n{{/if}}\n"
    "Adjust your prompt or requested output to stay within these limits.\n"
    "{{/if}}\n"
    "{{#if moderation_reasons}}\n"
    "\n**Moderation reasons:**\n"
    "{moderation_reasons}\n"
    "Please review the flagged content or contact your administrator if you believe this is a mistake.\n"
    "{{/if}}\n"
    "{{#if flagged_excerpt}}\n"
    "\n**Flagged text excerpt:**\n"
    "```\n{flagged_excerpt}\n```\n"
    "Provide this excerpt when following up with your administrator.\n"
    "{{/if}}\n"
    "{{#if raw_body}}\n"
    "\n**Raw provider response:**\n"
    "```\n{raw_body}\n```\n"
    "{{/if}}\n"
    "{{#if metadata_json}}\n"
    "\n**Metadata:**\n"
    "```\n{metadata_json}\n```\n"
    "{{/if}}\n"
    "{{#if provider_raw_json}}\n"
    "\n**Provider raw error:**\n"
    "```\n{provider_raw_json}\n```\n"
    "{{/if}}\n\n"
    "Please adjust the request and try again, or ask your admin to enable the middle-out option.\n"
    "{{#if request_id_reference}}\n"
    "{request_id_reference}\n"
    "{{/if}}\n"
)

DEFAULT_NETWORK_TIMEOUT_TEMPLATE = (
    "### ⏱️ Request Timeout\n\n"
    "The request to OpenRouter took too long to complete.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if timeout_seconds}}\n"
    "**Timeout:** {timeout_seconds}s\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**Possible causes:**\n"
    "- OpenRouter's servers are slow or overloaded\n"
    "- Network congestion\n"
    "- Large request taking longer than expected\n\n"
    "**What to do:**\n"
    "- Wait a few moments and try again\n"
    "- Try a smaller request if possible\n"
    "- Check [OpenRouter Status](https://status.openrouter.ai/)\n"
    "{{#if support_email}}\n"
    "- Contact support: {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_CONNECTION_ERROR_TEMPLATE = (
    "### 🔌 Connection Failed\n\n"
    "Unable to reach OpenRouter's servers.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if error_type}}\n"
    "**Error type:** `{error_type}`\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**Possible causes:**\n"
    "- Network connectivity issues\n"
    "- Firewall blocking HTTPS traffic\n"
    "- DNS resolution failure\n"
    "- OpenRouter service outage\n\n"
    "**What to do:**\n"
    "1. Check your internet connection\n"
    "2. Verify firewall allows HTTPS (port 443)\n"
    "3. Check [OpenRouter Status](https://status.openrouter.ai/)\n"
    "4. Contact your network administrator if the issue persists\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_SERVICE_ERROR_TEMPLATE = (
    "### 🔴 OpenRouter Service Error\n\n"
    "OpenRouter's servers are experiencing issues.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if status_code}}\n"
    "**Status:** {status_code} {reason}\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "This is **not** a problem with your request. The issue is on OpenRouter's side.\n\n"
    "**What to do:**\n"
    "- Wait a few minutes and try again\n"
    "- Check [OpenRouter Status](https://status.openrouter.ai/) for updates\n"
    "- If the problem persists for more than 15 minutes, contact OpenRouter support\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_INTERNAL_ERROR_TEMPLATE = (
    "### ⚠️ Unexpected Error\n\n"
    "Something unexpected went wrong while processing your request.\n\n"
    "**Error ID:** `{error_id}` — Share this with support\n"
    "{{#if error_type}}\n"
    "**Error type:** `{error_type}`\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "The error has been logged and will be investigated.\n\n"
    "**What to do:**\n"
    "- Try your request again\n"
    "- If the problem persists, contact support with the Error ID above\n"
    "{{#if support_email}}\n"
    "- Email: {support_email}\n"
    "{{/if}}\n"
    "{{#if support_url}}\n"
    "- Support: {support_url}\n"
    "{{/if}}\n"
)

DEFAULT_AUTHENTICATION_ERROR_TEMPLATE = (
    "### 🔐 Authentication Failed\n\n"
    "OpenRouter rejected your credentials.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if openrouter_code}}\n"
    "**Status:** {openrouter_code}\n"
    "{{/if}}\n"
    "{{#if openrouter_message}}\n"
    "**Details:** {openrouter_message}\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**What to do:**\n"
    "1. Verify the API key configured for this pipe\n"
    "2. Generate a new key at https://openrouter.ai/keys if needed\n"
    "3. If using OAuth, re-authenticate your session\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_INSUFFICIENT_CREDITS_TEMPLATE = (
    "### 💳 Insufficient Credits\n\n"
    "OpenRouter could not run this request because the account is out of credits.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if openrouter_code}}\n"
    "**Status:** {openrouter_code}\n"
    "{{/if}}\n"
    "{{#if openrouter_message}}\n"
    "**Details:** {openrouter_message}\n"
    "{{/if}}\n"
    "{{#if required_cost}}\n"
    "**Estimated cost:** ${required_cost}\n"
    "{{/if}}\n"
    "{{#if account_balance}}\n"
    "**Current balance:** ${account_balance}\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**What to do:**\n"
    "- Add credits at https://openrouter.ai/credits\n"
    "- Review usage at https://openrouter.ai/usage\n"
    "- Consider enabling auto-recharge to avoid interruptions\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_RATE_LIMIT_TEMPLATE = (
    "### ⏸️ Rate Limit Exceeded\n\n"
    "OpenRouter is protecting the service because too many requests were made quickly.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if openrouter_code}}\n"
    "**Status:** {openrouter_code}\n"
    "{{/if}}\n"
    "{{#if retry_after_seconds}}\n"
    "**Retry after:** {retry_after_seconds}s\n"
    "{{/if}}\n"
    "{{#if rate_limit_type}}\n"
    "**Limit type:** {rate_limit_type}\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**Tips:**\n"
    "- Back off and retry with exponential delays\n"
    "- Queue requests or lower parallelism\n"
    "- Contact OpenRouter if you need higher limits\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_SERVER_TIMEOUT_TEMPLATE = (
    "### 🕒 OpenRouter Timed Out\n\n"
    "OpenRouter started the request but couldn't finish within its timeout window.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if openrouter_code}}\n"
    "**Status:** {openrouter_code}\n"
    "{{/if}}\n"
    "{{#if openrouter_message}}\n"
    "**Details:** {openrouter_message}\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**Next steps:**\n"
    "- Retry shortly; the upstream provider may be busy\n"
    "- Reduce prompt size or requested work\n"
    "- Contact support if the issue persists\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)


def _render_error_template(template: str, values: dict[str, Any]) -> str:
    """Render a user-supplied template, honoring {{#if}} conditionals."""
    if not template:
        template = DEFAULT_OPENROUTER_ERROR_TEMPLATE

    rendered_lines: list[str] = []
    condition_stack: list[bool] = []

    def _conditions_active() -> bool:
        """Return True when the current {{#if}} stack has no falsy guards."""
        return all(condition_stack) if condition_stack else True

    for raw_line in template.splitlines():
        last_index = 0
        line_parts: list[str] = []

        for match in _TEMPLATE_IF_TOKEN_RE.finditer(raw_line):
            segment = raw_line[last_index:match.start()]
            if segment and _conditions_active():
                line_parts.append(segment)

            token = match.group(1) or ""
            var_name = match.group(2)
            if token.startswith("#if"):
                condition_stack.append(bool(values.get(var_name or "")))
            else:
                if condition_stack:
                    condition_stack.pop()

            last_index = match.end()

        tail_segment = raw_line[last_index:]
        if tail_segment and _conditions_active():
            line_parts.append(tail_segment)

        if not line_parts:
            if raw_line.strip():
                continue
            if not _conditions_active():
                continue
            rendered_lines.append("")
            continue

        line = "".join(line_parts)

        drop_line = False
        for name, value in values.items():
            placeholder = f"{{{name}}}"
            if placeholder in line:
                if not _template_value_present(value):
                    drop_line = True
                line = line.replace(placeholder, "" if value is None else str(value))
        if drop_line:
            continue
        rendered_lines.append(line)
    return "\n".join(rendered_lines).strip()


def _pretty_json(value: Any) -> str:
    """Return a human-readable JSON string or an empty string when not applicable."""
    if value is None:
        return ""
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        return text.strip()
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except Exception:
        return str(value)


def _template_value_present(value: Any) -> bool:
    """Return True when a placeholder value should be rendered."""
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    if isinstance(value, (int, float)):
        return True
    return bool(value)


def _build_error_template_values(
    error: "OpenRouterAPIError",
    *,
    heading: str,
    diagnostics: list[str],
    metrics: dict[str, Any],
    model_identifier: Optional[str],
    normalized_model_id: Optional[str],
    api_model_id: Optional[str],
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Prepare placeholder values for the customizable error template."""
    detail = (error.upstream_message or error.openrouter_message or str(error)).strip()
    sanitized_detail = detail.replace("`", "\\`")

    moderation_lines = [reason for reason in (error.moderation_reasons or []) if reason]
    flagged_excerpt = (error.flagged_input or "").strip()
    raw_body = (error.raw_body or "").strip()

    detail_lower = detail.lower()
    include_model_limits = "or use the \"middle-out\"" in detail_lower
    context_limit_value = metrics.get("context_limit") if include_model_limits else None
    max_output_tokens_value = metrics.get("max_output_tokens") if include_model_limits else None
    include_model_limits = include_model_limits and bool(
        (context_limit_value or max_output_tokens_value)
    )

    metadata_json = error.metadata_json or _pretty_json(error.metadata)
    provider_raw_json = error.provider_raw_json or _pretty_json(error.provider_raw)
    context = context or {}
    retry_after = (
        context.get("retry_after_seconds")
        or error.metadata.get("retry_after_seconds")
        or error.metadata.get("retry_after")
    )
    replacements: dict[str, Any] = {
        "heading": heading,
        "detail": detail,
        "sanitized_detail": sanitized_detail,
        "provider": (error.provider or "").strip(),
        "reason": str(error),
        "raw_body": raw_body,
        "model_identifier": model_identifier or "",
        "requested_model": error.requested_model or "",
        "openrouter_code": str(error.openrouter_code or ""),
        "upstream_type": error.upstream_type or "",
        "upstream_message": (error.upstream_message or "").strip(),
        "openrouter_message": (error.openrouter_message or "").strip(),
        "request_id": error.request_id or "",
        "request_id_reference": f"Request reference: `{error.request_id}`" if error.request_id else "",
        "moderation_reasons": "\n".join(f"- {reason}" for reason in moderation_lines),
        "flagged_excerpt": flagged_excerpt,
        "context_limit_tokens": f"{context_limit_value:,}" if context_limit_value else "",
        "max_output_tokens": f"{max_output_tokens_value:,}" if max_output_tokens_value else "",
        "include_model_limits": bool(include_model_limits),
        "api_model_id": api_model_id or "",
        "normalized_model_id": normalized_model_id or "",
        "metadata_json": metadata_json,
        "provider_raw_json": provider_raw_json,
        "native_finish_reason": error.native_finish_reason or "",
        "error_chunk_id": error.chunk_id or "",
        "error_chunk_created": error.chunk_created or "",
        "streaming_provider": error.chunk_provider or (error.provider or ""),
        "streaming_model": error.chunk_model or "",
        "is_streaming_error": bool(error.is_streaming_error),
        "error_id": context.get("error_id", ""),
        "timestamp": context.get("timestamp", ""),
        "session_id": context.get("session_id", ""),
        "user_id": context.get("user_id", ""),
        "support_email": context.get("support_email", ""),
        "support_url": context.get("support_url", ""),
        "retry_after_seconds": retry_after or "",
        "rate_limit_type": error.metadata.get("rate_limit_type") or "",
        "required_cost": error.metadata.get("required_cost") or "",
        "account_balance": error.metadata.get("account_balance") or "",
        "diagnostics": "\n".join(diagnostics),
    }
    return replacements


class _RetryableHTTPStatusError(Exception):
    """Wrapper that marks an HTTPStatusError as retryable."""

    def __init__(self, original: httpx.HTTPStatusError, retry_after: Optional[float] = None):
        """Capture the original HTTP error plus optional Retry-After hint."""
        self.original = original
        self.retry_after = retry_after
        status_code = getattr(original.response, "status_code", "unknown")
        super().__init__(f"Retryable HTTP error ({status_code})")


class _RetryWait:
    """Custom Tenacity wait strategy honoring Retry-After headers."""

    def __init__(self, base_wait):
        """Store the wrapped Tenacity wait callable used as a baseline."""
        self._base_wait = base_wait

    def __call__(self, retry_state):
        """Return the greater of base delay or Retry-After header guidance."""
        base_delay = self._base_wait(retry_state) if self._base_wait else 0
        exc = None
        if retry_state.outcome is not None:
            try:
                exc = retry_state.outcome.exception()
            except Exception:
                exc = None
        if isinstance(exc, _RetryableHTTPStatusError):
            retry_after = exc.retry_after
            if isinstance(retry_after, (int, float)) and retry_after > 0:
                return max(base_delay, retry_after)
        return base_delay


_TWait = TypeVar("_TWait")


async def _wait_for(
    value: Awaitable[_TWait] | _TWait,
    *,
    timeout: Optional[float] = None,
) -> _TWait:
    """Return ``value`` immediately when it's synchronous, otherwise await it.

    Redis' asyncio client returns synchronous fallbacks (bool/str/list) when a
    pipeline is configured for immediate execution, which caused ``await`` to be
    applied to non-awaitables.  This helper centralizes the guard so call sites
    stay tidy and Pyright no longer reports \"X is not awaitable\" diagnostics.
    """
    if inspect.isawaitable(value):
        coroutine = cast(Awaitable[_TWait], value)
        if timeout is None:
            return cast(_TWait, await coroutine)
        return cast(_TWait, await asyncio.wait_for(coroutine, timeout=timeout))
    return cast(_TWait, value)


_OPEN_WEBUI_CONFIG_MODULE: Any | None = None


def _get_open_webui_config_module() -> Any | None:
    """Return the cached open_webui.config module if available."""
    global _OPEN_WEBUI_CONFIG_MODULE
    if _OPEN_WEBUI_CONFIG_MODULE is not None:
        return _OPEN_WEBUI_CONFIG_MODULE
    try:
        import open_webui.config as ow_config  # type: ignore
    except Exception:
        return None
    _OPEN_WEBUI_CONFIG_MODULE = ow_config
    return ow_config


def _unwrap_config_value(value: Any) -> Any:
    """Return the raw value from a PersistentConfig-like object."""
    if value is None:
        return None
    return getattr(value, "value", value)


def _coerce_positive_int(value: Any) -> Optional[int]:
    """Convert strings/bools into positive integers (MB)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def _coerce_bool(value: Any) -> Optional[bool]:
    """Best-effort coercion of truthy string/int flags into booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, int):
        return bool(value)
    return None


def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
    """Convert Retry-After header value into seconds."""
    if not value:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    try:
        seconds = float(trimmed)
        return max(0.0, seconds)
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(trimmed)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        seconds = (dt - now).total_seconds()
        return max(0.0, seconds)
    except (TypeError, ValueError, OverflowError):
        return None


def _classify_retryable_http_error(
    exc: httpx.HTTPStatusError,
) -> tuple[bool, Optional[float]]:
    """Return (is_retryable, retry_after_seconds) for an HTTPStatusError."""
    response = exc.response
    if response is None:
        return False, None
    status_code = response.status_code
    if status_code >= 500 or status_code in {408, 425, 429}:
        return True, _retry_after_seconds(response.headers.get("retry-after"))
    return False, None


def _read_rag_file_constraints() -> tuple[bool, Optional[int]]:
    """Return (rag_enabled, rag_file_size_mb) gleaned from Open WebUI config."""
    module = _get_open_webui_config_module()
    if module is None:
        return False, None

    bypass_raw = _unwrap_config_value(getattr(module, "BYPASS_EMBEDDING_AND_RETRIEVAL", None))
    bypass_bool = _coerce_bool(bypass_raw)
    rag_enabled = True if bypass_bool is None else not bypass_bool

    limit_mb: Optional[int] = None
    for attr_name in ("RAG_FILE_MAX_SIZE", "FILE_MAX_SIZE"):
        attr_value = _unwrap_config_value(getattr(module, attr_name, None))
        limit_mb = _coerce_positive_int(attr_value)
        if limit_mb is not None:
            break

    if limit_mb is not None:
        limit_mb = min(limit_mb, _REMOTE_FILE_MAX_SIZE_MAX_MB)

    return rag_enabled, limit_mb


# ─────────────────────────────────────────────────────────────────────────────
# Status Message Constants
# ─────────────────────────────────────────────────────────────────────────────


class StatusMessages:
    """Centralized status messages for multimodal processing."""

    # Image processing
    IMAGE_BASE64_SAVED = "📥 Saved base64 image to storage"
    IMAGE_REMOTE_SAVED = "📥 Downloaded and saved image from remote URL"

    # File processing
    FILE_BASE64_SAVED = "📥 Saved base64 file to storage"
    FILE_REMOTE_SAVED = "📥 Downloaded and saved file from remote URL"

    # Video processing
    VIDEO_BASE64 = "🎥 Processing base64 video input"
    VIDEO_YOUTUBE = "🎥 Processing YouTube video input"
    VIDEO_REMOTE = "🎥 Processing video input"

    # Audio processing
    AUDIO_BASE64_SAVED = "🎵 Saved base64 audio to storage"
    AUDIO_REMOTE_SAVED = "🎵 Downloaded and saved audio from remote URL"


# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class _PipeJob:
    """Encapsulate a single OpenRouter request scheduled through the queue."""

    pipe: "Pipe"
    body: dict[str, Any]
    user: dict[str, Any]
    request: Request | None
    event_emitter: EventEmitter | None
    event_call: Callable[[dict[str, Any]], Awaitable[Any]] | None
    metadata: dict[str, Any]
    tools: list[dict[str, Any]] | dict[str, Any] | None
    task: Optional[dict[str, Any]]
    task_body: Optional[dict[str, Any]]
    valves: "Pipe.Valves"
    future: asyncio.Future
    stream_queue: asyncio.Queue[dict[str, Any] | str | None] | None = None
    request_id: str = field(default_factory=lambda: secrets.token_hex(8))

    @property
    def session_id(self) -> str:
        """Convenience accessor for the metadata session identifier."""
        return str(self.metadata.get("session_id") or "")

    @property
    def user_id(self) -> str:
        """Return the Open WebUI user id associated with the job."""
        return str(self.user.get("id") or self.metadata.get("user_id") or "")


# Tool Execution ----------------------------------------------------
@dataclass(slots=True)
class _QueuedToolCall:
    """Stores a pending tool call plus execution metadata for worker pools."""
    call: dict[str, Any]
    tool_cfg: dict[str, Any]
    args: dict[str, Any]
    future: asyncio.Future
    allow_batch: bool


@dataclass(slots=True)
class _ToolExecutionContext:
    """Holds shared state for executing tool calls within breaker limits."""
    queue: asyncio.Queue[_QueuedToolCall | None]
    per_request_semaphore: asyncio.Semaphore
    global_semaphore: asyncio.Semaphore | None
    timeout: float
    batch_timeout: float | None
    idle_timeout: float | None
    user_id: str
    event_emitter: EventEmitter | None
    batch_cap: int
    workers: list[asyncio.Task] = field(default_factory=list)
    timeout_error: Optional[str] = None


ToolCallable = Callable[..., Awaitable[Any]] | Callable[..., Any]


@dataclass(slots=True)
class _SessionLogArchiveJob:
    """Represents a single session log archive request destined for the writer thread."""

    base_dir: str
    zip_password: bytes
    zip_compression: str
    zip_compresslevel: Optional[int]
    user_id: str
    session_id: str
    chat_id: str
    message_id: str
    request_id: str
    created_at: float
    log_lines: list[str]


class EncryptedStr(str):
    """String wrapper that automatically encrypts/decrypts valve values."""

    _ENCRYPTION_PREFIX = "encrypted:"

    @classmethod
    def _get_encryption_key(cls) -> Optional[bytes]:
        """Return the Fernet key derived from ``WEBUI_SECRET_KEY``.

        Returns:
            Optional[bytes]: URL-safe base64 Fernet key or ``None`` when unset.
        """
        secret = os.getenv("WEBUI_SECRET_KEY")
        if not secret:
            return None
        hashed_key = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hashed_key)

    @classmethod
    def encrypt(cls, value: str) -> str:
        """Encrypt ``value`` when an application secret is configured.

        Args:
            value: Plain-text string supplied by the user.

        Returns:
            str: Ciphertext prefixed with ``encrypted:`` or the original value.
        """
        if not value or value.startswith(cls._ENCRYPTION_PREFIX):
            return value
        key = cls._get_encryption_key()
        if not key:
            return value
        fernet = Fernet(key)
        encrypted = fernet.encrypt(value.encode())
        return f"{cls._ENCRYPTION_PREFIX}{encrypted.decode()}"

    @classmethod
    def decrypt(cls, value: str) -> str:
        """Decrypt values produced by :meth:`encrypt`.

        Args:
            value: Ciphertext string, typically prefixed with ``encrypted:``.

        Returns:
            str: Decrypted plain text or the original value when keyless.
        """
        if not value or not value.startswith(cls._ENCRYPTION_PREFIX):
            return value
        key = cls._get_encryption_key()
        if not key:
            return value[len(cls._ENCRYPTION_PREFIX) :]
        try:
            encrypted_part = value[len(cls._ENCRYPTION_PREFIX) :]
            fernet = Fernet(key)
            decrypted = fernet.decrypt(encrypted_part.encode())
            return decrypted.decode()
        except InvalidToken:
            # Invalid encryption key or corrupted data - return original value
            LOGGER.warning("Failed to decrypt value: invalid token or key mismatch")
            return value
        except (ValueError, UnicodeDecodeError) as e:
            # Decoding or encoding error - return original value
            LOGGER.warning(f"Failed to decrypt value: {type(e).__name__}: {e}")
            return value

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Expose a union schema so plain strings auto-wrap as EncryptedStr."""
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.chain_schema(
                    [
                        core_schema.str_schema(),
                        core_schema.no_info_plain_validator_function(
                            lambda value: cls(cls.encrypt(value) if value else value)
                        ),
                    ]
                ),
            ],
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: str(instance)
            ),
        )

_ALLOWED_LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _default_api_key() -> EncryptedStr:
    """Return the API key env default as EncryptedStr."""
    return EncryptedStr((os.getenv("OPENROUTER_API_KEY") or "").strip())


def _default_artifact_encryption_key() -> EncryptedStr:
    """Provide an EncryptedStr placeholder for artifact encryption."""
    return EncryptedStr("")


def _resolve_log_level_default() -> Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    """Normalize env-provided log level to the allowed literal set."""
    value = (os.getenv("GLOBAL_LOG_LEVEL") or "INFO").strip().upper()
    if value not in _ALLOWED_LOG_LEVELS:
        value = "INFO"
    return cast(Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], value)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Constants & Global Configuration
# ─────────────────────────────────────────────────────────────────────────────
class ModelFamily:
    """
    One place for base capabilities + alias mapping (with effort defaults).
    """

    _DATE_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")
    _PIPE_ID: ContextVar[Optional[str]] = ContextVar(
        "owui_pipe_id_ctx",
        default=None,
    )
    _DYNAMIC_SPECS: Dict[str, Dict[str, Any]] = {}

    # ── tiny, intuitive helpers ──────────────────────────────────────────────
    @classmethod
    def _norm(cls, model_id: str) -> str:
        """Normalize model ids by stripping pipe prefixes and date suffixes."""
        m = (model_id or "").strip()
        if "/" in m:
            m = m.replace("/", ".")
        pipe_id = cls._PIPE_ID.get()
        if pipe_id:
            pref = f"{pipe_id}."
            if m.startswith(pref):
                m = m[len(pref):]
        return cls._DATE_RE.sub("", m.lower())

    @classmethod
    def base_model(cls, model_id: str) -> str:
        """Canonical base model id (prefix/date stripped)."""
        return cls._norm(model_id)

    @classmethod
    def features(cls, model_id: str) -> frozenset[str]:
        """Capabilities for the base model behind this id."""
        spec = cls._lookup_spec(model_id)
        return frozenset(spec.get("features", set()))

    @classmethod
    def max_completion_tokens(cls, model_id: str) -> Optional[int]:
        """Return max completion tokens reported by the provider, if any."""
        spec = cls._lookup_spec(model_id)
        return spec.get("max_completion_tokens")

    @classmethod
    def supports(cls, feature: str, model_id: str) -> bool:
        """Check if a model supports a given feature."""
        return feature in cls.features(model_id)

    @classmethod
    def capabilities(cls, model_id: str) -> dict[str, bool]:
        """Return derived capability checkboxes for the given model."""
        spec = cls._lookup_spec(model_id)
        caps = spec.get("capabilities") or {}
        # Return a shallow copy so downstream code can mutate safely.
        return dict(caps)

    @classmethod
    def supported_parameters(cls, model_id: str) -> frozenset[str]:
        """Return the raw `supported_parameters` set from the OpenRouter catalog."""
        spec = cls._lookup_spec(model_id)
        params = spec.get("supported_parameters")
        if isinstance(params, frozenset):
            return params
        if isinstance(params, (set, list, tuple)):
            return frozenset(params)
        return frozenset()

    @classmethod
    def set_dynamic_specs(cls, specs: Dict[str, Dict[str, Any]] | None) -> None:
        """Update cached OpenRouter specs shared with :class:`ModelFamily`."""
        cls._DYNAMIC_SPECS = specs or {}

    @classmethod
    def _lookup_spec(cls, model_id: str) -> Dict[str, Any]:
        """Return the stored spec for ``model_id`` or an empty dict."""
        norm = cls.base_model(model_id)
        return cls._DYNAMIC_SPECS.get(norm) or {}


def sanitize_model_id(model_id: str) -> str:
    """Convert `author/model` ids into dot-friendly ids for Open WebUI."""
    if not model_id:
        return model_id
    if "/" not in model_id:
        return model_id
    head, tail = model_id.split("/", 1)
    return f"{head}.{tail.replace('/', '.')}"


_PAYLOAD_FLAG_PLAIN = 0
_PAYLOAD_FLAG_LZ4 = 1
_ENCRYPTED_PAYLOAD_VERSION = 1
_PAYLOAD_HEADER_SIZE = 1

# DB Persistence constants
_REDIS_FLUSH_CHANNEL = "db-flush"
# Valves (defaults pulled at runtime)


class OpenRouterModelRegistry:
    """Fetches and caches the OpenRouter model catalog."""

    _models: list[dict[str, Any]] = []
    _specs: Dict[str, Dict[str, Any]] = {}
    _id_map: Dict[str, str] = {}  # normalized sanitized id -> original id
    _last_fetch: float = 0.0
    _lock: asyncio.Lock = asyncio.Lock()
    _next_refresh_after: float = 0.0
    _consecutive_failures: int = 0
    _last_error_time: float = 0.0

    @classmethod
    async def ensure_loaded(
        cls,
        session: aiohttp.ClientSession,
        *,
        base_url: str,
        api_key: str,
        cache_seconds: int,
        logger: logging.Logger,
    ) -> None:
        """Refresh the model catalog if the cache is empty or stale."""
        if not api_key:
            raise ValueError("OpenRouter API key is required.")

        now = time.time()
        next_refresh = cls._next_refresh_after or (cls._last_fetch + cache_seconds)
        if cls._specs and now < next_refresh:
            return

        async with cls._lock:
            now = time.time()
            next_refresh = cls._next_refresh_after or (cls._last_fetch + cache_seconds)
            if cls._specs and now < next_refresh:
                return
            try:
                await cls._refresh(session, base_url=base_url, api_key=api_key, logger=logger)
            except Exception as exc:
                # Catch all refresh errors (network, JSON, API errors) to use cache if available
                cls._record_refresh_failure(exc, cache_seconds)
                if not cls._models:
                    raise
                logger.warning(
                    "OpenRouter catalog refresh failed (%s). Serving %d cached model(s).",
                    exc,
                    len(cls._models),
                )
                return
            cls._record_refresh_success(cache_seconds)

    @classmethod
    async def _refresh(
        cls,
        session: aiohttp.ClientSession,
        *,
        base_url: str,
        api_key: str,
        logger: logging.Logger,
    ) -> None:
        """Fetch and cache the OpenRouter catalog."""
        url = base_url.rstrip("/") + "/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Title": _OPENROUTER_TITLE,
            "HTTP-Referer": _OPENROUTER_REFERER,
        }
        _debug_print_request(headers, {"method": "GET", "url": url})
        try:
            async with session.get(url, headers=headers) as resp:
                if resp.status >= 400:
                    await _debug_print_error_response(resp)
                resp.raise_for_status()
                payload = await resp.json()
        except Exception as exc:
            logger.error("Failed to load OpenRouter model catalog: %s", exc)
            raise

        data = payload.get("data") or []
        raw_specs: Dict[str, Dict[str, Any]] = {}
        models: list[dict[str, Any]] = []
        id_map: Dict[str, str] = {}

        for item in data:
            original_id = item.get("id")
            if not original_id:
                continue

            sanitized = sanitize_model_id(original_id)
            norm_id = ModelFamily.base_model(sanitized)

            # Store the FULL model object - it's only 3.5MB total for all models
            # This gives us access to: description, context_length, created, canonical_slug,
            # hugging_face_id, per_request_limits, default_parameters, and everything else
            raw_specs[norm_id] = dict(item)

            id_map[norm_id] = original_id
            models.append(
                {
                    "id": sanitized,
                    "norm_id": norm_id,
                    "original_id": original_id,
                    "name": item.get("name") or original_id,
                }
            )

        # Finalize specs shared with ModelFamily (features + max completion tokens).
        # Now we keep the FULL model object plus derived features for fast lookups
        specs: Dict[str, Dict[str, Any]] = {}
        for norm_id, full_model in raw_specs.items():
            supported_parameters = set(full_model.get("supported_parameters") or [])
            architecture = full_model.get("architecture") or {}
            pricing = full_model.get("pricing") or {}

            # Derive feature flags from model metadata
            features = cls._derive_features(
                supported_parameters,
                architecture,
                pricing,
            )
            capabilities = cls._derive_capabilities(
                architecture,
                pricing,
            )

            # Extract max_completion_tokens from top_provider
            max_completion_tokens: Optional[int] = None
            top_provider = full_model.get("top_provider")
            if isinstance(top_provider, dict):
                max_completion_tokens = top_provider.get("max_completion_tokens")

            # Store full model + derived data for downstream use
            specs[norm_id] = {
                # Derived features for fast capability checks
                "features": features,
                "capabilities": capabilities,
                "max_completion_tokens": max_completion_tokens,
                "supported_parameters": frozenset(supported_parameters),

                # Keep full model object for any future needs
                "full_model": full_model,

                # Quick access to commonly used fields
                "context_length": full_model.get("context_length"),
                "description": full_model.get("description"),
                "pricing": pricing,
                "architecture": architecture,
            }

        models.sort(key=lambda m: m["name"].lower())
        if not models:
            raise RuntimeError("OpenRouter returned an empty model catalog.")
        cls._models = models
        cls._specs = specs
        cls._id_map = id_map

        # Share dynamic specs with ModelFamily for downstream feature checks.
        ModelFamily.set_dynamic_specs(specs)

    @classmethod
    def _record_refresh_success(cls, cache_seconds: int) -> None:
        """Reset refresh backoff bookkeeping after a successful catalog fetch."""
        now = time.time()
        cls._last_fetch = now
        cls._next_refresh_after = now + max(5, cache_seconds)
        cls._consecutive_failures = 0
        cls._last_error_time = 0.0

    @classmethod
    def _record_refresh_failure(cls, exc: Exception, cache_seconds: int) -> None:
        """Increase backoff delay and track the most recent catalog error."""
        cls._consecutive_failures += 1
        cls._last_error_time = time.time()
        exponent = min(cls._consecutive_failures - 1, 5)
        base_backoff = 5.0
        raw_backoff = base_backoff * (2 ** exponent)
        capped_backoff = min(cache_seconds, raw_backoff)
        backoff_until = cls._last_error_time + max(base_backoff, capped_backoff)
        cls._next_refresh_after = max(cls._next_refresh_after, backoff_until)

    @staticmethod
    def _derive_features(
        supported_parameters: set[str],
        architecture: Dict[str, Any],
        pricing: Dict[str, Any],
    ) -> set[str]:
        """Translate OpenRouter metadata into capability flags.

        Features include:
        - function_calling: Model supports tools/function calling
        - reasoning: Model supports extended reasoning
        - reasoning_summary: Model supports reasoning summaries
        - web_search_tool: Model has web search capability
        - image_gen_tool: Model can generate images (output)
        - vision: Model accepts image inputs
        - audio_input: Model accepts audio inputs
        - video_input: Model accepts video inputs
        - file_input: Model accepts file/document inputs
        """
        features: set[str] = set()

        # Check supported parameters for function calling and reasoning
        if {"tools", "tool_choice"} & supported_parameters:
            features.add("function_calling")
        if "reasoning" in supported_parameters:
            features.add("reasoning")
        if "include_reasoning" in supported_parameters:
            features.add("reasoning_summary")

        # Check pricing for built-in tools
        if pricing.get("web_search") is not None:
            features.add("web_search_tool")

        # Check output modalities
        output_modalities = architecture.get("output_modalities") or []
        if "image" in output_modalities:
            features.add("image_gen_tool")

        # Check input modalities - CRITICAL for multimodal support validation
        input_modalities = architecture.get("input_modalities") or []
        if "image" in input_modalities:
            features.add("vision")
        if "audio" in input_modalities:
            features.add("audio_input")
        if "video" in input_modalities:
            features.add("video_input")
        if "file" in input_modalities:
            features.add("file_input")

        return features

    @staticmethod
    def _supports_web_search(pricing: Dict[str, Any]) -> bool:
        """Return True when the provider exposes paid web-search support."""
        value = pricing.get("web_search")
        if value is None:
            return False
        if isinstance(value, str):
            value = value.strip() or "0"
        try:
            return float(value) > 0.0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _derive_capabilities(
        architecture: Dict[str, Any],
        pricing: Dict[str, Any],
    ) -> dict[str, bool]:
        """Translate metadata into Open WebUI capability checkboxes."""

        def _normalize(values: list[Any]) -> set[str]:
            """Return a normalized lowercase set from the provider metadata."""
            normalized: set[str] = set()
            for item in values:
                if isinstance(item, str):
                    normalized.add(item.strip().lower())
            return normalized

        input_modalities = _normalize(architecture.get("input_modalities") or [])
        output_modalities = _normalize(architecture.get("output_modalities") or [])

        vision_capable = "image" in input_modalities or "video" in input_modalities
        file_upload_capable = "file" in input_modalities
        image_generation_capable = "image" in output_modalities
        web_search_capable = OpenRouterModelRegistry._supports_web_search(pricing)

        return {
            "vision": vision_capable,
            "file_upload": file_upload_capable,
            "web_search": web_search_capable,
            "image_generation": image_generation_capable,
            "code_interpreter": True,
            "citations": True,
            "status_updates": True,
            "usage": True,
        }

    @classmethod
    def list_models(cls) -> list[dict[str, Any]]:
        """Return a shallow copy of the cached catalog."""
        enriched: list[dict[str, Any]] = []
        for model in cls._models:
            item = dict(model)
            spec = cls._specs.get(model["norm_id"])
            if spec and spec.get("capabilities"):
                item["capabilities"] = dict(spec["capabilities"])
            enriched.append(item)
        return enriched


    @classmethod
    def api_model_id(cls, model_id: str) -> Optional[str]:
        """Map sanitized Open WebUI ids back to provider ids."""
        norm = ModelFamily.base_model(model_id)
        return cls._id_map.get(norm)

# Temporary debug helpers shared by registry + pipe (safe to remove later)
def _debug_print_request(headers: Dict[str, str], payload: Optional[Dict[str, Any]]) -> None:
    """Log sanitized request metadata when DEBUG logging is enabled."""
    redacted_headers = dict(headers or {})
    if "Authorization" in redacted_headers:
        token = redacted_headers["Authorization"]
        redacted_headers["Authorization"] = f"{token[:10]}..." if len(token) > 10 else "***"
    LOGGER.debug("OpenRouter request headers: %s", json.dumps(redacted_headers, indent=2))
    if payload is not None:
        LOGGER.debug("OpenRouter request payload: %s", json.dumps(payload, indent=2))


async def _debug_print_error_response(resp: aiohttp.ClientResponse) -> str:
    """Log the response payload and return the response body for debugging."""
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


def _safe_json_loads(payload: Optional[str]) -> Any:
    """Return parsed JSON or None without raising."""
    if not payload:
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


def _normalize_optional_str(value: Any) -> Optional[str]:
    """Convert arbitrary input into a trimmed string or None."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def _normalize_string_list(value: Any) -> list[str]:
    """Return a list of trimmed strings."""
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for entry in value:
        text = _normalize_optional_str(entry)
        if text:
            items.append(text)
    return items


def _extract_plain_text_content(content: Any) -> str:
    """Collapse Open WebUI-style content blocks into a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                text_val = block.get("text") or block.get("content")
                if isinstance(text_val, str):
                    parts.append(text_val)
        return "\n".join(parts)
    if isinstance(content, dict):
        text_val = content.get("text") or content.get("content")
        if isinstance(text_val, str):
            return text_val
    return str(content or "")


def _extract_openrouter_error_details(body_text: Optional[str]) -> dict[str, Any]:
    """Normalize OpenRouter error payloads into structured metadata."""
    parsed = _safe_json_loads(body_text) if body_text else None
    error_section = parsed.get("error", {}) if isinstance(parsed, dict) else {}
    metadata = error_section.get("metadata", {}) if isinstance(error_section, dict) else {}
    metadata_dict = metadata if isinstance(metadata, dict) else {}

    raw_meta = metadata_dict.get("raw")
    if isinstance(raw_meta, str):
        raw_details = _safe_json_loads(raw_meta)
    elif isinstance(raw_meta, dict):
        raw_details = raw_meta
    else:
        raw_details = None

    upstream_error = raw_details.get("error", {}) if isinstance(raw_details, dict) else {}
    upstream_message = (
        upstream_error.get("message")
        if isinstance(upstream_error, dict)
        else None
    ) or (raw_details.get("message") if isinstance(raw_details, dict) else None)
    upstream_type = upstream_error.get("type") if isinstance(upstream_error, dict) else None

    request_id = (
        metadata.get("request_id")
        or (raw_details.get("request_id") if isinstance(raw_details, dict) else None)
        or (parsed.get("request_id") if isinstance(parsed, dict) else None)
    )

    return {
        "provider": metadata_dict.get("provider_name") or metadata_dict.get("provider"),
        "openrouter_message": error_section.get("message"),
        "openrouter_code": error_section.get("code"),
        "upstream_message": upstream_message,
        "upstream_type": upstream_type,
        "request_id": request_id,
        "raw_body": body_text or "",
        "metadata": metadata_dict,
        "metadata_json": _pretty_json(metadata_dict),
        "moderation_reasons": _normalize_string_list(metadata_dict.get("reasons")),
        "flagged_input": _normalize_optional_str(metadata_dict.get("flagged_input")),
        "model_slug": _normalize_optional_str(metadata_dict.get("model_slug")),
        "provider_raw": raw_details,
        "provider_raw_json": _pretty_json(raw_details),
    }


def _build_openrouter_api_error(
    status: int,
    reason: str,
    body_text: Optional[str],
    *,
    requested_model: Optional[str] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> "OpenRouterAPIError":
    """Create a structured error wrapper for OpenRouter 4xx responses."""
    details = _extract_openrouter_error_details(body_text)
    metadata_block = details.get("metadata") or {}
    if extra_metadata:
        metadata_block = {**metadata_block, **extra_metadata}
    return OpenRouterAPIError(
        status=status,
        reason=reason,
        provider=details.get("provider"),
        openrouter_message=details.get("openrouter_message"),
        openrouter_code=details.get("openrouter_code"),
        upstream_message=details.get("upstream_message"),
        upstream_type=details.get("upstream_type"),
        request_id=details.get("request_id"),
        raw_body=details.get("raw_body"),
        metadata=metadata_block,
        moderation_reasons=details.get("moderation_reasons") or [],
        flagged_input=details.get("flagged_input"),
        model_slug=details.get("model_slug"),
        requested_model=requested_model,
        metadata_json=details.get("metadata_json"),
        provider_raw=details.get("provider_raw"),
        provider_raw_json=details.get("provider_raw_json"),
    )


def _is_reasoning_effort_error(error_details: dict[str, Any]) -> bool:
    """Return True when the provider rejected reasoning.effort as unsupported."""
    provider_raw = error_details.get("provider_raw")
    if not isinstance(provider_raw, dict):
        return False
    upstream_error = provider_raw.get("error", {})
    if not isinstance(upstream_error, dict):
        return False
    return (
        upstream_error.get("param") == "reasoning.effort"
        and upstream_error.get("code") == "unsupported_value"
        and upstream_error.get("type") == "invalid_request_error"
    )


def _parse_supported_effort_values(error_message: str) -> list[str]:
    """Extract the list of supported reasoning efforts from an OpenRouter error."""
    import re

    match = re.search(r"Supported values are:\s*(.+?)\.", error_message, re.IGNORECASE)
    if not match:
        return []
    values_str = match.group(1)
    return re.findall(r"'([^']+)'", values_str)


def _select_best_effort_fallback(requested: str, supported: list[str]) -> Optional[str]:
    """Choose the closest supported effort to retry with."""
    ordering = ["none", "minimal", "low", "medium", "high", "xhigh"]
    if not supported:
        return None
    requested_lower = (requested or "").strip().lower()
    supported_lower = [value.strip().lower() for value in supported if value]
    if requested_lower in supported_lower:
        return requested_lower
    try:
        requested_idx = ordering.index(requested_lower)
    except ValueError:
        return supported_lower[0] if supported_lower else None
    indexed: list[tuple[int, str]] = []
    for value in supported_lower:
        try:
            indexed.append((ordering.index(value), value))
        except ValueError:
            continue
    if not indexed:
        return supported_lower[0] if supported_lower else None
    indexed.sort()
    min_idx, min_value = indexed[0]
    max_idx, max_value = indexed[-1]
    if requested_idx <= min_idx:
        return min_value
    if requested_idx >= max_idx:
        return max_value
    for idx, value in indexed:
        if idx > requested_idx:
            return value
    closest = None
    distance = float("inf")
    for idx, value in indexed:
        delta = abs(idx - requested_idx)
        if delta < distance:
            closest = value
            distance = delta
    return closest


def _classify_gemini_thinking_family(normalized_model_id: str) -> Optional[str]:
    """Return the Gemini thinking family for the provided normalized model id."""
    lowered = (normalized_model_id or "").lower()
    if lowered.startswith("google.gemini-3-") or lowered == "google.gemini-3":
        return "gemini-3"
    if lowered.startswith("google.gemini-2.5-") or lowered == "google.gemini-2.5":
        return "gemini-2.5"
    return None


def _map_effort_to_gemini_level(effort: str, valve_level: str) -> Optional[str]:
    """Map our reasoning effort preference onto Gemini 3 thinking levels."""
    effective = (valve_level or "auto").strip().lower()
    if effective in {"low", "high"}:
        return effective.upper()
    normalized = (effort or "").strip().lower()
    if normalized in {"none", ""}:
        return None
    if normalized in {"minimal", "low"}:
        return "LOW"
    return "HIGH"


def _map_effort_to_gemini_budget(effort: str, base_budget: int) -> Optional[int]:
    """Scale the configured Gemini 2.5 thinking budget from the requested effort."""
    if base_budget <= 0:
        return 0 if base_budget == 0 else None
    normalized = (effort or "").strip().lower()
    if normalized == "none":
        return None
    scalars = {
        "minimal": 0.25,
        "low": 0.5,
        "medium": 1.0,
        "high": 2.0,
        "xhigh": 4.0,
    }
    scalar = scalars.get(normalized, 1.0)
    return int(max(1, round(base_budget * scalar)))


class OpenRouterAPIError(RuntimeError):
    """User-facing error raised when OpenRouter rejects a request with status 400."""

    def __init__(
        self,
        *,
        status: int,
        reason: str,
        provider: Optional[str] = None,
        openrouter_message: Optional[str] = None,
        openrouter_code: Optional[Any] = None,
        upstream_message: Optional[str] = None,
        upstream_type: Optional[str] = None,
        request_id: Optional[str] = None,
        raw_body: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        moderation_reasons: Optional[list[str]] = None,
        flagged_input: Optional[str] = None,
        model_slug: Optional[str] = None,
        requested_model: Optional[str] = None,
        metadata_json: Optional[str] = None,
        provider_raw: Optional[Any] = None,
        provider_raw_json: Optional[str] = None,
        native_finish_reason: Optional[str] = None,
        chunk_id: Optional[str] = None,
        chunk_created: Optional[Any] = None,
        chunk_provider: Optional[str] = None,
        chunk_model: Optional[str] = None,
        is_streaming_error: bool = False,
    ) -> None:
        """Normalize raw OpenRouter metadata into convenient attributes."""
        self.status = status
        self.reason = reason
        self.provider = provider
        self.openrouter_message = (openrouter_message or "").strip() or None
        self.openrouter_code = openrouter_code
        self.upstream_message = (upstream_message or "").strip() or None
        self.upstream_type = (upstream_type or "").strip() or None
        self.request_id = (request_id or "").strip() or None
        self.raw_body = raw_body or ""
        self.metadata = metadata or {}
        self.moderation_reasons = moderation_reasons or []
        self.flagged_input = (flagged_input or "").strip() or None
        self.model_slug = (model_slug or "").strip() or None
        self.requested_model = (requested_model or "").strip() or None
        self.metadata_json = metadata_json or _pretty_json(self.metadata)
        self.provider_raw = provider_raw
        self.provider_raw_json = provider_raw_json or _pretty_json(provider_raw)
        self.native_finish_reason = native_finish_reason
        self.chunk_id = chunk_id
        self.chunk_created = chunk_created
        self.chunk_provider = chunk_provider
        self.chunk_model = chunk_model
        self.is_streaming_error = is_streaming_error
        summary = (
            self.upstream_message
            or self.openrouter_message
            or f"OpenRouter request failed ({self.status} {self.reason})"
        )
        super().__init__(summary)

    def to_markdown(
        self,
        *,
        model_label: Optional[str] = None,
        diagnostics: Optional[list[str]] = None,
        fallback_model: Optional[str] = None,
        template: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
        normalized_model_id: Optional[str] = None,
        api_model_id: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Return a user-friendly markdown block describing the failure."""
        provider_label = (self.provider or "").strip()
        effective_model = model_label or fallback_model or self.model_slug or self.requested_model
        if provider_label and effective_model:
            heading = f"{provider_label}: {effective_model}"
        elif effective_model:
            heading = effective_model
        elif provider_label:
            heading = provider_label
        else:
            heading = "OpenRouter"

        replacements = _build_error_template_values(
            self,
            heading=heading,
            diagnostics=diagnostics or [],
            metrics=metrics or {},
            model_identifier=self.model_slug or self.requested_model or fallback_model,
            normalized_model_id=normalized_model_id,
            api_model_id=api_model_id,
            context=context,
        )
        return _render_error_template(template or DEFAULT_OPENROUTER_ERROR_TEMPLATE, replacements)


def _resolve_error_model_context(
    error: OpenRouterAPIError,
    *,
    normalized_model_id: Optional[str] = None,
    api_model_id: Optional[str] = None,
) -> tuple[Optional[str], list[str], dict[str, Any]]:
    """Return (display_label, diagnostics_lines, metrics) for the affected model."""
    diagnostics: list[str] = []
    display_label: Optional[str] = None

    norm_id = ModelFamily.base_model(normalized_model_id or "") if normalized_model_id else None
    spec = ModelFamily._lookup_spec(norm_id or "")
    full_model = spec.get("full_model") or {}

    context_limit = spec.get("context_length") or full_model.get("context_length")
    max_output_tokens = spec.get("max_completion_tokens") or full_model.get("max_completion_tokens")

    if context_limit:
        diagnostics.append(f"- **Context window**: {context_limit:,} tokens")
    if max_output_tokens:
        diagnostics.append(f"- **Max output tokens**: {max_output_tokens:,} tokens")

    display_label = (
        full_model.get("name")
        or api_model_id
        or error.model_slug
        or error.requested_model
        or normalized_model_id
    )

    return display_label, diagnostics, {
        "context_limit": context_limit,
        "max_output_tokens": max_output_tokens,
    }


def _format_openrouter_error_markdown(
    error: OpenRouterAPIError,
    *,
    normalized_model_id: Optional[str],
    api_model_id: Optional[str],
    template: str,
    context: Optional[dict[str, Any]] = None,
) -> str:
    """Render a provider error into markdown after enriching with model context."""
    model_display, diagnostics, metrics = _resolve_error_model_context(
        error,
        normalized_model_id=normalized_model_id,
        api_model_id=api_model_id,
    )
    return error.to_markdown(
        model_label=model_display,
        diagnostics=diagnostics or None,
        fallback_model=api_model_id or normalized_model_id,
        template=template,
        metrics=metrics,
        normalized_model_id=normalized_model_id,
        api_model_id=api_model_id,
        context=context,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Models
# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models for validating request and response payloads
class CompletionsBody(BaseModel):
    """
    Represents the body of a completions request to OpenAI completions API.
    """
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False
    response_format: Optional[Dict[str, Any]] = None
    parallel_tool_calls: Optional[bool] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    model_config = ConfigDict(extra="allow")  # pass through additional OpenAI parameters

class ResponsesBody(BaseModel):
    """
    Represents the body of a responses request to OpenAI Responses API.
    """
    
    # Core parameters
    model: str
    instructions: Optional[str] = None  # system/developer instructions
    input: Union[str, List[Dict[str, Any]]]  # plain text, or rich array

    stream: bool = False                          # SSE chunking
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    reasoning: Optional[Dict[str, Any]] = None    # {"effort":"high", ...}
    include_reasoning: Optional[bool] = None
    thinking_config: Optional[Dict[str, Any]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    plugins: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    parallel_tool_calls: Optional[bool] = None
    transforms: Optional[List[str]] = None
    model_config = ConfigDict(extra="allow")  # allow additional OpenAI parameters automatically

    @model_validator(mode='after')
    def _normalize_model_id(self) -> "ResponsesBody":
        """Ensure the model name references the canonical base id (prefix/date stripped)."""
        normalized = ModelFamily.base_model(self.model or "")
        if normalized and normalized != self.model:
            self.model = normalized
        return self

    @staticmethod
    def transform_owui_tools(__tools__: Dict[str, dict] | None, *, strict: bool = False) -> List[dict]:
        """
        Convert Open WebUI __tools__ registry (dict of entries with {"spec": {...}}) into
        OpenAI Responses-API tool specs: {"type": "function", "name", ...}.

        """
        if not __tools__:
            return []

        tools: List[dict] = []
        for item in __tools__.values():
            spec = item.get("spec") or {}
            name = spec.get("name")
            if not name:
                continue  # skip malformed entries

            params = spec.get("parameters") or {"type": "object", "properties": {}}

            tool = {
                "type": "function",
                "name": name,
                "description": spec.get("description") or name,
                "parameters": _strictify_schema(params) if strict else params,
            }
            if strict:
                tool["strict"] = True

            tools.append(tool)

        return tools

    @staticmethod
    def _convert_function_call_to_tool_choice(function_call: Any) -> Optional[Any]:
        """
        Translate legacy OpenAI `function_call` payloads into modern `tool_choice` values.

        Returns either "auto"/"none" (strings) or {"type": "function", "name": "..."}.
        """
        if function_call is None:
            return None
        if isinstance(function_call, str):
            lowered = function_call.strip().lower()
            if lowered in {"auto", "none"}:
                return lowered
            return None

        if isinstance(function_call, dict):
            name = function_call.get("name")
            if not name and isinstance(function_call.get("function"), dict):
                name = function_call["function"].get("name")
            if isinstance(name, str) and name.strip():
                return {"type": "function", "name": name.strip()}
        return None

    @classmethod
    async def from_completions(
        cls,
        completions_body: "CompletionsBody",
        chat_id: Optional[str] = None,
        openwebui_model_id: Optional[str] = None,
        *,
        request: Optional[Request] = None,
        user_obj: Optional[Any] = None,
        event_emitter: Optional[Callable] = None,
        artifact_loader: Optional[
            Callable[[Optional[str], Optional[str], List[str]], Awaitable[Dict[str, Dict[str, Any]]]]
        ] = None,
        pruning_turns: int = 0,
        transformer_context: Optional[Any] = None,
        transformer_valves: Optional["Pipe.Valves"] = None,
        **extra_params,
    ) -> "ResponsesBody":
        """
        Convert CompletionsBody → ResponsesBody.

        - Drops unsupported fields (clearly logged).
        - Converts max_tokens → max_output_tokens.
        - Converts reasoning_effort → reasoning.effort (without overwriting).
        - Builds messages in Responses API format.
        - Allows explicit overrides via kwargs.
        - Replays persisted artifacts by awaiting `artifact_loader` when provided.
        """
        completions_dict = completions_body.model_dump(exclude_none=True)

        # Step 1: Remove unsupported fields
        unsupported_fields = {
            # Fields that are not supported by OpenAI Responses API
            "frequency_penalty", "presence_penalty", "seed", "logit_bias",
            "logprobs", "top_logprobs", "n", "stop",
            "suffix", # Responses API does not support suffix
            "stream_options", # Responses API does not support stream options
            "function_call", # Deprecated in favor of 'tool_choice'.
            "functions", # Deprecated in favor of 'tools'.

            # Fields that are dropped and manually handled in step 2.
            "reasoning_effort", "max_tokens",

            # Fields that are dropped and manually handled later in the pipe()
            "tools",
            "extra_tools", # Not a real OpenAI parm. Upstream filters may use it to add tools. The are appended to body["tools"] later in the pipe()

            # Fields not documented in OpenRouter's Responses API reference
            "store", "truncation",
            "user",
        }
        sanitized_params = {}
        for key, value in completions_dict.items():
            if key in unsupported_fields:
                logging.warning(f"Dropping unsupported parameter: '{key}'")
            else:
                sanitized_params[key] = value

        # Step 2: Apply transformations
        # Rename max_tokens → max_output_tokens
        if "max_tokens" in completions_dict:
            sanitized_params["max_output_tokens"] = completions_dict["max_tokens"]

        # reasoning_effort → reasoning.effort (without overwriting existing effort)
        effort = completions_dict.get("reasoning_effort")
        if effort:
            reasoning = sanitized_params.get("reasoning", {})
            reasoning.setdefault("effort", effort)
            sanitized_params["reasoning"] = reasoning

        # Legacy function_call → modern tool_choice
        if "tool_choice" not in sanitized_params and "function_call" in completions_dict:
            converted_choice = ResponsesBody._convert_function_call_to_tool_choice(
                completions_dict.get("function_call")
            )
            if converted_choice is not None:
                sanitized_params["tool_choice"] = converted_choice

        # Transform input messages to OpenAI Responses API format
        if "messages" in completions_dict:
            sanitized_params.pop("messages", None)
            replayed_reasoning_refs: list[Tuple[str, str]] = []
            # Resolve the transformer context. When provided, the transformer_context is typically
            # the Pipe instance so helper methods (upload, emit_status, etc.) are available.
            if transformer_context is None:
                raise RuntimeError(
                    "ResponsesBody.from_completions requires a transformer_context (usually the Pipe instance) "
                    "so multimodal helpers (file uploads, status events, etc.) are available."
                )
            transformer_owner = transformer_context
            raw_messages = completions_dict.get("messages", [])
            filtered_messages: list[dict[str, Any]] = []
            if isinstance(raw_messages, list):
                for msg in raw_messages:
                    if not isinstance(msg, dict):
                        continue
                    role = (msg.get("role") or "").lower()
                    if not role:
                        continue
                    filtered_messages.append(msg)

            sanitized_params["input"] = await transformer_owner.transform_messages_to_input(
                filtered_messages,
                chat_id=chat_id,
                openwebui_model_id=openwebui_model_id,
                artifact_loader=artifact_loader,
                pruning_turns=pruning_turns,
                replayed_reasoning_refs=replayed_reasoning_refs,
                __request__=request,
                user_obj=user_obj,
                event_emitter=event_emitter,
                model_id=completions_dict.get("model"),
                valves=transformer_valves or getattr(transformer_owner, "valves", None),
            )
            if replayed_reasoning_refs:
                sanitized_params["_replayed_reasoning_refs"] = replayed_reasoning_refs

        # Build the final ResponsesBody directly
        return cls(
            **sanitized_params,
            **extra_params  # Extra parameters that are passed to the ResponsesBody (e.g., custom parameters configured in Open WebUI model settings)
        )

ALLOWED_OPENROUTER_FIELDS = {
    "model",
    "models",
    "input",
    "instructions",
    "metadata",
    "stream",
    "max_output_tokens",
    "temperature",
    "top_k",
    "top_p",
    "reasoning",
    "include_reasoning",
    "tools",
    "tool_choice",
    "plugins",
    "response_format",
    "parallel_tool_calls",
    "user",
    "session_id",
    "transforms",
}

def _parse_model_fallback_csv(value: Any) -> list[str]:
    """Parse a comma-separated model list into a normalized array (order-preserving)."""
    if not isinstance(value, str):
        return []
    raw = value.strip()
    if not raw:
        return []
    models: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        candidate = part.strip()
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        models.append(candidate)
    return models


def _apply_model_fallback_to_payload(payload: dict[str, Any], *, logger: logging.Logger = LOGGER) -> None:
    """Map OWUI custom `model_fallback` (CSV string) to OpenRouter `models` (array).

    OpenRouter supports `model` plus `models` where `models` is treated as the fallback list.
    This helper never prepends `model` into `models`.
    """
    if not isinstance(payload, dict):
        return
    raw_fallback = payload.pop("model_fallback", None)
    fallback_models = _parse_model_fallback_csv(raw_fallback)

    existing_models_raw = payload.get("models")
    existing_models: list[str] = []
    if isinstance(existing_models_raw, list):
        for entry in existing_models_raw:
            if isinstance(entry, str) and entry.strip():
                existing_models.append(entry.strip())

    if not fallback_models and not existing_models:
        return

    # Merge existing + fallback (dedupe, preserve order). Existing models come first.
    merged: list[str] = []
    seen: set[str] = set()
    for candidate in existing_models + fallback_models:
        if candidate in seen:
            continue
        seen.add(candidate)
        merged.append(candidate)

    if not merged:
        payload.pop("models", None)
        return

    payload["models"] = merged
    if raw_fallback not in (None, "", []):
        logger.debug("Applied model_fallback -> models (%d fallback(s))", len(fallback_models))

def _sanitize_openrouter_metadata(raw: Any) -> Optional[dict[str, str]]:
    """Return a validated OpenRouter `metadata` dict or None.

    OpenRouter's Responses schema documents `metadata` as a string→string map with:
    - max 16 pairs
    - key <= 64 chars, no brackets
    - value <= 512 chars
    """
    if not isinstance(raw, dict):
        return None

    sanitized: dict[str, str] = {}
    for key, value in raw.items():
        if len(sanitized) >= _MAX_OPENROUTER_METADATA_PAIRS:
            break
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if len(key) > _MAX_OPENROUTER_METADATA_KEY_CHARS:
            continue
        if "[" in key or "]" in key:
            continue
        if len(value) > _MAX_OPENROUTER_METADATA_VALUE_CHARS:
            continue
        sanitized[key] = value

    return sanitized or None


def _apply_identifier_valves_to_payload(
    payload: dict[str, Any],
    *,
    valves: "Pipe.Valves",
    owui_metadata: dict[str, Any],
    owui_user_id: str,
    logger: logging.Logger = LOGGER,
) -> None:
    """Mutate request payload to include valve-gated identifiers.

    Rules (per operator requirements):
    - Only emit `metadata` when at least one identifier valve contributes a value.
    - When `SEND_END_USER_ID` is enabled, emit both top-level `user` and `metadata.user_id`.
    - When `SEND_SESSION_ID` is enabled, emit both top-level `session_id` and `metadata.session_id`.
    - `chat_id` and `message_id` have no OpenRouter top-level fields; they go into `metadata` only.
    """
    if not isinstance(payload, dict):
        return
    if not isinstance(owui_metadata, dict):
        owui_metadata = {}

    metadata_out: dict[str, str] = {}

    if valves.SEND_END_USER_ID:
        candidate = (owui_user_id or "").strip()
        if candidate and len(candidate) <= _MAX_OPENROUTER_ID_CHARS:
            payload["user"] = candidate
            metadata_out["user_id"] = candidate
        else:
            payload.pop("user", None)
            logger.debug("SEND_END_USER_ID enabled but OWUI user id missing/invalid; omitting `user`.")
    else:
        payload.pop("user", None)

    if valves.SEND_SESSION_ID:
        session_id = owui_metadata.get("session_id")
        if isinstance(session_id, str):
            candidate = session_id.strip()
            if candidate and len(candidate) <= _MAX_OPENROUTER_ID_CHARS:
                payload["session_id"] = candidate
                metadata_out["session_id"] = candidate
            else:
                payload.pop("session_id", None)
                logger.debug("SEND_SESSION_ID enabled but OWUI session_id missing/invalid; omitting `session_id`.")
        else:
            payload.pop("session_id", None)
    else:
        payload.pop("session_id", None)

    if valves.SEND_CHAT_ID:
        chat_id = owui_metadata.get("chat_id")
        if isinstance(chat_id, str):
            candidate = chat_id.strip()
            if candidate:
                metadata_out["chat_id"] = candidate[:_MAX_OPENROUTER_METADATA_VALUE_CHARS]

    if valves.SEND_MESSAGE_ID:
        message_id = owui_metadata.get("message_id")
        if isinstance(message_id, str):
            candidate = message_id.strip()
            if candidate:
                metadata_out["message_id"] = candidate[:_MAX_OPENROUTER_METADATA_VALUE_CHARS]

    if metadata_out:
        # Let the central sanitizer enforce length/bracket/pair constraints.
        payload["metadata"] = metadata_out
    else:
        payload.pop("metadata", None)


def _filter_openrouter_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Drop any keys not documented for the OpenRouter Responses API."""
    filtered: Dict[str, Any] = {}
    for key, value in payload.items():
        if key not in ALLOWED_OPENROUTER_FIELDS:
            continue

        # Drop explicit nulls; OpenRouter rejects nulls for optional fields.
        if value is None:
            continue

        if key == "top_k":
            if isinstance(value, (int, float)):
                filtered[key] = float(value)
                continue
            if isinstance(value, str):
                candidate = value.strip()
                if not candidate:
                    continue
                try:
                    filtered[key] = float(candidate)
                except ValueError:
                    continue
                continue
            continue

        if key == "metadata":
            value = _sanitize_openrouter_metadata(value)
            if value is None:
                continue

        if key == "reasoning":
            if not isinstance(value, dict):
                continue
            allowed_reasoning = {}
            for field_name in ("effort", "max_tokens", "exclude", "enabled", "summary"):
                if field_name in value:
                    allowed_reasoning[field_name] = value[field_name]
            if not allowed_reasoning:
                continue
            value = allowed_reasoning
        filtered[key] = value
    return filtered


def _filter_replayable_input_items(
    items: Any,
    *,
    logger: logging.Logger = LOGGER,
) -> Any:
    """Strip tool artifacts we must not replay back to the provider."""
    if not isinstance(items, list):
        return items

    filtered: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            filtered.append(item)
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type in _NON_REPLAYABLE_TOOL_ARTIFACTS:
            logger.debug(
                "Input sanitizer removed %s artifact at index %d (id=%s).",
                item_type,
                idx,
                item.get("id"),
            )
            continue
        filtered.append(item)

    if len(filtered) != len(items):
        return filtered
    return items


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Controller: Pipe
# ─────────────────────────────────────────────────────────────────────────────
# Primary interface implementing the Responses manifold
class Pipe:
    """Manifold entrypoint that adapts Open WebUI requests to OpenRouter."""
    id: str = _PIPE_RUNTIME_ID
    # 4.1 Configuration Schemas
    class Valves(BaseModel):
        """Global valve configuration shared across sessions."""
        # Connection & Auth
        BASE_URL: str = Field(
            default=((os.getenv("OPENROUTER_API_BASE_URL") or "").strip() or "https://openrouter.ai/api/v1"),
            description="OpenRouter API base URL. Override this if you are using a gateway or proxy.",
        )
        API_KEY: EncryptedStr = Field(
            default_factory=_default_api_key,
            description="Your OpenRouter API key. Defaults to the OPENROUTER_API_KEY environment variable.",
        )
        HTTP_CONNECT_TIMEOUT_SECONDS: int = Field(
            default=10,
            ge=1,
            description="Seconds to wait for the TCP/TLS connection to OpenRouter before failing.",
        )
        HTTP_TOTAL_TIMEOUT_SECONDS: Optional[int] = Field(
            default=None,
            ge=1,
            description="Overall HTTP timeout (seconds) for OpenRouter requests. Set to null to disable the total timeout so long-running streaming responses are not interrupted.",
        )
        HTTP_SOCK_READ_SECONDS: int = Field(
            default=300,
            ge=1,
            description="Idle read timeout (seconds) applied to active streams when HTTP_TOTAL_TIMEOUT_SECONDS is disabled. Generous default favors UX for slow providers.",
        )

        # Remote File/Image Download Settings
        REMOTE_DOWNLOAD_MAX_RETRIES: int = Field(
            default=3,
            ge=0,
            le=10,
            description="Maximum number of retry attempts for downloading remote images and files. Set to 0 to disable retries.",
        )
        REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS: int = Field(
            default=5,
            ge=1,
            le=60,
            description="Initial delay in seconds before the first retry attempt. Subsequent retries use exponential backoff (delay * 2^attempt).",
        )
        REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS: int = Field(
            default=45,
            ge=5,
            le=300,
            description="Maximum total time in seconds to spend on retry attempts. Retries will stop if this time limit is exceeded.",
        )

        REMOTE_FILE_MAX_SIZE_MB: int = Field(
            default=_REMOTE_FILE_MAX_SIZE_DEFAULT_MB,
            ge=1,
            le=_REMOTE_FILE_MAX_SIZE_MAX_MB,
            description="Maximum size in MB for downloading remote files/images. Files exceeding this limit are skipped. When Open WebUI RAG is enabled, the pipe automatically caps downloads to Open WebUI's FILE_MAX_SIZE (if set).",
        )
        SAVE_REMOTE_FILE_URLS: bool = Field(
            default=True,
            description="When True, remote URLs and data URLs in the file_url field are downloaded/parsed and re-hosted in Open WebUI storage. When False, file_url values pass through untouched. Note: This valve only affects the file_url field; see SAVE_FILE_DATA_CONTENT for file_data behavior. Recommended: Keep disabled to avoid unexpected storage growth.",
        )
        SAVE_FILE_DATA_CONTENT: bool = Field(
            default=True,
            description="When True, base64 content and URLs in the file_data field are parsed/downloaded and re-hosted in Open WebUI storage to prevent chat history bloat. When False, file_data values pass through untouched. Recommended: Keep enabled to avoid large inline payloads in chat history.",
        )
        BASE64_MAX_SIZE_MB: int = Field(
            default=50,
            ge=1,
            le=500,
            description="Maximum size in MB for base64-encoded files/images before decoding. Larger payloads will be rejected to prevent memory issues and excessive HTTP request sizes.",
        )
        IMAGE_UPLOAD_CHUNK_BYTES: int = Field(
            default=1 * 1024 * 1024,
            ge=64 * 1024,
            le=8 * 1024 * 1024,
            description="Maximum number of bytes to buffer at a time when loading Open WebUI-hosted images before forwarding them to a provider. Lower values reduce peak memory usage when multiple users edit images concurrently.",
        )
        VIDEO_MAX_SIZE_MB: int = Field(
            default=100,
            ge=1,
            le=1000,
            description="Maximum size in MB for video files (remote URLs or data URLs). Videos exceeding this limit will be rejected to prevent memory and bandwidth issues.",
        )
        FALLBACK_STORAGE_EMAIL: str = Field(
            default=(os.getenv("OPENROUTER_STORAGE_USER_EMAIL") or "openrouter-pipe@system.local"),
            description="Owner email used when multimodal uploads occur without a chat user (e.g., API automations).",
        )
        FALLBACK_STORAGE_NAME: str = Field(
            default=(os.getenv("OPENROUTER_STORAGE_USER_NAME") or "OpenRouter Pipe Storage"),
            description="Display name for the fallback storage owner.",
        )
        FALLBACK_STORAGE_ROLE: str = Field(
            default=(os.getenv("OPENROUTER_STORAGE_USER_ROLE") or "pending"),
            description="Role assigned to the fallback storage account when auto-created. Defaults to the low-privilege 'pending' role; override if your deployment needs a custom service role.",
        )
        ENABLE_SSRF_PROTECTION: bool = Field(
            default=True,
            description="Enable SSRF (Server-Side Request Forgery) protection for remote URL downloads. When enabled, blocks requests to private IP ranges (localhost, 192.168.x.x, 10.x.x.x, etc.) to prevent internal network probing.",
        )

        # Models
        MODEL_ID: str = Field(
            default="auto",
            description=(
                "Comma separated OpenRouter model IDs to expose in Open WebUI. "
                "Set to 'auto' to import every available Responses-capable model."
            ),
        )
        MODEL_CATALOG_REFRESH_SECONDS: int = Field(
            default=60 * 60,
            ge=60,
            description="How long to cache the OpenRouter model catalog (in seconds) before refreshing.",
        )

        ENABLE_REASONING: bool = Field(
            default=True,
            title="Show live reasoning",
            description="Request live reasoning traces whenever the selected model supports them.",
        )
        THINKING_OUTPUT_MODE: Literal["open_webui", "status", "both"] = Field(
            default="open_webui",
            title="Thinking output",
            description=(
                "Controls where in-progress thinking is surfaced while a response is being generated. "
                "'open_webui' streams reasoning in the Open WebUI reasoning box only; "
                "'status' emits thinking as status events only; "
                "'both' enables both outputs."
            ),
        )
        ENABLE_ANTHROPIC_INTERLEAVED_THINKING: bool = Field(
            default=True,
            title="Anthropic interleaved thinking",
            description=(
                "When True, enables Claude's interleaved thinking mode by sending "
                "`x-anthropic-beta: interleaved-thinking-2025-05-14` for `anthropic/...` models."
            ),
        )
        AUTO_CONTEXT_TRIMMING: bool = Field(
            default=True,
            title="Auto context trimming",
            description=(
                "When enabled, automatically attaches OpenRouter's `middle-out` transform so long prompts "
                "are trimmed from the middle instead of failing with context errors. Disable if your deployment "
                "manages `transforms` manually."
            ),
        )
        REASONING_EFFORT: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = Field(
            default="medium",
            title="Reasoning effort",
            description=(
                "Default reasoning effort to request from supported models. Use 'none' to skip reasoning entirely "
                "or 'xhigh' when maximum depth is desired (only on supporting models)."
            ),
        )
        REASONING_SUMMARY_MODE: Literal["auto", "concise", "detailed", "disabled"] = Field(
            default="auto",
            title="Reasoning summary",
            description="Controls the reasoning summary emitted by supported models (auto/concise/detailed). Set to 'disabled' to skip requesting reasoning summaries.",
        )
        GEMINI_THINKING_LEVEL: Literal["auto", "low", "high"] = Field(
            default="auto",
            title="Gemini 3 thinking level",
            description=(
                "Controls the thinking_level sent to Gemini 3.x models. 'auto' maps minimal/low effort to LOW and "
                "everything else to HIGH. Set explicitly to 'low' or 'high' to override."
            ),
        )
        GEMINI_THINKING_BUDGET: int = Field(
            default=1024,
            ge=0,
            le=65536,
            title="Gemini 2.5 thinking budget",
            description=(
                "Base thinking budget (tokens) for Gemini 2.5 models. When 0, thinking is disabled. "
                "When non-zero, the pipe scales this value based on reasoning effort (minimal → smaller, xhigh → larger)."
            ),
        )
        PERSIST_REASONING_TOKENS: Literal["disabled", "next_reply", "conversation"] = Field(
            default="conversation",
            title="Reasoning retention",
            description="Reasoning retention: 'disabled' keeps nothing, 'next_reply' keeps thoughts only until the following assistant reply finishes, and 'conversation' keeps them for the full chat history.",
        )
        TASK_MODEL_REASONING_EFFORT: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = Field(
            default="low",
            title="Task reasoning effort",
            description=(
                "Reasoning effort requested for Open WebUI task payloads (titles, tags, etc.) when they target this pipe's models. "
                "Low is the default balance between speed and quality; set to 'minimal' to prioritize fastest runs (and disable auto web-search), "
                "or use medium/high for progressively deeper background reasoning at higher cost."
            ),
        )
        
        # Tool execution behavior
        PERSIST_TOOL_RESULTS: bool = Field(
            default=True,
            title="Keep tool results",
            description="Persist tool call results across conversation turns. When disabled, tool results stay ephemeral.",
        )
        ARTIFACT_ENCRYPTION_KEY: EncryptedStr = Field(
            default_factory=_default_artifact_encryption_key,
            description="Min 16 chars. Encrypt reasoning tokens (and optionally all persisted artifacts). Changing the key creates a new table; prior artifacts become inaccessible.",
        )
        ENCRYPT_ALL: bool = Field(
            default=True,
            description="Encrypt every persisted artifact when ARTIFACT_ENCRYPTION_KEY is set. When False, only reasoning tokens are encrypted.",
        )
        ENABLE_LZ4_COMPRESSION: bool = Field(
            default=True,
            description="When True (and lz4 is available), compress large encrypted artifacts to reduce database read/write overhead.",
        )
        MIN_COMPRESS_BYTES: int = Field(
            default=0,
            ge=0,
            description="Payloads at or above this size (in bytes) are candidates for LZ4 compression before encryption. The default 0 always attempts compression; raise the value to skip tiny payloads.",
        )

        ENABLE_STRICT_TOOL_CALLING: bool = Field(
            default=True,
            description=(
                "When True, converts Open WebUI registry tools to strict JSON Schema for OpenAI tools, "
                "enforcing explicit types, required fields, and disallowing additionalProperties."
            ),
        )
        MAX_FUNCTION_CALL_LOOPS: int = Field(
            default=10,
            description=(
                "Maximum number of full execution cycles (loops) allowed per request. "
                "Each loop involves the model generating one or more function/tool calls, "
                "executing all requested functions, and feeding the results back into the model. "
                "Looping stops when this limit is reached or when the model no longer requests "
                "additional tool or function calls."
            )
        )

        # Web search
        ENABLE_WEB_SEARCH_TOOL: bool = Field(
            default=True,
            description="Enable the OpenRouter web-search plugin (id='web') when supported by the selected model.",
        )
        WEB_SEARCH_MAX_RESULTS: Optional[int] = Field(
            default=3,
            ge=1,
            le=10,
            description="Number of web results to request when the web-search plugin is enabled (1-10). Set to null to use the provider default.",
        )

        # Logging
        LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
            default_factory=_resolve_log_level_default,
            description="Select logging level.  Recommend INFO or WARNING for production use. DEBUG is useful for development and debugging.",
        )
        SESSION_LOG_STORE_ENABLED: bool = Field(
            default=False,
            description=(
                "When True, persist per-request SessionLogger output to encrypted zip files on disk. "
                "Persistence is skipped when any required IDs are missing (user_id, session_id, chat_id, message_id)."
            ),
        )
        SESSION_LOG_DIR: str = Field(
            default="session_logs",
            description=(
                "Base directory for encrypted session log archives. "
                "Files are stored under <SESSION_LOG_DIR>/<user_id>/<chat_id>/<message_id>.zip."
            ),
        )
        SESSION_LOG_ZIP_PASSWORD: EncryptedStr = Field(
            default=EncryptedStr(""),
            description=(
                "Password used to encrypt session log zip files (pyzipper AES). "
                "Recommend using a long random passphrase and encrypting the value (requires WEBUI_SECRET_KEY)."
            ),
        )
        SESSION_LOG_RETENTION_DAYS: int = Field(
            default=90,
            ge=1,
            description="Retention window for stored session log archives. Cleanup deletes zip files older than this many days.",
        )
        SESSION_LOG_CLEANUP_INTERVAL_SECONDS: int = Field(
            default=3600,
            ge=60,
            description="How often (in seconds) to run the session log cleanup loop when storage is enabled.",
        )
        SESSION_LOG_ZIP_COMPRESSION: Literal["stored", "deflated", "bzip2", "lzma"] = Field(
            default="lzma",
            description="Zip compression algorithm for session log archives (default lzma).",
        )
        SESSION_LOG_ZIP_COMPRESSLEVEL: Optional[int] = Field(
            default=None,
            ge=0,
            le=9,
            description=(
                "Compression level (0-9) for deflated/bzip2 zip compression. "
                "Ignored for stored/lzma."
            ),
        )
        SESSION_LOG_MAX_LINES: int = Field(
            default=20000,
            ge=100,
            le=200000,
            description="Maximum number of in-memory SessionLogger lines retained per request (older lines are dropped).",
        )
        MAX_CONCURRENT_REQUESTS: int = Field(
            default=200,
            ge=1,
            le=2000,
            description="Maximum number of in-flight OpenRouter requests allowed per process.",
        )
        SSE_WORKERS_PER_REQUEST: int = Field(
            default=4,
            ge=1,
            le=8,
            description="Number of per-request SSE worker tasks that parse streamed chunks.",
        )
        STREAMING_CHUNK_QUEUE_MAXSIZE: int = Field(
            default=0,
            ge=0,
            description="Maximum number of raw SSE chunks buffered before applying backpressure to the OpenRouter stream. 0=unbounded (deadlock-proof, recommended); bounded values &lt;500 risk hangs on tool-heavy loads or slow DB/emit (drain block → event full → workers block → chunk full → producer halt).",
        )
        STREAMING_EVENT_QUEUE_MAXSIZE: int = Field(
            default=0,
            ge=0,
            description="Maximum number of parsed SSE events buffered ahead of downstream processing. 0=unbounded (deadlock-proof, recommended); bounded values &lt;500 risk hangs on tool-heavy loads or slow DB/emit (drain block → event full → workers block → chunk full → producer halt).",

        )
        STREAMING_EVENT_QUEUE_WARN_SIZE: int = Field(
            default=1000,
            ge=100,
            description="Log warning when event_queue.qsize() hits this threshold (unbounded queue monitoring); ge=100 avoids spam on sustained high load. Tune higher for noisy envs.",
        )
        OPENROUTER_ERROR_TEMPLATE: str = Field(
            default=DEFAULT_OPENROUTER_ERROR_TEMPLATE,
            description=(
                "Markdown template used when OpenRouter rejects a request with status 400. "
                "Placeholders such as {heading}, {detail}, {sanitized_detail}, {provider}, {model_identifier}, "
                "{requested_model}, {api_model_id}, {normalized_model_id}, {openrouter_code}, {upstream_type}, "
                "{reason}, {request_id}, {request_id_reference}, {openrouter_message}, {upstream_message}, "
                "{moderation_reasons}, {flagged_excerpt}, {raw_body}, {context_limit_tokens}, {max_output_tokens}, "
                "{include_model_limits}, {metadata_json}, {provider_raw_json}, {error_id}, {timestamp}, {session_id}, {user_id}, "
                "{native_finish_reason}, {error_chunk_id}, {error_chunk_created}, {streaming_provider}, {streaming_model}, "
                "{retry_after_seconds}, {rate_limit_type}, {required_cost}, and {account_balance} are replaced when values are available. "
                "Lines containing placeholders are omitted automatically when the referenced value is missing or empty. "
                "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to render only when truthy."
            ),
        )

        AUTHENTICATION_ERROR_TEMPLATE: str = Field(
            default=DEFAULT_AUTHENTICATION_ERROR_TEMPLATE,
            description=(
                "Markdown template for HTTP 401 errors. Available placeholders include {error_id}, {timestamp}, {openrouter_code}, {openrouter_message}, "
                "{session_id}, {user_id}, {support_email}, {support_url}, and any metadata extracted from the OpenRouter payload."
            ),
        )

        INSUFFICIENT_CREDITS_TEMPLATE: str = Field(
            default=DEFAULT_INSUFFICIENT_CREDITS_TEMPLATE,
            description=(
                "Markdown template for HTTP 402 errors when the account is out of credits. Supports {error_id}, {timestamp}, {openrouter_code}, "
                "{openrouter_message}, {required_cost}, {account_balance}, {support_email}, and other shared context variables."
            ),
        )

        RATE_LIMIT_TEMPLATE: str = Field(
            default=DEFAULT_RATE_LIMIT_TEMPLATE,
            description=(
                "Markdown template for HTTP 429 rate-limit errors. Use placeholders such as {error_id}, {timestamp}, {openrouter_code}, {retry_after_seconds}, "
                "{rate_limit_type}, {support_email}, and the standard context variables."
            ),
        )

        SERVER_TIMEOUT_TEMPLATE: str = Field(
            default=DEFAULT_SERVER_TIMEOUT_TEMPLATE,
            description=(
                "Markdown template for HTTP 408 errors returned by OpenRouter (server-side timeout). Supports the common context variables plus "
                "{openrouter_message}, {openrouter_code}, and support contact placeholders."
            ),
        )

        # Support configuration
        SUPPORT_EMAIL: str = Field(
            default="",
            description=(
                "Support email displayed in error messages. "
                "Leave empty if self-hosted without dedicated support."
            )
        )

        SUPPORT_URL: str = Field(
            default="",
            description=(
                "Support URL (e.g., internal ticket system, Slack channel). "
                "Shown in error messages if provided."
            )
        )

        # Additional error templates
        NETWORK_TIMEOUT_TEMPLATE: str = Field(
            default=DEFAULT_NETWORK_TIMEOUT_TEMPLATE,
            description=(
                "Markdown template for network timeout errors. "
                "Available variables: {error_id}, {timeout_seconds}, {timestamp}, "
                "{session_id}, {user_id}, {support_email}. "
                "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to render only when truthy."
            )
        )

        CONNECTION_ERROR_TEMPLATE: str = Field(
            default=DEFAULT_CONNECTION_ERROR_TEMPLATE,
            description=(
                "Markdown template for connection failures. "
                "Available variables: {error_id}, {error_type}, {timestamp}, "
                "{session_id}, {user_id}, {support_email}. "
                "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to render only when truthy."
            )
        )

        SERVICE_ERROR_TEMPLATE: str = Field(
            default=DEFAULT_SERVICE_ERROR_TEMPLATE,
            description=(
                "Markdown template for OpenRouter 5xx errors. "
                "Available variables: {error_id}, {status_code}, {reason}, {timestamp}, "
                "{session_id}, {user_id}, {support_email}. "
                "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to render only when truthy."
            )
        )

        INTERNAL_ERROR_TEMPLATE: str = Field(
            default=DEFAULT_INTERNAL_ERROR_TEMPLATE,
            description=(
                "Markdown template for unexpected internal errors. "
                "Available variables: {error_id}, {error_type}, {timestamp}, "
                "{session_id}, {user_id}, {support_email}, {support_url}. "
                "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to render only when truthy."
            )
        )

        MAX_PARALLEL_TOOLS_GLOBAL: int = Field(
            default=200,
            ge=1,
            le=2000,
            description="Global ceiling for simultaneously executing tool calls.",
        )
        MAX_PARALLEL_TOOLS_PER_REQUEST: int = Field(
            default=5,
            ge=1,
            le=50,
            description="Per-request concurrency limit for tool execution workers.",
        )
        BREAKER_MAX_FAILURES: int = Field(
            default=5,
            ge=1,
            le=50,
            description=(
                "Number of failures allowed per breaker window before requests, tools, or DB ops are temporarily blocked. "
                "Set higher to reduce trip frequency in noisy environments."
            ),
        )
        BREAKER_WINDOW_SECONDS: int = Field(
            default=60,
            ge=5,
            le=900,
            description="Sliding window length (in seconds) used when counting breaker failures.",
        )
        BREAKER_HISTORY_SIZE: int = Field(
            default=5,
            ge=1,
            le=200,
            description=(
                "Maximum failures remembered per user/tool breaker. Increase when using very high BREAKER_MAX_FAILURES so history is not truncated."
            ),
        )
        TOOL_BATCH_CAP: int = Field(
            default=4,
            ge=1,
            le=32,
            description="Maximum number of compatible tool calls that may be executed in a single batch.",
        )
        TOOL_OUTPUT_RETENTION_TURNS: int = Field(
            default=10,
            ge=0,
            description=(
                "Number of most recent logical turns whose tool outputs are sent in full. "
                "A turn starts when a user speaks and includes the assistant/tool responses "
                "that follow until the next user message. Older turns have their persisted "
                "tool outputs pruned to save tokens. Set to 0 to keep every tool output."
            ),
        )
        TOOL_TIMEOUT_SECONDS: int = Field(
            default=60,
            ge=1,
            le=600,
            description="Max seconds to wait for an individual tool to finish before timing out. Generous default reduces disruption for real-world tools.",
        )
        TOOL_BATCH_TIMEOUT_SECONDS: int = Field(
            default=120,
            ge=1,
            description="Max seconds to wait for a batch of tool calls to complete before timing out. Longer default keeps complex batches from being interrupted prematurely.",
        )
        TOOL_IDLE_TIMEOUT_SECONDS: Optional[int] = Field(
            default=None,
            ge=1,
            description="Idle timeout (seconds) between tool executions in a queue. Set to null for unlimited idle time so intermittent tool usage does not fail unexpectedly.",
        )
        ENABLE_REDIS_CACHE: bool = Field(
            default=True,
            description="Enable Redis write-behind cache when REDIS_URL + multi-worker detected.",
        )
        REDIS_CACHE_TTL_SECONDS: int = Field(
            default=600,
            ge=60,
            le=3600,
            description="TTL applied to Redis artifact cache entries (seconds).",
        )
        REDIS_PENDING_WARN_THRESHOLD: int = Field(
            default=100,
            ge=1,
            le=10000,
            description="Emit a warning when the Redis pending queue exceeds this number of artifacts.",
        )
        REDIS_FLUSH_FAILURE_LIMIT: int = Field(
            default=5,
            ge=1,
            le=50,
            description="Disable Redis caching after this many consecutive flush failures (falls back to direct DB writes).",
        )
        COSTS_REDIS_DUMP: bool = Field(
            default=False,
            description="When True, push per-request usage snapshots into Redis for downstream cost analytics.",
        )
        COSTS_REDIS_TTL_SECONDS: int = Field(
            default=900,
            ge=60,
            le=3600,
            description="TTL (seconds) applied to cost analytics Redis snapshots.",
        )
        ARTIFACT_CLEANUP_DAYS: int = Field(
            default=90,
            ge=1,
            le=365,
            description="Retention window (days) for artifact cleanup scheduler.",
        )
        ARTIFACT_CLEANUP_INTERVAL_HOURS: float = Field(
            default=1.0,
            ge=0.5,
            le=24,
            description="Frequency (hours) for the artifact cleanup worker to wake up.",
        )
        DB_BATCH_SIZE: int = Field(
            default=10,
            ge=5,
            le=20,
            description="Number of artifacts to commit per DB batch.",
        )
        USE_MODEL_MAX_OUTPUT_TOKENS: bool = Field(
            default=False,
            description="When enabled, automatically include the provider's max_output_tokens in each request. Disable to omit the parameter entirely.",
        )
        SHOW_FINAL_USAGE_STATUS: bool = Field(
            default=True,
            description="When True, the final status message includes elapsed time, cost, and token usage.",
        )
        ENABLE_STATUS_CSS_PATCH: bool = Field(
            default=True,
            description="When True, injects a CSS tweak via __event_call__ to show multi-line status descriptions in Open WebUI (experimental).",
        )
        SEND_END_USER_ID: bool = Field(
            default=False,
            description="When True, send OpenRouter `user` using the OWUI user GUID, and also include `metadata.user_id`.",
        )
        SEND_SESSION_ID: bool = Field(
            default=False,
            description="When True, send OpenRouter `session_id` using OWUI metadata and also include `metadata.session_id`.",
        )
        SEND_CHAT_ID: bool = Field(
            default=False,
            description="When True, include OWUI chat_id as `metadata.chat_id` (metadata only).",
        )
        SEND_MESSAGE_ID: bool = Field(
            default=False,
            description="When True, include OWUI message_id as `metadata.message_id` (metadata only).",
        )
        MAX_INPUT_IMAGES_PER_REQUEST: int = Field(
            default=5,
            ge=1,
            le=20,
            description="Maximum number of image inputs (user attachments plus assistant fallbacks) to include in a single provider request.",
        )
        IMAGE_INPUT_SELECTION: Literal["user_turn_only", "user_then_assistant"] = Field(
            default="user_then_assistant",
            description=(
                "Controls which images are forwarded to the provider. "
                "'user_turn_only' restricts inputs to the images supplied with the current user message. "
                "'user_then_assistant' falls back to the most recent assistant-generated images when the user did not attach any."
            ),
        )


    class UserValves(BaseModel):
        """Per-user valve overrides."""

        model_config = ConfigDict(populate_by_name=True)
        @model_validator(mode="before")
        @classmethod
        def _normalize_inherit(cls, values):
            """Treat the literal string 'inherit' (any case) as an unset value.

            """
            if not isinstance(values, dict):
                return values

            normalized: dict[str, Any] = {}
            for key, val in values.items():
                if isinstance(val, str):
                    stripped = val.strip()
                    lowered = stripped.lower()
                    if lowered == "inherit":
                        normalized[key] = None
                        continue
                normalized[key] = val
            return normalized

        SHOW_FINAL_USAGE_STATUS: bool = Field(
            default=True,
            title="Show usage details",
            description="Display tokens, time, and cost at the end of each reply.",
        )
        ENABLE_REASONING: bool = Field(
            default=True,
            title="Show reasoning steps",
            description="While the AI works, show its step-by-step reasoning when supported.",
        )
        THINKING_OUTPUT_MODE: Literal["open_webui", "status", "both"] = Field(
            default="open_webui",
            title="Thinking output",
            description=(
                "Choose where to show the model's thinking while it works: "
                "'open_webui' uses the Open WebUI reasoning box, "
                "'status' uses status messages, "
                "or 'both' shows both."
            ),
        )
        ENABLE_ANTHROPIC_INTERLEAVED_THINKING: bool = Field(
            default=True,
            title="Interleaved thinking (Claude)",
            description=(
                "When enabled, request Claude's interleaved thinking stream by sending "
                "`x-anthropic-beta: interleaved-thinking-2025-05-14` for `anthropic/...` models."
            ),
        )
        REASONING_EFFORT: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = Field(
            default="medium",
            title="Reasoning depth",
            description="Choose how much thinking the AI should do before answering (higher depth is slower but more thorough). Use 'none' to disable reasoning or 'xhigh' for maximum depth when available.",
        )
        REASONING_SUMMARY_MODE: Literal["auto", "concise", "detailed", "disabled"] = Field(
            default="auto",
            title="Reasoning explanation detail",
            description="Pick how detailed the reasoning summary should be (auto, concise, detailed, or hidden).",
        )
        PERSIST_REASONING_TOKENS: Literal["disabled", "next_reply", "conversation"] = Field(
            default="next_reply",
            title="How long to keep reasoning",
            description="Choose whether reasoning is kept just for the next reply or the entire conversation.",
        )
        PERSIST_TOOL_RESULTS: bool = Field(
            default=True,
            title="Remember tool and search results",
            description="Let the AI reuse outputs from tools (for example web searches or other apps) later in the conversation.",
        )

    # Core Structure — shared concurrency primitives
    _QUEUE_MAXSIZE = 500
    _request_queue: asyncio.Queue[_PipeJob] | None = None
    _queue_worker_task: asyncio.Task | None = None
    _queue_worker_lock: asyncio.Lock | None = None
    _global_semaphore: asyncio.Semaphore | None = None
    _semaphore_limit: int = 0
    _tool_global_semaphore: asyncio.Semaphore | None = None
    _tool_global_limit: int = 0
    _log_queue: asyncio.Queue[logging.LogRecord] | None = None
    _log_queue_loop: asyncio.AbstractEventLoop | None = None
    _log_worker_task: asyncio.Task | None = None
    _log_worker_lock: asyncio.Lock | None = None
    _TOOL_CONTEXT: ContextVar[Optional[_ToolExecutionContext]] = ContextVar(
        "openrouter_tool_context",
        default=None,
    )
    _cleanup_task: asyncio.Task | None = None

    # 4.2 Constructor and Entry Points
    def __init__(self):
        """Initialize valve defaults and lazy-initialized resources."""
        self.type = "manifold"
        self.valves = self.Valves()  # Note: valve values are not accessible in __init__. Access from pipes() or pipe() methods.
        self.logger = SessionLogger.get_logger(__name__)
        decrypted_encryption_key = EncryptedStr.decrypt(self.valves.ARTIFACT_ENCRYPTION_KEY)
        self._encryption_key: str = (decrypted_encryption_key or "").strip()
        self._encrypt_all: bool = bool(self.valves.ENCRYPT_ALL)
        self._compression_min_bytes: int = self.valves.MIN_COMPRESS_BYTES
        self._compression_enabled: bool = bool(
            self.valves.ENABLE_LZ4_COMPRESSION and lz4frame is not None
        )
        self._fernet: Fernet | None = None
        self._engine: Engine | None = None
        self._session_factory: sessionmaker | None = None
        self._item_model: Type[Any] | None = None
        self._artifact_table_name: str | None = None
        self._db_executor: ThreadPoolExecutor | None = None
        self._artifact_store_signature: tuple[str, str] | None = None
        self._lz4_warning_emitted = False
        breaker_history_size = self.valves.BREAKER_HISTORY_SIZE
        self._breaker_records: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=breaker_history_size)
        )
        self._breaker_threshold = self.valves.BREAKER_MAX_FAILURES
        self._breaker_window_seconds = self.valves.BREAKER_WINDOW_SECONDS
        self._tool_breakers: dict[str, dict[str, deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=breaker_history_size))
        )
        self._db_breakers: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=breaker_history_size)
        )
        self._user_insert_param_names: tuple[str, ...] | None = None
        self._startup_task: asyncio.Task | None = None
        self._startup_checks_started = False
        self._startup_checks_pending = False
        self._startup_checks_complete = False
        self._warmup_failed = False
        self._redis_url = (os.getenv("REDIS_URL") or "").strip()
        self._websocket_manager = (os.getenv("WEBSOCKET_MANAGER") or "").strip().lower()
        self._websocket_redis_url = (os.getenv("WEBSOCKET_REDIS_URL") or "").strip()
        raw_uvicorn_workers = (os.getenv("UVICORN_WORKERS") or "1").strip()
        try:
            uvicorn_workers = int(raw_uvicorn_workers or "1")
        except ValueError:
            self.logger.warning("Invalid UVICORN_WORKERS value '%s'; defaulting to 1.", raw_uvicorn_workers)
            uvicorn_workers = 1
        multi_worker = uvicorn_workers > 1
        redis_url_configured = bool(self._redis_url)
        websocket_ready = (
            self._websocket_manager == "redis"
            and bool(self._websocket_redis_url)
        )
        redis_valve_enabled = bool(self.valves.ENABLE_REDIS_CACHE)
        if multi_worker and not redis_valve_enabled:
            self.logger.warning("Multiple UVicorn workers detected but ENABLE_REDIS_CACHE is disabled; Redis cache remains off.")
        if multi_worker and redis_valve_enabled:
            if not redis_url_configured:
                self.logger.warning("Multiple UVicorn workers detected but REDIS_URL is unset; Redis cache remains off.")
            elif self._websocket_manager != "redis":
                self.logger.warning("Multiple UVicorn workers detected but WEBSOCKET_MANAGER is not 'redis'; Redis cache remains off.")
            elif not websocket_ready:
                self.logger.warning("Multiple UVicorn workers detected but WEBSOCKET_REDIS_URL is unset; Redis cache remains off.")
        self._redis_candidate = (
            redis_url_configured
            and multi_worker
            and websocket_ready
            and aioredis is not None
            and redis_valve_enabled
        )
        self._redis_enabled = False
        self._redis_client: Optional[_RedisClient] = None
        self._redis_listener_task: asyncio.Task | None = None
        self._redis_flush_task: asyncio.Task | None = None
        self._redis_ready_task: asyncio.Task | None = None
        self._redis_namespace = (getattr(self, "id", None ) or "openrouter").lower()
        self._redis_pending_key = f"{self._redis_namespace}:pending"
        self._redis_cache_prefix = f"{self._redis_namespace}:artifact"
        self._redis_flush_lock_key = f"{self._redis_namespace}:flush_lock"
        self._redis_ttl = self.valves.REDIS_CACHE_TTL_SECONDS
        self._cleanup_task = None
        self._legacy_tool_warning_emitted = False
        self._storage_user_cache: Optional[Any] = None
        self._storage_user_lock: Optional[asyncio.Lock] = None
        self._storage_role_warning_emitted: bool = False
        self._session_log_queue: queue.Queue[_SessionLogArchiveJob] | None = None
        self._session_log_stop_event: threading.Event | None = None
        self._session_log_worker_thread: threading.Thread | None = None
        self._session_log_cleanup_thread: threading.Thread | None = None
        self._session_log_lock = threading.Lock()
        self._session_log_cleanup_interval_seconds = self.valves.SESSION_LOG_CLEANUP_INTERVAL_SECONDS
        self._session_log_retention_days = self.valves.SESSION_LOG_RETENTION_DAYS
        self._session_log_dirs: set[str] = set()
        self._session_log_warning_emitted = False
        self._maybe_start_log_worker()
        self._maybe_start_startup_checks()

    # Core Structure -------------------------------------------------
    def _maybe_start_startup_checks(self) -> None:
        """Schedule background warmup checks once an event loop is available."""
        if self._startup_checks_complete:
            return
        if self._startup_task and not self._startup_task.done():
            return
        if self._startup_task and self._startup_task.done():
            self._startup_task = None
        api_key_available = bool(EncryptedStr.decrypt(self.valves.API_KEY))
        if not api_key_available:
            if not self._startup_checks_pending:
                self.logger.debug("Deferring OpenRouter warmup until an API key is configured.")
            self._startup_checks_pending = True
            self._startup_checks_started = False
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (import-time). Defer until the first async entrypoint.
            self._startup_checks_pending = True
            return

        if self._startup_checks_started and not self._startup_checks_pending:
            return

        self._startup_checks_started = True
        self._startup_checks_pending = False
        self._startup_task = loop.create_task(self._run_startup_checks(), name="openrouter-warmup")

    def _maybe_start_log_worker(self) -> None:
        """Ensure the async logging queue + worker are started."""
        cls = type(self)
        if cls._log_worker_lock is None:
            cls._log_worker_lock = asyncio.Lock()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if cls._log_queue is None or cls._log_queue_loop is not loop:
            stale_worker = cls._log_worker_task
            if stale_worker and not stale_worker.done():
                with contextlib.suppress(Exception):
                    stale_worker.cancel()
            cls._log_worker_task = None
            cls._log_queue = asyncio.Queue(maxsize=1000)
            cls._log_queue_loop = loop
            SessionLogger.set_log_queue(cls._log_queue)
        SessionLogger.set_main_loop(loop)

        async def _ensure_worker() -> None:
            async with cls._log_worker_lock:  # type: ignore[arg-type]
                if cls._log_worker_task and not cls._log_worker_task.done():
                    return
                if cls._log_queue is None:
                    cls._log_queue = asyncio.Queue(maxsize=1000)
                    SessionLogger.set_log_queue(cls._log_queue)
                cls._log_worker_task = loop.create_task(
                    cls._log_worker_loop(),
                    name="openrouter-log-worker",
                )

        loop.create_task(_ensure_worker())

    def _maybe_start_session_log_workers(self) -> None:
        """Start session log writer + cleanup threads if not already running."""
        if self._session_log_worker_thread and self._session_log_worker_thread.is_alive():
            return
        if self._session_log_cleanup_thread and self._session_log_cleanup_thread.is_alive():
            return
        if self._session_log_queue is None:
            self._session_log_queue = queue.Queue(maxsize=500)
        if self._session_log_stop_event is None:
            self._session_log_stop_event = threading.Event()

        def _writer_loop() -> None:
            while True:
                if self._session_log_stop_event and self._session_log_stop_event.is_set():
                    break
                item: _SessionLogArchiveJob | None = None
                try:
                    item = self._session_log_queue.get(timeout=0.5) if self._session_log_queue else None
                except queue.Empty:
                    continue
                except Exception:
                    continue
                if item is None:
                    with contextlib.suppress(Exception):
                        if self._session_log_queue:
                            self._session_log_queue.task_done()
                    continue
                try:
                    self._write_session_log_archive(item)
                except Exception:
                    # Never allow the writer thread to crash the process.
                    self.logger.debug("Session log writer failed", exc_info=True)
                finally:
                    with contextlib.suppress(Exception):
                        if self._session_log_queue:
                            self._session_log_queue.task_done()

        def _cleanup_loop() -> None:
            while True:
                if self._session_log_stop_event and self._session_log_stop_event.is_set():
                    break
                try:
                    self._cleanup_session_log_archives()
                except Exception:
                    # Never allow cleanup to crash the process.
                    self.logger.debug("Session log cleanup failed", exc_info=True)
                interval = 3600
                with contextlib.suppress(Exception):
                    with self._session_log_lock:
                        interval = self._session_log_cleanup_interval_seconds
                try:
                    time.sleep(interval)
                except Exception:
                    time.sleep(60)

        self._session_log_worker_thread = threading.Thread(
            target=_writer_loop,
            name="openrouter-session-log-writer",
            daemon=True,
        )
        self._session_log_worker_thread.start()

        self._session_log_cleanup_thread = threading.Thread(
            target=_cleanup_loop,
            name="openrouter-session-log-cleanup",
            daemon=True,
        )
        self._session_log_cleanup_thread.start()

    def _stop_session_log_workers(self) -> None:
        """Stop session log background threads (best effort)."""
        if self._session_log_stop_event:
            with contextlib.suppress(Exception):
                self._session_log_stop_event.set()
        if self._session_log_queue:
            with contextlib.suppress(Exception):
                self._session_log_queue.put_nowait(None)  # type: ignore[arg-type]
        for thread in (self._session_log_worker_thread, self._session_log_cleanup_thread):
            if thread and thread.is_alive():
                with contextlib.suppress(Exception):
                    thread.join(timeout=2.0)
        self._session_log_worker_thread = None
        self._session_log_cleanup_thread = None

    def _enqueue_session_log_archive(
        self,
        valves: "Pipe.Valves",
        *,
        user_id: str,
        session_id: str,
        chat_id: str,
        message_id: str,
        request_id: str,
        log_lines: list[str],
    ) -> None:
        """Queue the current request's session logs for encrypted zip persistence."""
        if not valves.SESSION_LOG_STORE_ENABLED:
            return
        if not (user_id and session_id and chat_id and message_id):
            return
        if not log_lines:
            return
        if pyzipper is None:
            if not self._session_log_warning_emitted:
                self.logger.warning("Session log storage is enabled but the 'pyzipper' package is not available; skipping persistence.")
                self._session_log_warning_emitted = True
            return

        base_dir = valves.SESSION_LOG_DIR.strip()
        if not base_dir:
            if not self._session_log_warning_emitted:
                self.logger.warning("Session log storage is enabled but SESSION_LOG_DIR is empty; skipping persistence.")
                self._session_log_warning_emitted = True
            return

        decrypted = EncryptedStr.decrypt(valves.SESSION_LOG_ZIP_PASSWORD)
        password = (decrypted or "").strip()
        if not password:
            if not self._session_log_warning_emitted:
                self.logger.warning("Session log storage is enabled but SESSION_LOG_ZIP_PASSWORD is not configured; skipping persistence.")
                self._session_log_warning_emitted = True
            return

        zip_compression = valves.SESSION_LOG_ZIP_COMPRESSION
        zip_compresslevel = valves.SESSION_LOG_ZIP_COMPRESSLEVEL
        if zip_compression in {"stored", "lzma"}:
            zip_compresslevel = None

        with contextlib.suppress(Exception):
            with self._session_log_lock:
                self._session_log_cleanup_interval_seconds = valves.SESSION_LOG_CLEANUP_INTERVAL_SECONDS
                self._session_log_retention_days = valves.SESSION_LOG_RETENTION_DAYS
                self._session_log_dirs.add(base_dir)

        if self._session_log_queue is None:
            self._session_log_queue = queue.Queue(maxsize=500)
        if self._session_log_queue.full():
            self.logger.warning("Session log archive queue is full; dropping archive for chat_id=%s message_id=%s", chat_id, message_id)
            return

        self._maybe_start_session_log_workers()
        job = _SessionLogArchiveJob(
            base_dir=base_dir,
            zip_password=password.encode("utf-8"),
            zip_compression=zip_compression,
            zip_compresslevel=zip_compresslevel,
            user_id=user_id,
            session_id=session_id,
            chat_id=chat_id,
            message_id=message_id,
            request_id=request_id,
            created_at=time.time(),
            log_lines=log_lines,
        )
        try:
            self._session_log_queue.put_nowait(job)
        except Exception:
            self.logger.debug("Failed to enqueue session log archive job", exc_info=True)

    def _maybe_start_redis(self) -> None:
        """Initialize Redis cache if enabled."""
        if not self._redis_candidate or self._redis_enabled:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._redis_ready_task and not self._redis_ready_task.done():
            return
        self._redis_ready_task = loop.create_task(self._init_redis_client(), name="openrouter-redis-init")

    def _maybe_start_cleanup(self) -> None:
        if self._cleanup_task and not self._cleanup_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._cleanup_task and self._cleanup_task.done():
            self._cleanup_task = None
        self._cleanup_task = loop.create_task(
            self._artifact_cleanup_worker(),
            name="openrouter-artifact-cleanup",
        )

    async def _run_startup_checks(self) -> None:
        """Warm OpenRouter connections and log readiness without blocking startup."""
        session: aiohttp.ClientSession | None = None
        try:
            api_key = EncryptedStr.decrypt(self.valves.API_KEY)
            if not api_key:
                self.logger.debug("Skipping OpenRouter warmup: API key missing (will retry when configured).")
                self._startup_checks_pending = True
                return
            session = self._create_http_session()
            await self._ping_openrouter(session, self.valves.BASE_URL, api_key)
            self.logger.debug("Warmed: success")
            self._warmup_failed = False
            self._startup_checks_complete = True
            self._startup_checks_pending = False
        except Exception as exc:  # pragma: no cover - depends on IO
            self.logger.warning("OpenRouter warmup failed: %s", exc)
            self._warmup_failed = True
            self._startup_checks_complete = False
            self._startup_checks_pending = True
        finally: 
            if session:
                with contextlib.suppress(Exception):
                    await session.close()
            self._startup_checks_started = False
            self._startup_task = None

    async def _ping_openrouter(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        api_key: str,
    ) -> None:
        """Issue a lightweight GET to prime DNS/TLS caches."""
        url = base_url.rstrip("/") + "/models?limit=1"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Title": _OPENROUTER_TITLE,
            "HTTP-Referer": _OPENROUTER_REFERER,
        }
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
            reraise=True,
        ):
            with attempt:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    resp.raise_for_status()
                    await resp.read()

    async def _init_redis_client(self) -> None:
        if not self._redis_candidate or self._redis_enabled or not self._redis_url:
            return
        if aioredis is None:
            self.logger.warning("Redis cache requested but redis-py is unavailable.")
            return
        client: Optional[_RedisClient] = None
        try:
            client = aioredis.from_url(self._redis_url, encoding="utf-8", decode_responses=True)
            if client is None:
                self.logger.warning("Redis client initialization returned None; Redis cache remains disabled.")
                return
            await _wait_for(client.ping(), timeout=5.0)
        except Exception as exc:
            if client is not None:
                with contextlib.suppress(Exception):
                    await client.close()
            self._redis_enabled = False
            self._redis_client = None
            self.logger.warning("Redis cache disabled (%s)", exc)
            return

        self._redis_client = client
        self._redis_enabled = True
        self.logger.info("Redis cache enabled for namespace '%s'", self._redis_namespace)
        loop = asyncio.get_running_loop()
        self._redis_listener_task = loop.create_task(self._redis_pubsub_listener(), name="openrouter-redis-listener")
        self._redis_flush_task = loop.create_task(self._redis_periodic_flusher(), name="openrouter-redis-flush")

    async def _redis_pubsub_listener(self) -> None:
        if not self._redis_client:
            return
        pubsub = self._redis_client.pubsub()
        try:
            await pubsub.subscribe(_REDIS_FLUSH_CHANNEL)
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                await self._flush_redis_queue()
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            pass
        except Exception as exc:
            self.logger.warning("Redis pub/sub listener stopped: %s (falling back to timer)", exc)
        finally:
            with contextlib.suppress(Exception):
                await pubsub.close()

    async def _redis_periodic_flusher(self) -> None:
        consecutive_failures = 0

        while self._redis_enabled:
            warn_threshold = self.valves.REDIS_PENDING_WARN_THRESHOLD
            failure_limit = self.valves.REDIS_FLUSH_FAILURE_LIMIT
            try:
                if not self._redis_client:
                    break

                queue_depth = await _wait_for(
                    self._redis_client.llen(self._redis_pending_key)
                )
                if queue_depth > warn_threshold:
                    self.logger.warning("⚠️ Redis pending queue backed up: %d items (threshold: %d)", queue_depth, warn_threshold)
                elif queue_depth > 0:
                    self.logger.debug("Redis pending queue depth: %d items", queue_depth)

                await self._flush_redis_queue()
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                self.logger.error("Periodic flush failed (%d/%d consecutive failures): %s", consecutive_failures, failure_limit, exc, exc_info=self.logger.isEnabledFor(logging.DEBUG))
                if consecutive_failures >= failure_limit:
                    self.logger.critical("🚨 Disabling Redis cache after %d consecutive flush failures. Falling back to direct DB writes.", failure_limit)
                    self._redis_enabled = False
                    break

            await asyncio.sleep(10)

        self.logger.debug("Redis periodic flusher stopped")


    async def _flush_redis_queue(self) -> None:
        if not (self._redis_enabled and self._redis_client):
            return

        lock_token = secrets.token_hex(16)
        lock_acquired = False
        try:
            lock_acquired = bool(
                await _wait_for(
                    self._redis_client.set(
                        self._redis_flush_lock_key,
                        lock_token,
                        nx=True,
                        ex=5,
                    )
                )
            )
            if not lock_acquired:
                self.logger.debug("Skipping Redis flush: another worker holds the lock")
                return

            rows: list[dict[str, Any]] = []
            raw_entries: list[str] = []
            batch_size = self.valves.DB_BATCH_SIZE
            while len(rows) < batch_size:
                data = await _wait_for(self._redis_client.lpop(self._redis_pending_key))
                if data is None:
                    break
                entry: str | None = None
                if isinstance(data, str):
                    entry = data
                elif isinstance(data, bytes):
                    entry = data.decode("utf-8", errors="replace")
                else:
                    self.logger.warning(
                        "Unexpected Redis queue payload type '%s'; skipping entry.",
                        type(data).__name__,
                    )
                    continue
                raw_entries.append(entry)
                try:
                    parsed = json.loads(entry)
                except json.JSONDecodeError as exc:
                    self.logger.warning("Malformed JSON in pending queue, skipping: %s", exc)
                    continue
                if not isinstance(parsed, dict):
                    self.logger.warning("Pending queue entry must be an object; skipping malformed payload.")
                    continue
                rows.append(parsed)
            if not raw_entries:
                return
            if not rows:
                self.logger.warning("Discarded %d malformed artifact(s) from Redis pending queue.", len(raw_entries))
                return

            self.logger.debug("Flushing %d artifact(s) from Redis pending queue to DB (table: %s)", len(rows), self._artifact_table_name or "unknown")
            try:
                await self._db_persist_direct(rows)
                self.logger.debug("✅ Successfully flushed %d artifacts to DB", len(rows))
            except Exception as exc:
                self.logger.error("❌ DB flush failed! %d artifacts could not be persisted: %s", len(rows), exc, exc_info=True)
                try:
                    await self._redis_requeue_entries(raw_entries)
                    self.logger.debug("Re-queued %d artifact(s) back to Redis pending queue after DB failure", len(raw_entries))
                except Exception as requeue_exc:  # pragma: no cover - defensive
                    self.logger.critical("Failed to re-queue %d artifact(s) after DB failure: %s", len(raw_entries), requeue_exc)
        finally:
            if lock_acquired and self._redis_client:
                release_script = (
                    "if redis.call('get', KEYS[1]) == ARGV[1] then "
                    "return redis.call('del', KEYS[1]) "
                    "else return 0 end"
                )
                try:
                    await _wait_for(
                        self._redis_client.eval(
                            release_script,
                            1,
                            self._redis_flush_lock_key,
                            lock_token,
                        )
                    )
                except Exception:
                    # Redis errors during lock release are non-fatal - continue pipe operation
                    self.logger.debug("Failed to release Redis flush lock", exc_info=self.logger.isEnabledFor(logging.DEBUG))

    def _redis_cache_key(self, chat_id: Optional[str], row_id: Optional[str]) -> Optional[str]:
        if not (chat_id and row_id):
            return None
        return f"{self._redis_cache_prefix}:{chat_id}:{row_id}"

    async def _redis_enqueue_rows(self, rows: list[dict[str, Any]]) -> list[str]:
        """Enqueue artifacts into Redis for asynchronous DB flushing."""
        if not rows:
            return []

        if not (self._redis_enabled and self._redis_client):
            return await self._db_persist_direct(rows)

        for row in rows:
            row.setdefault("id", generate_item_id())

        try:
            pipe = self._redis_client.pipeline()
            for row in rows:
                serialized = json.dumps(row, ensure_ascii=False)
                pipe.rpush(self._redis_pending_key, serialized)
            await _wait_for(pipe.execute())

            await self._redis_cache_rows(rows)
            await _wait_for(self._redis_client.publish(_REDIS_FLUSH_CHANNEL, "flush"))

            self.logger.debug("Enqueued %d artifacts to Redis pending queue", len(rows))
            return [row["id"] for row in rows]
        except Exception as exc:
            self.logger.warning("Redis enqueue failed, falling back to direct DB write: %s", exc)
            return await self._db_persist_direct(rows)

    async def _redis_cache_rows(self, rows: list[dict[str, Any]], *, chat_id: Optional[str] = None) -> None:
        if not (self._redis_enabled and self._redis_client):
            return
        pipe = self._redis_client.pipeline()
        for row in rows:
            row_payload = row if "payload" in row else {"payload": row}
            cache_key = self._redis_cache_key(row.get("chat_id") or chat_id, row.get("id"))
            if not cache_key:
                continue
            pipe.setex(cache_key, self._redis_ttl, json.dumps(row_payload, ensure_ascii=False))
        await _wait_for(pipe.execute())

    async def _redis_requeue_entries(self, entries: list[str]) -> None:
        """Push raw JSON entries back onto the pending queue after a DB failure."""
        if not (entries and self._redis_client):
            return
        pipe = self._redis_client.pipeline()
        for payload in reversed(entries):
            pipe.lpush(self._redis_pending_key, payload)
        pipe.expire(self._redis_pending_key, max(self._redis_ttl, 60))
        await _wait_for(pipe.execute())

    async def _redis_fetch_rows(
        self,
        chat_id: Optional[str],
        item_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        if not (self._redis_enabled and self._redis_client and chat_id and item_ids):
            return {}
        keys: list[str] = []
        id_lookup: list[str] = []
        for item_id in item_ids:
            cache_key = self._redis_cache_key(chat_id, item_id)
            if cache_key:
                keys.append(cache_key)
                id_lookup.append(item_id)
        if not keys:
            return {}
        values = await _wait_for(self._redis_client.mget(keys))
        cached: dict[str, dict[str, Any]] = {}
        for item_id, raw in zip(id_lookup, values):
            if not raw:
                continue
            try:
                row_data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            payload = row_data.get("payload", row_data) if isinstance(row_data, dict) else row_data
            is_encrypted = False
            if isinstance(row_data, dict):
                is_encrypted = bool(row_data.get("is_encrypted"))
            if not is_encrypted and isinstance(payload, dict):
                is_encrypted = "ciphertext" in payload
            if is_encrypted:
                ciphertext = ""
                if isinstance(payload, dict):
                    ciphertext = payload.get("ciphertext", "") or ""
                elif isinstance(row_data, dict) and isinstance(row_data.get("payload"), dict):
                    ciphertext = row_data["payload"].get("ciphertext", "") or ""
                try:
                    payload = self._decrypt_payload(ciphertext)
                except Exception as exc:
                    self.logger.warning("Failed to decrypt cached artifact %s: %s", item_id, exc, exc_info=self.logger.isEnabledFor(logging.DEBUG))
                    continue
            if isinstance(payload, dict):
                cached[item_id] = payload
        return cached

    async def _artifact_cleanup_worker(self) -> None:
        while True:
            try:
                await self._run_cleanup_once()
            except asyncio.CancelledError:  # pragma: no cover - shutdown
                break
            except Exception as exc:
                self.logger.warning("Artifact cleanup failed: %s", exc)
            interval_hours = self.valves.ARTIFACT_CLEANUP_INTERVAL_HOURS
            interval_seconds = interval_hours * 3600
            jitter = min(600.0, interval_seconds * 0.25)
            await asyncio.sleep(interval_seconds + random.uniform(0, jitter))

    async def _run_cleanup_once(self) -> None:
        if not (self._item_model and self._session_factory):
            return
        cutoff_days = self.valves.ARTIFACT_CLEANUP_DAYS
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=cutoff_days)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._db_executor,
            functools.partial(self._cleanup_sync, cutoff),
        )

    def _cleanup_sync(self, cutoff: datetime.datetime) -> None:
        if not (self._session_factory and self._item_model):
            return
        session: Session = self._session_factory()  # type: ignore[call-arg]
        try:
            deleted = (
                session.query(self._item_model)
                .filter(self._item_model.created_at < cutoff)
                .delete(synchronize_session=False)
            )
            session.commit()
            if deleted:
                self.logger.debug("Cleanup removed %s rows older than %s", deleted, cutoff)
        except Exception as exc:
            session.rollback()
            raise exc
        finally:
            session.close()

    async def _ensure_concurrency_controls(self, valves: "Pipe.Valves") -> None:
        """Lazy-initialize queue worker and semaphore with the latest valves."""
        cls = type(self)
        if cls._queue_worker_lock is None:
            cls._queue_worker_lock = asyncio.Lock()

        async with cls._queue_worker_lock:
            if cls._request_queue is None:
                cls._request_queue = asyncio.Queue(maxsize=self._QUEUE_MAXSIZE)
                self.logger.debug("Created request queue (maxsize=%s)", self._QUEUE_MAXSIZE)

            if cls._queue_worker_task is None or cls._queue_worker_task.done():
                loop = asyncio.get_running_loop()
                cls._queue_worker_task = loop.create_task(
                    cls._request_worker_loop(),
                    name="openrouter-pipe-dispatch",
                )
                self.logger.debug("Started request queue worker")

            target = valves.MAX_CONCURRENT_REQUESTS
            if cls._global_semaphore is None:
                cls._global_semaphore = asyncio.Semaphore(target)
                cls._semaphore_limit = target
                self.logger.debug("Initialized semaphore (limit=%s)", target)
            elif target > cls._semaphore_limit:
                delta = target - cls._semaphore_limit
                for _ in range(delta):
                    cls._global_semaphore.release()
                cls._semaphore_limit = target
                self.logger.info("Increased MAX_CONCURRENT_REQUESTS to %s", target)
            elif target < cls._semaphore_limit:
                self.logger.warning("Lower MAX_CONCURRENT_REQUESTS (%s→%s) requires restart to take full effect.", cls._semaphore_limit, target)

            target_tool = valves.MAX_PARALLEL_TOOLS_GLOBAL
            if cls._tool_global_semaphore is None:
                cls._tool_global_semaphore = asyncio.Semaphore(target_tool)
                cls._tool_global_limit = target_tool
                self.logger.debug("Initialized tool semaphore (limit=%s)", target_tool)
            elif target_tool > cls._tool_global_limit:
                delta = target_tool - cls._tool_global_limit
                for _ in range(delta):
                    cls._tool_global_semaphore.release()
                cls._tool_global_limit = target_tool
                self.logger.info("Increased MAX_PARALLEL_TOOLS_GLOBAL to %s", target_tool)
            elif target_tool < cls._tool_global_limit:
                self.logger.warning("Lower MAX_PARALLEL_TOOLS_GLOBAL (%s→%s) requires restart to take full effect.", cls._tool_global_limit, target_tool)

    def _enqueue_job(self, job: _PipeJob) -> bool:
        """Attempt to enqueue a request, returning False when the queue is full."""
        queue = type(self)._request_queue
        if queue is None:
            raise RuntimeError("Request queue not initialized")
        try:
            queue.put_nowait(job)
            self.logger.debug("Enqueued request %s (depth=%s)", job.request_id, queue.qsize())
            return True
        except asyncio.QueueFull:
            self.logger.warning("Request queue full (max=%s)", queue.maxsize)
            return False

    @classmethod
    async def _request_worker_loop(cls) -> None:
        """Background worker that dequeues jobs and spawns per-request tasks."""
        queue = cls._request_queue
        if queue is None:
            return
        try:
            while True:
                job = await queue.get()
                if job.future.cancelled():
                    queue.task_done()
                    continue
                task = asyncio.create_task(job.pipe._execute_pipe_job(job))

                def _mark_done(_task: asyncio.Task, q=queue) -> None:
                    q.task_done()

                task.add_done_callback(_mark_done)

                def _propagate_cancel(fut: asyncio.Future, _task: asyncio.Task = task, _job_id: str = job.request_id) -> None:
                    if fut.cancelled() and not _task.done():
                        job.pipe.logger.debug("Cancelling in-flight request (request_id=%s)", _job_id)
                        _task.cancel()

                job.future.add_done_callback(_propagate_cancel)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            return

    async def _stop_request_worker(self) -> None:
        """Stop the shared queue worker and drain pending items."""
        cls = type(self)
        worker = cls._queue_worker_task
        if worker:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            cls._queue_worker_task = None
        cls._request_queue = None

    @classmethod
    async def _log_worker_loop(cls) -> None:
        """Drain log records asynchronously to keep handlers non-blocking."""
        queue = cls._log_queue
        if queue is None:
            return
        try:
            while True:
                record = await queue.get()
                try:
                    SessionLogger.process_record(record)
                finally:
                    queue.task_done()
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            pass
        finally:
            if queue is not None:
                while not queue.empty():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        record = queue.get_nowait()
                        SessionLogger.process_record(record)
                        queue.task_done()

    async def _stop_log_worker(self) -> None:
        """Stop the log worker and clear the queue."""
        cls = type(self)
        worker = cls._log_worker_task
        if worker:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            cls._log_worker_task = None
        cls._log_queue = None
        cls._log_queue_loop = None
        SessionLogger.set_log_queue(None)

    async def _shutdown_tool_context(self, context: _ToolExecutionContext) -> None:
        """Gracefully stop per-request tool workers."""
        try:
            active_workers = [task for task in context.workers if not task.done()]
            worker_count = len(active_workers)
            if worker_count:
                for _ in range(worker_count):
                    await context.queue.put(None)
                await context.queue.join()
        except Exception:
            # Ignore errors during graceful shutdown - workers will be cancelled anyway
            pass
        for task in context.workers:
            if not task.done():
                task.cancel()
        if context.workers:
            await asyncio.gather(*context.workers, return_exceptions=True)

    async def _tool_worker_loop(self, context: _ToolExecutionContext) -> None:
        """Process queued tool calls with batching/timeouts."""
        pending: list[tuple[_QueuedToolCall | None, bool]] = []
        batch_cap = context.batch_cap
        idle_timeout = context.idle_timeout
        try:
            while True:
                if pending:
                    item, from_queue = pending.pop(0)
                else:
                    try:
                        get_coro = context.queue.get()
                        queued = (
                            await get_coro
                            if idle_timeout is None
                            else await asyncio.wait_for(get_coro, timeout=idle_timeout)
                        )
                    except asyncio.TimeoutError:
                        message = (
                            f"Tool queue idle for {idle_timeout:.0f}s; cancelling pending work."
                            if idle_timeout
                            else "Tool queue idle timeout triggered."
                        )
                        context.timeout_error = context.timeout_error or message
                        self.logger.warning("%s", message)
                        break
                    item, from_queue = queued, True
                if item is None:
                    if from_queue:
                        context.queue.task_done()
                    if pending:
                        continue
                    break

                batch: list[tuple[_QueuedToolCall, bool]] = [(item, from_queue)]
                if item.allow_batch:
                    while len(batch) < batch_cap:
                        try:
                            nxt = context.queue.get_nowait()
                            from_queue_next = True
                        except asyncio.QueueEmpty:
                            break
                        if nxt is None:
                            pending.insert(0, (None, True))
                            break
                        if nxt.allow_batch and self._can_batch_tool_calls(batch[0][0], nxt):
                            batch.append((nxt, from_queue_next))
                        else:
                            pending.insert(0, (nxt, from_queue_next))
                            break

                await self._execute_tool_batch([itm for itm, _ in batch], context)
                for _, flag in batch:
                    if flag:
                        context.queue.task_done()
        finally:
            # Resolve remaining futures if the worker is stopping unexpectedly
            while pending:
                leftover, from_queue = pending.pop(0)
                if from_queue:
                    context.queue.task_done()
                if leftover is None:
                    continue
                if not leftover.future.done():
                    error_msg = context.timeout_error or "Tool execution cancelled"
                    leftover.future.set_result(
                        self._build_tool_output(
                            leftover.call,
                            error_msg,
                            status="cancelled",
                        )
                    )

    def _can_batch_tool_calls(self, first: _QueuedToolCall, candidate: _QueuedToolCall) -> bool:
        if first.call.get("name") != candidate.call.get("name"):
            return False

        dep_keys = {"depends_on", "_depends_on", "sequential", "no_batch"}
        if any(key in first.args or key in candidate.args for key in dep_keys):
            return False

        first_id = first.call.get("call_id")
        candidate_id = candidate.call.get("call_id")
        if first_id and self._args_reference_call(candidate.args, first_id):
            return False
        if candidate_id and self._args_reference_call(first.args, candidate_id):
            return False
        return True

    def _args_reference_call(self, args: Any, call_id: str) -> bool:
        if isinstance(args, str):
            return call_id in args
        if isinstance(args, dict):
            return any(self._args_reference_call(value, call_id) for value in args.values())
        if isinstance(args, list):
            return any(self._args_reference_call(item, call_id) for item in args)
        return False

    async def _execute_tool_batch(
        self,
        batch: list[_QueuedToolCall],
        context: _ToolExecutionContext,
    ) -> None:
        if not batch:
            return
        self.logger.debug("Batched %s tool(s) for %s", len(batch), batch[0].call.get("name"))
        tasks = [self._invoke_tool_call(item, context) for item in batch]
        gather_coro = asyncio.gather(*tasks, return_exceptions=True)
        # TODO: Add soak tests that cover multi-minute batch executions to validate these generous timeouts.
        results: list[tuple[str, str] | BaseException] = []
        try:
            if context.batch_timeout:
                results = await asyncio.wait_for(gather_coro, timeout=context.batch_timeout)
            else:
                results = await gather_coro
        except asyncio.TimeoutError:
            message = (
                f"Tool batch '{batch[0].call.get('name')}' exceeded {context.batch_timeout:.0f}s and was cancelled."
                if context.batch_timeout
                else "Tool batch timed out."
            )
            context.timeout_error = context.timeout_error or message
            self.logger.warning("%s", message)
            for item in batch:
                tool_type = (item.tool_cfg.get("type") or "function").lower()
                self._record_tool_failure_type(context.user_id, tool_type)
                if not item.future.done():
                    item.future.set_result(
                        self._build_tool_output(
                            item.call,
                            message,
                            status="failed",
                        )
                    )
            return
        for item, result in zip(batch, results):
            if item.future.done():
                continue
            if isinstance(result, BaseException):
                payload = self._build_tool_output(
                    item.call,
                    f"Tool error: {result}",
                    status="failed",
                )
            else:
                status, text = result
                payload = self._build_tool_output(item.call, text, status=status)
                tool_type = (item.tool_cfg.get("type") or "function").lower()
                self._reset_tool_failure_type(context.user_id, tool_type)
            item.future.set_result(payload)

    async def _invoke_tool_call(
        self,
        item: _QueuedToolCall,
        context: _ToolExecutionContext,
    ) -> tuple[str, str]:
        tool_type = (item.tool_cfg.get("type") or "function").lower()
        if not self._tool_type_allows(context.user_id, tool_type):
            await self._notify_tool_breaker(context, tool_type, item.call.get("name"))
            return (
                "skipped",
                f"Tool '{item.call.get('name')}' temporarily disabled due to repeated errors.",
            )

        async with context.per_request_semaphore:
            if context.global_semaphore is not None:
                async with self._acquire_tool_global(context.global_semaphore, item.call.get("name")):
                    return await self._run_tool_with_retries(item, context, tool_type)
            return await self._run_tool_with_retries(item, context, tool_type)

    async def _run_tool_with_retries(
        self,
        item: _QueuedToolCall,
        context: _ToolExecutionContext,
        tool_type: str,
    ) -> tuple[str, str]:
        fn = item.tool_cfg.get("callable")
        if not callable(fn):
            message = f"Tool '{item.call.get('name')}' is missing a callable handler."
            self.logger.warning("%s", message)
            self._record_tool_failure_type(context.user_id, tool_type)
            return ("failed", message)
        fn_to_call = cast(ToolCallable, fn)
        timeout = float(context.timeout)
        retryer = AsyncRetrying(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=0.2, min=0.2, max=1),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        try:
            async for attempt in retryer:
                with attempt:
                    result = await asyncio.wait_for(
                        self._call_tool_callable(fn_to_call, item.args),
                        timeout=timeout,
                    )
                    self._reset_tool_failure_type(context.user_id, tool_type)
                    text = "" if result is None else str(result)
                    return ("completed", text)
        except Exception as exc:
            self._record_tool_failure_type(context.user_id, tool_type)
            raise exc
        return ("failed", "Tool execution produced no output.")

    async def _call_tool_callable(self, fn: ToolCallable, args: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(fn):
            return await fn(**args)
        result = await asyncio.to_thread(fn, **args)
        if inspect.isawaitable(result):
            return await result
        return result

    @contextlib.asynccontextmanager
    async def _acquire_tool_global(self, semaphore: asyncio.Semaphore, tool_name: str | None):
        self.logger.debug("Waiting for global tool slot (%s)", tool_name)
        await semaphore.acquire()
        try:
            yield
        finally:
            semaphore.release()


    async def _execute_pipe_job(self, job: _PipeJob) -> None:
        """Isolate per-request context, HTTP session, and semaphore slot."""
        semaphore = type(self)._global_semaphore
        if semaphore is None:
            job.future.set_exception(RuntimeError("Semaphore unavailable"))
            return

        session: aiohttp.ClientSession | None = None
        tokens: list[tuple[ContextVar[Any], contextvars.Token[Any]]] = []
        tool_context: _ToolExecutionContext | None = None
        tool_token: contextvars.Token[Optional[_ToolExecutionContext]] | None = None
        stream_queue = job.stream_queue
        stream_emitter = (
            self._make_middleware_stream_emitter(job, stream_queue)
            if stream_queue is not None
            else None
        )
        try:
            async with self._acquire_semaphore(semaphore, job.request_id):
                session = self._create_http_session(job.valves)
                tokens = self._apply_logging_context(job)
                tool_queue: asyncio.Queue[_QueuedToolCall | None] = asyncio.Queue(maxsize=50)
                per_request_tool_sem = asyncio.Semaphore(job.valves.MAX_PARALLEL_TOOLS_PER_REQUEST)
                per_tool_timeout = float(job.valves.TOOL_TIMEOUT_SECONDS)
                batch_timeout = float(max(per_tool_timeout, job.valves.TOOL_BATCH_TIMEOUT_SECONDS))
                idle_timeout_value = job.valves.TOOL_IDLE_TIMEOUT_SECONDS
                idle_timeout = float(idle_timeout_value) if idle_timeout_value else None
                self.logger.debug("Tool timeouts (request=%s): per_call=%ss batch=%ss idle=%s", job.request_id, per_tool_timeout, batch_timeout, idle_timeout if idle_timeout is not None else "disabled")
                tool_context = _ToolExecutionContext(
                    queue=tool_queue,
                    per_request_semaphore=per_request_tool_sem,
                    global_semaphore=type(self)._tool_global_semaphore,
                    timeout=float(per_tool_timeout),
                    batch_timeout=batch_timeout,
                    idle_timeout=idle_timeout,
                    user_id=job.user_id,
                    event_emitter=stream_emitter or job.event_emitter,
                    batch_cap=job.valves.TOOL_BATCH_CAP,
                )
                worker_count = job.valves.MAX_PARALLEL_TOOLS_PER_REQUEST
                for worker_idx in range(worker_count):
                    tool_context.workers.append(
                        asyncio.create_task(
                            self._tool_worker_loop(tool_context),
                            name=f"openrouter-tool-worker-{job.request_id}-{worker_idx}",
                        )
                    )
                tool_token = self._TOOL_CONTEXT.set(tool_context)
                result = await self._handle_pipe_call(
                    job.body,
                    job.user,
                    job.request,
                    stream_emitter or job.event_emitter,
                    job.event_call,
                    job.metadata,
                    job.tools,
                    job.task,
                    job.task_body,
                    valves=job.valves,
                    session=session,
                )
                if not job.future.done():
                    job.future.set_result(result)
                self._reset_failure_counter(job.user_id)
        except Exception as exc:
            self._record_failure(job.user_id)
            if stream_queue is not None and not job.future.cancelled():
                with contextlib.suppress(Exception):
                    await stream_queue.put({"error": {"detail": str(exc)}})
            if not job.future.done():
                job.future.set_exception(exc)
        finally:
            if stream_queue is not None:
                with contextlib.suppress(Exception):
                    await stream_queue.put(None)
            if tool_context:
                await self._shutdown_tool_context(tool_context)
            if tool_token is not None:
                self._TOOL_CONTEXT.reset(tool_token)
            for var, token in tokens:
                with contextlib.suppress(Exception):
                    var.reset(token)
            if session:
                with contextlib.suppress(Exception):
                    await session.close()

    def _create_http_session(self, valves: Pipe.Valves | None = None) -> aiohttp.ClientSession:
        """Return a fresh ClientSession with sane defaults for per-request use."""
        valves = valves or self.valves
        connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=10,
            keepalive_timeout=75,
            ttl_dns_cache=300,
        )
        connect_timeout = float(valves.HTTP_CONNECT_TIMEOUT_SECONDS)
        total_timeout_value = valves.HTTP_TOTAL_TIMEOUT_SECONDS
        total_timeout = float(total_timeout_value) if total_timeout_value else None
        sock_read = float(valves.HTTP_SOCK_READ_SECONDS) if total_timeout is None else None
        timeout = aiohttp.ClientTimeout(total=total_timeout, connect=connect_timeout, sock_read=sock_read)
        self.logger.debug("HTTP timeouts: connect=%ss total=%s sock_read=%s", connect_timeout, total_timeout if total_timeout is not None else "disabled", sock_read if sock_read is not None else "disabled")
        return aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            json_serialize=json.dumps,
        )

    @contextlib.asynccontextmanager
    async def _acquire_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        request_id: str,
    ):
        """Async context manager that logs semaphore acquisition/release."""
        self.logger.debug("Waiting for semaphore (request=%s)", request_id)
        await semaphore.acquire()
        self.logger.debug("Semaphore acquired (request=%s)", request_id)
        try:
            yield
        finally:
            semaphore.release()
            self.logger.debug("Semaphore released (request=%s)", request_id)

    def _apply_logging_context(self, job: _PipeJob) -> list[tuple[ContextVar[Any], contextvars.Token[Any]]]:
        """Set SessionLogger contextvars based on the incoming request."""
        session_id = job.session_id or None
        request_id = job.request_id or None
        user_id = job.user_id or None
        log_level = getattr(logging, job.valves.LOG_LEVEL)
        with contextlib.suppress(Exception):
            SessionLogger.set_max_lines(job.valves.SESSION_LOG_MAX_LINES)
        tokens: list[tuple[ContextVar[Any], contextvars.Token[Any]]] = []
        tokens.append((SessionLogger.session_id, SessionLogger.session_id.set(session_id)))
        tokens.append((SessionLogger.request_id, SessionLogger.request_id.set(request_id)))
        tokens.append((SessionLogger.user_id, SessionLogger.user_id.set(user_id)))
        tokens.append((SessionLogger.log_level, SessionLogger.log_level.set(log_level)))
        return tokens

    def _breaker_allows(self, user_id: str) -> bool:
        """Per-user breaker governed by BREAKER_MAX_FAILURES and BREAKER_WINDOW_SECONDS."""
        if not user_id:
            return True
        window = self._breaker_records[user_id]
        now = time.time()
        while window and now - window[0] > self._breaker_window_seconds:
            window.popleft()
        return len(window) < self._breaker_threshold

    def _record_failure(self, user_id: str) -> None:
        if not user_id:
            return
        self._breaker_records[user_id].append(time.time())

    def _reset_failure_counter(self, user_id: str) -> None:
        if user_id and user_id in self._breaker_records:
            self._breaker_records[user_id].clear()

    def _tool_type_allows(self, user_id: str, tool_type: str) -> bool:
        if not user_id or not tool_type:
            return True
        window = self._tool_breakers[user_id][tool_type]
        now = time.time()
        while window and now - window[0] > self._breaker_window_seconds:
            window.popleft()
        return len(window) < self._breaker_threshold

    def _record_tool_failure_type(self, user_id: str, tool_type: str) -> None:
        if not user_id or not tool_type:
            return
        self._tool_breakers[user_id][tool_type].append(time.time())

    def _reset_tool_failure_type(self, user_id: str, tool_type: str) -> None:
        if user_id and tool_type and user_id in self._tool_breakers:
            self._tool_breakers[user_id][tool_type].clear()

    async def _notify_tool_breaker(
        self,
        context: _ToolExecutionContext,
        tool_type: str,
        tool_name: Optional[str],
    ) -> None:
        if not context.event_emitter:
            return
        try:
            await context.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "description": (
                            f"Skipping {tool_name or tool_type} tools due to repeated failures"
                        ),
                        "done": False,
                    },
                }
            )
        except Exception:
            # Event emitter failures (client disconnect, etc.) shouldn't stop pipe
            self.logger.debug("Failed to emit breaker notification", exc_info=True)

    def _db_breaker_allows(self, user_id: str) -> bool:
        if not user_id:
            return True
        window = self._db_breakers[user_id]
        now = time.time()
        while window and now - window[0] > self._breaker_window_seconds:
            window.popleft()
        return len(window) < self._breaker_threshold

    def _record_db_failure(self, user_id: str) -> None:
        if user_id:
            self._db_breakers[user_id].append(time.time())

    def _reset_db_failure(self, user_id: str) -> None:
        if user_id and user_id in self._db_breakers:
            self._db_breakers[user_id].clear()

    def _ensure_artifact_store(self, valves: "Pipe.Valves", pipe_identifier: Optional[str] = None) -> None:
        """Configure encryption/compression + ensure the backing table exists."""
        decrypted_encryption_key = EncryptedStr.decrypt(valves.ARTIFACT_ENCRYPTION_KEY)
        encryption_key = (decrypted_encryption_key or "").strip()
        self._encryption_key = encryption_key
        self._encrypt_all = bool(valves.ENCRYPT_ALL)
        self._compression_min_bytes = valves.MIN_COMPRESS_BYTES

        wants_compression = bool(valves.ENABLE_LZ4_COMPRESSION)
        compression_enabled = wants_compression and lz4frame is not None
        if wants_compression and lz4frame is None and not self._lz4_warning_emitted:
            self.logger.warning("LZ4 compression requested but the 'lz4' package is not available. Artifacts will be stored without compression.")
            self._lz4_warning_emitted = True
        self._compression_enabled = compression_enabled

        pipe_identifier = pipe_identifier or getattr(self, "id", None)
        if not pipe_identifier:
            raise RuntimeError("Pipe identifier is missing; Open WebUI did not assign an id to this manifold.")
        table_fragment = _sanitize_table_fragment(pipe_identifier)
        desired_signature = (table_fragment, self._encryption_key)
        if (
            self._artifact_store_signature == desired_signature
            and self._item_model is not None
            and self._session_factory is not None
            and self._engine is not None
        ):
            return

        self._init_artifact_store(
            pipe_identifier=pipe_identifier,
            table_fragment=table_fragment,
        )

    def _init_artifact_store(
        self,
        pipe_identifier: Optional[str] = None,
        *,
        table_fragment: Optional[str] = None,
    ) -> None:
        """Initialize the per-pipe SQLAlchemy model + executor for artifact storage."""
        engine: Engine | None = None
        session_factory: sessionmaker | None = None
        base: Any | None = None

        try:
            from open_webui.internal import db as owui_db  # type: ignore
        except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
            owui_db = None

        if owui_db is not None:
            engine = getattr(owui_db, "engine", None)
            session_factory = getattr(owui_db, "SessionLocal", None)
            base = getattr(owui_db, "Base", None)

        if not (engine and session_factory and base):
            self.logger.warning("Artifact persistence disabled: Open WebUI database helpers are unavailable.")
            self._engine = None
            self._session_factory = None
            self._item_model = None
            self._artifact_table_name = None
            self._artifact_store_signature = None
            return

        pipe_identifier = pipe_identifier or getattr(self, "id", None)
        if not pipe_identifier:
            raise RuntimeError("Pipe identifier is required to initialize the artifact store.")
        encryption_key = self._encryption_key
        hash_source = f"{encryption_key}{pipe_identifier}".encode("utf-8", "ignore")
        key_hash = hashlib.sha256(hash_source).hexdigest()
        table_fragment = table_fragment or _sanitize_table_fragment(pipe_identifier)
        table_name = f"response_items_{table_fragment}_{key_hash[:8]}"
        class_name = f"ResponseItem_{table_fragment}_{key_hash[:4]}"

        existing_table = base.metadata.tables.get(table_name)
        if existing_table is not None:
            base.metadata.remove(existing_table)

        attrs: dict[str, Any] = {
            "__tablename__": table_name,
            "__table_args__": {"extend_existing": True, "sqlite_autoincrement": False},
            "id": Column(String(ULID_LENGTH), primary_key=True),
            "chat_id": Column(String(64), index=True, nullable=False),
            "message_id": Column(String(64), index=True, nullable=False),
            "model_id": Column(String(128), nullable=True),
            "item_type": Column(String(64), nullable=False),
            "payload": Column(JSON, nullable=False, default=dict),
            "is_encrypted": Column(Boolean, nullable=False, default=False),
            "created_at": Column(DateTime, nullable=False, default=datetime.datetime.utcnow),
        }

        item_model = type(class_name, (base,), attrs)

        schema_name = item_model.__table__.schema
        table_exists = True
        try:
            table_exists = sa_inspect(engine).has_table(table_name, schema=schema_name)
        except SQLAlchemyError:
            table_exists = True
        except Exception:  # pragma: no cover - defensive; inspector uses plugins
            table_exists = True

        try:
            item_model.__table__.create(bind=engine, checkfirst=True)
        except Exception as exc:  # pragma: no cover - database-specific errors
            if self._maybe_heal_index_conflict(engine, item_model.__table__, exc):
                try:
                    item_model.__table__.create(bind=engine, checkfirst=True)
                except Exception as retry_exc:
                    self.logger.warning("Artifact persistence disabled (table init failed after index cleanup): %s", retry_exc)
                    self._engine = None
                    self._session_factory = None
                    self._item_model = None
                    self._artifact_table_name = None
                    self._artifact_store_signature = None
                    return
            else:
                self.logger.warning("Artifact persistence disabled (table init failed): %s", exc)
                self._engine = None
                self._session_factory = None
                self._item_model = None
                self._artifact_table_name = None
                self._artifact_store_signature = None
                return

        self._engine = engine
        self._session_factory = session_factory
        self._item_model = item_model
        self._artifact_table_name = table_name
        self._artifact_store_signature = (table_fragment, self._encryption_key)
        if not table_exists:
            self.logger.info("Artifact table ready: %s (key hash: %s). Changing ARTIFACT_ENCRYPTION_KEY creates a new table; old artifacts become inaccessible.", table_name, key_hash[:8])
        if self._db_executor is None:
            self._db_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="responses-db")
        self.logger.debug("Artifact table ready: %s", table_name)

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        """Return a double-quoted identifier safe for direct SQL execution."""
        value = (identifier or "").replace('"', '""')
        return f'"{value}"'

    def _maybe_heal_index_conflict(
        self,
        engine: Engine | None,
        table: Any | None,
        exc: Exception,
    ) -> bool:
        """Attempt to drop orphaned indexes when table creation hits duplicates."""
        if not engine or table is None:
            return False

        root_exc = getattr(exc, "orig", exc)
        message = str(root_exc) or str(exc) or ""
        lowered = message.lower()
        if "ix_" not in lowered:
            return False

        raw_index_objects = [
            idx for idx in getattr(table, "indexes", set()) if getattr(idx, "name", None)
        ]
        names_from_metadata = {
            (idx.name or "").strip()
            for idx in raw_index_objects
            if (idx.name or "").strip()
        }
        names_from_error = {
            name.lower()
            for name in re.findall(r"ix_[0-9a-z_]+", message, flags=re.IGNORECASE)
        }
        names_from_columns = {
            f"ix_{table.name}_{column.name}"
            for column in getattr(table, "columns", [])
            if getattr(column, "index", False)
        }
        normalized_map = {name.lower(): name for name in names_from_metadata}
        for column_name in names_from_columns:
            normalized_map.setdefault(column_name.lower(), column_name)

        names_to_drop: dict[str, str] = {}
        for lowered, original in normalized_map.items():
            names_to_drop[lowered] = original
        for lowered in names_from_error:
            if lowered not in names_to_drop:
                names_to_drop[lowered] = lowered

        if not names_to_drop:
            return False

        dropped: list[str] = []
        failed_any = False
        for original_name in names_to_drop.values():
            if not original_name:
                continue
            qualified = self._quote_identifier(original_name)
            schema = getattr(table, "schema", None)
            if schema:
                qualified = f"{self._quote_identifier(schema)}.{qualified}"
            drop_sql = text(f"DROP INDEX IF EXISTS {qualified}")
            try:
                with engine.begin() as connection:
                    connection.execute(drop_sql)
                dropped.append(original_name)
            except SQLAlchemyError as raw_exc:
                failed_any = True
                self.logger.warning("Failed to drop index %s while healing %s: %s", original_name, getattr(table, "name", "?"), raw_exc)

        if dropped:
            self.logger.info("Dropped orphaned index(es) %s before recreating %s.", ", ".join(dropped), getattr(table, "name", "?"))
            return True

        if failed_any:
            return False

        # Targets were found but none were dropped (e.g., not mentioned and already gone).
        return False

    def shutdown(self) -> None:
        """Public method to shut down background resources."""
        executor = self._db_executor
        self._db_executor = None
        if executor:
            executor.shutdown(wait=True)
        self._stop_session_log_workers()

    def _write_session_log_archive(self, job: _SessionLogArchiveJob) -> None:
        """Write a single encrypted zip archive containing session logs + metadata."""
        if pyzipper is None:
            return
        base_dir = (job.base_dir or "").strip()
        if not base_dir:
            return

        user_id = _sanitize_path_component(job.user_id, fallback="user")
        chat_id = _sanitize_path_component(job.chat_id, fallback="chat")
        message_id = _sanitize_path_component(job.message_id, fallback="message")
        session_id = str(job.session_id or "")

        root = Path(base_dir).expanduser()
        out_dir = root / user_id / chat_id
        out_path = out_dir / f"{message_id}.zip"
        tmp_path = out_dir / f"{message_id}.zip.tmp"

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        compression_map = {
            "stored": pyzipper.ZIP_STORED,
            "deflated": pyzipper.ZIP_DEFLATED,
            "bzip2": pyzipper.ZIP_BZIP2,
            "lzma": pyzipper.ZIP_LZMA,
        }
        compression = compression_map.get((job.zip_compression or "lzma").lower(), pyzipper.ZIP_LZMA)

        meta = {
            "created_at": datetime.datetime.fromtimestamp(job.created_at, tz=datetime.timezone.utc).isoformat(),
            "ids": {
                "user_id": str(job.user_id or ""),
                "session_id": str(session_id),
                "chat_id": str(job.chat_id or ""),
                "message_id": str(job.message_id or ""),
            },
            "request_id": str(job.request_id or ""),
            "log_format": "text",
        }
        meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
        logs_payload = "\n".join([line.rstrip("\n") for line in (job.log_lines or [])])
        if logs_payload and not logs_payload.endswith("\n"):
            logs_payload += "\n"

        zip_kwargs: dict[str, Any] = {
            "mode": "w",
            "compression": compression,
            "encryption": pyzipper.WZ_AES,
        }
        if job.zip_compresslevel is not None and compression in {pyzipper.ZIP_DEFLATED, pyzipper.ZIP_BZIP2}:
            zip_kwargs["compresslevel"] = int(job.zip_compresslevel)

        try:
            with pyzipper.AESZipFile(tmp_path, **zip_kwargs) as zf:
                zf.setpassword(job.zip_password or b"")
                zf.writestr("meta.json", meta_json)
                zf.writestr("logs.txt", logs_payload)
        except Exception:
            with contextlib.suppress(Exception):
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            return

        try:
            os.replace(tmp_path, out_path)
        except Exception:
            with contextlib.suppress(Exception):
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]

    def _cleanup_session_log_archives(self) -> None:
        """Delete expired session log archives and prune empty directories."""
        with self._session_log_lock:
            dirs = set(self._session_log_dirs)
            retention_days = self._session_log_retention_days
        if not dirs:
            return
        cutoff = time.time() - retention_days * 86400

        for base_dir in dirs:
            base_dir = (base_dir or "").strip()
            if not base_dir:
                continue
            root = Path(base_dir).expanduser()
            if not root.exists():
                continue
            try:
                for path in root.rglob("*.zip"):
                    with contextlib.suppress(Exception):
                        stat = path.stat()
                        if stat.st_mtime < cutoff:
                            path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                continue

            # Prune empty directories, including the root if it's emptied out.
            try:
                for dirpath, dirnames, filenames in os.walk(root, topdown=False):
                    if dirnames or filenames:
                        continue
                    with contextlib.suppress(Exception):
                        os.rmdir(dirpath)
            except Exception:
                continue

    def _get_fernet(self) -> Fernet | None:
        """Return (and cache) the Fernet helper derived from the encryption key."""
        if not self._encryption_key:
            return None
        if self._fernet is None:
            digest = hashlib.sha256(self._encryption_key.encode("utf-8")).digest()
            key = base64.urlsafe_b64encode(digest)
            self._fernet = Fernet(key)
        return self._fernet

    def _should_encrypt(self, item_type: str) -> bool:
        """Determine whether a payload of ``item_type`` must be encrypted."""
        if not self._encryption_key:
            return False
        if self._encrypt_all:
            return True
        return (item_type or "").lower() == "reasoning"

    def _serialize_payload_bytes(self, payload: dict[str, Any]) -> bytes:
        """Return compact JSON bytes for ``payload``."""
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def _maybe_compress_payload(self, serialized: bytes) -> tuple[bytes, bool]:
        """Compress serialized bytes when LZ4 is available and thresholds are met."""
        if not serialized:
            return serialized, False
        if not self._compression_enabled:
            return serialized, False
        if self._compression_min_bytes and len(serialized) < self._compression_min_bytes:
            return serialized, False
        if lz4frame is None:
            return serialized, False
        try:
            compressed = lz4frame.compress(serialized)
        except Exception as exc:  # pragma: no cover - depends on native lib
            self.logger.warning("LZ4 compression failed; disabling compression for the remainder of this process: %s", exc, exc_info=self.logger.isEnabledFor(logging.DEBUG))
            self._compression_enabled = False
            return serialized, False
        if not compressed or len(compressed) >= len(serialized):
            return serialized, False
        return compressed, True

    def _encode_payload_bytes(self, payload: dict[str, Any]) -> bytes:
        """Serialize payload bytes and prepend a compression flag header."""
        serialized = self._serialize_payload_bytes(payload)
        data, compressed = self._maybe_compress_payload(serialized)
        flag = _PAYLOAD_FLAG_LZ4 if compressed else _PAYLOAD_FLAG_PLAIN
        return bytes([flag]) + data

    def _decode_payload_bytes(self, payload_bytes: bytes) -> dict[str, Any]:
        """Decode stored payload bytes into dictionaries."""
        if not payload_bytes:
            return {}
        if len(payload_bytes) <= _PAYLOAD_HEADER_SIZE:
            body = payload_bytes
        else:
            flag = payload_bytes[0]
            body = payload_bytes[_PAYLOAD_HEADER_SIZE:]
            if flag == _PAYLOAD_FLAG_LZ4:
                body = self._lz4_decompress(body)
            elif flag != _PAYLOAD_FLAG_PLAIN:
                raise ValueError(f"Invalid artifact payload flag: {flag}")
        try:
            return json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Unable to decode persisted artifact payload.") from exc

    def _lz4_decompress(self, data: bytes) -> bytes:
        """Decompress LZ4 payloads or raise descriptive errors."""
        if not data:
            return b""
        if lz4frame is None:
            raise RuntimeError(
                "Encountered compressed artifact, but the 'lz4' package is unavailable."
            )
        try:
            return lz4frame.decompress(data)
        except Exception as exc:  # pragma: no cover - depends on native lib
            raise ValueError("Failed to decompress persisted artifact payload.") from exc

    def _encrypt_payload(self, payload: dict[str, Any]) -> str:
        """Encrypt payload bytes using the configured Fernet helper."""
        fernet = self._get_fernet()
        if not fernet:
            raise RuntimeError("Encryption requested but ARTIFACT_ENCRYPTION_KEY is not configured.")
        encoded = self._encode_payload_bytes(payload)
        return fernet.encrypt(encoded).decode("utf-8")

    def _decrypt_payload(self, ciphertext: str) -> dict[str, Any]:
        """Decrypt ciphertext previously produced by :meth:`_encrypt_payload`."""
        fernet = self._get_fernet()
        if not fernet:
            raise RuntimeError("Decryption requested but ARTIFACT_ENCRYPTION_KEY is not configured.")
        try:
            plaintext = fernet.decrypt(ciphertext.encode("utf-8"))
        except InvalidToken as exc:
            raise ValueError("Unable to decrypt payload (invalid token).") from exc
        return self._decode_payload_bytes(plaintext)

    def _encrypt_if_needed(self, item_type: str, payload: dict[str, Any]) -> tuple[Any, bool]:
        """Optionally encrypt ``payload`` depending on the item type."""
        if not self._should_encrypt(item_type):
            return payload, False
        encrypted = self._encrypt_payload(payload)
        return {"ciphertext": encrypted, "enc_v": _ENCRYPTED_PAYLOAD_VERSION}, True

    def _prepare_rows_for_storage(self, rows: Iterable[dict[str, Any]]) -> None:
        """Normalize row payloads so Redis/DB always receive the stored schema."""
        if not rows:
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            payload = row.get("payload")
            if (
                row.get("is_encrypted")
                and isinstance(payload, dict)
                and "ciphertext" in payload
            ):
                payload.setdefault("enc_v", _ENCRYPTED_PAYLOAD_VERSION)
                continue
            if not isinstance(payload, dict):
                continue
            stored_payload, is_encrypted = self._encrypt_if_needed(row.get("item_type", ""), payload)
            row["payload"] = stored_payload
            row["is_encrypted"] = is_encrypted

    def _make_db_row(
        self,
        chat_id: Optional[str],
        message_id: Optional[str],
        model_id: str,
        payload: Dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Construct a persistence-ready row dict or return ``None`` when invalid."""
        if not (chat_id and self._item_model):
            return None
        if not message_id:
            self.logger.warning("Skipping artifact persistence for chat_id=%s: missing message_id.", chat_id)
            return None
        if not isinstance(payload, dict):
            return None
        item_type = payload.get("type", "unknown")
        return {
            "chat_id": chat_id,
            "message_id": message_id,
            "model_id": model_id,
            "item_type": item_type,
            "payload": payload,
        }

    def _db_persist_sync(self, rows: list[dict[str, Any]]) -> list[str]:
        """Persist prepared rows once; intentionally no automatic retry logic."""
        if not rows or not self._item_model or not self._session_factory:
            return []

        cleanup_rows = False
        try:
            ulids: list[str] = []
            for row in rows:
                if not row.get("_persisted"):
                    continue
                identifier = row.get("id")
                if isinstance(identifier, str) and identifier:
                    ulids.append(identifier)

            batch_size = self.valves.DB_BATCH_SIZE
            pending_rows = [row for row in rows if not row.get("_persisted")]
            if not pending_rows:
                if ulids:
                    self.logger.debug(
                        "Persisted %d response artifact(s) to %s.",
                        len(ulids),
                        self._artifact_table_name,
                    )
                cleanup_rows = True
                return ulids

            for start in range(0, len(pending_rows), batch_size):
                chunk = pending_rows[start : start + batch_size]
                now = datetime.datetime.utcnow()
                instances = []
                chunk_ulids: list[str] = []
                persisted_rows: list[dict[str, Any]] = []
                for row in chunk:
                    payload = row.get("payload")
                    if payload is None:
                        self.logger.warning(
                            "Skipping artifact persist for chat_id=%s message_id=%s: payload missing or invalid.",
                            row.get("chat_id"),
                            row.get("message_id"),
                        )
                        continue
                    ulid = row.get("id") or generate_item_id()
                    stored_payload = payload
                    is_encrypted = bool(row.get("is_encrypted"))
                    needs_encryption = (
                        not is_encrypted
                        or not isinstance(stored_payload, dict)
                        or "ciphertext" not in stored_payload
                    )
                    if needs_encryption:
                        raw_payload = payload if isinstance(payload, dict) else {}
                        stored_payload, is_encrypted = self._encrypt_if_needed(row.get("item_type", ""), raw_payload)
                    instances.append(
                        self._item_model(  # type: ignore[call-arg]
                            id=ulid,
                            chat_id=row.get("chat_id"),
                            message_id=row.get("message_id"),
                            model_id=row.get("model_id"),
                            item_type=row.get("item_type"),
                            payload=stored_payload,
                            is_encrypted=is_encrypted,
                            created_at=now,
                        )
                    )
                    chunk_ulids.append(ulid)
                    persisted_rows.append(row)

                if not instances:
                    continue

                session: Session = self._session_factory()  # type: ignore[call-arg]
                try:
                    session.add_all(instances)
                    session.commit()
                except SQLAlchemyError as exc:  # pragma: no cover
                    session.rollback()
                    self.logger.error(
                        "Failed to persist response artifacts: %s",
                        exc,
                        exc_info=self.logger.isEnabledFor(logging.DEBUG),
                    )
                    raise
                finally:
                    session.close()

                for row in persisted_rows:
                    row["_persisted"] = True
                ulids.extend(chunk_ulids)

            if ulids:
                self.logger.debug(
                    "Persisted %d response artifact(s) to %s.",
                    len(ulids),
                    self._artifact_table_name,
                )
            cleanup_rows = True
            return ulids
        finally:
            if cleanup_rows:
                for row in rows:
                    row.pop("_persisted", None)

    async def _db_persist(self, rows: list[dict[str, Any]]) -> list[str]:
        """Persist artifacts, optionally via Redis write-behind."""
        if not rows:
            return []

        user_id = SessionLogger.user_id.get() or ""
        if not self._db_breaker_allows(user_id):
            self.logger.warning("DB writes disabled for user_id=%s due to repeated failures", user_id)
            context = self._TOOL_CONTEXT.get()
            await self._emit_notification(
                context.event_emitter if context else None,
                "DB ops skipped due to repeated errors.",
                level="warning",
            )
            self._record_failure(user_id)
            return []

        for row in rows:
            row.setdefault("id", generate_item_id())

        self._prepare_rows_for_storage(rows)

        try:
            if self._redis_enabled:
                return await self._redis_enqueue_rows(rows)
            return await self._db_persist_direct(rows, user_id=user_id)
        except Exception as exc:
            self._record_db_failure(user_id)
            self.logger.warning("Artifact persist failed: %s", exc)
            return []

    async def _db_persist_direct(self, rows: list[dict[str, Any]], user_id: str = "") -> list[str]:
        if not rows or not self._db_executor or not self._item_model or not self._session_factory:
            return []

        retryer = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        loop = asyncio.get_running_loop()
        async for attempt in retryer:
            with attempt:
                try:
                    ulids = await loop.run_in_executor(
                        self._db_executor, self._db_persist_sync, rows
                    )
                except Exception as exc:
                    if self._is_duplicate_key_error(exc):
                        self.logger.debug("Duplicate key detected during DB persist; assuming prior flush succeeded")
                        return [
                            identifier
                            for row in rows
                            for identifier in [row.get("id")]
                            if isinstance(identifier, str) and identifier
                        ]
                    raise
                if self._redis_enabled:
                    await self._redis_cache_rows(rows)
                self._reset_db_failure(user_id)
                return ulids
        return []

    def _is_duplicate_key_error(self, exc: Exception) -> bool:
        if isinstance(exc, SQLAlchemyError):
            messages = [str(exc)]
            orig = getattr(exc, "orig", None)
            if orig:
                messages.append(str(orig))
            lowered = " ".join(messages).lower()
            keywords = ("duplicate key", "unique constraint", "already exists")
            return any(keyword in lowered for keyword in keywords)
        return False

    def _db_fetch_sync(
        self,
        chat_id: str,
        message_id: Optional[str],
        item_ids: list[str],
    ) -> dict[str, dict]:
        """Synchronously fetch persisted artifacts for ``chat_id``."""
        if not item_ids or not self._item_model or not self._session_factory:
            return {}
        model = self._item_model
        session: Session = self._session_factory()  # type: ignore[call-arg]
        try:
            query = session.query(model).filter(model.chat_id == chat_id)
            if item_ids:
                query = query.filter(model.id.in_(item_ids))
            if message_id:
                query = query.filter(model.message_id == message_id)
            rows = query.all()
        finally:
            session.close()

        results: dict[str, dict] = {}
        for row in rows:
            payload = row.payload
            if row.is_encrypted:
                ciphertext = ""
                if isinstance(payload, dict):
                    ciphertext = payload.get("ciphertext", "")
                elif isinstance(payload, str):
                    ciphertext = payload
                try:
                    payload = self._decrypt_payload(ciphertext or "")
                except Exception as exc:
                    self.logger.warning("Failed to decrypt artifact %s: %s", row.id, exc, exc_info=self.logger.isEnabledFor(logging.DEBUG))
                    continue
            if isinstance(payload, dict):
                results[row.id] = payload
        return results

    async def _db_fetch(
        self,
        chat_id: Optional[str],
        message_id: Optional[str],
        item_ids: list[str],
    ) -> dict[str, dict]:
        """Fetch artifacts with Redis cache + retries."""
        if not (chat_id and item_ids):
            return {}

        cached: dict[str, dict] = {}
        if self._redis_enabled:
            cached = await self._redis_fetch_rows(chat_id, item_ids)
            missing_ids = [item_id for item_id in item_ids if item_id not in cached]
        else:
            missing_ids = item_ids

        if not missing_ids:
            return cached

        if not (self._db_executor and self._item_model and self._session_factory):
            return cached

        user_id = SessionLogger.user_id.get() or ""
        if not self._db_breaker_allows(user_id):
            self.logger.warning("DB reads disabled for user_id=%s due to repeated failures", user_id)
            context = self._TOOL_CONTEXT.get()
            await self._emit_notification(
                context.event_emitter if context else None,
                "DB ops skipped due to repeated errors.",
                level="warning",
            )
            self._record_failure(user_id)
            return {}

        try:
            fetched = await self._db_fetch_direct(chat_id, message_id, missing_ids)
            if fetched and self._redis_enabled:
                cache_rows = [
                    {
                        "id": item_id,
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "item_type": (payload or {}).get("type", "unknown") if isinstance(payload, dict) else "unknown",
                        "payload": payload,
                    }
                    for item_id, payload in fetched.items()
                ]
                self._prepare_rows_for_storage(cache_rows)
                await self._redis_cache_rows(cache_rows, chat_id=chat_id)
            if user_id:
                self._reset_db_failure(user_id)
            cached.update(fetched)
        except Exception as exc:
            self._record_db_failure(user_id)
            self.logger.warning("Artifact fetch failed: %s", exc, exc_info=self.logger.isEnabledFor(logging.DEBUG))
        return cached

    async def _db_fetch_direct(
        self,
        chat_id: str,
        message_id: Optional[str],
        item_ids: list[str],
    ) -> dict[str, dict]:
        retryer = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        loop = asyncio.get_running_loop()
        async for attempt in retryer:
            with attempt:
                fetch_call = functools.partial(self._db_fetch_sync, chat_id, message_id, item_ids)
                return await loop.run_in_executor(self._db_executor, fetch_call)
        return {}

    def _delete_artifacts_sync(self, artifact_ids: list[str]) -> None:
        """Synchronously delete artifacts by ULID."""
        if not (artifact_ids and self._session_factory and self._item_model):
            return
        session: Session = self._session_factory()  # type: ignore[call-arg]
        try:
            (
                session.query(self._item_model)
                .filter(self._item_model.id.in_(artifact_ids))
                .delete(synchronize_session=False)
            )
            session.commit()
        except SQLAlchemyError:
            # Rollback on any database error, then re-raise
            session.rollback()
            raise
        finally:
            session.close()

    async def _delete_artifacts(self, refs: list[Tuple[str, str]]) -> None:
        """Delete persisted artifacts (and cached copies) once they have been replayed."""
        if not refs:
            return
        ids = sorted({artifact_id for _, artifact_id in refs if artifact_id})
        if not ids or not self._db_executor:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._db_executor, functools.partial(self._delete_artifacts_sync, ids))
        if self._redis_enabled and self._redis_client:
            keys = [self._redis_cache_key(chat_id, artifact_id) for chat_id, artifact_id in refs]
            keys = [key for key in keys if key]
            if keys:
                await _wait_for(self._redis_client.delete(*keys))

    # ─────────────────────────────────────────────────────────────────────────────
    # File and Image Handling Helpers
    # ─────────────────────────────────────────────────────────────────────────────

    async def _get_user_by_id(self, user_id: str):
        """Fetch user record from database for file upload operations.

        Args:
            user_id: The unique identifier for the user

        Returns:
            UserModel object if found, None otherwise

        Note:
            Uses run_in_threadpool to avoid blocking async operations.
            Failures are logged but do not raise exceptions.
        """
        try:
            return await run_in_threadpool(Users.get_user_by_id, user_id)
        except Exception as exc:
            self.logger.error(f"Failed to load user {user_id}: {exc}")
            return None

    async def _get_file_by_id(self, file_id: str):
        """Look up file metadata from Open WebUI's file storage.

        Args:
            file_id: The unique identifier for the file

        Returns:
            FileModel object if found, None otherwise

        Note:
            Uses run_in_threadpool to avoid blocking async operations.
            Failures are logged but do not raise exceptions.
        """
        try:
            return await run_in_threadpool(Files.get_file_by_id, file_id)
        except Exception as exc:
            self.logger.error(f"Failed to load file {file_id}: {exc}")
            return None

    def _infer_file_mime_type(self, file_obj: Any) -> str:
        """Return the best-known MIME type for a stored Open WebUI file."""
        candidates = [
            getattr(file_obj, "mime_type", None),
            getattr(file_obj, "content_type", None),
        ]
        meta = getattr(file_obj, "meta", None) or {}
        if isinstance(meta, dict):
            candidates.extend(
                [
                    meta.get("content_type"),
                    meta.get("mimeType"),
                    meta.get("mime_type"),
                ]
            )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                normalized = candidate.strip().lower()
                if normalized == "image/jpg":
                    return "image/jpeg"
                return normalized
        return "application/octet-stream"

    async def _inline_internal_file_url(
        self,
        url: str,
        *,
        chunk_size: int,
        max_bytes: int,
    ) -> Optional[str]:
        """Convert an Open WebUI file URL into a data URL for providers."""
        file_id = _extract_internal_file_id(url)
        if not file_id:
            return None
        file_obj = await self._get_file_by_id(file_id)
        if not file_obj:
            return None
        mime_type = self._infer_file_mime_type(file_obj)
        try:
            b64 = await self._read_file_record_base64(file_obj, chunk_size, max_bytes)
        except ValueError as exc:
            self.logger.warning("Failed to inline file %s: %s", file_id, exc)
            return None
        if not b64:
            return None
        return f"data:{mime_type};base64,{b64}"

    async def _read_file_record_base64(
        self,
        file_obj: Any,
        chunk_size: int,
        max_bytes: int,
    ) -> Optional[str]:
        """Return a base64 string for a stored Open WebUI file."""
        if max_bytes <= 0:
            raise ValueError("BASE64_MAX_SIZE_MB must be greater than zero")

        def _from_bytes(raw: bytes) -> str:
            if len(raw) > max_bytes:
                raise ValueError("File exceeds BASE64_MAX_SIZE_MB limit")
            return base64.b64encode(raw).decode("ascii")

        data_field = getattr(file_obj, "data", None)
        if isinstance(data_field, dict):
            for key in ("b64", "base64", "data"):
                inline_value = data_field.get(key)
                if isinstance(inline_value, str) and inline_value.strip():
                    if not self._validate_base64_size(inline_value):
                        raise ValueError("Stored base64 payload exceeds configured limit")
                    return inline_value.strip()
            blob_value = data_field.get("bytes")
            if isinstance(blob_value, (bytes, bytearray)):
                return _from_bytes(bytes(blob_value))

        prefer_paths = [
            getattr(file_obj, attr, None)
            for attr in (
                "path",
                "file_path",
                "absolute_path",
            )
        ]
        for candidate in prefer_paths:
            if not isinstance(candidate, str):
                continue
            path = Path(candidate)
            if not path.exists():
                continue
            return await self._encode_file_path_base64(path, chunk_size, max_bytes)

        raw_bytes = None
        for attr in ("content", "blob", "data"):
            value = getattr(file_obj, attr, None)
            if isinstance(value, (bytes, bytearray)):
                raw_bytes = bytes(value)
                break
        if raw_bytes is not None:
            return _from_bytes(raw_bytes)
        return None

    async def _encode_file_path_base64(
        self,
        path: Path,
        chunk_size: int,
        max_bytes: int,
    ) -> str:
        """Read ``path`` in chunks and return a base64 string."""
        chunk_size = max(64 * 1024, min(chunk_size, max_bytes))

        def _encode_stream() -> str:
            total = 0
            buffer = io.StringIO()
            leftover = b""
            with path.open("rb") as source:
                while True:
                    chunk = source.read(chunk_size)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("File exceeds BASE64_MAX_SIZE_MB limit")
                    chunk = leftover + chunk
                    whole_bytes = (len(chunk) // 3) * 3
                    if whole_bytes:
                        buffer.write(base64.b64encode(chunk[:whole_bytes]).decode("ascii"))
                    leftover = chunk[whole_bytes:]
            if leftover:
                buffer.write(base64.b64encode(leftover).decode("ascii"))
            return buffer.getvalue()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _encode_stream)

    async def _upload_to_owui_storage(
        self,
        request: Request,
        user,
        file_data: bytes,
        filename: str,
        mime_type: str
    ) -> Optional[str]:
        """Upload file or image to Open WebUI storage and return internal URL.

        This method ensures that files and images are persistently stored in Open WebUI's
        file system, preventing data loss from:
        - Remote URLs that may become inaccessible
        - Base64 data that is temporary
        - External image hosts that delete images after a period

        Args:
            request: FastAPI Request object for URL generation
            user: UserModel object representing the file owner
            file_data: Raw bytes of the file content
            filename: Desired filename (will be prefixed with UUID)
            mime_type: MIME type of the file (e.g., 'image/jpeg', 'application/pdf')

        Returns:
            Internal URL path to the uploaded file (e.g., '/api/v1/files/{id}'),
            or None if upload fails

        Note:
            - File processing is disabled (process=False) to avoid unnecessary overhead
            - Uses run_in_threadpool to prevent blocking async event loop
            - Failures are logged but return None rather than raising exceptions

        Example:
            >>> url = await self._upload_to_owui_storage(
            ...     request, user, image_bytes, "photo.jpg", "image/jpeg"
            ... )
            >>> # url = '/api/v1/files/abc123...'
        """
        try:
            file_item = await run_in_threadpool(
                upload_file_handler,
                request=request,
                file=UploadFile(
                    file=io.BytesIO(file_data),
                    filename=filename,
                    headers=Headers({"content-type": mime_type}),
                ),
                metadata={"mime_type": mime_type},
                process=False,  # Disable processing to avoid overhead
                process_in_background=False,
                user=user,
                background_tasks=BackgroundTasks(),
            )
            # Generate internal URL path
            file_id: Optional[str] = None
            if hasattr(file_item, "id"):
                candidate = getattr(file_item, "id", None)
                if isinstance(candidate, str) and candidate.strip():
                    file_id = candidate.strip()
            elif isinstance(file_item, dict):
                raw_id = file_item.get("id")
                if isinstance(raw_id, str) and raw_id.strip():
                    file_id = raw_id.strip()
            if not file_id:
                self.logger.error("Upload handler returned an object without an id; aborting OWUI storage write.")
                return None
            internal_url = request.app.url_path_for("get_file_content_by_id", id=file_id)
            self.logger.info(
                f"Uploaded {filename} ({len(file_data):,} bytes) to OWUI storage: {internal_url}"
            )
            return internal_url
        except Exception as exc:
            self.logger.error(f"Failed to upload {filename} to OWUI storage: {exc}")
            return None

    async def _resolve_storage_context(
        self,
        request: Optional[Request],
        user_obj: Optional[Any],
    ) -> tuple[Optional[Request], Optional[Any]]:
        """Return a `(request, user)` tuple suitable for OWUI uploads."""
        if request is None:
            if user_obj:
                self.logger.debug("Storage upload skipped: request context missing.")
            return None, None
        if user_obj is not None:
            return request, user_obj

        fallback_user = await self._ensure_storage_user()
        if fallback_user is None:
            return None, None
        self.logger.debug("Using fallback storage user '%s' for upload.", fallback_user.email)
        return request, fallback_user

    async def _ensure_storage_user(self) -> Optional[Any]:
        """Ensure the fallback storage user exists (lazy creation)."""
        if self._storage_user_cache is not None:
            return self._storage_user_cache

        if self._storage_user_lock is None:
            self._storage_user_lock = asyncio.Lock()

        async with self._storage_user_lock:
            if self._storage_user_cache is not None:
                return self._storage_user_cache

            fallback_email = (self.valves.FALLBACK_STORAGE_EMAIL or "").strip() or "openrouter-pipe@system.local"
            fallback_name = (self.valves.FALLBACK_STORAGE_NAME or "").strip() or "OpenRouter Pipe Storage"
            fallback_role = (self.valves.FALLBACK_STORAGE_ROLE or "").strip() or "pending"

            if (
                fallback_role.lower() in {"admin", "system", "owner"}
                and not self._storage_role_warning_emitted
            ):
                self.logger.warning(
                    "Fallback storage role '%s' is highly privileged. Configure FALLBACK_STORAGE_ROLE to a least-privilege service role if possible.",
                    fallback_role,
                )
                self._storage_role_warning_emitted = True

            try:
                fallback_user = await run_in_threadpool(
                    Users.get_user_by_email,
                    fallback_email,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                self.logger.error("Failed to load fallback storage user: %s", exc)
                return None

            if fallback_user is None:
                user_id = f"openrouter-pipe-{uuid.uuid4().hex}"
                try:
                    oauth_marker = f"openrouter-pipe-storage:{uuid.uuid4().hex}"
                    insert_fn = Users.insert_new_user
                    insert_kwargs: dict[str, Any] = {}
                    try:
                        if self._user_insert_param_names is None:
                            sig = inspect.signature(insert_fn)
                            self._user_insert_param_names = tuple(sig.parameters.keys())
                    except (TypeError, ValueError):
                        self._user_insert_param_names = ()

                    param_names = self._user_insert_param_names or ()
                    if "oauth" in param_names:
                        insert_kwargs["oauth"] = {"sub": oauth_marker}
                    elif "oauth_sub" in param_names:
                        insert_kwargs["oauth_sub"] = oauth_marker

                    fallback_user = await run_in_threadpool(
                        insert_fn,
                        user_id,
                        fallback_name,
                        fallback_email,
                        "/user.png",
                        fallback_role or "pending",
                        **insert_kwargs,
                    )
                    self.logger.info(
                        "Created fallback storage user '%s' (%s) for multimodal uploads.",
                        fallback_name,
                        fallback_email,
                    )
                except Exception as exc:  # pragma: no cover - defensive guard
                    self.logger.error("Failed to create fallback storage user: %s", exc)
                    return None

            self._storage_user_cache = fallback_user
            return fallback_user

    async def _is_safe_url(self, url: str) -> bool:
        """Async wrapper to validate URLs without blocking the event loop."""
        if not self.valves.ENABLE_SSRF_PROTECTION:
            return True
        return await asyncio.to_thread(self._is_safe_url_blocking, url)

    def _is_safe_url_blocking(self, url: str) -> bool:
        """Blocking implementation of the SSRF guard (runs in a thread)."""
        try:
            import ipaddress
            import socket
            from ipaddress import IPv4Address, IPv6Address

            parsed = urlparse(url)
            host = parsed.hostname

            if not host:
                self.logger.warning(f"URL has no hostname: {url}")
                return False

            ip_objects: list[IPv4Address | IPv6Address] = []
            seen_ips: set[str] = set()

            def _record_ip(candidate: IPv4Address | IPv6Address) -> None:
                comp = candidate.compressed
                if comp not in seen_ips:
                    seen_ips.add(comp)
                    ip_objects.append(candidate)

            # Fast-path literal IPv4/IPv6 hosts
            try:
                literal_ip = ipaddress.ip_address(host)
            except ValueError:
                literal_ip = None
            else:
                _record_ip(literal_ip)

            # Resolve hostname to all available IPs (IPv4 + IPv6) when not a literal
            if literal_ip is None:
                try:
                    addrinfo = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
                except (socket.gaierror, UnicodeError):
                    self.logger.warning(f"DNS resolution failed for: {host}")
                    return False
                except Exception as exc:  # pragma: no cover - defensive guard
                    self.logger.error(f"Unexpected DNS error for {host}: {exc}")
                    return False

                for _, _, _, _, sockaddr in addrinfo:
                    if not sockaddr:
                        continue
                    ip_str = sockaddr[0]
                    try:
                        resolved_ip = ipaddress.ip_address(ip_str)
                    except ValueError:
                        self.logger.warning(f"Invalid IP address format: {ip_str}")
                        return False
                    _record_ip(resolved_ip)

            if not ip_objects:
                self.logger.warning(f"No IP addresses resolved for: {host}")
                return False

            for ip in ip_objects:
                if ip.is_private:
                    reason = "private"
                elif ip.is_loopback:
                    reason = "loopback"
                elif ip.is_link_local:
                    reason = "link-local"
                elif ip.is_multicast:
                    reason = "multicast"
                elif ip.is_reserved:
                    reason = "reserved"
                elif ip.is_unspecified:
                    reason = "unspecified"
                else:
                    continue

                self.logger.warning(f"Blocked SSRF attempt to {reason} IP: {url} ({ip})")
                return False

            return True

        except Exception as exc:
            # Defensive: treat validation errors as unsafe
            self.logger.error(f"URL safety validation failed for {url}: {exc}")
            return False

    def _is_youtube_url(self, url: Optional[str]) -> bool:
        """Check if URL is a valid YouTube video URL.

        Supports both standard and short YouTube URL formats:
            - https://www.youtube.com/watch?v=VIDEO_ID
            - https://youtu.be/VIDEO_ID
            - http://youtube.com/watch?v=VIDEO_ID (http variant)

        Args:
            url: URL to validate

        Returns:
            True if URL matches YouTube video pattern, False otherwise

        Note:
            - Does not validate that the video ID exists or is accessible
            - Only checks URL format, not video availability
            - Query parameters (like &t=30s) are allowed

        Example:
            >>> self._is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            True
            >>> self._is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
            True
            >>> self._is_youtube_url("https://vimeo.com/123456")
            False
        """
        if not url:
            return False

        patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
        ]

        return any(re.match(pattern, url, re.IGNORECASE) for pattern in patterns)


    async def _download_remote_url(
        self,
        url: str,
        timeout_seconds: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Download file or image from remote URL with exponential backoff retry logic.

        This method fetches content from HTTP/HTTPS URLs with automatic retry on transient
        failures using exponential backoff. Retry behavior is configurable via valves.

        Args:
            url: The HTTP or HTTPS URL to download
            timeout_seconds: Optional timeout in seconds per attempt (defaults to valve setting, max 60s)

        Returns:
            Dictionary containing:
                - 'data': Raw bytes of the downloaded content
                - 'mime_type': Normalized MIME type from Content-Type header
                - 'url': Original URL for reference
            Returns None if download fails or URL is invalid

        Retry Behavior (configurable via valves):
            - REMOTE_DOWNLOAD_MAX_RETRIES: Maximum retry attempts (default: 3)
            - REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS: Initial delay before first retry (default: 5s)
            - REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS: Maximum total retry time (default: 45s)
            - Uses exponential backoff: delay * 2^attempt

        Retryable Errors:
            - Network errors (connection timeout, DNS failure, etc.)
            - HTTP 5xx server errors
            - HTTP 429 rate limit errors

        Non-Retryable Errors:
            - HTTP 4xx client errors (except 429)
            - Invalid URLs
            - Files exceeding configured size limit (REMOTE_FILE_MAX_SIZE_MB)

        Size Limits:
            - Maximum size configurable via REMOTE_FILE_MAX_SIZE_MB valve (default: 50MB)
            - When Open WebUI RAG uploads enforce FILE_MAX_SIZE, the limit auto-aligns (never exceeding 500MB)
            - Files exceeding the effective limit are rejected with a warning

        Supported Protocols:
            - http:// and https:// only
            - Other protocols return None

        MIME Type Normalization:
            - 'image/jpg' is normalized to 'image/jpeg'
            - MIME type extracted from Content-Type header (charset ignored)

        Note:
            - Timeout is capped at 60 seconds per attempt to prevent hanging requests
            - All exceptions are caught and logged, returning None
            - Empty or non-HTTP URLs return None immediately
            - Retry delays use exponential backoff to be respectful of remote servers

        Example:
            >>> result = await self._download_remote_url(
            ...     "https://example.com/image.jpg"
            ... )
            >>> if result:
            ...     print(f"Downloaded {len(result['data'])} bytes")
            ...     print(f"MIME type: {result['mime_type']}")
        """
        url = (url or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            return None

        # SSRF protection: Validate URL is not targeting private networks
        if not await self._is_safe_url(url):
            self.logger.error(f"SSRF protection blocked download from: {url}")
            return None

        # Get retry configuration from valves
        max_retries = self.valves.REMOTE_DOWNLOAD_MAX_RETRIES
        initial_delay = self.valves.REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS
        max_retry_time = self.valves.REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS

        # Cap timeout at 60 seconds per attempt
        if timeout_seconds is None:
            timeout_seconds = self.valves.HTTP_CONNECT_TIMEOUT_SECONDS
        timeout_seconds = min(timeout_seconds, 60)

        attempt = 0
        start_time = time.perf_counter()

        try:
            # Use tenacity for retry logic with exponential backoff
            async for attempt_info in AsyncRetrying(
                retry=retry_if_exception_type((_RetryableHTTPStatusError, httpx.NetworkError, httpx.TimeoutException)),
                stop=stop_after_attempt(max_retries + 1),  # +1 because first attempt doesn't count as retry
                wait=_RetryWait(wait_exponential(multiplier=initial_delay, min=initial_delay, max=max_retry_time)),
                reraise=True
            ):
                with attempt_info:
                    attempt += 1

                    # Check if we've exceeded max retry time
                    elapsed = time.perf_counter() - start_time
                    if attempt > 1 and elapsed > max_retry_time:
                        self.logger.warning(
                            f"Download retry timeout exceeded for {url} after {elapsed:.1f}s"
                        )
                        return None

                    # Log retry attempts
                    if attempt > 1:
                        self.logger.info(
                            f"Retry attempt {attempt - 1}/{max_retries} for {url} after {elapsed:.1f}s"
                        )

                    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                        async with client.stream("GET", url) as response:
                            try:
                                response.raise_for_status()
                            except httpx.HTTPStatusError as exc:
                                retryable, retry_after = _classify_retryable_http_error(exc)
                                if retryable:
                                    raise _RetryableHTTPStatusError(exc, retry_after=retry_after) from exc
                                raise

                            # Extract and normalize MIME type
                            mime_type = response.headers.get("content-type", "").split(";")[0].lower().strip()
                            if mime_type == "image/jpg":
                                mime_type = "image/jpeg"

                            # Enforce configurable size limit (valve + optional RAG cap)
                            effective_limit_mb = self._get_effective_remote_file_limit_mb()
                            max_size_bytes = effective_limit_mb * 1024 * 1024

                            content_length = response.headers.get("content-length")
                            if content_length:
                                try:
                                    if int(content_length) > max_size_bytes:
                                        self.logger.warning(
                                            "Remote file %s exceeds configured limit based on Content-Length header "
                                            "(%s bytes > %s bytes); aborting download.",
                                            url,
                                            content_length,
                                            max_size_bytes,
                                        )
                                        return None
                                except ValueError:
                                    pass

                            payload = bytearray()
                            async for chunk in response.aiter_bytes():
                                if not chunk:
                                    continue
                                payload.extend(chunk)
                                if len(payload) > max_size_bytes:
                                    size_mb = len(payload) / (1024 * 1024)
                                    self.logger.warning(
                                        f"Remote file {url} exceeds configured limit "
                                        f"({size_mb:.1f}MB > {effective_limit_mb}MB), aborting download."
                                    )
                                    return None

                    # Success
                    if attempt > 1:
                        elapsed = time.perf_counter() - start_time
                        self.logger.info(
                            f"Successfully downloaded {url} after {attempt} attempt(s) in {elapsed:.1f}s"
                        )

                    return {
                        "data": bytes(payload),
                        "mime_type": mime_type,
                        "url": url
                    }

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            self.logger.error(
                f"Failed to download {url} after {attempt} attempt(s) in {elapsed:.1f}s: {exc}"
            )
            return None

    def _get_effective_remote_file_limit_mb(self) -> int:
        """Return the active remote download limit, honoring RAG constraints."""
        base_limit_mb = self.valves.REMOTE_FILE_MAX_SIZE_MB
        rag_enabled, rag_limit_mb = _read_rag_file_constraints()
        if not rag_enabled or rag_limit_mb is None:
            return base_limit_mb

        # Never exceed Open WebUI's configured FILE_MAX_SIZE when RAG is active.
        if base_limit_mb > rag_limit_mb:
            return rag_limit_mb

        # If the valve is still using the default, upgrade to the RAG cap for consistency.
        if (
            base_limit_mb == _REMOTE_FILE_MAX_SIZE_DEFAULT_MB
            and rag_limit_mb > base_limit_mb
        ):
            return rag_limit_mb
        return base_limit_mb

    def _validate_base64_size(self, b64_data: str) -> bool:
        """Validate base64 data size is within configured limits.

        Estimates the decoded size of base64 data and compares it against the
        configured BASE64_MAX_SIZE_MB valve to prevent memory issues from huge payloads.

        Args:
            b64_data: Base64-encoded string to validate

        Returns:
            True if within limits, False if too large

        Note:
            Base64 encoding increases size by approximately 33% (4/3 ratio).
            This method estimates the original size before validation to avoid
            decoding potentially huge strings just to reject them.

        Example:
            >>> if self._validate_base64_size(huge_base64_string):
            ...     decoded = base64.b64decode(huge_base64_string)
            ... else:
            ...     # Reject without decoding
            ...     return None
        """
        if not b64_data:
            return True  # Empty string is valid

        # Base64 is ~1.33x the original size (4/3 ratio)
        # Estimate original size: (base64_length * 3) / 4
        estimated_size_bytes = (len(b64_data) * 3) / 4
        max_size_bytes = self.valves.BASE64_MAX_SIZE_MB * 1024 * 1024

        if estimated_size_bytes > max_size_bytes:
            estimated_size_mb = estimated_size_bytes / (1024 * 1024)
            self.logger.warning(
                f"Base64 data size (~{estimated_size_mb:.1f}MB) exceeds configured limit "
                f"({self.valves.BASE64_MAX_SIZE_MB}MB), rejecting to prevent memory issues"
            )
            return False

        return True

    def _parse_data_url(self, data_url: str) -> Optional[Dict[str, Any]]:
        """Extract base64 data from data URL.

        Parses data URLs in the format: data:<mime_type>;base64,<base64_data>

        Args:
            data_url: Data URL string to parse

        Returns:
            Dictionary containing:
                - 'data': Decoded bytes from base64
                - 'mime_type': Normalized MIME type
                - 'b64': Original base64 string (without prefix)
            Returns None if parsing fails or format is invalid

        Format Requirements:
            - Must start with 'data:'
            - Must contain ';base64,' separator
            - Base64 data must be valid
            - Size must not exceed BASE64_MAX_SIZE_MB valve (default: 50MB)

        MIME Type Normalization:
            - 'image/jpg' is normalized to 'image/jpeg'
            - MIME type extracted from prefix (e.g., 'data:image/png;base64,...')

        Size Validation:
            - Validates size before decoding to prevent memory issues
            - Uses BASE64_MAX_SIZE_MB valve for limit
            - Returns None if size exceeds limit

        Note:
            - Invalid base64 data results in None return
            - Oversized data results in None return
            - All exceptions are caught and logged
            - Non-data URLs return None immediately

        Example:
            >>> result = self._parse_data_url(
            ...     "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
            ... )
            >>> if result:
            ...     print(f"MIME: {result['mime_type']}")
            ...     print(f"Size: {len(result['data'])} bytes")
        """
        try:
            if not data_url or not data_url.startswith("data:"):
                return None

            parts = data_url.split(";base64,", 1)
            if len(parts) != 2:
                return None

            # Extract and normalize MIME type
            mime_type = parts[0].replace("data:", "", 1).lower().strip()
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"

            b64_data = parts[1]

            # Validate base64 size before decoding to prevent memory issues
            if not self._validate_base64_size(b64_data):
                return None  # Size validation failed, already logged

            file_data = base64.b64decode(b64_data)

            return {
                "data": file_data,
                "mime_type": mime_type,
                "b64": b64_data
            }
        except Exception as exc:
            self.logger.error(f"Failed to parse data URL: {exc}")
            return None

    async def _emit_status(
        self,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]],
        message: str,
        done: bool = False
    ):
        """Emit status updates to the Open WebUI client.

        Sends progress indicators to the UI during file/image processing operations.

        Args:
            event_emitter: Async callable for sending events to the client,
                          or None if no emitter available
            message: Status message to display (supports emoji for visual indicators)
            done: Whether this status represents completion (default: False)

        Status Message Conventions:
            - 📥 Download/upload in progress
            - ✅ Successful completion
            - ⚠️ Warning or non-critical error
            - 🔴 Critical error

        Note:
            - If event_emitter is None, this method is a no-op
            - Errors during emission are caught and logged
            - Does not interrupt processing flow

        Example:
            >>> await self._emit_status(
            ...     emitter,
            ...     "📥 Downloading remote image...",
            ...     done=False
            ... )
        """
        if event_emitter:
            try:
                await event_emitter({
                    "type": "status",
                    "data": {
                        "description": message,
                        "done": done
                    }
                })
            except Exception as exc:
                self.logger.error(f"Failed to emit status: {exc}")

    async def _maybe_dump_costs_snapshot(
        self,
        valves: "Pipe.Valves",
        *,
        user_id: str,
        model_id: Optional[str],
        usage: dict[str, Any] | None,
        user_obj: Optional[Any] = None,
        pipe_id: Optional[str] = None,
    ) -> None:
        """Push usage snapshots to Redis when enabled, namespaced per pipe."""
        if not valves.COSTS_REDIS_DUMP:
            return
        if not (self._redis_enabled and self._redis_client):
            return
        if not user_id:
            return

        def _user_field(obj: Any, field: str) -> Optional[str]:
            if obj is None:
                return None
            if isinstance(obj, dict):
                value = obj.get(field)
            else:
                value = getattr(obj, field, None)
            return str(value) if value is not None else None

        email = _user_field(user_obj, "email")
        name = _user_field(user_obj, "name")
        snapshot_usage = usage if isinstance(usage, dict) else {}
        model_value = (model_id or "").strip() if isinstance(model_id, str) else (model_id or "")

        missing_fields: list[str] = []
        if not user_id:
            missing_fields.append("guid")
        if not email:
            missing_fields.append("email")
        if not name:
            missing_fields.append("name")
        if not model_value:
            missing_fields.append("model")
        if not snapshot_usage:
            missing_fields.append("usage")
        if missing_fields:
            self.logger.debug(
                "Skipping cost snapshot due to missing fields: %s",
                ", ".join(sorted(missing_fields)),
            )
            return

        ttl = valves.COSTS_REDIS_TTL_SECONDS
        ts = int(time.time())
        raw_pipe_id = pipe_id or getattr(self, "id", None)
        if not raw_pipe_id:
            self.logger.debug("Skipping cost snapshot due to missing pipe identifier.")
            return
        pipe_namespace = _sanitize_table_fragment(raw_pipe_id)
        key = f"costs:{pipe_namespace}:{user_id}:{uuid.uuid4()}:{ts}"
        payload = {
            "guid": user_id,
            "email": str(email),
            "name": str(name),
            "model": model_value,
            "usage": snapshot_usage,
            "ts": ts,
        }
        try:
            await _wait_for(
                self._redis_client.set(key, json.dumps(payload, default=str), ex=ttl)
            )
        except Exception as exc:  # pragma: no cover - Redis failures logged, not fatal
            self.logger.debug("Cost snapshot write failed for user=%s: %s", user_id, exc)

    def _qualify_model_for_pipe(
        self,
        pipe_identifier: Optional[str],
        model_id: Optional[str],
    ) -> Optional[str]:
        """Return a dot-prefixed Open WebUI model id for this pipe."""
        if not isinstance(model_id, str):
            return None
        trimmed = model_id.strip()
        if not trimmed:
            return None
        if not pipe_identifier:
            return trimmed
        prefix = f"{pipe_identifier}."
        if trimmed.startswith(prefix):
            return trimmed
        normalized = ModelFamily.base_model(trimmed) or trimmed
        return f"{pipe_identifier}.{normalized}"

    async def pipes(self):
        """Return the list of models exposed to Open WebUI."""
        self._maybe_start_startup_checks()
        self._maybe_start_redis()
        self._maybe_start_cleanup()
        session = self._create_http_session()
        refresh_error: Exception | None = None
        try:
            await OpenRouterModelRegistry.ensure_loaded(
                session,
                base_url=self.valves.BASE_URL,
                api_key=EncryptedStr.decrypt(self.valves.API_KEY),
                cache_seconds=self.valves.MODEL_CATALOG_REFRESH_SECONDS,
                logger=self.logger,
            )
        except ValueError as exc:
            refresh_error = exc
            self.logger.error("OpenRouter configuration error: %s", exc)
        except Exception as exc:
            refresh_error = exc
            self.logger.warning("OpenRouter catalog refresh failed: %s", exc)
        finally:
            await session.close()

        available_models = OpenRouterModelRegistry.list_models()
        if refresh_error and available_models:
            self.logger.warning("Serving %d cached OpenRouter model(s) due to refresh failure.", len(available_models))
        if refresh_error and not available_models:
            return []

        selected_models = self._select_models(self.valves.MODEL_ID, available_models)
        enriched: list[dict[str, Any]] = []
        for model in selected_models:
            capabilities = ModelFamily.capabilities(model["id"])
            entry: dict[str, Any] = {"id": model["id"], "name": model["name"]}
            if capabilities:
                entry["capabilities"] = capabilities
                entry["meta"] = {"capabilities": capabilities}
            enriched.append(entry)
        return enriched

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __request__: Request | None,
        __event_emitter__: EventEmitter | None,
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        __metadata__: dict[str, Any],
        __tools__: list[dict[str, Any]] | dict[str, Any] | None,
        __task__: Optional[dict[str, Any]] = None,
        __task_body__: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[dict[str, Any] | str, None] | str | None | JSONResponse:
        """Entry point that enqueues work and awaits the isolated job result."""

        self._maybe_start_log_worker()
        self._maybe_start_startup_checks()
        self._maybe_start_redis()
        self._maybe_start_cleanup()

        if not isinstance(body, dict):
            body = {}
        if not isinstance(__user__, dict):
            __user__ = {}
        if not isinstance(__metadata__, dict):
            __metadata__ = {}

        user_valves_raw = __user__.get("valves") or {}
        user_valves = self.UserValves.model_validate(user_valves_raw)
        valves = self._merge_valves(self.valves, user_valves)
        user_id = str(__user__.get("id") or __metadata__.get("user_id") or "")
        safe_event_emitter = self._wrap_safe_event_emitter(__event_emitter__)
        wants_stream = bool(body.get("stream"))

        if not self._breaker_allows(user_id):
            message = "Temporarily disabled due to repeated errors. Please retry later."
            if safe_event_emitter:
                await self._emit_notification(safe_event_emitter, message, level="warning")
            SessionLogger.cleanup()
            return message

        if self._warmup_failed:
            message = "Service unavailable due to startup issues"
            if safe_event_emitter:
                await self._emit_error(
                    safe_event_emitter,
                    message,
                    show_error_message=True,
                    done=True,
                )
            SessionLogger.cleanup()
            return message

        await self._ensure_concurrency_controls(valves)
        queue = type(self)._request_queue
        if queue is None:
            raise RuntimeError("request queue not initialized")

        loop = asyncio.get_running_loop()
        stream_queue: asyncio.Queue[dict[str, Any] | str | None] | None = None
        future = loop.create_future()
        if wants_stream:
            stream_queue = asyncio.Queue()

        job = _PipeJob(
            pipe=self,
            body=body,
            user=__user__,
            request=__request__,
            event_emitter=None if wants_stream else safe_event_emitter,
            event_call=__event_call__,
            metadata=__metadata__,
            tools=__tools__,
            task=__task__,
            task_body=__task_body__,
            valves=valves,
            future=future,
            stream_queue=stream_queue,
        )

        if not self._enqueue_job(job):
            self.logger.warning("Request queue full; rejecting request_id=%s", job.request_id)
            if safe_event_emitter:
                await self._emit_error(
                    safe_event_emitter,
                    "Server busy (503)",
                    show_error_message=True,
                    done=True,
                )
            SessionLogger.cleanup()
            return "Server busy (503)"

        if wants_stream and stream_queue is not None:
            async def _stream() -> AsyncGenerator[dict[str, Any] | str, None]:
                try:
                    while True:
                        item = await stream_queue.get()
                        stream_queue.task_done()
                        if item is None:
                            break
                        yield item
                finally:
                    if not future.done():
                        future.cancel()
                    SessionLogger.cleanup()

            return _stream()

        try:
            result = await future
            return result
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            self.logger.debug("Pipe request cancelled by caller (request_id=%s)", job.request_id)
            raise
        except Exception as exc:  # pragma: no cover - defensive top-level guard
            self.logger.error("Pipe request failed (request_id=%s): %s", job.request_id, exc)
            if safe_event_emitter:
                await self._emit_error(
                    safe_event_emitter,
                    f"Pipe request failed: {exc}",
                    show_error_message=True,
                    done=True,
                )
            return "Request failed. Please retry."
        finally:
            SessionLogger.cleanup()

    async def _handle_pipe_call(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __request__: Request | None,
        __event_emitter__: EventEmitter | None,
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        __metadata__: dict[str, Any],
        __tools__: list[dict[str, Any]] | dict[str, Any] | None,
        __task__: Optional[dict[str, Any]] = None,
        __task_body__: Optional[dict[str, Any]] = None,
        *,
        valves: Pipe.Valves | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> AsyncGenerator[str, None] | str | None:
        """Process a user request and return either a stream or final text.

        When ``body['stream']`` is ``True`` the method yields deltas from
        ``_run_streaming_loop``.  Otherwise it falls back to
        ``_run_nonstreaming_loop`` and returns the aggregated response.
        """
        if not isinstance(body, dict):
            body = {}
        if not isinstance(__user__, dict):
            __user__ = {}
        if not isinstance(__metadata__, dict):
            __metadata__ = {}

        if valves is None:
            user_valves_raw = __user__.get("valves") or {}
            valves = self._merge_valves(
                self.valves,
                self.UserValves.model_validate(user_valves_raw),
            )
        if session is None:
            raise RuntimeError("HTTP session is required for _handle_pipe_call")

        model_block = __metadata__.get("model")
        openwebui_model_id = model_block.get("id", "") if isinstance(model_block, dict) else ""
        pipe_identifier = self.id
        self._ensure_artifact_store(valves, pipe_identifier=pipe_identifier)

        try:
            await OpenRouterModelRegistry.ensure_loaded(
                session,
                base_url=valves.BASE_URL,
                api_key=EncryptedStr.decrypt(valves.API_KEY),
                cache_seconds=valves.MODEL_CATALOG_REFRESH_SECONDS,
                logger=self.logger,
            )
        except ValueError as exc:
            await self._emit_error(
                __event_emitter__,
                f"OpenRouter configuration error: {exc}",
                show_error_message=True,
                done=True,
            )
            return ""
        except Exception as exc:
            available_models = OpenRouterModelRegistry.list_models()
            if not available_models:
                await self._emit_error(
                    __event_emitter__,
                    "OpenRouter model catalog unavailable. Please retry shortly.",
                    show_error_message=True,
                    done=True,
                )
                self.logger.error("OpenRouter model catalog unavailable: %s", exc)
                return ""
            self.logger.warning("OpenRouter catalog refresh failed (%s). Serving %d cached model(s).", exc, len(available_models))
        else:
            available_models = OpenRouterModelRegistry.list_models()
        allowed_models = self._select_models(valves.MODEL_ID, available_models) or available_models
        allowed_norm_ids = {m["norm_id"] for m in allowed_models}

        # Full model ID, e.g. "<pipe-id>.gpt-4o"
        pipe_token = ModelFamily._PIPE_ID.set(pipe_identifier)
        # Features are nested under the pipe id key – read them dynamically.
        _features_all = __metadata__.get("features", {}) or {}
        features = dict(_features_all.get(pipe_identifier, {}) or {})
        # Custom location that this manifold uses to store feature flags
        user_id = str(__user__.get("id") or __metadata__.get("user_id") or "")

        try:
            result = await self._process_transformed_request(
                body,
                __user__,
                __request__,
                __event_emitter__,
                __event_call__,
                __metadata__,
                __tools__,
                __task__,
                __task_body__,
                valves,
                session,
                openwebui_model_id,
                pipe_identifier,
                allowed_norm_ids,
                features,
                user_id=user_id,
            )
        # OpenRouter 400 errors (already have templates)
        except OpenRouterAPIError as e:
            await self._report_openrouter_error(
                e,
                event_emitter=__event_emitter__,
                normalized_model_id=body.get("model"),
                api_model_id=None,
            )
            return ""

        # Network timeouts
        except httpx.TimeoutException as e:
            await self._emit_templated_error(
                __event_emitter__,
                template=valves.NETWORK_TIMEOUT_TEMPLATE,
                variables={
                    "timeout_seconds": getattr(e, 'timeout', valves.HTTP_TOTAL_TIMEOUT_SECONDS or 120),
                    "endpoint": "https://openrouter.ai/api/v1/responses",
                },
                log_message=f"Network timeout: {e}",
            )
            return ""

        # Connection failures
        except httpx.ConnectError as e:
            await self._emit_templated_error(
                __event_emitter__,
                template=valves.CONNECTION_ERROR_TEMPLATE,
                variables={
                    "error_type": type(e).__name__,
                    "endpoint": "https://openrouter.ai",
                },
                log_message=f"Connection failed: {e}",
            )
            return ""

        # HTTP 5xx errors
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code if e.response else None
            reason_phrase = e.response.reason_phrase if e.response else None
            if status_code and status_code >= 500:
                await self._emit_templated_error(
                    __event_emitter__,
                    template=valves.SERVICE_ERROR_TEMPLATE,
                    variables={
                        "status_code": status_code,
                        "reason": reason_phrase or "Server Error",
                    },
                    log_message=f"OpenRouter service error: {status_code} {reason_phrase}",
                )
                return ""

            handled_statuses = {400, 401, 402, 403, 408, 429}
            if status_code in handled_statuses:
                body_text = None
                if e.response is not None:
                    try:
                        raw_bytes = await e.response.aread()
                        body_text = raw_bytes.decode("utf-8", errors="replace") if isinstance(raw_bytes, bytes) else str(raw_bytes)
                    except Exception:
                        body_text = None
                extra_meta: dict[str, Any] = {}
                if e.response is not None:
                    retry_after = e.response.headers.get("Retry-After") or e.response.headers.get("retry-after")
                    if retry_after:
                        extra_meta["retry_after"] = retry_after
                        extra_meta["retry_after_seconds"] = retry_after
                    rate_scope = (
                        e.response.headers.get("X-RateLimit-Scope")
                        or e.response.headers.get("x-ratelimit-scope")
                    )
                    if rate_scope:
                        extra_meta["rate_limit_type"] = rate_scope
                error = _build_openrouter_api_error(
                    status=status_code,
                    reason=reason_phrase or "HTTP error",
                    body_text=body_text,
                    requested_model=body.get("model"),
                    extra_metadata=extra_meta or None,
                )
                await self._report_openrouter_error(
                    error,
                    event_emitter=__event_emitter__,
                    normalized_model_id=body.get("model"),
                    api_model_id=None,
                )
                return ""

            raise

        # Generic catch-all
        except Exception as e:
            await self._emit_templated_error(
                __event_emitter__,
                template=valves.INTERNAL_ERROR_TEMPLATE,
                variables={
                    "error_type": type(e).__name__,
                },
                log_message=f"Unexpected error: {e}",
            )
            return ""

        finally:
            ModelFamily._PIPE_ID.reset(pipe_token)
        return result


    def _build_direct_tool_server_registry(
        self,
        __metadata__: dict[str, Any],
        *,
        valves: "Pipe.Valves",
        event_call: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        event_emitter: EventEmitter | None,
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        """Return OWUI-style "direct tool server" entries (callables + tool specs).

        Open WebUI direct tool servers are executed client-side via Socket.IO.
        The model-visible tool names are plain OpenAPI ``operationId`` values
        (no namespacing). Collisions overwrite ("last wins"), matching OWUI.
        """

        direct_registry: dict[str, dict[str, Any]] = {}
        direct_tool_specs: list[dict[str, Any]] = []

        try:
            if not isinstance(__metadata__, dict):
                return {}, []
            tool_servers = __metadata__.get("tool_servers")
            if not isinstance(tool_servers, list) or not tool_servers:
                return {}, []
            if event_call is None:
                # No Socket.IO bridge means direct tools cannot run; don't advertise them.
                return {}, []

            for server in tool_servers:
                try:
                    if not isinstance(server, dict):
                        continue
                    specs = server.get("specs")
                    if not isinstance(specs, list) or not specs:
                        # Best-effort fallback: derive specs from raw OpenAPI if present.
                        openapi = server.get("openapi")
                        if isinstance(openapi, dict):
                            try:
                                from open_webui.utils.tools import convert_openapi_to_tool_payload  # type: ignore
                            except Exception:
                                convert_openapi_to_tool_payload = None  # type: ignore[assignment]
                            if callable(convert_openapi_to_tool_payload):
                                try:
                                    specs = convert_openapi_to_tool_payload(openapi)  # type: ignore[misc]
                                except Exception:
                                    specs = []
                    if not isinstance(specs, list) or not specs:
                        continue

                    server_payload = dict(server)
                    with contextlib.suppress(Exception):
                        server_payload.pop("specs", None)

                    for spec in specs:
                        try:
                            if not isinstance(spec, dict):
                                continue
                            raw_name = spec.get("name")
                            name = raw_name.strip() if isinstance(raw_name, str) else ""
                            if not name:
                                continue

                            allowed_params: set[str] = set()
                            try:
                                parameters = spec.get("parameters")
                                if isinstance(parameters, dict):
                                    props = parameters.get("properties")
                                    if isinstance(props, dict):
                                        allowed_params = {k for k in props.keys() if isinstance(k, str)}
                            except Exception:
                                allowed_params = set()

                            spec_payload = dict(spec)
                            spec_payload["name"] = name

                            async def _direct_tool_callable(  # noqa: ANN001 - tool kwargs are dynamic
                                _allowed_params: set[str] = allowed_params,
                                _tool_name: str = name,
                                _server_payload: dict[str, Any] = server_payload,
                                _metadata: dict[str, Any] = __metadata__,
                                _event_call: Callable[[dict[str, Any]], Awaitable[Any]] | None = event_call,
                                _event_emitter: EventEmitter | None = event_emitter,
                                **kwargs,
                            ) -> Any:
                                try:
                                    filtered: dict[str, Any] = {}
                                    try:
                                        filtered = {k: v for k, v in kwargs.items() if k in _allowed_params}
                                    except Exception:
                                        filtered = {}

                                    session_id = None
                                    try:
                                        session_id = _metadata.get("session_id")
                                    except Exception:
                                        session_id = None

                                    payload = {
                                        "type": "execute:tool",
                                        "data": {
                                            "id": str(uuid.uuid4()),
                                            "name": _tool_name,
                                            "params": filtered,
                                            "server": _server_payload,
                                            "session_id": session_id,
                                        },
                                    }
                                    if _event_call is None:
                                        return [{"error": "Direct tool execution unavailable."}, None]
                                    return await _event_call(payload)  # type: ignore[misc]
                                except Exception as exc:
                                    # Never let tool failures crash the pipe/session.
                                    self.logger.debug("Direct tool '%s' failed: %s", _tool_name, exc, exc_info=True)
                                    with contextlib.suppress(Exception):
                                        await self._emit_notification(
                                            _event_emitter,
                                            f"Tool '{_tool_name}' failed: {exc}",
                                            level="warning",
                                        )
                                    return [{"error": str(exc)}, None]

                            direct_registry[name] = {
                                "spec": spec_payload,
                                "direct": True,
                                "server": server_payload,
                                "callable": _direct_tool_callable,
                            }
                        except Exception:
                            # Skip malformed tool specs safely.
                            self.logger.debug("Skipping malformed direct tool spec", exc_info=True)
                            continue
                except Exception:
                    # Skip malformed server entries safely.
                    self.logger.debug("Skipping malformed direct tool server entry", exc_info=True)
                    continue

            if direct_registry:
                try:
                    direct_tool_specs = ResponsesBody.transform_owui_tools(
                        direct_registry,
                        strict=valves.ENABLE_STRICT_TOOL_CALLING,
                    )
                except Exception:
                    direct_tool_specs = []
            return direct_registry, direct_tool_specs
        except Exception:
            self.logger.debug("Direct tool server registry build failed", exc_info=True)
            return {}, []


    async def _process_transformed_request(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __request__: Request | None,
        __event_emitter__: EventEmitter | None,
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        __metadata__: dict[str, Any],
        __tools__: list[dict[str, Any]] | dict[str, Any] | None,
        __task__: Optional[dict[str, Any]],
        __task_body__: Optional[dict[str, Any]],
        valves: "Pipe.Valves",
        session: aiohttp.ClientSession,
        openwebui_model_id: str,
        pipe_identifier: str,
        allowed_norm_ids: set[str],
        features: dict[str, Any],
        *,
        user_id: str = "",
    ) -> AsyncGenerator[str, None] | str | None:
        user_id = user_id or str(__user__.get("id") or __metadata__.get("user_id") or "")
        # Optional: inject CSS tweak for multi-line statuses when enabled.
        if valves.ENABLE_STATUS_CSS_PATCH:
            if __event_call__:
                payload_user_id = user_id or __metadata__.get("user_id") or __user__.get("id") or "anonymous"
                try:
                    await __event_call__({
                        "type": "execute",
                        "user_id": str(payload_user_id),
                        "data": {
                            "code": """
                            (() => {
                                if (document.getElementById("owui-status-unclamp")) return "ok";
                                const style = document.createElement("style");
                                style.id = "owui-status-unclamp";
                                style.textContent = `
                                    .status-description .line-clamp-1,
                                    .status-description .text-base.line-clamp-1,
                                    .status-description .text-gray-500.text-base.line-clamp-1 {
                                        display: block !important;
                                        overflow: visible !important;
                                        -webkit-line-clamp: unset !important;
                                        -webkit-box-orient: initial !important;
                                        white-space: pre-wrap !important;
                                        word-break: break-word;
                                    }
                                    .status-description .text-base::first-line,
                                    .status-description .text-gray-500.text-base::first-line {
                                        font-weight: 500 !important;
                                    }
                                `;
                                document.head.appendChild(style);
                                return "ok";
                            })();
                            """
                        }
                    })
                except Exception as exc:  # pragma: no cover - UI injection optional
                    self.logger.debug("Status CSS injection failed: %s", exc)
            else:
                self.logger.debug("Status CSS injection skipped: __event_call__ unavailable.")

        # STEP 1: Transform request body (Completions API -> Responses API).
        completions_body = CompletionsBody.model_validate(body)

        # Resolve full Open WebUI user model for uploads/status events.
        user_model = None
        if user_id:
            try:
                user_model = await self._get_user_by_id(user_id)
            except Exception:  # pragma: no cover - defensive guard
                user_model = None
        responses_body = await ResponsesBody.from_completions(
            completions_body=completions_body,

            # If chat_id and openwebui_model_id are provided, from_completions() uses them to fetch previously persisted items (function_calls, reasoning, etc.) from DB and reconstruct the input array in the correct order.
            **({"chat_id": __metadata__["chat_id"]} if __metadata__.get("chat_id") else {}),
            **({"openwebui_model_id": openwebui_model_id} if openwebui_model_id else {}),
            artifact_loader=self._db_fetch,
            pruning_turns=valves.TOOL_OUTPUT_RETENTION_TURNS,
            transformer_context=self,
            request=__request__,
            user_obj=user_model,
            event_emitter=__event_emitter__,
            transformer_valves=valves,

        )
        self._sanitize_request_input(responses_body)
        self._apply_reasoning_preferences(responses_body, valves)
        self._apply_gemini_thinking_config(responses_body, valves)
        self._apply_context_transforms(responses_body, valves)
        if valves.USE_MODEL_MAX_OUTPUT_TOKENS:
            if responses_body.max_output_tokens is None:
                default_max = ModelFamily.max_completion_tokens(responses_body.model)
                if default_max:
                    responses_body.max_output_tokens = default_max
        else:
            responses_body.max_output_tokens = None

        normalized_model_id = ModelFamily.base_model(responses_body.model)
        task_mode = bool(__task__)
        if allowed_norm_ids and normalized_model_id not in allowed_norm_ids:
            if task_mode:
                self.logger.debug(
                    "Bypassing model whitelist for task request (model=%s, task=%s)",
                    normalized_model_id,
                    (__task__ or {}).get("type") or "task",
                )
            else:
                await self._emit_error(
                    __event_emitter__,
                    f"Model '{responses_body.model}' is not enabled for this pipe. Please choose one of the allowed models.",
                    show_error_message=True,
                    done=True,
                )
                return ""
        if not features:
            fallback_caps = (
                ModelFamily.capabilities(openwebui_model_id or "")
                or ModelFamily.capabilities(responses_body.model)
            )
            if fallback_caps:
                features = dict(fallback_caps)

        # STEP 2: Detect if task model (generate title, generate tags, etc.), handle it separately
        task_effort = None
        if __task__:
            self.logger.debug("Detected task model: %s", __task__)

            requested_model = responses_body.model
            owns_task_model = ModelFamily.base_model(requested_model) in allowed_norm_ids if allowed_norm_ids else True
            if owns_task_model:
                task_effort = valves.TASK_MODEL_REASONING_EFFORT
                self._apply_task_reasoning_preferences(responses_body, task_effort)
                self._apply_gemini_thinking_config(responses_body, valves)

            result = await self._run_task_model_request(
                responses_body.model_dump(),
                valves,
                session=session,
                task_context=__task__,
                user_id=user_id or "",
                user_obj=user_model,
                pipe_id=pipe_identifier,
                snapshot_model_id=self._qualify_model_for_pipe(
                    pipe_identifier,
                    responses_body.model,
                ),
            )
            return result

        # STEP 3: Build OpenAI Tools JSON (from __tools__, valves, and completions_body.extra_tools)
        tools_registry = __tools__
        if inspect.isawaitable(tools_registry):
            try:
                tools_registry = await tools_registry
            except Exception as exc:
                self.logger.warning("Tool registry unavailable; continuing without tools: %s", exc)
                await self._emit_notification(
                    __event_emitter__,
                    "Tool registry unavailable; continuing without tools.",
                    level="warning",
                )
                tools_registry = {}
        __tools__ = tools_registry

        # STEP 3a: Merge Open WebUI "direct tool servers" (user-configured OpenAPI servers).
        # These are executed client-side via Socket.IO (execute:tool) and must not crash the pipe.
        direct_registry: dict[str, dict[str, Any]] = {}
        direct_tool_specs: list[dict[str, Any]] = []
        try:
            direct_registry, direct_tool_specs = self._build_direct_tool_server_registry(
                __metadata__,
                valves=valves,
                event_call=__event_call__,
                event_emitter=__event_emitter__,
            )
        except Exception:
            direct_registry, direct_tool_specs = {}, []
        if direct_registry:
            try:
                if isinstance(__tools__, dict):
                    __tools__.update(direct_registry)
                elif isinstance(__tools__, list):
                    __tools__.extend(direct_registry.values())
                elif __tools__ is None:
                    __tools__ = dict(direct_registry)
            except Exception:
                self.logger.debug("Failed to merge direct tool servers into registry", exc_info=True)

        merged_extra_tools: list[dict[str, Any]] = []
        try:
            upstream_extra = getattr(completions_body, "extra_tools", None)
            if isinstance(upstream_extra, list):
                merged_extra_tools.extend([t for t in upstream_extra if isinstance(t, dict)])
        except Exception:
            merged_extra_tools = []
        if direct_tool_specs:
            merged_extra_tools.extend(direct_tool_specs)
        if not merged_extra_tools:
            merged_extra_tools = []

        tools = build_tools(
            responses_body,
            valves,
            __tools__=__tools__,
            features=features,
            extra_tools=merged_extra_tools or None,
        )

        # STEP 4: Auto-enable native function calling if tools are used but `native` function calling is not enabled in Open WebUI model settings.
        if tools and ModelFamily.supports("function_calling", openwebui_model_id):
            try:
                model = await run_in_threadpool(Models.get_model_by_id, openwebui_model_id)
            except Exception as exc:
                self.logger.warning("Failed to inspect model '%s' for function calling: %s", openwebui_model_id, exc)
                await self._emit_notification(
                    __event_emitter__,
                    f"Unable to verify function calling for {openwebui_model_id}. Please ensure it is set to native.",
                    level="warning",
                )
                model = None
            if model:
                params = dict(model.params or {})
                if params.get("function_calling") != "native":
                    await self._emit_notification(
                        __event_emitter__,
                        content=f"Enabling native function calling for model: {openwebui_model_id}. Please re-run your query.",
                        level="info"
                    )
                    params["function_calling"] = "native"
                    form_data = model.model_dump()
                    form_data["params"] = params
                    try:
                        await run_in_threadpool(
                            Models.update_model_by_id,
                            openwebui_model_id,
                            ModelForm(**form_data),
                        )
                    except Exception as exc:
                        self.logger.warning("Failed to update model '%s' function calling setting: %s", openwebui_model_id, exc)
                        await self._emit_notification(
                            __event_emitter__,
                            f"Unable to persist function calling setting for {openwebui_model_id}. Please update it manually.",
                            level="warning",
                        )

        # STEP 5: Add tools to responses body, if supported
        if ModelFamily.supports("function_calling", responses_body.model):
            if tools:
                responses_body.tools = tools

        # STEP 6: Configure OpenRouter web-search plugin if supported/enabled
        if ModelFamily.supports("web_search_tool", responses_body.model) and (
            valves.ENABLE_WEB_SEARCH_TOOL or features.get("web_search", False)
        ):
            reasoning_cfg = responses_body.reasoning if isinstance(responses_body.reasoning, dict) else {}
            effort = (reasoning_cfg.get("effort") or "").strip().lower()
            if not effort:
                effort = valves.REASONING_EFFORT
            if effort == "minimal":
                self.logger.debug(
                    "Skipping web-search plugin because reasoning.effort is set to 'minimal' (model=%s)",
                    responses_body.model,
                )
            else:
                plugin_payload: dict[str, Any] = {"id": "web"}
                if valves.WEB_SEARCH_MAX_RESULTS is not None:
                    plugin_payload["max_results"] = valves.WEB_SEARCH_MAX_RESULTS
                plugins = list(responses_body.plugins or [])
                plugins.append(plugin_payload)
                responses_body.plugins = plugins

        # Convert the normalized model id back to the original OpenRouter id for the API request.
        setattr(responses_body, "api_model", OpenRouterModelRegistry.api_model_id(normalized_model_id) or normalized_model_id)

        # STEP 7: Send to OpenAI Responses API (with provider-specific fallbacks)
        reasoning_retry_attempted = False
        reasoning_effort_retry_attempted = False
        while True:
            try:
                if responses_body.stream:
                    # Return async generator for partial text
                    return await self._run_streaming_loop(
                        responses_body,
                        valves,
                        __event_emitter__,
                        __metadata__,
                        __tools__,
                        session=session,
                        user_id=user_id,
                        request_context=__request__,
                        user_obj=user_model,
                        pipe_identifier=pipe_identifier,
                    )
                # Return final text (non-streaming)
                return await self._run_nonstreaming_loop(
                    responses_body,
                    valves,
                    __event_emitter__,
                    __metadata__,
                    __tools__,
                    session=session,
                    user_id=user_id,
                    request_context=__request__,
                    user_obj=user_model,
                    pipe_identifier=pipe_identifier,
                )
            except OpenRouterAPIError as exc:
                if not reasoning_effort_retry_attempted:
                    error_details = {"provider_raw": exc.provider_raw}
                    if _is_reasoning_effort_error(error_details):
                        original_effort = None
                        if isinstance(responses_body.reasoning, dict):
                            original_effort = responses_body.reasoning.get("effort")

                        error_message = exc.upstream_message or exc.openrouter_message or ""
                        supported_values = _parse_supported_effort_values(error_message)

                        if supported_values:
                            fallback_effort = _select_best_effort_fallback(
                                original_effort or "",
                                supported_values,
                            )
                            if fallback_effort:
                                self.logger.info(
                                    "Reasoning effort '%s' not supported by model %s. Retrying with '%s'. Supported values: %s",
                                    original_effort,
                                    responses_body.model,
                                    fallback_effort,
                                    ", ".join(supported_values),
                                )
                                if __event_emitter__:
                                    try:
                                        await __event_emitter__(
                                            {
                                                "type": "status",
                                                "data": {
                                                    "description": (
                                                        f"Adjusting reasoning effort from '{original_effort}' to "
                                                        f"'{fallback_effort}' (model doesn't support '{original_effort}')"
                                                    ),
                                                    "done": False,
                                                },
                                            }
                                        )
                                    except Exception as emit_error:
                                        self.logger.debug("Failed to emit status update: %s", emit_error)

                                if not isinstance(responses_body.reasoning, dict):
                                    responses_body.reasoning = {}
                                responses_body.reasoning["effort"] = fallback_effort
                                reasoning_effort_retry_attempted = True
                                self._apply_gemini_thinking_config(responses_body, valves)
                                continue

                if (
                    not reasoning_retry_attempted
                    and self._should_retry_without_reasoning(exc, responses_body)
                ):
                    reasoning_retry_attempted = True
                    continue

                await self._report_openrouter_error(
                    exc,
                    event_emitter=__event_emitter__,
                    normalized_model_id=responses_body.model,
                    api_model_id=getattr(responses_body, "api_model", None),
                    template=valves.OPENROUTER_ERROR_TEMPLATE,
                )
                return ""


    def _apply_reasoning_preferences(self, responses_body: ResponsesBody, valves: "Pipe.Valves") -> None:
        """Automatically request reasoning traces when supported and enabled."""
        if not valves.ENABLE_REASONING:
            return

        supported = ModelFamily.supported_parameters(responses_body.model)
        supports_reasoning = "reasoning" in supported
        supports_legacy_only = "include_reasoning" in supported and not supports_reasoning
        summary_mode = valves.REASONING_SUMMARY_MODE
        requested_summary: Optional[str] = None
        if summary_mode != "disabled":
            requested_summary = summary_mode

        target_effort = valves.REASONING_EFFORT

        if supports_reasoning:
            cfg: dict[str, Any] = {}
            if isinstance(responses_body.reasoning, dict):
                cfg = dict(responses_body.reasoning)
            if target_effort and "effort" not in cfg:
                cfg["effort"] = target_effort
            if requested_summary and "summary" not in cfg:
                cfg["summary"] = requested_summary
            cfg.setdefault("enabled", True)
            responses_body.reasoning = cfg or None
            if getattr(responses_body, "include_reasoning", None) is not None:
                setattr(responses_body, "include_reasoning", None)
        elif supports_legacy_only:
            responses_body.reasoning = None
            desired = target_effort not in {"none", ""}
            setattr(responses_body, "include_reasoning", desired)
        else:
            responses_body.reasoning = None
            setattr(responses_body, "include_reasoning", False)

    def _apply_task_reasoning_preferences(self, responses_body: ResponsesBody, effort: str) -> None:
        """Override reasoning effort for task models."""
        if not effort:
            return
        supported = ModelFamily.supported_parameters(responses_body.model)
        supports_reasoning = "reasoning" in supported
        supports_legacy_only = "include_reasoning" in supported and not supports_reasoning
        target_effort = effort.strip().lower()

        if supports_reasoning:
            cfg = (
                responses_body.reasoning
                if isinstance(responses_body.reasoning, dict)
                else {}
            )
            cfg = dict(cfg) if cfg else {}
            cfg["effort"] = target_effort
            cfg.setdefault("enabled", True)
            responses_body.reasoning = cfg
            if getattr(responses_body, "include_reasoning", None) is not None:
                setattr(responses_body, "include_reasoning", None)
        elif supports_legacy_only:
            responses_body.reasoning = None
            desired = target_effort not in {"none", "minimal"}
            setattr(responses_body, "include_reasoning", desired)
        else:
            responses_body.reasoning = None
            setattr(responses_body, "include_reasoning", False)

    def _apply_gemini_thinking_config(self, responses_body: ResponsesBody, valves: "Pipe.Valves") -> None:
        """Translate reasoning preferences into Vertex thinking_config for Gemini models."""
        normalized_model = ModelFamily.base_model(responses_body.model)
        family = _classify_gemini_thinking_family(normalized_model)
        if not family:
            responses_body.thinking_config = None
            return

        reasoning_cfg = (
            responses_body.reasoning if isinstance(responses_body.reasoning, dict) else {}
        )
        include_flag = getattr(responses_body, "include_reasoning", None)
        effort_hint = (reasoning_cfg.get("effort") or "").strip().lower()
        if not effort_hint:
            effort_hint = valves.REASONING_EFFORT

        enabled = reasoning_cfg.get("enabled", True)
        exclude = reasoning_cfg.get("exclude", False)
        if effort_hint == "none":
            enabled = False

        reasoning_requested = bool(include_flag) or (reasoning_cfg and enabled and not exclude)
        if not reasoning_requested:
            responses_body.thinking_config = None
            setattr(responses_body, "include_reasoning", False)
            return

        thinking_config: dict[str, Any] = {"include_thoughts": True}
        if family == "gemini-3":
            level = _map_effort_to_gemini_level(effort_hint, valves.GEMINI_THINKING_LEVEL)
            if level is None:
                responses_body.thinking_config = None
                setattr(responses_body, "include_reasoning", False)
                return
            thinking_config["thinking_level"] = level
        elif family == "gemini-2.5":
            budget = _map_effort_to_gemini_budget(effort_hint, valves.GEMINI_THINKING_BUDGET)
            if budget is None:
                responses_body.thinking_config = None
                setattr(responses_body, "include_reasoning", False)
                return
            thinking_config["thinking_budget"] = budget

        responses_body.thinking_config = thinking_config
        responses_body.reasoning = None
        setattr(responses_body, "include_reasoning", None)

    def _apply_context_transforms(self, responses_body: ResponsesBody, valves: "Pipe.Valves") -> None:
        """Attach OpenRouter's middle-out transform when auto trimming is enabled."""

        if not valves.AUTO_CONTEXT_TRIMMING:
            return
        if responses_body.transforms is not None:
            return
        responses_body.transforms = ["middle-out"]

    def _sanitize_request_input(self, body: ResponsesBody) -> None:
        """Remove non-replayable artifacts that may have snuck into body.input."""
        items = getattr(body, "input", None)
        if not isinstance(items, list):
            return
        sanitized = _filter_replayable_input_items(items, logger=self.logger)
        if sanitized is items:
            return
        removed = len(items) - len(sanitized)
        self.logger.debug(
            "Sanitized provider input: removed %d non-replayable artifact(s).",
            removed,
        )
        body.input = sanitized

    async def transform_messages_to_input(
        self: "Pipe",
        messages: List[Dict[str, Any]],
        chat_id: Optional[str] = None,
        openwebui_model_id: Optional[str] = None,
        artifact_loader: Optional[
            Callable[[Optional[str], Optional[str], List[str]], Awaitable[Dict[str, Dict[str, Any]]]]
        ] = None,
        pruning_turns: int = 0,
        replayed_reasoning_refs: Optional[List[Tuple[str, str]]] = None,
        __request__: Optional[Request] = None,
        user_obj: Optional[Any] = None,
        event_emitter: Optional[Callable] = None,
        *,
        model_id: Optional[str] = None,
        valves: Optional["Pipe.Valves"] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build an OpenAI Responses-API `input` array from Open WebUI-style messages.

        Parameters `chat_id` and `openwebui_model_id` are optional. When both are
        supplied and the messages contain empty-link encoded item references, the
        function fetches persisted items from the database and injects them in the
        correct order. When either parameter is missing, the messages are simply
        converted without attempting to fetch persisted items.

        When provided, `artifact_loader` is awaited with `(chat_id, message_id, ulids)`
        for each assistant message so database-backed artifacts can be replayed.
        When `replayed_reasoning_refs` is supplied, the function appends each
        `(chat_id, artifact_id)` pair for reasoning items so the caller can clean
        them up after they have been replayed once.

        Returns
        -------
        List[dict] : The fully-formed `input` list for the OpenAI Responses API.
        """

        logger = LOGGER
        active_valves = valves or self.valves
        image_limit = getattr(
            active_valves,
            "MAX_INPUT_IMAGES_PER_REQUEST",
            self.valves.MAX_INPUT_IMAGES_PER_REQUEST,
        )
        selection_mode = getattr(
            active_valves,
            "IMAGE_INPUT_SELECTION",
            self.valves.IMAGE_INPUT_SELECTION,
        )
        chunk_size = getattr(
            active_valves,
            "IMAGE_UPLOAD_CHUNK_BYTES",
            self.valves.IMAGE_UPLOAD_CHUNK_BYTES,
        )
        max_inline_bytes = (
            getattr(
                active_valves,
                "BASE64_MAX_SIZE_MB",
                self.valves.BASE64_MAX_SIZE_MB,
            )
            * 1024
            * 1024
        )
        target_model_id = model_id or openwebui_model_id or ""
        if target_model_id:
            vision_supported = ModelFamily.supports("vision", target_model_id)
        else:
            vision_supported = True

        openai_input: list[dict] = []
        last_assistant_images: list[dict[str, Any]] = []

        def _message_identifier(entry: dict[str, Any]) -> Optional[str]:
            """Return the most specific identifier available on ``entry``."""
            for key in ("id", "_id", "message_id"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return None

        def _compute_turn_indices() -> tuple[list[Optional[int]], int]:
            """Label each message with a turn index and return the total count."""
            indices: list[Optional[int]] = []
            current_turn = -1
            max_turn = -1
            last_dialog_role: Optional[str] = None

            for msg in messages:
                role = (msg.get("role") or "").lower()
                turn_idx: Optional[int] = None

                if role == "user":
                    if last_dialog_role != "user":
                        current_turn += 1
                    turn_idx = current_turn
                    last_dialog_role = "user"
                elif role == "assistant":
                    if current_turn < 0:
                        current_turn = 0
                    turn_idx = current_turn
                    last_dialog_role = "assistant"
                else:
                    turn_idx = current_turn if current_turn >= 0 else None

                if turn_idx is not None and turn_idx > max_turn:
                    max_turn = turn_idx
                indices.append(turn_idx)

            total_turns = max_turn + 1 if max_turn >= 0 else 0
            return indices, total_turns

        def _markdown_images_from_text(text: str) -> list[str]:
            """Extract inline Markdown image URLs from a text block."""
            if not isinstance(text, str):
                return []
            return [
                match.group("url").strip()
                for match in _MARKDOWN_IMAGE_RE.finditer(text)
                if match.group("url").strip()
            ]

        def _is_old_turn(turn_index: Optional[int], *, threshold: Optional[int]) -> bool:
            """Return True when a message turn falls outside the retention window."""
            return (
                threshold is not None
                and turn_index is not None
                and turn_index < threshold
            )

        def _prune_tool_output(
            item: Dict[str, Any],
            *,
            marker: Optional[str],
            turn_index: Optional[int],
            retention_turns: int,
        ) -> bool:
            """Shorten oversized tool output strings while leaving markers intact."""
            if item.get("type") != "function_call_output":
                return False

            output_value = item.get("output")
            if output_value is None:
                return False
            if not isinstance(output_value, str):
                try:
                    output_text = json.dumps(output_value, ensure_ascii=False)
                except (TypeError, ValueError):
                    # Fallback to str() if object isn't JSON serializable
                    output_text = str(output_value)
            else:
                output_text = output_value

            if len(output_text) < _TOOL_OUTPUT_PRUNE_MIN_LENGTH:
                return False

            head = output_text[:_TOOL_OUTPUT_PRUNE_HEAD_CHARS].rstrip()
            tail = output_text[-_TOOL_OUTPUT_PRUNE_TAIL_CHARS:].lstrip()
            removed_chars = len(output_text) - len(head) - len(tail)
            if removed_chars <= 0:
                return False

            ellipsis = "..."
            turn_label = (
                f"{retention_turns} turn" if retention_turns == 1 else f"{retention_turns} turns"
            )
            note = (
                f"{ellipsis}\n"
                f"[tool output pruned: removed {removed_chars} char"
                f"{'' if removed_chars == 1 else 's'}, older than {turn_label}]\n"
                f"{ellipsis}"
            )
            item["output"] = "\n".join(
                part for part in (head, note, tail) if part
            )
            LOGGER.debug("Pruned tool output (marker=%s, call_id=%s, turn=%s, removed_chars=%d, retention=%d)", marker, item.get("call_id"), turn_index, removed_chars, retention_turns)
            return True

        turn_indices, total_turns = _compute_turn_indices()
        prune_before_turn: Optional[int] = None
        if pruning_turns > 0 and total_turns > pruning_turns:
            prune_before_turn = total_turns - pruning_turns

        for idx, msg in enumerate(messages):
            raw_role = msg.get("role")
            role = (raw_role or "").lower()
            raw_content = msg.get("content", "")
            msg_id = msg.get("message_id") or _message_identifier(msg)
            msg_turn_index = turn_indices[idx]

            # -------- system / developer messages --------------------------- #
            if role in {"system", "developer"}:
                # Preserve system/developer content exactly as supplied by Open WebUI.
                # We still adapt it into Responses API "input_text" blocks, but we do
                # not strip/trim/normalize whitespace or merge blocks.
                blocks: list[dict[str, Any]] = []

                if isinstance(raw_content, str):
                    blocks.append({"type": "input_text", "text": raw_content})
                elif isinstance(raw_content, list):
                    for entry in raw_content:
                        if isinstance(entry, str):
                            blocks.append({"type": "input_text", "text": entry})
                            continue
                        if isinstance(entry, dict):
                            text_val = entry.get("text")
                            if not isinstance(text_val, str):
                                text_val = entry.get("content")
                            if isinstance(text_val, str):
                                blocks.append({"type": "input_text", "text": text_val})
                elif isinstance(raw_content, dict):
                    text_val = raw_content.get("text")
                    if not isinstance(text_val, str):
                        text_val = raw_content.get("content")
                    if isinstance(text_val, str):
                        blocks.append({"type": "input_text", "text": text_val})

                if blocks:
                    openai_input.append(
                        {
                            "type": "message",
                            "role": role,
                            "content": blocks,
                        }
                    )
                continue

            # -------- user message ---------------------------------------- #
            if role == "user":
                # Convert string content to a block list (["Hello"] → [{"type": "text", "text": "Hello"}])
                content_blocks = msg.get("content") or []
                if isinstance(content_blocks, str):
                    content_blocks = [{"type": "text", "text": content_blocks}]

                # Only transform known types; leave all others unchanged
                async def _to_input_image(block: dict) -> dict:
                    """Convert Open WebUI image block into Responses format.

                    Handles image URLs and base64 data URLs, downloading remote images and
                    saving all images to OWUI storage to prevent data loss.

                    Supported Image Formats (per OpenRouter docs):
                        - image/png
                        - image/jpeg
                        - image/webp
                        - image/gif

                    Args:
                        block: Content block from Open WebUI message

                    Returns:
                        Responses API input_image block with internal OWUI storage URL

                    Processing Flow:
                        1. Extract image URL from block (nested or flat structure)
                        2. If data URL: Parse, upload to OWUI storage
                        3. If remote URL: Download, upload to OWUI storage
                        4. If OWUI file reference: Keep as-is
                        5. Return Responses API format with detail level

                    Note:
                        Image data URLs and remote URLs are ALWAYS saved to storage
                        to prevent chat history bloat, similar to the default behavior of
                        SAVE_FILE_DATA_CONTENT for file inputs. This cannot be disabled
                        via valve configuration as inline image payloads significantly
                        degrade UI performance and storage efficiency.
                        All errors are caught and logged with status emissions.
                        Failed processing returns empty image_url rather than crashing.
                    """
                    try:
                        image_payload = block.get("image_url")
                        detail: Optional[str] = None
                        url: str = ""

                        if isinstance(image_payload, dict):
                            url = image_payload.get("url", "")
                            detail = image_payload.get("detail")
                        elif isinstance(image_payload, str):
                            url = image_payload

                        if not url:
                            return {"type": "input_image", "image_url": "", "detail": "auto"}

                        storage_context: Optional[Tuple[Optional[Request], Optional[Any]]] = None

                        async def _get_storage_context() -> tuple[Optional[Request], Optional[Any]]:
                            """Resolve (request,user) tuple only once for storage uploads."""
                            nonlocal storage_context
                            if storage_context is None:
                                storage_context = await self._resolve_storage_context(__request__, user_obj)
                            return storage_context

                        async def _save_image_bytes(
                            payload: bytes,
                            mime_type: str,
                            preferred_name: str,
                            status_message: str,
                        ) -> Optional[str]:
                            """Upload image bytes to Open WebUI storage and emit status."""
                            upload_request, upload_user = await _get_storage_context()
                            if not (upload_request and upload_user):
                                return None
                            internal_url = await self._upload_to_owui_storage(
                                request=upload_request,
                                user=upload_user,
                                file_data=payload,
                                filename=preferred_name,
                                mime_type=mime_type,
                            )
                            if internal_url:
                                await self._emit_status(event_emitter, status_message, done=False)
                            return internal_url

                        # Handle data URLs (base64)
                        if url.startswith("data:"):
                            try:
                                parsed = self._parse_data_url(url)
                                if parsed:
                                    ext = parsed["mime_type"].split("/")[-1]
                                    internal_url = await _save_image_bytes(
                                        parsed["data"],
                                        parsed["mime_type"],
                                        f"image-{uuid.uuid4().hex}.{ext}",
                                        StatusMessages.IMAGE_BASE64_SAVED,
                                    )
                                    if internal_url:
                                        url = internal_url
                            except Exception as exc:
                                self.logger.error(f"Failed to process base64 image: {exc}")
                                await self._emit_error(
                                    event_emitter,
                                    f"Failed to save base64 image: {exc}",
                                    show_error_message=False
                                )

                        # Handle remote URLs (download and save to prevent deletion)
                        elif url.startswith(("http://", "https://")) and not _is_internal_file_url(url):
                            try:
                                downloaded = await self._download_remote_url(url)
                                if downloaded:
                                    filename = url.split("/")[-1].split("?")[0] or f"image-{uuid.uuid4().hex}"
                                    if "." not in filename:
                                        ext = downloaded["mime_type"].split("/")[-1]
                                        filename = f"{filename}.{ext}"

                                    internal_url = await _save_image_bytes(
                                        downloaded["data"],
                                        downloaded["mime_type"],
                                        filename,
                                        StatusMessages.IMAGE_REMOTE_SAVED,
                                    )
                                    if internal_url:
                                        url = internal_url
                            except Exception as exc:
                                self.logger.error(f"Failed to download remote image {url}: {exc}")
                                await self._emit_error(
                                    event_emitter,
                                    f"Failed to download image: {exc}",
                                    show_error_message=False
                                )
                        if _is_internal_file_url(url):
                            inlined = await self._inline_internal_file_url(
                                url,
                                chunk_size=chunk_size,
                                max_bytes=max_inline_bytes,
                            )
                            if inlined:
                                url = inlined

                        result: dict[str, Any] = {"type": "input_image", "image_url": url}
                        if isinstance(detail, str) and detail in {"auto", "low", "high"}:
                            result["detail"] = detail
                        elif url:
                            result["detail"] = "auto"

                        return result

                    except Exception as exc:
                        self.logger.error(f"Error in _to_input_image: {exc}")
                        await self._emit_error(
                            event_emitter,
                            f"Image processing error: {exc}",
                            show_error_message=False
                        )
                        return {"type": "input_image", "image_url": "", "detail": "auto"}

                async def _to_input_file(block: dict) -> dict:
                    """Convert Open WebUI file blocks into Responses API format.

                    Handles file content blocks from multiple sources, downloading remote files
                    and saving base64 data to OWUI storage for persistence.

                    Responses API File Input Fields (per OpenAPI spec):
                        - type: "input_file" (required)
                        - file_id: string | null (optional)
                        - file_data: string (optional) - base64 or data URL
                        - filename: string (optional) - for model context
                        - file_url: string (optional) - URL to file

                    Args:
                        block: Content block from Open WebUI message

                    Returns:
                        Responses API input_file block with all available fields

                    Processing Flow:
                        1. Extract fields from nested or flat block structure
                        2. If file_data provided AND SAVE_FILE_DATA_CONTENT enabled:
                           - If data URL: Parse, upload to OWUI storage, set file_url
                           - If remote URL: Download, upload to OWUI storage, set file_url
                        3. If file_id: Keep as-is (already in OWUI storage)
                        4. If file_url provided AND SAVE_REMOTE_FILE_URLS enabled:
                           - If data URL: Parse, upload to OWUI storage
                           - If remote URL: Download, upload to OWUI storage
                        5. Return all available fields to Responses API

                    Note:
                        All errors are caught and logged with status emissions.
                        Failed processing returns minimal valid block rather than crashing.
                        Size limits follow BASE64_MAX_SIZE_MB (default 50MB) for inline payloads.
                    """
                    try:
                        result = {"type": "input_file"}

                        # Check for nested "file" object (Chat Completions format)
                        nested_file = block.get("file")
                        source = nested_file if isinstance(nested_file, dict) else block

                        # Extract all available fields
                        file_id = source.get("file_id")
                        file_data = source.get("file_data")
                        filename = source.get("filename")
                        file_url = source.get("file_url")

                        def _is_internal_storage(url: str) -> bool:
                            """Return True when a URL already references internal storage."""
                            return isinstance(url, str) and ("/api/v1/files/" in url or "/files/" in url)

                        storage_context: Optional[Tuple[Optional[Request], Optional[Any]]] = None

                        async def _get_storage_context() -> tuple[Optional[Request], Optional[Any]]:
                            """Lazy-load the request/user pair used for uploads."""
                            nonlocal storage_context
                            if storage_context is None:
                                storage_context = await self._resolve_storage_context(__request__, user_obj)
                            return storage_context

                        async def _save_bytes_to_storage(
                            payload: bytes,
                            mime_type: str,
                            *,
                            preferred_name: Optional[str],
                            status_message: str,
                        ) -> Optional[str]:
                            """Persist arbitrary bytes to Open WebUI storage and emit status."""
                            upload_request, upload_user = await _get_storage_context()
                            if not (upload_request and upload_user):
                                return None
                            safe_mime = mime_type or "application/octet-stream"
                            fname = preferred_name or filename or f"file-{uuid.uuid4().hex}"
                            if "." not in fname and safe_mime:
                                ext = safe_mime.split("/")[-1]
                                fname = f"{fname}.{ext}"

                            internal_url = await self._upload_to_owui_storage(
                                request=upload_request,
                                user=upload_user,
                                file_data=payload,
                                filename=fname,
                                mime_type=safe_mime,
                            )
                            if internal_url:
                                await self._emit_status(
                                    event_emitter,
                                    status_message,
                                    done=False,
                                )
                            return internal_url

                        async def _download_and_store(
                            remote_url: str,
                            *,
                            name_hint: Optional[str] = None,
                        ) -> Optional[str]:
                            """Download a remote file and persist it via `_save_bytes_to_storage`."""
                            downloaded = await self._download_remote_url(remote_url)
                            if not downloaded:
                                return None
                            derived_name = (
                                name_hint
                                or remote_url.split("/")[-1].split("?")[0]
                                or f"file-{uuid.uuid4().hex}"
                            )
                            return await _save_bytes_to_storage(
                                downloaded["data"],
                                downloaded.get("mime_type") or "application/octet-stream",
                                preferred_name=derived_name,
                                status_message=StatusMessages.FILE_REMOTE_SAVED,
                            )

                        # Process file_data if it's a data URL or remote URL
                        if (
                            file_data
                            and isinstance(file_data, str)
                            and self.valves.SAVE_FILE_DATA_CONTENT
                        ):
                            # Handle data URL (base64)
                            if file_data.startswith("data:"):
                                try:
                                    parsed = self._parse_data_url(file_data)
                                    if parsed:
                                        fname = filename or f"file-{uuid.uuid4().hex}"
                                        internal_url = await _save_bytes_to_storage(
                                            parsed["data"],
                                            parsed["mime_type"],
                                            preferred_name=fname,
                                            status_message=StatusMessages.FILE_BASE64_SAVED,
                                        )
                                        if internal_url:
                                            file_url = internal_url
                                            file_data = None  # Clear base64, use URL instead
                                except Exception as exc:
                                    self.logger.error(f"Failed to process base64 file: {exc}")
                                    await self._emit_error(
                                        event_emitter,
                                        f"Failed to save base64 file: {exc}",
                                        show_error_message=False
                                    )

                            # Handle remote URL (not OWUI file reference)
                            elif file_data.startswith(("http://", "https://")) and not _is_internal_storage(file_data):
                                try:
                                    fname = filename or file_data.split("/")[-1].split("?")[0]
                                    internal_url = await _download_and_store(file_data, name_hint=fname)
                                    if internal_url:
                                        file_url = internal_url
                                        file_data = None  # Clear, use URL instead
                                except Exception as exc:
                                    self.logger.error(f"Failed to download remote file: {exc}")
                                    await self._emit_error(
                                        event_emitter,
                                        f"Failed to download file: {exc}",
                                        show_error_message=False
                                    )

                        # Process file_url when it's provided and valve permits re-hosting
                        if (
                            file_url
                            and isinstance(file_url, str)
                            and self.valves.SAVE_REMOTE_FILE_URLS
                        ):
                            if file_url.startswith("data:"):
                                try:
                                    parsed = self._parse_data_url(file_url)
                                    if parsed:
                                        fname = filename or f"file-{uuid.uuid4().hex}"
                                        internal_url = await _save_bytes_to_storage(
                                            parsed["data"],
                                            parsed["mime_type"],
                                            preferred_name=fname,
                                            status_message=StatusMessages.FILE_BASE64_SAVED,
                                        )
                                        if internal_url:
                                            file_url = internal_url
                                except Exception as exc:
                                    self.logger.error(f"Failed to process base64 file_url: {exc}")
                                    await self._emit_error(
                                        event_emitter,
                                        f"Failed to save base64 file URL: {exc}",
                                        show_error_message=False
                                    )
                            elif file_url.startswith(("http://", "https://")) and not _is_internal_storage(file_url):
                                try:
                                    name_hint = filename or file_url.split("/")[-1].split("?")[0]
                                    internal_url = await _download_and_store(file_url, name_hint=name_hint)
                                    if internal_url:
                                        file_url = internal_url
                                except Exception as exc:
                                    self.logger.error(f"Failed to download remote file_url: {exc}")
                                    await self._emit_error(
                                        event_emitter,
                                        f"Failed to download file URL: {exc}",
                                        show_error_message=False
                                    )

                        # Build result with all available fields (per Responses API spec)
                        if file_id:
                            result["file_id"] = file_id
                        if file_data:
                            result["file_data"] = file_data
                        if filename:
                            result["filename"] = filename
                        if file_url:
                            result["file_url"] = file_url

                        return result

                    except Exception as exc:
                        self.logger.error(f"Error in _to_input_file: {exc}")
                        await self._emit_error(
                            event_emitter,
                            f"File processing error: {exc}",
                            show_error_message=False
                        )
                        return {"type": "input_file"}

                async def _to_input_audio(block: dict) -> dict:
                    """Convert Open WebUI audio blocks into Responses API format.

                    Handles audio content blocks, transforming various input formats into
                    the Responses API audio input format.

                    Responses API Audio Input Format (per OpenAPI spec):
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": "<base64_audio_data>",
                                "format": "mp3" | "wav"
                            }
                        }

                    OpenRouter Audio Requirements (per documentation):
                        - Audio must be base64-encoded (URLs NOT supported)
                        - Supported formats: wav, mp3 only
                        - See: https://openrouter.ai/docs/guides/overview/multimodal/audio

                    Input Formats Handled:
                        1. Chat Completions: {"type": "input_audio", "input_audio": "<base64>"}
                        2. Tool output: {"type": "audio", "mimeType": "audio/mp3", "data": "<base64>"}
                        3. Already correct: {"type": "input_audio", "input_audio": {"data": "...", "format": "..."}}

                    Args:
                        block: Content block from Open WebUI message

                    Returns:
                        Responses API input_audio block with data and format

                    MIME Type to Format Mapping:
                        - audio/mpeg, audio/mp3 → "mp3"
                        - audio/wav, audio/wave, audio/x-wav → "wav"
                        - Unknown types default to "mp3"

                    Note:
                        All errors are caught and logged with status emissions.
                        Failed processing returns minimal valid block rather than crashing.
                    """
                    format_map = {
                        "audio/mpeg": "mp3",
                        "audio/mp3": "mp3",
                        "audio/wav": "wav",
                        "audio/wave": "wav",
                        "audio/x-wav": "wav",
                    }
                    supported_formats = {"mp3", "wav"}

                    def _map_format(mime: Optional[str]) -> str:
                        if not isinstance(mime, str):
                            return "mp3"
                        return format_map.get(mime.lower(), "mp3")

                    def _empty_audio_block() -> dict[str, Any]:
                        return {
                            "type": "input_audio",
                            "input_audio": {
                                "data": "",
                                "format": "mp3",
                            },
                        }

                    def _normalize_base64(data: str) -> Optional[str]:
                        if not data:
                            return None
                        cleaned = "".join(data.split())
                        if not cleaned:
                            return None
                        if not self._validate_base64_size(cleaned):
                            return None
                        try:
                            base64.b64decode(cleaned, validate=True)
                        except (binascii.Error, ValueError):
                            return None
                        return cleaned

                    def _resolved_mime_hint(payload: Optional[dict[str, Any]] = None) -> Optional[str]:
                        candidates: list[Any] = []
                        if isinstance(payload, dict):
                            candidates.extend(
                                [
                                    payload.get("mimeType"),
                                    payload.get("mime_type"),
                                ]
                            )
                        candidates.extend(
                            [
                                block.get("mimeType"),
                                block.get("mime_type"),
                                block.get("contentType"),
                                block.get("content_type"),
                            ]
                        )
                        for value in candidates:
                            if isinstance(value, str):
                                stripped = value.strip()
                                if stripped:
                                    return stripped
                        return None

                    def _normalize_format(
                        explicit_format: Optional[str],
                        mime_hint: Optional[str],
                    ) -> str:
                        if isinstance(explicit_format, str):
                            normalized = explicit_format.strip().lower()
                            if normalized in supported_formats:
                                return normalized
                        return _map_format(mime_hint)

                    def _build_audio_block(data: str, audio_format: str) -> dict[str, Any]:
                        return {
                            "type": "input_audio",
                            "input_audio": {
                                "data": data,
                                "format": audio_format,
                            },
                        }

                    try:
                        audio_payload = block.get("input_audio") or block.get("data") or block.get("blob")

                        # If payload is already a dict with both data/format, validate and passthrough
                        if isinstance(audio_payload, dict) and "data" in audio_payload and "format" in audio_payload:
                            cleaned = _normalize_base64(audio_payload.get("data", ""))
                            if not cleaned:
                                self.logger.warning("Audio payload rejected: invalid base64 data.")
                                await self._emit_error(
                                    event_emitter,
                                    "Audio input was not valid base64.",
                                    show_error_message=False,
                                )
                                return _empty_audio_block()
                            audio_format = _normalize_format(audio_payload.get("format"), _resolved_mime_hint(audio_payload))
                            return _build_audio_block(cleaned, audio_format)

                        # Payload dict without format but containing data (tool outputs)
                        if isinstance(audio_payload, dict):
                            raw_data = audio_payload.get("data")
                            if isinstance(raw_data, str):
                                cleaned = _normalize_base64(raw_data)
                                if not cleaned:
                                    self.logger.warning("Audio payload rejected: invalid base64 data.")
                                    await self._emit_error(
                                        event_emitter,
                                        "Audio input was not valid base64.",
                                        show_error_message=False,
                                    )
                                    return _empty_audio_block()
                                mime_hint = _resolved_mime_hint(audio_payload)
                                audio_format = _normalize_format(audio_payload.get("format"), mime_hint)
                                return _build_audio_block(cleaned, audio_format)

                        # Need to convert to Responses format for string payloads
                        if isinstance(audio_payload, str):
                            sanitized = audio_payload.strip()
                            lowercase = sanitized.lower()
                            if lowercase.startswith(("http://", "https://")):
                                self.logger.warning("Audio payload rejected: remote URLs are not supported.")
                                await self._emit_error(
                                    event_emitter,
                                    "Audio input must be base64-encoded. URLs are not supported.",
                                    show_error_message=False,
                                )
                                return _empty_audio_block()

                            # Handle data URLs
                            if lowercase.startswith("data:"):
                                parsed = self._parse_data_url(sanitized if sanitized.startswith("data:") else f"data:{sanitized.split(':', 1)[1]}")
                                if not parsed or not parsed.get("mime_type", "").startswith("audio/"):
                                    self.logger.warning("Audio payload rejected: invalid data URL.")
                                    await self._emit_error(
                                        event_emitter,
                                        "Audio input must be base64-encoded audio data.",
                                        show_error_message=False,
                                    )
                                    return _empty_audio_block()
                                if not self._validate_base64_size(parsed.get("b64", "")):
                                    return _empty_audio_block()
                                audio_format = _map_format(parsed.get("mime_type"))
                                return _build_audio_block(parsed.get("b64", ""), audio_format)

                            cleaned = _normalize_base64(sanitized)
                            if not cleaned:
                                self.logger.warning("Audio payload rejected: invalid base64 data.")
                                await self._emit_error(
                                    event_emitter,
                                    "Audio input was not valid base64.",
                                    show_error_message=False,
                                )
                                return _empty_audio_block()

                            mime_type = _resolved_mime_hint()
                            audio_format = _map_format(mime_type)
                            return _build_audio_block(cleaned, audio_format)

                        # Invalid/empty
                        self.logger.warning("Invalid audio payload format, returning empty audio block")
                        return _empty_audio_block()

                    except Exception as exc:
                        self.logger.error(f"Error in _to_input_audio: {exc}")
                        await self._emit_error(
                            event_emitter,
                            f"Audio processing error: {exc}",
                            show_error_message=False,
                        )
                        return _empty_audio_block()

                async def _to_input_video(block: dict) -> dict:
                    """Convert Open WebUI video blocks into Chat Completions video format.

                    Note: The Responses API doesn't have explicit `input_video` type.
                    Videos use the Chat Completions `video_url` format, which OpenRouter
                    handles internally.

                    Video Support by Provider (per OpenRouter docs):
                        - Gemini AI Studio: YouTube links only
                        - Most providers: Limited or no video support
                        - Check model's input_modalities for "video" capability

                    Supported Video Formats:
                        - Remote URLs: YouTube links, direct video URLs
                        - Data URLs: data:video/mp4;base64,... (rarely used due to size)
                        - OWUI file references: /api/v1/files/...

                    Chat Completions Video Format:
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": "https://youtube.com/..." or "data:video/mp4;base64,..."
                            }
                        }

                    Input Formats Handled:
                        1. {"type": "video_url", "video_url": {"url": "..."}}
                        2. {"type": "video_url", "video_url": "..."}
                        3. {"type": "video", "url": "...", "mimeType": "video/mp4"}

                    Args:
                        block: Content block from Open WebUI message

                    Returns:
                        Chat Completions video_url block

                    Note:
                        Videos are NOT downloaded/stored due to large size.
                        URL validation is minimal - OpenRouter will validate provider support.
                    """
                    try:
                        video_payload = block.get("video_url")
                        url: str = ""

                        # Handle nested video_url object
                        if isinstance(video_payload, dict):
                            url = video_payload.get("url", "")
                        elif isinstance(video_payload, str):
                            url = video_payload

                        # Fallback: check for direct URL in block
                        if not url:
                            url = block.get("url", "")

                        if not url:
                            self.logger.warning("Video block has no URL")
                            return {"type": "video_url", "video_url": {"url": ""}}

                        # Validate data URL size for base64 videos
                        if url.startswith("data:"):
                            # Estimate base64 size before processing
                            if "," in url:
                                b64_data = url.split(",", 1)[1]
                                # Estimate decoded size (base64 is ~33% larger than raw)
                                estimated_size_bytes = (len(b64_data) * 3) // 4
                                max_size_bytes = self.valves.VIDEO_MAX_SIZE_MB * 1024 * 1024
                                if estimated_size_bytes > max_size_bytes:
                                    estimated_size_mb = estimated_size_bytes / (1024 * 1024)
                                    self.logger.warning(
                                        f"Base64 video size (~{estimated_size_mb:.1f}MB) exceeds configured limit "
                                        f"({self.valves.VIDEO_MAX_SIZE_MB}MB), rejecting to prevent memory issues"
                                    )
                                    await self._emit_error(
                                        event_emitter,
                                        f"Video too large (~{estimated_size_mb:.1f}MB, max: {self.valves.VIDEO_MAX_SIZE_MB}MB)",
                                        show_error_message=True
                                    )
                                    return {"type": "video_url", "video_url": {"url": ""}}

                            await self._emit_status(
                                event_emitter,
                                StatusMessages.VIDEO_BASE64,
                                done=False
                            )
                        # Validate YouTube URLs
                        elif self._is_youtube_url(url):
                            # Note: YouTube videos only work with Gemini models (per OpenRouter docs)
                            await self._emit_status(
                                event_emitter,
                                StatusMessages.VIDEO_YOUTUBE,
                                done=False
                            )
                        # Validate remote URLs (SSRF protection)
                        elif url.startswith(("http://", "https://")) and not ("/api/v1/files/" in url or "/files/" in url):
                            # Apply SSRF protection for non-OWUI URLs
                            if not await self._is_safe_url(url):
                                self.logger.error(f"SSRF protection blocked video URL: {url}")
                                await self._emit_error(
                                    event_emitter,
                                    "Video URL blocked by security policy (private network)",
                                    show_error_message=True
                                )
                                return {"type": "video_url", "video_url": {"url": ""}}

                            await self._emit_status(
                                event_emitter,
                                StatusMessages.VIDEO_REMOTE,
                                done=False
                            )
                        else:
                            # OWUI file reference or other format
                            await self._emit_status(
                                event_emitter,
                                StatusMessages.VIDEO_REMOTE,
                                done=False
                            )

                        return {
                            "type": "video_url",
                            "video_url": {
                                "url": url
                            }
                        }

                    except Exception as exc:
                        self.logger.error(f"Error in _to_input_video: {exc}")
                        await self._emit_error(
                            event_emitter,
                            f"Video processing error: {exc}",
                            show_error_message=False
                        )
                        return {"type": "video_url", "video_url": {"url": ""}}

                # Block transformers (async functions for image/file processing)
                def _identity_block(b: dict[str, Any]) -> dict[str, Any]:
                    return b

                block_transform = {
                    "text":       lambda b: {"type": "input_text",  "text": b.get("text", "")},
                    "image_url":  _to_input_image,
                    "input_file": _to_input_file,
                    "file":       _to_input_file,  # Chat Completions format
                    "input_audio": _to_input_audio,  # Responses API audio format
                    "audio":      _to_input_audio,   # Open WebUI tool audio output format
                    "video_url":  _to_input_video,   # Chat Completions video format
                    "video":      _to_input_video,   # Alternative video format
                }

                # Process blocks (handle async transformers)
                converted_blocks: list[dict[str, Any]] = []
                user_images_used = 0
                dropped_images = 0
                encountered_user_images = False
                vision_warning_sent = False
                latest_user_message = role == "user" and idx == len(messages) - 1
                include_user_images = (
                    latest_user_message and vision_supported and image_limit > 0
                )

                for block in content_blocks:
                    if not block:
                        continue
                    raw_block_type = block.get("type")
                    block_type = raw_block_type if isinstance(raw_block_type, str) else ""
                    transformer = block_transform.get(block_type, _identity_block)
                    is_image_block = block_type in {"image_url", "input_image"}

                    if is_image_block:
                        encountered_user_images = True
                        if not include_user_images:
                            if latest_user_message and not vision_supported and not vision_warning_sent:
                                await self._emit_status(
                                    event_emitter,
                                    "Model does not accept image inputs; skipping user attachments.",
                                    done=False,
                                )
                                vision_warning_sent = True
                            continue
                        if user_images_used >= image_limit:
                            dropped_images += 1
                            continue

                    try:
                        if asyncio.iscoroutinefunction(transformer):
                            result = await transformer(block)
                        else:
                            result = transformer(block)
                        if is_image_block and result:
                            user_images_used += 1
                        converted_blocks.append(result)
                    except Exception as exc:
                        self.logger.error(f"Failed to transform block type '{block_type}': {exc}")
                        await self._emit_error(
                            event_emitter,
                            f"Block transformation error for '{block_type}': {exc}",
                            show_error_message=False
                        )
                        if not is_image_block:
                            converted_blocks.append(block)

                if (
                    latest_user_message
                    and selection_mode == "user_then_assistant"
                    and include_user_images
                    and user_images_used == 0
                    and last_assistant_images
                ):
                    fallback_slots = min(image_limit, len(last_assistant_images))
                    fallback_blocks: list[dict[str, Any]] = []
                    for source_block in last_assistant_images[:fallback_slots]:
                        try:
                            transformed = await _to_input_image(source_block)
                            fallback_blocks.append(transformed)
                        except Exception as exc:
                            self.logger.error("Failed to reuse assistant image: %s", exc)
                    if fallback_blocks:
                        self.logger.debug(
                            "Rehydrating %d assistant-generated image(s) due to empty user attachments (selection_mode=%s, limit=%d).",
                            len(fallback_blocks),
                            selection_mode,
                            image_limit,
                        )
                        converted_blocks = fallback_blocks + converted_blocks
                        user_images_used = len(fallback_blocks)

                if dropped_images and latest_user_message:
                    await self._emit_status(
                        event_emitter,
                        f"Dropped {dropped_images} extra image{'s' if dropped_images != 1 else ''}; limit is {image_limit}.",
                        done=False,
                    )
                if (
                    latest_user_message
                    and encountered_user_images
                    and not vision_supported
                    and not vision_warning_sent
                ):
                    await self._emit_status(
                        event_emitter,
                        "Model does not accept image inputs; skipping user attachments.",
                        done=False,
                    )

                openai_input.append({
                    "type": "message",
                    "role": "user",
                    "content": converted_blocks,
                })
                continue

            # -------- assistant message ----------------------------------- #
            assistant_text = (
                raw_content
                if isinstance(raw_content, str)
                else _extract_plain_text_content(raw_content)
            )
            is_old_message = _is_old_turn(msg_turn_index, threshold=prune_before_turn)
            assistant_image_urls = _markdown_images_from_text(assistant_text)
            if assistant_image_urls:
                last_assistant_images = [
                    {"type": "image_url", "image_url": url, "detail": "auto"}
                    for url in assistant_image_urls
                ]
            else:
                last_assistant_images = []

            if contains_marker(assistant_text):
                segments = split_text_by_markers(assistant_text)
                markers = [seg["marker"] for seg in segments if seg.get("type") == "marker"]

                db_artifacts: dict[str, dict] = {}
                orphaned_call_ids: set[str] = set()
                orphaned_output_ids: set[str] = set()
                if artifact_loader and chat_id and openwebui_model_id and markers:
                    try:
                        db_artifacts = await artifact_loader(chat_id, msg_id, markers)
                        (
                            _,
                            orphaned_call_ids,
                            orphaned_output_ids,
                        ) = _classify_function_call_artifacts(db_artifacts)
                        if orphaned_call_ids:
                            logger.warning(
                                "Dropping %d persisted function_call artifact(s) missing outputs (chat_id=%s message_id=%s call_ids=%s)",
                                len(orphaned_call_ids),
                                chat_id,
                                msg_id,
                                sorted(orphaned_call_ids),
                            )
                        if orphaned_output_ids:
                            logger.warning(
                                "Dropping %d persisted function_call_output artifact(s) missing calls (chat_id=%s message_id=%s call_ids=%s)",
                                len(orphaned_output_ids),
                                chat_id,
                                msg_id,
                                sorted(orphaned_output_ids),
                            )
                    except Exception:
                        # Catch all loader errors (DB, network, etc.) - pipe must continue
                        logger.warning("Artifact loader failed for chat_id=%s message_id=%s", chat_id, msg_id, exc_info=True)
                        db_artifacts = {}

                for segment in segments:
                    if segment["type"] == "marker":
                        payload = db_artifacts.get(segment["marker"])
                        if payload is None:
                            logger.warning("Missing artifact %s for chat_id=%s message_id=%s", segment["marker"], chat_id, msg_id)
                            continue
                        if (
                            payload.get("type") == "reasoning"
                            and replayed_reasoning_refs is not None
                            and chat_id
                        ):
                            replayed_reasoning_refs.append((chat_id, segment["marker"]))
                        item = _normalize_persisted_item(payload)
                        if item is not None:
                            item_type = ((item.get("type") or "").lower())
                            if item_type in _NON_REPLAYABLE_TOOL_ARTIFACTS:
                                self.logger.debug(
                                    "Skipping %s artifact when rebuilding provider context (not replayable).",
                                    item_type,
                                )
                                continue
                            if (
                                item_type == "function_call"
                                and item.get("call_id") in orphaned_call_ids
                            ):
                                logger.debug(
                                    "Skipping orphaned function_call artifact (call_id=%s chat_id=%s message_id=%s)",
                                    item.get("call_id"),
                                    chat_id,
                                    msg_id,
                                )
                                continue
                            if (
                                item_type == "function_call_output"
                                and item.get("call_id") in orphaned_output_ids
                            ):
                                logger.debug(
                                    "Skipping orphaned function_call_output artifact (call_id=%s chat_id=%s message_id=%s)",
                                    item.get("call_id"),
                                    chat_id,
                                    msg_id,
                                )
                                continue
                            if (
                                is_old_message
                                and pruning_turns > 0
                                and prune_before_turn is not None
                            ):
                                _prune_tool_output(
                                    item,
                                    marker=segment["marker"],
                                    turn_index=msg_turn_index,
                                    retention_turns=pruning_turns,
                                )
                            openai_input.append(item)
                    elif segment["type"] == "text" and segment["text"].strip():
                        openai_input.append({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": segment["text"].strip()}]
                        })
            else:
                # Plain assistant text (no markers detected)
                if assistant_text:
                    openai_input.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": assistant_text}],
                        }
                    )

        return openai_input

    def _should_retry_without_reasoning(
        self,
        error: OpenRouterAPIError,
        responses_body: ResponsesBody,
    ) -> bool:
        """Return True when we can retry the request after disabling reasoning."""

        include_flag = getattr(responses_body, "include_reasoning", None)
        has_reasoning_dict = bool(getattr(responses_body, "reasoning", None))
        has_thinking_config = bool(getattr(responses_body, "thinking_config", None))
        if not any((include_flag, has_reasoning_dict, has_thinking_config)):
            return False

        trigger_phrases = (
            "thinking_config.include_thoughts is only enabled when thinking is enabled",
            "include_thoughts is only enabled when thinking is enabled",
        )
        message_candidates = [
            error.upstream_message,
            error.openrouter_message,
            str(error),
        ]

        for message in message_candidates:
            if not isinstance(message, str):
                continue
            lowered = message.lower()
            if any(trigger in lowered for trigger in trigger_phrases):
                setattr(responses_body, "include_reasoning", False)
                responses_body.reasoning = None
                responses_body.thinking_config = None
                self.logger.info(
                    "Retrying without reasoning for model '%s' after provider rejected include_reasoning without thinking.",
                    responses_body.model,
                )
                return True

        return False


    def _select_models(self, filter_value: str, available_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter OpenRouter catalog entries based on the valve string."""
        if not available_models:
            return []

        filter_value = (filter_value or "").strip()
        if not filter_value or filter_value.lower() == "auto":
            return available_models

        requested = {
            ModelFamily.base_model(sanitize_model_id(model_id.strip()))
            for model_id in filter_value.split(",")
            if model_id.strip()
        }
        if not requested:
            return available_models

        selected = [model for model in available_models if model["norm_id"] in requested]
        missing = requested - {model["norm_id"] for model in selected}
        if missing:
            self.logger.warning("Requested models not found in OpenRouter catalog: %s", ", ".join(sorted(missing)))
        return selected or available_models

    # 4.3 Core Multi-Turn Handlers
    @no_type_check
    async def _run_streaming_loop(  # pyright: ignore[reportGeneralTypeIssues]
        self,
        body: ResponsesBody,
        valves: Pipe.Valves,
        event_emitter: EventEmitter | None,
        metadata: dict[str, Any] = {},
        tools: dict[str, Dict[str, Any]] | list[dict[str, Any]] | None = None,
        session: aiohttp.ClientSession | None = None,
        user_id: str = "",
        *,
        request_context: Optional[Request] = None,
        user_obj: Optional[Any] = None,
        pipe_identifier: Optional[str] = None,
    ):
        """
        Stream assistant responses incrementally, handling function calls, status updates, and tool usage.
        """
        if session is None:
            raise RuntimeError("HTTP session is required for streaming")

        if not isinstance(metadata, dict):
            metadata = {}

        self.logger.debug("🔧 PERSIST_TOOL_RESULTS=%s", valves.PERSIST_TOOL_RESULTS)
        self.logger.debug("Streaming config: direct pass-through (no server batching)")

        raw_tools = tools or {}
        tool_registry: dict[str, dict[str, Any]] = {}
        if isinstance(raw_tools, dict):
            tool_registry = raw_tools
        elif isinstance(raw_tools, list):
            skipped_no_callable = False
            for entry in raw_tools:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not name and isinstance(entry.get("spec"), dict):
                    name = entry["spec"].get("name")
                callable_obj = entry.get("callable")
                if callable_obj is None:
                    skipped_no_callable = True
                    continue
                if name:
                    tool_registry[name] = entry
            if skipped_no_callable and not tool_registry:
                self.logger.warning("Received list-based tools without callables; tool execution will be disabled for this request.")
        model_block = metadata.get("model")
        openwebui_model = model_block.get("id", "") if isinstance(model_block, dict) else ""
        assistant_message = ""
        pending_ulids: list[str] = []
        pending_items: list[dict[str, Any]] = []
        total_usage: dict[str, Any] = {}
        reasoning_buffer = ""
        reasoning_completed_emitted = False
        reasoning_stream_active = False
        ordinal_by_url: dict[str, int] = {}
        emitted_citations: list[dict] = []
        chat_id = metadata.get("chat_id")
        message_id = metadata.get("message_id")
        model_started = asyncio.Event()
        responding_status_sent = False
        provider_status_seen = False
        generation_started_at: float | None = None
        generation_last_event_at: float | None = None
        response_completed_at: float | None = None
        stream_started_at: float | None = None
        surrogate_carry: dict[str, str] = {"assistant": "", "reasoning": ""}
        storage_context_cache: Optional[tuple[Optional[Request], Optional[Any]]] = None
        processed_image_item_ids: set[str] = set()
        generated_image_count = 0
        reasoning_status_buffer = ""
        reasoning_status_last_emit: float | None = None
        thinking_mode = valves.THINKING_OUTPUT_MODE
        thinking_box_enabled = thinking_mode in {"open_webui", "both"}
        thinking_status_enabled = thinking_mode in {"status", "both"}

        async def _maybe_emit_reasoning_status(delta_text: str, *, force: bool = False) -> None:
            """Emit readable status updates for reasoning text without flooding."""
            nonlocal reasoning_status_buffer, reasoning_status_last_emit
            if not thinking_status_enabled:
                return
            if not event_emitter:
                return
            reasoning_status_buffer += delta_text
            text = reasoning_status_buffer.strip()
            if not text:
                return
            should_emit = force
            now = perf_counter()
            if not should_emit:
                if delta_text.rstrip().endswith(_REASONING_STATUS_PUNCTUATION):
                    should_emit = True
                elif len(text) >= _REASONING_STATUS_MAX_CHARS:
                    should_emit = True
                else:
                    elapsed = None if reasoning_status_last_emit is None else (now - reasoning_status_last_emit)
                    if len(text) >= _REASONING_STATUS_MIN_CHARS:
                        if elapsed is None or elapsed >= _REASONING_STATUS_IDLE_SECONDS:
                            should_emit = True
            if not should_emit:
                return
            await event_emitter(
                {
                    "type": "status",
                    "data": {"description": text, "done": False},
                }
            )
            reasoning_status_buffer = ""
            reasoning_status_last_emit = now

        async def _get_storage_context() -> tuple[Optional[Request], Optional[Any]]:
            nonlocal storage_context_cache
            if storage_context_cache is None:
                storage_context_cache = await self._resolve_storage_context(request_context, user_obj)
            return storage_context_cache or (None, None)

        async def _persist_generated_image(data: bytes, mime_type: str) -> Optional[str]:
            upload_request, upload_user = await _get_storage_context()
            if not upload_request or not upload_user:
                return None
            ext = "png"
            if isinstance(mime_type, str) and "/" in mime_type:
                ext = mime_type.split("/")[-1] or "png"
            if ext == "jpg":
                ext = "jpeg"
            filename = f"generated-image-{uuid.uuid4().hex}.{ext}"
            return await self._upload_to_owui_storage(
                request=upload_request,
                user=upload_user,
                file_data=data,
                filename=filename,
                mime_type=mime_type,
            )

        async def _materialize_image_from_str(data_str: str) -> Optional[str]:
            text = (data_str or "").strip()
            if not text:
                return None
            if text.startswith("data:"):
                parsed = self._parse_data_url(text)
                if parsed:
                    stored = await _persist_generated_image(parsed["data"], parsed["mime_type"])
                    if stored:
                        await self._emit_status(event_emitter, StatusMessages.IMAGE_BASE64_SAVED, done=False)
                        return stored
                    return text
                return None
            if text.startswith(("http://", "https://", "/")):
                return text
            cleaned = text
            if "," in cleaned and ";base64" in cleaned.split(",", 1)[0]:
                cleaned = cleaned.split(",", 1)[1]
            cleaned = cleaned.strip()
            if not cleaned:
                return None
            if not self._validate_base64_size(cleaned):
                return None
            try:
                decoded = base64.b64decode(cleaned, validate=True)
            except (binascii.Error, ValueError):
                return None
            mime_type = "image/png"
            stored = await _persist_generated_image(decoded, mime_type)
            if stored:
                await self._emit_status(event_emitter, StatusMessages.IMAGE_BASE64_SAVED, done=False)
                return stored
            return f"data:{mime_type};base64,{cleaned}"

        async def _materialize_image_entry(entry: Any) -> Optional[str]:
            if entry is None:
                return None
            if isinstance(entry, str):
                return await _materialize_image_from_str(entry)
            if isinstance(entry, dict):
                for key in ("url", "image_url", "imageUrl", "content_url"):
                    candidate = entry.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
                    if isinstance(candidate, dict):
                        nested = await _materialize_image_entry(candidate)
                        if nested:
                            return nested
                for key in ("b64_json", "b64", "base64", "data", "image_base64"):
                    b64_val = entry.get(key)
                    if isinstance(b64_val, str) and b64_val.strip():
                        cleaned = b64_val.strip()
                        if not self._validate_base64_size(cleaned):
                            continue
                        try:
                            decoded = base64.b64decode(cleaned, validate=True)
                        except (binascii.Error, ValueError):
                            continue
                        mime_type = (
                            entry.get("mime_type")
                            or entry.get("mimeType")
                            or "image/png"
                        ).lower()
                        if mime_type == "image/jpg":
                            mime_type = "image/jpeg"
                        stored = await _persist_generated_image(decoded, mime_type)
                        if stored:
                            await self._emit_status(event_emitter, StatusMessages.IMAGE_BASE64_SAVED, done=False)
                            return stored
                        return f"data:{mime_type};base64,{cleaned}"
                nested_result = entry.get("result")
                if nested_result is not None:
                    return await _materialize_image_entry(nested_result)
            return None

        async def _collect_image_output_urls(item: dict[str, Any]) -> list[str]:
            payload = item.get("result")
            urls: list[str] = []
            if isinstance(payload, list):
                for entry in payload:
                    resolved = await _materialize_image_entry(entry)
                    if resolved:
                        urls.append(resolved)
            else:
                resolved = await _materialize_image_entry(payload)
                if resolved:
                    urls.append(resolved)
            return urls

        async def _render_image_markdown(item: dict[str, Any]) -> list[str]:
            nonlocal generated_image_count
            urls = await _collect_image_output_urls(item)
            markdowns: list[str] = []
            for url in urls:
                generated_image_count += 1
                label = item.get("label") or f"Generated image {generated_image_count}"
                alt_text = re.sub(r"[\r\n]+", " ", str(label)).strip() or f"Generated image {generated_image_count}"
                markdowns.append(f"![{alt_text}]({url})")
            return markdowns

        def _append_output_block(current: str, block: str) -> str:
            snippet = (block or "").strip()
            if not snippet:
                return current
            if current:
                if not current.endswith("\n"):
                    current += "\n"
                if not current.endswith("\n\n"):
                    current += "\n"
            return f"{current}{snippet}\n"

        def _normalize_surrogate_chunk(text: str, bucket: str) -> str:
            """Coalesce surrogate pairs in streaming chunks to keep UTF-8 happy."""
            prev = surrogate_carry.get(bucket, "")
            combined = f"{prev}{text or ''}"
            if not combined:
                surrogate_carry[bucket] = ""
                return ""
            new_carry = ""
            try:
                normalized = combined.encode("utf-16", "surrogatepass").decode("utf-16")
            except UnicodeDecodeError:
                if combined:
                    last_char = combined[-1]
                    if 0xD800 <= ord(last_char) <= 0xDBFF:
                        new_carry = last_char
                        combined = combined[:-1]
                normalized = combined.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
            surrogate_carry[bucket] = new_carry
            return normalized

        def _extract_reasoning_text(event: dict[str, Any]) -> str:
            """Return best-effort reasoning text from assorted event payloads."""
            if not isinstance(event, dict):
                return ""
            for key in ("delta", "text"):
                value = event.get(key)
                if isinstance(value, str) and value:
                    return value
            part = event.get("part")
            if isinstance(part, dict):
                part_text = part.get("text")
                if isinstance(part_text, str) and part_text:
                    return part_text
                content = part.get("content")
                if isinstance(content, list):
                    fragments: list[str] = []
                    for entry in content:
                        if isinstance(entry, dict):
                            text_val = entry.get("text")
                            if isinstance(text_val, str):
                                fragments.append(text_val)
                        elif isinstance(entry, str):
                            fragments.append(entry)
                    if fragments:
                        return "".join(fragments)
            return ""

        async def _flush_pending(reason: str) -> None:
            """Persist buffered artifacts and emit a warning when the DB fails."""
            if not pending_items:
                return
            rows = pending_items[:]
            # Clear immediately so a failure here does not cause the same rows
            # to be re-attempted during later flushes. Callers report the error
            # to the UI instead of silently retrying.
            pending_items.clear()
            try:
                ulids = await self._db_persist(rows)
            except Exception as exc:  # pragma: no cover - DB errors handled later
                self.logger.error("Failed to persist response artifacts (%s): %s", reason, exc, exc_info=self.logger.isEnabledFor(logging.DEBUG))
                if event_emitter:
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {"description": "⚠️ Tool storage unavailable", "done": False},
                        }
                    )
                return
            pending_ulids.extend(ulids)

        thinking_tasks: list[asyncio.Task] = []
        thinking_cancelled = False
        if event_emitter:
            async def _later(delay: float, msg: str) -> None:
                """Emit a delayed status update to reassure the user during long thoughts."""
                try:
                    await asyncio.wait_for(model_started.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    if model_started.is_set():
                        return
                await event_emitter({"type": "status", "data": {"description": msg}})

            thinking_tasks = []
            for delay, msg in [
                (0, "Thinking…"),
                (1.5, "Reading the user's question…"),
                (4.0, "Gathering my thoughts…"),
                (6.0, "Exploring possible responses…"),
                (7.0, "Building a plan…"),
            ]:
                if delay == 0:
                    await event_emitter({"type": "status", "data": {"description": msg}})
                    continue
                thinking_tasks.append(
                    asyncio.create_task(_later(delay + random.uniform(0, 0.5), msg))
                )

        def cancel_thinking() -> None:
            """Cancel any scheduled reasoning status updates once the loop completes."""
            nonlocal thinking_cancelled
            if thinking_cancelled:
                return
            thinking_cancelled = True
            for t in thinking_tasks:
                t.cancel()

        def note_model_activity() -> None:
            """Mark the stream as active and stop any pending thinking statuses."""
            if not model_started.is_set():
                model_started.set()
                cancel_thinking()

        def note_generation_activity() -> None:
            """Record when output tokens start/continue streaming."""
            nonlocal generation_started_at, generation_last_event_at
            now = perf_counter()
            generation_last_event_at = now
            if generation_started_at is None:
                generation_started_at = now

        request_started_at = perf_counter()

        # Send OpenAI Responses API request, parse and emit response
        error_occurred = False
        was_cancelled = False
        try:
            for _ in range(valves.MAX_FUNCTION_CALL_LOOPS):
                final_response: dict[str, Any] | None = None
                self._sanitize_request_input(body)
                request_payload = body.model_dump(exclude_none=True)
                api_model_override = getattr(body, "api_model", None)
                if api_model_override:
                    request_payload["model"] = api_model_override
                    request_payload.pop("api_model", None)
                _apply_identifier_valves_to_payload(
                    request_payload,
                    valves=valves,
                    owui_metadata=metadata,
                    owui_user_id=user_id,
                    logger=self.logger,
                )
                _apply_model_fallback_to_payload(request_payload, logger=self.logger)
                request_payload = _filter_openrouter_request(request_payload)

                api_key_value = EncryptedStr.decrypt(valves.API_KEY)
                async for event in self.send_openai_responses_streaming_request(
                    session,
                    request_payload,
                    api_key=api_key_value,
                    base_url=valves.BASE_URL,
                    valves=valves,
                    workers=valves.SSE_WORKERS_PER_REQUEST,
                    breaker_key=user_id or None,
                    delta_char_limit=0,
                    idle_flush_ms=0,
                    chunk_queue_maxsize=valves.STREAMING_CHUNK_QUEUE_MAXSIZE,
                    event_queue_maxsize=valves.STREAMING_EVENT_QUEUE_MAXSIZE,
                    event_queue_warn_size=valves.STREAMING_EVENT_QUEUE_WARN_SIZE,
                ):
                    if stream_started_at is None:
                        stream_started_at = perf_counter()
                    etype = event.get("type")
                    # Note: Don't call note_model_activity() here for ALL events.
                    # We only want to cancel thinking tasks when actual output or action starts,
                    # not during reasoning phases. Moved to specific event handlers below.

                    # Emit OpenRouter SSE frames at DEBUG (non-delta) only; skip delta spam entirely.
                    is_delta_event = bool(etype and etype.endswith(".delta"))
                    if not is_delta_event and self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug("OpenRouter payload: %s",json.dumps(event, indent=2, ensure_ascii=False))

                    if etype:
                        is_reasoning_event = (
                            etype.startswith("response.reasoning")
                            and etype != "response.reasoning_summary_text.done"
                        )
                        part = event.get("part") if isinstance(event, dict) else None
                        reasoning_part_types = {"reasoning_text", "reasoning_summary_text", "summary_text"}
                        is_reasoning_part_event = (
                            etype.startswith("response.content_part")
                            and isinstance(part, dict)
                            and part.get("type") in reasoning_part_types
                        )
                        if is_reasoning_event or is_reasoning_part_event:
                            reasoning_stream_active = True
                            # Stop "Thinking..." placeholders when reasoning starts
                            note_model_activity()

                            delta_text = _extract_reasoning_text(event)
                            normalized_delta = _normalize_surrogate_chunk(delta_text, "reasoning") if delta_text else ""
                            if normalized_delta:
                                note_generation_activity()
                                reasoning_buffer += normalized_delta

                                if event_emitter and thinking_box_enabled:
                                    await event_emitter(
                                        {
                                            "type": "reasoning:delta",
                                            "data": {
                                                "content": reasoning_buffer,
                                                "delta": normalized_delta,
                                                "event": etype,
                                            },
                                        }
                                    )

                                await _maybe_emit_reasoning_status(normalized_delta)
                            if etype.endswith(".done") or etype.endswith(".completed"):
                                if event_emitter:
                                    await _maybe_emit_reasoning_status("", force=True)

                                    if thinking_box_enabled:
                                        await event_emitter(
                                            {
                                                "type": "reasoning:completed",
                                                "data": {"content": reasoning_buffer},
                                            }
                                        )
                                reasoning_completed_emitted = True
                            continue

                    # ─── Emit partial delta assistant message
                    if etype == "response.output_text.delta":
                        note_model_activity()  # Cancel thinking tasks when actual output starts
                        delta = event.get("delta") or ""
                        normalized_delta = _normalize_surrogate_chunk(delta, "assistant") if delta else ""
                        if normalized_delta:
                            note_generation_activity()
                            if (
                                not provider_status_seen
                                and not responding_status_sent
                                and not reasoning_stream_active
                                and event_emitter
                            ):
                                provider_status_seen = True
                                responding_status_sent = True
                                await event_emitter(
                                    {
                                        "type": "status",
                                        "data": {"description": "Responding to the user…"},
                                    }
                                )
                            assistant_message += normalized_delta
                            await event_emitter(
                                {
                                    "type": "chat:message",
                                    "data": {
                                        "content": assistant_message,
                                        "delta": normalized_delta,
                                    },
                                }
                            )
                        continue

                    # ─── Emit reasoning summary once done ───────────────────────
                    if etype == "response.reasoning_summary_text.done":
                        text = (event.get("text") or "").strip()
                        if text:
                            title_match = re.findall(r"\*\*(.+?)\*\*", text)
                            title = title_match[-1].strip() if title_match else "Thinking…"
                            content = re.sub(r"\*\*(.+?)\*\*", "", text).strip()
                            summary = title if not content else f"{title}\n{content}"
                            if event_emitter:
                                note_model_activity()
                                if thinking_box_enabled and (not reasoning_buffer):
                                    normalized_summary = (
                                        _normalize_surrogate_chunk(summary, "reasoning") if summary else ""
                                    )
                                    if normalized_summary:
                                        note_generation_activity()
                                        reasoning_stream_active = True
                                        reasoning_buffer += normalized_summary
                                        await event_emitter(
                                            {
                                                "type": "reasoning:delta",
                                                "data": {
                                                    "content": reasoning_buffer,
                                                    "delta": normalized_summary,
                                                    "event": etype,
                                                },
                                            }
                                        )
                                    if not reasoning_completed_emitted:
                                        await event_emitter(
                                            {
                                                "type": "reasoning:completed",
                                                "data": {"content": reasoning_buffer},
                                            }
                                        )
                                        reasoning_completed_emitted = True
                                if thinking_status_enabled:
                                    cancel_thinking()
                                    await event_emitter(
                                        {
                                            "type": "status",
                                            "data": {"description": summary},
                                        }
                                    )
                        continue

                    # ─── Citations from inline annotations (emit metadata only) ───────────────
                    if etype == "response.output_text.annotation.added":
                        ann = event.get("annotation") or {}
                        if ann.get("type") == "url_citation":
                            # Basic fields
                            url = (ann.get("url") or "").strip()
                            if url.endswith("?utm_source=openai"):
                                url = url[: -len("?utm_source=openai")]
                            title = (ann.get("title") or url).strip()

                            if url in ordinal_by_url:
                                continue

                            ordinal_by_url[url] = len(ordinal_by_url) + 1

                            # Emit a citation event so Open WebUI can render
                            # references in the footer instead of mutating the
                            # streaming text (which was causing inline [n] markers).
                            host = url.split("//", 1)[-1].split("/", 1)[0].lower().lstrip("www.")
                            citation = {
                                "source": {"name": host or "source", "url": url},
                                "document": [title],
                                "metadata": [{
                                    "source": url,
                                    "date_accessed": datetime.date.today().isoformat(),
                                }],
                            }
                            await self._emit_citation(event_emitter, citation)
                            emitted_citations.append(citation)

                        continue


                    # ─── Emit status updates for in-progress items ──────────────────────
                    if etype == "response.output_item.added":
                        item_raw = event.get("item")
                        item = item_raw if isinstance(item_raw, dict) else {}
                        item_type = item.get("type", "")
                        item_status = item.get("status", "")

                        if item_type == "message" and item_status == "in_progress":
                            provider_status_seen = True
                            if (not responding_status_sent) and (not reasoning_stream_active) and event_emitter:
                                responding_status_sent = True
                                await event_emitter(
                                    {
                                        "type": "status",
                                        "data": {"description": "Responding to the user…"},
                                    }
                                )
                            continue

                    # ─── Emit detailed tool status upon completion ────────────────────────
                    if etype == "response.output_item.done":
                        item_raw = event.get("item")
                        item = item_raw if isinstance(item_raw, dict) else {}
                        item_type = item.get("type", "")
                        item_name = item.get("name", "unnamed_tool")

                        # Skip irrelevant item types
                        if item_type in ("message"):
                            continue

                        # Decide persistence policy
                        should_persist = False
                        if item_type == "reasoning":
                            # Persist reasoning when retention is enabled
                            should_persist = valves.PERSIST_REASONING_TOKENS in {"next_reply", "conversation"}

                        elif item_type in ("message", "web_search_call"):
                            # Never persist assistant/user messages or ephemeral search calls
                            should_persist = False

                        else:
                            # Persist all other non-message items if valve enabled
                            should_persist = valves.PERSIST_TOOL_RESULTS

                        if item_type == "function_call":
                            # Defer persistence until the corresponding tool result is stored
                            should_persist = False

                        if should_persist:
                            normalized_item = _normalize_persisted_item(item)
                            if normalized_item:
                                row = self._make_db_row(
                                    chat_id, message_id, openwebui_model, normalized_item
                                )
                                if row:
                                    pending_items.append(row)


                        # Default empty content
                        title = f"Running `{item_name}`"
                        content = ""
                        image_markdowns: list[str] = []

                        # Prepare detailed content per item_type
                        if item_type == "function_call":
                            title = f"Running the {item_name} tool…"
                            arguments = json.loads(item.get("arguments") or "{}")
                            args_formatted = ", ".join(f"{k}={json.dumps(v)}" for k, v in arguments.items())
                            content = wrap_code_block(f"{item_name}({args_formatted})", "python")

                        elif item_type == "web_search_call":
                            action = item.get("action", {}) or {}

                            if action.get("type") == "search":
                                query = action.get("query")
                                sources = action.get("sources") or []
                                urls = [s.get("url") for s in sources if s.get("url")]

                                if event_emitter:
                                    # Emit 'searching' status update along with the search query if available
                                    if query:
                                        await event_emitter({
                                            "type": "status",
                                            "data": {
                                                "action": "web_search_queries_generated",
                                                "description": "Searching",
                                                "queries": [query],
                                                "done": False,
                                            },
                                        })

                                    # If API returned sources, emit the panel now
                                    if urls:
                                        await event_emitter({
                                            "type": "status",
                                            "data": {
                                                "action": "web_search",
                                                "description": "Reading through {{count}} sites",
                                                "query": query,
                                                "urls": urls,
                                                "done": False,
                                            },
                                        })

                            elif action.get("type") == "open_page":
                                if event_emitter:
                                    raw_title = action.get("title")
                                    raw_host = action.get("host")
                                    raw_url = action.get("url")
                                    title = raw_title.strip() if isinstance(raw_title, str) else ""
                                    host = raw_host.strip() if isinstance(raw_host, str) else ""
                                    url = raw_url.strip() if isinstance(raw_url, str) else ""
                                    description = "Opening page"
                                    if host:
                                        description = f"Opening {host}"
                                    elif title:
                                        description = f"Opening {title}"
                                    elif url:
                                        description = f"Opening {url}"
                                    await event_emitter({
                                        "type": "status",
                                        "data": {
                                            "action": "open_page",
                                            "description": description,
                                            "title": raw_title or (title or None),
                                            "host": raw_host or (host or None),
                                            "url": raw_url or (url or None),
                                            "done": False,
                                        },
                                    })
                                continue
                            elif action.get("type") == "find_in_page":
                                if event_emitter:
                                    raw_query = next(
                                        (
                                            value
                                            for value in (
                                                action.get("needle"),
                                                action.get("query"),
                                                action.get("text"),
                                            )
                                            if isinstance(value, str) and value.strip()
                                        ),
                                        "",
                                    )
                                    query = raw_query.strip() if isinstance(raw_query, str) else ""
                                    raw_page_url = action.get("url")
                                    page_url = raw_page_url.strip() if isinstance(raw_page_url, str) else ""
                                    description = "Searching within page"
                                    if query:
                                        description = f"Searching for {query}"
                                    await event_emitter({
                                        "type": "status",
                                        "data": {
                                            "action": "find_in_page",
                                            "description": description,
                                            "needle": raw_query or (query or None),
                                            "url": raw_page_url or (page_url or None),
                                            "done": False,
                                        },
                                    })
                                continue
                                    
                            continue

                        elif item_type == "file_search_call":
                            title = "Let me skim those files…"
                        elif item_type == "image_generation_call":
                            title = "Let me create that image…"
                            item_id = item.get("id")
                            if item_id and item_id in processed_image_item_ids:
                                self.logger.debug("Skipping duplicate image item '%s'", item_id)
                            else:
                                if item_id:
                                    processed_image_item_ids.add(item_id)
                                try:
                                    image_markdowns = await _render_image_markdown(item)
                                except Exception as exc:
                                    self.logger.error(
                                        "Failed to process generated image for item '%s': %s",
                                        item_id or "<unknown>",
                                        exc,
                                        exc_info=self.logger.isEnabledFor(logging.DEBUG),
                                    )
                                    await self._emit_status(
                                        event_emitter,
                                        "⚠️ Unable to process generated image output",
                                        done=False,
                                    )
                                    image_markdowns = []
                        elif item_type == "local_shell_call":
                            title = "Let me run that command…"
                        elif item_type == "reasoning":
                            title = None # Don't emit a title for reasoning items
                            if (
                                event_emitter
                                and thinking_box_enabled
                                and reasoning_buffer
                                and not reasoning_completed_emitted
                            ):
                                await event_emitter(
                                    {
                                        "type": "reasoning:completed",
                                        "data": {"content": reasoning_buffer},
                                    }
                                )
                                reasoning_completed_emitted = True

                        # Log the status with prepared title and detailed content instead of emitting it
                        if title:
                            desc = title if not content else f"{title}\n{content}"
                            if thinking_tasks:
                                cancel_thinking()
                            self.logger.debug("Tool status update: %s", desc)

                        if image_markdowns:
                            note_model_activity()
                            note_generation_activity()
                            for snippet in image_markdowns:
                                assistant_message = _append_output_block(assistant_message, snippet)
                            if event_emitter:
                                await event_emitter({"type": "chat:message", "data": {"content": assistant_message}})

                        continue

                    # ─── Capture final response payload for this loop
                    if etype == "response.completed":
                        note_model_activity()  # Ensure thinking tasks are cancelled when response completes
                        final_response = event.get("response", {})
                        response_completed_at = perf_counter()
                        if generation_started_at is not None:
                            generation_last_event_at = response_completed_at
                        break

                if final_response is None:
                    raise ValueError("No final response received from OpenAI Responses API.")

                # Extract usage information from OpenAI response and pass-through to Open WebUI
                raw_usage = final_response.get("usage") or {}
                usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}

                if usage:
                    usage["turn_count"] = 1
                    usage["function_call_count"] = sum(
                        1 for i in final_response["output"] if i["type"] == "function_call"
                    )
                    total_usage = merge_usage_stats(total_usage, usage)
                    await self._emit_completion(
                        event_emitter,
                        content=assistant_message,
                        usage=total_usage,
                        done=False,
                    )

                metadata_model = None
                if isinstance(metadata, dict):
                    model_block = metadata.get("model")
                    if isinstance(model_block, dict):
                        metadata_model = model_block.get("id")
                snapshot_model_id = self._qualify_model_for_pipe(
                    pipe_identifier,
                    metadata_model or body.model,
                )

                await self._maybe_dump_costs_snapshot(
                    valves,
                    user_id=user_id or "",
                    model_id=snapshot_model_id,
                    usage=usage if usage else {},
                    user_obj=user_obj,
                    pipe_id=pipe_identifier,
                )

                # Execute tool calls (if any), persist results (if valve enabled), and append to body.input.
                call_items = []
                for item in final_response.get("output", []):
                    if item.get("type") == "function_call":
                        normalized_call = _normalize_persisted_item(item)
                        if normalized_call:
                            call_items.append(normalized_call)
                if call_items:
                    note_model_activity()  # Cancel thinking tasks when function calls begin
                    body.input.extend(call_items)
                    self._sanitize_request_input(body)

                calls = [i for i in final_response.get("output", []) if i.get("type") == "function_call"]
                self.logger.debug("📞 Found %d function_call items in response", len(calls))
                function_outputs: list[dict[str, Any]] = []
                if calls:
                    function_outputs = await self._execute_function_calls(calls, tool_registry)
                    if valves.PERSIST_TOOL_RESULTS:
                        self.logger.debug("💾 Persisting %d tool results", len(function_outputs))
                        persist_payloads: list[dict] = []
                        for idx, output in enumerate(function_outputs):
                            if idx < len(call_items):
                                persist_payloads.append(call_items[idx])
                            persist_payloads.append(output)

                        if persist_payloads:
                            self.logger.debug("🔍 Processing %d persist_payloads", len(persist_payloads))
                            for idx, payload in enumerate(persist_payloads, start=1):
                                payload_type = payload.get("type")
                                self.logger.debug("🔍 [%d/%d] Payload type=%s", idx, len(persist_payloads), payload_type)
                                normalized_payload = _normalize_persisted_item(payload)
                                if not normalized_payload:
                                    self.logger.warning("❌ [%d/%d] Normalization returned None for type=%s", idx, len(persist_payloads), payload_type)
                                    continue
                                self.logger.debug("✅ [%d/%d] Normalized successfully (type=%s)", idx, len(persist_payloads), payload_type)
                                row = self._make_db_row(
                                    chat_id, message_id, openwebui_model, normalized_payload
                                )
                                if not row:
                                    self.logger.warning("❌ [%d/%d] _make_db_row returned None (chat_id=%s, message_id=%s, type=%s)", idx, len(persist_payloads), chat_id, message_id, payload_type)
                                    continue
                                self.logger.debug("✅ [%d/%d] Row created; enqueueing for persistence", idx, len(persist_payloads))
                                pending_items.append(row)
                            self.logger.debug("📦 Total pending_items after loop: %d", len(pending_items))
                            if thinking_tasks:
                                cancel_thinking()
                            await _flush_pending("function_outputs")

                    for output in function_outputs:
                        result_text = wrap_code_block(output.get("output", ""))
                        if thinking_tasks:
                            cancel_thinking()
                        self.logger.debug("Received tool result\n%s", result_text)
                    body.input.extend(function_outputs)
                    self._sanitize_request_input(body)
                else:
                    break

        # Catch any exceptions during the streaming loop and emit an error
        except asyncio.CancelledError:
            was_cancelled = True
            raise
        except OpenRouterAPIError as exc:
            error_occurred = True
            assistant_message = ""
            cancel_thinking()
            await self._report_openrouter_error(
                exc,
                event_emitter=event_emitter,
                normalized_model_id=body.model,
                api_model_id=getattr(body, "api_model", None),
                usage=total_usage,
            )
        except Exception as e:  # pragma: no cover - network errors
            error_occurred = True
            await self._emit_error(event_emitter, f"Error: {str(e)}", show_error_message=True, show_error_log_citation=True, done=True)

        finally:
            cancel_thinking()
            for t in thinking_tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
            if (
                reasoning_buffer
                and reasoning_stream_active
                and not reasoning_completed_emitted
                and event_emitter
                and thinking_box_enabled
            ):
                await event_emitter(
                    {
                        "type": "reasoning:completed",
                        "data": {"content": reasoning_buffer},
                    }
                )
                reasoning_completed_emitted = True
            surrogate_carry["assistant"] = ""
            surrogate_carry["reasoning"] = ""
            if (not error_occurred) and (not was_cancelled):
                await self._cleanup_replayed_reasoning(body, valves)
            if (not error_occurred) and (not was_cancelled) and event_emitter:
                effective_start = stream_started_at or request_started_at
                elapsed = max(0.0, perf_counter() - effective_start)
                stream_window = None
                last_generation_stamp = generation_last_event_at or response_completed_at
                if last_generation_stamp is not None:
                    duration = max(0.0, last_generation_stamp - effective_start)
                    if duration > 0:
                        stream_window = duration
                description = self._format_final_status_description(
                    elapsed=elapsed,
                    total_usage=total_usage,
                    valves=valves,
                    stream_duration=stream_window,
                )
                await event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": description,
                            "done": True,
                        },
                    }
                    )

            request_id = SessionLogger.request_id.get() or ""
            if request_id:
                logs = list(SessionLogger.logs.get(request_id, []))
                if logs and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug("Collected %d session log entries for request %s.", len(logs), request_id)
                resolved_user_id = str(user_id or metadata.get("user_id") or "")
                resolved_session_id = str(metadata.get("session_id") or "")
                resolved_chat_id = str(metadata.get("chat_id") or "")
                resolved_message_id = str(metadata.get("message_id") or "")
                with contextlib.suppress(Exception):
                    self._enqueue_session_log_archive(
                        valves,
                        user_id=resolved_user_id,
                        session_id=resolved_session_id,
                        chat_id=resolved_chat_id,
                        message_id=resolved_message_id,
                        request_id=request_id,
                        log_lines=logs,
                    )

            if (not error_occurred) and (not was_cancelled):
                # Emit completion (middleware.py also does this so this just covers if there is a downstream error)
                # Emit the final completion frame with the last assistant snapshot so
                # late-arriving emitters (middleware, other workers) cannot wipe the UI.
                await self._emit_completion(
                    event_emitter,
                    content=assistant_message,
                    usage=total_usage,
                    done=True,
                )

            # Clear logs
            if request_id:
                SessionLogger.logs.pop(request_id, None)
            SessionLogger.cleanup()

            chat_id = metadata.get("chat_id")
            message_id = metadata.get("message_id")
            if (not was_cancelled) and chat_id and message_id and emitted_citations:
                try:
                    Chats.upsert_message_to_chat_by_id_and_message_id(
                        chat_id, message_id, {"sources": emitted_citations}
                    )
                except Exception as exc:
                    self.logger.warning("Failed to persist citations for chat_id=%s message_id=%s: %s", chat_id, message_id, exc)
                    await self._emit_notification(
                        event_emitter,
                        "Unable to save citations for this response. Output was delivered successfully.",
                        level="warning",
                    )

            if not was_cancelled:
                await _flush_pending("finalize")
                if pending_ulids:
                    marker_lines = [_serialize_marker(ulid) for ulid in pending_ulids]
                    ulid_block = "\n".join(marker_lines)
                    if assistant_message:
                        if not assistant_message.endswith("\n"):
                            assistant_message += "\n"
                        if not assistant_message.endswith("\n\n"):
                            assistant_message += "\n"
                    assistant_message += ulid_block
                    if not assistant_message.endswith("\n"):
                        assistant_message += "\n"

        # Return the final output to ensure persistence (unless cancelled, in which case the exception propagates).
        return assistant_message

    async def _cleanup_replayed_reasoning(self, body: ResponsesBody, valves: Pipe.Valves) -> None:
        """Delete once-used reasoning artifacts when retention is limited to the next reply."""
        if valves.PERSIST_REASONING_TOKENS != "next_reply":
            return
        refs = getattr(body, "_replayed_reasoning_refs", None)
        if not refs:
            return
        setattr(body, "_replayed_reasoning_refs", [])
        await self._delete_artifacts(refs)

    async def _run_nonstreaming_loop(
        self,
        body: ResponsesBody,
        valves: Pipe.Valves,
        event_emitter: EventEmitter | None,
        metadata: Dict[str, Any] = {},
        tools: dict[str, Dict[str, Any]] | list[dict[str, Any]] | None = None,
        session: aiohttp.ClientSession | None = None,
        user_id: str = "",
        *,
        request_context: Optional[Request] = None,
        user_obj: Optional[Any] = None,
        pipe_identifier: Optional[str] = None,
    ) -> str:
        """Unified implementation: reuse the streaming path.

        We force `stream=True` and delegate to `_run_streaming_loop`, but wrap the
        emitter so incremental `chat:message` frames are suppressed. The final
        message text is returned (same contract as before).
        """

        # Force SSE so we can reuse the streaming machinery
        body.stream = True

        # Pass through status / citations / usage, but do NOT emit partial text
        wrapped_emitter = _wrap_event_emitter(
            event_emitter,
            suppress_chat_messages=True,
            suppress_completion=False,
        )

        if session is None:
            raise RuntimeError("HTTP session is required for non-streaming")

        return await self._run_streaming_loop(
            body,
            valves,
            wrapped_emitter,
            metadata,
            tools or {},
            session=session,
            user_id=user_id,
            request_context=request_context,
            user_obj=user_obj,
            pipe_identifier=pipe_identifier,
        )

    # 4.4 Task Model Handling
    async def _run_task_model_request(
        self,
        body: Dict[str, Any],
        valves: Pipe.Valves,
        *,
        session: aiohttp.ClientSession | None = None,
        task_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        user_obj: Optional[Any] = None,
        pipe_id: Optional[str] = None,
        snapshot_model_id: Optional[str] = None,
    ) -> str:
        """Process a task model request via the Responses API.

        Task models (e.g. generating a chat title or tags) return their
        information as standard Responses output.  This helper performs a single
        non-streaming call and extracts the plain text from the response items.
        """

        task_body = dict(body or {})
        source_model_id = task_body.get("model", "")
        task_body["model"] = OpenRouterModelRegistry.api_model_id(source_model_id) or source_model_id
        task_body.setdefault("input", "")
        task_body["stream"] = False
        if valves.USE_MODEL_MAX_OUTPUT_TOKENS:
            if task_body.get("max_output_tokens") is None:
                default_max = ModelFamily.max_completion_tokens(source_model_id)
                if default_max:
                    task_body["max_output_tokens"] = default_max
        else:
            task_body.pop("max_output_tokens", None)
        task_body = _filter_openrouter_request(task_body)

        attempts = 2
        delay_seconds = 0.2  # keep retries snappy; task models run in latency-sensitive contexts
        last_error: Optional[Exception] = None

        if session is None:
            raise RuntimeError("HTTP session is required for task model requests")

        for attempt in range(1, attempts + 1):
            try:
                response = await self.send_openai_responses_nonstreaming_request(
                    session,
                    task_body,
                    api_key=EncryptedStr.decrypt(valves.API_KEY),
                    base_url=valves.BASE_URL,
                )

                usage = response.get("usage") if isinstance(response, dict) else None
                if usage and isinstance(usage, dict) and user_id:
                    safe_model_id = snapshot_model_id or self._qualify_model_for_pipe(
                        pipe_id,
                        source_model_id,
                    )
                    if safe_model_id:
                        try:
                            await self._maybe_dump_costs_snapshot(
                                valves,
                                user_id=user_id,
                                model_id=safe_model_id,
                                usage=usage,
                                user_obj=user_obj,
                                pipe_id=pipe_id,
                            )
                        except Exception as exc:  # pragma: no cover - guard against Redis-side issues
                            self.logger.debug("Task cost snapshot failed: %s", exc)

                message = self._extract_task_output_text(response).strip()
                if message:
                    return message

                raise ValueError(
                    "Task model returned no output_text content."
                )

            except Exception as exc:
                last_error = exc
                self.logger.warning("Task model attempt %d/%d failed: %s", attempt, attempts, exc, exc_info=self.logger.isEnabledFor(logging.DEBUG))
                if attempt < attempts:
                    await asyncio.sleep(delay_seconds)
                    delay_seconds = min(delay_seconds * 2, 0.8)

        task_type = (task_context or {}).get("type") or "task"
        error_message = (
            f"Task model '{task_type}' failed after {attempts} attempt(s): {last_error}"
        )
        self.logger.error(error_message, exc_info=self.logger.isEnabledFor(logging.DEBUG))
        return f"[Task error] Unable to generate {task_type}. Please retry later."

    @staticmethod
    def _extract_task_output_text(response: Dict[str, Any]) -> str:
        """
        Normalize Responses API payloads into a plain text string for task models.
        """
        if not isinstance(response, dict):
            return ""

        text_parts: list[str] = []
        output_items = response.get("output", [])
        if isinstance(output_items, list):
            for item in output_items:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for content in item.get("content", []):
                    if not isinstance(content, dict):
                        continue
                    if content.get("type") == "output_text":
                        text_parts.append(content.get("text", "") or "")

        # Fallback: some providers return a collapsed output_text field
        fallback_text = response.get("output_text")
        if isinstance(fallback_text, str):
            text_parts.append(fallback_text)

        joined = "\n".join(part for part in text_parts if part)
        return joined
      
    def _maybe_apply_anthropic_beta_headers(
        self,
        headers: dict[str, str],
        model: Any,
        *,
        valves: "Pipe.Valves",
    ) -> None:
        """Apply provider-specific beta headers when needed.

        Currently used to opt into Claude's interleaved thinking mode when requested.
        """
        if not isinstance(headers, dict):
            return
        if not getattr(valves, "ENABLE_ANTHROPIC_INTERLEAVED_THINKING", True):
            return
        if not isinstance(model, str):
            return
        model_id = model.strip()
        if not model_id.startswith("anthropic/"):
            return

        feature = "interleaved-thinking-2025-05-14"
        existing = headers.get("x-anthropic-beta") or headers.get("X-Anthropic-Beta") or ""
        values = [part.strip() for part in existing.split(",") if part.strip()] if existing else []
        if feature not in values:
            values.append(feature)
        if values:
            headers["x-anthropic-beta"] = ",".join(values)
        headers.pop("X-Anthropic-Beta", None)

    # 4.5 LLM HTTP Request Helpers
    async def send_openai_responses_streaming_request(
        self,
        session: aiohttp.ClientSession,
        request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: "Pipe.Valves | None" = None,
        workers: int = 4,
        breaker_key: Optional[str] = None,
        delta_char_limit: int = 0,
        idle_flush_ms: int = 0,
        chunk_queue_maxsize: int = 100,
        event_queue_maxsize: int = 100,
        event_queue_warn_size: int = 1000,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Producer/worker SSE pipeline with configurable delta batching."""

        effective_valves = valves or self.valves
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Title": _OPENROUTER_TITLE,
            "HTTP-Referer": _OPENROUTER_REFERER,
        }
        self._maybe_apply_anthropic_beta_headers(
            headers,
            request_body.get("model"),
            valves=effective_valves,
        )
        _debug_print_request(headers, request_body)
        url = base_url.rstrip("/") + "/responses"

        workers = max(1, min(int(workers or 1), 8))
        chunk_queue_size = max(0, int(chunk_queue_maxsize))
        event_queue_size = max(0, int(event_queue_maxsize))
        chunk_queue: asyncio.Queue[tuple[Optional[int], bytes]] = asyncio.Queue(maxsize=chunk_queue_size)
        event_queue: asyncio.Queue[tuple[Optional[int], Optional[dict[str, Any]]]] = asyncio.Queue(maxsize=event_queue_size)
        chunk_sentinel = (None, b"")
        delta_batch_threshold = max(1, int(delta_char_limit))
        idle_flush_seconds = float(idle_flush_ms) / 1000 if idle_flush_ms > 0 else None
        passthrough_deltas = delta_char_limit <= 0 and idle_flush_ms <= 0
        requested_model = request_body.get("model")

        async def _producer() -> None:
            seq = 0
            retryer = AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
                retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
                reraise=True,
            )
            try:
                async for attempt in retryer:
                    if breaker_key and not self._breaker_allows(breaker_key):
                        raise RuntimeError("Breaker open for user during stream")
                    with attempt:
                        buf = bytearray()
                        event_data_parts: list[bytes] = []
                        stream_complete = False
                        try:
                            async with session.post(url, json=request_body, headers=headers) as resp:
                                if resp.status >= 400:
                                    error_body = await _debug_print_error_response(resp)
                                    if breaker_key:
                                        self._record_failure(breaker_key)
                                    special_statuses = {400, 401, 402, 403, 408, 429}
                                    if resp.status in special_statuses:
                                        extra_meta: dict[str, Any] = {}
                                        retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                                        if retry_after:
                                            extra_meta["retry_after"] = retry_after
                                            extra_meta["retry_after_seconds"] = retry_after
                                        rate_scope = (
                                            resp.headers.get("X-RateLimit-Scope")
                                            or resp.headers.get("x-ratelimit-scope")
                                        )
                                        if rate_scope:
                                            extra_meta["rate_limit_type"] = rate_scope
                                        reason_text = resp.reason or "HTTP error"
                                        raise _build_openrouter_api_error(
                                            resp.status,
                                            reason_text,
                                            error_body,
                                            requested_model=request_body.get("model"),
                                            extra_metadata=extra_meta or None,
                                        )
                                    if resp.status < 500:
                                        raise RuntimeError(
                                            f"OpenRouter request failed ({resp.status}): {resp.reason}"
                                        )
                                resp.raise_for_status()

                                async for chunk in resp.content.iter_chunked(4096):
                                    view = memoryview(chunk)
                                    if breaker_key and not self._breaker_allows(breaker_key):
                                        raise RuntimeError("Breaker open during stream")
                                    buf.extend(view)
                                    start_idx = 0
                                    while True:
                                        newline_idx = buf.find(b"\n", start_idx)
                                        if newline_idx == -1:
                                            break
                                        line = buf[start_idx:newline_idx]
                                        start_idx = newline_idx + 1
                                        stripped = line.strip()
                                        if not stripped:
                                            if event_data_parts:
                                                data_blob = b"\n".join(event_data_parts).strip()
                                                event_data_parts.clear()
                                                if not data_blob:
                                                    continue
                                                if data_blob == b"[DONE]":
                                                    stream_complete = True
                                                    break
                                                await chunk_queue.put((seq, data_blob))
                                                # self.logger.debug("Enqueued SSE chunk seq=%s size=%s",seq,len(data_blob))
                                                seq += 1
                                            continue
                                        if stripped.startswith(b":"):
                                            continue
                                        if stripped.startswith(b"data:"):
                                            event_data_parts.append(bytes(stripped[5:].lstrip()))
                                            continue
                                    if start_idx > 0:
                                        del buf[:start_idx]
                                    if stream_complete:
                                        break

                                if event_data_parts and not stream_complete:
                                    data_blob = b"\n".join(event_data_parts).strip()
                                    event_data_parts.clear()
                                    if data_blob and data_blob != b"[DONE]":
                                        await chunk_queue.put((seq, data_blob))
                                        # self.logger.debug("Enqueued SSE chunk seq=%s size=%s",seq,len(data_blob))
                                        seq += 1
                        except Exception as producer_exc:
                            self.logger.error(
                                "Producer encountered error while streaming from OpenRouter: %s",
                                producer_exc,
                                exc_info=True,
                            )
                            if breaker_key:
                                self._record_failure(breaker_key)
                            raise
                        if stream_complete:
                            break
            finally:
                for _ in range(workers):
                    with contextlib.suppress(asyncio.CancelledError):
                        await chunk_queue.put(chunk_sentinel)

        async def _worker(worker_idx: int) -> None:
            try:
                while True:
                    seq, data = await chunk_queue.get()
                    try:
                        if seq is None:
                            break
                        if data == b"[DONE]":
                            continue
                        try:
                            event = json.loads(data.decode("utf-8"))
                        except json.JSONDecodeError as exc:
                            self.logger.warning("Chunk parse failed (seq=%s): %s", seq, exc)
                            continue
                        await event_queue.put((seq, event))
                        # self.logger.debug("Worker %s emitted seq=%s",worker_idx,seq)
                    finally:
                        chunk_queue.task_done()
            finally:
                with contextlib.suppress(asyncio.CancelledError):
                    await event_queue.put((None, None))

        producer_task = asyncio.create_task(_producer(), name="openrouter-sse-producer")
        worker_tasks = [
            asyncio.create_task(_worker(idx), name=f"openrouter-sse-worker-{idx}")
            for idx in range(workers)
        ]

        pending_events: dict[int, dict[str, Any]] = {}
        next_seq = 0
        done_workers = 0
        delta_buffer: list[str] = []
        delta_template: Optional[dict[str, Any]] = None
        delta_length = 0
        event_queue_warn_last_ts: float = 0.0

        def flush_delta(force: bool = False) -> Optional[dict[str, Any]]:
            if passthrough_deltas:
                return None
            nonlocal delta_buffer, delta_template, delta_length
            if delta_buffer and (force or delta_length >= delta_batch_threshold):
                combined = "".join(delta_buffer)
                base = dict(delta_template or {"type": "response.output_text.delta"})
                base["delta"] = combined
                delta_buffer = []
                delta_template = None
                delta_length = 0
                return base
            return None

        try:
            while True:
                timeout = idle_flush_seconds if (idle_flush_seconds and delta_buffer) else None
                timed_out = False
                seq: int | None = None
                event: dict[str, Any] | None = None
                if timeout is not None:
                    try:
                        seq, event = await asyncio.wait_for(event_queue.get(), timeout=timeout)
                    except asyncio.TimeoutError:
                        timed_out = True
                else:
                    seq, event = await event_queue.get()

                if timed_out:
                    batched = flush_delta(force=True)
                    if batched:
                        yield batched
                    continue

                event_queue.task_done()

                # Non-spammy queue monitoring
                now = perf_counter()
                if event_queue.qsize() >= event_queue_warn_size and (now - event_queue_warn_last_ts) >= 30.0:
                    self.logger.warning(
                        "Event queue backlog high: %d items (session=%s)",
                        event_queue.qsize(),
                        SessionLogger.session_id.get() or "unknown",
                    )
                    event_queue_warn_last_ts = now

                if seq is None:
                    done_workers += 1
                    if done_workers >= workers and not pending_events:
                        break
                    continue
                if event is None:
                    self.logger.debug("Skipping empty SSE event (seq=%s)", seq)
                    continue
                pending_events[seq] = event
                while next_seq in pending_events:
                    current = pending_events.pop(next_seq)
                    next_seq += 1
                    streaming_error = self._extract_streaming_error_event(current, requested_model)
                    if streaming_error is not None:
                        raise streaming_error
                    etype = current.get("type")
                    if etype == "response.output_text.delta":
                        delta_chunk = current.get("delta", "")
                        if passthrough_deltas:
                            if delta_chunk:
                                yield current
                            continue
                        if delta_chunk:
                            delta_buffer.append(delta_chunk)
                            delta_length += len(delta_chunk)
                            if delta_template is None:
                                delta_template = {k: v for k, v in current.items() if k != "delta"}
                        batched = flush_delta()
                        if batched:
                            yield batched
                        continue

                    batched = flush_delta(force=True)
                    if batched:
                        yield batched
                    yield current

            final_delta = flush_delta(force=True)
            if final_delta:
                yield final_delta

            await producer_task
        finally:
            if not producer_task.done():
                producer_task.cancel()
            for task in worker_tasks:
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(producer_task, *worker_tasks, return_exceptions=True)
            for idx, result in enumerate(results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    task_name = "producer" if idx == 0 else f"worker-{idx - 1}"
                    self.logger.error(
                        "SSE %s task failed during cleanup: %s",
                        task_name,
                        result,
                        exc_info=result,
                    )

    async def send_openai_responses_nonstreaming_request(
        self,
        session: aiohttp.ClientSession,
        request_params: dict[str, Any],
        api_key: str,
        base_url: str,
    ) -> Dict[str, Any]:
        """Send a blocking request to the Responses API and return the JSON payload."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": _OPENROUTER_TITLE,
            "HTTP-Referer": _OPENROUTER_REFERER,
        }
        _debug_print_request(headers, request_params)
        url = base_url.rstrip("/") + "/responses"

        retryer = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
            reraise=True,
        )

        async for attempt in retryer:
            with attempt:
                async with session.post(url, json=request_params, headers=headers) as resp:
                    if resp.status >= 400:
                        error_body = await _debug_print_error_response(resp)
                        if resp.status < 500:
                            special_statuses = {400, 401, 402, 403, 408, 429}
                            if resp.status in special_statuses:
                                extra_meta: dict[str, Any] = {}
                                retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                                if retry_after:
                                    extra_meta["retry_after"] = retry_after
                                    extra_meta["retry_after_seconds"] = retry_after
                                rate_scope = (
                                    resp.headers.get("X-RateLimit-Scope")
                                    or resp.headers.get("x-ratelimit-scope")
                                )
                                if rate_scope:
                                    extra_meta["rate_limit_type"] = rate_scope
                                reason_text = resp.reason or "HTTP error"
                                raise _build_openrouter_api_error(
                                    resp.status,
                                    reason_text,
                                    error_body,
                                    requested_model=request_params.get("model"),
                                    extra_metadata=extra_meta or None,
                                )
                            raise RuntimeError(
                                f"OpenRouter request failed ({resp.status}): {resp.reason}"
                            )
                    resp.raise_for_status()
                    return await resp.json()
        self.logger.error("Responses API call completed without yielding a response body; returning empty payload.")
        return {}

    async def close(self) -> None:
        """Shutdown background resources (DB executor, queue worker)."""
        self.shutdown()
        await self._stop_request_worker()
        await self._stop_log_worker()
        await self._stop_redis()
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

    def __del__(self) -> None:
        """Best-effort cleanup hook for garbage collection."""
        self.shutdown()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            try:
                loop.create_task(self.close())
            except RuntimeError:
                pass
        else:
            try:
                new_loop = asyncio.new_event_loop()
                try:
                    new_loop.run_until_complete(self.close())
                finally:
                    new_loop.close()
            except RuntimeError:
                pass

    async def _stop_redis(self) -> None:
        self._redis_enabled = False
        tasks = []
        for task in (self._redis_listener_task, self._redis_flush_task, self._redis_ready_task):
            if task:
                task.cancel()
                tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._redis_listener_task = None
        self._redis_flush_task = None
        self._redis_ready_task = None
        if self._redis_client:
            with contextlib.suppress(Exception):
                await _wait_for(self._redis_client.close())
            self._redis_client = None

    # 4.6 Tool Execution Logic
    async def _execute_function_calls(
        self,
        calls: list[dict],
        tools: dict[str, dict[str, Any]],
    ) -> list[dict]:
        """Execute tool calls via the per-request queue/worker pipeline."""

        context = self._TOOL_CONTEXT.get()
        if context is None:
            self.logger.debug("Using legacy tool execution path")
            # Fallback: legacy direct execution
            return await self._execute_function_calls_legacy(calls, tools)

        loop = asyncio.get_running_loop()
        pending: list[tuple[dict[str, Any], asyncio.Future]] = []
        outputs: list[dict[str, Any]] = []
        enqueued_any = False
        breaker_only_skips = True

        for call in calls:
            raw_name = call.get("name")
            tool_name = raw_name.strip() if isinstance(raw_name, str) else ""
            if not tool_name:
                breaker_only_skips = False
                outputs.append(
                    self._build_tool_output(
                        call,
                        "Tool call missing name",
                        status="failed",
                    )
                )
                continue
            tool_cfg = tools.get(tool_name)
            if not tool_cfg:
                breaker_only_skips = False
                outputs.append(
                    self._build_tool_output(
                        call,
                        "Tool not found",
                        status="failed",
                    )
                )
                continue
            tool_type = (tool_cfg.get("type") or "function").lower()
            if not self._tool_type_allows(context.user_id, tool_type):
                await self._notify_tool_breaker(context, tool_type, call.get("name"))
                outputs.append(
                    self._build_tool_output(
                        call,
                        f"Tool '{call.get('name')}' skipped due to repeated failures.",
                        status="skipped",
                    )
                )
                continue
            fn = tool_cfg.get("callable")
            if fn is None:
                breaker_only_skips = False
                outputs.append(
                    self._build_tool_output(
                        call,
                        f"Tool '{call.get('name')}' has no callable configured.",
                        status="failed",
                    )
                )
                continue
            try:
                raw_args = call.get("arguments") or "{}"
                args = self._parse_tool_arguments(raw_args)
            except Exception as exc:
                breaker_only_skips = False
                outputs.append(
                    self._build_tool_output(
                        call,
                        f"Invalid arguments: {exc}",
                        status="failed",
                    )
                )
                continue

            future: asyncio.Future = loop.create_future()
            allow_batch = self._is_batchable_tool_call(args)
            queued = _QueuedToolCall(
                call=call,
                tool_cfg=tool_cfg,
                args=args,
                future=future,
                allow_batch=allow_batch,
            )
            await context.queue.put(queued)
            self.logger.debug("Enqueued tool %s (batch=%s)", call.get("name"), allow_batch)
            pending.append((call, future))
            enqueued_any = True
            breaker_only_skips = False

        if not enqueued_any and breaker_only_skips and context.user_id:
            self._record_failure(context.user_id)

        for call, future in pending:
            try:
                if context and context.idle_timeout:
                    result = await asyncio.wait_for(future, timeout=context.idle_timeout)
                else:
                    result = await future
            except asyncio.TimeoutError:
                message = (
                    f"Tool '{call.get('name')}' idle timeout after {context.idle_timeout:.0f}s."
                    if context and context.idle_timeout
                    else "Tool idle timeout exceeded."
                )
                if context:
                    context.timeout_error = context.timeout_error or message
                raise RuntimeError(message)
            except Exception as exc:  # pragma: no cover - defensive
                result = self._build_tool_output(
                    call,
                    f"Tool error: {exc}",
                    status="failed",
                )
            outputs.append(result)

        if context and context.timeout_error:
            raise RuntimeError(context.timeout_error)

        return outputs

    async def _execute_function_calls_legacy(
        self,
        calls: list[dict],
        tools: dict[str, dict[str, Any]],
    ) -> list[dict]:
        """Legacy direct execution path used when tool context is unavailable."""

        if not self._legacy_tool_warning_emitted:
            self._legacy_tool_warning_emitted = True
            self.logger.warning("Tool queue unavailable; falling back to direct execution.")

        tasks: list[Awaitable] = []
        for call in calls:
            raw_name = call.get("name")
            tool_name = raw_name.strip() if isinstance(raw_name, str) else ""
            if not tool_name:
                tasks.append(asyncio.sleep(0, result=RuntimeError("Tool call missing name")))
                continue
            tool_cfg = tools.get(tool_name)
            if not tool_cfg:
                tasks.append(asyncio.sleep(0, result=RuntimeError("Tool not found")))
                continue
            fn = tool_cfg.get("callable")
            if fn is None:
                tasks.append(asyncio.sleep(0, result=RuntimeError("Tool has no callable configured")))
                continue
            raw_args = call.get("arguments") or "{}"
            args = self._parse_tool_arguments(raw_args)
            if inspect.iscoroutinefunction(fn):
                tasks.append(fn(**args))
            else:
                tasks.append(asyncio.to_thread(fn, **args))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs: list[dict] = []
        for call, result in zip(calls, results):
            if isinstance(result, Exception):
                status = "failed"
                output_text = f"Tool error: {result}"
            else:
                status = "completed"
                output_text = "" if result is None else str(result)
            outputs.append(
                self._build_tool_output(
                    call,
                    output_text,
                    status=status,
                )
            )
        return outputs

    def _build_tool_output(
        self,
        call: dict[str, Any],
        output_text: str,
        *,
        status: str = "completed",
    ) -> dict[str, Any]:
        call_id = call.get("call_id") or generate_item_id()
        return {
            "type": "function_call_output",
            "id": generate_item_id(),
            "status": status,
            "call_id": call_id,
            "output": output_text,
        }

    def _is_batchable_tool_call(self, args: dict[str, Any]) -> bool:
        blockers = {"depends_on", "_depends_on", "sequential", "no_batch"}
        return not any(key in args for key in blockers)

    def _parse_tool_arguments(self, raw_args: Any) -> dict[str, Any]:
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args)
            except json.JSONDecodeError:
                try:
                    literal_value = ast.literal_eval(raw_args)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError("Unable to parse tool arguments") from exc
                if not isinstance(literal_value, dict):
                    raise ValueError("Tool arguments must evaluate to an object")
                return literal_value
        raise ValueError(f"Unsupported argument type: {type(raw_args).__name__}")
    # 4.7 Emitters (Front-end communication)

    def _wrap_safe_event_emitter(
        self,
        emitter: EventEmitter | None,
    ) -> EventEmitter | None:
        """Return an emitter wrapper that swallows downstream transport errors."""

        if emitter is None:
            return None

        async def _guarded(event: dict[str, Any]) -> None:
            try:
                await emitter(event)
            except Exception as exc:  # pragma: no cover - emitter transport errors
                evt_type = event.get("type") if isinstance(event, dict) else None
                suffix = f" ({evt_type})" if evt_type else ""
                self.logger.warning("Event emitter failure%s: %s", suffix, exc)

        return _guarded

    def _make_middleware_stream_emitter(
        self,
        job: _PipeJob,
        stream_queue: asyncio.Queue[dict[str, Any] | str | None],
    ) -> EventEmitter:
        """Translate internal events into middleware-supported streaming output.

        Open WebUI's `process_chat_response` middleware consumes OpenAI-style
        streaming chunks (``choices[].delta``) and supports out-of-band events
        via a top-level ``event`` key. This adapter ensures:

        - assistant deltas become ``delta.content`` chunks
        - reasoning traces become ``delta.reasoning_content`` chunks
        - status/citation/notification/etc are forwarded via ``{"event": ...}``
        """

        model_id = ""
        metadata_model = job.metadata.get("model") if isinstance(job.metadata, dict) else None
        if isinstance(metadata_model, dict):
            model_id = str(metadata_model.get("id") or "")
        if not model_id:
            model_id = str(job.body.get("model") or "pipe")

        assistant_sent = ""
        answer_started = False
        reasoning_status_buffer = ""
        reasoning_status_last_emit: float | None = None

        thinking_mode = job.valves.THINKING_OUTPUT_MODE
        thinking_box_enabled = thinking_mode in {"open_webui", "both"}
        thinking_status_enabled = thinking_mode in {"status", "both"}

        async def _emit(event: dict[str, Any]) -> None:
            nonlocal assistant_sent, answer_started
            if not isinstance(event, dict):
                return

            etype = event.get("type")
            raw_data = event.get("data")
            data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}

            async def _maybe_emit_reasoning_status(delta_text: str, *, force: bool = False) -> None:
                """Emit status updates for late-arriving reasoning without spamming the UI."""
                nonlocal reasoning_status_buffer, reasoning_status_last_emit
                if not isinstance(delta_text, str):
                    return
                reasoning_status_buffer += delta_text
                text = reasoning_status_buffer.strip()
                if not text:
                    return
                should_emit = force
                now = perf_counter()
                if not should_emit:
                    if delta_text.rstrip().endswith(_REASONING_STATUS_PUNCTUATION):
                        should_emit = True
                    elif len(text) >= _REASONING_STATUS_MAX_CHARS:
                        should_emit = True
                    else:
                        elapsed = None if reasoning_status_last_emit is None else (now - reasoning_status_last_emit)
                        if len(text) >= _REASONING_STATUS_MIN_CHARS:
                            if elapsed is None or elapsed >= _REASONING_STATUS_IDLE_SECONDS:
                                should_emit = True
                if not should_emit:
                    return
                await stream_queue.put(
                    {
                        "event": {
                            "type": "status",
                            "data": {
                                "description": text,
                                "done": False,
                            },
                        }
                    }
                )
                reasoning_status_buffer = ""
                reasoning_status_last_emit = now

            if etype == "chat:message":
                delta = data.get("delta")
                content = data.get("content")
                delta_text: str | None = None

                if isinstance(delta, str) and delta:
                    delta_text = delta
                    if isinstance(content, str) and content.startswith(assistant_sent):
                        assistant_sent = content
                    else:
                        assistant_sent = assistant_sent + delta
                elif isinstance(content, str) and content:
                    if content.startswith(assistant_sent):
                        delta_text = content[len(assistant_sent) :]
                        assistant_sent = content
                if isinstance(delta_text, str) and delta_text:
                    answer_started = True
                    if thinking_status_enabled and reasoning_status_buffer:
                        await _maybe_emit_reasoning_status("", force=True)
                    await stream_queue.put(
                        openai_chat_chunk_message_template(model_id, delta_text)
                    )
                return

            if etype == "reasoning:delta":
                delta = data.get("delta")
                if isinstance(delta, str) and delta:
                    if thinking_box_enabled:
                        await stream_queue.put(
                            openai_chat_chunk_message_template(
                                model_id,
                                reasoning_content=delta,
                            )
                        )
                    if thinking_status_enabled:
                        await _maybe_emit_reasoning_status(delta)
                return

            if etype == "reasoning:completed":
                if thinking_status_enabled and reasoning_status_buffer:
                    await _maybe_emit_reasoning_status("", force=True)
                return

            if etype == "chat:completion":
                if thinking_status_enabled and reasoning_status_buffer:
                    await _maybe_emit_reasoning_status("", force=True)
                error = data.get("error")
                if isinstance(error, dict) and error:
                    await stream_queue.put({"error": error})

                usage = data.get("usage")
                if isinstance(usage, dict) and usage:
                    await stream_queue.put({"usage": usage})
                return

            await stream_queue.put({"event": event})

        return _emit

    async def _report_openrouter_error(
        self,
        exc: OpenRouterAPIError,
        *,
        event_emitter: EventEmitter | None,
        normalized_model_id: Optional[str],
        api_model_id: Optional[str],
        usage: Optional[dict[str, Any]] = None,
        template: Optional[str] = None,
    ) -> None:
        """Emit a user-facing markdown message for OpenRouter 400 responses."""
        error_id, context_defaults = self._build_error_context()
        template_to_use = template or self._select_openrouter_template(exc.status)
        retry_after_hint = (
            exc.metadata.get("retry_after_seconds")
            or exc.metadata.get("retry_after")
        )
        if retry_after_hint and not context_defaults.get("retry_after_seconds"):
            context_defaults["retry_after_seconds"] = retry_after_hint
        self.logger.warning("[%s] OpenRouter rejected the request: %s", error_id, exc)
        if event_emitter:
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "description": "Encountered a provider error. See details below.",
                        "done": True,
                    },
                }
            )
            content = _format_openrouter_error_markdown(
                exc,
                normalized_model_id=normalized_model_id,
                api_model_id=api_model_id,
                template=template_to_use or DEFAULT_OPENROUTER_ERROR_TEMPLATE,
                context=context_defaults,
            )
            await event_emitter({"type": "chat:message", "data": {"content": content}})
            await self._emit_completion(
                event_emitter,
                content="",
                usage=usage or None,
                done=True,
            )

    def _select_openrouter_template(self, status: Optional[int]) -> str:
        """Return the appropriate template based on the HTTP status."""
        if status == 401:
            return self.valves.AUTHENTICATION_ERROR_TEMPLATE
        if status == 402:
            return self.valves.INSUFFICIENT_CREDITS_TEMPLATE
        if status == 408:
            return self.valves.SERVER_TIMEOUT_TEMPLATE
        if status == 429:
            return self.valves.RATE_LIMIT_TEMPLATE
        return self.valves.OPENROUTER_ERROR_TEMPLATE
    async def _emit_error(
        self,
        event_emitter: EventEmitter | None,
        error_obj: Exception | str,
        *,
        show_error_message: bool = True,
        show_error_log_citation: bool = False,
        done: bool = False,
    ) -> None:
        """Log an error and optionally surface it to the UI.

        When ``show_error_log_citation`` is true the collected debug logs are
        dumped to the server log (instead of the UI) so developers can inspect
        what went wrong.
        """
        error_message = str(error_obj)  # If it's an exception, convert to string
        self.logger.error("Error: %s", error_message)

        if show_error_message and event_emitter:
            await event_emitter(
                {
                    "type": "chat:completion",
                    "data": {
                        "error": {"message": error_message},
                        "done": done,
                    },
                }
            )

        # 2) Optionally dump the collected logs to the backend logger
        if show_error_log_citation:
            request_id = SessionLogger.request_id.get()
            logs = SessionLogger.logs.get(request_id or "", [])
            if logs:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug("Error logs for request %s:\n%s", request_id, "\n".join(logs))
            else:
                self.logger.warning("No debug logs found for request_id %s", request_id)

    async def _emit_templated_error(
        self,
        event_emitter: EventEmitter | None,
        *,
        template: str,
        variables: dict[str, Any],
        log_message: str,
        log_level: int = logging.ERROR,
    ) -> None:
        """Render and emit an error using the template system.

        Automatically enriches variables with:
        - error_id: Unique identifier for support correlation
        - timestamp: ISO 8601 timestamp (UTC)
        - session_id: Current session identifier
        - user_id: Current user identifier
        - support_email: From valves
        - support_url: From valves

        Args:
            event_emitter: Event emitter for UI messages
            template: Markdown template with {{#if}} conditionals
            variables: Dictionary of template variables
            log_message: Technical message for operator logs
            log_level: Logging level (default: ERROR)
        """
        error_id, context_defaults = self._build_error_context()
        enriched_variables = {**context_defaults, **variables}

        # Log with error ID for correlation
        self.logger.log(
            log_level,
            f"[{error_id}] {log_message} (session={enriched_variables['session_id']}, user={enriched_variables['user_id']})"
        )

        if not event_emitter:
            return

        # Render template
        try:
            markdown = _render_error_template(template, enriched_variables)
        except Exception as e:
            self.logger.error(f"[{error_id}] Template rendering failed: {e}")
            markdown = (
                f"### ⚠️ Error\n\n"
                f"An error occurred, but we couldn't format the error message properly.\n\n"
                f"**Error ID:** `{error_id}` (share this with support)\n\n"
                f"Please contact your administrator."
            )

        # Emit to UI
        try:
            await event_emitter({
                "type": "chat:message",
                "data": {"content": markdown}
            })
            await event_emitter({
                "type": "chat:completion",
                "data": {"done": True}
            })
        except Exception as e:
            self.logger.error(f"[{error_id}] Failed to emit error message: {e}")

    def _build_error_context(self) -> tuple[str, dict[str, Any]]:
        """Return a unique error id plus contextual metadata for templates."""
        error_id = secrets.token_hex(8)
        context = {
            "error_id": error_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "session_id": SessionLogger.session_id.get() or "",
            "user_id": SessionLogger.user_id.get() or "",
            "support_email": self.valves.SUPPORT_EMAIL,
            "support_url": self.valves.SUPPORT_URL,
        }
        return error_id, context

    def _extract_streaming_error_event(
        self,
        event: dict[str, Any] | None,
        requested_model: Optional[str],
    ) -> Optional[OpenRouterAPIError]:
        """Return an OpenRouterAPIError for SSE error payloads, if present."""
        if not isinstance(event, dict):
            return None
        event_data: dict[str, Any] = event
        event_type = (event_data.get("type") or "").strip()
        response_raw = event_data.get("response")
        response_block = response_raw if isinstance(response_raw, dict) else None
        error_raw = event_data.get("error")
        error_block = error_raw if isinstance(error_raw, dict) else None
        has_error = error_block is not None
        if isinstance(response_block, dict):
            if response_block.get("status") == "failed" or isinstance(response_block.get("error"), dict):
                has_error = True
        if event_type in {"response.failed", "response.error", "error"}:
            has_error = True
        if not has_error:
            return None
        return self._build_streaming_openrouter_error(event_data, requested_model=requested_model)

    def _build_streaming_openrouter_error(
        self,
        event: dict[str, Any],
        *,
        requested_model: Optional[str],
    ) -> OpenRouterAPIError:
        """Normalize SSE error events into an OpenRouterAPIError."""
        response_value = event.get("response")
        response_block: dict[str, Any] = response_value if isinstance(response_value, dict) else {}
        error_value = event.get("error")
        error_block = error_value if isinstance(error_value, dict) else None
        if not error_block and isinstance(response_block.get("error"), dict):
            error_block = response_block.get("error")
        message = ""
        if isinstance(error_block, dict):
            message = (error_block.get("message") or "").strip()
        if not message and isinstance(response_block.get("error"), dict):
            message = (response_block.get("error", {}).get("message") or "").strip()
        if not message:
            message = (event.get("message") or "").strip() or "Streaming error"
        code = error_block.get("code") if isinstance(error_block, dict) else None
        choices = event.get("choices") or response_block.get("choices")
        native_finish_reason = None
        if isinstance(choices, list) and choices:
            first_choice = choices[0] or {}
            native_finish_reason = first_choice.get("native_finish_reason") or first_choice.get("finish_reason")
        chunk_id = event.get("id") or response_block.get("id")
        chunk_created = event.get("created") or response_block.get("created")
        chunk_model = event.get("model") or response_block.get("model")
        chunk_provider = event.get("provider") or response_block.get("provider")
        metadata: dict[str, Any] = {
            "stream_event_type": event.get("type") or "",
            "raw": event,
        }
        if isinstance(response_block, dict):
            if response_block.get("status"):
                metadata["response_status"] = response_block.get("status")
            if response_block.get("error"):
                metadata["response_error"] = response_block.get("error")
            if response_block.get("id"):
                metadata.setdefault("request_id", response_block.get("id"))
        raw_body = _pretty_json(event)
        return OpenRouterAPIError(
            status=400,
            reason=message,
            provider=chunk_provider,
            openrouter_message=message,
            openrouter_code=code,
            upstream_message=message,
            upstream_type=(code or event.get("type") or "stream_error"),
            request_id=response_block.get("id") or event.get("response_id") or event.get("request_id"),
            raw_body=raw_body,
            metadata=metadata,
            moderation_reasons=None,
            flagged_input=None,
            model_slug=chunk_model,
            requested_model=requested_model,
            metadata_json=_pretty_json(metadata),
            provider_raw=event,
            provider_raw_json=raw_body,
            native_finish_reason=native_finish_reason,
            chunk_id=chunk_id,
            chunk_created=chunk_created,
            chunk_provider=chunk_provider,
            chunk_model=chunk_model,
            is_streaming_error=True,
        )

    async def _emit_citation(
        self,
        event_emitter: EventEmitter | None,
        citation: Dict[str, Any],
    ) -> None:
        """Send a normalized citation block to the UI if an emitter is available."""
        if event_emitter is None or not isinstance(citation, dict):
            return

        documents_raw = citation.get("document")
        if isinstance(documents_raw, str):
            documents = [documents_raw] if documents_raw.strip() else []
        elif isinstance(documents_raw, list):
            documents = [str(doc).strip() for doc in documents_raw if str(doc).strip()]
        else:
            documents = []
        if not documents:
            documents = ["Citation"]

        metadata_raw = citation.get("metadata")
        if isinstance(metadata_raw, list):
            metadata = [m for m in metadata_raw if isinstance(m, dict)]
        else:
            metadata = []

        source_info = citation.get("source")
        if isinstance(source_info, dict):
            source_name = (source_info.get("name") or source_info.get("url") or "source").strip() or "source"
            if not source_info.get("name"):
                source_info = dict(source_info)
                source_info["name"] = source_name
        else:
            source_name = "source"
            source_info = {"name": source_name}

        if not metadata:
            metadata = [
                {
                    "date_accessed": datetime.datetime.now().isoformat(),
                    "source": source_info.get("url") or source_name,
                }
            ]

        await event_emitter(
            {
                "type": "citation",
                "data": {
                    "document": documents,
                    "metadata": metadata,
                    "source": source_info,
                },
            }
        )

    async def _emit_completion(
        self,
        event_emitter: EventEmitter | None,
        *,
        content: str | None = "",                       # always included (may be "").  UI will stall if you leave it out.
        title:   str | None = None,                     # optional title.
        usage:   dict[str, Any] | None = None,          # optional usage block
        done:    bool = True,                           # True → final frame
    ) -> None:
        """Emit a ``chat:completion`` event if an emitter is present.

        The ``done`` flag indicates whether this is the final frame for the
        request.  When ``usage`` information is provided it is forwarded as part
        of the event data. Callers should pass the latest assistant snapshot so
        downstream emitters cannot overwrite the UI with an empty string.
        """
        if event_emitter is None:
            return

        # Note: Open WebUI emits a final "chat:completion" event after the stream ends, which overwrites any previously emitted completion events' content and title in the UI.
        await event_emitter(
            {
                "type": "chat:completion",
                "data": {
                    "done": done,
                    "content": content,
                    **({"title": title} if title is not None else {}),
                    **({"usage": usage} if usage is not None else {}),
                }
            }
        )

    async def _emit_notification(
        self,
        event_emitter: EventEmitter | None,
        content: str,
        *,
        level: Literal["info", "success", "warning", "error"] = "info",
    ) -> None:
        """Emit a toast-style notification to the UI.

        The ``level`` argument controls the styling of the notification banner.
        """
        if event_emitter is None:
            return

        await event_emitter(
            {"type": "notification", "data": {"type": level, "content": content}}
        )

    def _format_final_status_description(
        self,
        *,
        elapsed: float,
        total_usage: Dict[str, Any],
        valves: "Pipe.Valves",
        stream_duration: Optional[float] = None,
    ) -> str:
        """Return the final status line respecting valve + available metrics.

        ``stream_duration`` is expected to mirror provider dashboards (request
        start → last output event, which includes first-token latency).
        """
        default_description = f"Thought for {elapsed:.1f} seconds"
        if not valves.SHOW_FINAL_USAGE_STATUS:
            return default_description

        usage = total_usage or {}
        time_segment = f"Time: {elapsed:.2f}s"
        tokens_for_tps: Optional[int] = None
        segments: list[str] = []

        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and cost > 0:
            cost_str = f"{cost:.6f}".rstrip("0").rstrip(".")
            segments.append(f"Cost ${cost_str}")

        def _to_int(value: Any) -> Optional[int]:
            """Best-effort conversion to ``int`` for usage counters."""
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            return None

        input_tokens = _to_int(usage.get("input_tokens"))
        output_tokens = _to_int(usage.get("output_tokens"))
        total_tokens = _to_int(usage.get("total_tokens"))
        if total_tokens is None:
            candidates = [v for v in (input_tokens, output_tokens) if v is not None]
            if candidates:
                total_tokens = sum(candidates)
        if output_tokens is not None:
            tokens_for_tps = output_tokens
        elif total_tokens is not None:
            tokens_for_tps = total_tokens

        cached_tokens = _to_int(
            (usage.get("input_tokens_details") or {}).get("cached_tokens")
        )
        reasoning_tokens = _to_int(
            (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        )

        token_details: list[str] = []
        if input_tokens is not None:
            token_details.append(f"Input: {input_tokens}")
        if output_tokens is not None:
            token_details.append(f"Output: {output_tokens}")
        if cached_tokens is not None and cached_tokens > 0:
            token_details.append(f"Cached: {cached_tokens}")
        if reasoning_tokens is not None and reasoning_tokens > 0:
            token_details.append(f"Reasoning: {reasoning_tokens}")

        if total_tokens is not None:
            token_segment = f"Total tokens: {total_tokens}"
            if token_details:
                token_segment += f" ({', '.join(token_details)})"
            segments.append(token_segment)
        elif token_details:
            segments.append("Tokens: " + ", ".join(token_details))

        if (
            tokens_for_tps is not None
            and stream_duration is not None
            and stream_duration > 0
        ):
            tokens_per_second = tokens_for_tps / stream_duration
            if tokens_per_second > 0:
                time_segment = f"{time_segment}  {tokens_per_second:.1f} tps"

        segments.insert(0, time_segment)

        description = " | ".join(segments)
        return description or default_description

    # 4.8 Internal Static Helpers
    def _merge_valves(self, global_valves, user_valves) -> "Pipe.Valves":
        """Merge user-level valves into the global defaults.

        Any field set to ``"INHERIT"`` (case-insensitive) is ignored so the
        corresponding global value is preserved.
        """
        if not user_valves:
            return global_valves

        overrides: dict[str, Any] = {}
        if isinstance(user_valves, BaseModel):
            fields_set = getattr(user_valves, "model_fields_set", set()) or set()
            for field_name in fields_set:
                value = getattr(user_valves, field_name, None)
                if value is None:
                    continue
                overrides[field_name] = value
        elif isinstance(user_valves, dict):
            overrides = {
                key: value
                for key, value in user_valves.items()
                if value is not None and str(value).lower() != "inherit"
            }

        if not overrides:
            return global_valves

        mapped: dict[str, Any] = {}
        for key, value in overrides.items():
            target_key = key
            if not hasattr(global_valves, target_key):
                if key == "next_reply":
                    target_key = "PERSIST_REASONING_TOKENS"
                elif key == "PERSIST_REASONING_TOKENS" and not hasattr(global_valves, key):
                    continue
                elif not hasattr(global_valves, target_key):
                    continue
            mapped[target_key] = value

        if not mapped:
            return global_valves

        # Do not allow per-user overrides of the global log level.
        mapped.pop("LOG_LEVEL", None)

        return global_valves.model_copy(update=mapped)


def _extract_internal_file_id(url: str) -> Optional[str]:
    """Return the Open WebUI file identifier embedded in a storage URL."""
    if not isinstance(url, str):
        return None
    match = _INTERNAL_FILE_ID_PATTERN.search(url)
    if match:
        return match.group(1)
    return None


def _is_internal_file_url(url: str) -> bool:
    """Heuristic to detect when a URL references Open WebUI file storage."""
    if not isinstance(url, str):
        return False
    return "/api/v1/files/" in url or "/files/" in url


# ─────────────────────────────────────────────────────────────────────────────
# 5. Utility & Helper Layer (organized, consistent docstrings)
#    NOTE: Logic is unchanged. Only docstrings/comments/sectioning were improved.
# ─────────────────────────────────────────────────────────────────────────────

# 5.1 Logging & Diagnostics
# -------------------------

class SessionLogger:
    """Per-request logger that captures console output and an in-memory log buffer.

    The logger tracks two identifiers via contextvars:
    - session_id: Open WebUI session identifier (for status/errors/debug).
    - request_id: Per-request unique id used to key the in-memory log buffer.

    Cleanup is intentional and explicit: request handlers call ``cleanup`` once
    they finish streaming so there is no background task silently pruning logs.

    Attributes:
        session_id: ContextVar storing the Open WebUI session id.
        request_id: ContextVar storing the per-request buffer key.
        log_level:  ContextVar storing the minimum level to emit for this request.
        logs:       Map of request_id -> fixed-size deque of formatted log strings.
    """

    session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
    request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
    user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
    log_level: ContextVar[int] = ContextVar("log_level", default=logging.INFO)
    max_lines: int = 2000
    logs: Dict[str, deque[str]] = {}
    _session_last_seen: Dict[str, float] = {}
    log_queue: asyncio.Queue[logging.LogRecord] | None = None
    _main_loop: asyncio.AbstractEventLoop | None = None
    _console_formatter = logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _memory_formatter = logging.Formatter("%(asctime)s [%(levelname)s] [user=%(user_id)s] %(message)s")

    @classmethod
    def get_logger(cls, name=__name__):
        """Create a logger wired to the current SessionLogger context.

        Args:
            name: Logger name; defaults to the current module name.

        Returns:
            logging.Logger: A configured logger that writes both to stdout and
            the in-memory `SessionLogger.logs` buffer. The buffer is keyed by
            the current `SessionLogger.request_id`.
        """
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.filters.clear()
        logger.setLevel(logging.DEBUG)
        root_logger = logging.getLogger()
        if not any(isinstance(handler, logging.NullHandler) for handler in root_logger.handlers):
            root_logger.addHandler(logging.NullHandler())
        logger.propagate = True

        # Single combined filter: attach session_id and respect per-session level.
        def filter(record):
            """Attach session metadata and capture the per-request console log level."""
            try:
                sid = cls.session_id.get()
                rid = cls.request_id.get()
                uid = cls.user_id.get()
                record.session_id = sid
                record.request_id = rid
                record.user_id = uid or "-"
                record.session_log_level = cls.log_level.get()
                if rid:
                    cls._session_last_seen[rid] = time.time()
            except Exception:
                # Logging must never break request handling.
                pass
            return True

        logger.addFilter(filter)

        async_handler = logging.Handler()

        def _emit(record: logging.LogRecord) -> None:
            cls._enqueue(record)

        async_handler.emit = _emit  # type: ignore[assignment]
        logger.addHandler(async_handler)

        return logger

    @classmethod
    def set_log_queue(cls, queue: asyncio.Queue[logging.LogRecord] | None) -> None:
        cls.log_queue = queue
    @classmethod
    def set_main_loop(cls, loop: asyncio.AbstractEventLoop | None) -> None:
        cls._main_loop = loop

    @classmethod
    def set_max_lines(cls, value: int) -> None:
        """Set the maximum in-memory lines retained per request (best effort)."""
        try:
            value_int = int(value)
        except Exception:
            return
        value_int = max(100, min(200000, value_int))
        cls.max_lines = value_int

    @classmethod
    def _enqueue(cls, record: logging.LogRecord) -> None:
        queue = cls.log_queue
        if queue is None:
            cls.process_record(record)
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop and running_loop is cls._main_loop:
            cls._safe_put(queue, record)
            return

        main_loop = cls._main_loop
        if main_loop and not main_loop.is_closed():
            main_loop.call_soon_threadsafe(cls._safe_put, queue, record)
        else:
            cls.process_record(record)

    @classmethod
    def _safe_put(cls, queue: asyncio.Queue[logging.LogRecord], record: logging.LogRecord) -> None:
        try:
            queue.put_nowait(record)
        except asyncio.QueueFull:
            cls.process_record(record)

    @classmethod
    def process_record(cls, record: logging.LogRecord) -> None:
        try:
            session_log_level = getattr(record, "session_log_level", logging.INFO)
            if record.levelno >= int(session_log_level):
                try:
                    console_line = cls._console_formatter.format(record)
                    sys.stdout.write(console_line + "\n")
                    sys.stdout.flush()
                except Exception:
                    pass
            request_id = getattr(record, "request_id", None)
            if request_id:
                buffer = cls.logs.get(request_id)
                if buffer is None or buffer.maxlen != cls.max_lines:
                    buffer = deque(maxlen=cls.max_lines)
                    cls.logs[request_id] = buffer
                try:
                    buffer.append(cls._memory_formatter.format(record))
                    cls._session_last_seen[request_id] = time.time()
                except Exception:
                    pass
        except Exception:
            # Never raise from logging hooks.
            return

    @classmethod
    def cleanup(cls, max_age_seconds: float = 3600) -> None:
        """Remove stale session logs to avoid unbounded growth."""
        cutoff = time.time() - max_age_seconds
        stale = [sid for sid, ts in cls._session_last_seen.items() if ts < cutoff]
        for sid in stale:
            cls.logs.pop(sid, None)
            cls._session_last_seen.pop(sid, None)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Framework Integration Helpers (Open WebUI DB operations)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 7. General-Purpose Utilities (data transforms & patches)
# ─────────────────────────────────────────────────────────────────────────────

def _wrap_event_emitter(
    emitter: EventEmitter | None,
    *,
    suppress_chat_messages: bool = False,
    suppress_completion: bool = False,
):
    """
    Wrap the given event emitter and optionally suppress specific event types.

    Use-case: reuse the streaming loop for non-stream requests by swallowing
    incremental 'chat:message' frames while allowing status/citation/usage
    events through.
    """
    if emitter is None:
        async def _noop(_event: Dict[str, Any]) -> None:
            """Swallow events when no emitter is provided."""
            return

        return _noop

    async def _wrapped(event: Dict[str, Any]) -> None:
        """Proxy emitter that suppresses selected event types."""
        etype = (event or {}).get("type")
        if suppress_chat_messages and etype == "chat:message":
            return  # swallow incremental deltas
        if suppress_completion and etype == "chat:completion":
            return  # optionally swallow completion frames
        await emitter(event)

    return _wrapped

def merge_usage_stats(total, new):
    """Recursively merge nested usage statistics.

    For numeric values, sums are accumulated; for dicts, the function recurses;
    other values overwrite the prior value when non-None.

    Args:
        total: Accumulator dictionary to update.
        new:   Newly reported usage block to merge into `total`.

    Returns:
        dict: The updated accumulator dictionary (`total`).
    """
    for k, v in new.items():
        if isinstance(v, dict):
            total[k] = merge_usage_stats(total.get(k, {}), v)
        elif isinstance(v, (int, float)):
            total[k] = total.get(k, 0) + v
        else:
            total[k] = v if v is not None else total.get(k, 0)
    return total


def wrap_code_block(text: str, language: str = "python") -> str:
    """Wrap text in a fenced Markdown code block.

    The fence length adapts to the longest backtick run within the text to avoid
    prematurely closing the block.

    Args:
        text:     The code or content to wrap.
        language: Markdown fence language tag.

    Returns:
        str: Markdown code block.
    """
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _normalize_persisted_item(item: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Ensure persisted response artifacts match the schema expected by the
    Responses API when replayed via the `input` array.
    """
    if not isinstance(item, dict):
        return None

    item_type = item.get("type")
    if not item_type:
        return None

    normalized = dict(item)

    def _ensure_identity(status_default: str = "completed") -> None:
        """Guarantee persisted artifacts include ``id`` and ``status`` fields."""
        normalized.setdefault("id", generate_item_id())
        status = normalized.get("status") or status_default
        normalized["status"] = status

    if item_type == "function_call_output":
        _ensure_identity()
        normalized["call_id"] = normalized.get("call_id") or generate_item_id()
        output_value = normalized.get("output")
        normalized["output"] = "" if output_value is None else str(output_value)
        return normalized

    if item_type == "function_call":
        name = normalized.get("name")
        arguments = normalized.get("arguments")
        if not name or arguments is None:
            return None
        if not isinstance(arguments, str):
            try:
                normalized["arguments"] = json.dumps(arguments)
            except (TypeError, ValueError):
                # Fallback to str() if arguments aren't JSON serializable
                normalized["arguments"] = str(arguments)
        normalized["call_id"] = normalized.get("call_id") or generate_item_id()
        _ensure_identity()
        return normalized

    if item_type == "reasoning":
        content = normalized.get("content")
        if isinstance(content, list):
            normalized["content"] = content
        elif content:
            normalized["content"] = [{"type": "reasoning_text", "text": str(content)}]
        else:
            normalized["content"] = []
        summary = normalized.get("summary")
        if not isinstance(summary, list):
            normalized["summary"] = [] if summary in (None, "") else [summary]
        _ensure_identity()
        return normalized

    if item_type in {
        "web_search_call",
        "file_search_call",
        "image_generation_call",
        "local_shell_call",
    }:
        _ensure_identity()
        if item_type == "file_search_call":
            queries = normalized.get("queries")
            if not isinstance(queries, list):
                normalized["queries"] = []
        if item_type == "web_search_call":
            action = normalized.get("action")
            if not isinstance(action, dict):
                normalized["action"] = {}
        return normalized

    return item


def _classify_function_call_artifacts(
    artifacts: Dict[str, Dict[str, Any]]
) -> tuple[set[str], set[str], set[str]]:
    """
    Inspect persisted artifacts and return three identifier sets:

    - valid_ids: call_ids that have both a function_call and function_call_output
    - orphaned_calls: call_ids that only have a function_call entry
    - orphaned_outputs: call_ids that only have a function_call_output entry
    """
    call_ids: set[str] = set()
    output_ids: set[str] = set()

    for payload in artifacts.values():
        if not isinstance(payload, dict):
            continue
        call_id = payload.get("call_id")
        if not isinstance(call_id, str):
            continue
        call_id = call_id.strip()
        if not call_id:
            continue
        payload_type = (payload.get("type") or "").lower()
        if payload_type == "function_call":
            call_ids.add(call_id)
        elif payload_type == "function_call_output":
            output_ids.add(call_id)

    valid_ids = call_ids & output_ids
    orphaned_calls = call_ids - valid_ids
    orphaned_outputs = output_ids - valid_ids
    return valid_ids, orphaned_calls, orphaned_outputs


# ─────────────────────────────────────────────────────────────────────────────
# 8. Persistent Item Markers (ULIDs & encoding helpers)
# ─────────────────────────────────────────────────────────────────────────────

# Constants for ULID-like IDs used in hidden markers.
ULID_LENGTH = 20
ULID_TIME_LENGTH = 16
ULID_RANDOM_LENGTH = ULID_LENGTH - ULID_TIME_LENGTH
CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_CROCKFORD_SET = frozenset(CROCKFORD_ALPHABET)
_ULID_TIME_MASK = (1 << (ULID_TIME_LENGTH * 5)) - 1
_MARKER_SUFFIX = "]: #"

_TOOL_OUTPUT_PRUNE_MIN_LENGTH = 800
_TOOL_OUTPUT_PRUNE_HEAD_CHARS = 256
_TOOL_OUTPUT_PRUNE_TAIL_CHARS = 128

def _encode_crockford(value: int, length: int) -> str:
    """Encode an integer into a fixed-width Crockford base32 string."""
    if value < 0:
        raise ValueError("value must be non-negative")
    chars = ["0"] * length
    for idx in range(length - 1, -1, -1):
        chars[idx] = CROCKFORD_ALPHABET[value & 0x1F]
        value >>= 5
    return "".join(chars)


def generate_item_id() -> str:
    """Generate a 20-char ULID using a 16-char time component + 4-char random tail.

    Returns:
        str: Crockford-encoded ULID (stateless + monotonic per timestamp).
    """
    timestamp = time.time_ns() & _ULID_TIME_MASK
    time_component = _encode_crockford(timestamp, ULID_TIME_LENGTH)
    random_bits = secrets.randbits(ULID_RANDOM_LENGTH * 5)
    random_component = _encode_crockford(random_bits, ULID_RANDOM_LENGTH)
    return f"{time_component}{random_component}"


def _serialize_marker(ulid: str) -> str:
    """Return the hidden marker representation for ``ulid``."""
    return f"[{ulid}{_MARKER_SUFFIX}"


def _extract_marker_ulid(line: str) -> str | None:
    """Return the ULID embedded in a hidden marker line, if present."""
    if not line:
        return None
    stripped = line.strip()
    if not stripped.startswith("[") or not stripped.endswith(_MARKER_SUFFIX):
        return None
    body = stripped[1 : -len(_MARKER_SUFFIX)]
    if len(body) != ULID_LENGTH:
        return None
    for char in body:
        if char not in _CROCKFORD_SET:
            return None
    return body



def contains_marker(text: str) -> bool:
    """Fast check: does the text contain any embedded ULID markers?

    Args:
        text: Text to scan.

    Returns:
        bool: True if the sentinel substring is present; otherwise False.
    """
    return bool(_iter_marker_spans(text))


def _iter_marker_spans(text: str) -> list[dict[str, Any]]:
    """Return ordered ULID marker spans."""
    if not text:
        return []

    spans: list[dict[str, Any]] = []
    cursor = 0
    for segment in text.splitlines(True):
        stripped = segment.strip()
        marker_ulid = _extract_marker_ulid(stripped)
        if marker_ulid:
            offset = segment.find(stripped)
            start = cursor + (offset if offset >= 0 else 0)
            spans.append(
                {
                    "start": start,
                    "end": start + len(stripped),
                    "marker": marker_ulid,
                }
            )
        cursor += len(segment)

    spans.sort(key=lambda span: span["start"])
    return spans



def split_text_by_markers(text: str) -> list[dict]:
    """Split text into a sequence of literal segments and marker segments.

    Args:
        text: Source text possibly containing embedded markers.

    Returns:
        list[dict]: A list like:
            [
              {"type": "text",   "text": "..."},
              {"type": "marker", "marker": "01H...Q4"},
              ...
            ]
    """
    segments: list[dict[str, Any]] = []
    last = 0
    for span in _iter_marker_spans(text):
        if span["start"] > last:
            segments.append({"type": "text", "text": text[last:span["start"]]})
        segments.append(
            {
                "type": "marker",
                "marker": span["marker"],
            }
        )
        last = span["end"]
    if last < len(text):
        segments.append({"type": "text", "text": text[last:]})
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# 9. Database & Persistence Infrastructure
# ─────────────────────────────────────────────────────────────────────────────


def _sanitize_table_fragment(value: str) -> str:
    """Normalize arbitrary identifiers into safe SQL table suffixes."""
    fragment = re.sub(r"[^a-z0-9_]", "_", (value or "").lower())
    fragment = fragment.strip("_") or "pipe"
    if len(fragment) > 62:
        fragment = fragment[:62].rstrip("_") or "pipe"
    return fragment


def _sanitize_path_component(value: str, *, fallback: str = "unknown", max_length: int = 128) -> str:
    """Return a filesystem-safe path component to prevent traversal/odd characters."""
    text = str(value or "").strip()
    if not text:
        return fallback
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    cleaned = cleaned.strip("._-") or fallback
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("._-") or fallback
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# 9. Tool & Schema Utilities (internal)
# ─────────────────────────────────────────────────────────────────────────────
def build_tools(
    responses_body: "ResponsesBody",
    valves: "Pipe.Valves",
    __tools__: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    *,
    features: Optional[Dict[str, Any]] = None,
    extra_tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build the OpenAI Responses-API tool spec list for this request.

    - Returns [] if the target model doesn't support function calling.
    - Includes Open WebUI registry tools (strictified if enabled).
    - Adds OpenAI web_search (if allowed + supported + not minimal effort).
    - Appends any caller-provided extra_tools (already-valid OpenAI tool specs).
    - Deduplicates by (type,name) identity; last one wins.

    NOTE: This builds the *schema* to send to OpenAI. For executing function
    calls at runtime, you can keep passing the raw `__tools__` registry into
    your streaming/non-streaming loops; those functions expect name→callable.
    """
    features = features or {}

    # 1) If model can't do function calling, no tools
    if not ModelFamily.supports("function_calling", responses_body.model):
        return []

    tools: List[Dict[str, Any]] = []

    # 2) Baseline: Open WebUI registry tools → OpenAI tool specs
    if isinstance(__tools__, dict) and __tools__:
        tools.extend(
            ResponsesBody.transform_owui_tools(
                __tools__,
                strict=valves.ENABLE_STRICT_TOOL_CALLING,
            )
        )
    elif isinstance(__tools__, list) and __tools__:
        tools.extend([tool for tool in __tools__ if isinstance(tool, dict)])

    # 3) Optional extra tools (already OpenAI-format)
    if isinstance(extra_tools, list) and extra_tools:
        tools.extend(extra_tools)

    return _dedupe_tools(tools)


_STRICT_SCHEMA_CACHE_SIZE = 128


@functools.lru_cache(maxsize=_STRICT_SCHEMA_CACHE_SIZE)
def _strictify_schema_cached(serialized_schema: str) -> str:
    """Cached worker that enforces strict schema rules on serialized JSON."""
    schema_dict = json.loads(serialized_schema)
    strict_schema = _strictify_schema_impl(schema_dict)
    return json.dumps(strict_schema, ensure_ascii=False)


def _strictify_schema(schema):
    """
    Minimal, predictable transformer to make a JSON schema strict-compatible.

    Rules for every object node (root + nested):
      - additionalProperties := false
      - required := all property keys
      - fields that were optional become nullable (add "null" to their type)

    We traverse properties, items (dict or list), and anyOf/oneOf branches.
    We do NOT rewrite anyOf/oneOf; we only enforce object rules inside them.

    Returns a new dict. Non-dict inputs return {}.
    """
    if not isinstance(schema, dict):
        return {}

    canonical = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    cached = _strictify_schema_cached(canonical)
    return json.loads(cached)


def _strictify_schema_impl(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Internal implementation for `_strictify_schema` that assumes input is a fresh dict.

    Applies strict-mode transformations to JSON schemas:
    - Ensures all object nodes have `additionalProperties: false`
    - Makes all properties required
    - Makes optional properties nullable by adding "null" to their type
    - **Auto-infers missing types**: Adds default types to properties without one:
      * Empty schemas `{}` → `{"type": "object"}`
      * Schemas with `properties` but no type → `{"type": "object"}`
      * Schemas with `items` but no type → `{"type": "array"}`

    This defensive type inference ensures schemas are valid for OpenAI strict mode.
    """
    root_t = schema.get("type")
    if not (
        root_t == "object"
        or (isinstance(root_t, list) and "object" in root_t)
        or "properties" in schema
    ):
        schema = {
            "type": "object",
            "properties": {"value": schema},
            "required": ["value"],
            "additionalProperties": False,
        }

    stack = [schema]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue

        t = node.get("type")
        is_object = ("properties" in node) or (t == "object") or (
            isinstance(t, list) and "object" in t
        )
        if is_object:
            props = node.get("properties")
            if not isinstance(props, dict):
                props = {}
                node["properties"] = props

            raw_required = node.get("required") or []
            raw_required_names: list[str] = [
                name for name in raw_required if isinstance(name, str)
            ]
            all_property_names = list(props.keys())

            node["additionalProperties"] = False
            node["required"] = all_property_names

            explicitly_required = {name for name in raw_required_names if name in props}
            optional_candidates = {
                name for name in all_property_names if name not in explicitly_required
            }

            for name, p in props.items():
                if not isinstance(p, dict):
                    continue

                # Ensure every property schema has a type key (strict mode requirement)
                if "type" not in p:
                    schema_structure_keys = {"properties", "items", "anyOf", "oneOf", "allOf"}
                    has_nested_structure = any(k in p for k in schema_structure_keys)

                    if has_nested_structure:
                        if "properties" in p:
                            p["type"] = "object"
                            LOGGER.debug(
                                "Added inferred type 'object' to property '%s' which has 'properties' but no explicit type. "
                                "Consider fixing the schema definition at the source.",
                                name
                            )
                        elif "items" in p:
                            p["type"] = "array"
                            LOGGER.debug(
                                "Added inferred type 'array' to property '%s' which has 'items' but no explicit type. "
                                "Consider fixing the schema definition at the source.",
                                name
                            )
                        # For anyOf/oneOf/allOf without type, don't add a default
                        # Let OpenAI validation handle these complex cases
                    else:
                        # Empty or minimal schema (e.g., {"description": "..."} or just {})
                        # Default to object as the safest, most flexible type
                        p["type"] = "object"
                        LOGGER.debug(
                            "Added default type 'object' to property '%s' with no type or schema structure. "
                            "This indicates an incomplete schema definition that should be fixed at the source.",
                            name
                        )

                # Handle optional fields by adding null to type
                if name in optional_candidates:
                    ptype = p.get("type")
                    if isinstance(ptype, str) and ptype != "null":
                        p["type"] = [ptype, "null"]
                    elif isinstance(ptype, list) and "null" not in ptype:
                        p["type"] = ptype + ["null"]
                stack.append(p)

        items = node.get("items")
        if isinstance(items, dict):
            # Ensure items schema has a type key
            if "type" not in items and "properties" not in items and "items" not in items:
                items["type"] = "object"
                LOGGER.debug("Added default type 'object' to empty items schema")
            stack.append(items)
        elif isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    stack.append(it)

        for key in ("anyOf", "oneOf"):
            branches = node.get(key)
            if isinstance(branches, list):
                for br in branches:
                    if isinstance(br, dict):
                        # Ensure branch schema has a type key
                        if "type" not in br and "properties" not in br and "items" not in br:
                            br["type"] = "object"
                            LOGGER.debug("Added default type 'object' to empty %s branch", key)
                        stack.append(br)

    return schema


def _dedupe_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """(Internal) Deduplicate a tool list with simple, stable identity keys.

    Identity:
      - Function tools → key = ("function", <name>)
      - Non-function tools → key = (<type>, None)

    Later entries win (last write wins).

    Args:
        tools: List of tool dicts (OpenAI Responses schema).

    Returns:
        list: Deduplicated list, preserving only the last occurrence per identity.
    """
    if not tools:
        return []
    canonical: Dict[tuple, Dict[str, Any]] = {}
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function":
            key = ("function", t.get("name"))
        else:
            key = (t.get("type"), None)
        if key[0]:
            canonical[key] = t
    return list(canonical.values())
