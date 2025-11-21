"""
title: OpenRouter Responses API Manifold
author: rbb-dev
author_url: https://github.com/rbb-dev
git_url: https://github.com/
description: Brings OpenRouter Responses API support to Open WebUI, auto-importing every Responses-capable model from OpenRouter and translating Open WebUI requests into the new API format.
required_open_webui_version: 0.6.28
version: 1.0.5
requirements: aiohttp, cryptography, fastapi, lz4, pydantic, pydantic_core, sqlalchemy, tenacity
license: MIT

- Auto-discovers and imports full OpenRouter Responses model catalog with capabilities and identifiers.
- Translates Completions to Responses API, persisting reasoning/tool artifacts per chat via scoped SQLAlchemy tables.
- Handles 100-500 concurrent users with per-request isolation, async queues, and global semaphores for overload protection (503 rejects).
- Non-blocking ops: Offloads sync DB to ThreadPool, async logging queue, per-request HTTP sessions with retries/breakers.
- Optional Redis cache (auto-detected via ENV/multi-worker): Write-behind with pub/sub/timed flushes, TTL for fast artifact reads.
- Secure artifact persistence: User-key encryption, LZ4 compression for large payloads, ULID markers for context replay.
- Tool execution: Per-request FIFO queues, parallel workers with semaphores/timeouts, per-user/type breakers, batching non-dependent calls.
- Streams SSE with producer-multi-consumer workers, delta batching/zero-copy, inline citations, and usage metrics.
- Strictifies tool schemas (Open WebUI/MCP/plugins) for predictable function calling; deduplicates definitions.
- Auto-enables web search plugin if model-supported; configurable MCP servers for global tools.
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
import re
import sys
import secrets
import random
import time
import hashlib
import base64
import functools
from time import perf_counter
from collections import defaultdict, deque
from contextvars import ContextVar
import contextvars
from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Literal, Optional, Type, Union, TYPE_CHECKING
from urllib.parse import urlparse
import ast

# Third-party imports
import aiohttp
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import Boolean, Column, DateTime, JSON, String, create_engine, inspect as sa_inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

try:
    import lz4.frame as lz4frame
except ImportError:  # pragma: no cover - optional dependency, handled via requirements metadata
    lz4frame = None

try:  # optional Redis cache
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover - optional dependency
    aioredis = None

if TYPE_CHECKING:
    from redis.asyncio import Redis as _RedisClient
else:
    _RedisClient = Any
# Open WebUI internals
from open_webui.models.chats import Chats
from open_webui.models.models import ModelForm, Models
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _PipeJob:
    """Encapsulate a single OpenRouter request scheduled through the queue."""

    pipe: "Pipe"
    body: dict[str, Any]
    user: dict[str, Any]
    request: Request
    event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None
    event_call: Callable[[dict[str, Any]], Awaitable[Any]] | None
    metadata: dict[str, Any]
    tools: list[dict[str, Any]] | dict[str, Any] | None
    task: Optional[dict[str, Any]]
    task_body: Optional[dict[str, Any]]
    valves: "Pipe.Valves"
    future: asyncio.Future
    request_id: str = field(default_factory=lambda: secrets.token_hex(8))

    @property
    def session_id(self) -> str:
        return str(self.metadata.get("session_id") or "")

    @property
    def user_id(self) -> str:
        return str(self.user.get("id") or self.metadata.get("user_id") or "")


# Tool Execution ----------------------------------------------------
@dataclass(slots=True)
class _QueuedToolCall:
    call: dict[str, Any]
    tool_cfg: dict[str, Any]
    args: dict[str, Any]
    future: asyncio.Future
    allow_batch: bool


@dataclass(slots=True)
class _ToolExecutionContext:
    queue: asyncio.Queue[_QueuedToolCall | None]
    per_request_semaphore: asyncio.Semaphore
    global_semaphore: asyncio.Semaphore | None
    timeout: float
    batch_timeout: float | None
    idle_timeout: float | None
    user_id: str
    event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None
    workers: list[asyncio.Task] = field(default_factory=list)
    timeout_error: Optional[str] = None


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
        except (InvalidToken, Exception):
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
    _last_error: Optional[str] = None
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
        headers = {"Authorization": f"Bearer {api_key}"}
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
            supported = set(item.get("supported_parameters") or [])
            architecture = dict(item.get("architecture") or {})
            pricing = dict(item.get("pricing") or {})
            limits = item.get("limits")
            max_completion_tokens: Optional[int] = None
            if isinstance(limits, dict):
                max_completion_tokens = limits.get("max_completion_tokens")
            if max_completion_tokens is None:
                top_provider = item.get("top_provider")
                if isinstance(top_provider, dict):
                    max_completion_tokens = top_provider.get("max_completion_tokens")

            raw_specs[norm_id] = {
                "supported_parameters": supported,
                "architecture": architecture,
                "pricing": pricing,
                "max_completion_tokens": max_completion_tokens,
            }
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
        specs: Dict[str, Dict[str, Any]] = {}
        for norm_id, spec in raw_specs.items():
            features = cls._derive_features(
                spec.get("supported_parameters") or set(),
                spec.get("architecture") or {},
                spec.get("pricing") or {},
            )
            specs[norm_id] = {
                "features": features,
                "max_completion_tokens": spec.get("max_completion_tokens"),
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
        now = time.time()
        cls._last_fetch = now
        cls._next_refresh_after = now + max(5, cache_seconds)
        cls._consecutive_failures = 0
        cls._last_error = None
        cls._last_error_time = 0.0

    @classmethod
    def _record_refresh_failure(cls, exc: Exception, cache_seconds: int) -> None:
        cls._consecutive_failures += 1
        cls._last_error = str(exc)
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
        """Translate OpenRouter metadata into capability flags."""
        features: set[str] = set()
        if {"tools", "tool_choice"} & supported_parameters:
            features.add("function_calling")
        if "reasoning" in supported_parameters:
            features.add("reasoning")
        if "include_reasoning" in supported_parameters:
            features.add("reasoning_summary")
        if pricing.get("web_search") is not None:
            features.add("web_search_tool")
        output_modalities = architecture.get("output_modalities") or []
        if "image" in output_modalities:
            features.add("image_gen_tool")
        return features

    @classmethod
    def list_models(cls) -> list[dict[str, Any]]:
        """Return a shallow copy of the cached catalog."""
        return list(cls._models)


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


async def _debug_print_error_response(resp: aiohttp.ClientResponse) -> None:
    """Log the response payload for easier debugging of HTTP failures."""
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

    class Config:
        """Permit passthrough of additional OpenAI parameters automatically."""

        extra = "allow" # Pass through additional OpenAI parameters automatically

class ResponsesBody(BaseModel):
    """
    Represents the body of a responses request to OpenAI Responses API.
    """
    
    # Required parameters
    model: str
    input: Union[str, List[Dict[str, Any]]] # plain text, or rich array

    # Optional parameters
    stream: bool = False                          # SSE chunking
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    reasoning: Optional[Dict[str, Any]] = None    # {"effort":"high", ...}
    tool_choice: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    plugins: Optional[List[Dict[str, Any]]] = None

    class Config:
        """Permit passthrough of additional OpenAI parameters automatically."""

        extra = "allow" # Allow additional OpenAI parameters automatically (future-proofing)

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

    # -----------------------------------------------------------------------
    # Helper: turn the JSON string into valid MCP tool dicts
    # -----------------------------------------------------------------------
    @staticmethod
    def _build_mcp_tools(mcp_json: str) -> list[dict]:
        """
        Parse ``REMOTE_MCP_SERVERS_JSON`` and return a list of ready-to-use
        tool objects (``{\"type\":\"mcp\", …}``).  Silently drops invalid items.
        """
        if not mcp_json or not mcp_json.strip():
            return []
        if len(mcp_json) > 1_000_000:
            LOGGER.warning("REMOTE_MCP_SERVERS_JSON ignored: payload exceeds 1MB.")
            return []

        try:
            data = json.loads(mcp_json)
        except Exception as exc:                             # malformed JSON
            LOGGER.warning("REMOTE_MCP_SERVERS_JSON could not be parsed (%s); ignoring.", exc)
            return []

        # Accept a single object or a list
        items = data if isinstance(data, list) else [data]

        valid_tools: list[dict] = []
        for idx, obj in enumerate(items, start=1):
            if not isinstance(obj, dict):
                LOGGER.warning("REMOTE_MCP_SERVERS_JSON item %d ignored: not an object.", idx)
                continue

            # Minimum viable keys
            label = obj.get("server_label")
            url   = obj.get("server_url")
            if not (label and url):
                LOGGER.warning("REMOTE_MCP_SERVERS_JSON item %d ignored: 'server_label' and 'server_url' are required.", idx)
                continue

            parsed_url = urlparse(url)
            scheme = (parsed_url.scheme or "").lower()
            if scheme not in {"http", "https", "ws", "wss"} or not parsed_url.netloc:
                LOGGER.warning("REMOTE_MCP_SERVERS_JSON item %d ignored: unsupported server_url '%s'. Only http(s) or ws(s) URLs with a host are allowed.", idx, url)
                continue

            # Whitelist only official MCP keys so users can copy-paste API examples
            allowed = {
                "server_label",
                "server_url",
                "require_approval",
                "allowed_tools",
                "headers",
            }
            tool = {"type": "mcp"}
            tool.update({k: v for k, v in obj.items() if k in allowed})

            valid_tools.append(tool)

        return valid_tools
    
    @staticmethod
    async def transform_messages_to_input(
        messages: List[Dict[str, Any]],
        chat_id: Optional[str] = None,
        openwebui_model_id: Optional[str] = None,
        artifact_loader: Optional[
            Callable[[Optional[str], Optional[str], List[str]], Awaitable[Dict[str, Dict[str, Any]]]]
        ] = None,
        pruning_turns: int = 0,
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

        Returns
        -------
        List[dict] : The fully-formed `input` list for the OpenAI Responses API.
        """

        def _extract_plain_text(content: Any) -> str:
            """Collapse Open WebUI content blocks into a single string."""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict):
                        text_val = block.get("text") or block.get("content")
                        if isinstance(text_val, str):
                            parts.append(text_val)
                return "\n".join(parts)
            if isinstance(content, dict):
                text_val = content.get("text") or content.get("content")
                if isinstance(text_val, str):
                    return text_val
            return str(content or "")

        logger = LOGGER

        pending_instructions: list[str] = []
        openai_input: list[dict] = []

        def _message_identifier(entry: dict[str, Any]) -> Optional[str]:
            """Return the most specific identifier available on ``entry``."""
            for key in ("id", "_id", "message_id"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return None

        def _compute_turn_indices() -> tuple[list[Optional[int]], int]:
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

        def _is_old_turn(turn_index: Optional[int], *, threshold: Optional[int]) -> bool:
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
            if item.get("type") != "function_call_output":
                return False

            output_value = item.get("output")
            if output_value is None:
                return False
            if not isinstance(output_value, str):
                try:
                    output_text = json.dumps(output_value, ensure_ascii=False)
                except Exception:
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
            role = (msg.get("role") or "").lower()
            raw_content = msg.get("content", "")
            msg_id = msg.get("message_id") or _message_identifier(msg)
            msg_turn_index = turn_indices[idx]

            # -------- system / developer messages --------------------------- #
            if role in {"system", "developer"}:
                text = _extract_plain_text(raw_content)
                if text.strip():
                    pending_instructions.append(text.strip())
                continue

            # -------- user message ---------------------------------------- #
            if role == "user":
                # Convert string content to a block list (["Hello"] → [{"type": "text", "text": "Hello"}])
                content_blocks = msg.get("content") or []
                if isinstance(content_blocks, str):
                    content_blocks = [{"type": "text", "text": content_blocks}]

                instruction_prefix = "\n\n".join(pending_instructions).strip()

                # Only transform known types; leave all others unchanged
                def _to_input_image(block: dict) -> dict:
                    """Convert an Open WebUI image block into Responses format."""
                    image_payload = block.get("image_url")
                    detail: Optional[str] = None
                    if isinstance(image_payload, dict):
                        url = image_payload.get("url", "")
                        detail = image_payload.get("detail")
                    elif isinstance(image_payload, str):
                        url = image_payload
                    else:
                        url = ""
                    result: dict[str, Any] = {"type": "input_image", "image_url": url}
                    if isinstance(detail, str) and detail in {"auto", "low", "high"}:
                        result["detail"] = detail
                    elif url:
                        result["detail"] = "auto"
                    return result

                block_transform = {
                    "text":       lambda b: {"type": "input_text",  "text": b.get("text", "")},
                    "image_url":  _to_input_image,
                    "input_file": lambda b: {"type": "input_file",  "file_id": b.get("file_id")},
                }

                converted_blocks = [
                    block_transform.get(block.get("type"), lambda b: b)(block)
                    for block in content_blocks if block
                ]

                if instruction_prefix:
                    pending_instructions.clear()
                    if converted_blocks and converted_blocks[0].get("type") == "input_text":
                        converted_blocks[0]["text"] = f"{instruction_prefix}\n\n{converted_blocks[0].get('text', '')}".strip()
                    else:
                        converted_blocks.insert(0, {"type": "input_text", "text": instruction_prefix})

                openai_input.append({
                    "type": "message",
                    "role": "user",
                    "content": converted_blocks,
                })
                continue

            # -------- assistant message ----------------------------------- #
            assistant_text = raw_content if isinstance(raw_content, str) else _extract_plain_text(raw_content)
            is_old_message = _is_old_turn(msg_turn_index, threshold=prune_before_turn)

            if contains_marker(assistant_text):
                segments = split_text_by_markers(assistant_text)
                markers = [seg["marker"] for seg in segments if seg.get("type") == "marker"]

                db_artifacts: dict[str, dict] = {}
                if artifact_loader and chat_id and openwebui_model_id and markers:
                    try:
                        db_artifacts = await artifact_loader(chat_id, msg_id, markers)
                    except Exception:
                        logger.warning("Artifact loader failed for chat_id=%s message_id=%s", chat_id, msg_id, exc_info=True)
                        db_artifacts = {}

                for segment in segments:
                    if segment["type"] == "marker":
                        payload = db_artifacts.get(segment["marker"])
                        if payload is None:
                            logger.warning("Missing artifact %s for chat_id=%s message_id=%s", segment["marker"], chat_id, msg_id)
                            continue
                        item = _normalize_persisted_item(payload)
                        if item is not None:
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

        if pending_instructions:
            instruction_text = "\n\n".join(s for s in pending_instructions if s).strip()
            if instruction_text:
                openai_input.insert(
                    0,
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": instruction_text}],
                    },
                )

        return openai_input

    @classmethod
    async def from_completions(
        ResponsesBody,
        completions_body: "CompletionsBody",
        chat_id: Optional[str] = None,
        openwebui_model_id: Optional[str] = None,
        *,
        artifact_loader: Optional[
            Callable[[Optional[str], Optional[str], List[str]], Awaitable[Dict[str, Dict[str, Any]]]]
        ] = None,
        pruning_turns: int = 0,
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
            "response_format", # Replaced with 'text' in Responses API
            "suffix", # Responses API does not support suffix
            "stream_options", # Responses API does not support stream options
            "audio", # Responses API does not support audio input
            "function_call", # Deprecated in favor of 'tool_choice'.
            "functions", # Deprecated in favor of 'tools'.

            # Fields that are dropped and manually handled in step 2.
            "reasoning_effort", "max_tokens",

            # Fields that are dropped and manually handled later in the pipe()
            "tools",
            "extra_tools", # Not a real OpenAI parm. Upstream filters may use it to add tools. The are appended to body["tools"] later in the pipe()

            # Fields not documented in OpenRouter's Responses API reference
            "instructions", "store", "parallel_tool_calls", "truncation",
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

        # Transform input messages to OpenAI Responses API format
        if "messages" in completions_dict:
            sanitized_params.pop("messages", None)
            sanitized_params["input"] = await ResponsesBody.transform_messages_to_input(
                completions_dict.get("messages", []),
                chat_id=chat_id,
                openwebui_model_id=openwebui_model_id,
                artifact_loader=artifact_loader,
                pruning_turns=pruning_turns,
            )

        # Build the final ResponsesBody directly
        return ResponsesBody(
            **sanitized_params,
            **extra_params  # Extra parameters that are passed to the ResponsesBody (e.g., custom parameters configured in Open WebUI model settings)
        )

ALLOWED_OPENROUTER_FIELDS = {
    "model",
    "input",
    "stream",
    "max_output_tokens",
    "temperature",
    "top_p",
    "reasoning",
    "tools",
    "tool_choice",
    "plugins",
}

def _filter_openrouter_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Drop any keys not documented for the OpenRouter Responses API."""
    filtered: Dict[str, Any] = {}
    for key, value in payload.items():
        if key not in ALLOWED_OPENROUTER_FIELDS:
            continue
        if key == "reasoning":
            if not isinstance(value, dict):
                continue
            allowed_reasoning = {}
            if value.get("effort"):
                allowed_reasoning["effort"] = value["effort"]
            if not allowed_reasoning:
                continue
            value = allowed_reasoning
        filtered[key] = value
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Controller: Pipe
# ─────────────────────────────────────────────────────────────────────────────
# Primary interface implementing the Responses manifold
class Pipe:
    """Manifold entrypoint that adapts Open WebUI requests to OpenRouter."""
    # 4.1 Configuration Schemas
    class Valves(BaseModel):
        """Global valve configuration shared across sessions."""
        # Connection & Auth
        BASE_URL: str = Field(
            default=((os.getenv("OPENROUTER_API_BASE_URL") or "").strip() or "https://openrouter.ai/api/v1"),
            description="OpenRouter API base URL. Override this if you are using a gateway or proxy.",
        )
        API_KEY: EncryptedStr = Field(
            default=(os.getenv("OPENROUTER_API_KEY") or "").strip(),
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

        PERSIST_REASONING_TOKENS: Literal["response", "conversation", "disabled"] = Field(
            default="disabled",
            description="Control whether encrypted reasoning tokens are requested from the model (response | conversation | disabled). Tokens are requested only if the model supports reasoning.",
        )
        
        # Tool execution behavior
        PERSIST_TOOL_RESULTS: bool = Field(
             default=False,
            description="Persist tool call results across conversation turns. When disabled, tool results are not stored in the chat history.",
        )
        ARTIFACT_ENCRYPTION_KEY: EncryptedStr = Field(
            default="",
            description="Min 16 chars. Encrypt reasoning tokens (and optionally all persisted artifacts). Changing the key creates a new table; prior artifacts become inaccessible.",
        )
        ENCRYPT_ALL: bool = Field(
            default=False,
            description="Encrypt every persisted artifact when ARTIFACT_ENCRYPTION_KEY is set. When False, only reasoning tokens are encrypted.",
        )
        ENABLE_LZ4_COMPRESSION: bool = Field(
            default=True,
            description="When True (and lz4 is available), compress large encrypted artifacts to reduce database read/write overhead.",
        )
        MIN_COMPRESS_BYTES: int = Field(
            default=512,
            ge=0,
            description="Payloads at or above this size (in bytes) are candidates for LZ4 compression before encryption. Set to 0 to disable size-based gating.",
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

        # Integrations
        REMOTE_MCP_SERVERS_JSON: Optional[str] = Field(
            default=None,
            description=(
                "[EXPERIMENTAL] A JSON-encoded list (or single JSON object) defining one or more "
                "remote MCP servers to be automatically attached to each request. This can be useful "
                "for globally enabling tools across all chats.\n\n"
                "Note: The Responses API currently caches MCP server definitions at the start of each chat. "
                "This means the first message in a new thread may be slower. A more efficient implementation is planned."
                "Each item must follow the MCP tool schema supported by the OpenAI Responses API, for example:\n"
                '[{"server_label":"deepwiki","server_url":"https://mcp.deepwiki.com/mcp","require_approval":"never","allowed_tools": ["ask_question"]}]'
            ),
        )

        # Logging
        LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
            default=os.getenv("GLOBAL_LOG_LEVEL", "INFO").upper(),
            description="Select logging level.  Recommend INFO or WARNING for production use. DEBUG is useful for development and debugging.",
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


    class UserValves(BaseModel):
        """Per-user valve overrides."""
        LOG_LEVEL: Literal[
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "INHERIT"
        ] = Field(
            default="INHERIT",
            description="Select logging level. 'INHERIT' uses the pipe default.",
        )
        SHOW_FINAL_USAGE_STATUS: Optional[bool] = Field(
            default=None,
            description="Override whether the final status message includes usage stats.",
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
    _log_queue: asyncio.Queue[logging.LogRecord] | None = None  # Fix: async logging queue
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
        self._compression_min_bytes: int = max(0, int(self.valves.MIN_COMPRESS_BYTES or 0))
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
        # Fix: breaker history bounded per user
        self._breaker_records: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=5))
        self._breaker_threshold = 5
        self._breaker_window_seconds = 60
        self._tool_breakers: dict[str, dict[str, deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=5))
        )
        self._db_breakers: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=5))
        self._startup_task: asyncio.Task | None = None
        self._startup_checks_started = False
        self._startup_checks_pending = False
        self._warmup_failed = False  # Fix: track warmup failures
        self._redis_url = os.getenv("REDIS_URL")
        raw_uvicorn_workers = (os.getenv("UVICORN_WORKERS") or "1").strip()
        try:
            uvicorn_workers = int(raw_uvicorn_workers or "1")
        except ValueError:
            self.logger.warning("Invalid UVICORN_WORKERS value '%s'; defaulting to 1.", raw_uvicorn_workers)
            uvicorn_workers = 1
        multi_worker = uvicorn_workers > 1
        redis_valve_enabled = bool(self.valves.ENABLE_REDIS_CACHE)
        if multi_worker and not redis_valve_enabled:
            self.logger.warning("Multiple UVicorn workers detected but ENABLE_REDIS_CACHE is disabled; Redis cache remains off.")
        self._redis_candidate = (
            bool(self._redis_url)
            and multi_worker
            and aioredis is not None
            and redis_valve_enabled
        )
        self._redis_enabled = False
        self._redis_client: Optional[_RedisClient] = None
        self._redis_listener_task: asyncio.Task | None = None
        self._redis_flush_task: asyncio.Task | None = None
        self._redis_ready_task: asyncio.Task | None = None
        self._redis_namespace = (getattr(self, "id", None) or "openrouter").lower()
        self._redis_pending_key = f"{self._redis_namespace}:pending"
        self._redis_cache_prefix = f"{self._redis_namespace}:artifact"
        self._redis_flush_lock_key = f"{self._redis_namespace}:flush_lock"
        self._redis_ttl = max(60, int(self.valves.REDIS_CACHE_TTL_SECONDS or 300))
        self._cleanup_task = None
        self._legacy_tool_warning_emitted = False
        self._maybe_start_log_worker()
        self._maybe_start_startup_checks()

    # Core Structure -------------------------------------------------
    def _maybe_start_startup_checks(self) -> None:
        """Schedule background warmup checks once an event loop is available."""
        if self._startup_task and not self._startup_task.done():
            return
        if self._startup_checks_started and self._startup_task and self._startup_task.done():
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
        """Ensure the async logging queue + worker are started."""  # Fix: async logging
        cls = type(self)
        if cls._log_queue is None:
            cls._log_queue = asyncio.Queue(maxsize=1000)
            SessionLogger.set_log_queue(cls._log_queue)
        if cls._log_worker_lock is None:
            cls._log_worker_lock = asyncio.Lock()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
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
        api_key = EncryptedStr.decrypt(self.valves.API_KEY)
        if not api_key:
            self.logger.warning("Skipping OpenRouter warmup: API key missing.")
            return
        session: aiohttp.ClientSession | None = None
        try:
            session = self._create_http_session()
            await self._ping_openrouter(session, self.valves.BASE_URL, api_key)
            self.logger.debug("Warmed: success")
            self._warmup_failed = False
        except Exception as exc:  # pragma: no cover - depends on IO
            self.logger.warning("OpenRouter warmup failed: %s", exc)
            self._warmup_failed = True  # Fix: track failures
        finally: 
            if session:
                with contextlib.suppress(Exception):
                    await session.close()

    async def _ping_openrouter(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        api_key: str,
    ) -> None:
        """Issue a lightweight GET to prime DNS/TLS caches."""
        url = base_url.rstrip("/") + "/models?limit=1"
        headers = {"Authorization": f"Bearer {api_key}"}
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
                    timeout=aiohttp.ClientTimeout(total=10),  # Fix: explicit ping timeout
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
            await asyncio.wait_for(client.ping(), timeout=5.0)
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
        max_consecutive_failures = 5

        while self._redis_enabled:
            try:
                if not self._redis_client:
                    break

                queue_depth = await self._redis_client.llen(self._redis_pending_key)
                if queue_depth > 100:
                    self.logger.warning("⚠️ Redis pending queue backed up: %d items (threshold: 100)", queue_depth)
                elif queue_depth > 0:
                    self.logger.debug("Redis pending queue depth: %d items", queue_depth)

                await self._flush_redis_queue()
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                self.logger.error("Periodic flush failed (%d/%d consecutive failures): %s", consecutive_failures, max_consecutive_failures, exc, exc_info=self.logger.isEnabledFor(logging.DEBUG))
                if consecutive_failures >= max_consecutive_failures:
                    self.logger.critical("🚨 Disabling Redis cache after %d consecutive flush failures. Falling back to direct DB writes.", max_consecutive_failures)
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
                await self._redis_client.set(
                    self._redis_flush_lock_key,
                    lock_token,
                    nx=True,
                    ex=5,
                )
            )
            if not lock_acquired:
                self.logger.debug("Skipping Redis flush: another worker holds the lock")
                return

            rows: list[dict[str, Any]] = []
            batch_size = max(5, min(int(self.valves.DB_BATCH_SIZE or 10), 20))
            while len(rows) < batch_size:
                data = await self._redis_client.lpop(self._redis_pending_key)
                if data is None:
                    break
                try:
                    rows.append(json.loads(data))
                except json.JSONDecodeError as exc:
                    self.logger.warning("Malformed JSON in pending queue, skipping: %s", exc)
                    continue
            if not rows:
                return

            self.logger.debug("Flushing %d artifact(s) from Redis pending queue to DB (table: %s)", len(rows), self._artifact_table_name or "unknown")
            try:
                await self._db_persist_direct(rows)
                self.logger.debug("✅ Successfully flushed %d artifacts to DB", len(rows))
            except Exception as exc:
                self.logger.error("❌ DB flush failed! %d artifacts may be lost: %s", len(rows), exc, exc_info=True)
        finally:
            if lock_acquired and self._redis_client:
                release_script = (
                    "if redis.call('get', KEYS[1]) == ARGV[1] then "
                    "return redis.call('del', KEYS[1]) "
                    "else return 0 end"
                )
                try:
                    await self._redis_client.eval(
                        release_script,
                        1,
                        self._redis_flush_lock_key,
                        lock_token,
                    )
                except Exception:
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
            await pipe.execute()

            await self._redis_cache_rows(rows)
            await self._redis_client.publish(_REDIS_FLUSH_CHANNEL, "flush")

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
        await pipe.execute()

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
        values = await self._redis_client.mget(keys)
        cached: dict[str, dict[str, Any]] = {}
        for item_id, raw in zip(id_lookup, values):
            if not raw:
                continue
            try:
                row_data = json.loads(raw)
                cached[item_id] = row_data.get("payload", row_data)
            except json.JSONDecodeError:
                continue
        return cached

    async def _artifact_cleanup_worker(self) -> None:
        while True:
            try:
                await self._run_cleanup_once()
            except asyncio.CancelledError:  # pragma: no cover - shutdown
                break
            except Exception as exc:
                self.logger.warning("Artifact cleanup failed: %s", exc)
            interval_hours = max(0.5, float(self.valves.ARTIFACT_CLEANUP_INTERVAL_HOURS or 1.0))
            interval_seconds = interval_hours * 3600
            jitter = min(600.0, interval_seconds * 0.25)
            await asyncio.sleep(interval_seconds + random.uniform(0, jitter))  # Fix: configurable cadence

    async def _run_cleanup_once(self) -> None:
        if not (self._item_model and self._session_factory):
            return
        cutoff_days = max(1, int(self.valves.ARTIFACT_CLEANUP_DAYS or 30))
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

            target = max(1, valves.MAX_CONCURRENT_REQUESTS)
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

            target_tool = max(1, valves.MAX_PARALLEL_TOOLS_GLOBAL)
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
        """Drain log records asynchronously to keep handlers non-blocking."""  # Fix: async logging
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
            pass
        for task in context.workers:
            if not task.done():
                task.cancel()
        if context.workers:
            await asyncio.gather(*context.workers, return_exceptions=True)

    async def _tool_worker_loop(self, context: _ToolExecutionContext) -> None:
        """Process queued tool calls with batching/timeouts."""
        pending: list[tuple[_QueuedToolCall | None, bool]] = []
        batch_cap = 4
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
            if isinstance(result, Exception):
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
        timeout = max(1, int(context.timeout))
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
                        self._call_tool_callable(fn, item.args),
                        timeout=timeout,
                    )
                    self._reset_tool_failure_type(context.user_id, tool_type)
                    text = "" if result is None else str(result)
                    return ("completed", text)
        except Exception as exc:
            self._record_tool_failure_type(context.user_id, tool_type)
            raise exc

    async def _call_tool_callable(self, fn: Callable, args: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(fn):
            return await fn(**args)
        return await asyncio.to_thread(fn, **args)

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
        tokens: list[tuple[ContextVar, object]] = []
        tool_context: _ToolExecutionContext | None = None
        tool_token: contextvars.Token | None = None  # type: ignore[name-defined]
        try:
            async with self._acquire_semaphore(semaphore, job.request_id):
                session = self._create_http_session(job.valves)
                tokens = self._apply_logging_context(job)
                tool_queue: asyncio.Queue[_QueuedToolCall | None] = asyncio.Queue(maxsize=50)
                per_request_tool_sem = asyncio.Semaphore(
                    max(1, job.valves.MAX_PARALLEL_TOOLS_PER_REQUEST)
                )
                per_tool_timeout = max(1, int(job.valves.TOOL_TIMEOUT_SECONDS or 1))
                batch_timeout = float(
                    max(per_tool_timeout, int(job.valves.TOOL_BATCH_TIMEOUT_SECONDS or per_tool_timeout))
                )
                idle_timeout_value = job.valves.TOOL_IDLE_TIMEOUT_SECONDS
                idle_timeout = float(max(1, int(idle_timeout_value))) if idle_timeout_value else None
                self.logger.debug("Tool timeouts (request=%s): per_call=%ss batch=%ss idle=%s", job.request_id, per_tool_timeout, batch_timeout, idle_timeout if idle_timeout is not None else "disabled")
                tool_context = _ToolExecutionContext(
                    queue=tool_queue,
                    per_request_semaphore=per_request_tool_sem,
                    global_semaphore=type(self)._tool_global_semaphore,
                    timeout=float(per_tool_timeout),
                    batch_timeout=batch_timeout,
                    idle_timeout=idle_timeout,
                    user_id=job.user_id,
                    event_emitter=job.event_emitter,
                )
                worker_count = max(1, job.valves.MAX_PARALLEL_TOOLS_PER_REQUEST)
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
                    job.event_emitter,
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
            if not job.future.done():
                job.future.set_exception(exc)
        finally:
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

    def _create_http_session(self, valves: "Pipe.Valves" | None = None) -> aiohttp.ClientSession:
        """Return a fresh ClientSession with sane defaults for per-request use."""
        valves = valves or self.valves
        connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=10,
            keepalive_timeout=75,
            ttl_dns_cache=300,
        )
        connect_timeout = max(1, int(getattr(valves, "HTTP_CONNECT_TIMEOUT_SECONDS", 10) or 10))
        total_timeout_value = getattr(valves, "HTTP_TOTAL_TIMEOUT_SECONDS", None)
        total_timeout = None
        if isinstance(total_timeout_value, (int, float)):
            if total_timeout_value > 0:
                total_timeout = float(total_timeout_value)
        sock_read = None
        sock_read_value = getattr(valves, "HTTP_SOCK_READ_SECONDS", None)
        if total_timeout is None and sock_read_value:
            sock_read = float(max(1, int(sock_read_value)))
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

    def _apply_logging_context(self, job: _PipeJob) -> list[tuple[ContextVar, object]]:
        """Set SessionLogger contextvars based on the incoming request."""
        session_id = job.session_id or None
        user_id = job.user_id or None
        log_level = getattr(logging, str(job.valves.LOG_LEVEL).upper(), logging.INFO)
        tokens: list[tuple[ContextVar, object]] = []
        tokens.append((SessionLogger.session_id, SessionLogger.session_id.set(session_id)))
        tokens.append((SessionLogger.user_id, SessionLogger.user_id.set(user_id)))
        tokens.append((SessionLogger.log_level, SessionLogger.log_level.set(log_level)))
        return tokens

    def _breaker_allows(self, user_id: str) -> bool:
        """Simple per-user breaker: 5 failures per minute."""
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
        self._compression_min_bytes = max(0, int(valves.MIN_COMPRESS_BYTES or 0))

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

    def _resolve_pipe_identifier(
        self,
        openwebui_model_id: Optional[str] = None,
        *,
        fallback_model_id: Optional[str] = None,
    ) -> str:
        """Derive the pipe identifier prefix used for storage namespaces."""
        def _extract_prefix(value: Optional[str]) -> Optional[str]:
            if not isinstance(value, str):
                return None
            candidate = value.split(".", 1)[0].strip()
            return candidate or None

        for source in (openwebui_model_id, fallback_model_id):
            candidate = _extract_prefix(source)
            if candidate:
                return candidate
        explicit_id = getattr(self, "id", None)
        if isinstance(explicit_id, str) and explicit_id.strip():
            return explicit_id.strip()
        fallback_identifier = "openrouter"
        self.logger.warning("Pipe identifier missing from metadata; defaulting to '%s'.", fallback_identifier)
        return fallback_identifier

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
        except Exception:  # pragma: no cover - optional dependency
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
        for lowered_name, original_name in names_to_drop.items():
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
                # Backward compatibility: old ciphertexts lack the header and are just JSON.
                body = payload_bytes
        try:
            return json.loads(body.decode("utf-8"))
        except Exception as exc:
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
        return {"ciphertext": encrypted}, True

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
            ulids: list[str] = [
                row.get("id")
                for row in rows
                if row.get("_persisted") and isinstance(row.get("id"), str)
            ]
            ulids = [ulid for ulid in ulids if ulid]
            batch_size = max(5, min(int(self.valves.DB_BATCH_SIZE or 10), 20))  # Fix: configurable batching
            pending_rows = [row for row in rows if not row.get("_persisted")]
            if not pending_rows:
                if ulids:
                    self.logger.debug("Persisted %d response artifact(s) to %s.", len(ulids), self._artifact_table_name)
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
                    if not isinstance(payload, dict):
                        self.logger.warning("Skipping artifact persist for chat_id=%s message_id=%s: payload is not a dict.", row.get("chat_id"), row.get("message_id"))
                        continue
                    ulid = row.get("id") or generate_item_id()
                    stored_payload, is_encrypted = self._encrypt_if_needed(row.get("item_type", ""), payload)
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
                    self.logger.error("Failed to persist response artifacts: %s", exc, exc_info=self.logger.isEnabledFor(logging.DEBUG))
                    raise
                finally:
                    session.close()

                for row in persisted_rows:
                    row["_persisted"] = True
                ulids.extend(chunk_ulids)

            if ulids:
                self.logger.debug("Persisted %d response artifact(s) to %s.", len(ulids), self._artifact_table_name)
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
            )  # Fix: notify on DB breaker
            self._record_failure(user_id)
            return []

        for row in rows:
            row.setdefault("id", generate_item_id())

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
                        return [row.get("id") for row in rows if row.get("id")]
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
            )  # Fix: notify on DB breaker
            self._record_failure(user_id)
            return {}

        try:
            fetched = await self._db_fetch_direct(chat_id, message_id, missing_ids)
            if fetched and self._redis_enabled:
                await self._redis_cache_rows(
                    [
                        {
                            "id": item_id,
                            "chat_id": chat_id,
                            "message_id": message_id,
                            "payload": payload,
                        }
                        for item_id, payload in fetched.items()
                    ],
                    chat_id=chat_id,
                )
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
        return [{"id": model["id"], "name": model["name"]} for model in selected_models]

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __request__: Request,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]],
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        __metadata__: dict[str, Any],
        __tools__: list[dict[str, Any]] | dict[str, Any] | None,
        __task__: Optional[dict[str, Any]] = None,
        __task_body__: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None] | str | None | JSONResponse:
        """Entry point that enqueues work and awaits the isolated job result."""

        self._maybe_start_log_worker()
        self._maybe_start_startup_checks()
        self._maybe_start_redis()
        self._maybe_start_cleanup()
        user_valves = self.UserValves.model_validate(__user__.get("valves", {}))
        valves = self._merge_valves(self.valves, user_valves)
        session_id = str(__metadata__.get("session_id") or "")
        user_id = str(__user__.get("id") or __metadata__.get("user_id") or "")
        safe_event_emitter = self._wrap_safe_event_emitter(__event_emitter__)

        if not self._breaker_allows(user_id):  # Fix: user-scoped breaker
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
        future = loop.create_future()
        job = _PipeJob(
            pipe=self,
            body=body,
            user=__user__,
            request=__request__,
            event_emitter=safe_event_emitter,
            event_call=__event_call__,
            metadata=__metadata__,
            tools=__tools__,
            task=__task__,
            task_body=__task_body__,
            valves=valves,
            future=future,
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
        __request__: Request,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]],
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        __metadata__: dict[str, Any],
        __tools__: list[dict[str, Any]] | dict[str, Any] | None,
        __task__: Optional[dict[str, Any]] = None,
        __task_body__: Optional[dict[str, Any]] = None,
        *,
        valves: "Pipe.Valves" | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> AsyncGenerator[str, None] | str | None:
        """Process a user request and return either a stream or final text.

        When ``body['stream']`` is ``True`` the method yields deltas from
        ``_run_streaming_loop``.  Otherwise it falls back to
        ``_run_nonstreaming_loop`` and returns the aggregated response.
        """
        valves = valves or self._merge_valves(self.valves, self.UserValves.model_validate(__user__.get("valves", {})))
        if session is None:
            raise RuntimeError("HTTP session is required for _handle_pipe_call")

        openwebui_model_id = __metadata__.get("model", {}).get("id", "")
        request_model_id = body.get("model") if isinstance(body.get("model"), str) else None
        pipe_identifier_for_artifacts = self._resolve_pipe_identifier(
            openwebui_model_id,
            fallback_model_id=request_model_id,
        )
        self._ensure_artifact_store(valves, pipe_identifier=pipe_identifier_for_artifacts)

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
        pipe_identifier = pipe_identifier_for_artifacts
        pipe_token = ModelFamily._PIPE_ID.set(pipe_identifier)
        # Features are nested under the pipe id key – read them dynamically.
        _features_all = __metadata__.get("features", {}) or {}
        features = _features_all.get(pipe_identifier, {})
        # Custom location that this manifold uses to store feature flags
        user_id = str(__user__.get("id") or __metadata__.get("user_id") or "")

        try:
            result = await self._process_transformed_request(
                body,
                __user__,
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
                pipe_identifier_for_artifacts,
                allowed_norm_ids,
                features,
                user_id=user_id,
            )
        finally:
            ModelFamily._PIPE_ID.reset(pipe_token)
        return result


    async def _process_transformed_request(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]] | None,
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        __metadata__: dict[str, Any],
        __tools__: list[dict[str, Any]] | dict[str, Any] | None,
        __task__: Optional[dict[str, Any]],
        __task_body__: Optional[dict[str, Any]],
        valves: "Pipe.Valves",
        session: aiohttp.ClientSession,
        openwebui_model_id: str,
        pipe_identifier: str,
        pipe_identifier_for_artifacts: str,
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
        responses_body = await ResponsesBody.from_completions(
            completions_body=completions_body,

            # If chat_id and openwebui_model_id are provided, from_completions() uses them to fetch previously persisted items (function_calls, reasoning, etc.) from DB and reconstruct the input array in the correct order.
            **({"chat_id": __metadata__["chat_id"]} if __metadata__.get("chat_id") else {}),
            **({"openwebui_model_id": openwebui_model_id} if openwebui_model_id else {}),
            artifact_loader=self._db_fetch,
            pruning_turns=valves.TOOL_OUTPUT_RETENTION_TURNS,

        )
        if valves.USE_MODEL_MAX_OUTPUT_TOKENS:
            if responses_body.max_output_tokens is None:
                default_max = ModelFamily.max_completion_tokens(responses_body.model)
                if default_max:
                    responses_body.max_output_tokens = default_max
        else:
            responses_body.max_output_tokens = None

        normalized_model_id = ModelFamily.base_model(responses_body.model)
        if allowed_norm_ids and normalized_model_id not in allowed_norm_ids:
            await self._emit_error(
                __event_emitter__,
                f"Model '{responses_body.model}' is not enabled for this pipe. Please choose one of the allowed models.",
                show_error_message=True,
                done=True,
            )
            return ""

        # STEP 2: Detect if task model (generate title, generate tags, etc.), handle it separately
        if __task__:
            self.logger.debug("Detected task model: %s", __task__)
            return await self._run_task_model_request(
                responses_body.model_dump(),
                valves,
                session=session,
                task_context=__task__,
            )  # Placeholder for task handling logic

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
        tools = build_tools(
            responses_body,
            valves,
            __tools__=__tools__,
            features=features,
            extra_tools=getattr(completions_body, "extra_tools", None),
        )

        # STEP 4: Auto-enable native function calling if tools are used but `native` function calling is not enabled in Open WebUI model settings.
        if tools and ModelFamily.supports("function_calling", openwebui_model_id):
            try:
                model = Models.get_model_by_id(openwebui_model_id)
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
                        Models.update_model_by_id(openwebui_model_id, ModelForm(**form_data))
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
            plugin_payload = {"id": "web"}
            if valves.WEB_SEARCH_MAX_RESULTS is not None:
                plugin_payload["max_results"] = valves.WEB_SEARCH_MAX_RESULTS
            plugins = list(responses_body.plugins or [])
            plugins.append(plugin_payload)
            responses_body.plugins = plugins

        # STEP 7: Log the transformed request body
        self.logger.debug("Transformed ResponsesBody: %s", json.dumps(responses_body.model_dump(exclude_none=True), indent=2, ensure_ascii=False))

        # Convert the normalized model id back to the original OpenRouter id for the API request.
        setattr(responses_body, "api_model", OpenRouterModelRegistry.api_model_id(normalized_model_id) or normalized_model_id)

        # STEP 8: Send to OpenAI Responses API
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
            )
        else:
            # Return final text (non-streaming)
            return await self._run_nonstreaming_loop(
                responses_body,
                valves,
                __event_emitter__,
                __metadata__,
                __tools__,
                session=session,
                user_id=user_id,
            )


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
    async def _run_streaming_loop(
        self,
        body: ResponsesBody,
        valves: Pipe.Valves,
        event_emitter: Callable[[Dict[str, Any]], Awaitable[None]],
        metadata: dict[str, Any] = {},
        tools: Optional[Dict[str, Dict[str, Any]]] = None,
        session: aiohttp.ClientSession | None = None,
        user_id: str = "",
    ):
        """
        Stream assistant responses incrementally, handling function calls, status updates, and tool usage.
        """
        if session is None:
            raise RuntimeError("HTTP session is required for streaming")

        self.logger.debug("🔧 PERSIST_TOOL_RESULTS=%s", valves.PERSIST_TOOL_RESULTS)

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
        openwebui_model = metadata.get("model", {}).get("id", "")
        assistant_message = ""
        pending_ulids: list[str] = []
        pending_items: list[dict[str, Any]] = []
        total_usage: dict[str, Any] = {}
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
        if ModelFamily.supports("reasoning", body.model) and event_emitter:
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
                thinking_tasks.append(
                    asyncio.create_task(_later(delay + random.uniform(0, 0.5), msg))
                )

        def cancel_thinking() -> None:
            """Cancel any scheduled reasoning status updates once the loop completes."""
            if thinking_tasks:
                for t in thinking_tasks:
                    t.cancel()
                thinking_tasks.clear()

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
                request_payload = body.model_dump(exclude_none=True)
                api_model_override = getattr(body, "api_model", None)
                if api_model_override:
                    request_payload["model"] = api_model_override
                    request_payload.pop("api_model", None)
                request_payload = _filter_openrouter_request(request_payload)

                api_key_value = EncryptedStr.decrypt(valves.API_KEY)
                async for event in self.send_openai_responses_streaming_request(
                    session,
                    request_payload,
                    api_key=api_key_value,
                    base_url=valves.BASE_URL,
                    workers=valves.SSE_WORKERS_PER_REQUEST,
                    breaker_key=user_id or None,
                ):
                    if stream_started_at is None:
                        stream_started_at = perf_counter()
                    etype = event.get("type")
                    if etype:
                        note_model_activity()

                    # Emit OpenRouter SSE frames at DEBUG (non-delta) only; skip delta spam entirely.
                    is_delta_event = bool(etype and etype.endswith(".delta"))
                    if not is_delta_event and self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug("OpenRouter payload: %s",json.dumps(event, indent=2, ensure_ascii=False))

                    # ─── Emit partial delta assistant message
                    if etype == "response.output_text.delta":
                        delta = event.get("delta", "")
                        if delta:
                            note_generation_activity()
                            if not provider_status_seen and not responding_status_sent and event_emitter:
                                provider_status_seen = True
                                responding_status_sent = True
                                await event_emitter(
                                    {
                                        "type": "status",
                                        "data": {"description": "Responding to the user…"},
                                    }
                                )
                            assistant_message += delta
                            await event_emitter({"type": "chat:message", "data": {"content": assistant_message}})
                        continue

                    # ─── Emit reasoning summary once done ───────────────────────
                    if etype == "response.reasoning_summary_text.done":
                        text = (event.get("text") or "").strip()
                        if text:
                            title_match = re.findall(r"\*\*(.+?)\*\*", text)
                            title = title_match[-1].strip() if title_match else "Thinking…"
                            content = re.sub(r"\*\*(.+?)\*\*", "", text).strip()
                            if event_emitter:
                                cancel_thinking()
                                await event_emitter(
                                    {
                                        "type": "status",
                                        "data": {"description": f"{title}\n{content}"},
                                    }
                                )
                        continue

                    # ─── Citations from inline annotations (simple, no helpers) ───────────────
                    if etype == "response.output_text.annotation.added":
                        ann = event.get("annotation") or {}
                        if ann.get("type") == "url_citation":
                            # Basic fields
                            url = (ann.get("url") or "").strip()
                            if url.endswith("?utm_source=openai"):
                                url = url[: -len("?utm_source=openai")]
                            title = (ann.get("title") or url).strip()

                            # Stable [n] per unique URL
                            if url in ordinal_by_url:
                                n = ordinal_by_url[url]
                            else:
                                n = len(ordinal_by_url) + 1
                                ordinal_by_url[url] = n

                                # First time seeing this URL → emit a 'source' event
                                # Minimal domain extraction (no urlparse)
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

                            # TODO: Add support for insert citation markers.
                            marker = f" [{n}]"
                            end_idx = ann.get("end_index")
                            if isinstance(end_idx, int) and 0 <= end_idx <= len(assistant_message):
                                assistant_message = (
                                    assistant_message[:end_idx]
                                    + marker
                                    + assistant_message[end_idx:]
                                )
                            else:
                                assistant_message += marker
                            if event_emitter:
                                await event_emitter(
                                    {
                                        "type": "chat:message",
                                        "data": {"content": assistant_message},
                                    }
                                )

                        continue


                    # ─── Emit status updates for in-progress items ──────────────────────
                    if etype == "response.output_item.added":
                        item = event.get("item", {})
                        item_type = item.get("type", "")
                        item_status = item.get("status", "")

                        if item_type == "message" and item_status == "in_progress":
                            provider_status_seen = True
                            if (not responding_status_sent) and event_emitter:
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
                        item = event.get("item", {})
                        item_type = item.get("type", "")
                        item_name = item.get("name", "unnamed_tool")

                        # Skip irrelevant item types
                        if item_type in ("message"):
                            continue

                        # Decide persistence policy
                        should_persist = False
                        if item_type == "reasoning":
                            # Persist reasoning only when explicitly allowed
                            should_persist = valves.PERSIST_REASONING_TOKENS == "conversation"

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
                                #TODO: emit status for open_page.  Only emitted by Deep Research models
                                continue
                            elif action.get("type") == "find_in_page":
                                #TODO: emit status for find_in_page.  Only emitted by Deep Research models
                                continue
                                    
                            continue

                        elif item_type == "file_search_call":
                            title = "Let me skim those files…"
                        elif item_type == "image_generation_call":
                            title = "Let me create that image…"
                        elif item_type == "local_shell_call":
                            title = "Let me run that command…"
                        elif item_type == "mcp_call":
                            title = "Let me query the MCP server…"
                        elif item_type == "reasoning":
                            title = None # Don't emit a title for reasoning items

                        # Log the status with prepared title and detailed content instead of emitting it
                        if title:
                            desc = title if not content else f"{title}\n{content}"
                            if thinking_tasks:
                                cancel_thinking()
                            self.logger.debug("Tool status update: %s", desc)

                        continue

                    # ─── Capture final response payload for this loop
                    if etype == "response.completed":
                        final_response = event.get("response", {})
                        response_completed_at = perf_counter()
                        if generation_started_at is not None:
                            generation_last_event_at = response_completed_at
                        break

                if final_response is None:
                    raise ValueError("No final response received from OpenAI Responses API.")

                # Extract usage information from OpenAI response and pass-through to Open WebUI
                usage = final_response.get("usage", {})
                if usage:
                    usage["turn_count"] = 1
                    usage["function_call_count"] = sum(
                        1 for i in final_response["output"] if i["type"] == "function_call"
                    )
                    total_usage = merge_usage_stats(total_usage, usage)
                    await self._emit_completion(event_emitter, content="", usage=total_usage, done=False)

                # Execute tool calls (if any), persist results (if valve enabled), and append to body.input.
                call_items = []
                for item in final_response.get("output", []):
                    if item.get("type") == "function_call":
                        normalized_call = _normalize_persisted_item(item)
                        if normalized_call:
                            call_items.append(normalized_call)
                if call_items:
                    body.input.extend(call_items)

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
                else:
                    break

        # Catch any exceptions during the streaming loop and emit an error
        except asyncio.CancelledError:
            was_cancelled = True
            raise
        except Exception as e:  # pragma: no cover - network errors
            error_occurred = True
            await self._emit_error(event_emitter, f"Error: {str(e)}", show_error_message=True, show_error_log_citation=True, done=True)

        finally:
            cancel_thinking()
            for t in thinking_tasks:
                with contextlib.suppress(Exception):
                    await t
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

            if valves.LOG_LEVEL != "INHERIT":
                session_id = SessionLogger.session_id.get()
                if session_id:
                    logs = SessionLogger.logs.get(session_id, [])
                    if logs and self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug("Collected %d session log entries for session %s.", len(logs), session_id)

            if not was_cancelled:
                # Emit completion (middleware.py also does this so this just covers if there is a downstream error)
                await self._emit_completion(event_emitter, content="", usage=total_usage, done=True)  # There must be an empty content to avoid breaking the UI

            # Clear logs
            SessionLogger.logs.pop(SessionLogger.session_id.get(), None)
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

    async def _run_nonstreaming_loop(
        self,
        body: ResponsesBody,
        valves: Pipe.Valves,
        event_emitter: Callable[[Dict[str, Any]], Awaitable[None]],
        metadata: Dict[str, Any] = {},
        tools: Optional[Dict[str, Dict[str, Any]]] = None,
        session: aiohttp.ClientSession | None = None,
        user_id: str = "",
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
        )

    # 4.4 Task Model Handling
    async def _run_task_model_request(
        self,
        body: Dict[str, Any],
        valves: Pipe.Valves,
        *,
        session: aiohttp.ClientSession | None = None,
        task_context: Optional[Dict[str, Any]] = None,
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
      
    # 4.5 LLM HTTP Request Helpers
    async def send_openai_responses_streaming_request(
        self,
        session: aiohttp.ClientSession,
        request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        workers: int = 4,
        breaker_key: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Producer/worker SSE pipeline with delta batching."""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        _debug_print_request(headers, request_body)
        url = base_url.rstrip("/") + "/responses"

        workers = max(1, min(int(workers or 1), 8))
        chunk_queue: asyncio.Queue[tuple[Optional[int], bytes]] = asyncio.Queue(maxsize=100)
        event_queue: asyncio.Queue[tuple[Optional[int], Optional[dict[str, Any]]]] = asyncio.Queue(maxsize=100)
        chunk_sentinel = (None, b"")
        DELTA_BATCH_THRESHOLD = 50

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
                                    await _debug_print_error_response(resp)
                                    if breaker_key:
                                        self._record_failure(breaker_key)
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
                                            event_data_parts.append(stripped[5:].lstrip())
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
                        except Exception as exc:
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

        def flush_delta(force: bool = False) -> Optional[dict[str, Any]]:
            nonlocal delta_buffer, delta_template, delta_length
            if delta_buffer and (force or delta_length >= DELTA_BATCH_THRESHOLD):
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
                seq, event = await event_queue.get()
                event_queue.task_done()
                if seq is None:
                    done_workers += 1
                    if done_workers >= workers and not pending_events:
                        break
                    continue
                pending_events[seq] = event
                while next_seq in pending_events:
                    current = pending_events.pop(next_seq)
                    next_seq += 1
                    etype = current.get("type")
                    if etype == "response.output_text.delta":
                        delta_chunk = current.get("delta", "")
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
            await asyncio.gather(producer_task, *worker_tasks, return_exceptions=True)

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
                        await _debug_print_error_response(resp)
                        if resp.status < 500:
                            raise RuntimeError(
                                f"OpenRouter request failed ({resp.status}): {resp.reason}"
                            )
                    resp.raise_for_status()
                    return await resp.json()

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
                await self._redis_client.close()
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
            tool_cfg = tools.get(call.get("name"))
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
            if not self._tool_type_allows(context.user_id, tool_type):  # Fix: breaker before enqueue
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
            tool_cfg = tools.get(call.get("name"))
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
        emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> Callable[[dict[str, Any]], Awaitable[None]] | None:
        """Return an emitter wrapper that swallows downstream transport errors."""

        if emitter is None:
            return None

        async def _guarded(event: dict[str, Any]) -> None:
            try:
                await emitter(event)
            except Exception as exc:  # pragma: no cover - emitter transport errors
                evt_type = event.get("type") if isinstance(event, dict) else None
                suffix = f" ({evt_type})" if evt_type else ""
                self.logger.debug("Event emitter failure%s: %s", suffix, exc)

        return _guarded
    async def _emit_error(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]],
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
            session_id = SessionLogger.session_id.get()
            logs = SessionLogger.logs.get(session_id, [])
            if logs:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug("Error logs for session %s:\n%s", session_id, "\n".join(logs))
            else:
                self.logger.warning("No debug logs found for session_id %s", session_id)

    async def _emit_citation(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
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
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        *,
        content: str | None = "",                       # always included (may be "").  UI will stall if you leave it out.
        title:   str | None = None,                     # optional title.
        usage:   dict[str, Any] | None = None,          # optional usage block
        done:    bool = True,                           # True → final frame
    ) -> None:
        """Emit a ``chat:completion`` event if an emitter is present.

        The ``done`` flag indicates whether this is the final frame for the
        request.  When ``usage`` information is provided it is forwarded as part
        of the event data.
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
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
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

        # Merge: update only fields not set to "INHERIT"
        update = {
            k: v
            for k, v in user_valves.model_dump().items()
            if v is not None and str(v).lower() != "inherit"
        }
        return global_valves.model_copy(update=update)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Utility & Helper Layer (organized, consistent docstrings)
#    NOTE: Logic is unchanged. Only docstrings/comments/sectioning were improved.
# ─────────────────────────────────────────────────────────────────────────────

# 5.1 Logging & Diagnostics
# -------------------------

class SessionLogger:
    """Per-request logger that captures console output and an in-memory log buffer.

    The logger is bound to a logical *session* via contextvars so that log lines
    can be collected and emitted (e.g., as citations) for the current request.
    Cleanup is intentional and explicit: request handlers call ``cleanup`` once
    they finish streaming so there is no background task silently pruning logs.

    Attributes:
        session_id: ContextVar storing the current logical session ID.
        log_level:  ContextVar storing the minimum level to emit for this session.
        logs:       Map of session_id -> fixed-size deque of formatted log strings.
    """

    session_id = ContextVar("session_id", default=None)
    user_id = ContextVar("user_id", default=None)
    log_level = ContextVar("log_level", default=logging.INFO)
    logs = defaultdict(lambda: deque(maxlen=2000))
    _session_last_seen: Dict[str, float] = {}
    log_queue: asyncio.Queue[logging.LogRecord] | None = None
    _main_loop: asyncio.AbstractEventLoop | None = None
    _console_formatter = logging.Formatter("[%(levelname)s] [session=%(session_label)s] [user=%(user_id)s] %(message)s")
    _memory_formatter = logging.Formatter("%(asctime)s [%(levelname)s] [user=%(user_id)s] %(message)s")

    @classmethod
    def get_logger(cls, name=__name__):
        """Create a logger wired to the current SessionLogger context.

        Args:
            name: Logger name; defaults to the current module name.

        Returns:
            logging.Logger: A configured logger that writes both to stdout and
            the in-memory `SessionLogger.logs` buffer. The buffer is keyed by
            the current `SessionLogger.session_id`.
        """
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.filters.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # Single combined filter: attach session_id and respect per-session level.
        def filter(record):
            """Attach session metadata and enforce per-session log levels."""
            sid = cls.session_id.get()
            uid = cls.user_id.get()
            record.session_id = sid
            record.session_label = sid or "-"
            record.user_id = uid or "-"
            if sid:
                cls._session_last_seen[sid] = time.time()
            return record.levelno >= cls.log_level.get()

        logger.addFilter(filter)

        async_handler = logging.Handler()

        def _emit(record: logging.LogRecord) -> None:  # Fix: enqueue log record
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
        console_line = cls._console_formatter.format(record)
        sys.stdout.write(console_line + "\n")
        sys.stdout.flush()
        session_id = getattr(record, "session_id", None)
        if session_id:
            cls.logs[session_id].append(cls._memory_formatter.format(record))
            cls._session_last_seen[session_id] = time.time()

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
    emitter: Callable[[Dict[str, Any]], Awaitable[None]] | None,
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
            except Exception:
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
        "mcp_call",
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


# ─────────────────────────────────────────────────────────────────────────────
# 9. Tool & Schema Utilities (internal)
# ─────────────────────────────────────────────────────────────────────────────
def build_tools(
    responses_body: "ResponsesBody",
    valves: "Pipe.Valves",
    __tools__: Optional[Dict[str, Any]] = None,
    *,
    features: Optional[Dict[str, Any]] = None,
    extra_tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build the OpenAI Responses-API tool spec list for this request.

    - Returns [] if the target model doesn't support function calling.
    - Includes Open WebUI registry tools (strictified if enabled).
    - Adds OpenAI web_search (if allowed + supported + not minimal effort).
    - Adds MCP tools from REMOTE_MCP_SERVERS_JSON.
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

    # 3) Optional MCP servers
    if valves.REMOTE_MCP_SERVERS_JSON:
        tools.extend(ResponsesBody._build_mcp_tools(valves.REMOTE_MCP_SERVERS_JSON))

    # 4) Optional extra tools (already OpenAI-format)
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

            original_required = set(node.get("required") or [])

            node["additionalProperties"] = False
            node["required"] = list(props.keys())

            for name, p in props.items():
                if not isinstance(p, dict):
                    continue
                if name not in original_required:
                    ptype = p.get("type")
                    if isinstance(ptype, str) and ptype != "null":
                        p["type"] = [ptype, "null"]
                    elif isinstance(ptype, list) and "null" not in ptype:
                        p["type"] = ptype + ["null"]
                stack.append(p)

        items = node.get("items")
        if isinstance(items, dict):
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
