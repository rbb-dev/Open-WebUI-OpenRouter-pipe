"""Main Pipe orchestrator for OpenRouter integration.

This module defines the Pipe class and coordinates:
- persistence, multimodal, streaming, and event subsystems
- request processing, tool execution, and model management
- lifecycle, background workers, and concurrency controls

Architecture:
- Pipe class manages initialization, lifecycle, and high-level orchestration
- Subsystems handle specific concerns (persistence, streaming, files, events)
- Delegating methods forward calls to appropriate subsystems
- Orchestration methods remain in Pipe (workers, queues, HTTP, tools)
"""

from __future__ import annotations

import ast
import asyncio
import base64
import binascii
import contextlib
import contextvars
import datetime
import fnmatch
import hashlib
import inspect
import itertools
import json
import logging
import os
import queue
import random
import re
import secrets
import threading
import time
import traceback
import uuid
from decimal import Decimal, InvalidOperation
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Type, TypeVar, Union, TYPE_CHECKING, cast, no_type_check
from urllib.parse import quote, urlparse

# Third-party imports
import aiohttp
import httpx
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from .core.timing_logger import timed

# Open WebUI internals (available when running as a pipe)
try:
    from open_webui.models.chats import Chats
except ImportError:
    Chats = None  # type: ignore
try:
    from open_webui.models.models import ModelForm, Models
except ImportError:
    ModelForm = None  # type: ignore
    Models = None  # type: ignore
try:
    from open_webui.models.files import Files
except ImportError:
    Files = None  # type: ignore
try:
    from open_webui.models.users import Users
except ImportError:
    Users = None  # type: ignore

# Optional Redis support
try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore

# Timing instrumentation
from .core.timing_logger import timed, timing_mark, timing_scope, configure_timing_file, ensure_timing_file_configured

if TYPE_CHECKING:
    from redis.asyncio import Redis as _RedisClient
else:
    _RedisClient = Any

# Optional pyzipper support for session log encryption
try:
    import pyzipper  # pyright: ignore[reportMissingImports]
except ImportError:
    pyzipper = None  # type: ignore

# Import subsystems
from .storage.persistence import ArtifactStore, generate_item_id, _sanitize_table_fragment
from .storage.multimodal import (
    MultimodalHandler,
    _guess_image_mime_type,
    _extract_openrouter_og_image,
    _classify_retryable_http_error,
)
from .streaming.streaming_core import StreamingHandler
from .streaming.event_emitter import EventEmitterHandler

# Import vendor integrations
from .integrations.anthropic import _maybe_apply_anthropic_prompt_caching

# Import request processing
from .requests.sanitizer import _sanitize_request_input
from .requests.transformer import transform_messages_to_input

# Import model management
from .models.catalog_manager import ModelCatalogManager
from .models.reasoning_config import ReasoningConfigManager

# Import error handling
from .core.error_formatter import ErrorFormatter
from .core.circuit_breaker import CircuitBreaker

# Import request handling
from .requests import NonStreamingAdapter, TaskModelAdapter

# Import configuration and core modules
from .core.config import (
    Valves,
    UserValves,
    EncryptedStr,
    _PIPE_RUNTIME_ID,
    _DEFAULT_PIPE_ID,
    _OPENROUTER_TITLE,
    _OPENROUTER_REFERER,
    _OPENROUTER_SITE_URL,
    _OPENROUTER_FRONTEND_MODELS_URL,
    _NON_REPLAYABLE_TOOL_ARTIFACTS,
    _MAX_OPENROUTER_ID_CHARS,
    _MAX_OPENROUTER_METADATA_VALUE_CHARS,
    _detect_runtime_pipe_id,
    DEFAULT_OPENROUTER_ERROR_TEMPLATE,
    _ORS_FILTER_FEATURE_FLAG,
    _ORS_FILTER_MARKER,
    _ORS_FILTER_PREFERRED_FUNCTION_ID,
    _DIRECT_UPLOADS_FILTER_MARKER,
    _DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID,
    _PROVIDER_ROUTING_FILTER_MARKER_PREFIX,
    _PROVIDER_ROUTING_FILTER_MARKER_VERSION,
    _PROVIDER_ROUTING_FILTER_ID_PREFIX,
    _select_openrouter_http_referer,
)
from .requests.debug import (
    _debug_print_request,
    _debug_print_error_response,
)
# Imports from storage.persistence (additional)
from .storage.persistence import (
    normalize_persisted_item as _normalize_persisted_item,
)
# Imports from tools.tool_schema
from .tools.tool_schema import (
    _classify_function_call_artifacts,
)
# Imports from api.transforms
from .api.transforms import (
    _get_disable_param,
    _responses_tools_to_chat_tools,
    _responses_tool_choice_to_chat_tool_choice,
    _responses_input_to_chat_messages,
    _chat_tools_to_responses_tools,
    _responses_payload_to_chat_completions_payload,
    _apply_identifier_valves_to_payload,
    _filter_openrouter_request,
    _filter_replayable_input_items,
    _filter_openrouter_chat_request,
)
# Imports from core.utils
from .core.utils import (
    _redact_payload_blobs,
    _extract_plain_text_content,
    _extract_feature_flags,
    _retry_after_seconds,
    _select_best_effort_fallback,
)
# Imports from core.errors
from .core.errors import (
    _resolve_error_model_context,
    _build_openrouter_api_error,
    _is_reasoning_effort_error,
    _parse_supported_effort_values,
    _format_openrouter_error_markdown,
    _wait_for,
)
# Imports from storage.multimodal
from .storage.multimodal import (
    _extract_internal_file_id,
    _is_internal_file_url,
)
# Imports from models.registry
from .models.registry import (
    OpenRouterModelRegistry,
    ModelFamily,
    sanitize_model_id,
)
from .tools.tool_registry import _build_collision_safe_tool_specs_and_registry
from .tools.tool_executor import _QueuedToolCall, _ToolExecutionContext
from .api.transforms import (
    ResponsesBody,
    CompletionsBody,
    ALLOWED_OPENROUTER_FIELDS,
    ALLOWED_OPENROUTER_CHAT_FIELDS,
    _normalise_openrouter_responses_text_format,
    _sanitize_openrouter_metadata,
)
from .core.errors import (
    OpenRouterAPIError,
    StatusMessages,
    _RetryWait,
    _RetryableHTTPStatusError,
)
from .core.logging_system import SessionLogger, _SessionLogArchiveJob

if TYPE_CHECKING:
    from .tools.tool_executor import ToolExecutor
    from .api.gateway.responses_adapter import ResponsesAdapter
    from .api.gateway.chat_completions_adapter import ChatCompletionsAdapter
    from .requests.orchestrator import RequestOrchestrator
from .api.filters import Filter as OpenRouterSearchFilter
from .core.utils import (
    _coerce_bool,
    _coerce_positive_int,
    _normalize_optional_str,
    _normalize_string_list,
    _pretty_json,
    _safe_json_loads,
    _render_error_template,
    _sanitize_path_component,
    contains_marker,
    split_text_by_markers,
    _serialize_marker,
)
from .tools.tool_registry import _dedupe_tools, _normalize_responses_function_tool_spec, _responses_spec_from_owui_tool_cfg
from .tools.tool_schema import _strictify_schema
from .tools.tool_worker import (
    _tool_worker_loop,
    _can_batch_tool_calls,
    _args_reference_call,
)

# Type hints
EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]
ToolCallable = Callable[..., Awaitable[Any]] | Callable[..., Any]

LOGGER = logging.getLogger(__name__)

# Regex patterns
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?P<url>[^)]+)\)")
# Security: Provider name sanitization patterns (compiled once for performance)
_PROVIDER_NAME_ALLOWLIST_RE = re.compile(r"[^A-Za-z0-9 \-_.]")
_PROVIDER_NAME_COLLAPSE_RE = re.compile(r"[ _]{2,}")
# Security: Provider slug validation (lowercase ASCII + hyphens, optional /segment)
# Used for validating API slugs like "openai", "amazon-bedrock", "google-vertex"
_PROVIDER_SLUG_PATTERN = re.compile(r"^[a-z0-9-]+(?:/[a-z0-9-]+)?$")
# Security: Quantization level validation (alphanumeric + underscore/hyphen)
# Used for validating quantization values like "int4", "int8", "fp16", "bf16"
_QUANTIZATION_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_PRICING_CATEGORY_KEYS = {
    "prompt",
    "completion",
    "request",
    "image",
    "image_token",
    "image_output",
    "audio",
    "audio_output",
    "input_audio_cache",
    "web_search",
    "internal_reasoning",
    "input_cache_read",
    "input_cache_write",
}
_PRICING_VALUE_KEYS = ("price", "amount", "value", "usd", "cost", "rate")


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

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
    @timed
    def session_id(self) -> str:
        """Convenience accessor for the metadata session identifier."""
        return str(self.metadata.get("session_id") or "")

    @property
    @timed
    def user_id(self) -> str:
        """Return the Open WebUI user id associated with the job."""
        return str(self.user.get("id") or self.metadata.get("user_id") or "")


# -----------------------------------------------------------------------------
# Module-level constants
# -----------------------------------------------------------------------------

# Tool output pruning constants
_TOOL_OUTPUT_PRUNE_MIN_LENGTH = 800  # Minimum length before pruning is applied
_TOOL_OUTPUT_PRUNE_HEAD_CHARS = 256  # Characters to keep from the beginning
_TOOL_OUTPUT_PRUNE_TAIL_CHARS = 128  # Characters to keep from the end

# -----------------------------------------------------------------------------
# Main Pipe Class
# -----------------------------------------------------------------------------

class Pipe:
    """Main orchestration class for OpenRouter pipe with subsystem delegation.

    This class:
    1. Manages lifecycle (init, startup checks, shutdown)
    2. Handles high-level request flow (pipes, pipe methods)
    3. Delegates specific functionality to subsystems
    4. Maintains shared state (HTTP sessions, concurrency controls)
    5. Orchestrates workers, queues, and background tasks

    Subsystem Delegation:
    - ArtifactStore: All persistence (DB, Redis, encryption, cleanup)
    - MultimodalHandler: All file/image operations (upload, download, inline)
    - StreamingHandler: All streaming loops (SSE parsing, delta handling)
    - EventEmitterHandler: All UI events (status, errors, citations, completion)

    Orchestration Kept in Pipe:
    - __init__, pipes(), pipe(), shutdown() - lifecycle
    - _handle_pipe_call(), _process_transformed_request() - request routing
    - transform_messages_to_input() - message transformation
    - send_openrouter_*_request() - HTTP request execution
    - _execute_function_calls() - tool execution
    - _request_worker_loop(), _enqueue_job() - worker management
    - _ensure_concurrency_controls() - semaphore/breaker setup
    - _refresh_model_catalog() - model catalog management
    """

    # Class variables (shared across instances)
    id: str = _PIPE_RUNTIME_ID or "open_webui_openrouter_pipe"
    name: str = "OpenRouter Responses API"

    # Valve classes (must be defined as nested classes for Open WebUI discovery)
    Valves = Valves
    UserValves = UserValves

    # Shared concurrency primitives (class-level for global rate limiting)
    _QUEUE_MAXSIZE = 1000
    _global_semaphore: asyncio.Semaphore | None = None
    _semaphore_limit: int = 0
    _tool_global_semaphore: asyncio.Semaphore | None = None
    _tool_global_limit: int = 0
    _TOOL_CONTEXT: ContextVar[Optional[_ToolExecutionContext]] = ContextVar(
        "openrouter_tool_context",
        default=None,
    )
    # Note: Worker-related state (_request_queue, _queue_worker_task, _queue_worker_lock,
    # _log_queue, _log_queue_loop, _log_worker_task, _log_worker_lock, _cleanup_task)
    # are now INSTANCE-level to prevent event loop contamination across tests.

    @timed
    def __init__(self):
        """Initialize Pipe with subsystem delegation architecture.

        Initialization Order:
        1. Core pipe state (logger, valves, type)
        2. Instance variables for persistence, multimodal, streaming, events
        3. Circuit breaker state
        4. Redis/cache configuration
        5. Startup check coordination
        6. Session logging setup
        """
        # Core pipe identity and configuration
        self.type = "manifold"
        self.valves = self.Valves()
        self.logger = SessionLogger.get_logger(__name__)

        # Instance variables that will be lazy-initialized
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        self._closed = False
        self._shutdown_lock: Optional[asyncio.Lock] = None

        # Instance-level worker state (prevents event loop contamination across tests)
        self._request_queue: asyncio.Queue[_PipeJob] | None = None
        self._queue_worker_task: asyncio.Task | None = None
        self._queue_worker_lock: asyncio.Lock | None = None
        self._log_queue: asyncio.Queue[logging.LogRecord] | None = None
        self._log_queue_loop: asyncio.AbstractEventLoop | None = None
        self._log_worker_task: asyncio.Task | None = None
        self._log_worker_lock: asyncio.Lock | None = None
        self._cleanup_task: asyncio.Task | None = None

        # Subsystem instances (created in __init__, configured later)
        pipe_id = getattr(self, "id", "openrouter")
        self._artifact_store = ArtifactStore(
            pipe_id=pipe_id,
            logger=self.logger,
            valves=self.valves,
            emit_notification_callback=None,  # Will be set immediately below
            tool_context_var=Pipe._TOOL_CONTEXT,
            user_id_context_var=SessionLogger.user_id,
        )

        # Connect ArtifactStore notification callback to Pipe's _emit_notification
        self._artifact_store._emit_notification = self._emit_notification

        # Initialize subsystem handlers synchronously (http_session=None, will be set in async init)
        self._multimodal_handler: Optional[MultimodalHandler] = MultimodalHandler(
            logger=self.logger,
            valves=self.valves,
            http_session=None,  # Will be set in _ensure_async_subsystems_initialized
            artifact_store=None,  # Will be set after artifact store initialization
            emit_status_callback=None,
        )
        self._streaming_handler: Optional[StreamingHandler] = StreamingHandler(
            logger=self.logger,
            valves=self.valves,
            model_registry=OpenRouterModelRegistry,  # Pass the class itself
            pipe_instance=self,
        )
        self._event_emitter_handler: Optional[EventEmitterHandler] = EventEmitterHandler(
            logger=self.logger,
            valves=self.valves,
            pipe_instance=self,
        )
        self._catalog_manager: Optional[ModelCatalogManager] = None  # Lazy init
        self._error_formatter: Optional["ErrorFormatter"] = None  # Lazy init
        self._reasoning_config_manager: Optional[ReasoningConfigManager] = None  # Lazy init
        self._nonstreaming_adapter: Optional["NonStreamingAdapter"] = None  # Lazy init
        self._task_model_adapter: Optional["TaskModelAdapter"] = None  # Lazy init
        self._tool_executor: Optional["ToolExecutor"] = None  # Lazy init
        self._responses_adapter: Optional["ResponsesAdapter"] = None  # Lazy init
        self._chat_completions_adapter: Optional["ChatCompletionsAdapter"] = None  # Lazy init
        self._request_orchestrator: Optional["RequestOrchestrator"] = None  # Lazy init

        # Circuit breaker state (per-user error tracking)
        self._circuit_breaker = CircuitBreaker(
            threshold=self.valves.BREAKER_MAX_FAILURES,
            window_seconds=self.valves.BREAKER_WINDOW_SECONDS,
        )
        # Synchronize circuit breaker config with ArtifactStore
        self._artifact_store._breaker_threshold = self.valves.BREAKER_MAX_FAILURES
        self._artifact_store._breaker_window_seconds = self.valves.BREAKER_WINDOW_SECONDS
        # DB breakers (separate from general circuit breaker)
        breaker_history_size = self.valves.BREAKER_HISTORY_SIZE
        self._db_breakers: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=breaker_history_size)
        )

        # Startup check coordination
        self._startup_task: asyncio.Task | None = None
        self._startup_checks_started = False
        self._startup_checks_pending = False
        self._startup_checks_complete = False
        self._warmup_failed = False

        # Redis configuration (detected from environment)
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
        redis_valve_enabled = self.valves.ENABLE_REDIS_CACHE
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
        self._redis_client = None
        self._redis_listener_task: asyncio.Task | None = None
        self._redis_flush_task: asyncio.Task | None = None
        self._redis_ready_task: asyncio.Task | None = None
        self._redis_namespace = (getattr(self, "id", None) or "openrouter").lower()
        self._redis_pending_key = f"{self._redis_namespace}:pending"
        self._redis_cache_prefix = f"{self._redis_namespace}:artifact"
        self._redis_flush_lock_key = f"{self._redis_namespace}:flush_lock"
        self._redis_ttl = self.valves.REDIS_CACHE_TTL_SECONDS

        # Cleanup tasks
        self._cleanup_task: asyncio.Task | None = None

        # Multimodal state
        self._storage_user_cache: Optional[Any] = None
        self._storage_user_lock: Optional[asyncio.Lock] = None
        self._storage_role_warning_emitted: bool = False

        # Session logging (thread-based background archival)
        self._session_log_queue: queue.Queue[_SessionLogArchiveJob] | None = None
        self._session_log_stop_event: threading.Event | None = None
        self._session_log_worker_thread: threading.Thread | None = None
        self._session_log_cleanup_thread: threading.Thread | None = None
        self._session_log_assembler_thread: threading.Thread | None = None
        self._session_log_lock = threading.Lock()
        self._session_log_cleanup_interval_seconds = self.valves.SESSION_LOG_CLEANUP_INTERVAL_SECONDS
        self._session_log_retention_days = self.valves.SESSION_LOG_RETENTION_DAYS
        self._session_log_dirs: set[str] = set()
        self._session_log_warning_emitted = False
        self._maybe_start_log_worker()

        # Configure timing file if enabled
        if self.valves.ENABLE_TIMING_LOG:
            file_path = self.valves.TIMING_LOG_FILE
            if configure_timing_file(file_path):
                self.logger.info("Timing log enabled: %s", file_path)
            else:
                self.logger.warning("Failed to open timing log file: %s", file_path)

        self._maybe_start_startup_checks()

        self.logger.debug(
            "Pipe initialized (subsystem delegation: ArtifactStore, MultimodalHandler, StreamingHandler, EventEmitterHandler)"
        )

    @timed
    async def _ensure_async_subsystems_initialized(self):
        """Lazy initialization of async-dependent subsystems."""
        if self._initialized:
            return

        # Initialize HTTP session if not already created
        if not self._http_session:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
            self._http_session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
            )

        # Update subsystem handlers with async resources
        if self._multimodal_handler:
            self._multimodal_handler._http_session = self._http_session
            self._multimodal_handler._artifact_store = self._artifact_store

        if not self._streaming_handler:
            self._streaming_handler = StreamingHandler(
                logger=self.logger,
                valves=self.valves,
                model_registry=OpenRouterModelRegistry,
                pipe_instance=self,
            )

        if not self._event_emitter_handler:
            self._event_emitter_handler = EventEmitterHandler(
                logger=self.logger,
                valves=self.valves,
                pipe_instance=self,
            )

        self._initialized = True
        self.logger.debug("Async subsystems initialized")

    # =============================================================================
    # LIFECYCLE & STARTUP HELPERS
    # =============================================================================

    @timed
    def _maybe_start_startup_checks(self) -> None:
        """Schedule background warmup checks once an event loop is available."""
        if self._startup_checks_complete:
            return
        if self._startup_task and not self._startup_task.done():
            return
        if self._startup_task and self._startup_task.done():
            self._startup_task = None
        api_key_value, api_key_error = self._resolve_openrouter_api_key(self.valves)
        api_key_available = bool(api_key_value) and (not api_key_error)
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

    @timed
    def _maybe_start_log_worker(self) -> None:
        """Ensure the async logging queue + worker are started."""
        if getattr(self, "_closed", False):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        # Check if existing lock is bound to a different (stale) event loop
        if self._log_worker_lock is not None:
            try:
                lock_loop = getattr(cast(Any, self._log_worker_lock), "_get_loop", lambda: None)()
                if lock_loop is not loop:
                    self._log_worker_lock = None
            except RuntimeError:
                # Lock is bound to a closed/different loop
                self._log_worker_lock = None

        if self._log_worker_lock is None:
            self._log_worker_lock = asyncio.Lock()

        if self._log_queue is None or self._log_queue_loop is not loop:
            stale_worker = self._log_worker_task
            if stale_worker and not stale_worker.done():
                with contextlib.suppress(Exception):
                    stale_worker.cancel()
            self._log_worker_task = None
            self._log_queue = asyncio.Queue(maxsize=1000)
            self._log_queue_loop = loop
            SessionLogger.set_log_queue(self._log_queue)
        SessionLogger.set_main_loop(loop)

        # Capture self for the nested async function
        pipe_self = self

        @timed
        async def _ensure_worker() -> None:
            if getattr(pipe_self, "_closed", False):
                return
            try:
                async with pipe_self._log_worker_lock:  # type: ignore[arg-type]
                    if getattr(pipe_self, "_closed", False):
                        return
                    if pipe_self._log_worker_task and not pipe_self._log_worker_task.done():
                        return
                    if pipe_self._log_queue is None:
                        pipe_self._log_queue = asyncio.Queue(maxsize=1000)
                        SessionLogger.set_log_queue(pipe_self._log_queue)
                    pipe_self._log_worker_task = loop.create_task(
                        Pipe._log_worker_loop(pipe_self._log_queue),
                        name="openrouter-log-worker",
                    )
            except Exception:
                # Never let background startup tasks create noisy "Task exception was never retrieved"
                pipe_self.logger.debug("Log worker startup task failed", exc_info=True)

        def _consume_background_exception(task: asyncio.Task) -> None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                task.exception()

        start_task = loop.create_task(_ensure_worker(), name="openrouter-log-worker-start")
        start_task.add_done_callback(_consume_background_exception)

    @timed
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

    @timed
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

    @timed
    def _stop_session_log_workers(self) -> None:
        """Stop session log background threads (best effort)."""
        if self._session_log_stop_event:
            with contextlib.suppress(Exception):
                self._session_log_stop_event.set()
        if self._session_log_queue:
            with contextlib.suppress(Exception):
                self._session_log_queue.put_nowait(None)  # type: ignore[arg-type]
        for thread in (self._session_log_worker_thread, self._session_log_cleanup_thread, self._session_log_assembler_thread):
            if thread and thread.is_alive():
                with contextlib.suppress(Exception):
                    thread.join(timeout=2.0)
        self._session_log_worker_thread = None
        self._session_log_cleanup_thread = None
        self._session_log_assembler_thread = None

    @classmethod
    @timed
    def _auth_failure_scope_key(cls) -> str:
        """Return an identifier used to suppress repeated auth failures.

        Prefer stable user ids, otherwise fall back to session id. When both are
        absent, return an empty string (no suppression).
        """
        user_id = (SessionLogger.user_id.get() or "").strip()
        if user_id:
            return f"user:{user_id}"
        session_id = (SessionLogger.session_id.get() or "").strip()
        if session_id:
            return f"session:{session_id}"
        return ""

    @timed
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
                    await client.aclose()
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

    @staticmethod
    @timed
    async def _log_worker_loop(queue: asyncio.Queue) -> None:
        """Drain log records asynchronously to keep handlers non-blocking."""
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

    @timed
    def _ensure_catalog_manager(self) -> ModelCatalogManager:
        """Lazy initialization of ModelCatalogManager."""
        if self._catalog_manager is None:
            self._catalog_manager = ModelCatalogManager(
                pipe=self,
                multimodal_handler=self._multimodal_handler,
                logger=self.logger,
            )
        return self._catalog_manager

    @timed
    def _ensure_error_formatter(self) -> ErrorFormatter:
        """Lazy initialization of ErrorFormatter."""
        if self._error_formatter is None:
            if self._event_emitter_handler is None:
                self._event_emitter_handler = EventEmitterHandler(
                    logger=self.logger,
                    valves=self.valves,
                    pipe_instance=self,
                )
            self._error_formatter = ErrorFormatter(
                pipe=self,
                event_emitter_handler=self._event_emitter_handler,
                logger=self.logger,
            )
        return self._error_formatter

    @timed
    def _ensure_reasoning_config_manager(self) -> ReasoningConfigManager:
        """Lazy initialization of ReasoningConfigManager."""
        if self._reasoning_config_manager is None:
            self._reasoning_config_manager = ReasoningConfigManager(
                pipe=self,
                logger=self.logger,
            )
        return self._reasoning_config_manager

    @timed
    def _ensure_nonstreaming_adapter(self) -> NonStreamingAdapter:
        """Lazy initialization of NonStreamingAdapter."""
        if self._nonstreaming_adapter is None:
            self._nonstreaming_adapter = NonStreamingAdapter(
                pipe=self,
                logger=self.logger,
            )
        return self._nonstreaming_adapter

    @timed
    def _ensure_task_model_adapter(self) -> TaskModelAdapter:
        """Lazy initialization of TaskModelAdapter."""
        if self._task_model_adapter is None:
            self._task_model_adapter = TaskModelAdapter(
                pipe=self,
                logger=self.logger,
            )
        return self._task_model_adapter

    @timed
    def _ensure_tool_executor(self) -> "ToolExecutor":
        """Lazy initialization of ToolExecutor."""
        if self._tool_executor is None:
            from .tools.tool_executor import ToolExecutor
            self._tool_executor = ToolExecutor(
                pipe=self,
                logger=self.logger,
            )
        return self._tool_executor


    @timed
    def _ensure_responses_adapter(self) -> "ResponsesAdapter":
        """Lazy initialization of ResponsesAdapter."""
        if self._responses_adapter is None:
            from .api.gateway.responses_adapter import ResponsesAdapter
            self._responses_adapter = ResponsesAdapter(
                pipe=self,
                logger=self.logger,
            )
        return self._responses_adapter

    @timed
    def _ensure_chat_completions_adapter(self) -> "ChatCompletionsAdapter":
        """Lazy initialization of ChatCompletionsAdapter."""
        if self._chat_completions_adapter is None:
            from .api.gateway.chat_completions_adapter import ChatCompletionsAdapter
            self._chat_completions_adapter = ChatCompletionsAdapter(
                pipe=self,
                logger=self.logger,
            )
        return self._chat_completions_adapter

    @timed
    def _ensure_request_orchestrator(self) -> "RequestOrchestrator":
        """Lazy initialization of RequestOrchestrator."""
        if self._request_orchestrator is None:
            from .requests.orchestrator import RequestOrchestrator
            self._request_orchestrator = RequestOrchestrator(
                pipe=self,
                logger=self.logger,
            )
        return self._request_orchestrator

    @timed
    def _maybe_schedule_model_metadata_sync(
        self,
        selected_models: list[dict[str, Any]],
        *,
        pipe_identifier: str,
    ) -> None:
        """Delegate to ModelCatalogManager."""
        return self._ensure_catalog_manager().maybe_schedule_model_metadata_sync(
            selected_models,
            pipe_identifier=pipe_identifier,
        )

    @staticmethod
    @timed
    def _render_ors_filter_source() -> str:
        """Return the canonical OWUI filter source for the OpenRouter Search toggle."""
        # NOTE: This file is inserted into Open WebUI's Functions table as a *filter* function.
        # It must not depend on this pipe module at runtime.
        return f'''"""
title: OpenRouter Search
author: Open-WebUI-OpenRouter-pipe
	author_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
	id: openrouter_search
	description: Enables OpenRouter's web-search plugin for the OpenRouter pipe and disables Open WebUI Web Search for this request (OpenRouter Search overrides Web Search).
	version: 0.1.0
	license: MIT
	"""

from __future__ import annotations

import logging
from typing import Any

from open_webui.env import SRC_LOG_LEVELS

OWUI_OPENROUTER_PIPE_MARKER = "{_ORS_FILTER_MARKER}"
_FEATURE_FLAG = "{_ORS_FILTER_FEATURE_FLAG}"


class Filter:
    # Toggleable filter (shows a switch in the Integrations menu).
    toggle = True

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.search.toggle")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True

    def inlet(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Signal the pipe via request metadata (preferred path).
        if __metadata__ is not None and not isinstance(__metadata__, dict):
            return body

        features = body.get("features")
        if not isinstance(features, dict):
            features = {{}}
            body["features"] = features

        # Enforce: OpenRouter Search overrides Web Search (prevent OWUI native web search handler).
        features["web_search"] = False

        if isinstance(__metadata__, dict):
            meta_features = __metadata__.get("features")
            if meta_features is None:
                meta_features = {{}}
                __metadata__["features"] = meta_features

            # OWUI builds __metadata__["features"] as a reference to body["features"].
            # Break the reference so we can preserve the marker for the pipe while forcing
            # body.features.web_search = False.
            if meta_features is features:
                meta_features = dict(meta_features)
                __metadata__["features"] = meta_features

            if isinstance(meta_features, dict):
                meta_features[_FEATURE_FLAG] = True

        self.log.debug("Enabled OpenRouter Search; disabled OWUI web_search")
        return body
'''

    @staticmethod
    @timed
    def _render_direct_uploads_filter_source() -> str:
        """Return the canonical OWUI filter source for the OpenRouter Direct Uploads toggle."""
        # NOTE: This file is inserted into Open WebUI's Functions table as a *filter* function.
        # It must not depend on this pipe module at runtime.
        template = '''"""
title: Direct Uploads
author: Open-WebUI-OpenRouter-pipe
author_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: __FILTER_ID__
description: Bypass Open WebUI RAG for chat uploads and forward them to OpenRouter as direct file/audio/video inputs (user-controlled via valves).
version: 0.1.0
license: MIT
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from open_webui.env import SRC_LOG_LEVELS

OWUI_OPENROUTER_PIPE_MARKER = "__MARKER__"


class Filter:
    # Toggleable filter (shows a switch in the Integrations menu).
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )
        DIRECT_TOTAL_PAYLOAD_MAX_MB: int = Field(
            default=50,
            ge=1,
            le=500,
            description="Maximum total size (MB) across all diverted direct uploads in a single request.",
        )
        DIRECT_FILE_MAX_UPLOAD_SIZE_MB: int = Field(
            default=50,
            ge=1,
            le=500,
            description="Maximum size (MB) for a single diverted direct file upload.",
        )
        DIRECT_AUDIO_MAX_UPLOAD_SIZE_MB: int = Field(
            default=25,
            ge=1,
            le=500,
            description="Maximum size (MB) for a single diverted direct audio upload.",
        )
        DIRECT_VIDEO_MAX_UPLOAD_SIZE_MB: int = Field(
            default=20,
            ge=1,
            le=500,
            description="Maximum size (MB) for a single diverted direct video upload.",
        )
        DIRECT_FILE_MIME_ALLOWLIST: str = Field(
            default="application/pdf,text/plain,text/markdown,application/json,text/csv",
            description="Comma-separated MIME allowlist for diverted direct generic files.",
        )
        DIRECT_AUDIO_MIME_ALLOWLIST: str = Field(
            default="audio/*",
            description="Comma-separated MIME allowlist for diverted direct audio files.",
        )
        DIRECT_VIDEO_MIME_ALLOWLIST: str = Field(
            default="video/mp4,video/mpeg,video/quicktime,video/webm",
            description="Comma-separated MIME allowlist for diverted direct video files.",
        )
        DIRECT_AUDIO_FORMAT_ALLOWLIST: str = Field(
            default="wav,mp3,aiff,aac,ogg,flac,m4a,pcm16,pcm24",
            description="Comma-separated audio format allowlist (derived from filename/MIME).",
        )
        DIRECT_RESPONSES_AUDIO_FORMAT_ALLOWLIST: str = Field(
            default="wav,mp3",
            description="Comma-separated audio formats eligible for /responses input_audio.format.",
        )

    class UserValves(BaseModel):
        DIRECT_FILES: bool = Field(
            default=False,
            description="When enabled, uploads files directly to the model.",
        )
        DIRECT_AUDIO: bool = Field(
            default=False,
            description="When enabled, uploads audio directly to the model.",
        )
        DIRECT_VIDEO: bool = Field(
            default=False,
            description="When enabled, uploads video directly to the model.",
        )
        DIRECT_PDF_PARSER: Literal["Native", "PDF Text", "Mistral OCR"] = Field(
            default="Native",
            description="OpenRouter PDF engine selection for direct uploads (requires Direct Files enabled).",
        )

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.direct.uploads")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return int(stripped)
            except ValueError:
                return None
        return None

    @staticmethod
    def _csv_set(value: Any) -> set[str]:
        if not isinstance(value, str):
            return set()
        parts = []
        for raw in value.split(","):
            item = (raw or "").strip().lower()
            if item:
                parts.append(item)
        return set(parts)

    @staticmethod
    def _mime_allowed(mime: str, allowlist_csv: str) -> bool:
        mime = (mime or "").strip().lower()
        if not mime:
            return False
        allowlist = Filter._csv_set(allowlist_csv)
        if not allowlist:
            return False
        for pattern in allowlist:
            if fnmatch.fnmatch(mime, pattern):
                return True
        return False

    @staticmethod
    def _infer_audio_format(name: Any, mime: Any) -> str:
        mime_str = (mime or "").strip().lower() if isinstance(mime, str) else ""
        if mime_str in {"audio/wav", "audio/wave", "audio/x-wav"}:
            return "wav"
        if mime_str in {"audio/mpeg", "audio/mp3"}:
            return "mp3"
        filename = (name or "").strip().lower() if isinstance(name, str) else ""
        if "." in filename:
            ext = filename.rsplit(".", 1)[-1].strip().lower()
            if ext:
                return ext
        return ""

    @staticmethod
    def _model_caps(__model__: Any) -> dict[str, bool]:
        if not isinstance(__model__, dict):
            return {}
        meta = __model__.get("info", {}).get("meta", {})
        if not isinstance(meta, dict):
            return {}
        pipe_meta = meta.get("openrouter_pipe", {})
        if not isinstance(pipe_meta, dict):
            return {}
        caps = pipe_meta.get("capabilities", {})
        return caps if isinstance(caps, dict) else {}

    def inlet(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
        __model__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(body, dict):
            return body
        if __metadata__ is not None and not isinstance(__metadata__, dict):
            return body
        if __user__ is not None and not isinstance(__user__, dict):
            __user__ = None

        user_valves = None
        if isinstance(__user__, dict):
            user_valves = __user__.get("valves")
        if not isinstance(user_valves, BaseModel):
            user_valves = self.UserValves()

        enable_files = bool(getattr(user_valves, "DIRECT_FILES", False))
        enable_audio = bool(getattr(user_valves, "DIRECT_AUDIO", False))
        enable_video = bool(getattr(user_valves, "DIRECT_VIDEO", False))
        pdf_parser = getattr(user_valves, "DIRECT_PDF_PARSER", "Native")

        files = body.get("files", None)
        if not isinstance(files, list) or not files:
            return body

        caps = self._model_caps(__model__)
        supports_files = bool(caps.get("file_input", False))
        supports_audio = bool(caps.get("audio_input", False))
        supports_video = bool(caps.get("video_input", False))

        diverted: dict[str, list[dict[str, Any]]] = {"files": [], "audio": [], "video": []}
        retained: list[Any] = []
        warnings: list[str] = []
        total_bytes = 0
        pdf_seen = False

        total_limit = int(self.valves.DIRECT_TOTAL_PAYLOAD_MAX_MB) * 1024 * 1024
        file_limit = int(self.valves.DIRECT_FILE_MAX_UPLOAD_SIZE_MB) * 1024 * 1024
        audio_limit = int(self.valves.DIRECT_AUDIO_MAX_UPLOAD_SIZE_MB) * 1024 * 1024
        video_limit = int(self.valves.DIRECT_VIDEO_MAX_UPLOAD_SIZE_MB) * 1024 * 1024

        audio_formats_allowed = self._csv_set(self.valves.DIRECT_AUDIO_FORMAT_ALLOWLIST)

        for item in files:
            if not isinstance(item, dict):
                retained.append(item)
                continue
            if bool(item.get("legacy", False)):
                retained.append(item)
                continue
            if (item.get("type") or "file") != "file":
                retained.append(item)
                continue
            file_id = item.get("id")
            if not isinstance(file_id, str) or not file_id.strip():
                retained.append(item)
                continue

            content_type = (
                item.get("content_type")
                or item.get("contentType")
                or item.get("mime_type")
                or item.get("mimeType")
                or ""
            )
            content_type = content_type.strip().lower() if isinstance(content_type, str) else ""
            name = item.get("name") or ""
            filename = name.strip().lower() if isinstance(name, str) else ""
            is_pdf = ("pdf" in content_type) or filename.endswith(".pdf")

            size_bytes = self._to_int(item.get("size"))
            if size_bytes is None or size_bytes < 0:
                raise Exception("Direct uploads: uploaded file missing a valid size.")

            kind = "files"
            if content_type.startswith("audio/"):
                kind = "audio"
            elif content_type.startswith("video/"):
                kind = "video"

            if kind == "files":
                if not enable_files:
                    retained.append(item)
                    continue
                if not supports_files:
                    warnings.append("Direct file uploads not supported by the selected model; falling back to Open WebUI.")
                    retained.append(item)
                    continue
                if not self._mime_allowed(content_type, self.valves.DIRECT_FILE_MIME_ALLOWLIST):
                    # Fail-open: leave unsupported types on the normal OWUI path (RAG/Knowledge).
                    retained.append(item)
                    continue
                if size_bytes > file_limit:
                    raise Exception(
                        f"Direct file '{name or file_id}' is too large ({size_bytes} bytes; max {self.valves.DIRECT_FILE_MAX_UPLOAD_SIZE_MB} MB)."
                    )
                total_bytes += size_bytes
                if total_bytes > total_limit:
                    raise Exception(
                        f"Direct uploads exceed total limit ({self.valves.DIRECT_TOTAL_PAYLOAD_MAX_MB} MB)."
                    )
                diverted["files"].append(
                    {
                        "id": file_id,
                        "name": name,
                        "size": size_bytes,
                        "content_type": content_type,
                    }
                )
                if is_pdf:
                    pdf_seen = True
                continue

            if kind == "audio":
                if not enable_audio:
                    retained.append(item)
                    continue
                if not supports_audio:
                    warnings.append("Direct audio uploads not supported by the selected model; falling back to Open WebUI.")
                    retained.append(item)
                    continue
                if not self._mime_allowed(content_type, self.valves.DIRECT_AUDIO_MIME_ALLOWLIST):
                    retained.append(item)
                    continue
                audio_format = self._infer_audio_format(name, content_type)
                if not audio_format or (audio_formats_allowed and audio_format not in audio_formats_allowed):
                    retained.append(item)
                    continue
                if size_bytes > audio_limit:
                    raise Exception(
                        f"Direct audio '{name or file_id}' is too large ({size_bytes} bytes; max {self.valves.DIRECT_AUDIO_MAX_UPLOAD_SIZE_MB} MB)."
                    )
                total_bytes += size_bytes
                if total_bytes > total_limit:
                    raise Exception(
                        f"Direct uploads exceed total limit ({self.valves.DIRECT_TOTAL_PAYLOAD_MAX_MB} MB)."
                    )
                diverted["audio"].append(
                    {
                        "id": file_id,
                        "name": name,
                        "size": size_bytes,
                        "content_type": content_type,
                        "format": audio_format,
                    }
                )
                continue

            if kind == "video":
                if not enable_video:
                    retained.append(item)
                    continue
                if not supports_video:
                    warnings.append("Direct video uploads not supported by the selected model; falling back to Open WebUI.")
                    retained.append(item)
                    continue
                if not self._mime_allowed(content_type, self.valves.DIRECT_VIDEO_MIME_ALLOWLIST):
                    retained.append(item)
                    continue
                if size_bytes > video_limit:
                    raise Exception(
                        f"Direct video '{name or file_id}' is too large ({size_bytes} bytes; max {self.valves.DIRECT_VIDEO_MAX_UPLOAD_SIZE_MB} MB)."
                    )
                total_bytes += size_bytes
                if total_bytes > total_limit:
                    raise Exception(
                        f"Direct uploads exceed total limit ({self.valves.DIRECT_TOTAL_PAYLOAD_MAX_MB} MB)."
                    )
                diverted["video"].append(
                    {
                        "id": file_id,
                        "name": name,
                        "size": size_bytes,
                        "content_type": content_type,
                    }
                )
                continue

            retained.append(item)

        diverted_any = bool(diverted["files"] or diverted["audio"] or diverted["video"])
        # OWUI "File Context" reads `body["metadata"]["files"]`, but OWUI also rebuilds metadata.files
        # from `body["files"]` after inlet filters. To reliably bypass OWUI RAG for diverted uploads,
        # update both.
        if diverted_any:
            body["files"] = retained
            if isinstance(__metadata__, dict):
                __metadata__["files"] = retained

        if isinstance(__metadata__, dict) and (diverted_any or warnings):
            prev_pipe_meta = __metadata__.get("openrouter_pipe")
            pipe_meta = dict(prev_pipe_meta) if isinstance(prev_pipe_meta, dict) else {}
            __metadata__["openrouter_pipe"] = pipe_meta

            if warnings:
                prev_warnings = pipe_meta.get("direct_uploads_warnings")
                merged_warnings: list[str] = []
                seen: set[str] = set()
                if isinstance(prev_warnings, list):
                    for warning in prev_warnings:
                        if isinstance(warning, str) and warning and warning not in seen:
                            seen.add(warning)
                            merged_warnings.append(warning)
                for warning in warnings:
                    if warning and warning not in seen:
                        seen.add(warning)
                        merged_warnings.append(warning)
                pipe_meta["direct_uploads_warnings"] = merged_warnings

            if diverted_any:
                prev_attachments = pipe_meta.get("direct_uploads")
                attachments = dict(prev_attachments) if isinstance(prev_attachments, dict) else {}
                pipe_meta["direct_uploads"] = attachments
                # Persist the /responses audio format allowlist into metadata so the pipe can honor it at injection time.
                attachments["responses_audio_format_allowlist"] = self.valves.DIRECT_RESPONSES_AUDIO_FORMAT_ALLOWLIST
                if pdf_seen and isinstance(pdf_parser, str) and pdf_parser.strip():
                    attachments["pdf_parser"] = pdf_parser.strip()

                for key in ("files", "audio", "video"):
                    items = diverted.get(key) or []
                    if items:
                        existing = attachments.get(key)
                        merged: list[dict[str, Any]] = []
                        seen: set[str] = set()
                        if isinstance(existing, list):
                            for entry in existing:
                                if isinstance(entry, dict):
                                    eid = entry.get("id")
                                    if isinstance(eid, str) and eid and eid not in seen:
                                        seen.add(eid)
                                        merged.append(entry)
                        for entry in items:
                            eid = entry.get("id")
                            if isinstance(eid, str) and eid and eid not in seen:
                                seen.add(eid)
                                merged.append(entry)
                        attachments[key] = merged

        if diverted_any:
            self.log.debug("Diverted %d byte(s) for direct upload forwarding", total_bytes)
        return body
'''

        return (
            template.replace("__FILTER_ID__", _DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID)
            .replace("__MARKER__", _DIRECT_UPLOADS_FILTER_MARKER)
        )

    @timed
    def _ensure_ors_filter_function_id(self) -> str | None:
        """Ensure the OpenRouter Search companion filter exists (and is up to date), returning its OWUI function id."""
        try:
            from open_webui.models.functions import Functions  # type: ignore
        except Exception:
            return None

        desired_source = self._render_ors_filter_source().strip() + "\n"
        desired_name = "OpenRouter Search"
        desired_meta = {
            "description": (
                "Enable OpenRouter native web search for this request. "
                "When OpenRouter Search is enabled, OWUI Web Search is disabled to avoid double-search."
            ),
            "toggle": True,
            "manifest": {
                "title": "OpenRouter Search",
                "id": "openrouter_search",
                "version": "0.1.0",
                "license": "MIT",
            },
        }

        @timed
        def _matches_candidate(content: str) -> bool:
            if not isinstance(content, str) or not content:
                return False
            if _ORS_FILTER_MARKER in content:
                return True
            # Back-compat for manual installs of earlier drafts: detect by the feature flag string.
            return _ORS_FILTER_FEATURE_FLAG in content and "class Filter" in content

        try:
            filters = Functions.get_functions_by_type("filter", active_only=False)
        except Exception:
            return None

        candidates = [f for f in filters if _matches_candidate(getattr(f, "content", ""))]
        chosen = None
        if candidates:
            # Prefer the canonical marker when present.
            marked = [f for f in candidates if _ORS_FILTER_MARKER in (getattr(f, "content", "") or "")]
            if marked:
                candidates = marked
            chosen = sorted(candidates, key=lambda f: int(getattr(f, "updated_at", 0) or 0), reverse=True)[0]
            if len(candidates) > 1:
                self.logger.warning(
                    "Multiple OpenRouter Search filter candidates found (%d); using '%s'.",
                    len(candidates),
                    getattr(chosen, "id", ""),
                )

        if chosen is None:
            if not getattr(self.valves, "AUTO_INSTALL_ORS_FILTER", False):
                return None

            candidate_id = _ORS_FILTER_PREFERRED_FUNCTION_ID
            suffix = 0
            while True:
                existing = None
                try:
                    existing = Functions.get_function_by_id(candidate_id)
                except Exception:
                    existing = None
                if existing is None:
                    break
                suffix += 1
                candidate_id = f"{_ORS_FILTER_PREFERRED_FUNCTION_ID}_{suffix}"
                if suffix > 50:
                    return None

            try:
                from open_webui.models.functions import FunctionForm, FunctionMeta  # type: ignore
            except Exception:
                return None

            meta_obj = FunctionMeta(**desired_meta)
            form = FunctionForm(
                id=candidate_id,
                name=desired_name,
                content=desired_source,
                meta=meta_obj,
            )
            created = Functions.insert_new_function("", "filter", form)
            if not created:
                return None
            Functions.update_function_by_id(candidate_id, {"is_active": True, "is_global": False, "name": desired_name, "meta": desired_meta})
            self.logger.info("Installed OpenRouter Search filter: %s", candidate_id)
            return candidate_id

        function_id = str(getattr(chosen, "id", "") or "").strip()
        if not function_id:
            return None

        if getattr(self.valves, "AUTO_INSTALL_ORS_FILTER", False):
            existing_content = (getattr(chosen, "content", "") or "").strip() + "\n"
            if existing_content != desired_source:
                self.logger.info("Updating OpenRouter Search filter: %s", function_id)
                Functions.update_function_by_id(
                    function_id,
                    {
                        "content": desired_source,
                        "name": desired_name,
                        "meta": desired_meta,
                        "type": "filter",
                        "is_active": True,
                        "is_global": False,
                    },
                )
            else:
                Functions.update_function_by_id(
                    function_id,
                    {
                        "name": desired_name,
                        "meta": desired_meta,
                        "type": "filter",
                        "is_active": True,
                        "is_global": False,
                    },
                )

        return function_id

    @timed
    def _ensure_direct_uploads_filter_function_id(self) -> str | None:
        """Ensure the OpenRouter Direct Uploads companion filter exists (and is up to date), returning its OWUI function id."""
        try:
            from open_webui.models.functions import Functions  # type: ignore
        except Exception:
            return None

        desired_source = self._render_direct_uploads_filter_source().strip() + "\n"
        desired_name = "Direct Uploads"
        desired_meta = {
            "description": (
                "Bypass Open WebUI RAG for chat uploads and forward them to OpenRouter as direct file/audio/video inputs. "
                "Enable files/audio/video via filter user valves."
            ),
            "toggle": True,
            "manifest": {
                "title": "Direct Uploads",
                "id": _DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID,
                "version": "0.1.0",
                "license": "MIT",
            },
        }

        @timed
        def _matches_candidate(content: str) -> bool:
            if not isinstance(content, str) or not content:
                return False
            return _DIRECT_UPLOADS_FILTER_MARKER in content and "class Filter" in content

        try:
            filters = Functions.get_functions_by_type("filter", active_only=False)
        except Exception:
            return None

        candidates = [f for f in filters if _matches_candidate(getattr(f, "content", ""))]
        chosen = None
        if candidates:
            chosen = sorted(candidates, key=lambda f: int(getattr(f, "updated_at", 0) or 0), reverse=True)[0]
            if len(candidates) > 1:
                self.logger.warning(
                    "Multiple OpenRouter Direct Uploads filter candidates found (%d); using '%s'.",
                    len(candidates),
                    getattr(chosen, "id", ""),
                )

        if chosen is None:
            if not getattr(self.valves, "AUTO_INSTALL_DIRECT_UPLOADS_FILTER", False):
                return None

            candidate_id = _DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID
            suffix = 0
            while True:
                existing = None
                try:
                    existing = Functions.get_function_by_id(candidate_id)
                except Exception:
                    existing = None
                if existing is None:
                    break
                suffix += 1
                candidate_id = f"{_DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID}_{suffix}"
                if suffix > 50:
                    return None

            try:
                from open_webui.models.functions import FunctionForm, FunctionMeta  # type: ignore
            except Exception:
                return None

            meta_obj = FunctionMeta(**desired_meta)
            form = FunctionForm(
                id=candidate_id,
                name=desired_name,
                content=desired_source,
                meta=meta_obj,
            )
            created = Functions.insert_new_function("", "filter", form)
            if not created:
                return None
            Functions.update_function_by_id(
                candidate_id,
                {
                    "is_active": True,
                    "is_global": False,
                    "name": desired_name,
                    "meta": desired_meta,
                },
            )
            self.logger.info("Installed OpenRouter Direct Uploads filter: %s", candidate_id)
            return candidate_id

        function_id = str(getattr(chosen, "id", "") or "").strip()
        if not function_id:
            return None

        if getattr(self.valves, "AUTO_INSTALL_DIRECT_UPLOADS_FILTER", False):
            existing_content = (getattr(chosen, "content", "") or "").strip() + "\n"
            if existing_content != desired_source:
                self.logger.info("Updating OpenRouter Direct Uploads filter: %s", function_id)
                Functions.update_function_by_id(
                    function_id,
                    {
                        "content": desired_source,
                        "name": desired_name,
                        "meta": desired_meta,
                        "type": "filter",
                        "is_active": True,
                        "is_global": False,
                    },
                )
            else:
                Functions.update_function_by_id(
                    function_id,
                    {
                        "name": desired_name,
                        "meta": desired_meta,
                        "type": "filter",
                        "is_active": True,
                        "is_global": False,
                    },
                )

        return function_id

    # -------------------------------------------------------------------------
    # Provider Routing Filter Generation
    # -------------------------------------------------------------------------

    # State hash to prevent unnecessary filter regeneration
    _provider_routing_state_hash: str = ""

    @staticmethod
    @timed
    def _sanitize_model_for_filter_id(model_slug: str) -> str:
        """Convert model slug to safe filter ID component.

        Example: 'openai/gpt-4o' -> 'openai_gpt_4o'
        """
        return model_slug.replace("/", "_").replace("-", "_").replace(".", "_")

    @staticmethod
    def _safe_literal_string(s: str) -> str:
        """Safely escape a string for use in Python Literal type annotations.

        Uses Python's built-in repr() to produce a safely escaped string
        representation that can be interpolated into generated Python source
        code without risk of code injection.

        Security Purpose:
            Provider names from the OpenRouter API are untrusted external input.
            When interpolating these into generated filter source code (particularly
            in Literal['...'] type annotations), malicious input like:
                "foo', None); import os; os.system('rm -rf /'); x = ('"
            could escape the string context and execute arbitrary code.

            Using repr() produces properly escaped output that remains a safe,
            quoted string literal.

        Args:
            s: The raw string to escape (e.g., provider name, model slug).

        Returns:
            A repr()-escaped string safe for interpolation into Python source.
            The returned string includes surrounding quotes.

        Example:
            >>> Pipe._safe_literal_string("Amazon Bedrock")
            "'Amazon Bedrock'"
            >>> Pipe._safe_literal_string("evil'; import os; #")
            "\"evil'; import os; #\""
        """
        if not isinstance(s, str):
            s = str(s) if s is not None else ""
        return repr(s)

    @staticmethod
    def _validate_provider_name(name: str, slug: str = "") -> str:
        """Validate and sanitize a provider name for safe use in generated code.

        Enforces strict character allowlists and length limits on provider names
        to prevent injection attacks and ensure predictable behavior in generated
        filter source code.

        Security Purpose:
            Provider names from external APIs may contain unexpected characters
            that could:
            1. Break Python syntax when interpolated into source code
            2. Exploit edge cases in string parsing
            3. Cause display issues in the UI
            4. Enable homograph attacks with Unicode lookalikes

            This function sanitizes to ASCII alphanumeric + limited punctuation,
            and truncates long names with a hash suffix to maintain uniqueness.

        Args:
            name: The raw provider name from the API.
            slug: Optional provider slug for hash uniqueness when truncating.

        Returns:
            A sanitized provider name safe for use in generated code.
            - Only ASCII letters, digits, spaces, hyphens, underscores, periods
            - Maximum 64 characters (truncated with hash suffix if longer)
            - Empty/whitespace-only input returns "Unknown"

        Example:
            >>> Pipe._validate_provider_name("Amazon Bedrock")
            'Amazon Bedrock'
            >>> Pipe._validate_provider_name("Evil<script>Provider")
            'EvilscriptProvider'
        """
        # Handle empty/None input
        if not name or not isinstance(name, str):
            return "Unknown"

        # Strip leading/trailing whitespace
        cleaned = name.strip()
        if not cleaned:
            return "Unknown"

        # Remove any characters not in our allowlist (use pre-compiled pattern)
        # Allow: ASCII letters, digits, space, hyphen, underscore, period
        cleaned = _PROVIDER_NAME_ALLOWLIST_RE.sub("", cleaned)

        # Collapse multiple spaces/underscores (use pre-compiled pattern)
        cleaned = _PROVIDER_NAME_COLLAPSE_RE.sub(" ", cleaned)
        cleaned = cleaned.strip()

        if not cleaned:
            return "Unknown"

        # Enforce maximum length with hash suffix for uniqueness
        max_length = 64
        if len(cleaned) > max_length:
            # Use slug for hash if provided, otherwise use the cleaned name itself
            hash_source = slug if slug else cleaned
            hash_suffix = hashlib.md5(hash_source.encode("utf-8")).hexdigest()[:8]
            # Truncate to leave room for underscore + 8-char hash
            truncated = cleaned[: max_length - 9].rstrip(" _-.")
            # Handle edge case: if truncated is empty after rstrip, use fallback
            if not truncated:
                cleaned = f"Provider_{hash_suffix}"
            else:
                cleaned = f"{truncated}_{hash_suffix}"

        return cleaned

    @staticmethod
    def _validate_filter_source(source: str) -> tuple[bool, str | None]:
        """Validate generated Python source code for syntactic correctness.

        Uses Python's ast.parse() to verify that generated filter source code
        is syntactically valid before it is stored and potentially executed
        by Open WebUI.

        Security Purpose:
            This serves as a defense-in-depth measure. Even if string escaping
            and validation functions work correctly, this provides a final
            safety check that:
            1. The generated code is valid Python syntax
            2. No injection attacks have produced malformed code
            3. Template rendering hasn't introduced syntax errors

        Note:
            This validates SYNTAX only, not semantic safety. It cannot detect
            syntactically valid but malicious code. The earlier sanitization
            functions (_safe_literal_string, _validate_provider_name) are the
            primary defense against injection.

        Args:
            source: The complete generated Python source code to validate.

        Returns:
            A tuple of (is_valid, error_message):
            - (True, None) if the source is syntactically valid
            - (False, error_description) if parsing fails

        Example:
            >>> Pipe._validate_filter_source("x = 1")
            (True, None)
            >>> Pipe._validate_filter_source("x = ")
            (False, "Line 1: unexpected EOF while parsing")
        """
        if not source or not isinstance(source, str):
            return False, "Empty or invalid source"

        try:
            ast.parse(source)
            return True, None
        except SyntaxError as e:
            # Include line number and message for debugging
            # Note: Use str(e) rather than e.msg for cross-version compatibility
            if e.lineno:
                error_msg = f"Line {e.lineno}: {e}"
            else:
                error_msg = str(e)
            return False, error_msg
        except Exception as e:
            # Catch any other parsing errors
            return False, f"Parse error: {str(e)}"

    @staticmethod
    @timed
    def _compute_provider_routing_hash(
        admin_models: str,
        user_models: str,
        provider_map: dict[str, dict[str, list[str]]],
    ) -> str:
        """Compute hash of provider routing state to detect changes."""
        # Include sorted versions of relevant data
        admin_sorted = sorted([m.strip() for m in admin_models.split(",") if m.strip()])
        user_sorted = sorted([m.strip() for m in user_models.split(",") if m.strip()])

        # Include provider info for models we care about
        relevant_slugs = set(admin_sorted) | set(user_sorted)
        provider_data = {
            slug: provider_map.get(slug, {})
            for slug in sorted(relevant_slugs)
        }

        data = f"{admin_sorted}|{user_sorted}|{provider_data}"
        return hashlib.md5(data.encode()).hexdigest()

    @staticmethod
    @timed
    def _render_provider_routing_filter_source(
        model_slug: str,
        providers: list[str],
        quantizations: list[str],
        visibility: str,  # "admin", "user", or "both"
        *,
        short_name: str = "",
        provider_names: dict[str, str] | None = None,
    ) -> str:
        """Generate filter source code for a specific model's provider routing.

        Args:
            model_slug: The OpenRouter model slug (e.g., 'openai/gpt-4o')
            providers: List of available provider slugs
            quantizations: List of available quantization levels
            visibility: Who can configure - 'admin' (enforced), 'user' (optional), or 'both'
            short_name: Human-readable model name for filter title (e.g., 'GPT-4o')
            provider_names: Mapping of provider slug to display name (e.g., {'openai': 'OpenAI'})
        """
        safe_id = Pipe._sanitize_model_for_filter_id(model_slug)
        filter_id = f"{_PROVIDER_ROUTING_FILTER_ID_PREFIX}{safe_id}"

        # SECURITY: Validate model_slug format before interpolating into generated code
        # Model slugs should be: vendor/model-name (e.g., "openai/gpt-4o")
        if not isinstance(model_slug, str) or not model_slug:
            raise ValueError("model_slug must be a non-empty string")
        # Use json.dumps for consistent double-quote escaping (repr uses single quotes
        # by default which causes quote-type mismatch when interpolated into "..." templates)
        safe_model_slug_escaped = json.dumps(model_slug)[1:-1]  # Strip surrounding quotes

        # Use short_name for display, fallback to model_slug if not provided
        display_name = short_name.strip() if short_name else model_slug.split("/")[-1]
        # SECURITY: Sanitize display name for use in filter title
        safe_display_name = Pipe._validate_provider_name(display_name, slug=model_slug)
        safe_display_name_escaped = json.dumps(safe_display_name)[1:-1]

        marker = f"{_PROVIDER_ROUTING_FILTER_MARKER_PREFIX}{model_slug}:{_PROVIDER_ROUTING_FILTER_MARKER_VERSION}"
        # Escape marker for safe interpolation (contains model_slug)
        safe_marker_escaped = json.dumps(marker)[1:-1]

        # SECURITY: Validate provider slugs before interpolation (defense-in-depth)
        # Provider slugs from API should be lowercase ASCII + hyphens (e.g., "amazon-bedrock")
        safe_providers = [
            p for p in providers
            if isinstance(p, str) and _PROVIDER_SLUG_PATTERN.match(p) and len(p) <= 64
        ]
        providers_str = ", ".join(safe_providers) if safe_providers else "none"

        # Build provider slug -> display name mapping (for dropdowns)
        prov_names = provider_names or {}
        # Create display options for ONLY/IGNORE dropdowns: "Display Name" -> "slug"
        # Format: Literal["(no preference)", "OpenAI", "Azure", ...]
        provider_display_options: list[str] = []
        provider_slug_map_entries: list[str] = []  # For the _PROVIDER_MAP dict
        for pslug in safe_providers:
            # Get display name, fallback to titlecased slug
            disp = prov_names.get(pslug, pslug.replace("-", " ").title())
            # SECURITY: Sanitize display name
            safe_disp = Pipe._validate_provider_name(disp, slug=pslug)
            provider_display_options.append(safe_disp)
            # Map: "OpenAI" -> "openai"
            provider_slug_map_entries.append(f'    {Pipe._safe_literal_string(safe_disp)}: {Pipe._safe_literal_string(pslug)}')

        # Build Literal type string for ONLY/IGNORE fields
        # Include "(no preference)" as the default/empty option
        no_pref = "(no preference)"
        only_ignore_options = [no_pref] + provider_display_options
        only_ignore_literal = ", ".join(Pipe._safe_literal_string(opt) for opt in only_ignore_options)

        # Build provider map code block
        provider_map_code = "{\n" + ",\n".join(provider_slug_map_entries) + "\n}" if provider_slug_map_entries else "{}"

        # Build ORDER permutations - FULL permutations with ALL providers in each entry
        # 2 providers = 2! = 2 choices, 3 providers = 3! = 6 choices
        # Display format: "OpenAI > Azure > Together" (all providers in priority order)
        order_display_options: list[str] = []
        order_map_entries: list[str] = []  # For the _ORDER_MAP dict

        # Create mapping from display name to slug for lookups
        display_to_slug = dict(zip(provider_display_options, safe_providers))

        # Generate all n! full permutations of ALL providers
        for perm in itertools.permutations(provider_display_options):
            # Display: "OpenAI > Azure > Together"
            perm_disp = " > ".join(perm)
            order_display_options.append(perm_disp)
            # Slugs: ["openai", "azure", "together"]
            perm_slugs = [display_to_slug[d] for d in perm]
            slugs_literal = ", ".join(Pipe._safe_literal_string(s) for s in perm_slugs)
            order_map_entries.append(f'    {Pipe._safe_literal_string(perm_disp)}: [{slugs_literal}]')

        # Build Literal type string for ORDER field
        order_options = [no_pref] + order_display_options
        order_literal = ", ".join(Pipe._safe_literal_string(opt) for opt in order_options)

        # Build order map code block
        order_map_code = "{\n" + ",\n".join(order_map_entries) + "\n}" if order_map_entries else "{}"

        # SECURITY: Validate quantization levels before interpolation
        # Quantizations should be alphanumeric (e.g., "int4", "fp16", "bf16")
        safe_quantizations = [
            q for q in quantizations
            if isinstance(q, str) and _QUANTIZATION_PATTERN.match(q) and len(q) <= 32
        ]

        # Build Literal type string for QUANTIZATIONS dropdown
        quant_options = [no_pref] + safe_quantizations
        quantizations_literal = ", ".join(Pipe._safe_literal_string(q) for q in quant_options)

        # Determine toggle setting based on visibility
        # ADMIN-only: toggle=False (always runs, user can't disable)
        # USER or BOTH: toggle=True (user can toggle per-chat)
        toggle_value = "False" if visibility == "admin" else "True"

        # Generate Valves class (for admin) if visibility is 'admin' or 'both'
        # Field order matches OpenRouter API docs: provider-selection.md
        valves_class = ""
        if visibility in ("admin", "both"):
            valves_class = f'''
    class Valves(BaseModel):
        """Admin-level provider routing preferences (cannot be disabled by users)."""
        ORDER: Literal[{order_literal}] = Field(default=_NO_PREF, description="Provider priority order")
        ALLOW_FALLBACKS: bool = Field(default=True, description="Allow backup providers if preferred unavailable")
        REQUIRE_PARAMETERS: bool = Field(default=False, description="Only use providers supporting all request params")
        DATA_COLLECTION: Literal[_NO_PREF, "allow", "deny"] = Field(default=_NO_PREF, description="Data collection policy")
        ZDR: bool = Field(default=False, description="Zero Data Retention - only ZDR endpoints")
        ONLY: Literal[{only_ignore_literal}] = Field(default=_NO_PREF, description="Use only this provider")
        IGNORE: Literal[{only_ignore_literal}] = Field(default=_NO_PREF, description="Avoid this provider")
        QUANTIZATION: Literal[{quantizations_literal}] = Field(default=_NO_PREF, description="Filter by quantization")
        SORT: Literal[_NO_PREF, "price", "throughput", "latency"] = Field(default=_NO_PREF, description="Sort providers by")
        MIN_THROUGHPUT: float = Field(default=0, ge=0, description="Min throughput (tokens/sec), 0=no pref")
        MAX_LATENCY: float = Field(default=0, ge=0, description="Max latency (seconds), 0=no pref")
        MAX_PRICE_PROMPT: float = Field(default=0, ge=0, description="Max price for prompt ($/M tokens), 0=no limit")
        MAX_PRICE_COMPLETION: float = Field(default=0, ge=0, description="Max price for completion ($/M tokens), 0=no limit")
        MAX_PRICE_IMAGE: float = Field(default=0, ge=0, description="Max price per image ($/image), 0=no limit")
        MAX_PRICE_AUDIO: float = Field(default=0, ge=0, description="Max price for audio ($/unit), 0=no limit")
        MAX_PRICE_REQUEST: float = Field(default=0, ge=0, description="Max price per request ($/request), 0=no limit")
'''

        # Generate UserValves class (for user) if visibility is 'user' or 'both'
        # Field order matches OpenRouter API docs: provider-selection.md
        user_valves_class = ""
        if visibility in ("user", "both"):
            user_valves_class = f'''
    class UserValves(BaseModel):
        """User-level provider routing preferences (can override admin defaults)."""
        ORDER: Literal[{order_literal}] = Field(default=_NO_PREF, description="Provider priority order")
        ALLOW_FALLBACKS: bool = Field(default=True, description="Allow backup providers if preferred unavailable")
        REQUIRE_PARAMETERS: bool = Field(default=False, description="Only use providers supporting all request params")
        DATA_COLLECTION: Literal[_NO_PREF, "allow", "deny"] = Field(default=_NO_PREF, description="Data collection policy")
        ZDR: bool = Field(default=False, description="Zero Data Retention - only ZDR endpoints")
        ONLY: Literal[{only_ignore_literal}] = Field(default=_NO_PREF, description="Use only this provider")
        IGNORE: Literal[{only_ignore_literal}] = Field(default=_NO_PREF, description="Avoid this provider")
        QUANTIZATION: Literal[{quantizations_literal}] = Field(default=_NO_PREF, description="Filter by quantization")
        SORT: Literal[_NO_PREF, "price", "throughput", "latency"] = Field(default=_NO_PREF, description="Sort providers by")
        MIN_THROUGHPUT: float = Field(default=0, ge=0, description="Min throughput (tokens/sec), 0=no pref")
        MAX_LATENCY: float = Field(default=0, ge=0, description="Max latency (seconds), 0=no pref")
        MAX_PRICE_PROMPT: float = Field(default=0, ge=0, description="Max price for prompt ($/M tokens), 0=no limit")
        MAX_PRICE_COMPLETION: float = Field(default=0, ge=0, description="Max price for completion ($/M tokens), 0=no limit")
        MAX_PRICE_IMAGE: float = Field(default=0, ge=0, description="Max price per image ($/image), 0=no limit")
        MAX_PRICE_AUDIO: float = Field(default=0, ge=0, description="Max price for audio ($/unit), 0=no limit")
        MAX_PRICE_REQUEST: float = Field(default=0, ge=0, description="Max price per request ($/request), 0=no limit")
'''

        # Generate init based on visibility
        init_body = "        self.log = logging.getLogger(f\"openrouter.provider.{MODEL_SLUG}\")\n        self.log.setLevel(SRC_LOG_LEVELS.get(\"OPENAI\", logging.INFO))"
        if visibility in ("admin", "both"):
            init_body += "\n        self.valves = self.Valves()"
        if visibility in ("user", "both"):
            init_body += "\n        self.user_valves = self.UserValves()"

        # Generate inlet logic based on visibility
        inlet_logic = Pipe._generate_inlet_logic(visibility)

        return f'''"""
title: Provider: {safe_display_name_escaped}
author: Open-WebUI-OpenRouter-pipe
author_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: {filter_id}
description: Provider routing for {safe_display_name_escaped}
version: 0.2.0
license: MIT
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field
from open_webui.env import SRC_LOG_LEVELS

OWUI_OPENROUTER_PIPE_MARKER = "{safe_marker_escaped}"
MODEL_SLUG = "{safe_model_slug_escaped}"

# Sentinel value for "no preference" dropdown option
_NO_PREF = "(no preference)"

# Map display names to provider slugs
_PROVIDER_MAP: dict[str, str] = {provider_map_code}

# Map ORDER display values to provider slug lists
_ORDER_MAP: dict[str, list[str]] = {order_map_code}


class Filter:
    toggle = {toggle_value}
{valves_class}{user_valves_class}
    def __init__(self) -> None:
{init_body}

    def inlet(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
        __model__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Inject provider routing preferences into request metadata."""
{inlet_logic}
        return body
'''

    @staticmethod
    def _generate_inlet_logic(visibility: str) -> str:
        """Generate the inlet method logic based on visibility.

        IMPORTANT: User valves are injected by OWUI into __user__["valves"], NOT into
        self.user_valves. The self.user_valves from __init__ only contains defaults.
        We must extract the injected valves from __user__ to get actual user settings.
        """
        # Common provider dict building logic
        logic = '''        provider: dict[str, Any] = {}

        # Determine which valve source to use
        admin_set: set[str] = set()
        user_set: set[str] = set()
'''
        if visibility in ("admin", "both"):
            logic += '''        if hasattr(self, "valves"):
            admin_set = self.valves.model_fields_set
'''
        if visibility in ("user", "both"):
            # Extract user valves from OWUI-injected __user__["valves"]
            logic += '''
        # OWUI injects user valves into __user__["valves"], not self.user_valves
        user_valves = __user__.get("valves") if __user__ else None
        if user_valves is not None:
            user_set = user_valves.model_fields_set
'''

        # Field processing - user overrides admin for 'both'
        logic += '''
        # String fields: user if set and non-empty, else admin if set and non-empty
        def get_str(field: str) -> str:
            val = ""
'''
        if visibility in ("user", "both"):
            logic += '''            if field in user_set and user_valves is not None:
                v = getattr(user_valves, field, "")
                if isinstance(v, str) and v.strip():
                    val = v.strip()
'''
        if visibility in ("admin", "both"):
            logic += '''            if not val and field in admin_set and hasattr(self, "valves"):
                v = getattr(self.valves, field, "")
                if isinstance(v, str) and v.strip():
                    val = v.strip()
'''
        logic += '''            return val

        def get_literal(field: str) -> str:
            """Get Literal field value, returning empty string if _NO_PREF."""
            val = get_str(field)
            return "" if val == _NO_PREF else val

        def get_bool(field: str, api_default: bool = False) -> bool | None:
            """Get boolean value. Returns None if not explicitly set or matches API default."""
'''
        if visibility in ("user", "both"):
            logic += '''            if field in user_set and user_valves is not None:
                val = getattr(user_valves, field, api_default)
                if val != api_default:
                    return val
'''
        if visibility in ("admin", "both"):
            logic += '''            if field in admin_set and hasattr(self, "valves"):
                val = getattr(self.valves, field, api_default)
                if val != api_default:
                    return val
'''
        logic += '''            return None

        def get_float(field: str) -> float:
'''
        if visibility in ("user", "both"):
            logic += '''            if field in user_set and user_valves is not None:
                val = getattr(user_valves, field, 0)
                if isinstance(val, (int, float)) and val > 0:
                    return float(val)
'''
        if visibility in ("admin", "both"):
            logic += '''            if field in admin_set and hasattr(self, "valves"):
                val = getattr(self.valves, field, 0)
                if isinstance(val, (int, float)) and val > 0:
                    return float(val)
'''
        logic += '''            return 0

        # Build provider object
        # ORDER: Map display value to provider slug list using _ORDER_MAP
        order_display = get_literal("ORDER")
        if order_display:
            order_slugs = _ORDER_MAP.get(order_display)
            if order_slugs:
                provider["order"] = order_slugs
            else:
                self.log.warning("ORDER value %r not found in _ORDER_MAP", order_display)

        # ONLY: Map display name to slug using _PROVIDER_MAP
        only_display = get_literal("ONLY")
        if only_display:
            only_slug = _PROVIDER_MAP.get(only_display)
            if only_slug:
                provider["only"] = [only_slug]
            else:
                self.log.warning("ONLY value %r not found in _PROVIDER_MAP", only_display)

        # IGNORE: Map display name to slug using _PROVIDER_MAP
        ignore_display = get_literal("IGNORE")
        if ignore_display:
            ignore_slug = _PROVIDER_MAP.get(ignore_display)
            if ignore_slug:
                provider["ignore"] = [ignore_slug]
            else:
                self.log.warning("IGNORE value %r not found in _PROVIDER_MAP", ignore_display)

        # SORT: Literal values map directly
        sort_val = get_literal("SORT")
        if sort_val:
            provider["sort"] = sort_val

        # QUANTIZATION: Literal dropdown maps directly
        quant_val = get_literal("QUANTIZATION")
        if quant_val:
            provider["quantizations"] = [quant_val]

        # DATA_COLLECTION: Literal values map directly
        data_collection = get_literal("DATA_COLLECTION")
        if data_collection:
            provider["data_collection"] = data_collection

        # Boolean fields (only include if explicitly set and differs from API default)
        allow_fallbacks = get_bool("ALLOW_FALLBACKS", api_default=True)
        if allow_fallbacks is not None:
            provider["allow_fallbacks"] = allow_fallbacks

        require_params = get_bool("REQUIRE_PARAMETERS", api_default=False)
        if require_params is not None:
            provider["require_parameters"] = require_params

        zdr = get_bool("ZDR", api_default=False)
        if zdr is not None:
            provider["zdr"] = zdr

        distillable = get_bool("ENFORCE_DISTILLABLE_TEXT", api_default=False)
        if distillable is not None:
            provider["enforce_distillable_text"] = distillable

        # Numeric fields (only include if > 0)
        min_throughput = get_float("MIN_THROUGHPUT")
        if min_throughput > 0:
            provider["preferred_min_throughput"] = min_throughput

        max_latency = get_float("MAX_LATENCY")
        if max_latency > 0:
            provider["preferred_max_latency"] = max_latency

        # Price limits
        max_price_prompt = get_float("MAX_PRICE_PROMPT")
        max_price_completion = get_float("MAX_PRICE_COMPLETION")
        max_price_image = get_float("MAX_PRICE_IMAGE")
        max_price_audio = get_float("MAX_PRICE_AUDIO")
        max_price_request = get_float("MAX_PRICE_REQUEST")
        if (
            max_price_prompt > 0
            or max_price_completion > 0
            or max_price_image > 0
            or max_price_audio > 0
            or max_price_request > 0
        ):
            provider["max_price"] = {}
            if max_price_prompt > 0:
                provider["max_price"]["prompt"] = max_price_prompt
            if max_price_completion > 0:
                provider["max_price"]["completion"] = max_price_completion
            if max_price_image > 0:
                provider["max_price"]["image"] = max_price_image
            if max_price_audio > 0:
                provider["max_price"]["audio"] = max_price_audio
            if max_price_request > 0:
                provider["max_price"]["request"] = max_price_request

        # Inject into metadata if we have any provider settings
        if provider:
            if __metadata__ is None:
                __metadata__ = {}
            pipe_meta = __metadata__.setdefault("openrouter_pipe", {})
            pipe_meta["provider"] = provider
            self.log.debug("Injected provider routing: %s", provider)
'''
        return logic

    @timed
    def _ensure_provider_routing_filters(
        self,
        admin_models_csv: str,
        user_models_csv: str,
        provider_map: dict[str, dict[str, list[str]]],
        models: list[dict[str, Any]],
        pipe_identifier: str,
    ) -> dict[str, str]:
        """Ensure provider routing filters exist for specified models.

        Creates, updates, or disables filters based on current valve configuration.

        Returns:
            Mapping of model slug -> filter function ID for filters that should be attached.
        """
        try:
            from open_webui.models.functions import Functions, FunctionForm, FunctionMeta
        except Exception:
            return {}

        # Parse model lists
        admin_models = {m.strip() for m in admin_models_csv.split(",") if m.strip()}
        user_models = {m.strip() for m in user_models_csv.split(",") if m.strip()}

        # Compute state hash - skip generation if unchanged but still return existing mappings
        current_hash = self._compute_provider_routing_hash(admin_models_csv, user_models_csv, provider_map)
        hash_unchanged = current_hash == Pipe._provider_routing_state_hash
        if hash_unchanged:
            self.logger.info("Provider routing state unchanged (hash=%s), returning existing filter mappings", current_hash[:8])
        else:
            self.logger.info(
                "Provider routing state changed (new_hash=%s), processing %d admin + %d user model(s)",
                current_hash[:8],
                len(admin_models),
                len(user_models),
            )
            # NOTE: Hash update moved to end of function to prevent race condition
            # (hash was being set before filter work completed)

        # Determine visibility for each model
        all_models = admin_models | user_models
        model_visibility: dict[str, str] = {}
        for slug in all_models:
            if slug in admin_models and slug in user_models:
                model_visibility[slug] = "both"
            elif slug in admin_models:
                model_visibility[slug] = "admin"
            else:
                model_visibility[slug] = "user"

        # Find existing provider routing filters
        try:
            all_filters = Functions.get_functions_by_type("filter", active_only=False)
        except Exception:
            all_filters = []

        existing_filters: dict[str, Any] = {}  # model_slug -> filter object
        for f in all_filters:
            content = getattr(f, "content", "") or ""
            if _PROVIDER_ROUTING_FILTER_MARKER_PREFIX in content:
                # Extract model slug from marker
                for line in content.split("\n"):
                    if _PROVIDER_ROUTING_FILTER_MARKER_PREFIX in line:
                        # Format: OWUI_OPENROUTER_PIPE_MARKER = "openrouter_pipe:provider_routing:slug:v1"
                        try:
                            marker_val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            parts = marker_val.split(":")
                            if len(parts) >= 4 and parts[0] == "openrouter_pipe" and parts[1] == "provider_routing":
                                slug = parts[2]
                                existing_filters[slug] = f
                        except Exception:
                            pass
                        break

        # Track slug -> filter_id mappings for attachment
        slug_to_filter_id: dict[str, str] = {}

        # Check if any filters are missing (deleted by user)
        missing_filters = {slug for slug in all_models if slug not in existing_filters}

        # If hash unchanged AND all filters exist, just return existing mappings
        if hash_unchanged and not missing_filters:
            for slug in all_models:
                existing = existing_filters.get(slug)
                if existing:
                    existing_id = getattr(existing, "id", "")
                    if existing_id:
                        slug_to_filter_id[slug] = existing_id
            self.logger.info(
                "Returning %d existing provider routing filter mappings",
                len(slug_to_filter_id),
            )
            return slug_to_filter_id

        # If filters are missing, we need to create them even if hash unchanged
        if missing_filters:
            self.logger.info(
                "Provider routing filters missing for %d model(s), recreating: %s",
                len(missing_filters),
                ", ".join(sorted(missing_filters)),
            )

        # Create/update filters for requested models
        created = 0
        updated = 0
        for slug, visibility in model_visibility.items():
            model_info = provider_map.get(slug, {})
            providers = model_info.get("providers", [])
            quantizations = model_info.get("quantizations", [])
            # Extract short_name and provider_names with type safety
            raw_short_name = model_info.get("short_name", "")
            short_name: str = raw_short_name if isinstance(raw_short_name, str) else ""
            raw_prov_names = model_info.get("provider_names", {})
            prov_names: dict[str, str] = raw_prov_names if isinstance(raw_prov_names, dict) else {}

            if not providers:
                self.logger.warning("Skipping filter for %s: no providers found in catalog (check slug spelling)", slug)
                continue

            safe_id = self._sanitize_model_for_filter_id(slug)
            filter_id = f"{_PROVIDER_ROUTING_FILTER_ID_PREFIX}{safe_id}"

            desired_source = self._render_provider_routing_filter_source(
                slug, providers, quantizations, visibility,
                short_name=short_name,
                provider_names=prov_names,
            ).strip() + "\n"

            # SECURITY: Validate generated source before storing (defense-in-depth)
            is_valid, validation_error = self._validate_filter_source(desired_source)
            if not is_valid:
                self.logger.error(
                    "Generated filter for %s failed syntax validation: %s. Skipping.",
                    slug,
                    validation_error,
                )
                continue  # Skip this filter, don't store invalid code

            # Use short_name for display, fallback to slug
            display_name = short_name if short_name else slug.split("/")[-1]
            # Sanitize for safety
            safe_display = self._validate_provider_name(display_name, slug=slug)
            desired_name = f"Provider: {safe_display}"
            desired_meta = {
                "description": f"Provider routing preferences for {slug}",
                "toggle": visibility != "admin",  # ADMIN-only: not toggleable
                "manifest": {
                    "title": f"Provider Routing: {slug}",
                    "id": filter_id,
                    "version": "0.1.0",
                    "license": "MIT",
                },
            }

            existing = existing_filters.get(slug)
            if existing:
                # Update existing filter
                existing_id = getattr(existing, "id", "")
                existing_content = (getattr(existing, "content", "") or "").strip() + "\n"
                if existing_content != desired_source:
                    Functions.update_function_by_id(
                        existing_id,
                        {
                            "content": desired_source,
                            "name": desired_name,
                            "meta": desired_meta,
                            "is_active": True,
                        },
                    )
                    updated += 1
                    self.logger.info("Updated provider routing filter: %s", existing_id)
                else:
                    # Just ensure it's active
                    Functions.update_function_by_id(existing_id, {"is_active": True})
                # Track for attachment
                if existing_id:
                    slug_to_filter_id[slug] = existing_id
            else:
                # Create new filter
                candidate_id = filter_id
                suffix = 0
                while True:
                    try:
                        existing_func = Functions.get_function_by_id(candidate_id)
                    except Exception:
                        existing_func = None
                    if existing_func is None:
                        break
                    suffix += 1
                    candidate_id = f"{filter_id}_{suffix}"
                    if suffix > 50:
                        self.logger.warning("Could not find unique ID for provider routing filter: %s", slug)
                        break

                if suffix <= 50:
                    meta_obj = FunctionMeta(**desired_meta)
                    form = FunctionForm(
                        id=candidate_id,
                        name=desired_name,
                        content=desired_source,
                        meta=meta_obj,
                    )
                    created_func = Functions.insert_new_function("", "filter", form)
                    if created_func:
                        Functions.update_function_by_id(candidate_id, {"is_active": True, "is_global": False})
                        created += 1
                        self.logger.info("Created provider routing filter: %s", candidate_id)
                        # Track for attachment
                        slug_to_filter_id[slug] = candidate_id

        # Disable filters for models no longer in the lists
        disabled = 0
        for slug, existing in existing_filters.items():
            if slug not in all_models:
                existing_id = getattr(existing, "id", "")
                if existing_id:
                    Functions.update_function_by_id(existing_id, {"is_active": False})
                    disabled += 1
                    self.logger.info("Disabled provider routing filter: %s", existing_id)

        if created or updated or disabled:
            self.logger.info(
                "Provider routing filters: created=%d, updated=%d, disabled=%d (total models=%d)",
                created, updated, disabled, len(all_models),
            )

        # Update hash AFTER all filter work completes (prevents race condition where
        # hash is set but work failed, causing next call to incorrectly skip)
        if created > 0 or updated > 0 or disabled > 0 or len(all_models) == 0:
            Pipe._provider_routing_state_hash = current_hash
            self.logger.debug("Provider routing state hash updated: %s", current_hash[:8])
        else:
            self.logger.warning(
                "No filter changes made; state hash NOT updated (will retry on next call)"
            )

        self.logger.info(
            "Returning %d provider routing filter mappings for attachment: %r",
            len(slug_to_filter_id),
            slug_to_filter_id,
        )
        return slug_to_filter_id

    # =============================================================================
    # ENTRY POINTS
    # =============================================================================

    @timed
    async def pipes(self):
        """Return the list of models exposed to Open WebUI."""
        self._maybe_start_startup_checks()
        self._maybe_start_redis()
        self._maybe_start_cleanup()
        session = self._create_http_session()
        refresh_error: Exception | None = None
        api_key_value, api_key_error = self._resolve_openrouter_api_key(self.valves)
        if api_key_error:
            refresh_error = ValueError(api_key_error)
        try:
            if api_key_value and not api_key_error:
                await OpenRouterModelRegistry.ensure_loaded(
                    session,
                    base_url=self.valves.BASE_URL,
                    api_key=api_key_value,
                    cache_seconds=self.valves.MODEL_CATALOG_REFRESH_SECONDS,
                    logger=self.logger,
                    http_referer=_select_openrouter_http_referer(self.valves),
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

        if self.valves.AUTO_INSTALL_ORS_FILTER:
            try:
                await run_in_threadpool(self._ensure_ors_filter_function_id)
            except Exception as exc:
                self.logger.debug("AUTO_INSTALL_ORS_FILTER failed: %s", exc)
        if self.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER:
            try:
                await run_in_threadpool(self._ensure_direct_uploads_filter_function_id)
            except Exception as exc:
                self.logger.debug("AUTO_INSTALL_DIRECT_UPLOADS_FILTER failed: %s", exc)

        selected_models = self._select_models(self.valves.MODEL_ID, available_models)
        selected_models = self._apply_model_filters(selected_models, self.valves)
        selected_models = self._expand_variant_models(selected_models, self.valves)

        # Provider routing filter creation (immediate, like ORS and Direct Uploads)
        admin_routing = (self.valves.ADMIN_PROVIDER_ROUTING_MODELS or "").strip()
        user_routing = (self.valves.USER_PROVIDER_ROUTING_MODELS or "").strip()
        if admin_routing or user_routing:
            try:
                catalog_mgr = self._ensure_catalog_manager()
                provider_map = catalog_mgr.get_cached_provider_map()
                if provider_map:
                    await run_in_threadpool(
                        self._ensure_provider_routing_filters,
                        admin_routing,
                        user_routing,
                        provider_map,
                        selected_models,
                        self.id,
                    )
            except Exception as exc:
                self.logger.debug("Provider routing filter creation failed: %s", exc)

        self._maybe_schedule_model_metadata_sync(
            selected_models,
            pipe_identifier=self.id,
        )

        # Return simple id/name list - OWUI's get_function_models() only reads these fields
        return [{"id": m["id"], "name": m["name"]} for m in selected_models]

    @timed
    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __request__: Request | None,
        __event_emitter__: EventEmitter | None,
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        __metadata__: dict[str, Any],
        __tools__: list[dict[str, Any]] | dict[str, Any] | None,
        __task__: Any = None,
        __task_body__: Any = None,
    ) -> AsyncGenerator[dict[str, Any] | str, None] | dict[str, Any] | str | None | JSONResponse:
        """Entry point that enqueues work and awaits the isolated job result."""
        # Set up timing context at the very start to capture full request flow
        from .core.timing_logger import set_timing_context, timing_mark, ensure_timing_file_configured
        _early_request_id = secrets.token_hex(8)
        # Ensure timing file is configured when enabled (handles runtime valve changes)
        if self.valves.ENABLE_TIMING_LOG:
            ensure_timing_file_configured(self.valves.TIMING_LOG_FILE)
        set_timing_context(_early_request_id, self.valves.ENABLE_TIMING_LOG)
        timing_mark("pipe_entry")

        self._maybe_start_log_worker()
        timing_mark("after_log_worker")
        self._maybe_start_startup_checks()
        timing_mark("after_startup_checks")
        self._maybe_start_redis()
        timing_mark("after_redis")
        self._maybe_start_cleanup()
        timing_mark("after_cleanup")

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

        http_referer_override = (valves.HTTP_REFERER_OVERRIDE or "").strip()
        referer_override_invalid = bool(
            http_referer_override
            and not http_referer_override.startswith(("http://", "https://"))
        )
        if referer_override_invalid and not wants_stream:
            await self._emit_notification(
                safe_event_emitter,
                "HTTP_REFERER_OVERRIDE must be a full URL including http(s)://. "
                "Falling back to the default pipe referer.",
                level="warning",
            )

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
        timing_mark("after_concurrency_controls")
        queue = self._request_queue
        if queue is None:
            raise RuntimeError("request queue not initialized")

        loop = asyncio.get_running_loop()
        stream_queue: asyncio.Queue[dict[str, Any] | str | None] | None = None
        future = loop.create_future()
        if wants_stream:
            stream_queue_maxsize = valves.MIDDLEWARE_STREAM_QUEUE_MAXSIZE
            stream_queue = (
                asyncio.Queue(maxsize=stream_queue_maxsize)
                if stream_queue_maxsize > 0
                else asyncio.Queue()
            )
            if referer_override_invalid:
                await stream_queue.put(
                    {
                        "event": {
                            "type": "notification",
                            "data": {
                                "type": "warning",
                                "content": (
                                    "HTTP_REFERER_OVERRIDE must be a full URL including http(s)://. "
                                    "Falling back to the default pipe referer."
                                ),
                            },
                        }
                    }
                )

        job = _PipeJob(
            pipe=self,
            body=body,
            user=__user__,
            request=__request__,
            event_emitter=safe_event_emitter,  # Keep emitter for both streaming and non-streaming
            event_call=__event_call__,
            metadata=__metadata__,
            tools=__tools__,
            task=__task__,
            task_body=__task_body__,
            valves=valves,
            future=future,
            stream_queue=stream_queue,
            request_id=_early_request_id,  # Use same ID as timing context
        )

        timing_mark("before_enqueue_job")
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
            @timed
            async def _stream() -> AsyncGenerator[dict[str, Any] | str, None]:
                try:
                    while True:
                        if future.done() and stream_queue.empty():
                            break
                        if stream_queue.maxsize > 0:
                            try:
                                item = await asyncio.wait_for(stream_queue.get(), timeout=0.25)
                            except asyncio.TimeoutError:
                                continue
                        else:
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

    @timed
    async def _stop_redis(self) -> None:
        """Stop Redis client and cancel related tasks.

        Cancels all Redis background tasks and closes the Redis client connection.
        Any errors during client close are logged but not propagated.
        """
        # Cancel background tasks first
        if self._redis_listener_task and not self._redis_listener_task.done():
            self._redis_listener_task.cancel()
        self._redis_listener_task = None

        if self._redis_flush_task and not self._redis_flush_task.done():
            self._redis_flush_task.cancel()
        self._redis_flush_task = None

        if self._redis_ready_task and not self._redis_ready_task.done():
            self._redis_ready_task.cancel()
        self._redis_ready_task = None

        # Close Redis client and handle errors gracefully
        if self._redis_client:
            try:
                await self._redis_client.close()
            except Exception as e:
                self.logger.debug(f"Failed to close Redis client: {e}")
            finally:
                self._redis_client = None

        # Update state
        self._redis_enabled = False

    @timed
    def shutdown(self) -> None:
        """Public method to shut down background resources."""
        # Delegate DB executor shutdown to artifact store
        executor = getattr(self._artifact_store, "_db_executor", None) if self._artifact_store else None
        if self._artifact_store and executor:
            self._artifact_store._db_executor = None
        if executor:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
            except Exception:
                self.logger.debug(
                    "Failed to shutdown DB executor cleanly",
                    exc_info=self.logger.isEnabledFor(logging.DEBUG),
                )
        self._stop_session_log_workers()

    @timed
    async def _stop_request_worker(self) -> None:
        """Stop this instance's queue worker and drain pending items."""
        worker = self._queue_worker_task
        if worker:
            worker.cancel()
            try:
                worker_loop = worker.get_loop()
            except Exception:  # pragma: no cover - defensive for older asyncio implementations
                worker_loop = None
            if worker_loop is None or worker_loop is asyncio.get_running_loop():
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
            else:
                # The queue worker (and its Queue) belong to a different event loop.
                # This can happen in test runners that create a new loop per test.
                # Do not await across loops; just drop references so a new loop can recreate them.
                self.logger.debug(
                    "Skipping await for request worker bound to a different event loop during close()."
                )
            self._queue_worker_task = None
        self._request_queue = None

    @timed
    async def _stop_log_worker(self) -> None:
        """Stop this instance's log worker and clear the queue."""
        worker = self._log_worker_task
        if worker:
            worker.cancel()
            try:
                worker_loop = worker.get_loop()
            except Exception:  # pragma: no cover - defensive for older asyncio implementations
                worker_loop = None
            if worker_loop is None or worker_loop is asyncio.get_running_loop():
                try:
                    with contextlib.suppress(asyncio.CancelledError):
                        await worker
                except RuntimeError as exc:
                    # Shutdown/reload edge-case: avoid noisy logs when a stale coroutine/task leaks through.
                    if "cannot reuse already awaited coroutine" not in str(exc):
                        raise
                    self.logger.debug("Ignoring log worker shutdown error: %s", exc)
            else:
                self.logger.debug(
                    "Skipping await for log worker bound to a different event loop during close()."
                )
            self._log_worker_task = None
        self._log_queue = None
        self._log_queue_loop = None
        # Import SessionLogger lazily to avoid circular dependency
        try:
            from .core.logging_system import SessionLogger
            SessionLogger.set_log_queue(None)
        except Exception:
            pass

    @timed
    async def close(self):
        """Shutdown background resources (DB executor, queue worker, log worker, Redis)."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.shutdown()
        await self._stop_request_worker()
        await self._stop_log_worker()
        await self._stop_redis()
        if self._http_session:
            with contextlib.suppress(Exception):
                await self._http_session.close()
            self._http_session = None
            if self._multimodal_handler:
                self._multimodal_handler._http_session = None
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

    @timed
    def __del__(self) -> None:
        """Best-effort cleanup hook for garbage collection."""
        if getattr(self, "_closed", False):
            return
        self.shutdown()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            try:
                task = loop.create_task(self.close(), name="openrouter-pipe-close")

                def _consume_background_exception(task: asyncio.Task) -> None:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        task.exception()

                task.add_done_callback(_consume_background_exception)
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

    # Compatibility properties for tests (delegate to subsystems)
    @property
    @timed
    def _db_executor(self):
        """Property for test compatibility - delegates to artifact store."""
        return self._artifact_store._db_executor if self._artifact_store else None

    @_db_executor.setter
    @timed
    def _db_executor(self, value):
        """Property setter for test compatibility - delegates to artifact store."""
        if self._artifact_store:
            self._artifact_store._db_executor = value

    @property
    @timed
    def _encryption_key(self):
        """Property for test compatibility - delegates to artifact store."""
        return self._artifact_store._encryption_key if self._artifact_store else ""

    @_encryption_key.setter
    @timed
    def _encryption_key(self, value):
        """Property setter for test compatibility - delegates to artifact store."""
        if self._artifact_store:
            self._artifact_store._encryption_key = value

    @property
    @timed
    def _fernet(self):
        """Property for test compatibility - delegates to artifact store."""
        return self._artifact_store._fernet if self._artifact_store else None

    @_fernet.setter
    @timed
    def _fernet(self, value):
        """Property setter for test compatibility - delegates to artifact store."""
        if self._artifact_store:
            self._artifact_store._fernet = value

    @property
    @timed
    def _encrypt_all(self):
        """Property for test compatibility - delegates to artifact store."""
        return self._artifact_store._encrypt_all if self._artifact_store else False

    @_encrypt_all.setter
    @timed
    def _encrypt_all(self, value):
        """Property setter for test compatibility - delegates to artifact store."""
        if self._artifact_store:
            self._artifact_store._encrypt_all = value

    @property
    @timed
    def _redis_client(self) -> Optional[_RedisClient]:
        """Property for test compatibility - delegates to artifact store."""
        return self._artifact_store._redis_client if self._artifact_store else None

    @_redis_client.setter
    @timed
    def _redis_client(self, value: Optional[_RedisClient]) -> None:
        """Property setter for test compatibility - delegates to artifact store."""
        if self._artifact_store:
            self._artifact_store._redis_client = value

    @property
    @timed
    def _redis_enabled(self):
        """Property for test compatibility - delegates to artifact store."""
        return self._artifact_store._redis_enabled if self._artifact_store else False

    @_redis_enabled.setter
    @timed
    def _redis_enabled(self, value):
        """Property setter for test compatibility - delegates to artifact store."""
        if self._artifact_store:
            self._artifact_store._redis_enabled = value

    @property
    @timed
    def _compression_enabled(self):
        """Property for test compatibility - delegates to artifact store."""
        return self._artifact_store._compression_enabled if self._artifact_store else False

    @_compression_enabled.setter
    @timed
    def _compression_enabled(self, value):
        """Property setter for test compatibility - delegates to artifact store."""
        if self._artifact_store:
            self._artifact_store._compression_enabled = value

    @property
    @timed
    def _compression_min_bytes(self):
        """Property for test compatibility - delegates to artifact store."""
        return self._artifact_store._compression_min_bytes if self._artifact_store else 0

    @_compression_min_bytes.setter
    @timed
    def _compression_min_bytes(self, value):
        """Property setter for test compatibility - delegates to artifact store."""
        if self._artifact_store:
            self._artifact_store._compression_min_bytes = value

    @property
    @timed
    def _breaker_threshold(self) -> int:
        """Breaker threshold (synchronized with CircuitBreaker and ArtifactStore)."""
        return self._circuit_breaker.threshold

    @_breaker_threshold.setter
    @timed
    def _breaker_threshold(self, value: int) -> None:
        """Set breaker threshold for CircuitBreaker and ArtifactStore."""
        self._circuit_breaker.threshold = value
        if hasattr(self, '_artifact_store') and self._artifact_store:
            self._artifact_store._breaker_threshold = value

    @property
    @timed
    def _breaker_window_seconds(self) -> float:
        """Breaker window seconds (synchronized with CircuitBreaker and ArtifactStore)."""
        return self._circuit_breaker.window_seconds

    @_breaker_window_seconds.setter
    @timed
    def _breaker_window_seconds(self, value: float) -> None:
        """Set breaker window for CircuitBreaker and ArtifactStore."""
        self._circuit_breaker.window_seconds = value
        if hasattr(self, '_artifact_store') and self._artifact_store:
            self._artifact_store._breaker_window_seconds = value

    # =============================================================================
    # DELEGATION METHODS - ArtifactStore (47 methods)
    # =============================================================================

    # 1. Artifact Store Initialization (5 methods)

    @timed
    def _ensure_artifact_store(self, valves: Any, pipe_identifier: Optional[str] = None) -> None:
        """Delegate to ArtifactStore._ensure_artifact_store."""
        self._artifact_store._ensure_artifact_store(valves, pipe_identifier)

    @timed
    def _init_artifact_store(
        self,
        pipe_identifier: Optional[str] = None,
        *,
        table_fragment: Optional[str] = None,
    ) -> None:
        """Delegate to ArtifactStore._init_artifact_store."""
        self._artifact_store._init_artifact_store(pipe_identifier, table_fragment=table_fragment)

    @staticmethod
    @timed
    def _discover_owui_engine_and_schema(owui_db: Any):
        """Delegate to ArtifactStore._discover_owui_engine_and_schema."""
        return ArtifactStore._discover_owui_engine_and_schema(owui_db)

    @staticmethod
    @timed
    def _quote_identifier(identifier: str) -> str:
        """Delegate to ArtifactStore._quote_identifier."""
        return ArtifactStore._quote_identifier(identifier)

    @staticmethod
    @timed
    def _should_warn_event_queue_backlog(
        qsize: int,
        warn_size: int,
        now: float,
        last_warn_ts: float,
        *,
        cooldown_seconds: float = 30.0,
    ) -> bool:
        """Return True when the event queue backlog should log a warning.

        This is a pure helper extracted for testability; behaviour is unchanged.
        """
        return qsize >= warn_size and (now - last_warn_ts) >= cooldown_seconds

    @staticmethod
    @timed
    def _supports_tool_calling(model_norm_id: str) -> bool:
        """Check if model supports tool calling by checking supported parameters."""
        return bool({"tools", "tool_choice"} & set(ModelFamily.supported_parameters(model_norm_id)))

    @staticmethod
    @timed
    def _sum_pricing_values(node: Any) -> tuple[Decimal, int]:
        """Return (sum, count) of numeric values found in a pricing node."""
        total = Decimal(0)
        count = 0

        if isinstance(node, dict):
            keys = set(node.keys())
            if keys & _PRICING_CATEGORY_KEYS:
                for key, value in node.items():
                    if key == "discount":
                        continue
                    child_total, child_count = Pipe._sum_pricing_values(value)
                    total += child_total
                    count += child_count
                return total, count

            found_value_key = False
            for key in _PRICING_VALUE_KEYS:
                if key in node:
                    found_value_key = True
                    child_total, child_count = Pipe._sum_pricing_values(node[key])
                    total += child_total
                    count += child_count
            if found_value_key:
                return total, count

            for key, value in node.items():
                if key == "discount":
                    continue
                child_total, child_count = Pipe._sum_pricing_values(value)
                total += child_total
                count += child_count
            return total, count

        if isinstance(node, (list, tuple, set)):
            for value in node:
                child_total, child_count = Pipe._sum_pricing_values(value)
                total += child_total
                count += child_count
            return total, count

        if isinstance(node, (int, float, Decimal)):
            try:
                return Decimal(str(node)), 1
            except (InvalidOperation, ValueError):
                return Decimal(0), 0

        if isinstance(node, str):
            raw = node.strip()
            if not raw:
                return Decimal(0), 0
            try:
                return Decimal(raw), 1
            except (InvalidOperation, ValueError):
                return Decimal(0), 0

        return Decimal(0), 0

    @staticmethod
    @timed
    def _extract_task_output_text(response: Dict[str, Any]) -> str:
        """Delegate to TaskModelAdapter._extract_task_output_text."""
        return TaskModelAdapter._extract_task_output_text(response)

    @timed
    def _maybe_heal_index_conflict(self, engine, table, exc: Exception) -> bool:
        """Delegate to ArtifactStore._maybe_heal_index_conflict."""
        return self._artifact_store._maybe_heal_index_conflict(engine, table, exc)

    # 2. Encryption/Compression (9 methods)

    @timed
    def _get_fernet(self):
        """Delegate to ArtifactStore._get_fernet."""
        return self._artifact_store._get_fernet()

    @timed
    def _should_encrypt(self, item_type: str) -> bool:
        """Delegate to ArtifactStore._should_encrypt."""
        return self._artifact_store._should_encrypt(item_type)

    @timed
    def _serialize_payload_bytes(self, payload: dict[str, Any]) -> bytes:
        """Delegate to ArtifactStore._serialize_payload_bytes."""
        return self._artifact_store._serialize_payload_bytes(payload)

    @timed
    def _maybe_compress_payload(self, serialized: bytes) -> tuple[bytes, bool]:
        """Delegate to ArtifactStore._maybe_compress_payload."""
        return self._artifact_store._maybe_compress_payload(serialized)

    @timed
    def _encode_payload_bytes(self, payload: dict[str, Any]) -> bytes:
        """Delegate to ArtifactStore._encode_payload_bytes."""
        return self._artifact_store._encode_payload_bytes(payload)

    @timed
    def _decode_payload_bytes(self, payload_bytes: bytes) -> dict[str, Any]:
        """Delegate to ArtifactStore._decode_payload_bytes."""
        return self._artifact_store._decode_payload_bytes(payload_bytes)

    @timed
    def _lz4_decompress(self, data: bytes) -> bytes:
        """Delegate to ArtifactStore._lz4_decompress."""
        return self._artifact_store._lz4_decompress(data)

    @timed
    def _encrypt_payload(self, payload: dict[str, Any]) -> str:
        """Delegate to ArtifactStore._encrypt_payload."""
        return self._artifact_store._encrypt_payload(payload)

    @timed
    def _decrypt_payload(self, ciphertext: str) -> dict[str, Any]:
        """Delegate to ArtifactStore._decrypt_payload."""
        return self._artifact_store._decrypt_payload(ciphertext)

    @timed
    def _encrypt_if_needed(self, item_type: str, payload: dict[str, Any]) -> tuple[Any, bool]:
        """Delegate to ArtifactStore._encrypt_if_needed."""
        return self._artifact_store._encrypt_if_needed(item_type, payload)

    # 3. Database Operations (11 methods)

    @timed
    def _prepare_rows_for_storage(self, rows: List[dict[str, Any]]) -> None:
        """Delegate to ArtifactStore._prepare_rows_for_storage."""
        self._artifact_store._prepare_rows_for_storage(rows)

    @timed
    def _make_db_row(
        self,
        chat_id: Optional[str],
        message_id: Optional[str],
        model_id: str,
        payload: Dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Delegate to ArtifactStore._make_db_row."""
        return self._artifact_store._make_db_row(chat_id, message_id, model_id, payload)

    @timed
    def _db_persist_sync(self, rows: list[dict[str, Any]]) -> list[str]:
        """Delegate to ArtifactStore._db_persist_sync."""
        return self._artifact_store._db_persist_sync(rows)

    @timed
    async def _db_persist(self, rows: list[dict[str, Any]]) -> list[str]:
        """Delegate to ArtifactStore._db_persist."""
        return await self._artifact_store._db_persist(rows)

    @timed
    async def _db_persist_direct(self, rows: list[dict[str, Any]], user_id: str = "") -> list[str]:
        """Delegate to ArtifactStore._db_persist_direct."""
        return await self._artifact_store._db_persist_direct(rows, user_id)

    @timed
    def _is_duplicate_key_error(self, exc: Exception) -> bool:
        """Delegate to ArtifactStore._is_duplicate_key_error."""
        return self._artifact_store._is_duplicate_key_error(exc)

    @timed
    def _try_acquire_lock_sync(self, lock_row: dict[str, Any]) -> bool:
        """Delegate to ArtifactStore._try_acquire_lock_sync."""
        return self._artifact_store._try_acquire_lock_sync(lock_row)

    @timed
    def _db_fetch_sync(
        self,
        chat_id: str,
        message_id: Optional[str],
        item_ids: list[str],
    ) -> dict[str, dict]:
        """Delegate to ArtifactStore._db_fetch_sync."""
        return self._artifact_store._db_fetch_sync(chat_id, message_id, item_ids)

    @timed
    async def _db_fetch(
        self,
        chat_id: Optional[str],
        message_id: Optional[str],
        item_ids: list[str],
    ) -> dict[str, dict]:
        """Delegate to ArtifactStore._db_fetch."""
        return await self._artifact_store._db_fetch(chat_id, message_id, item_ids)

    @timed
    async def _db_fetch_direct(
        self,
        chat_id: str,
        message_id: Optional[str],
        item_ids: list[str],
    ) -> dict[str, dict]:
        """Delegate to ArtifactStore._db_fetch_direct."""
        return await self._artifact_store._db_fetch_direct(chat_id, message_id, item_ids)

    @timed
    def _delete_artifacts_sync(self, artifact_ids: list[str]) -> None:
        """Delegate to ArtifactStore._delete_artifacts_sync."""
        self._artifact_store._delete_artifacts_sync(artifact_ids)

    @timed
    async def _delete_artifacts(self, refs: list[tuple[str, str]]) -> None:
        """Delegate to ArtifactStore._delete_artifacts."""
        await self._artifact_store._delete_artifacts(refs)

    # 4. Redis Cache (8 methods)

    @timed
    async def _redis_pubsub_listener(self) -> None:
        """Delegate to ArtifactStore._redis_pubsub_listener."""
        await self._artifact_store._redis_pubsub_listener()

    @timed
    async def _redis_periodic_flusher(self) -> None:
        """Delegate to ArtifactStore._redis_periodic_flusher."""
        await self._artifact_store._redis_periodic_flusher()

    @timed
    async def _flush_redis_queue(self) -> None:
        """Delegate to ArtifactStore._flush_redis_queue."""
        await self._artifact_store._flush_redis_queue()

    @timed
    def _redis_cache_key(self, chat_id: Optional[str], row_id: Optional[str]) -> Optional[str]:
        """Delegate to ArtifactStore._redis_cache_key."""
        return self._artifact_store._redis_cache_key(chat_id, row_id)

    @timed
    async def _redis_enqueue_rows(self, rows: list[dict[str, Any]]) -> list[str]:
        """Delegate to ArtifactStore._redis_enqueue_rows."""
        return await self._artifact_store._redis_enqueue_rows(rows)

    @timed
    async def _redis_cache_rows(self, rows: list[dict[str, Any]], *, chat_id: Optional[str] = None) -> None:
        """Delegate to ArtifactStore._redis_cache_rows."""
        await self._artifact_store._redis_cache_rows(rows, chat_id=chat_id)

    @timed
    async def _redis_requeue_entries(self, entries: list[str]) -> None:
        """Delegate to ArtifactStore._redis_requeue_entries."""
        await self._artifact_store._redis_requeue_entries(entries)

    @timed
    async def _redis_fetch_rows(
        self,
        chat_id: Optional[str],
        item_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Delegate to ArtifactStore._redis_fetch_rows."""
        return await self._artifact_store._redis_fetch_rows(chat_id, item_ids)

    # 5. Cleanup Workers (3 methods)

    @timed
    async def _artifact_cleanup_worker(self) -> None:
        """Delegate to ArtifactStore._artifact_cleanup_worker."""
        await self._artifact_store._artifact_cleanup_worker()

    @timed
    async def _run_cleanup_once(self) -> None:
        """Delegate to ArtifactStore._run_cleanup_once."""
        await self._artifact_store._run_cleanup_once()

    @timed
    def _cleanup_sync(self, cutoff: Any) -> None:
        """Delegate to ArtifactStore._cleanup_sync."""
        self._artifact_store._cleanup_sync(cutoff)

    # 6. Circuit Breakers (4 methods)

    @timed
    def _db_breaker_allows(self, user_id: str) -> bool:
        """Delegate to ArtifactStore._db_breaker_allows."""
        return self._artifact_store._db_breaker_allows(user_id)

    @timed
    def _record_db_failure(self, user_id: str) -> None:
        """Delegate to ArtifactStore._record_db_failure."""
        self._artifact_store._record_db_failure(user_id)

    @timed
    def _reset_db_failure(self, user_id: str) -> None:
        """Delegate to ArtifactStore._reset_db_failure."""
        self._artifact_store._reset_db_failure(user_id)

    # =============================================================================
    # DELEGATION METHODS - MultimodalHandler (20 methods)
    # =============================================================================

    # 1. File Operations (7 methods)

    @timed
    async def _get_file_by_id(self, file_id: str):
        """Delegate to MultimodalHandler._get_file_by_id."""
        if not self._multimodal_handler:
            return None
        return await self._multimodal_handler._get_file_by_id(file_id)

    @timed
    def _infer_file_mime_type(self, file_obj: Any) -> str:
        """Delegate to MultimodalHandler._infer_file_mime_type."""
        if not self._multimodal_handler:
            return "application/octet-stream"
        return self._multimodal_handler._infer_file_mime_type(file_obj)

    @timed
    async def _inline_owui_file_id(
        self,
        file_id: str,
        *,
        chunk_size: int,
        max_bytes: int,
    ) -> Optional[str]:
        """Delegate to MultimodalHandler._inline_owui_file_id."""
        if not self._multimodal_handler:
            return None
        return await self._multimodal_handler._inline_owui_file_id(
            file_id,
            chunk_size=chunk_size,
            max_bytes=max_bytes,
        )

    @timed
    async def _inline_internal_file_url(
        self,
        url: str,
        *,
        chunk_size: int,
        max_bytes: int,
    ) -> Optional[str]:
        """Delegate to MultimodalHandler._inline_internal_file_url."""
        if not self._multimodal_handler:
            return None
        return await self._multimodal_handler._inline_internal_file_url(
            url,
            chunk_size=chunk_size,
            max_bytes=max_bytes,
        )

    @timed
    async def _inline_internal_responses_input_files_inplace(
        self,
        request_body: dict[str, Any],
        *,
        chunk_size: int,
        max_bytes: int,
    ) -> None:
        """Delegate to MultimodalHandler._inline_internal_responses_input_files_inplace."""
        if not self._multimodal_handler:
            return
        await self._multimodal_handler._inline_internal_responses_input_files_inplace(
            request_body,
            chunk_size=chunk_size,
            max_bytes=max_bytes,
        )

    @timed
    async def _read_file_record_base64(
        self,
        file_obj: Any,
        chunk_size: int,
        max_bytes: int,
    ) -> Optional[str]:
        """Delegate to MultimodalHandler._read_file_record_base64."""
        if not self._multimodal_handler:
            return None
        return await self._multimodal_handler._read_file_record_base64(
            file_obj,
            chunk_size,
            max_bytes,
        )

    @timed
    async def _encode_file_path_base64(
        self,
        path: Any,
        chunk_size: int,
        max_bytes: int,
    ) -> str:
        """Delegate to MultimodalHandler._encode_file_path_base64."""
        if not self._multimodal_handler:
            raise RuntimeError("MultimodalHandler not initialized")
        return await self._multimodal_handler._encode_file_path_base64(
            path,
            chunk_size,
            max_bytes,
        )

    # 2. Upload to OWUI Storage (4 methods)

    @timed
    async def _upload_to_owui_storage(
        self,
        request: Any,
        user: Any,
        file_data: bytes,
        filename: str,
        mime_type: str,
        chat_id: Optional[str] = None,
        message_id: Optional[str] = None,
        owui_user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Delegate to MultimodalHandler._upload_to_owui_storage."""
        if not self._multimodal_handler:
            return None
        return await self._multimodal_handler._upload_to_owui_storage(
            request,
            user,
            file_data,
            filename,
            mime_type,
            chat_id,
            message_id,
            owui_user_id,
        )

    @timed
    def _try_link_file_to_chat(
        self,
        *,
        chat_id: Optional[str],
        message_id: Optional[str],
        file_id: str,
        user_id: Optional[str],
    ) -> bool:
        """Delegate to MultimodalHandler._try_link_file_to_chat."""
        if not self._multimodal_handler:
            return False
        return self._multimodal_handler._try_link_file_to_chat(
            chat_id=chat_id,
            message_id=message_id,
            file_id=file_id,
            user_id=user_id,
        )

    @timed
    async def _resolve_storage_context(
        self,
        request: Optional[Any],
        user_obj: Optional[Any],
    ) -> tuple[Optional[Any], Optional[Any]]:
        """Delegate to MultimodalHandler._resolve_storage_context."""
        if not self._multimodal_handler:
            return None, None
        return await self._multimodal_handler._resolve_storage_context(request, user_obj)

    @timed
    async def _ensure_storage_user(self) -> Optional[Any]:
        """Delegate to MultimodalHandler._ensure_storage_user."""
        if not self._multimodal_handler:
            return None
        return await self._multimodal_handler._ensure_storage_user()

    # 3. Remote Downloads (5 methods)

    @timed
    async def _download_remote_url(
        self,
        url: str,
        timeout_seconds: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Delegate to MultimodalHandler._download_remote_url."""
        if not self._multimodal_handler:
            return None
        return await self._multimodal_handler._download_remote_url(url, timeout_seconds)

    @timed
    async def _is_safe_url(self, url: str) -> bool:
        """Delegate to MultimodalHandler._is_safe_url."""
        if not self._multimodal_handler:
            return False
        return await self._multimodal_handler._is_safe_url(url)

    @timed
    def _is_insecure_http_allowed(self, url: str) -> bool:
        """Delegate to MultimodalHandler._is_insecure_http_allowed."""
        if not self._multimodal_handler:
            return False
        return self._multimodal_handler._is_insecure_http_allowed(url)

    @timed
    def _is_safe_url_blocking(self, url: str) -> bool:
        """Delegate to MultimodalHandler._is_safe_url_blocking."""
        if not self._multimodal_handler:
            return False
        return self._multimodal_handler._is_safe_url_blocking(url)

    @timed
    def _is_youtube_url(self, url: Optional[str]) -> bool:
        """Delegate to MultimodalHandler._is_youtube_url."""
        if not self._multimodal_handler:
            return False
        return self._multimodal_handler._is_youtube_url(url)

    @timed
    def _get_effective_remote_file_limit_mb(self) -> int:
        """Delegate to MultimodalHandler._get_effective_remote_file_limit_mb."""
        if not self._multimodal_handler:
            return 50  # Default
        return self._multimodal_handler._get_effective_remote_file_limit_mb()

    # 4. Image Processing (2 methods)

    @timed
    async def _fetch_image_as_data_url(
        self,
        session: Any,
        url: str,
    ) -> Optional[str]:
        """Delegate to MultimodalHandler._fetch_image_as_data_url."""
        if not self._multimodal_handler:
            return None
        return await self._multimodal_handler._fetch_image_as_data_url(session, url)

    @timed
    async def _fetch_maker_profile_image_url(
        self,
        session: Any,
        maker_id: str,
    ) -> Optional[str]:
        """Delegate to MultimodalHandler._fetch_maker_profile_image_url."""
        if not self._multimodal_handler:
            return None
        return await self._multimodal_handler._fetch_maker_profile_image_url(session, maker_id)

    # 5. Data URL Handling (2 methods)

    @timed
    def _validate_base64_size(self, b64_data: str) -> bool:
        """Delegate to MultimodalHandler._validate_base64_size."""
        if not self._multimodal_handler:
            return False
        return self._multimodal_handler._validate_base64_size(b64_data)

    @timed
    def _parse_data_url(self, data_url: str) -> Optional[Dict[str, Any]]:
        """Delegate to MultimodalHandler._parse_data_url."""
        if not self._multimodal_handler:
            return None
        return self._multimodal_handler._parse_data_url(data_url)

    # =============================================================================
    # DELEGATION METHODS - StreamingHandler (6 methods)
    # =============================================================================

    @timed
    async def _run_streaming_loop(self, *args, **kwargs):
        """Delegate to StreamingHandler._run_streaming_loop."""
        if not self._streaming_handler:
            raise RuntimeError("StreamingHandler not initialized")
        return await self._streaming_handler._run_streaming_loop(*args, **kwargs)

    @timed
    async def _run_nonstreaming_loop(self, *args, **kwargs):
        """Delegate to StreamingHandler._run_nonstreaming_loop."""
        if not self._streaming_handler:
            raise RuntimeError("StreamingHandler not initialized")
        return await self._streaming_handler._run_nonstreaming_loop(*args, **kwargs)

    @timed
    async def _cleanup_replayed_reasoning(self, body: Any, valves: Any) -> None:
        """Delegate to StreamingHandler._cleanup_replayed_reasoning."""
        if not self._streaming_handler:
            return
        await self._streaming_handler._cleanup_replayed_reasoning(body, valves)

    @timed
    def _select_llm_endpoint(self, model_id: str, *, valves: Any):
        """Delegate to StreamingHandler._select_llm_endpoint."""
        if not self._streaming_handler:
            return "responses"
        return self._streaming_handler._select_llm_endpoint(model_id, valves=valves)

    @timed
    def _select_llm_endpoint_with_forced(self, model_id: str, *, valves: Any):
        """Delegate to StreamingHandler._select_llm_endpoint_with_forced."""
        if not self._streaming_handler:
            return "responses", False
        return self._streaming_handler._select_llm_endpoint_with_forced(model_id, valves=valves)

    @staticmethod
    @timed
    def _looks_like_responses_unsupported(exc: Exception) -> bool:
        """Delegate to StreamingHandler._looks_like_responses_unsupported."""
        return StreamingHandler._looks_like_responses_unsupported(exc)

    # =============================================================================
    # DELEGATION METHODS - EventEmitterHandler (11 methods)
    # =============================================================================

    @timed
    async def _emit_status(
        self,
        event_emitter: Optional[EventEmitter],
        message: str,
        done: bool = False,
    ):
        """Delegate to EventEmitterHandler._emit_status."""
        if not self._event_emitter_handler:
            return
        await self._event_emitter_handler._emit_status(event_emitter, message, done)

    @timed
    async def _emit_error(
        self,
        event_emitter: Optional[EventEmitter],
        error_obj: Exception | str,
        *,
        show_error_message: bool = True,
        show_error_log_citation: bool = False,
        done: bool = False,
    ):
        """Delegate to ErrorFormatter._emit_error."""
        return await self._ensure_error_formatter()._emit_error(
            event_emitter,
            error_obj,
            show_error_message=show_error_message,
            show_error_log_citation=show_error_log_citation,
            done=done,
        )

    @timed
    async def _emit_templated_error(
        self,
        event_emitter: Optional[EventEmitter],
        *,
        template: str,
        variables: dict[str, Any],
        log_message: str,
        log_level: int = logging.ERROR,
    ):
        """Delegate to ErrorFormatter._emit_templated_error."""
        return await self._ensure_error_formatter()._emit_templated_error(
            event_emitter,
            template=template,
            variables=variables,
            log_message=log_message,
            log_level=log_level,
        )

    @timed
    def _build_error_context(self) -> tuple[str, dict[str, Any]]:
        """Delegate to ErrorFormatter._build_error_context."""
        return self._ensure_error_formatter()._build_error_context()

    @timed
    def _select_openrouter_template(self, status: Optional[int]) -> str:
        """Delegate to ErrorFormatter._select_openrouter_template."""
        return self._ensure_error_formatter()._select_openrouter_template(status)

    @timed
    def _build_streaming_openrouter_error(
        self,
        event: dict[str, Any],
        *,
        requested_model: Optional[str],
    ) -> OpenRouterAPIError:
        """Delegate to ErrorFormatter._build_streaming_openrouter_error."""
        return self._ensure_error_formatter()._build_streaming_openrouter_error(
            event,
            requested_model=requested_model,
        )

    @timed
    async def _emit_citation(
        self,
        event_emitter: Optional[EventEmitter],
        citation: Dict[str, Any],
    ):
        """Delegate to EventEmitterHandler._emit_citation."""
        if not self._event_emitter_handler:
            return
        await self._event_emitter_handler._emit_citation(event_emitter, citation)

    @timed
    async def _emit_files(
        self,
        event_emitter: Optional[EventEmitter],
        files: list[dict[str, Any]],
    ):
        """Emit extracted files from tool results to the UI.

        Files are typically images/audio from MCP tools or OpenAPI responses.
        Delegates to EventEmitterHandler._emit_files.
        """
        if not self._event_emitter_handler:
            return
        try:
            await self._event_emitter_handler._emit_files(event_emitter, files)
        except Exception as exc:
            self.logger.debug("Failed to emit files: %s", exc)

    @timed
    async def _emit_embeds(
        self,
        event_emitter: Optional[EventEmitter],
        embeds: list[str],
    ):
        """Emit embedded HTML content from tool results to the UI.

        Embeds are HTML snippets from tools returning HTMLResponse.
        Delegates to EventEmitterHandler._emit_embeds.
        """
        if not self._event_emitter_handler:
            return
        try:
            await self._event_emitter_handler._emit_embeds(event_emitter, embeds)
        except Exception as exc:
            self.logger.debug("Failed to emit embeds: %s", exc)

    @timed
    async def _emit_completion(
        self,
        event_emitter: Optional[EventEmitter],
        *,
        content: str | None = "",
        title: str | None = None,
        usage: dict[str, Any] | None = None,
        done: bool = True,
    ):
        """Delegate to EventEmitterHandler._emit_completion."""
        if not self._event_emitter_handler:
            return
        await self._event_emitter_handler._emit_completion(
            event_emitter,
            content=content,
            title=title,
            usage=usage,
            done=done,
        )

    @timed
    async def _emit_notification(
        self,
        event_emitter: Optional[EventEmitter],
        content: str,
        *,
        level: Literal["info", "success", "warning", "error"] = "info",
    ):
        """Delegate to EventEmitterHandler._emit_notification."""
        if not self._event_emitter_handler:
            return
        await self._event_emitter_handler._emit_notification(
            event_emitter,
            content,
            level=level,
        )

    @timed
    def _wrap_safe_event_emitter(self, emitter: Optional[EventEmitter]):
        """Delegate to EventEmitterHandler._wrap_safe_event_emitter."""
        if not self._event_emitter_handler:
            return emitter
        return self._event_emitter_handler._wrap_safe_event_emitter(emitter)

    @timed
    def _try_put_middleware_stream_nowait(
        self,
        stream_queue: Any,
        item: Any,
    ):
        """Delegate to EventEmitterHandler._try_put_middleware_stream_nowait."""
        if not self._event_emitter_handler:
            return
        self._event_emitter_handler._try_put_middleware_stream_nowait(stream_queue, item)

    @timed
    async def _put_middleware_stream_item(
        self,
        job: Any,
        stream_queue: Any,
        item: Any,
    ):
        """Delegate to EventEmitterHandler._put_middleware_stream_item."""
        if not self._event_emitter_handler:
            return
        await self._event_emitter_handler._put_middleware_stream_item(job, stream_queue, item)

    @timed
    def _make_middleware_stream_emitter(
        self,
        job: Any,
        stream_queue: Any,
    ):
        """Delegate to EventEmitterHandler._make_middleware_stream_emitter."""
        if not self._event_emitter_handler:
            return None
        return self._event_emitter_handler._make_middleware_stream_emitter(job, stream_queue)

    # =============================================================================
    # INTERNAL CALLBACKS (for subsystem use)
    # =============================================================================

    @timed
    async def _emit_status_callback(self, emitter, message: str, done: bool = False):
        """Internal callback for subsystems to emit status."""
        if self._event_emitter_handler:
            await self._event_emitter_handler._emit_status(emitter, message, done)

    @timed
    async def _emit_notification_callback(
        self,
        emitter,
        content: str,
        *,
        level: Literal["info", "success", "warning", "error"] = "info",
    ):
        """Internal callback for subsystems to emit notifications."""
        if self._event_emitter_handler:
            await self._event_emitter_handler._emit_notification(
                emitter,
                content,
                level=level,
            )

    # =============================================================================
    # ORCHESTRATION METHODS
    # =============================================================================

    # ======================================================================
    # _maybe_start_session_log_workers (69 lines)
    # ======================================================================

    @timed
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

        @timed
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

        @timed
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

    # ======================================================================
    # _write_session_log_archive (191 lines)
    # ======================================================================

    @timed
    def _write_session_log_archive(self, job: _SessionLogArchiveJob) -> None:
        """Delegate to logging_system.write_session_log_archive."""
        from .core.logging_system import write_session_log_archive
        return write_session_log_archive(job)


    # ======================================================================
    # _enqueue_session_log_archive (76 lines)
    # ======================================================================

    @timed
    def _enqueue_session_log_archive(
        self,
        valves: "Pipe.Valves",
        *,
        user_id: str,
        session_id: str,
        chat_id: str,
        message_id: str,
        request_id: str,
        log_events: list[dict[str, Any]],
    ) -> None:
        """Queue the current request's session logs for encrypted zip persistence."""
        if not valves.SESSION_LOG_STORE_ENABLED:
            return
        if not (user_id and session_id and chat_id and message_id):
            return
        if not log_events:
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
            log_format=valves.SESSION_LOG_FORMAT,
            log_events=log_events,
        )
        try:
            self._session_log_queue.put_nowait(job)
        except Exception:
            self.logger.debug("Failed to enqueue session log archive job", exc_info=True)

    # ======================================================================
    # _persist_session_log_segment_to_db (99 lines)
    # ======================================================================

    @timed
    def _resolve_session_log_archive_settings(
        self,
        valves: "Pipe.Valves",
    ) -> tuple[str, bytes, str, int | None] | None:
        """Resolve the session log archive settings required for eventual zip writing."""
        if not valves.SESSION_LOG_STORE_ENABLED:
            return None
        if pyzipper is None:
            if not self._session_log_warning_emitted:
                self.logger.warning(
                    "Session log storage is enabled but the 'pyzipper' package is not available; skipping persistence."
                )
                self._session_log_warning_emitted = True
            return None

        base_dir = valves.SESSION_LOG_DIR.strip()
        if not base_dir:
            if not self._session_log_warning_emitted:
                self.logger.warning(
                    "Session log storage is enabled but SESSION_LOG_DIR is empty; skipping persistence."
                )
                self._session_log_warning_emitted = True
            return None

        decrypted = EncryptedStr.decrypt(valves.SESSION_LOG_ZIP_PASSWORD)
        password = (decrypted or "").strip()
        if not password:
            if not self._session_log_warning_emitted:
                self.logger.warning(
                    "Session log storage is enabled but SESSION_LOG_ZIP_PASSWORD is not configured; skipping persistence."
                )
                self._session_log_warning_emitted = True
            return None

        zip_compression = valves.SESSION_LOG_ZIP_COMPRESSION
        zip_compresslevel = valves.SESSION_LOG_ZIP_COMPRESSLEVEL
        if zip_compression in {"stored", "lzma"}:
            zip_compresslevel = None

        with contextlib.suppress(Exception):
            with self._session_log_lock:
                self._session_log_cleanup_interval_seconds = valves.SESSION_LOG_CLEANUP_INTERVAL_SECONDS
                self._session_log_retention_days = valves.SESSION_LOG_RETENTION_DAYS
                self._session_log_dirs.add(base_dir)

        return base_dir, password.encode("utf-8"), zip_compression, zip_compresslevel

    @timed
    async def _persist_session_log_segment_to_db(
        self,
        valves: "Pipe.Valves",
        *,
        user_id: str,
        session_id: str,
        chat_id: str,
        message_id: str,
        request_id: str,
        log_events: list[dict[str, Any]],
        terminal: bool,
        status: str,
        reason: str = "",
        pipe_identifier: str | None = None,
    ) -> None:
        """Persist one invocation's session log events into the DB for later assembly.

        The assembler thread merges all segments for a (chat_id, message_id) into a
        single `<SESSION_LOG_DIR>/<user_id>/<chat_id>/<message_id>.zip`.
        """
        if not getattr(valves, "SESSION_LOG_STORE_ENABLED", False):
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Session log segment skipped (SESSION_LOG_STORE_ENABLED=false): chat_id=%s message_id=%s request_id=%s",
                    chat_id,
                    message_id,
                    request_id,
                )
            return
        if not (user_id and chat_id and message_id and request_id):
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Session log segment skipped (missing ids): user_id=%s chat_id=%s message_id=%s request_id=%s",
                    bool(user_id),
                    bool(chat_id),
                    bool(message_id),
                    bool(request_id),
                )
            return
        if not log_events:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Session log segment skipped (no events): chat_id=%s message_id=%s request_id=%s",
                    chat_id,
                    message_id,
                    request_id,
                )
            return
        archive_settings = self._resolve_session_log_archive_settings(valves)
        if archive_settings is None:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Session log segment skipped (archive settings unavailable): chat_id=%s message_id=%s request_id=%s",
                    chat_id,
                    message_id,
                    request_id,
                )
            return

        # Ensure ArtifactStore is initialized so multi-worker assemblers can coordinate.
        with contextlib.suppress(Exception):
            self._ensure_artifact_store(valves, pipe_identifier=pipe_identifier)

        item_type = "session_log_segment_terminal" if terminal else "session_log_segment"
        payload: dict[str, Any] = {
            "type": item_type,
            "status": str(status or ""),
            "reason": str(reason or ""),
            "user_id": str(user_id or ""),
            "session_id": str(session_id or ""),
            "chat_id": str(chat_id or ""),
            "message_id": str(message_id or ""),
            "request_id": str(request_id or ""),
            "created_at": time.time(),
            "log_format": str(getattr(valves, "SESSION_LOG_FORMAT", "") or ""),
            "events": log_events,
        }
        if pipe_identifier:
            payload["pipe_id"] = str(pipe_identifier)

        row: dict[str, Any] = {
            "id": generate_item_id(),
            "chat_id": chat_id,
            "message_id": message_id,
            "model_id": None,
            "item_type": item_type,
            "payload": payload,
        }

        try:
            # Best-effort: background workers may not be long-lived in some OWUI deployment
            # modes, so we also attempt to assemble terminal bundles inline (below).
            self._maybe_start_session_log_workers()
            self._maybe_start_session_log_assembler_worker()
            persisted = await self._db_persist([row])
            if not persisted:
                # Survivability guarantee: if DB staging isn't available (or breaker blocks writes),
                # fall back to direct zip persistence so operators still get logs.
                self.logger.warning(
                    "Session log DB staging returned no ids; falling back to direct zip write (chat_id=%s message_id=%s request_id=%s).",
                    chat_id,
                    message_id,
                    request_id,
                )
                from .core.logging_system import _SessionLogArchiveJob, write_session_log_archive

                base_dir, zip_password, zip_compression, zip_compresslevel = archive_settings
                fallback_message_id = f"{message_id}.{request_id}"
                write_session_log_archive(
                    _SessionLogArchiveJob(
                        base_dir=base_dir,
                        zip_password=zip_password,
                        zip_compression=zip_compression,
                        zip_compresslevel=zip_compresslevel,
                        user_id=user_id,
                        session_id=session_id,
                        chat_id=chat_id,
                        message_id=fallback_message_id,
                        request_id=request_id,
                        created_at=time.time(),
                        log_format=str(getattr(valves, "SESSION_LOG_FORMAT", "jsonl") or "jsonl"),
                        log_events=log_events,
                    )
                )
            # Note: Inline assembly removed - the background worker handles all assembly
            # with archive merging to prevent data loss from multiple invocations.
        except Exception:
            self.logger.debug(
                "Failed to persist session log segment (chat_id=%s message_id=%s request_id=%s terminal=%s)",
                chat_id,
                message_id,
                request_id,
                terminal,
                exc_info=True,
            )

    # ======================================================================
    # Session Log Assembler Worker (DB -> single message zip)
    # ======================================================================

    @timed
    def _maybe_start_session_log_assembler_worker(self) -> None:
        """Start the DB-backed session log assembler thread (multi-worker safe)."""
        if self._session_log_assembler_thread and self._session_log_assembler_thread.is_alive():
            return
        if self._session_log_stop_event is None:
            self._session_log_stop_event = threading.Event()

        @timed
        def _wait(stop_event: threading.Event, seconds: float) -> bool:
            try:
                return stop_event.wait(timeout=max(0.0, float(seconds)))
            except Exception:
                time.sleep(max(0.0, float(seconds)))
                return stop_event.is_set()

        @timed
        def _assembler_loop() -> None:
            stop_event = self._session_log_stop_event
            if stop_event is None:
                return

            # Desynchronize workers so multiple UVicorn processes don't spike the DB at once.
            jitter = 0.0
            with contextlib.suppress(Exception):
                jitter = float(getattr(self.valves, "SESSION_LOG_ASSEMBLER_JITTER_SECONDS", 0) or 0)
            jitter = max(0.0, jitter)
            if jitter:
                initial = random.uniform(0.0, jitter)
                if _wait(stop_event, initial):
                    return

            while True:
                if stop_event.is_set():
                    break
                try:
                    self._run_session_log_assembler_once()
                except Exception:
                    self.logger.debug("Session log assembler failed", exc_info=True)
                interval = 15.0
                extra = 0.0
                with contextlib.suppress(Exception):
                    interval = float(getattr(self.valves, "SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS", 15) or 15)
                with contextlib.suppress(Exception):
                    extra = float(getattr(self.valves, "SESSION_LOG_ASSEMBLER_JITTER_SECONDS", 0) or 0)
                interval = max(1.0, interval)
                extra = max(0.0, extra)
                delay = interval + (random.uniform(0.0, extra) if extra else 0.0)
                if _wait(stop_event, delay):
                    break

        self._session_log_assembler_thread = threading.Thread(
            target=_assembler_loop,
            name="openrouter-session-log-assembler",
            daemon=True,
        )
        self._session_log_assembler_thread.start()

    @timed
    def _session_log_db_handles(self) -> tuple[Any | None, Any | None]:
        """Return (model, session_factory) for direct DB queries (best effort)."""
        model = getattr(self._artifact_store, "_item_model", None)
        session_factory = getattr(self._artifact_store, "_session_factory", None)
        return model, session_factory

    @timed
    def _run_session_log_assembler_once(self) -> None:
        """One assembler tick: cleanup stale locks, assemble terminal + stale bundles."""
        if not getattr(self.valves, "SESSION_LOG_STORE_ENABLED", False):
            return
        model, session_factory = self._session_log_db_handles()
        if not model or not session_factory:
            return
        probe = None
        with contextlib.suppress(Exception):
            probe = session_factory()  # type: ignore[call-arg]
        if probe is None or not hasattr(probe, "query"):
            return
        with contextlib.suppress(Exception):
            probe.close()

        batch_size = 25
        lock_stale_seconds = 1800.0
        stale_finalize_seconds = 6 * 3600.0
        with contextlib.suppress(Exception):
            batch_size = int(getattr(self.valves, "SESSION_LOG_ASSEMBLER_BATCH_SIZE", 25) or 25)
        with contextlib.suppress(Exception):
            lock_stale_seconds = float(getattr(self.valves, "SESSION_LOG_LOCK_STALE_SECONDS", 1800) or 1800)
        with contextlib.suppress(Exception):
            stale_finalize_seconds = float(getattr(self.valves, "SESSION_LOG_STALE_FINALIZE_SECONDS", 6 * 3600) or 6 * 3600)
        batch_size = max(1, min(500, batch_size))
        lock_stale_seconds = max(60.0, lock_stale_seconds)
        stale_finalize_seconds = max(300.0, stale_finalize_seconds)

        self._cleanup_stale_session_log_locks(model, session_factory, lock_stale_seconds)

        terminals = self._list_session_log_terminal_messages(model, session_factory, limit=batch_size)
        for chat_id, message_id in terminals:
            if self._assemble_and_write_session_log_bundle(chat_id, message_id, terminal=True):
                continue

        stale = self._list_session_log_stale_messages(
            model,
            session_factory,
            stale_finalize_seconds=stale_finalize_seconds,
            limit=batch_size,
        )
        for chat_id, message_id in stale:
            self._assemble_and_write_session_log_bundle(chat_id, message_id, terminal=False, stale_finalize_seconds=stale_finalize_seconds)

    @timed
    def _cleanup_stale_session_log_locks(
        self,
        model: Any,
        session_factory: Any,
        lock_stale_seconds: float,
    ) -> None:
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=float(lock_stale_seconds))
        session = session_factory()  # type: ignore[call-arg]
        ids: list[str] = []
        try:
            rows = (
                session.query(model.id)  # type: ignore[attr-defined]
                .filter(model.item_type == "session_log_lock")  # type: ignore[attr-defined]
                .filter(model.created_at < cutoff)  # type: ignore[attr-defined]
                .limit(500)
                .all()
            )
            ids = [row[0] for row in rows if row and isinstance(row[0], str)]
        finally:
            with contextlib.suppress(Exception):
                session.close()
        if ids:
            with contextlib.suppress(Exception):
                self._delete_artifacts_sync(ids)

    @timed
    def _list_session_log_terminal_messages(
        self,
        model: Any,
        session_factory: Any,
        *,
        limit: int,
    ) -> list[tuple[str, str]]:
        session = session_factory()  # type: ignore[call-arg]
        try:
            rows = (
                session.query(model.chat_id, model.message_id)  # type: ignore[attr-defined]
                .filter(model.item_type == "session_log_segment_terminal")  # type: ignore[attr-defined]
                .order_by(model.created_at.asc())  # type: ignore[attr-defined]
                .limit(int(limit))
                .all()
            )
            seen: set[tuple[str, str]] = set()
            out: list[tuple[str, str]] = []
            for chat_id, message_id in rows:
                if not (isinstance(chat_id, str) and isinstance(message_id, str)):
                    continue
                key = (chat_id, message_id)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
            return out
        finally:
            session.close()

    @timed
    def _list_session_log_stale_messages(
        self,
        model: Any,
        session_factory: Any,
        *,
        stale_finalize_seconds: float,
        limit: int,
    ) -> list[tuple[str, str]]:
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=float(stale_finalize_seconds))
        session = session_factory()  # type: ignore[call-arg]
        try:
            # Candidates (best effort): any message that has at least one segment.
            candidates = (
                session.query(model.chat_id, model.message_id)  # type: ignore[attr-defined]
                .filter(model.item_type == "session_log_segment")  # type: ignore[attr-defined]
                .distinct()
                .limit(int(limit) * 5)
                .all()
            )
        finally:
            with contextlib.suppress(Exception):
                session.close()

        out: list[tuple[str, str]] = []
        if not candidates:
            return out

        # Filter: no terminal segment, and last activity < cutoff.
        session = session_factory()  # type: ignore[call-arg]
        try:
            for chat_id, message_id in candidates:
                if not (isinstance(chat_id, str) and isinstance(message_id, str)):
                    continue
                exists_terminal = (
                    session.query(model.id)  # type: ignore[attr-defined]
                    .filter(model.chat_id == chat_id)  # type: ignore[attr-defined]
                    .filter(model.message_id == message_id)  # type: ignore[attr-defined]
                    .filter(model.item_type == "session_log_segment_terminal")  # type: ignore[attr-defined]
                    .first()
                )
                if exists_terminal is not None:
                    continue
                last_row = (
                    session.query(model.created_at)  # type: ignore[attr-defined]
                    .filter(model.chat_id == chat_id)  # type: ignore[attr-defined]
                    .filter(model.message_id == message_id)  # type: ignore[attr-defined]
                    .filter(model.item_type.in_(["session_log_segment", "session_log_segment_terminal"]))  # type: ignore[attr-defined]
                    .order_by(model.created_at.desc())  # type: ignore[attr-defined]
                    .limit(1)
                    .first()
                )
                last_created = last_row[0] if last_row else None
                if last_created is None or last_created >= cutoff:
                    continue
                out.append((chat_id, message_id))
                if len(out) >= int(limit):
                    break
        finally:
            session.close()
        return out

    # ------------------------------------------------------------------
    # Session log archive merge helpers
    # ------------------------------------------------------------------

    def _read_session_log_archive_events(
        self,
        zip_path: Path,
        settings: tuple[str, bytes, str, int | None],
    ) -> list[dict[str, Any]]:
        """Read events from an existing session log archive.

        Used during assembly to merge existing events with newly fetched DB events,
        preventing data loss when multiple invocations share the same message_id.
        """
        import pyzipper

        _, zip_password, _, _ = settings
        events: list[dict[str, Any]] = []

        with pyzipper.AESZipFile(zip_path, "r") as zf:
            zf.setpassword(zip_password)
            if "logs.jsonl" in zf.namelist():
                content = zf.read("logs.jsonl").decode("utf-8")
                for line in content.strip().split("\n"):
                    if line.strip():
                        try:
                            evt = json.loads(line)
                            events.append(self._convert_jsonl_to_internal(evt))
                        except Exception:
                            pass  # Skip malformed lines
        return events

    def _convert_jsonl_to_internal(self, evt: dict[str, Any]) -> dict[str, Any]:
        """Convert JSONL archive format back to internal event format.

        JSONL uses 'ts' (ISO timestamp), internal uses 'created' (epoch float).
        """
        internal = dict(evt)
        if "ts" in internal and "created" not in internal:
            try:
                ts_str = internal.pop("ts")
                dt = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                internal["created"] = dt.timestamp()
            except Exception:
                internal["created"] = time.time()
        return internal

    def _dedupe_session_log_events(
        self,
        events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove duplicate events based on content signature.

        Duplicates can occur when the worker writes to the archive but fails to
        delete DB rows, then runs again and re-reads the same events.
        """
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for evt in events:
            # Key: timestamp + request_id + lineno + message hash
            key = "{}:{}:{}:{}".format(
                evt.get("created", 0),
                evt.get("request_id", ""),
                evt.get("lineno", 0),
                hash(str(evt.get("message", ""))),
            )
            if key not in seen:
                seen.add(key)
                unique.append(evt)
        return unique

    @timed
    def _assemble_and_write_session_log_bundle(
        self,
        chat_id: str,
        message_id: str,
        *,
        terminal: bool,
        stale_finalize_seconds: float = 0.0,
        archive_settings: tuple[str, bytes, str, int | None] | None = None,
    ) -> bool:
        """Assemble all segments for one message into a single zip, then delete DB rows."""
        if not (chat_id and message_id):
            return False
        model, session_factory = self._session_log_db_handles()
        if not model or not session_factory:
            return False

        from .core.utils import _stable_crockford_id, _sanitize_path_component

        lock_id = _stable_crockford_id(f"{chat_id}:{message_id}:session_log_lock")
        lock_row: dict[str, Any] = {
            "id": lock_id,
            "chat_id": chat_id,
            "message_id": message_id,
            "model_id": None,
            "item_type": "session_log_lock",
            "payload": {
                "type": "session_log_lock",
                "claimed_at": time.time(),
                "pid": os.getpid(),
                "thread": threading.get_ident(),
            },
        }
        # Use upsert-based lock acquisition (INSERT ON CONFLICT DO NOTHING)
        # to avoid noisy duplicate key errors in multi-worker environments
        if not self._try_acquire_lock_sync(lock_row):
            return False  # Another worker holds the lock

        # Fetch all segment ids for this message (including any terminal markers).
        session = session_factory()  # type: ignore[call-arg]
        ids: list[str] = []
        try:
            rows = (
                session.query(model.id)  # type: ignore[attr-defined]
                .filter(model.chat_id == chat_id)  # type: ignore[attr-defined]
                .filter(model.message_id == message_id)  # type: ignore[attr-defined]
                .filter(model.item_type.in_(["session_log_segment", "session_log_segment_terminal"]))  # type: ignore[attr-defined]
                .order_by(model.created_at.asc())  # type: ignore[attr-defined]
                .all()
            )
            ids = [row[0] for row in rows if row and isinstance(row[0], str)]
        finally:
            with contextlib.suppress(Exception):
                session.close()

        if not ids:
            with contextlib.suppress(Exception):
                self._delete_artifacts_sync([lock_id])
            return False

        payloads = self._db_fetch_sync(chat_id, message_id, ids)
        segments: list[dict[str, Any]] = []
        for item_id in ids:
            payload = payloads.get(item_id)
            if not isinstance(payload, dict):
                continue
            if payload.get("type") in {"session_log_segment", "session_log_segment_terminal"}:
                segments.append(payload)

        if not segments:
            with contextlib.suppress(Exception):
                self._delete_artifacts_sync(ids + [lock_id])
            return False

        resolved_user_id = ""
        resolved_session_id = ""
        preferred_request_id = ""
        merged_events: list[dict[str, Any]] = []

        for seg in segments:
            if not resolved_user_id:
                raw_uid = seg.get("user_id")
                if isinstance(raw_uid, str) and raw_uid.strip():
                    resolved_user_id = raw_uid.strip()
            if not resolved_session_id:
                raw_sid = seg.get("session_id")
                if isinstance(raw_sid, str) and raw_sid.strip():
                    resolved_session_id = raw_sid.strip()
            if seg.get("type") == "session_log_segment_terminal":
                rid = seg.get("request_id")
                if isinstance(rid, str) and rid.strip():
                    preferred_request_id = rid.strip()
            events = seg.get("events")
            if isinstance(events, list):
                for evt in events:
                    if isinstance(evt, dict):
                        merged_events.append(evt)

        if not preferred_request_id:
            for seg in segments:
                rid = seg.get("request_id")
                if isinstance(rid, str) and rid.strip():
                    preferred_request_id = rid.strip()
                    break

        def _event_ts(evt: dict[str, Any]) -> float:
            created = evt.get("created")
            try:
                return float(created) if created is not None else 0.0
            except Exception:
                return 0.0

        merged_events.sort(key=_event_ts)

        # Add a final synthetic marker to make incomplete bundles explicit.
        if not terminal:
            msg = "Session log finalized as incomplete"
            if stale_finalize_seconds:
                msg = f"{msg} (no terminal segment after {int(stale_finalize_seconds)}s)"
            merged_events.append(
                {
                    "created": time.time(),
                    "level": "WARNING",
                    "logger": __name__,
                    "request_id": preferred_request_id or "",
                    "session_id": resolved_session_id or "",
                    "user_id": resolved_user_id or "",
                    "event_type": "pipe",
                    "module": __name__,
                    "func": "_assemble_and_write_session_log_bundle",
                    "lineno": 0,
                    "message": msg,
                }
            )

        settings = archive_settings or self._resolve_session_log_archive_settings(self.valves)
        if settings is None:
            with contextlib.suppress(Exception):
                self._delete_artifacts_sync([lock_id])
            return False
        base_dir, zip_password, zip_compression, zip_compresslevel = settings

        out_dir = Path(base_dir).expanduser() / _sanitize_path_component(resolved_user_id, fallback="user") / _sanitize_path_component(chat_id, fallback="chat")
        out_path = out_dir / f"{_sanitize_path_component(message_id, fallback='message')}.zip"
        before_stat = None
        with contextlib.suppress(Exception):
            before_stat = out_path.stat()

        # Merge with existing archive events if the zip already exists.
        # This prevents data loss when multiple pipe invocations for the same
        # message_id (main response, title generation, tool calls) each persist
        # their own session log segments.
        if out_path.exists():
            try:
                existing_events = self._read_session_log_archive_events(out_path, settings)
                if existing_events:
                    merged_events = existing_events + merged_events
                    merged_events = self._dedupe_session_log_events(merged_events)
                    merged_events.sort(key=_event_ts)
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            "Merged %d existing archive events with %d DB events (chat_id=%s message_id=%s)",
                            len(existing_events),
                            len(merged_events) - len(existing_events),
                            chat_id,
                            message_id,
                        )
            except Exception:
                # If reading fails, proceed with DB events only (safe fallback).
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "Failed to read existing session log archive, proceeding with DB events only (path=%s)",
                        str(out_path),
                        exc_info=True,
                    )

        job = _SessionLogArchiveJob(
            base_dir=base_dir,
            zip_password=zip_password,
            zip_compression=zip_compression,
            zip_compresslevel=zip_compresslevel,
            user_id=resolved_user_id or "user",
            session_id=resolved_session_id or "",
            chat_id=chat_id,
            message_id=message_id,
            request_id=preferred_request_id or "",
            created_at=time.time(),
            log_format=str(getattr(self.valves, "SESSION_LOG_FORMAT", "jsonl") or "jsonl"),
            log_events=merged_events,
        )
        self._write_session_log_archive(job)

        after_stat = None
        with contextlib.suppress(Exception):
            after_stat = out_path.stat()
        wrote = after_stat is not None and (
            before_stat is None
            or after_stat.st_mtime_ns != before_stat.st_mtime_ns
            or after_stat.st_size != before_stat.st_size
        )
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "Session log archive write attempted (chat_id=%s message_id=%s terminal=%s wrote=%s)",
                chat_id,
                message_id,
                terminal,
                wrote,
            )

        if wrote:
            with contextlib.suppress(Exception):
                self._delete_artifacts_sync(ids + [lock_id])
            return True

        # If writing failed, keep segments for retry and allow lock reaping.
        with contextlib.suppress(Exception):
            self._delete_artifacts_sync([lock_id])
        return False


    # ======================================================================
    # _run_startup_checks (30 lines)
    # ======================================================================

    @timed
    async def _run_startup_checks(self) -> None:
        """Warm OpenRouter connections and log readiness without blocking startup."""
        session: aiohttp.ClientSession | None = None
        try:
            api_key, api_key_error = self._resolve_openrouter_api_key(self.valves)
            if api_key_error or not api_key:
                self.logger.debug(
                    "Skipping OpenRouter warmup: %s",
                    api_key_error or "API key missing (will retry when configured).",
                )
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


    # ======================================================================
    # _ensure_concurrency_controls (47 lines)
    # ======================================================================

    @timed
    async def _ensure_concurrency_controls(self, valves: "Pipe.Valves") -> None:
        """Lazy-initialize queue worker and semaphore with the latest valves."""
        cls = type(self)
        current_loop = asyncio.get_running_loop()

        # Check if existing lock is bound to a different (stale) event loop
        if self._queue_worker_lock is not None:
            try:
                lock_loop = getattr(cast(Any, self._queue_worker_lock), "_get_loop", lambda: None)()
                if lock_loop is not current_loop:
                    self._queue_worker_lock = None
            except RuntimeError:
                # Lock is bound to a closed/different loop
                self._queue_worker_lock = None

        if self._queue_worker_lock is None:
            self._queue_worker_lock = asyncio.Lock()

        async with self._queue_worker_lock:
            if self._queue_worker_task is not None and not self._queue_worker_task.done():
                try:
                    worker_loop = self._queue_worker_task.get_loop()
                except Exception:  # pragma: no cover - defensive for older asyncio implementations
                    worker_loop = None
                if worker_loop is not None and worker_loop is not current_loop:
                    # The worker task belongs to a different event loop (common in test runners).
                    # Drop the stale references so a new loop can recreate them.
                    self.logger.debug(
                        "Dropping stale request worker bound to a different event loop during setup."
                    )
                    self._queue_worker_task = None
                    self._request_queue = None

            if self._request_queue is not None:
                try:
                    queue_loop = self._request_queue._get_loop()  # type: ignore[attr-defined]
                except Exception:
                    queue_loop = getattr(self._request_queue, "_loop", None)
                if queue_loop is not None and queue_loop is not current_loop:
                    self.logger.debug(
                        "Dropping stale request queue bound to a different event loop during setup."
                    )
                    self._request_queue = None
                    self._queue_worker_task = None

            if self._request_queue is None:
                self._request_queue = asyncio.Queue(maxsize=self._QUEUE_MAXSIZE)
                self.logger.debug("Created request queue (maxsize=%s)", self._QUEUE_MAXSIZE)

            if self._queue_worker_task is None or self._queue_worker_task.done():
                self._queue_worker_task = current_loop.create_task(
                    Pipe._request_worker_loop(self._request_queue),
                    name="openrouter-pipe-dispatch",
                )
                self.logger.debug("Started request queue worker")

            # Semaphores remain class-level for global rate limiting
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
                self.logger.warning("Lower MAX_CONCURRENT_REQUESTS (%s->%s) requires restart to take full effect.", cls._semaphore_limit, target)

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
                self.logger.warning("Lower MAX_PARALLEL_TOOLS_GLOBAL (%s->%s) requires restart to take full effect.", cls._tool_global_limit, target_tool)


    # ======================================================================
    # _enqueue_job (14 lines)
    # ======================================================================

    @timed
    def _enqueue_job(self, job: _PipeJob) -> bool:
        """Attempt to enqueue a request, returning False when the queue is full."""
        queue = self._request_queue
        if queue is None:
            raise RuntimeError("Request queue not initialized")
        try:
            queue.put_nowait(job)
            self.logger.debug("Enqueued request %s (depth=%s)", job.request_id, queue.qsize())
            return True
        except asyncio.QueueFull:
            self.logger.warning("Request queue full (max=%s)", queue.maxsize)
            return False

    # ======================================================================
    # _request_worker_loop (27 lines)
    # ======================================================================

    @staticmethod
    @timed
    async def _request_worker_loop(queue: asyncio.Queue) -> None:
        """Background worker that dequeues jobs and spawns per-request tasks."""
        if queue is None:
            return
        try:
            while True:
                job = await queue.get()
                if job.future.cancelled():
                    queue.task_done()
                    continue
                task = asyncio.create_task(job.pipe._execute_pipe_job(job))

                @timed
                def _mark_done(_task: asyncio.Task, q=queue) -> None:
                    q.task_done()

                task.add_done_callback(_mark_done)

                @timed
                def _propagate_cancel(fut: asyncio.Future, _task: asyncio.Task = task, _job_id: str = job.request_id) -> None:
                    if fut.cancelled() and not _task.done():
                        job.pipe.logger.debug("Cancelling in-flight request (request_id=%s)", _job_id)
                        _task.cancel()

                job.future.add_done_callback(_propagate_cancel)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            return


    # ======================================================================
    # _execute_pipe_job (86 lines)
    # ======================================================================

    @timed
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
                    # Phase 3: Add context for process_tool_result() integration
                    request=job.request,
                    user=job.user,
                    metadata=job.metadata,
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
                self._try_put_middleware_stream_nowait(
                    stream_queue,
                    {"error": {"detail": str(exc)}},
                )
            if not job.future.done():
                job.future.set_exception(exc)
        finally:
            if stream_queue is not None:
                if stream_emitter is not None:
                    flush_reasoning_status = getattr(stream_emitter, "flush_reasoning_status", None)
                    if callable(flush_reasoning_status):
                        with contextlib.suppress(Exception):
                            maybe_awaitable = flush_reasoning_status()
                            if inspect.isawaitable(maybe_awaitable):
                                await maybe_awaitable
                self._try_put_middleware_stream_nowait(stream_queue, None)
            if tool_context:
                await self._shutdown_tool_context(tool_context)

            # Non-streaming fallback: if the streaming subsystem didn't consume and
            # persist session logs, stage them here so the assembler can produce
            # the per-message zip. Task requests (title/tags/followups) are included
            # when they share the same message_id - the archive merger will combine
            # all events from all invocations into a single comprehensive archive.
            rid = SessionLogger.request_id.get() or ""
            if rid:
                with SessionLogger._state_lock:
                    fallback_events = list(SessionLogger.logs.get(rid, []))
                if fallback_events:
                    status = "complete"
                    reason = ""
                    if job.future.cancelled():
                        status = "cancelled"
                        reason = "cancelled"
                    else:
                        with contextlib.suppress(Exception):
                            exc = job.future.exception()
                            if exc is not None:
                                status = "error"
                                reason = str(exc)

                    resolved_user_id = str(job.user_id or job.user.get("id") or job.metadata.get("user_id") or "")
                    resolved_session_id = str(job.session_id or job.metadata.get("session_id") or "")
                    resolved_chat_id = str(job.metadata.get("chat_id") or "")
                    resolved_message_id = str(job.metadata.get("message_id") or "")
                    try:
                        await asyncio.shield(
                            self._persist_session_log_segment_to_db(
                                job.valves,
                                user_id=resolved_user_id,
                                session_id=resolved_session_id,
                                chat_id=resolved_chat_id,
                                message_id=resolved_message_id,
                                request_id=rid,
                                log_events=fallback_events,
                                terminal=True,
                                status=status,
                                reason=reason,
                                pipe_identifier=self.id,
                            )
                        )
                    except Exception:
                        self.logger.debug(
                            "Failed to persist session log segment (chat_id=%s message_id=%s request_id=%s terminal=%s)",
                            resolved_chat_id,
                            resolved_message_id,
                            rid,
                            True,
                            exc_info=True,
                        )
                    with SessionLogger._state_lock:
                        SessionLogger.logs.pop(rid, None)

            if tool_token is not None:
                self._TOOL_CONTEXT.reset(tool_token)
            for var, token in tokens:
                with contextlib.suppress(Exception):
                    var.reset(token)
            if session:
                with contextlib.suppress(Exception):
                    await session.close()


    # ======================================================================
    # _acquire_semaphore (15 lines)
    # ======================================================================

    @contextlib.asynccontextmanager
    @timed
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

    @timed
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

        # Set timing context if timing logging is enabled
        if request_id:
            with contextlib.suppress(Exception):
                from .core.timing_logger import set_timing_context

                set_timing_context(
                    request_id=request_id,
                    enabled=bool(job.valves.ENABLE_TIMING_LOG),
                )

        return tokens


    # ======================================================================
    # _breaker_allows (10 lines)
    # ======================================================================

    @timed
    def _breaker_allows(self, user_id: str) -> bool:
        """Delegate to CircuitBreaker.allows."""
        return self._circuit_breaker.allows(user_id)


    # ======================================================================
    # _record_failure (1 line)
    # ======================================================================

    @timed
    def _record_failure(self, user_id: str) -> None:
        """Delegate to CircuitBreaker.record_failure."""
        return self._circuit_breaker.record_failure(user_id)

    # ======================================================================
    # _reset_failure_counter (reset request breaker)
    # ======================================================================

    @timed
    def _reset_failure_counter(self, user_id: str) -> None:
        """Delegate to CircuitBreaker.reset."""
        return self._circuit_breaker.reset(user_id)

    # ======================================================================
    # _tool_type_allows (check tool breaker)
    # ======================================================================

    @timed
    def _tool_type_allows(self, user_id: str, tool_type: str) -> bool:
        """Delegate to CircuitBreaker.tool_allows."""
        return self._circuit_breaker.tool_allows(user_id, tool_type)

    # ======================================================================
    # _record_tool_failure_type (record tool failure)
    # ======================================================================

    @timed
    def _record_tool_failure_type(self, user_id: str, tool_type: str) -> None:
        """Delegate to CircuitBreaker.record_tool_failure."""
        return self._circuit_breaker.record_tool_failure(user_id, tool_type)

    # ======================================================================
    # _reset_tool_failure_type (reset tool breaker)
    # ======================================================================

    @timed
    def _reset_tool_failure_type(self, user_id: str, tool_type: str) -> None:
        """Delegate to CircuitBreaker.reset_tool."""
        return self._circuit_breaker.reset_tool(user_id, tool_type)


    # ======================================================================
    # _maybe_dump_costs_snapshot (73 lines)
    # ======================================================================

    @timed
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

        @timed
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

    # ======================================================================
    # _handle_pipe_call (273 lines)
    # ======================================================================

    @timed
    async def _handle_pipe_call(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __request__: Request | None,
        __event_emitter__: EventEmitter | None,
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        __metadata__: dict[str, Any],
        __tools__: list[dict[str, Any]] | dict[str, Any] | None,
        __task__: Any = None,
        __task_body__: Any = None,
        *,
        valves: Pipe.Valves | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> AsyncGenerator[str, None] | dict[str, Any] | str | None:
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

        task_name = self._task_name(__task__)
        if task_name and self._auth_failure_active():
            # Suppress background task calls after an auth failure to avoid log spam
            # and repeated upstream requests.
            fallback = self._build_task_fallback_content(task_name)
            return self._build_chat_completion_payload(
                model=str(body.get("model") or openwebui_model_id or "pipe"),
                content=fallback,
            )

        api_key_value, api_key_error = self._resolve_openrouter_api_key(valves)
        if api_key_error:
            self._note_auth_failure()
            # Task calls are background; return a safe stub without emitting UI errors.
            if task_name:
                fallback = self._build_task_fallback_content(task_name)
                return self._build_chat_completion_payload(
                    model=str(body.get("model") or openwebui_model_id or "pipe"),
                    content=fallback,
                )

            template = valves.AUTHENTICATION_ERROR_TEMPLATE
            variables = {
                "openrouter_code": 401,
                "openrouter_message": api_key_error,
            }
            # For streaming requests we must emit and finish the stream.
            if bool(body.get("stream")) and __event_emitter__:
                await self._emit_templated_error(
                    __event_emitter__,
                    template=template,
                    variables=variables,
                    log_message=f"Auth configuration error: {api_key_error}",
                    log_level=logging.WARNING,
                )
                return ""

            # Non-streaming: return a normal chat completion payload with the markdown.
            error_id, context_defaults = self._build_error_context()
            enriched_variables = {**context_defaults, **variables}
            try:
                markdown = _render_error_template(template, enriched_variables)
            except Exception:
                markdown = (
                    "### 🔐 Authentication Failed\n\n"
                    f"{api_key_error}\n\n"
                    f"**Error ID:** `{error_id}`\n\n"
                    "Verify the API key configured for this pipe."
                )
            self.logger.warning(
                "[%s] Auth configuration error (session=%s, user=%s): %s",
                error_id,
                enriched_variables.get("session_id") or "",
                enriched_variables.get("user_id") or "",
                api_key_error,
            )
            return self._build_chat_completion_payload(
                model=str(body.get("model") or openwebui_model_id or "pipe"),
                content=markdown,
            )

        try:
            await OpenRouterModelRegistry.ensure_loaded(
                session,
                base_url=valves.BASE_URL,
                api_key=api_key_value or "",
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
        catalog_norm_ids = {m["norm_id"] for m in available_models if isinstance(m, dict) and m.get("norm_id")}
        allowlist_models = self._select_models(valves.MODEL_ID, available_models) or available_models
        allowlist_norm_ids = {m["norm_id"] for m in allowlist_models if isinstance(m, dict) and m.get("norm_id")}
        enforced_models = self._apply_model_filters(allowlist_models, valves)
        enforced_norm_ids = {m["norm_id"] for m in enforced_models if isinstance(m, dict) and m.get("norm_id")}

        # Full model ID, e.g. "<pipe-id>.gpt-4o"
        pipe_token = ModelFamily._PIPE_ID.set(pipe_identifier)
        features = _extract_feature_flags(__metadata__)
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
                allowlist_norm_ids,
                enforced_norm_ids,
                catalog_norm_ids,
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



    # ======================================================================
    # _process_transformed_request (725 lines)
    # ======================================================================

    @timed
    async def _process_transformed_request(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __request__: Request | None,
        __event_emitter__: EventEmitter | None,
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        __metadata__: dict[str, Any],
        __tools__: list[dict[str, Any]] | dict[str, Any] | None,
        __task__: Any,
        __task_body__: Any,
        valves: "Pipe.Valves",
        session: aiohttp.ClientSession,
        openwebui_model_id: str,
        pipe_identifier: str,
        allowlist_norm_ids: set[str],
        enforced_norm_ids: set[str],
        catalog_norm_ids: set[str],
        features: dict[str, Any],
        *,
        user_id: str = "",
    ) -> AsyncGenerator[str, None] | dict[str, Any] | str | None:
        """Delegate to RequestOrchestrator.process_request."""
        return await self._ensure_request_orchestrator().process_request(
            body, __user__, __request__, __event_emitter__, __event_call__, __metadata__, __tools__,
            __task__, __task_body__, valves, session, openwebui_model_id, pipe_identifier,
            allowlist_norm_ids, enforced_norm_ids, catalog_norm_ids, features, user_id=user_id
        )

    # ======================================================================
    # Anthropic Integration (bound from integrations/anthropic.py)
    # ======================================================================

    # Anthropic prompt caching integration
    _maybe_apply_anthropic_prompt_caching = _maybe_apply_anthropic_prompt_caching

    # ======================================================================
    # Request Processing (bound from request/ subdirectory)
    # ======================================================================

    # Request sanitization
    _sanitize_request_input = _sanitize_request_input

    # Message transformation
    transform_messages_to_input = transform_messages_to_input

    # ======================================================================
    # Model Management
    # ======================================================================

    @timed
    def _qualify_model_for_pipe(
        self,
        pipe_identifier: Optional[str],
        model_id: Optional[str],
    ) -> Optional[str]:
        """Return a dot-prefixed Open WebUI model id for this pipe.

        Args:
            pipe_identifier: The pipe identifier prefix (e.g., "openrouter")
            model_id: The model ID to qualify

        Returns:
            Qualified model ID with pipe prefix, or None if invalid
        """
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

    # ======================================================================
    # OpenRouter API Adapters
    # ======================================================================

    @timed
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
        """Delegate to ResponsesAdapter.send_openai_responses_streaming_request."""
        async for event in self._ensure_responses_adapter().send_openai_responses_streaming_request(
            session, request_body, api_key, base_url, valves=valves, workers=workers,
            breaker_key=breaker_key, delta_char_limit=delta_char_limit, idle_flush_ms=idle_flush_ms,
            chunk_queue_maxsize=chunk_queue_maxsize, event_queue_maxsize=event_queue_maxsize,
            event_queue_warn_size=event_queue_warn_size
        ):
            yield event

    @timed
    async def send_openai_chat_completions_streaming_request(
        self,
        session: aiohttp.ClientSession,
        responses_request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: "Pipe.Valves | None" = None,
        breaker_key: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Delegate to ChatCompletionsAdapter.send_openai_chat_completions_streaming_request."""
        async for event in self._ensure_chat_completions_adapter().send_openai_chat_completions_streaming_request(
            session, responses_request_body, api_key, base_url, valves=valves, breaker_key=breaker_key
        ):
            yield event

    @timed
    async def send_openai_chat_completions_nonstreaming_request(
        self,
        session: aiohttp.ClientSession,
        responses_request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: "Pipe.Valves | None" = None,
        breaker_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Delegate to ChatCompletionsAdapter.send_openai_chat_completions_nonstreaming_request."""
        return await self._ensure_chat_completions_adapter().send_openai_chat_completions_nonstreaming_request(
            session, responses_request_body, api_key, base_url, valves=valves, breaker_key=breaker_key
        )

    @timed
    async def send_openrouter_nonstreaming_request_as_events(
        self,
        session: aiohttp.ClientSession,
        responses_request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: "Pipe.Valves | None" = None,
        endpoint_override: Literal["responses", "chat_completions"] | None = None,
        breaker_key: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Delegate to NonStreamingAdapter.send_openrouter_nonstreaming_request_as_events."""
        async for event in self._ensure_nonstreaming_adapter().send_openrouter_nonstreaming_request_as_events(
            session,
            responses_request_body,
            api_key,
            base_url,
            valves=valves,
            endpoint_override=endpoint_override,
            breaker_key=breaker_key,
        ):
            yield event


    # ======================================================================
    # send_openrouter_streaming_request (113 lines)
    # ======================================================================

    @timed
    async def send_openrouter_streaming_request(
        self,
        session: aiohttp.ClientSession,
        responses_request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: "Pipe.Valves | None" = None,
        endpoint_override: Literal["responses", "chat_completions"] | None = None,
        workers: int = 4,
        breaker_key: Optional[str] = None,
        delta_char_limit: int = 0,
        idle_flush_ms: int = 0,
        chunk_queue_maxsize: int = 100,
        event_queue_maxsize: int = 100,
        event_queue_warn_size: int = 1000,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Delegate to ChatCompletionsAdapter.send_openrouter_streaming_request."""
        async for event in self._ensure_chat_completions_adapter().send_openrouter_streaming_request(
            session, responses_request_body, api_key, base_url, valves=valves,
            endpoint_override=endpoint_override, workers=workers, breaker_key=breaker_key,
            delta_char_limit=delta_char_limit, idle_flush_ms=idle_flush_ms,
            chunk_queue_maxsize=chunk_queue_maxsize, event_queue_maxsize=event_queue_maxsize,
            event_queue_warn_size=event_queue_warn_size
        ):
            yield event

    @timed
    async def _shutdown_tool_context(self, context: _ToolExecutionContext) -> None:
        """Stop per-request tool workers (bounded wait, then cancel)."""

        @timed
        async def _graceful() -> None:
            active_workers = [task for task in context.workers if not task.done()]
            worker_count = len(active_workers)
            if not worker_count:
                return
            for _ in range(worker_count):
                await context.queue.put(None)
            await context.queue.join()

        timeout = self.valves.TOOL_SHUTDOWN_TIMEOUT_SECONDS
        try:
            if timeout <= 0:
                raise asyncio.TimeoutError()
            await asyncio.wait_for(_graceful(), timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.warning(
                "Tool shutdown exceeded %.1fs; cancelling workers.",
                timeout,
            )
        except Exception:
            self.logger.debug(
                "Tool shutdown encountered error; cancelling workers.",
                exc_info=self.logger.isEnabledFor(logging.DEBUG),
            )

        for task in context.workers:
            if not task.done():
                task.cancel()
        if context.workers:
            await asyncio.gather(*context.workers, return_exceptions=True)

    # ======================================================================
    # Tool Execution Methods
    # ======================================================================

    # Tool worker methods (bound from tools/tool_worker.py)
    _tool_worker_loop = _tool_worker_loop
    _can_batch_tool_calls = _can_batch_tool_calls
    _args_reference_call = _args_reference_call

    @timed
    async def _execute_tool_batch(
        self,
        batch: list[_QueuedToolCall],
        context: _ToolExecutionContext,
    ) -> None:
        """Execute a batch of tool calls in parallel."""
        if not batch:
            return
        self.logger.debug("Batched %s tool(s) for %s", len(batch), batch[0].call.get("name"))
        tasks = [self._invoke_tool_call(item, context) for item in batch]
        gather_coro = asyncio.gather(*tasks, return_exceptions=True)
        results: list[tuple[str, str, list[dict[str, Any]], list[str]] | BaseException] = []
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
                if self.logger.isEnabledFor(logging.DEBUG):
                    call_name = item.call.get("name")
                    call_id = item.call.get("call_id")
                    self.logger.debug(
                        "Tool execution raised exception (name=%s, call_id=%s)",
                        call_name,
                        call_id,
                        exc_info=(type(result), result, result.__traceback__),
                    )
                payload = self._build_tool_output(
                    item.call,
                    f"Tool error: {result}",
                    status="failed",
                )
            else:
                status, text, files, embeds = result
                payload = self._build_tool_output(item.call, text, status=status, files=files, embeds=embeds)
                tool_type = (item.tool_cfg.get("type") or "function").lower()
                self._reset_tool_failure_type(context.user_id, tool_type)
            item.future.set_result(payload)

    @timed
    async def _invoke_tool_call(
        self,
        item: _QueuedToolCall,
        context: _ToolExecutionContext,
    ) -> tuple[str, str, list[dict[str, Any]], list[str]]:
        """Invoke a single tool call with circuit breaker protection."""
        tool_type = (item.tool_cfg.get("type") or "function").lower()
        if not self._tool_type_allows(context.user_id, tool_type):
            await self._notify_tool_breaker(context, tool_type, item.call.get("name"))
            return (
                "skipped",
                f"Tool '{item.call.get('name')}' temporarily disabled due to repeated errors.",
                [],
                [],
            )

        async with context.per_request_semaphore:
            if context.global_semaphore is not None:
                async with self._acquire_tool_global(context.global_semaphore, item.call.get("name")):
                    return await self._run_tool_with_retries(item, context, tool_type)
            return await self._run_tool_with_retries(item, context, tool_type)

    @timed
    async def _run_tool_with_retries(
        self,
        item: _QueuedToolCall,
        context: _ToolExecutionContext,
        tool_type: str,
    ) -> tuple[str, str, list[dict[str, Any]], list[str]]:
        """Run a tool with retry logic.

        This method executes the tool callable with timeout and optional retries,
        then processes the result to extract text, files, and embeds.
        Files and embeds are emitted to UI via event_emitter AND returned for
        inclusion in tool card HTML attributes.

        Returns:
            Tuple of (status, text, files, embeds) where:
            - status: "completed", "failed", or "skipped"
            - text: Processed tool output as string
            - files: List of file dicts (e.g., [{"type": "image", "url": "..."}])
            - embeds: List of HTML embed strings
        """
        tool_name = item.call.get("name", "unknown")
        timing_mark(f"tool_run:{tool_name}:start")

        fn = item.tool_cfg.get("callable")
        if not callable(fn):
            message = f"Tool '{tool_name}' is missing a callable handler."
            self.logger.warning("%s", message)
            self._record_tool_failure_type(context.user_id, tool_type)
            return ("failed", message, [], [])
        fn_to_call = cast(ToolCallable, fn)
        timeout = float(context.timeout)

        # Helper to process result and emit files/embeds
        async def _process_and_emit(raw_result: Any) -> tuple[str, list[dict[str, Any]], list[str]]:
            timing_mark(f"tool_run:{tool_name}:processing")
            try:
                # Use the tool executor's safe processing method
                executor = self._ensure_tool_executor()
                text, files, embeds = await executor._process_tool_result_safe(
                    tool_name=tool_name,
                    tool_type=tool_type,
                    raw_result=raw_result,
                    context=context,
                )

                # Emit files if any were extracted
                if files and context.event_emitter:
                    try:
                        await self._emit_files(context.event_emitter, files)
                        timing_mark(f"tool_run:{tool_name}:files_emitted")
                    except Exception as emit_exc:
                        self.logger.debug("Failed to emit files for '%s': %s", tool_name, emit_exc)

                # Emit embeds if any were extracted
                if embeds and context.event_emitter:
                    try:
                        await self._emit_embeds(context.event_emitter, embeds)
                        timing_mark(f"tool_run:{tool_name}:embeds_emitted")
                    except Exception as emit_exc:
                        self.logger.debug("Failed to emit embeds for '%s': %s", tool_name, emit_exc)

                return text, files, embeds
            except Exception as proc_exc:
                # Safety net - never crash, just return stringified result
                self.logger.debug("Result processing failed for '%s': %s", tool_name, proc_exc)
                return ("" if raw_result is None else str(raw_result)), [], []

        # Import tenacity for retries
        try:
            from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type
        except ImportError:
            # Fallback without retries if tenacity not available
            try:
                timing_mark(f"tool_run:{tool_name}:executing_no_retry")
                result = await asyncio.wait_for(
                    self._call_tool_callable(fn_to_call, item.args),
                    timeout=timeout,
                )
                self._reset_tool_failure_type(context.user_id, tool_type)
                text, files, embeds = await _process_and_emit(result)
                timing_mark(f"tool_run:{tool_name}:done")
                return ("completed", text, files, embeds)
            except Exception as exc:
                self._record_tool_failure_type(context.user_id, tool_type)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "Tool '%s' execution failed.",
                        tool_name,
                        exc_info=True,
                    )
                raise

        retryer = AsyncRetrying(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=0.2, min=0.2, max=1),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        try:
            async for attempt in retryer:
                with attempt:
                    timing_mark(f"tool_run:{tool_name}:attempt_{attempt.retry_state.attempt_number}")
                    result = await asyncio.wait_for(
                        self._call_tool_callable(fn_to_call, item.args),
                        timeout=timeout,
                    )
                    self._reset_tool_failure_type(context.user_id, tool_type)
                    text, files, embeds = await _process_and_emit(result)
                    timing_mark(f"tool_run:{tool_name}:done")
                    return ("completed", text, files, embeds)
        except Exception as exc:
            self._record_tool_failure_type(context.user_id, tool_type)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Tool '%s' execution failed.",
                    tool_name,
                    exc_info=True,
                )
            raise
        return ("failed", "Tool execution produced no output.", [], [])

    @timed
    async def _call_tool_callable(self, fn: ToolCallable, args: dict[str, Any]) -> Any:
        """Call a tool callable (sync or async)."""
        if inspect.iscoroutinefunction(fn):
            return await fn(**args)
        result = await asyncio.to_thread(fn, **args)
        if inspect.isawaitable(result):
            return await result
        return result

    @contextlib.asynccontextmanager
    @timed
    async def _acquire_tool_global(self, semaphore: asyncio.Semaphore, tool_name: str | None):
        """Acquire global tool semaphore slot."""
        self.logger.debug("Waiting for global tool slot (%s)", tool_name)
        await semaphore.acquire()
        try:
            yield
        finally:
            semaphore.release()

    # ======================================================================
    # _execute_function_calls (159 lines)
    # ======================================================================

    @timed
    async def _execute_function_calls(
        self,
        calls: list[dict],
        tools: dict[str, dict[str, Any]],
    ) -> list[dict]:
        """Delegate to ToolExecutor._execute_function_calls."""
        return await self._ensure_tool_executor()._execute_function_calls(calls, tools)


    # ======================================================================
    # _report_openrouter_error (47 lines)
    # ======================================================================

    @timed
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
        """Delegate to ErrorFormatter._report_openrouter_error."""
        return await self._ensure_error_formatter()._report_openrouter_error(
            exc,
            event_emitter=event_emitter,
            normalized_model_id=normalized_model_id,
            api_model_id=api_model_id,
            usage=usage,
            template=template,
        )


    # ======================================================================
    # _format_final_status_description (89 lines)
    # ======================================================================

    @timed
    def _format_final_status_description(
        self,
        *,
        elapsed: float,
        total_usage: Dict[str, Any],
        valves: "Pipe.Valves",
        stream_duration: Optional[float] = None,
    ) -> str:
        """Delegate to ErrorFormatter._format_final_status_description."""
        return self._ensure_error_formatter()._format_final_status_description(
            elapsed=elapsed,
            total_usage=total_usage,
            valves=valves,
            stream_duration=stream_duration,
        )

    # 4.8 Internal Static Helpers



    # ======================================================================
    # ----------------------------------------------------------------------
    # send_openai_responses_nonstreaming_request (69 lines)
    # ----------------------------------------------------------------------

    @timed
    async def send_openai_responses_nonstreaming_request(
        self,
        session: aiohttp.ClientSession,
        request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: "Pipe.Valves | None" = None,
        breaker_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Delegate to ResponsesAdapter.send_openai_responses_nonstreaming_request."""
        return await self._ensure_responses_adapter().send_openai_responses_nonstreaming_request(
            session, request_body, api_key, base_url, valves=valves, breaker_key=breaker_key
        )

    @timed
    async def _run_task_model_request(
        self,
        body: Dict[str, Any],
        valves: Pipe.Valves,
        *,
        session: aiohttp.ClientSession | None = None,
        task_context: Any = None,
        owui_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        user_obj: Optional[Any] = None,
        pipe_id: Optional[str] = None,
        snapshot_model_id: Optional[str] = None,
    ) -> str:
        """Delegate to TaskModelAdapter._run_task_model_request."""
        return await self._ensure_task_model_adapter()._run_task_model_request(
            body,
            valves,
            session=session,
            task_context=task_context,
            owui_metadata=owui_metadata,
            user_id=user_id,
            user_obj=user_obj,
            pipe_id=pipe_id,
            snapshot_model_id=snapshot_model_id,
        )

    # ADDITIONAL HELPER METHODS (called by orchestration methods)
    # ======================================================================

    # ----------------------------------------------------------------------
    # _ping_openrouter (28 lines)
    # ----------------------------------------------------------------------

    @timed
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


    # ----------------------------------------------------------------------
    # _create_http_session (21 lines)
    # ----------------------------------------------------------------------

    @timed
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


    # ----------------------------------------------------------------------
    # _notify_tool_breaker (24 lines)
    # ----------------------------------------------------------------------

    @timed
    async def _notify_tool_breaker(
        self,
        context: _ToolExecutionContext,
        tool_type: str,
        tool_name: Optional[str],
    ) -> None:
        """Delegate to ToolExecutor._notify_tool_breaker."""
        return await self._ensure_tool_executor()._notify_tool_breaker(context, tool_type, tool_name)


    # ----------------------------------------------------------------------
    # _cleanup_session_log_archives (38 lines)
    # ----------------------------------------------------------------------

    @timed
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
                    # Don't rely on os.walk's cached dirnames/filenames; check
                    # the filesystem live so we can prune parent directories in
                    # the same pass.
                    with contextlib.suppress(Exception):
                        if any(Path(dirpath).iterdir()):
                            continue
                        os.rmdir(dirpath)
            except Exception:
                continue


    # ----------------------------------------------------------------------
    # _get_user_by_id (19 lines)
    # ----------------------------------------------------------------------

    @timed
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
        if Users is None:
            return None
        try:
            return await run_in_threadpool(Users.get_user_by_id, user_id)
        except Exception as exc:
            self.logger.error(f"Failed to load user {user_id}: {exc}")
            return None


    # ----------------------------------------------------------------------
    # _build_direct_tool_server_registry (154 lines)
    # ----------------------------------------------------------------------

    @timed
    def _build_direct_tool_server_registry(
        self,
        __metadata__: dict[str, Any],
        *,
        valves: "Pipe.Valves",
        event_call: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        event_emitter: EventEmitter | None,
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        """Delegate to ToolExecutor._build_direct_tool_server_registry."""
        return self._ensure_tool_executor()._build_direct_tool_server_registry(
            __metadata__, valves=valves, event_call=event_call, event_emitter=event_emitter
        )



    # ----------------------------------------------------------------------
    # _apply_reasoning_preferences (35 lines)
    # ----------------------------------------------------------------------

    @timed
    def _apply_reasoning_preferences(self, responses_body: ResponsesBody, valves: "Pipe.Valves") -> None:
        """Delegate to ReasoningConfigManager._apply_reasoning_preferences."""
        return self._ensure_reasoning_config_manager()._apply_reasoning_preferences(
            responses_body, valves
        )


    # ----------------------------------------------------------------------
    # _apply_task_reasoning_preferences (29 lines)
    # ----------------------------------------------------------------------

    @timed
    def _apply_task_reasoning_preferences(self, responses_body: ResponsesBody, effort: str) -> None:
        """Delegate to ReasoningConfigManager._apply_task_reasoning_preferences."""
        return self._ensure_reasoning_config_manager()._apply_task_reasoning_preferences(
            responses_body, effort
        )


    # ----------------------------------------------------------------------
    # _apply_gemini_thinking_config (47 lines)
    # ----------------------------------------------------------------------

    @timed
    def _apply_gemini_thinking_config(self, responses_body: ResponsesBody, valves: "Pipe.Valves") -> None:
        """Delegate to ReasoningConfigManager._apply_gemini_thinking_config."""
        return self._ensure_reasoning_config_manager()._apply_gemini_thinking_config(
            responses_body, valves
        )

    # ----------------------------------------------------------------------
    # _should_retry_without_reasoning (38 lines)
    # ----------------------------------------------------------------------

    @timed
    def _should_retry_without_reasoning(
        self,
        error: OpenRouterAPIError,
        responses_body: ResponsesBody,
    ) -> bool:
        """Delegate to ReasoningConfigManager._should_retry_without_reasoning."""
        return self._ensure_reasoning_config_manager()._should_retry_without_reasoning(
            error, responses_body
        )

    # ----------------------------------------------------------------------
    # _apply_context_transforms (9 lines)
    # ----------------------------------------------------------------------

    @timed
    def _apply_context_transforms(self, responses_body: ResponsesBody, valves: "Pipe.Valves") -> None:
        """Attach OpenRouter's middle-out transform when auto trimming is enabled."""

        if not valves.AUTO_CONTEXT_TRIMMING:
            return
        if responses_body.transforms is not None:
            return
        responses_body.transforms = ["middle-out"]


    # ----------------------------------------------------------------------
    # _is_free_model (6 lines)
    # ----------------------------------------------------------------------

    @timed
    def _is_free_model(self, model_norm_id: str) -> bool:
        """Check if a model has free pricing (all pricing values sum to 0)."""
        pricing = OpenRouterModelRegistry.spec(model_norm_id).get("pricing") or {}
        total, numeric_count = self._sum_pricing_values(pricing)
        if numeric_count <= 0:
            return False
        return total == Decimal(0)

    # ----------------------------------------------------------------------
    # _select_models (22 lines)
    # ----------------------------------------------------------------------

    @timed
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

    # ----------------------------------------------------------------------
    # _apply_model_filters (34 lines)
    # ----------------------------------------------------------------------

    @timed
    def _apply_model_filters(self, models: list[dict[str, Any]], valves: "Pipe.Valves") -> list[dict[str, Any]]:
        """Apply model capability filters (free pricing/tool calling) to a model list."""
        if not models:
            return []

        free_mode = (getattr(valves, "FREE_MODEL_FILTER", "all") or "all").strip().lower()
        tool_mode = (getattr(valves, "TOOL_CALLING_FILTER", "all") or "all").strip().lower()
        zdr_only = bool(getattr(valves, "ZDR_MODELS_ONLY", False))
        zdr_model_ids = None
        if zdr_only:
            zdr_model_ids = OpenRouterModelRegistry.zdr_model_ids()
            if zdr_model_ids is None:
                self.logger.warning(
                    "ZDR model filter enabled but ZDR endpoint list is unavailable; skipping ZDR filtering."
                )
        if free_mode == "all" and tool_mode == "all" and not (zdr_only and zdr_model_ids is not None):
            return models

        filtered: list[dict[str, Any]] = []
        for model in models:
            norm_id = model.get("norm_id") or ""
            if not norm_id:
                continue

            if zdr_only and zdr_model_ids is not None and norm_id not in zdr_model_ids:
                continue

            if free_mode != "all":
                is_free = self._is_free_model(norm_id)
                if free_mode == "only" and not is_free:
                    continue
                if free_mode == "exclude" and is_free:
                    continue

            if tool_mode != "all":
                supports_tools = self._supports_tool_calling(norm_id)
                if tool_mode == "only" and not supports_tools:
                    continue
                if tool_mode == "exclude" and supports_tools:
                    continue

            filtered.append(model)

        return filtered

    @timed
    def _expand_variant_models(
        self,
        models: list[dict[str, Any]],
        valves: "Pipe.Valves"
    ) -> list[dict[str, Any]]:
        """Expand model list by adding virtual variant and preset model entries.

        Supports two syntaxes:
        - Variants: "openai/gpt-4o:nitro" (uses : separator)
        - Presets: "openai/gpt-4o@preset/email-copywriter" (uses @ separator)

        Args:
            models: List of base models from catalog
            valves: Pipe configuration valves

        Returns:
            Extended list with both base models and variant/preset models
        """
        variant_specs_csv = (valves.VARIANT_MODELS or "").strip()
        if not variant_specs_csv:
            return models  # No variants configured

        # Parse CSV into (base_id, variant_tag, is_preset) tuples
        # Presets use @ separator, variants use : separator
        variant_specs: list[tuple[str, str, bool]] = []
        for spec in variant_specs_csv.split(","):
            spec = spec.strip()
            if not spec:
                continue

            # Try preset syntax first (@ separator)
            if "@" in spec:
                parts = spec.rsplit("@", 1)
                base_id = parts[0].strip()
                raw_tag = parts[1].strip()  # e.g., "preset/email-copywriter"
                is_preset = raw_tag.startswith("preset/")
                # Keep preset tags as-is (case-sensitive slugs)
                variant_tag = raw_tag
                if base_id and variant_tag:
                    variant_specs.append((base_id, variant_tag, is_preset))
            # Fall back to variant syntax (: separator)
            elif ":" in spec:
                parts = spec.rsplit(":", 1)
                base_id = parts[0].strip()
                variant_tag = parts[1].strip().lower()
                if base_id and variant_tag:
                    variant_specs.append((base_id, variant_tag, False))
            # Skip invalid entries (no separator)

        if not variant_specs:
            return models  # Nothing to expand

        # Build lookup map: original_id -> model dict
        model_map: dict[str, dict[str, Any]] = {}
        for model in models:
            original_id = model.get("original_id", "")
            if original_id:
                model_map[original_id] = model

        # Expand variants and presets
        expanded: list[dict[str, Any]] = list(models)  # Start with base models

        for base_id, variant_tag, is_preset in variant_specs:
            # Find base model
            base_model = model_map.get(base_id)
            if not base_model:
                separator = "@" if is_preset else ":"
                self.logger.warning(
                    "Variant model base not found: %s (skipping %s%s%s)",
                    base_id,
                    base_id,
                    separator,
                    variant_tag
                )
                continue

            # Clone base model and modify for variant/preset
            variant_model = dict(base_model)  # Shallow copy

            # Update ID to include variant suffix (always use : internally)
            base_sanitized_id = variant_model.get("id", "")
            variant_model["id"] = f"{base_sanitized_id}:{variant_tag}"

            # Keep original_id pointing to BASE (for icon/description lookups)
            # Do NOT change original_id - it must stay as base_id for catalog lookups

            # Update display name with tag
            # Use exact base name and append variant tag or preset label
            base_name = variant_model.get("name", base_id)
            if is_preset:
                # "preset/email-copywriter" → "Preset: email-copywriter"
                preset_slug = variant_tag.replace("preset/", "")
                tag_display = f"Preset: {preset_slug}"
            else:
                # Standard variant: "nitro" → "Nitro"
                tag_display = variant_tag.capitalize()
            variant_model["name"] = f"{base_name} {tag_display}"

            # Keep norm_id pointing to base (for capability lookups)
            # norm_id is used by ModelFamily.supports() to check features

            # Add to expanded list
            expanded.append(variant_model)

            self.logger.debug(
                "Added %s model: %s (from %s)",
                "preset" if is_preset else "variant",
                variant_model["name"],
                base_name
            )

        return expanded


    # ----------------------------------------------------------------------
    # _model_restriction_reasons (37 lines)
    # ----------------------------------------------------------------------

    @timed
    def _model_restriction_reasons(
        self,
        model_norm_id: str,
        *,
        valves: "Pipe.Valves",
        allowlist_norm_ids: set[str],
        catalog_norm_ids: set[str],
    ) -> list[str]:
        reasons: list[str] = []
        if catalog_norm_ids and model_norm_id not in catalog_norm_ids:
            reasons.append("not_in_catalog")

        model_id_filter = (valves.MODEL_ID or "").strip()
        if model_id_filter and model_id_filter.lower() != "auto":
            if model_norm_id not in allowlist_norm_ids:
                reasons.append("MODEL_ID")

        free_mode = (getattr(valves, "FREE_MODEL_FILTER", "all") or "all").strip().lower()
        if free_mode != "all" and model_norm_id in catalog_norm_ids:
            is_free = self._is_free_model(model_norm_id)
            if free_mode == "only" and not is_free:
                reasons.append("FREE_MODEL_FILTER=only")
            elif free_mode == "exclude" and is_free:
                reasons.append("FREE_MODEL_FILTER=exclude")

        tool_mode = (getattr(valves, "TOOL_CALLING_FILTER", "all") or "all").strip().lower()
        if tool_mode != "all" and model_norm_id in catalog_norm_ids:
            supports_tools = self._supports_tool_calling(model_norm_id)
            if tool_mode == "only" and not supports_tools:
                reasons.append("TOOL_CALLING_FILTER=only")
            elif tool_mode == "exclude" and supports_tools:
                reasons.append("TOOL_CALLING_FILTER=exclude")

        zdr_only = bool(getattr(valves, "ZDR_MODELS_ONLY", False))
        if zdr_only and model_norm_id in catalog_norm_ids:
            zdr_model_ids = OpenRouterModelRegistry.zdr_model_ids()
            if zdr_model_ids is not None and model_norm_id not in zdr_model_ids:
                reasons.append("ZDR_MODELS_ONLY")

        return reasons

    @timed
    def _build_icon_mapping(self, frontend_data: dict[str, Any] | None) -> dict[str, str]:
        """Delegate to ModelCatalogManager."""
        return self._ensure_catalog_manager()._build_icon_mapping(frontend_data)

    @timed
    def _build_web_search_support_mapping(
        self,
        frontend_data: dict[str, Any] | None,
    ) -> dict[str, bool]:
        """Delegate to ModelCatalogManager."""
        return self._ensure_catalog_manager()._build_web_search_support_mapping(frontend_data)

    @timed
    async def _fetch_frontend_model_catalog(
        self,
        session: Any,
    ) -> dict[str, Any] | None:
        """Delegate to ModelCatalogManager."""
        return await self._ensure_catalog_manager()._fetch_frontend_model_catalog(session)

    @timed
    async def _build_maker_profile_image_mapping(
        self,
        session: Any,
        maker_ids: Iterable[str],
    ) -> dict[str, str]:
        """Delegate to ModelCatalogManager."""
        return await self._ensure_catalog_manager()._build_maker_profile_image_mapping(
            session,
            maker_ids,
        )

    @timed
    async def _sync_model_metadata_to_owui(
        self,
        models: list[dict[str, Any]],
        *,
        pipe_identifier: str,
    ) -> None:
        """Delegate to ModelCatalogManager."""
        return await self._ensure_catalog_manager()._sync_model_metadata_to_owui(
            models,
            pipe_identifier=pipe_identifier,
        )

    @timed
    def _update_or_insert_model_with_metadata(
        self,
        openwebui_model_id: str,
        name: str,
        capabilities: Optional[dict],
        profile_image_url: Optional[str],
        update_capabilities: bool,
        update_images: bool,
        *,
        filter_function_id: str | None = None,
        filter_supported: bool = False,
        auto_attach_filter: bool = False,
        auto_default_filter: bool = False,
        direct_uploads_filter_function_id: str | None = None,
        direct_uploads_filter_supported: bool = False,
        auto_attach_direct_uploads_filter: bool = False,
        openrouter_pipe_capabilities: dict[str, bool] | None = None,
        description: str | None = None,
        update_descriptions: bool = False,
    ):
        """Delegate to ModelCatalogManager."""
        return self._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            openwebui_model_id,
            name,
            capabilities,
            profile_image_url,
            update_capabilities,
            update_images,
            filter_function_id=filter_function_id,
            filter_supported=filter_supported,
            auto_attach_filter=auto_attach_filter,
            auto_default_filter=auto_default_filter,
            direct_uploads_filter_function_id=direct_uploads_filter_function_id,
            direct_uploads_filter_supported=direct_uploads_filter_supported,
            auto_attach_direct_uploads_filter=auto_attach_direct_uploads_filter,
            openrouter_pipe_capabilities=openrouter_pipe_capabilities,
            description=description,
            update_descriptions=update_descriptions,
        )

    @staticmethod
    @timed
    def _extract_openrouter_og_image(html: str) -> str | None:
        """Wrapper for multimodal function (for test compatibility)."""
        return _extract_openrouter_og_image(html)

    @staticmethod
    @timed
    def _guess_image_mime_type(url: str, content_type: str | None, data: bytes) -> str | None:
        """Wrapper for multimodal function (for test compatibility)."""
        return _guess_image_mime_type(url, content_type, data)

    # 4.3 Core Multi-Turn Handlers
    @no_type_check

    # ----------------------------------------------------------------------
    # _maybe_apply_anthropic_beta_headers (31 lines)
    # ----------------------------------------------------------------------

    @timed
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
        if not self._is_anthropic_model_id(model_id):
            return

        feature = "interleaved-thinking-2025-05-14"
        existing = headers.get("x-anthropic-beta") or headers.get("X-Anthropic-Beta") or ""
        values = [part.strip() for part in existing.split(",") if part.strip()] if existing else []
        if feature not in values:
            values.append(feature)
        if values:
            headers["x-anthropic-beta"] = ",".join(values)
        headers.pop("X-Anthropic-Beta", None)

    @staticmethod

    # ----------------------------------------------------------------------
    # _is_anthropic_model_id (7 lines)
    # ----------------------------------------------------------------------

    @timed
    def _is_anthropic_model_id(model_id: Any) -> bool:
        if not isinstance(model_id, str):
            return False
        normalized = model_id.strip()
        return normalized.startswith("anthropic/") or normalized.startswith("anthropic.")

    @staticmethod
    @timed
    def _task_name(task: Any) -> str:
        """Delegate to TaskModelAdapter._task_name."""
        return TaskModelAdapter._task_name(task)

    @staticmethod

    # ----------------------------------------------------------------------
    # _chat_usage_to_responses_usage (38 lines)
    # ----------------------------------------------------------------------

    @staticmethod
    @timed
    def _chat_usage_to_responses_usage(raw_usage: Any) -> dict[str, Any]:
        """Delegate to ChatCompletionsAdapter._chat_usage_to_responses_usage."""
        from .api.gateway.chat_completions_adapter import ChatCompletionsAdapter
        return ChatCompletionsAdapter._chat_usage_to_responses_usage(raw_usage)


    # ----------------------------------------------------------------------
    # _execute_function_calls_legacy (69 lines)
    # ----------------------------------------------------------------------

    @timed
    async def _execute_function_calls_legacy(
        self,
        calls: list[dict],
        tools: dict[str, dict[str, Any]],
    ) -> list[dict]:
        """Delegate to ToolExecutor._execute_function_calls_legacy."""
        return await self._ensure_tool_executor()._execute_function_calls_legacy(calls, tools)


    # ----------------------------------------------------------------------
    # _build_tool_output (21 lines)
    # ----------------------------------------------------------------------

    @timed
    def _build_tool_output(
        self,
        call: dict[str, Any],
        output_text: str,
        *,
        status: str = "completed",
        files: list[dict[str, Any]] | None = None,
        embeds: list[str] | None = None,
    ) -> dict[str, Any]:
        """Delegate to ToolExecutor._build_tool_output."""
        return self._ensure_tool_executor()._build_tool_output(
            call, output_text, status=status, files=files, embeds=embeds
        )


    # ----------------------------------------------------------------------
    # _is_batchable_tool_call (4 lines)
    # ----------------------------------------------------------------------

    @timed
    def _is_batchable_tool_call(self, args: dict[str, Any]) -> bool:
        blockers = {"depends_on", "_depends_on", "sequential", "no_batch"}
        return not any(key in args for key in blockers)


    # ----------------------------------------------------------------------
    # _parse_tool_arguments (17 lines)
    # ----------------------------------------------------------------------

    @timed
    def _parse_tool_arguments(self, raw_args: Any) -> dict[str, Any]:
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise ValueError("Unable to parse tool arguments") from exc
        raise ValueError(f"Unsupported argument type: {type(raw_args).__name__}")
    # 4.7 Emitters (Front-end communication)


    # ----------------------------------------------------------------------
    # _note_auth_failure (12 lines)
    # ----------------------------------------------------------------------

    @classmethod
    @timed
    def _note_auth_failure(cls, *, ttl_seconds: Optional[int] = None) -> None:
        """Delegate to CircuitBreaker.note_auth_failure."""
        key = cls._auth_failure_scope_key()
        if not key:
            return
        CircuitBreaker.note_auth_failure(key, ttl_seconds=ttl_seconds)

    @classmethod

    # ----------------------------------------------------------------------
    # _auth_failure_active (15 lines)
    # ----------------------------------------------------------------------

    @timed
    def _auth_failure_active(cls) -> bool:
        """Delegate to CircuitBreaker.auth_failure_active."""
        key = cls._auth_failure_scope_key()
        if not key:
            return False
        return CircuitBreaker.auth_failure_active(key)

    @staticmethod
    @timed
    def _resolve_openrouter_api_key(valves: "Pipe.Valves") -> tuple[str | None, str | None]:
        """Return (api_key, error_message) where api_key is a usable bearer token.

        This guards against cases where `API_KEY` is stored encrypted but cannot
        be decrypted (WEBUI_SECRET_KEY mismatch / missing), which would
        otherwise be sent upstream as `Bearer encrypted:...` and cause noisy 401s.
        """
        raw_value = str(getattr(valves, "API_KEY", "") or "")
        raw_value = raw_value.strip()
        decrypted = EncryptedStr.decrypt(raw_value)
        decrypted = decrypted.strip() if isinstance(decrypted, str) else ""

        if not decrypted:
            return None, "OpenRouter API key is not configured."

        if raw_value.startswith(EncryptedStr._ENCRYPTION_PREFIX):
            # If the key was stored encrypted but decryption did not yield a plausible plaintext key,
            # treat it as an operator config issue.
            if decrypted.startswith(EncryptedStr._ENCRYPTION_PREFIX) or (not decrypted.startswith("sk-")):
                return (
                    None,
                    "OpenRouter API key is encrypted but cannot be decrypted. "
                    "This usually means WEBUI_SECRET_KEY changed. Re-enter the API key in this pipe's settings.",
                )

        return decrypted, None

    @staticmethod
    @timed
    def _input_contains_cache_control(value: Any) -> bool:
        """Recursively check if value contains cache_control markers.

        Used by Anthropic prompt caching to detect if cache breakpoints
        have already been inserted into the input.

        Args:
            value: Input value to check (dict, list, or other).

        Returns:
            bool: True if any cache_control markers are found.
        """
        if isinstance(value, dict):
            if "cache_control" in value:
                return True
            return any(Pipe._input_contains_cache_control(v) for v in value.values())
        if isinstance(value, list):
            return any(Pipe._input_contains_cache_control(v) for v in value)
        return False

    @staticmethod
    @timed
    def _strip_cache_control_from_input(value: Any) -> None:
        """Recursively remove cache_control markers from value.

        Used when retrying Anthropic requests that failed due to
        unsupported cache_control parameters.

        Args:
            value: Input value to strip markers from (modified in place).
        """
        if isinstance(value, dict):
            value.pop("cache_control", None)
            for v in value.values():
                Pipe._strip_cache_control_from_input(v)
            return
        if isinstance(value, list):
            for item in value:
                Pipe._strip_cache_control_from_input(item)


    # ----------------------------------------------------------------------
    # _build_chat_completion_payload (18 lines)
    # ----------------------------------------------------------------------

    @timed
    def _build_chat_completion_payload(self, *, model: str, content: str) -> dict[str, Any]:
        """Return a minimal OpenAI chat.completions-style payload."""
        model_id = (model or "pipe").strip() if isinstance(model, str) else "pipe"
        return {
            "id": f"{model_id}-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
        }


    # ----------------------------------------------------------------------
    # _build_task_fallback_content (12 lines)
    # ----------------------------------------------------------------------

    @timed
    def _build_task_fallback_content(self, task_name: str) -> str:
        """Return OWUI-parseable JSON content for known task types."""
        name = (task_name or "").strip().lower()
        if not name:
            return ""
        if "follow" in name:
            return json.dumps({"follow_ups": []})
        if "tag" in name:
            return json.dumps({"tags": ["General"]})
        if "title" in name:
            return json.dumps({"title": "Chat"})
        return ""

    # ----------------------------------------------------------------------
    # _extract_streaming_error_event (24 lines)
    # ----------------------------------------------------------------------

    @timed
    def _extract_streaming_error_event(
        self,
        event: dict[str, Any] | None,
        requested_model: Optional[str],
    ) -> Optional[OpenRouterAPIError]:
        """Delegate to ErrorFormatter._extract_streaming_error_event."""
        return self._ensure_error_formatter()._extract_streaming_error_event(
            event,
            requested_model,
        )


    # ----------------------------------------------------------------------
    # _merge_valves (104 lines)
    # ----------------------------------------------------------------------

    @timed
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
