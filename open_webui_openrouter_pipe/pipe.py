"""Main Pipe orchestrator for OpenRouter integration.

This module defines the Pipe class and coordinates:
- persistence, multimodal, streaming, and event subsystems
- request processing, tool execution, and model management
- lifecycle, background workers, and concurrency controls

Architecture:
- Pipe class manages initialization, lifecycle, and high-level orchestration
- Subsystems handle specific concerns (persistence, streaming, files, events)
- Subsystems are first-class modules with direct method calls (no shims)
- Orchestration remains in Pipe (workers, queues, HTTP, tools)
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import json
import logging
import secrets
import time
import uuid
from collections import defaultdict, deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable, Literal, Optional, TYPE_CHECKING, cast, no_type_check

# Third-party imports
import aiohttp
import httpx
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

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

# Optional Redis support
try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore

# Timing instrumentation
from .core.timing_logger import timed, timing_mark, configure_timing_file
from .storage.persistence import _RedisClient, _detect_redis_config

# Optional pyzipper support for session log encryption
try:
    import pyzipper  # pyright: ignore[reportMissingImports]
except ImportError:
    pyzipper = None  # type: ignore

# Import subsystems
from .storage.persistence import ArtifactStore
from .storage.multimodal import MultimodalHandler
from .streaming.streaming_core import StreamingHandler
from .streaming.event_emitter import EventEmitterHandler

# Import vendor integrations
from .integrations.anthropic import _is_anthropic_model_id

# Import model management
from .models.catalog_manager import ModelCatalogManager
from .models.reasoning_config import ReasoningConfigManager

# Import error handling
from .core.error_formatter import ErrorFormatter
from .core.circuit_breaker import CircuitBreaker

# Import logging
from .logging.session_log_manager import SessionLogManager

# Import request handling
from .requests import NonStreamingAdapter, TaskModelAdapter

# Import configuration and core modules
from .core.config import (
    Valves,
    UserValves,
    EncryptedStr,
    _PIPE_RUNTIME_ID,
    _OPENROUTER_TITLE,
    _OPENROUTER_REFERER,
    _select_openrouter_http_referer,
)
from .core.utils import _extract_feature_flags, _await_if_needed, _render_error_template
from .core.errors import _build_openrouter_api_error, OpenRouterAPIError
from .models.registry import (
    OpenRouterModelRegistry,
    ModelFamily,
    sanitize_model_id,
    is_free_model,
    supports_tool_calling,
)
from .tools.tool_executor import _QueuedToolCall, _ToolExecutionContext
from .core.logging_system import SessionLogger
from .streaming.event_emitter import EventEmitter

if TYPE_CHECKING:
    from .tools.tool_executor import ToolExecutor
    from .api.gateway.responses_adapter import ResponsesAdapter
    from .api.gateway.chat_completions_adapter import ChatCompletionsAdapter
    from .requests.orchestrator import RequestOrchestrator
    from .filters import FilterManager

ToolCallable = Callable[..., Awaitable[Any]] | Callable[..., Any]


def _consume_background_task_exception(task: asyncio.Task) -> None:
    """Silently consume exceptions from background tasks to avoid 'Task exception was never retrieved' warnings."""
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.exception()


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

        # Initialize EventEmitterHandler first (provides notification callback)
        self._event_emitter_handler: EventEmitterHandler = EventEmitterHandler(
            logger=self.logger,
            valves=self.valves,
            pipe_instance=self,
        )

        # Create ArtifactStore with notification callback from EventEmitterHandler
        self._artifact_store = ArtifactStore(
            pipe_id=pipe_id,
            logger=self.logger,
            valves=self.valves,
            emit_notification_callback=self._event_emitter_handler._emit_notification,
            tool_context_var=Pipe._TOOL_CONTEXT,
            user_id_context_var=SessionLogger.user_id,
        )

        # Initialize subsystem handlers synchronously (http_session=None, will be set in async init)
        self._multimodal_handler: MultimodalHandler = MultimodalHandler(
            logger=self.logger,
            valves=self.valves,
            http_session=None,  # Will be set in _ensure_async_subsystems_initialized
            artifact_store=None,  # Will be set after artifact store initialization
            emit_status_callback=None,
        )
        self._streaming_handler: StreamingHandler = StreamingHandler(
            logger=self.logger,
            valves=self.valves,
            model_registry=OpenRouterModelRegistry,  # Pass the class itself
            pipe_instance=self,
        )
        self._catalog_manager: Optional[ModelCatalogManager] = None
        self._error_formatter: Optional["ErrorFormatter"] = None
        self._reasoning_config_manager: Optional[ReasoningConfigManager] = None
        self._nonstreaming_adapter: Optional["NonStreamingAdapter"] = None
        self._task_model_adapter: Optional["TaskModelAdapter"] = None
        self._tool_executor: Optional["ToolExecutor"] = None
        self._responses_adapter: Optional["ResponsesAdapter"] = None
        self._chat_completions_adapter: Optional["ChatCompletionsAdapter"] = None
        self._request_orchestrator: Optional["RequestOrchestrator"] = None
        self._filter_manager: Optional["FilterManager"] = None

        # Circuit breaker state (per-user error tracking)
        self._circuit_breaker = CircuitBreaker(
            threshold=self.valves.BREAKER_MAX_FAILURES,
            window_seconds=self.valves.BREAKER_WINDOW_SECONDS,
        )
        # Synchronize circuit breaker config with ArtifactStore
        self._artifact_store.configure_breaker(
            threshold=self.valves.BREAKER_MAX_FAILURES,
            window_seconds=self.valves.BREAKER_WINDOW_SECONDS,
        )
        # DB breakers (separate from general circuit breaker)
        breaker_history_size = self.valves.BREAKER_HISTORY_SIZE
        self._db_breakers: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=breaker_history_size)
        )

        # One-time stale filter ID pruning (runs on first pipes() call)
        self._stale_filter_ids_pruned = False

        # Startup check coordination
        self._startup_task: asyncio.Task | None = None
        self._startup_checks_started = False
        self._startup_checks_pending = False
        self._startup_checks_complete = False
        self._warmup_failed = False

        # Redis configuration (detected from environment)
        self._redis_url, self._websocket_manager, self._websocket_redis_url, self._redis_candidate = (
            _detect_redis_config(self.valves, self.logger)
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
        self._session_log_manager = SessionLogManager(
            logger=self.logger,
            pipe=self,
            artifact_store=self._artifact_store,
        )
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
            self._multimodal_handler.set_http_session(self._http_session)
            self._multimodal_handler.set_artifact_store(self._artifact_store)

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
        SessionLogger.SESSION_LOG_MAX_LINES = self.valves.SESSION_LOG_MAX_LINES

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

        start_task = loop.create_task(_ensure_worker(), name="openrouter-log-worker-start")
        start_task.add_done_callback(_consume_background_task_exception)

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
            self._artifact_store._artifact_cleanup_worker(),
            name="openrouter-artifact-cleanup",
        )

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
            await _await_if_needed(client.ping(), timeout=5.0)
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
        self._redis_listener_task = loop.create_task(self._artifact_store._redis_pubsub_listener(), name="openrouter-redis-listener")
        self._redis_flush_task = loop.create_task(self._artifact_store._redis_periodic_flusher(), name="openrouter-redis-flush")

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

    def _ensure_catalog_manager(self) -> ModelCatalogManager:
        if self._catalog_manager is None:
            self._catalog_manager = ModelCatalogManager(
                pipe=self,
                multimodal_handler=self._multimodal_handler,
                logger=self.logger,
                task_done_callback=_consume_background_task_exception,
            )
        return self._catalog_manager

    def _ensure_error_formatter(self) -> ErrorFormatter:
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

    def _ensure_reasoning_config_manager(self) -> ReasoningConfigManager:
        if self._reasoning_config_manager is None:
            self._reasoning_config_manager = ReasoningConfigManager(
                pipe=self,
                logger=self.logger,
            )
        return self._reasoning_config_manager

    def _ensure_nonstreaming_adapter(self) -> NonStreamingAdapter:
        if self._nonstreaming_adapter is None:
            self._nonstreaming_adapter = NonStreamingAdapter(
                pipe=self,
                logger=self.logger,
            )
        return self._nonstreaming_adapter

    def _ensure_task_model_adapter(self) -> TaskModelAdapter:
        if self._task_model_adapter is None:
            self._task_model_adapter = TaskModelAdapter(
                pipe=self,
                logger=self.logger,
            )
        return self._task_model_adapter

    def _ensure_tool_executor(self) -> "ToolExecutor":
        if self._tool_executor is None:
            from .tools.tool_executor import ToolExecutor
            self._tool_executor = ToolExecutor(
                pipe=self,
                logger=self.logger,
            )
        return self._tool_executor

    def _ensure_responses_adapter(self) -> "ResponsesAdapter":
        if self._responses_adapter is None:
            from .api.gateway.responses_adapter import ResponsesAdapter
            self._responses_adapter = ResponsesAdapter(
                pipe=self,
                logger=self.logger,
            )
        return self._responses_adapter

    def _ensure_chat_completions_adapter(self) -> "ChatCompletionsAdapter":
        if self._chat_completions_adapter is None:
            from .api.gateway.chat_completions_adapter import ChatCompletionsAdapter
            self._chat_completions_adapter = ChatCompletionsAdapter(
                pipe=self,
                logger=self.logger,
            )
        return self._chat_completions_adapter

    def _ensure_request_orchestrator(self) -> "RequestOrchestrator":
        if self._request_orchestrator is None:
            from .requests.orchestrator import RequestOrchestrator
            self._request_orchestrator = RequestOrchestrator(
                pipe=self,
                logger=self.logger,
            )
        return self._request_orchestrator

    def _ensure_filter_manager(self) -> "FilterManager":
        if self._filter_manager is None:
            from .filters import FilterManager
            self._filter_manager = FilterManager(
                pipe=self,
                valves=self.valves,
                logger=self.logger,
            )
        return self._filter_manager

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
                await run_in_threadpool(self._ensure_filter_manager().ensure_ors_filter_function_id)
            except Exception as exc:
                self.logger.debug("AUTO_INSTALL_ORS_FILTER failed: %s", exc)
        if self.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER:
            try:
                await run_in_threadpool(self._ensure_filter_manager().ensure_direct_uploads_filter_function_id)
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
                        self._ensure_filter_manager().ensure_provider_routing_filters,
                        admin_routing,
                        user_routing,
                        provider_map,
                        selected_models,
                        self.id,
                    )
            except Exception as exc:
                self.logger.debug("Provider routing filter creation failed: %s", exc)

        # One-time cleanup of stale openrouter_* filter IDs in model metadata.
        # Must run inside pipes() — before OWUI's get_all_models() reads model
        # overlays and pre-warms the function cache — to prevent "Function not
        # found" crashes on OWUI 0.8.0+.
        if not self._stale_filter_ids_pruned:
            self._stale_filter_ids_pruned = True
            try:
                count = await run_in_threadpool(
                    self._ensure_catalog_manager().prune_stale_openrouter_filter_ids
                )
                if count:
                    self.logger.info(
                        "Pruned stale openrouter_* filter IDs from %d model(s) on startup.", count
                    )
            except Exception as exc:
                self.logger.debug("Startup stale filter ID pruning failed: %s", exc)

        self._ensure_catalog_manager().maybe_schedule_model_metadata_sync(
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
        safe_event_emitter = None

        try:
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

            safe_event_emitter = self._event_emitter_handler._wrap_safe_event_emitter(__event_emitter__)
            user_valves_raw = __user__.get("valves") or {}
            user_valves = self.UserValves.model_validate(user_valves_raw)
            valves = self._merge_valves(self.valves, user_valves)
            user_id = str(__user__.get("id") or __metadata__.get("user_id") or "")
            wants_stream = bool(body.get("stream"))

            http_referer_override = (valves.HTTP_REFERER_OVERRIDE or "").strip()
            referer_override_invalid = bool(
                http_referer_override
                and not http_referer_override.startswith(("http://", "https://"))
            )
            if referer_override_invalid and not wants_stream:
                await self._event_emitter_handler._emit_notification(
                    safe_event_emitter,
                    "HTTP_REFERER_OVERRIDE must be a full URL including http(s)://. "
                    "Falling back to the default pipe referer.",
                    level="warning",
                )

            if not self._circuit_breaker.allows(user_id):
                message = "Temporarily disabled due to repeated errors. Please retry later."
                if safe_event_emitter:
                    await self._event_emitter_handler._emit_notification(safe_event_emitter, message, level="warning")
                SessionLogger.cleanup()
                return message

            if self._warmup_failed:
                message = "Service unavailable due to startup issues"
                if safe_event_emitter:
                    await self._ensure_error_formatter()._emit_error(
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
                self.logger.error("Request queue not initialized after concurrency setup")
                if safe_event_emitter:
                    await self._ensure_error_formatter()._emit_error(
                        safe_event_emitter,
                        "Service temporarily unavailable",
                        show_error_message=True,
                        done=True,
                    )
                SessionLogger.cleanup()
                return "Service temporarily unavailable"

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
                            },
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
                    await self._ensure_error_formatter()._emit_error(
                        safe_event_emitter,
                        "Server busy (503)",
                        show_error_message=True,
                        done=True,
                    )
                SessionLogger.cleanup()
                return "Server busy (503)"
        except Exception as exc:
            self.logger.error("Pre-enqueue setup failed: %s", exc)
            if safe_event_emitter:
                try:
                    await self._ensure_error_formatter()._emit_error(
                        safe_event_emitter,
                        "Request setup failed. Please retry.",
                        show_error_message=True,
                        done=True,
                    )
                except Exception:
                    self.logger.debug("Failed to emit error during pre-enqueue recovery", exc_info=True)
            try:
                SessionLogger.cleanup()
            except Exception:
                self.logger.debug("SessionLogger.cleanup failed during pre-enqueue recovery", exc_info=True)
            return "Request setup failed. Please retry."

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
                await self._ensure_error_formatter()._emit_error(
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
        if self._artifact_store:
            self._artifact_store.close()
        self._session_log_manager.stop_workers()

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
                self._multimodal_handler.set_http_session(None)
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
                task.add_done_callback(_consume_background_task_exception)
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

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

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

    # =============================================================================
    # ORCHESTRATION METHODS
    # =============================================================================


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



    @timed
    def _enqueue_job(self, job: _PipeJob) -> bool:
        """Attempt to enqueue a request, returning False when the queue is full."""
        queue = self._request_queue
        if queue is None:
            self.logger.error("Request queue not initialized in _enqueue_job")
            return False
        try:
            queue.put_nowait(job)
            self.logger.debug("Enqueued request %s (depth=%s)", job.request_id, queue.qsize())
            return True
        except asyncio.QueueFull:
            self.logger.warning("Request queue full (max=%s)", queue.maxsize)
            return False


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
            self._event_emitter_handler._make_middleware_stream_emitter(job, stream_queue)
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
                tool_executor = self._ensure_tool_executor()
                for worker_idx in range(worker_count):
                    tool_context.workers.append(
                        asyncio.create_task(
                            tool_executor._tool_worker_loop(tool_context),
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
                self._circuit_breaker.reset(job.user_id)
        except Exception as exc:
            self._circuit_breaker.record_failure(job.user_id)
            if stream_queue is not None and not job.future.cancelled():
                self._event_emitter_handler._try_put_middleware_stream_nowait(
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
                self._event_emitter_handler._try_put_middleware_stream_nowait(stream_queue, None)
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
                            self._session_log_manager.persist_segment_to_db(
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
        SessionLogger.SESSION_LOG_MAX_LINES = job.valves.SESSION_LOG_MAX_LINES
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
        self._artifact_store._ensure_artifact_store(valves, pipe_identifier)

        task_name = TaskModelAdapter._task_name(__task__)
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
                await self._ensure_error_formatter()._emit_templated_error(
                    __event_emitter__,
                    template=template,
                    variables=variables,
                    log_message=f"Auth configuration error: {api_key_error}",
                    log_level=logging.WARNING,
                )
                return ""

            # Non-streaming: return a normal chat completion payload with the markdown.
            error_id, context_defaults = self._ensure_error_formatter()._build_error_context()
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
            await self._ensure_error_formatter()._emit_error(
                __event_emitter__,
                f"OpenRouter configuration error: {exc}",
                show_error_message=True,
                done=True,
            )
            return ""
        except Exception as exc:
            available_models = OpenRouterModelRegistry.list_models()
            if not available_models:
                await self._ensure_error_formatter()._emit_error(
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
            await self._ensure_error_formatter()._report_openrouter_error(
                e,
                event_emitter=__event_emitter__,
                normalized_model_id=body.get("model"),
                api_model_id=None,
            )
            return ""

        # Network timeouts
        except httpx.TimeoutException as e:
            await self._ensure_error_formatter()._emit_templated_error(
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
            await self._ensure_error_formatter()._emit_templated_error(
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
                await self._ensure_error_formatter()._emit_templated_error(
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
                await self._ensure_error_formatter()._report_openrouter_error(
                    error,
                    event_emitter=__event_emitter__,
                    normalized_model_id=body.get("model"),
                    api_model_id=None,
                )
                return ""

            raise

        # Generic catch-all
        except Exception as e:
            await self._ensure_error_formatter()._emit_templated_error(
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
        return await self._ensure_request_orchestrator().process_request(
            body, __user__, __request__, __event_emitter__, __event_call__, __metadata__, __tools__,
            __task__, __task_body__, valves, session, openwebui_model_id, pipe_identifier,
            allowlist_norm_ids, enforced_norm_ids, catalog_norm_ids, features, user_id=user_id
        )

    # Model Management

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

    # OpenRouter API Adapters

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
        chunk_queue_warn_size: int = 1000,
        event_queue_maxsize: int = 100,
        event_queue_warn_size: int = 1000,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._ensure_responses_adapter().send_openai_responses_streaming_request(
            session, request_body, api_key, base_url, valves=valves, workers=workers,
            breaker_key=breaker_key, delta_char_limit=delta_char_limit, idle_flush_ms=idle_flush_ms,
            chunk_queue_maxsize=chunk_queue_maxsize, chunk_queue_warn_size=chunk_queue_warn_size,
            event_queue_maxsize=event_queue_maxsize, event_queue_warn_size=event_queue_warn_size
        ):
            yield event

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
        async for event in self._ensure_chat_completions_adapter().send_openai_chat_completions_streaming_request(
            session, responses_request_body, api_key, base_url, valves=valves, breaker_key=breaker_key
        ):
            yield event

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
        return await self._ensure_chat_completions_adapter().send_openai_chat_completions_nonstreaming_request(
            session, responses_request_body, api_key, base_url, valves=valves, breaker_key=breaker_key
        )

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
        chunk_queue_warn_size: int = 1000,
        event_queue_maxsize: int = 100,
        event_queue_warn_size: int = 1000,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._ensure_chat_completions_adapter().send_openrouter_streaming_request(
            session, responses_request_body, api_key, base_url, valves=valves,
            endpoint_override=endpoint_override, workers=workers, breaker_key=breaker_key,
            delta_char_limit=delta_char_limit, idle_flush_ms=idle_flush_ms,
            chunk_queue_maxsize=chunk_queue_maxsize, chunk_queue_warn_size=chunk_queue_warn_size,
            event_queue_maxsize=event_queue_maxsize, event_queue_warn_size=event_queue_warn_size
        ):
            yield event

    async def _shutdown_tool_context(self, context: _ToolExecutionContext) -> None:
        """Stop per-request tool workers (bounded wait, then cancel)."""

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

    # Tool Execution Methods

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
                self._circuit_breaker.record_tool_failure(context.user_id, tool_type)
                if not item.future.done():
                    item.future.set_result(
                        self._ensure_tool_executor()._build_tool_output(
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
                payload = self._ensure_tool_executor()._build_tool_output(
                    item.call,
                    f"Tool error: {result}",
                    status="failed",
                )
            else:
                status, text, files, embeds = result
                payload = self._ensure_tool_executor()._build_tool_output(item.call, text, status=status, files=files, embeds=embeds)
                tool_type = (item.tool_cfg.get("type") or "function").lower()
                self._circuit_breaker.reset_tool(context.user_id, tool_type)
            item.future.set_result(payload)

    @timed
    async def _invoke_tool_call(
        self,
        item: _QueuedToolCall,
        context: _ToolExecutionContext,
    ) -> tuple[str, str, list[dict[str, Any]], list[str]]:
        """Invoke a single tool call with circuit breaker protection."""
        tool_type = (item.tool_cfg.get("type") or "function").lower()
        if not self._circuit_breaker.tool_allows(context.user_id, tool_type):
            await self._ensure_tool_executor()._notify_tool_breaker(context, tool_type, item.call.get("name"))
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
            self._circuit_breaker.record_tool_failure(context.user_id, tool_type)
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
                        await self._event_emitter_handler._emit_files(context.event_emitter, files)
                        timing_mark(f"tool_run:{tool_name}:files_emitted")
                    except Exception as emit_exc:
                        self.logger.debug("Failed to emit files for '%s': %s", tool_name, emit_exc)

                # Emit embeds if any were extracted
                if embeds and context.event_emitter:
                    try:
                        await self._event_emitter_handler._emit_embeds(context.event_emitter, embeds)
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
            from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_not_exception_type
        except ImportError:
            # Fallback without retries if tenacity not available
            try:
                timing_mark(f"tool_run:{tool_name}:executing_no_retry")
                result = await asyncio.wait_for(
                    self._call_tool_callable(fn_to_call, item.args),
                    timeout=timeout,
                )
                self._circuit_breaker.reset_tool(context.user_id, tool_type)
                text, files, embeds = await _process_and_emit(result)
                timing_mark(f"tool_run:{tool_name}:done")
                return ("completed", text, files, embeds)
            except Exception as exc:
                self._circuit_breaker.record_tool_failure(context.user_id, tool_type)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "Tool '%s' execution failed.",
                        tool_name,
                        exc_info=True,
                    )
                timing_mark(f"tool_run:{tool_name}:failed")
                return ("failed", f"Tool error: {exc}", [], [])

        retryer = AsyncRetrying(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=0.2, min=0.2, max=1),
            retry=(
                retry_if_exception_type(Exception)
                & retry_if_not_exception_type(asyncio.TimeoutError)
            ),
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
                    self._circuit_breaker.reset_tool(context.user_id, tool_type)
                    text, files, embeds = await _process_and_emit(result)
                    timing_mark(f"tool_run:{tool_name}:done")
                    return ("completed", text, files, embeds)
        except Exception as exc:
            self._circuit_breaker.record_tool_failure(context.user_id, tool_type)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Tool '%s' execution failed.",
                    tool_name,
                    exc_info=True,
                )
            timing_mark(f"tool_run:{tool_name}:failed")
            return ("failed", f"Tool error: {exc}", [], [])
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

    # 4.8 Internal Static Helpers

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
        return await self._ensure_responses_adapter().send_openai_responses_nonstreaming_request(
            session, request_body, api_key, base_url, valves=valves, breaker_key=breaker_key
        )

    # ADDITIONAL HELPER METHODS (called by orchestration methods)

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
                is_free = is_free_model(norm_id)
                if free_mode == "only" and not is_free:
                    continue
                if free_mode == "exclude" and is_free:
                    continue

            if tool_mode != "all":
                supports_tools = supports_tool_calling(norm_id)
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
            is_free = is_free_model(model_norm_id)
            if free_mode == "only" and not is_free:
                reasons.append("FREE_MODEL_FILTER=only")
            elif free_mode == "exclude" and is_free:
                reasons.append("FREE_MODEL_FILTER=exclude")

        tool_mode = (getattr(valves, "TOOL_CALLING_FILTER", "all") or "all").strip().lower()
        if tool_mode != "all" and model_norm_id in catalog_norm_ids:
            supports_tools = supports_tool_calling(model_norm_id)
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

    # 4.3 Core Multi-Turn Handlers
    @no_type_check
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
        if not _is_anthropic_model_id(model_id):
            return

        feature = "interleaved-thinking-2025-05-14"
        existing = headers.get("x-anthropic-beta") or headers.get("X-Anthropic-Beta") or ""
        values = [part.strip() for part in existing.split(",") if part.strip()] if existing else []
        if feature not in values:
            values.append(feature)
        if values:
            headers["x-anthropic-beta"] = ",".join(values)
        headers.pop("X-Anthropic-Beta", None)

    # 4.7 Emitters (Front-end communication)


    @classmethod
    @timed
    def _note_auth_failure(cls, *, ttl_seconds: Optional[int] = None) -> None:
        key = cls._auth_failure_scope_key()
        if not key:
            return
        CircuitBreaker.note_auth_failure(key, ttl_seconds=ttl_seconds)

    @classmethod
    @timed
    def _auth_failure_active(cls) -> bool:
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
