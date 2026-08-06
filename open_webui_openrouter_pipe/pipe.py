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
import concurrent.futures
import contextlib
import contextvars
import inspect
import json
import logging
import os
import secrets
import sys
import threading
import time
import uuid
import weakref
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, no_type_check

# Third-party imports
import aiohttp
import httpx
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    from open_webui.models.chats import Chats
except ImportError:
    Chats = None  # type: ignore
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.models.chats failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    Chats = None  # type: ignore
try:
    from open_webui.models.models import ModelForm, Models
except ImportError:
    ModelForm = None  # type: ignore
    Models = None  # type: ignore
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.models.models failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    ModelForm = None  # type: ignore
    Models = None  # type: ignore
try:
    from open_webui.models.files import Files
except ImportError:
    Files = None  # type: ignore
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.models.files failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    Files = None  # type: ignore

# Optional Redis support
try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore

# Timing instrumentation
from .core.timing_logger import timed, timing_mark
from .storage.persistence import _detect_redis_config, _RedisClient

try:
    import pyzipper  # pyright: ignore[reportMissingImports]
except ImportError:
    pyzipper = None  # type: ignore

# Import subsystems
from .core.circuit_breaker import CircuitBreaker
from .core.config import (
    _OPENROUTER_CATEGORIES,
    _OPENROUTER_REFERER,
    _OPENROUTER_TITLE,
    _PIPE_RUNTIME_ID,
    EncryptedStr,
    UserValves,
    Valves,
    _select_openrouter_http_referer,
    parse_user_valves,
)

# Import error handling
from .core.error_formatter import ErrorFormatter
from .core.errors import (
    OpenRouterAPIError,
    RequiredInternalFileError,
    _build_openrouter_api_error,
)
from .core.logging_system import SessionLogger, resolve_level
from .core.utils import (
    _apply_retry_after_metadata,
    _await_if_needed,
    _extract_feature_flags,
    _render_error_template,
)
from .core.warn_latch import warn_level

# Import vendor integrations
from .integrations.anthropic import _is_anthropic_model_id

# Import logging
from .logging.session_log_manager import SessionLogManager

# Import model management
from .models.catalog_manager import ModelCatalogManager
from .models.reasoning_config import ReasoningConfigManager
from .models.registry import (
    ModelFamily,
    OpenRouterModelRegistry,
    is_free_model,
    sanitize_model_id,
    supports_tool_calling,
)

# Import request handling
from .requests import NonStreamingAdapter, TaskModelAdapter
from .storage.multimodal import MultimodalHandler
from .storage.owui_files import OwuiFileGateway
from .storage.persistence import ArtifactStore
from .streaming.event_emitter import EventEmitter, EventEmitterHandler
from .streaming.streaming_core import StreamingHandler
from .tools.tool_executor import _QueuedToolCall, _ToolExecutionContext

if TYPE_CHECKING:
    from .api.gateway.chat_completions_adapter import ChatCompletionsAdapter
    from .api.gateway.responses_adapter import ResponsesAdapter
    from .filters import FilterManager
    from .integrations.video import VideoGenerationAdapter
    from .plugins.registry import PluginRegistry
    from .requests.orchestrator import RequestOrchestrator
    from .tools.tool_executor import ToolExecutor

ToolCallable = Callable[..., Awaitable[Any]] | Callable[..., Any]


def _consume_background_task_exception(task: asyncio.Task) -> None:
    """Silently consume exceptions from background tasks to avoid 'Task exception was never retrieved' warnings."""
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.exception()


_LIFECYCLE_REGISTRY_KEY = "_openrouter_pipe_lifecycle"


class _LifecycleRegistry:
    def __init__(self) -> None:
        self._current: dict[str, weakref.ref[Pipe]] = {}
        self._lock = threading.Lock()

    def swap_in(self, new: Pipe) -> Pipe | None:
        with self._lock:
            old_ref = self._current.get(new.id)
            self._current[new.id] = weakref.ref(new)
            return old_ref() if old_ref else None

    def current(self, pipe_id: str) -> Pipe | None:
        with self._lock:
            ref = self._current.get(pipe_id)
            return ref() if ref else None


def _fallback_tool_text(raw_result: Any) -> str:
    candidate: Any = raw_result
    if not isinstance(candidate, (str, list)):
        content_attr = getattr(raw_result, "content", None)
        if content_attr is not None:
            candidate = content_attr
    if isinstance(candidate, list):
        texts: list[str] = []
        for block in candidate:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    texts.append(block["text"])
            elif getattr(block, "type", None) == "text":
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str):
                    texts.append(block_text)
        if texts:
            return "\n".join(texts)
    return "" if raw_result is None else str(raw_result)


def _get_lifecycle_registry():
    reg = sys.modules.get(_LIFECYCLE_REGISTRY_KEY)
    if reg is None:
        reg = _LifecycleRegistry()
        sys.modules[_LIFECYCLE_REGISTRY_KEY] = reg  # type: ignore[assignment]
    return reg


_warned_plugin_dispatch: set[str] = set()
_warned_pipes_maintenance: set[str] = set()
_warned_user_valves: set[str] = set()
_warned_timing_file: set[str] = set()


# Data Classes

@dataclass(slots=True)
class _PipeJob:
    """Encapsulate a single OpenRouter request scheduled through the queue."""

    pipe: Pipe
    body: dict[str, Any]
    user: dict[str, Any]
    request: Request | None
    event_emitter: EventEmitter | None
    event_call: Callable[[dict[str, Any]], Awaitable[Any]] | None
    metadata: dict[str, Any]
    tools: list[dict[str, Any]] | dict[str, Any] | None
    task: dict[str, Any] | None
    task_body: dict[str, Any] | None
    valves: Pipe.Valves
    future: asyncio.Future
    stream_queue: asyncio.Queue[dict[str, Any] | str | None] | None = None
    request_id: str = field(default_factory=lambda: secrets.token_hex(8))
    # Parsed once at the entry point and carried, so the orchestrator does not repeat a
    # full user-row read -- and so one request sees ONE snapshot of the user's valves.
    user_valves: Pipe.UserValves | None = None
    rejected_user_valves: list[str] = field(default_factory=list)

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


# Main Pipe Class

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

    id: str = _PIPE_RUNTIME_ID or "open_webui_openrouter_pipe"

    Valves = Valves
    UserValves = UserValves

    _QUEUE_MAXSIZE = 1000
    _global_semaphore: asyncio.Semaphore | None = None
    _semaphore_limit: int = 0
    _tool_global_semaphore: asyncio.Semaphore | None = None
    _tool_global_limit: int = 0
    _video_global_semaphore: asyncio.Semaphore | None = None
    _video_global_limit: int = 0
    _TOOL_CONTEXT: ContextVar[_ToolExecutionContext | None] = ContextVar(
        "openrouter_tool_context",
        default=None,
    )
    _active_jobs: ClassVar[set[asyncio.Task[None]]] = set()

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
        self._draining: bool = False
        self._closing: bool = False
        self._close_lock: threading.Lock = threading.Lock()
        self._close_done: concurrent.futures.Future | None = None
        self._active_pipes_calls: int = 0

        if os.environ.get("OWUI_PIPE_TEST_MODE") == "1":
            self._init_minimal_for_tests()
            return

        self.type = "manifold"
        self.valves = self.Valves()
        self._refresh_process_log_level()
        self.logger = SessionLogger.get_logger(__name__.split(".")[0])

        self._http_session: aiohttp.ClientSession | None = None
        self._initialized = False
        self._closed = False
        self._shutdown_lock: asyncio.Lock | None = None

        self._request_queue: asyncio.Queue[_PipeJob] | None = None
        self._queue_worker_task: asyncio.Task | None = None
        self._queue_worker_lock: asyncio.Lock | None = None
        self._log_queue: asyncio.Queue[logging.LogRecord] | None = None
        self._log_queue_loop: asyncio.AbstractEventLoop | None = None
        self._log_worker_task: asyncio.Task | None = None
        self._log_worker_lock: asyncio.Lock | None = None
        self._log_worker_start_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

        pipe_id = getattr(self, "id", "openrouter")

        self._event_emitter_handler: EventEmitterHandler = EventEmitterHandler(
            logger=self.logger,
            valves=self.valves,
            pipe_instance=self,
        )

        self._artifact_store = ArtifactStore(
            pipe_id=pipe_id,
            logger=self.logger,
            valves=self.valves,
            emit_notification_callback=self._event_emitter_handler._emit_notification,
            tool_context_var=Pipe._TOOL_CONTEXT,
            user_id_context_var=SessionLogger.user_id,
        )

        self._file_gateway = OwuiFileGateway(logger=self.logger, valves=self.valves)
        self._multimodal_handler: MultimodalHandler = MultimodalHandler(
            logger=self.logger,
            valves=self.valves,
            http_session=None,
            artifact_store=None,
            emit_status_callback=None,
            file_gateway=self._file_gateway,
        )
        self._streaming_handler: StreamingHandler = StreamingHandler(
            logger=self.logger,
            valves=self.valves,
            model_registry=OpenRouterModelRegistry,
            pipe_instance=self,
        )
        self._catalog_manager: ModelCatalogManager | None = None
        self._error_formatter: ErrorFormatter | None = None
        self._reasoning_config_manager: ReasoningConfigManager | None = None
        self._nonstreaming_adapter: NonStreamingAdapter | None = None
        self._task_model_adapter: TaskModelAdapter | None = None
        self._tool_executor: ToolExecutor | None = None
        self._responses_adapter: ResponsesAdapter | None = None
        self._chat_completions_adapter: ChatCompletionsAdapter | None = None
        self._request_orchestrator: RequestOrchestrator | None = None
        self._filter_manager: FilterManager | None = None
        self._video_generation_adapter: VideoGenerationAdapter | None = None
        self._plugin_registry: PluginRegistry | None = None
        self._video_active_tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._video_active_tasks_dict_lock: asyncio.Lock = asyncio.Lock()
        self._video_message_locks_dict_lock: asyncio.Lock = asyncio.Lock()
        self._video_user_locks_dict_lock: asyncio.Lock = asyncio.Lock()
        self._video_user_locks: dict[str, asyncio.Lock] = {}
        self._video_user_active_counts: dict[str, int] = {}
        self._video_user_active_jobs: dict[str, set[str]] = {}
        self._video_message_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._video_message_lock_refs: dict[tuple[str, str], int] = {}

        self._circuit_breaker = CircuitBreaker(
            threshold=self.valves.BREAKER_MAX_FAILURES,
            window_seconds=self.valves.BREAKER_WINDOW_SECONDS,
        )
        self._artifact_store.configure_breaker(
            threshold=self.valves.BREAKER_MAX_FAILURES,
            window_seconds=self.valves.BREAKER_WINDOW_SECONDS,
        )

        self._stale_filter_ids_pruned = False

        # Startup check coordination
        self._startup_task: asyncio.Task | None = None
        self._startup_checks_started = False
        self._startup_checks_pending = False
        self._startup_checks_complete = False
        self._warmup_failed = False

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

        self._session_log_manager = SessionLogManager(
            logger=self.logger,
            pipe=self,
            artifact_store=self._artifact_store,
        )
        self._maybe_start_log_worker()

        # Configure timing file if enabled
        if self._maybe_configure_timing_file(reopen=True):
            self.logger.info("Timing log enabled: %s", self.valves.TIMING_LOG_FILE)

        self._maybe_start_startup_checks()

        self.logger.debug(
            "Pipe initialized (subsystem delegation: ArtifactStore, MultimodalHandler, StreamingHandler, EventEmitterHandler)"
        )

        self._attach_to_lifecycle_registry()

    @timed
    async def _ensure_async_subsystems_initialized(self):
        if self._initialized:
            return

        if not self._http_session:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
            self._http_session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
            )

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

    # LIFECYCLE & STARTUP HELPERS

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

        if self._log_worker_lock is not None:
            try:
                lock_loop = getattr(cast(Any, self._log_worker_lock), "_get_loop", lambda: None)()
                if lock_loop is not loop:
                    self._log_worker_lock = None
            except RuntimeError:
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

        pipe_ref = weakref.ref(self)

        @timed
        async def _ensure_worker() -> None:
            pipe = pipe_ref()
            if pipe is None or getattr(pipe, "_closed", False):
                return
            try:
                async with pipe._log_worker_lock:  # type: ignore[arg-type]
                    if getattr(pipe, "_closed", False):
                        return
                    if pipe._log_worker_task and not pipe._log_worker_task.done():
                        return
                    if pipe._log_queue is None:
                        pipe._log_queue = asyncio.Queue(maxsize=1000)
                        SessionLogger.set_log_queue(pipe._log_queue)
                    pipe._log_worker_task = loop.create_task(
                        Pipe._log_worker_loop(pipe._log_queue),
                        name="openrouter-log-worker",
                    )
            except Exception:
                pipe.logger.debug("Log worker startup task failed", exc_info=True)

        prev_start_task = self._log_worker_start_task
        if prev_start_task and not prev_start_task.done():
            prev_start_task.cancel()
        self._log_worker_start_task = loop.create_task(_ensure_worker(), name="openrouter-log-worker-start")
        self._log_worker_start_task.add_done_callback(_consume_background_task_exception)

    async def _stored_user_valves(self, __user__: dict[str, Any]) -> Any:
        """The user's valves as STORED, because Open WebUI destroys them on the way in.

        `functions.py` builds `UserValves(**stored)` inside a try/except and substitutes a
        default-constructed instance when pydantic refuses -- so a single stale value for
        one Literal field, left behind by any past option rename, silently collapses EVERY
        setting the user has, including REQUEST_ZDR. By the time the pipe is called, which
        field failed and what the others held is already gone: `parse_user_valves` gets an
        instance whose fields all equal their defaults and cannot tell that from a user
        who set nothing.

        Reading the row ourselves is the only place the per-field information still
        exists. The instance Open WebUI supplied is the fallback for when that read is
        unavailable or fails.
        """
        supplied = __user__.get("valves")
        user_id = str(__user__.get("id") or "")
        if not user_id:
            return supplied
        try:
            from open_webui.models.functions import Functions

            reader = getattr(Functions, "get_user_valves_by_id_and_user_id", None)
            if reader is None:
                # An Open WebUI whose Functions API does not offer this. Nothing to
                # recover, and no diagnostic: the supplied instance is the documented
                # input, not a failure -- so this is readable, not unreadable.
                return supplied
            # `_await_if_needed` rather than a bare await: this reader is `async` in
            # every Open WebUI the manifest supports that I can check, but the floor is
            # 0.9.1 and a sync one would raise TypeError here -- which lands in the
            # except below and silently swaps every setting the user has for its
            # default. The tolerant call keeps a sync reader from doing that.
            stored = await _await_if_needed(reader(self.id, user_id))
        except Exception:
            self.logger.log(
                warn_level(_warned_user_valves, "stored_read"),
                "Could not read the stored user valves; falling back to what Open WebUI "
                "supplied, which cannot report a field it failed to parse",
                exc_info=True,
            )
            return supplied
        return stored if isinstance(stored, Mapping) else supplied

    def _user_valve_blob_is_unreadable(self, __user__: dict[str, Any], stored: Any) -> bool:
        """True when this user HAS a saved valve blob for this pipe that did not decode.

        `decrypt_valves` returns `{}` on InvalidToken exactly as it does for a row with
        nothing stored, so the decoded value cannot tell a user who never opened the
        valve panel from one whose settings a WEBUI_SECRET_KEY rotation made unreadable.
        The still-encrypted blob can: Open WebUI builds `__user__` as
        `UserModel.model_dump()` and `UserSettings` allows extra keys, so the ciphertext
        travels on `settings.functions.valves.<pipe id>` and no second user-row fetch is
        needed to tell the two apart.

        A non-empty STRING is the only positive evidence. With valve encryption off the
        column holds a plain dict and `decrypt_valves` returns it unchanged, so an empty
        decode there really is an empty row. A blob under some other pipe id reads as
        absent, which degrades to the previous behaviour rather than enforcing ZDR on a
        user who never asked for it.
        """
        if stored != {}:
            return False
        settings = __user__.get("settings")
        if not isinstance(settings, Mapping):
            return False
        functions = settings.get("functions")
        if not isinstance(functions, Mapping):
            return False
        valves = functions.get("valves")
        if not isinstance(valves, Mapping):
            return False
        blob = valves.get(self.id)
        return isinstance(blob, str) and bool(blob.strip())

    async def _read_user_valves(self, __user__: dict[str, Any]) -> tuple[Any, list[str]]:
        """The one place a request turns `__user__` into (UserValves, rejected).

        One return statement, on purpose. An earlier version of this pair returned a
        2-tuple from one of four paths inside `_stored_user_valves` and a scalar from
        the other three; `parse_user_valves` recognised the tuple as neither a model nor
        a Mapping and every user valve silently reverted to its default.
        `_stored_user_valves` keeps its single-value contract and the classification
        happens here.

        A blob that will not decode is reported as EVERY field rejected, because that is
        what happened: the row was read and nothing in it could be recovered. Callers
        that fail closed on a specific field -- the orchestrator's
        `"REQUEST_ZDR" in rejected` -- then do so with no new signal to thread through
        the job.
        """
        stored = await self._stored_user_valves(__user__)
        user_valves, rejected = parse_user_valves(stored, model=self.UserValves)
        if self._user_valve_blob_is_unreadable(__user__, stored):
            self.logger.log(
                warn_level(_warned_user_valves, "undecodable_blob"),
                "The stored user valves did not decode (a rotated WEBUI_SECRET_KEY does "
                "this); every setting is falling back to its default and preferences "
                "that fail closed will do so",
            )
            rejected = sorted(set(rejected) | set(self.UserValves.model_fields))
        return user_valves, rejected

    def _maybe_configure_timing_file(self, *, reopen: bool = False) -> bool:
        """Open the timing log file when ENABLE_TIMING_LOG is on, warning once per bad path.

        ``reopen`` forces a close and re-open, which is what a fresh pipe load needs so an
        externally rotated or deleted log file is not written to through a stale handle.

        The latch is keyed on the path and cleared on success, so a corrected path warns
        again if it breaks again, and a still-broken path repeats at DEBUG rather than
        going silent at every level.
        """
        if not self.valves.ENABLE_TIMING_LOG:
            _warned_timing_file.clear()
            return False

        from .core.timing_logger import (
            configure_timing_file,
            ensure_timing_file_configured,
        )

        timing_path = self.valves.TIMING_LOG_FILE
        opener = configure_timing_file if reopen else ensure_timing_file_configured
        if opener(timing_path):
            _warned_timing_file.clear()
            return True
        self.logger.log(
            warn_level(_warned_timing_file, str(timing_path)),
            "Failed to open timing log file: %s. ENABLE_TIMING_LOG is on but no timing "
            "data will be recorded until TIMING_LOG_FILE points at a writable path.",
            timing_path,
        )
        return False

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
        client: _RedisClient | None = None
        try:
            client = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
                health_check_interval=30,
            )
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
            store = getattr(self, "_artifact_store", None)
            if store is not None:
                store._redis_enabled = False
                store._redis_client = None
            self.logger.warning("Redis cache disabled (%s)", exc, exc_info=True)
            return

        self._redis_client = client
        self._redis_enabled = True
        store = getattr(self, "_artifact_store", None)
        if store is not None:
            store._redis_client = client
            store._redis_enabled = True
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

    def _ensure_tool_executor(self) -> ToolExecutor:
        if self._tool_executor is None:
            from .tools.tool_executor import ToolExecutor
            self._tool_executor = ToolExecutor(
                pipe=self,
                logger=self.logger,
            )
        return self._tool_executor

    def _ensure_responses_adapter(self) -> ResponsesAdapter:
        if self._responses_adapter is None:
            from .api.gateway.responses_adapter import ResponsesAdapter
            self._responses_adapter = ResponsesAdapter(
                pipe=self,
                logger=self.logger,
            )
        return self._responses_adapter

    def _ensure_chat_completions_adapter(self) -> ChatCompletionsAdapter:
        if self._chat_completions_adapter is None:
            from .api.gateway.chat_completions_adapter import ChatCompletionsAdapter
            self._chat_completions_adapter = ChatCompletionsAdapter(
                pipe=self,
                logger=self.logger,
            )
        return self._chat_completions_adapter

    def _ensure_request_orchestrator(self) -> RequestOrchestrator:
        if self._request_orchestrator is None:
            from .requests.orchestrator import RequestOrchestrator
            self._request_orchestrator = RequestOrchestrator(
                pipe=self,
                logger=self.logger,
            )
        return self._request_orchestrator

    def _ensure_filter_manager(self) -> FilterManager:
        if self._filter_manager is None:
            from .filters import FilterManager
            self._filter_manager = FilterManager(
                pipe=self,
                valves=self.valves,
                logger=self.logger,
            )
        return self._filter_manager

    def _ensure_video_generation_adapter(self) -> VideoGenerationAdapter:
        if self._video_generation_adapter is None:
            from .integrations.video import VideoGenerationAdapter
            self._video_generation_adapter = VideoGenerationAdapter(
                pipe=self,
                logger=self.logger,
            )
        return self._video_generation_adapter

    async def _dispatch_plugin_event(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Guarded plugin-event dispatch; never raises into the request path.

        No-op when the plugin system is disabled or the registry was never
        created — deliberately does NOT force lazy registry creation.
        """
        try:
            if not getattr(self.valves, "ENABLE_PLUGIN_SYSTEM", False):
                return
            registry = self._plugin_registry
            if registry is None:
                return
            await getattr(registry, method)(*args, **kwargs)
        except Exception as exc:
            level = warn_level(_warned_plugin_dispatch, f"{method}:{type(exc).__name__}")
            self.logger.log(level, "Plugin event %s dispatch failed", method, exc_info=True)

    def _ensure_plugin_registry(self) -> PluginRegistry:
        if self._plugin_registry is None:
            from .plugins.registry import PluginRegistry
            self._plugin_registry = PluginRegistry()
            self._plugin_registry.init_plugins(self)
        return self._plugin_registry

    # ENTRY POINTS

    @timed
    async def pipes(self):
        """Return the list of models exposed to Open WebUI."""
        self._refresh_process_log_level()
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
                if self.valves.ENABLE_VIDEO_GENERATION:
                    from .integrations.video_catalog import ensure_video_catalog_loaded

                    await ensure_video_catalog_loaded(
                        session,
                        valves=self.valves,
                        api_key=api_key_value,
                        logger=self.logger,
                        cache_seconds=self.valves.MODEL_CATALOG_REFRESH_SECONDS,
                    )
                if self.valves.ENABLE_OPENROUTER_IMAGE_GENERATION:
                    from .integrations.image_catalog import ensure_image_catalog_loaded

                    await ensure_image_catalog_loaded(
                        session,
                        valves=self.valves,
                        api_key=api_key_value,
                        logger=self.logger,
                        cache_seconds=self.valves.MODEL_CATALOG_REFRESH_SECONDS,
                    )
        except ValueError as exc:
            refresh_error = exc
            self.logger.exception("OpenRouter configuration error")
        except Exception as exc:
            refresh_error = exc
            level = warn_level(_warned_pipes_maintenance, f"catalog_refresh:{type(exc).__name__}")
            self.logger.log(level, "OpenRouter catalog refresh failed: %s", exc, exc_info=True)
        finally:
            await session.close()

        available_models = OpenRouterModelRegistry.list_models()
        if refresh_error and available_models:
            level = warn_level(_warned_pipes_maintenance, f"catalog_cached:{type(refresh_error).__name__}")
            self.logger.log(level, "Serving %d cached OpenRouter model(s) due to refresh failure.", len(available_models))
        if refresh_error and not available_models:
            return []

        try:
            from open_webui.models.functions import Functions as _Funcs
            old_ors = await _Funcs.get_function_by_id("openrouter_search")
            if old_ors and getattr(old_ors, "is_active", False):
                content = getattr(old_ors, "content", "") or ""
                if "openrouter_pipe:ors_filter:" in content:
                    await _Funcs.update_function_by_id("openrouter_search", {"is_active": False})
                    self.logger.info("Disabled old OpenRouter Search filter (replaced by OpenRouter Web Tools)")
        except Exception:
            self.logger.debug("Old OpenRouter Search filter cleanup failed", exc_info=True)

        all_web_tools_disabled = not (
            self.valves.ENABLE_WEB_SEARCH
            or self.valves.ENABLE_WEB_FETCH
            or self.valves.ENABLE_DATETIME
            or self.valves.ENABLE_ADVISOR
            or self.valves.ENABLE_SUBAGENT
            or self.valves.ENABLE_SEARCH_MODELS
        )
        if self.valves.AUTO_INSTALL_WEB_TOOLS_FILTER and not all_web_tools_disabled:
            try:
                await self._ensure_filter_manager().ensure_openrouter_web_tools_filter_function_id(
                    enable_web_search=self.valves.ENABLE_WEB_SEARCH,
                    enable_web_fetch=self.valves.ENABLE_WEB_FETCH,
                    enable_datetime=self.valves.ENABLE_DATETIME,
                    enable_advisor=self.valves.ENABLE_ADVISOR,
                    enable_subagent=self.valves.ENABLE_SUBAGENT,
                    enable_search_models=self.valves.ENABLE_SEARCH_MODELS,
                )
            except Exception as exc:
                level = warn_level(_warned_pipes_maintenance, f"web_tools:{type(exc).__name__}")
                self.logger.log(level, "AUTO_INSTALL_WEB_TOOLS_FILTER failed: %s", exc, exc_info=True)
        elif all_web_tools_disabled:
            try:
                from open_webui.models.functions import Functions as _Funcs
                wt = await _Funcs.get_function_by_id("openrouter_web_tools")
                if wt and getattr(wt, "is_active", False):
                    await _Funcs.update_function_by_id("openrouter_web_tools", {"is_active": False})
                    self.logger.info("Disabled OpenRouter Web Tools filter (all tools disabled)")
            except Exception:
                self.logger.debug("Disabling OpenRouter Web Tools filter failed", exc_info=True)
        if self.valves.ENABLE_OPENROUTER_FUSION and self.valves.AUTO_INSTALL_FUSION_FILTER:
            try:
                await self._ensure_filter_manager().ensure_openrouter_fusion_filter_function_id()
            except Exception as exc:
                level = warn_level(_warned_pipes_maintenance, f"fusion:{type(exc).__name__}")
                self.logger.log(level, "AUTO_INSTALL_FUSION_FILTER failed: %s", exc, exc_info=True)
        elif not self.valves.ENABLE_OPENROUTER_FUSION:
            try:
                from open_webui.models.functions import Functions as _Funcs
                ff = await _Funcs.get_function_by_id("openrouter_fusion")
                if ff and getattr(ff, "is_active", False):
                    await _Funcs.update_function_by_id("openrouter_fusion", {"is_active": False})
                    self.logger.info("Disabled OpenRouter Fusion filter (ENABLE_OPENROUTER_FUSION=False)")
            except Exception:
                self.logger.debug("Disabling OpenRouter Fusion filter failed", exc_info=True)
        if self.valves.AUTO_INSTALL_IMAGE_GEN_FILTER and self.valves.ENABLE_IMAGE_GENERATION:
            try:
                await self._ensure_filter_manager().ensure_openrouter_image_gen_filter_function_id()
            except Exception as exc:
                level = warn_level(_warned_pipes_maintenance, f"image_gen:{type(exc).__name__}")
                self.logger.log(level, "AUTO_INSTALL_IMAGE_GEN_FILTER failed: %s", exc, exc_info=True)
        elif not self.valves.ENABLE_IMAGE_GENERATION:
            try:
                from open_webui.models.functions import Functions as _Funcs
                ig = await _Funcs.get_function_by_id("openrouter_image_gen")
                if ig and getattr(ig, "is_active", False):
                    await _Funcs.update_function_by_id("openrouter_image_gen", {"is_active": False})
                    self.logger.info("Disabled OpenRouter Image Generation filter (ENABLE_IMAGE_GENERATION=False)")
            except Exception:
                self.logger.debug("Disabling OpenRouter Image Generation filter failed", exc_info=True)
        if self.valves.AUTO_INSTALL_VIDEO_FILTERS and self.valves.ENABLE_VIDEO_GENERATION:
            try:
                await self._ensure_filter_manager().ensure_openrouter_video_gen_filter_function_ids(available_models)
            except Exception as exc:
                level = warn_level(_warned_pipes_maintenance, f"video:{type(exc).__name__}")
                self.logger.log(level, "AUTO_INSTALL_VIDEO_FILTERS per-model failed: %s", exc, exc_info=True)
        elif not self.valves.ENABLE_VIDEO_GENERATION:
            try:
                from open_webui.models.functions import Functions as _Funcs
                vg = await _Funcs.get_function_by_id("openrouter_video_gen")
                if vg and getattr(vg, "is_active", False):
                    await _Funcs.update_function_by_id("openrouter_video_gen", {"is_active": False})
                    self.logger.info("Disabled OpenRouter Video Generation filter (ENABLE_VIDEO_GENERATION=False)")
            except Exception:
                self.logger.debug("Disabling OpenRouter Video Generation filter failed", exc_info=True)
        try:
            from open_webui.models.functions import Functions as _Funcs
            legacy = await _Funcs.get_function_by_id("openrouter_video_openrouter_video")
            if legacy is not None:
                await _Funcs.delete_function_by_id("openrouter_video_openrouter_video")
                self.logger.info(
                    "Removed legacy generic OpenRouter Video Generation filter row 'openrouter_video_openrouter_video'"
                )
        except Exception as exc:
            self.logger.debug("Legacy video filter cleanup failed: %s", exc, exc_info=True)
        if self.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER:
            try:
                await self._ensure_filter_manager().ensure_direct_uploads_filter_function_id()
            except Exception as exc:
                level = warn_level(_warned_pipes_maintenance, f"direct_uploads:{type(exc).__name__}")
                self.logger.log(level, "AUTO_INSTALL_DIRECT_UPLOADS_FILTER failed: %s", exc, exc_info=True)

        selected_models = self._select_models(self.valves.MODEL_ID, available_models)
        selected_models = self._apply_model_filters(selected_models, self.valves)
        selected_models = self._expand_variant_models(selected_models, self.valves)

        admin_routing = (self.valves.ADMIN_PROVIDER_ROUTING_MODELS or "").strip()
        user_routing = (self.valves.USER_PROVIDER_ROUTING_MODELS or "").strip()
        if admin_routing or user_routing:
            try:
                catalog_mgr = self._ensure_catalog_manager()
                provider_map = catalog_mgr.get_cached_provider_map()
                if provider_map:
                    await self._ensure_filter_manager().ensure_provider_routing_filters(
                        admin_routing,
                        user_routing,
                        provider_map,
                        selected_models,
                        self.id,
                    )
            except Exception as exc:
                level = warn_level(_warned_pipes_maintenance, f"provider_routing:{type(exc).__name__}")
                self.logger.log(level, "Provider routing filter creation failed: %s", exc, exc_info=True)

        if not self._stale_filter_ids_pruned:
            self._stale_filter_ids_pruned = True
            try:
                count = await self._ensure_catalog_manager().prune_stale_openrouter_filter_ids()
                if count:
                    self.logger.info(
                        "Pruned stale openrouter_* filter IDs from %d model(s) on startup.", count
                    )
            except Exception as exc:
                level = warn_level(_warned_pipes_maintenance, f"stale_prune:{type(exc).__name__}")
                self.logger.log(level, "Startup stale filter ID pruning failed: %s", exc, exc_info=True)

        self._ensure_catalog_manager().maybe_schedule_model_metadata_sync(
            selected_models,
            pipe_identifier=self.id,
        )

        if self.valves.ENABLE_PLUGIN_SYSTEM:
            try:
                await self._ensure_plugin_registry().dispatch_on_models(selected_models)
            except Exception:
                level = warn_level(_warned_pipes_maintenance, "on_models")
                self.logger.log(level, "Plugin on_models dispatch failed", exc_info=True)
        return [
            {"id": m["id"], "name": m.get("name", m["id"])}
            for m in selected_models
            if isinstance(m, dict) and "id" in m
        ]

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
        if self._draining or self._closing:
            raise RuntimeError(
                "This pipe instance has been superseded by a newer version; please retry."
            )
        self._active_pipes_calls += 1
        counter_transferred = False
        try:
            result = await self._pipe_impl(
                body,
                __user__,
                __request__,
                __event_emitter__,
                __event_call__,
                __metadata__,
                __tools__,
                __task__,
                __task_body__,
            )
            if inspect.isasyncgen(result):
                counter_transferred = True
                state = {"released": False}
                wrapped = self._wrap_stream_with_counter_release(result, state)
                weakref.finalize(
                    wrapped,
                    Pipe._release_stream_counter,
                    self,
                    state,
                )
                return wrapped
            return result
        finally:
            if not counter_transferred:
                self._active_pipes_calls = max(0, self._active_pipes_calls - 1)
                self._maybe_trigger_drain_close()

    @staticmethod
    def _release_stream_counter(pipe: Pipe, state: dict) -> None:
        if state.get("released"):
            return
        state["released"] = True
        pipe._active_pipes_calls = max(0, pipe._active_pipes_calls - 1)
        pipe._maybe_trigger_drain_close()

    async def _wrap_stream_with_counter_release(
        self,
        inner: AsyncGenerator[dict[str, Any] | str, None],
        state: dict,
    ) -> AsyncGenerator[dict[str, Any] | str, None]:
        try:
            async for item in inner:
                yield item
        finally:
            Pipe._release_stream_counter(self, state)

    @timed
    async def _pipe_impl(
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
            from .core.timing_logger import set_timing_context, timing_mark
            _early_request_id = secrets.token_hex(8)
            self._maybe_configure_timing_file()
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
            user_valves, rejected_user_valves = await self._read_user_valves(__user__)
            for name in rejected_user_valves:
                level = warn_level(_warned_user_valves, name)
                self.logger.log(
                    level,
                    "User valve %s could not be read and is using its default",
                    name,
                )
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

            if self._warmup_failed and (self._startup_task is None or self._startup_task.done()):
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
                event_emitter=safe_event_emitter,
                event_call=__event_call__,
                metadata=__metadata__,
                tools=__tools__,
                task=__task__,
                task_body=__task_body__,
                valves=valves,
                user_valves=user_valves,
                rejected_user_valves=rejected_user_valves,
                future=future,
                stream_queue=stream_queue,
                request_id=_early_request_id,
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
        except Exception:
            self.logger.exception("Pre-enqueue setup failed")
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
                            except TimeoutError:
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
            self.logger.exception("Pipe request failed (request_id=%s)", job.request_id)
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
        cancelled_tasks: list[asyncio.Task] = []
        for attr in ("_redis_listener_task", "_redis_flush_task", "_redis_ready_task"):
            task = getattr(self, attr, None)
            if task and not task.done():
                task.cancel()
                cancelled_tasks.append(task)
            setattr(self, attr, None)

        if cancelled_tasks:
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            same_loop = [t for t in cancelled_tasks if running_loop is not None and t.get_loop() is running_loop]
            if same_loop:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        asyncio.gather(*same_loop, return_exceptions=True),
                        timeout=2.0,
                    )

        if self._redis_client:
            try:
                await self._redis_client.close()
            except Exception as e:
                self.logger.debug(f"Failed to close Redis client: {e}", exc_info=True)
            finally:
                self._redis_client = None

        # Update state
        self._redis_enabled = False
        store = getattr(self, "_artifact_store", None)
        if store is not None:
            store._redis_enabled = False
            store._redis_client = None

    def _init_minimal_for_tests(self) -> None:
        self.type = "manifold"
        self.logger = SessionLogger.get_logger(__name__.split(".")[0])
        self._http_session = None
        self._initialized = False
        self._closed = False
        self._shutdown_lock = None
        self._request_queue = None
        self._queue_worker_task = None
        self._queue_worker_lock = None
        self._log_queue = None
        self._log_queue_loop = None
        self._log_worker_task = None
        self._log_worker_lock = None
        self._log_worker_start_task = None
        self._cleanup_task = None
        self._startup_task = None
        self._startup_checks_started = False
        self._startup_checks_pending = False
        self._startup_checks_complete = False
        self._warmup_failed = False
        self._redis_url = None
        self._websocket_manager = None
        self._websocket_redis_url = None
        self._redis_candidate = False
        self._redis_enabled = False
        self._redis_client = None
        self._redis_listener_task = None
        self._redis_flush_task = None
        self._redis_ready_task = None
        self._video_active_tasks: dict[tuple[str, str], asyncio.Task] = {}

    def _attach_to_lifecycle_registry(self) -> None:
        registry = _get_lifecycle_registry()
        try:
            predecessor = registry.swap_in(self)
        except Exception:
            self.logger.warning("hot-reload: lifecycle registry swap_in failed", exc_info=True)
            return
        if predecessor is None:
            return
        self.logger.info(
            "hot-reload: sent close to predecessor (id=%s, predecessor_active_calls=%d)",
            getattr(predecessor, "id", "?"),
            getattr(predecessor, "_active_pipes_calls", -1),
        )
        try:
            predecessor.close_when_idle()
        except Exception:
            self.logger.warning("hot-reload: predecessor close_when_idle failed", exc_info=True)

    def _schedule_close(self) -> None:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        target = running or getattr(self, "_log_queue_loop", None)
        if target is None:
            if hasattr(self, "logger"):
                self.logger.debug("Close scheduling skipped: no loop available")
            return
        try:
            if target.is_closed():
                if hasattr(self, "logger"):
                    self.logger.debug("Close scheduling skipped: loop already closed")
                return
        except Exception:
            self.logger.debug("Close scheduling skipped: loop state check failed", exc_info=True)
            return
        if target is running:
            try:
                task = target.create_task(self.close(), name="openrouter-pipe-drain-close")
                task.add_done_callback(_consume_background_task_exception)
            except RuntimeError:
                if hasattr(self, "logger"):
                    self.logger.debug("create_task close scheduling failed", exc_info=True)
        else:
            try:
                asyncio.run_coroutine_threadsafe(self.close(), target)
            except Exception:
                if hasattr(self, "logger"):
                    self.logger.debug("run_coroutine_threadsafe close scheduling failed", exc_info=True)

    def close_when_idle(self) -> None:
        if self._closing or self._draining:
            return
        print(
            f"[openrouter pid={os.getpid()} id={getattr(self, 'id', '?')}] "
            f"hot-reload: close received (active_calls={self._active_pipes_calls})",
            file=sys.stderr,
            flush=True,
        )
        self._draining = True
        if self._active_pipes_calls <= 0:
            self._schedule_close()

    def _maybe_trigger_drain_close(self) -> None:
        if not self._draining or self._closing:
            return
        if self._active_pipes_calls > 0:
            return
        self._schedule_close()

    def shutdown(self) -> list[Any]:
        pending: list[Any] = []
        plugin_registry = getattr(self, "_plugin_registry", None)
        if plugin_registry is not None:
            pending = plugin_registry.dispatch_on_shutdown()
        cleanup_task = getattr(self, "_cleanup_task", None)
        if cleanup_task is not None and not cleanup_task.done():
            cleanup_task.cancel()
        artifact_store = getattr(self, "_artifact_store", None)
        if artifact_store:
            artifact_store.close()
        session_log_manager = getattr(self, "_session_log_manager", None)
        if session_log_manager:
            session_log_manager.stop_workers()
        return pending

    @timed
    async def _stop_request_worker(self) -> None:
        """Stop this instance's queue worker and drain pending items."""
        worker = self._queue_worker_task
        if worker:
            worker.cancel()
            try:
                worker_loop = worker.get_loop()
            except AttributeError:  # pragma: no cover - defensive for older asyncio implementations
                worker_loop = None
            if worker_loop is None or worker_loop is asyncio.get_running_loop():
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
            else:
                self.logger.debug(
                    "Skipping await for request worker bound to a different event loop during close()."
                )
            self._queue_worker_task = None
        self._request_queue = None

    @timed
    async def _stop_log_worker(self) -> None:
        """Stop this instance's log worker and clear the queue."""
        owned_queue = self._log_queue
        owned_loop = self._log_queue_loop
        worker = self._log_worker_task
        if worker:
            worker.cancel()
            try:
                worker_loop = worker.get_loop()
            except AttributeError:  # pragma: no cover - defensive for older asyncio implementations
                worker_loop = None
            if worker_loop is None or worker_loop is asyncio.get_running_loop():
                try:
                    with contextlib.suppress(asyncio.CancelledError):
                        await worker
                except RuntimeError as exc:
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
        try:
            from .core.logging_system import SessionLogger
            if owned_queue is not None and SessionLogger.log_queue is owned_queue:
                SessionLogger.set_log_queue(None)
            if owned_loop is not None and getattr(SessionLogger, "_main_loop", None) is owned_loop:
                SessionLogger.set_main_loop(None)
        except Exception:
            self.logger.debug("Releasing global log queue/loop references failed", exc_info=True)

    @timed
    async def _stop_video_tasks(self) -> None:
        tasks = list(getattr(self, "_video_active_tasks", {}).values())
        self._video_active_tasks.clear()
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                if task.get_loop() is asyncio.get_running_loop():
                    await task

    @timed
    async def close(self):
        """Shutdown background resources (DB executor, queue worker, log worker, Redis)."""
        lock = getattr(self, "_close_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._close_lock = lock
        with lock:
            already_closing = getattr(self, "_closing", False)
            if already_closing:
                existing = getattr(self, "_close_done", None)
            else:
                self._closing = True
                self._closed = True
                self._close_done = concurrent.futures.Future()
                existing = None
        if already_closing:
            if existing is not None and not existing.done():
                with contextlib.suppress(Exception):
                    await asyncio.wrap_future(existing)
            return
        print(
            f"[openrouter pid={os.getpid()} id={getattr(self, 'id', '?')}] "
            f"hot-reload: closing",
            file=sys.stderr,
            flush=True,
        )
        try:
            await self._do_close()
        finally:
            done = self._close_done
            if done is not None and not done.done():
                with contextlib.suppress(Exception):
                    done.set_result(None)

    async def _do_close(self) -> None:
        extra_tasks: list[asyncio.Task] = []
        startup = getattr(self, "_startup_task", None)
        if startup and not startup.done():
            startup.cancel()
            extra_tasks.append(startup)
        self._startup_task = None

        log_start = getattr(self, "_log_worker_start_task", None)
        if log_start and not log_start.done():
            log_start.cancel()
            extra_tasks.append(log_start)
        self._log_worker_start_task = None

        catalog = getattr(self, "_catalog_manager", None)
        catalog_sync = getattr(catalog, "_model_metadata_sync_task", None) if catalog else None
        if catalog_sync and not catalog_sync.done():
            catalog_sync.cancel()
            extra_tasks.append(catalog_sync)
        if catalog is not None:
            with contextlib.suppress(Exception):
                catalog._model_metadata_sync_task = None

        if extra_tasks:
            with contextlib.suppress(Exception):
                await asyncio.gather(*extra_tasks, return_exceptions=True)

        pending_shutdown: list[Any] = []
        with contextlib.suppress(Exception):
            pending_shutdown = self.shutdown() or []
        if pending_shutdown:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    asyncio.gather(*pending_shutdown, return_exceptions=True),
                    timeout=5.0,
                )
        await self._stop_video_tasks()
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

        catalog_mgr = getattr(self, "_catalog_manager", None)
        if catalog_mgr is not None:
            with contextlib.suppress(Exception):
                catalog_mgr._pipe = None  # type: ignore[assignment]
        slm = getattr(self, "_session_log_manager", None)
        if slm is not None:
            with contextlib.suppress(Exception):
                slm._pipe = None  # type: ignore[assignment]

    def __del__(self) -> None:
        """Best-effort cleanup hook for garbage collection."""
        if getattr(self, "_closed", False):
            return
        self.shutdown()
        self._schedule_close()

    # UTILITY METHODS

    @staticmethod
    @timed
    def _should_warn_event_queue_backlog(qsize: int, warn_size: int) -> bool:
        """Whether the backlog has reached the threshold worth reporting.

        Threshold only. The "have I said this recently" half used to live here as a
        hand-rolled timestamp cooldown, which meant an operator inside the 30 s window
        saw nothing at any level -- and no census could find it, because the condition
        was spelled as a plain call rather than as a latch lookup. That half now goes
        through `warn_level`, which demotes the repeat to DEBUG instead of dropping it.
        """
        return qsize >= warn_size

    # ORCHESTRATION METHODS


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
            self.logger.warning("OpenRouter warmup failed: %s", exc, exc_info=True)
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
    async def _ensure_concurrency_controls(self, valves: Pipe.Valves) -> None:
        """Lazy-initialize queue worker and semaphore with the latest valves."""
        cls = type(self)
        current_loop = asyncio.get_running_loop()

        if self._queue_worker_lock is not None:
            try:
                lock_loop = getattr(cast(Any, self._queue_worker_lock), "_get_loop", lambda: None)()
                if lock_loop is not current_loop:
                    self._queue_worker_lock = None
            except RuntimeError:
                self._queue_worker_lock = None

        if self._queue_worker_lock is None:
            self._queue_worker_lock = asyncio.Lock()

        async with self._queue_worker_lock:
            if self._queue_worker_task is not None and not self._queue_worker_task.done():
                try:
                    worker_loop = self._queue_worker_task.get_loop()
                except AttributeError:  # pragma: no cover - defensive for older asyncio implementations
                    worker_loop = None
                if worker_loop is not None and worker_loop is not current_loop:
                    self.logger.debug(
                        "Dropping stale request worker bound to a different event loop during setup."
                    )
                    self._queue_worker_task = None
                    self._request_queue = None

            if self._request_queue is not None:
                try:
                    queue_loop = self._request_queue._get_loop()  # type: ignore[attr-defined]
                except (RuntimeError, AttributeError):
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

                active = type(job.pipe)._active_jobs
                active.add(task)

                @timed
                def _mark_done(_task: asyncio.Task, q=queue, _active: set = active) -> None:
                    _active.discard(_task)
                    q.task_done()

                task.add_done_callback(_mark_done)

                @timed
                def _propagate_cancel(fut: asyncio.Future, _task: asyncio.Task = task, _job_id: str = job.request_id, _logger: logging.Logger = job.pipe.logger) -> None:
                    if fut.cancelled() and not _task.done():
                        _logger.debug("Cancelling in-flight request (request_id=%s)", _job_id)
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
        tool_token: contextvars.Token[_ToolExecutionContext | None] | None = None
        stream_queue = job.stream_queue
        stream_emitter = (
            self._event_emitter_handler._make_middleware_stream_emitter(job, stream_queue)
            if stream_queue is not None
            else None
        )
        try:
            if (
                self.valves.ENABLE_PLUGIN_SYSTEM
                and stream_emitter is not None
                and not job.task
            ):
                try:
                    wrapped = await self._ensure_plugin_registry().dispatch_on_emitter_wrap(
                        stream_emitter,
                        raw_emitter=job.event_emitter,
                        job_metadata={
                            "user_id": job.user_id,
                            "chat_id": job.metadata.get("chat_id", ""),
                            "message_id": job.metadata.get("message_id", ""),
                            "request_id": job.request_id,
                        },
                        valves=job.valves,
                    )
                    if wrapped is not None and wrapped is not stream_emitter:
                        stream_emitter = wrapped
                except Exception:
                    self.logger.debug("Plugin on_emitter_wrap dispatch failed", exc_info=True)

            async with self._acquire_semaphore(semaphore, job.request_id):
                session = self._create_http_session(job.valves)
                tokens = self._apply_logging_context(job)
                tokens.append(
                    (ModelFamily._PIPE_ID, ModelFamily._PIPE_ID.set(self.id))
                )
                tool_queue: asyncio.Queue[_QueuedToolCall | None] = asyncio.Queue(maxsize=50)
                per_request_tool_sem = asyncio.Semaphore(job.valves.MAX_PARALLEL_TOOLS_PER_REQUEST)
                per_tool_timeout = job.valves.TOOL_TIMEOUT_SECONDS
                batch_timeout = max(per_tool_timeout, job.valves.TOOL_BATCH_TIMEOUT_SECONDS)
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
                    request=job.request,
                    user=job.user,
                    metadata=job.metadata,
                    request_id=job.request_id,
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
                    user_valves=job.user_valves,
                    rejected_user_valves=job.rejected_user_valves,
                    )
                if not job.future.done():
                    job.future.set_result(result)
                self._circuit_breaker.reset(job.user_id)
        except asyncio.CancelledError:
            if not job.future.done():
                job.future.cancel()
            if stream_queue is not None:
                self._event_emitter_handler._try_put_middleware_stream_nowait(
                    stream_queue,
                    None,
                )
            raise
        except Exception as exc:
            self.logger.exception("Request job failed (request_id=%s)", job.request_id)
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
                self._event_emitter_handler._try_put_middleware_stream_nowait(stream_queue, None)
            if tool_context:
                await self._shutdown_tool_context(tool_context)

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

                    from .logging.session_log_manager import resolve_message_id
                    resolved_user_id = str(job.user_id or job.user.get("id") or job.metadata.get("user_id") or "")
                    resolved_session_id = str(job.session_id or job.metadata.get("session_id") or "")
                    resolved_chat_id = str(job.metadata.get("chat_id") or "")
                    resolved_message_id = resolve_message_id(job.metadata)
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

            backstop_rid = job.request_id or SessionLogger.request_id.get() or ""
            if backstop_rid:
                if job.future.cancelled():
                    backstop_status = "cancelled"
                else:
                    backstop_status = "ok"
                    with contextlib.suppress(Exception):
                        if job.future.exception() is not None:
                            backstop_status = "failed"
                await self._dispatch_plugin_event(
                    "dispatch_on_generation_complete",
                    None,
                    backstop_status,
                    request_id=backstop_rid,
                    metadata=job.metadata,
                    task=job.task,
                )

            if tool_token is not None:
                with contextlib.suppress(ValueError):
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

    def _refresh_process_log_level(self) -> None:
        """Point SessionLogger's out-of-request floor at the operator's LOG_LEVEL.

        Everything `pipes()` logs -- filter auto-install failures, plugin dispatch
        failures, startup pruning -- runs with no request in scope, so without this
        it is judged against the process default rather than the configured valve.

        The only writer of `process_log_level`. A request must not redefine it: the
        attribute is process-wide with no token to restore it, so one request would set
        the floor for every out-of-request logger in the worker from then on. The
        per-request level rides the `log_level` ContextVar instead, which
        `effective_log_level` prefers and `_apply_logging_context` resets.
        """
        try:
            SessionLogger.process_log_level = resolve_level(
                str(self.valves.LOG_LEVEL), SessionLogger.process_log_level
            )
        except Exception:  # noqa: BLE001, S110 - a valve read must not break startup
            pass

    @timed
    def _apply_logging_context(self, job: _PipeJob) -> list[tuple[ContextVar[Any], contextvars.Token[Any]]]:
        """Set SessionLogger contextvars based on the incoming request."""
        session_id = job.session_id or None
        request_id = job.request_id or None
        user_id = job.user_id or None
        log_level = resolve_level(str(job.valves.LOG_LEVEL), SessionLogger.process_log_level)
        SessionLogger.SESSION_LOG_MAX_LINES = job.valves.SESSION_LOG_MAX_LINES
        tokens: list[tuple[ContextVar[Any], contextvars.Token[Any]]] = []
        tokens.append((SessionLogger.session_id, SessionLogger.session_id.set(session_id)))
        tokens.append((SessionLogger.request_id, SessionLogger.request_id.set(request_id)))
        tokens.append((SessionLogger.user_id, SessionLogger.user_id.set(user_id)))
        tokens.append((SessionLogger.log_level, SessionLogger.log_level.set(log_level)))

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
        user_valves: Pipe.UserValves | None = None,
        rejected_user_valves: list[str] | None = None,
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
            valves = self._merge_valves(
                self.valves,
                parse_user_valves(__user__.get("valves"), model=self.UserValves)[0],
            )
        if session is None:
            raise RuntimeError("HTTP session is required for _handle_pipe_call")

        model_block = __metadata__.get("model")
        openwebui_model_id = model_block.get("id", "") if isinstance(model_block, dict) else ""
        pipe_identifier = self.id
        self._artifact_store._ensure_artifact_store(valves, pipe_identifier)

        plugin_result = None
        if self.valves.ENABLE_PLUGIN_SYSTEM:
            try:
                plugin_result = await self._ensure_plugin_registry().dispatch_on_request(
                    body, __user__, __metadata__, __event_emitter__, __task__,
                    valves=valves,
                    request_id=SessionLogger.request_id.get() or "",
                )
            except Exception:
                self.logger.debug("Plugin on_request dispatch failed", exc_info=True)
        if plugin_result is not None:
            if bool(body.get("stream")) and __event_emitter__:
                _pcontent: str | None = None
                if isinstance(plugin_result, dict):
                    _pchoices = plugin_result.get("choices")
                    if isinstance(_pchoices, list) and _pchoices:
                        _pmsg = _pchoices[0].get("message")
                        if isinstance(_pmsg, dict):
                            _pcontent = _pmsg.get("content")
                elif isinstance(plugin_result, str):
                    _pcontent = plugin_result
                if isinstance(_pcontent, str) and _pcontent:
                    await __event_emitter__(
                        {"type": "chat:message:delta", "data": {"content": _pcontent}}
                    )
                await __event_emitter__(
                    {"type": "chat:completion", "data": {"done": True}}
                )
            return plugin_result

        task_name = TaskModelAdapter._task_name(__task__)
        use_task_model_adapter = TaskModelAdapter._uses_task_model_adapter(__task__)
        if use_task_model_adapter and self._auth_failure_active():
            fallback = self._build_task_fallback_content(task_name)
            return self._build_chat_completion_payload(
                model=str(body.get("model") or openwebui_model_id or "pipe"),
                content=fallback,
            )

        api_key_value, api_key_error = self._resolve_openrouter_api_key(valves)
        if api_key_error:
            self._note_auth_failure()
            if use_task_model_adapter:
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
            if bool(body.get("stream")) and __event_emitter__:
                await self._ensure_error_formatter()._emit_templated_error(
                    __event_emitter__,
                    template=template,
                    variables=variables,
                    log_message=f"Auth configuration error: {api_key_error}",
                    log_level=logging.WARNING,
                )
                return ""

            error_id, context_defaults = self._ensure_error_formatter()._build_error_context()
            enriched_variables = {**context_defaults, **variables}
            try:
                markdown = _render_error_template(template, enriched_variables)
            except Exception:
                self.logger.debug("Auth error template rendering failed; using fallback", exc_info=True)
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
            if valves.ENABLE_VIDEO_GENERATION:
                from .integrations.video_catalog import ensure_video_catalog_loaded

                await ensure_video_catalog_loaded(
                    session,
                    valves=valves,
                    api_key=api_key_value or "",
                    logger=self.logger,
                    cache_seconds=valves.MODEL_CATALOG_REFRESH_SECONDS,
                )
            if valves.ENABLE_OPENROUTER_IMAGE_GENERATION:
                from .integrations.image_catalog import ensure_image_catalog_loaded

                await ensure_image_catalog_loaded(
                    session,
                    valves=valves,
                    api_key=api_key_value or "",
                    logger=self.logger,
                    cache_seconds=valves.MODEL_CATALOG_REFRESH_SECONDS,
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
                self.logger.exception("OpenRouter model catalog unavailable")
                return ""
            self.logger.warning("OpenRouter catalog refresh failed (%s). Serving %d cached model(s).", exc, len(available_models), exc_info=True)
        else:
            available_models = OpenRouterModelRegistry.list_models()
        catalog_norm_ids = {m["norm_id"] for m in available_models if isinstance(m, dict) and m.get("norm_id")}
        allowlist_models = self._select_models(valves.MODEL_ID, available_models) or available_models
        allowlist_models, virtual_variant_bases = self._expand_variants_for_enforcement(
            allowlist_models, valves, available_models,
        )
        allowlist_norm_ids = {m["norm_id"] for m in allowlist_models if isinstance(m, dict) and m.get("norm_id")}
        enforced_models = self._apply_model_filters(allowlist_models, valves)
        enforced_norm_ids = {m["norm_id"] for m in enforced_models if isinstance(m, dict) and m.get("norm_id")}

        features = _extract_feature_flags(__metadata__)
        user_id = str(__user__.get("id") or __metadata__.get("user_id") or "")

        if self.valves.ENABLE_PLUGIN_SYSTEM:
            try:
                await self._ensure_plugin_registry().dispatch_on_request_transform(
                    body, str(body.get("model", "")), valves,
                    user=__user__, metadata=__metadata__,
                )
            except Exception:
                self.logger.debug("Plugin on_request_transform dispatch failed", exc_info=True)

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
                virtual_variant_bases=virtual_variant_bases,
                user_valves=user_valves,
                rejected_user_valves=rejected_user_valves,
            )
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
                    "timeout_seconds": getattr(e, 'timeout', valves.HTTP_TOTAL_TIMEOUT_SECONDS),
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

            body_text = None
            if e.response is not None:
                try:
                    raw_bytes = await e.response.aread()
                    body_text = raw_bytes.decode("utf-8", errors="replace") if isinstance(raw_bytes, bytes) else str(raw_bytes)
                except Exception:
                    self.logger.debug("Failed to read HTTP error response body", exc_info=True)
                    body_text = None
            extra_meta: dict[str, Any] = {}
            if e.response is not None:
                _apply_retry_after_metadata(extra_meta, e.response.headers)
                rate_scope = (
                    e.response.headers.get("X-RateLimit-Scope")
                    or e.response.headers.get("x-ratelimit-scope")
                )
                if rate_scope:
                    extra_meta["rate_limit_type"] = rate_scope
            error = _build_openrouter_api_error(
                status=status_code or 0,
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

        except RequiredInternalFileError as e:
            await self._ensure_error_formatter()._emit_error(
                __event_emitter__,
                e.user_message,
                show_error_message=True,
                done=True,
            )
            self.logger.warning("Required internal file unavailable: %s", e.user_message)
            return ""

        # Generic catch-all
        except Exception as e:
            self.logger.exception("Unexpected error in _handle_pipe_call request processing")
            await self._ensure_error_formatter()._emit_templated_error(
                __event_emitter__,
                template=valves.INTERNAL_ERROR_TEMPLATE,
                variables={
                    "error_type": type(e).__name__,
                },
                log_message=f"Unexpected error: {e}",
            )
            return ""

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
        valves: Pipe.Valves,
        session: aiohttp.ClientSession,
        openwebui_model_id: str,
        pipe_identifier: str,
        allowlist_norm_ids: set[str],
        enforced_norm_ids: set[str],
        catalog_norm_ids: set[str],
        features: dict[str, Any],
        *,
        user_id: str = "",
        virtual_variant_bases: dict[str, str] | None = None,
        user_valves: Pipe.UserValves | None = None,
        rejected_user_valves: list[str] | None = None,
    ) -> AsyncGenerator[str, None] | dict[str, Any] | str | None:
        return await self._ensure_request_orchestrator().process_request(
            body, __user__, __request__, __event_emitter__, __event_call__, __metadata__, __tools__,
            __task__, __task_body__, valves, session, openwebui_model_id, pipe_identifier,
            allowlist_norm_ids, enforced_norm_ids, catalog_norm_ids, features,
            user_id=user_id, virtual_variant_bases=virtual_variant_bases,
            user_valves=user_valves, rejected_user_valves=rejected_user_valves,
        )

    # Model Management

    @timed
    def _qualify_model_for_pipe(
        self,
        pipe_identifier: str | None,
        model_id: str | None,
    ) -> str | None:
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
        valves: Pipe.Valves | None = None,
        workers: int = 4,
        breaker_key: str | None = None,
        delta_char_limit: int = 0,
        idle_flush_ms: int = 0,
        nagle_min_chars: int = 1,
        chunk_queue_maxsize: int = 100,
        chunk_queue_warn_size: int = 1000,
        event_queue_maxsize: int = 100,
        event_queue_warn_size: int = 1000,
        user: Any = None,
        owui_chat_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._ensure_responses_adapter().send_openai_responses_streaming_request(
            session, request_body, api_key, base_url, valves=valves, workers=workers,
            breaker_key=breaker_key, delta_char_limit=delta_char_limit, idle_flush_ms=idle_flush_ms,
            nagle_min_chars=nagle_min_chars,
            chunk_queue_maxsize=chunk_queue_maxsize, chunk_queue_warn_size=chunk_queue_warn_size,
            event_queue_maxsize=event_queue_maxsize, event_queue_warn_size=event_queue_warn_size,
            user=user,
            owui_chat_id=owui_chat_id,
        ):
            yield event

    async def send_openai_chat_completions_streaming_request(
        self,
        session: aiohttp.ClientSession,
        responses_request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: Pipe.Valves | None = None,
        breaker_key: str | None = None,
        user: Any = None,
        owui_chat_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._ensure_chat_completions_adapter().send_openai_chat_completions_streaming_request(
            session, responses_request_body, api_key, base_url, valves=valves, breaker_key=breaker_key,
            user=user,
            owui_chat_id=owui_chat_id,
        ):
            yield event

    async def send_openai_chat_completions_nonstreaming_request(
        self,
        session: aiohttp.ClientSession,
        responses_request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: Pipe.Valves | None = None,
        breaker_key: str | None = None,
        user: Any = None,
        owui_chat_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._ensure_chat_completions_adapter().send_openai_chat_completions_nonstreaming_request(
            session, responses_request_body, api_key, base_url, valves=valves, breaker_key=breaker_key,
            user=user,
            owui_chat_id=owui_chat_id,
        )

    async def send_openrouter_nonstreaming_request_as_events(
        self,
        session: aiohttp.ClientSession,
        responses_request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: Pipe.Valves | None = None,
        endpoint_override: Literal["responses", "chat_completions"] | None = None,
        breaker_key: str | None = None,
        user: Any = None,
        owui_chat_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._ensure_nonstreaming_adapter().send_openrouter_nonstreaming_request_as_events(
            session,
            responses_request_body,
            api_key,
            base_url,
            valves=valves,
            endpoint_override=endpoint_override,
            breaker_key=breaker_key,
            user=user,
            owui_chat_id=owui_chat_id,
        ):
            yield event

    async def send_openrouter_streaming_request(
        self,
        session: aiohttp.ClientSession,
        responses_request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: Pipe.Valves | None = None,
        endpoint_override: Literal["responses", "chat_completions"] | None = None,
        workers: int = 4,
        breaker_key: str | None = None,
        delta_char_limit: int = 0,
        idle_flush_ms: int = 0,
        nagle_min_chars: int = 1,
        chunk_queue_maxsize: int = 100,
        chunk_queue_warn_size: int = 1000,
        event_queue_maxsize: int = 100,
        event_queue_warn_size: int = 1000,
        user: Any = None,
        owui_chat_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._ensure_chat_completions_adapter().send_openrouter_streaming_request(
            session, responses_request_body, api_key, base_url, valves=valves,
            endpoint_override=endpoint_override, workers=workers, breaker_key=breaker_key,
            delta_char_limit=delta_char_limit, idle_flush_ms=idle_flush_ms,
            nagle_min_chars=nagle_min_chars,
            chunk_queue_maxsize=chunk_queue_maxsize, chunk_queue_warn_size=chunk_queue_warn_size,
            event_queue_maxsize=event_queue_maxsize, event_queue_warn_size=event_queue_warn_size,
            user=user,
            owui_chat_id=owui_chat_id,
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
                raise TimeoutError()
            await asyncio.wait_for(_graceful(), timeout=timeout)
        except TimeoutError:
            self.logger.warning(
                "Tool shutdown exceeded %.1fs; cancelling workers.",
                timeout,
            )
        except asyncio.CancelledError:
            self.logger.debug("Tool shutdown interrupted by cancellation.")
        except Exception:
            self.logger.debug(
                "Tool shutdown encountered error; cancelling workers.",
                exc_info=True,
            )
        finally:
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
        except TimeoutError:
            message = (
                f"Tool batch '{batch[0].call.get('name')}' exceeded {context.batch_timeout:.0f}s and was cancelled."
                if context.batch_timeout
                else "Tool batch timed out."
            )
            context.timeout_error = context.timeout_error or message
            self.logger.warning("%s", message)
            for item in batch:
                tool_type = (item.tool_cfg.get("type") or "function").lower()
                if not context.fusion_inner:
                    self._circuit_breaker.record_tool_failure(
                        context.user_id, tool_type, str(item.call.get("name") or "")
                    )
                if not item.future.done():
                    item.future.set_result(
                        self._ensure_tool_executor()._build_tool_output(
                            item.call,
                            message,
                            status="failed",
                        )
                    )
                    await self._dispatch_plugin_event(
                        "dispatch_on_tool_result",
                        str(item.call.get("name") or "?"),
                        "failed",
                        request_id=context.request_id,
                        metadata=context.metadata or {},
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
                resolved_status = "failed"
            else:
                status, text, files, embeds = result
                payload = self._ensure_tool_executor()._build_tool_output(item.call, text, status=status, files=files, embeds=embeds)
                tool_type = (item.tool_cfg.get("type") or "function").lower()
                self._circuit_breaker.reset_tool(
                    context.user_id, tool_type, str(item.call.get("name") or "")
                )
                resolved_status = str(status or "completed")
            item.future.set_result(payload)
            await self._dispatch_plugin_event(
                "dispatch_on_tool_result",
                str(item.call.get("name") or "?"),
                resolved_status,
                request_id=context.request_id,
                metadata=context.metadata or {},
            )

    @timed
    async def _invoke_tool_call(
        self,
        item: _QueuedToolCall,
        context: _ToolExecutionContext,
    ) -> tuple[str, str, list[dict[str, Any]], list[str]]:
        """Invoke a single tool call with circuit breaker protection."""
        tool_type = (item.tool_cfg.get("type") or "function").lower()
        gate_name = str(item.call.get("name") or "")
        if not context.fusion_inner and not self._circuit_breaker.tool_allows(
            context.user_id, tool_type, gate_name
        ):
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
                    return await self._run_tool_with_retries(item, context, tool_type, gate_name)
            return await self._run_tool_with_retries(item, context, tool_type, gate_name)

    @timed
    async def _run_tool_with_retries(
        self,
        item: _QueuedToolCall,
        context: _ToolExecutionContext,
        tool_type: str,
        breaker_name: str | None = None,
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
        breaker_key = (
            breaker_name
            if breaker_name is not None
            else str(item.call.get("name") or "")
        )
        timing_mark(f"tool_run:{tool_name}:start")

        fn = item.tool_cfg.get("callable")
        if not callable(fn):
            message = f"Tool '{tool_name}' is missing a callable handler."
            self.logger.warning("%s", message)
            if not context.fusion_inner:
                self._circuit_breaker.record_tool_failure(
                    context.user_id, tool_type, breaker_key
                )
            return ("failed", message, [], [])
        fn_to_call = cast(ToolCallable, fn)
        timeout = float(context.timeout)

        async def _process_and_emit(raw_result: Any) -> tuple[str, list[dict[str, Any]], list[str]]:
            timing_mark(f"tool_run:{tool_name}:processing")
            try:
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
                        self.logger.debug("Failed to emit files for '%s': %s", tool_name, emit_exc, exc_info=True)

                # Emit embeds if any were extracted
                if embeds and context.event_emitter:
                    try:
                        await self._event_emitter_handler._emit_embeds(context.event_emitter, embeds)
                        timing_mark(f"tool_run:{tool_name}:embeds_emitted")
                    except Exception as emit_exc:
                        self.logger.debug("Failed to emit embeds for '%s': %s", tool_name, emit_exc, exc_info=True)

                return text, files, embeds
            except Exception as proc_exc:
                self.logger.debug("Result processing failed for '%s': %s", tool_name, proc_exc, exc_info=True)
                return _fallback_tool_text(raw_result), [], []

        try:
            timing_mark(f"tool_run:{tool_name}:executing")
            result = await asyncio.wait_for(
                self._call_tool_callable(fn_to_call, item.args),
                timeout=timeout,
            )
            self._circuit_breaker.reset_tool(context.user_id, tool_type, breaker_key)
            text, files, embeds = await _process_and_emit(result)
            timing_mark(f"tool_run:{tool_name}:done")
            return ("completed", text, files, embeds)
        except Exception as exc:
            mcp_disconnected = (
                tool_type == "mcp"
                and isinstance(exc, RuntimeError)
                and "not connected" in str(exc)
            )
            if not context.fusion_inner and not mcp_disconnected:
                self._circuit_breaker.record_tool_failure(
                    context.user_id, tool_type, breaker_key
                )
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Tool '%s' execution failed.",
                    tool_name,
                    exc_info=True,
                )
            timing_mark(f"tool_run:{tool_name}:failed")
            if mcp_disconnected:
                return (
                    "failed",
                    f"Tool '{tool_name}' is no longer available in this session.",
                    [],
                    [],
                )
            return ("failed", f"Tool error: {exc}", [], [])

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
        valves: Pipe.Valves | None = None,
        breaker_key: str | None = None,
        user: Any = None,
        owui_chat_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._ensure_responses_adapter().send_openai_responses_nonstreaming_request(
            session, request_body, api_key, base_url, valves=valves, breaker_key=breaker_key,
            user=user,
            owui_chat_id=owui_chat_id,
        )


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
            "X-OpenRouter-Title": _OPENROUTER_TITLE,
            "X-OpenRouter-Categories": _OPENROUTER_CATEGORIES,
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
        connect_timeout = valves.HTTP_CONNECT_TIMEOUT_SECONDS
        total_timeout_value = valves.HTTP_TOTAL_TIMEOUT_SECONDS
        total_timeout = total_timeout_value if total_timeout_value else None
        sock_read = valves.HTTP_SOCK_READ_SECONDS if total_timeout is None else None
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
    def _apply_model_filters(self, models: list[dict[str, Any]], valves: Pipe.Valves) -> list[dict[str, Any]]:
        """Apply model capability filters (free pricing/tool calling) to a model list."""
        if not models:
            return []

        free_mode = valves.FREE_MODEL_FILTER
        tool_mode = valves.TOOL_CALLING_FILTER
        zdr_only = valves.ZDR_MODELS_ONLY
        if zdr_only and not OpenRouterModelRegistry.zdr_list_available():
            # latched: runs from pipes() and the chat path; the condition is stable
            self.logger.log(
                warn_level(_warned_pipes_maintenance, "zdr_list_unavailable"),
                "ZDR model filter enabled but ZDR endpoint list is unavailable; skipping ZDR filtering.",
            )
        if free_mode == "all" and tool_mode == "all" and not zdr_only:
            return models

        filtered: list[dict[str, Any]] = []
        for model in models:
            norm_id = model.get("norm_id") or ""
            if not norm_id:
                continue

            is_virtual = model.get("variant_is_virtual", False)
            if is_virtual:
                spec_lookup_id = model.get("variant_base_norm_id") or norm_id.rsplit(":", 1)[0]
            else:
                spec_lookup_id = norm_id

            if zdr_only:
                is_zdr_capable = OpenRouterModelRegistry.is_zdr_capable(norm_id)
                if is_zdr_capable is False:
                    continue

            if free_mode != "all":
                is_free = is_free_model(spec_lookup_id)
                if free_mode == "only" and not is_free:
                    continue
                if free_mode == "exclude" and is_free:
                    continue

            if tool_mode != "all":
                supports_tools = supports_tool_calling(spec_lookup_id)
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
        valves: Pipe.Valves
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
        variant_specs_csv = valves.VARIANT_MODELS
        if not variant_specs_csv:
            return models

        variant_specs: list[tuple[str, str, bool]] = []
        for spec in variant_specs_csv.split(","):
            spec = spec.strip()
            if not spec:
                continue

            if "@" in spec:
                parts = spec.rsplit("@", 1)
                base_id = parts[0].strip()
                raw_tag = parts[1].strip()
                is_preset = raw_tag.startswith("preset/")
                variant_tag = raw_tag
                if base_id and variant_tag:
                    variant_specs.append((base_id, variant_tag, is_preset))
            elif ":" in spec:
                parts = spec.rsplit(":", 1)
                base_id = parts[0].strip()
                variant_tag = parts[1].strip().lower()
                if base_id and variant_tag:
                    variant_specs.append((base_id, variant_tag, False))

        if not variant_specs:
            return models

        model_map: dict[str, dict[str, Any]] = {}
        for model in models:
            original_id = model.get("original_id", "")
            if original_id:
                model_map[original_id] = model

        # Expand variants and presets
        expanded: list[dict[str, Any]] = list(models)

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

            variant_model = dict(base_model)

            base_sanitized_id = variant_model.get("id", "")
            variant_model["id"] = f"{base_sanitized_id}:{variant_tag}"


            # Update display name with tag
            base_name = variant_model.get("name", base_id)
            if is_preset:
                preset_slug = variant_tag.replace("preset/", "")
                tag_display = f"Preset: {preset_slug}"
            else:
                tag_display = variant_tag.capitalize()
            variant_model["name"] = f"{base_name} {tag_display}"


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
    def _expand_variants_for_enforcement(
        self,
        allowlist_models: list[dict[str, Any]],
        valves: Pipe.Valves,
        available_models: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Expand MODEL_ID + VARIANT_MODELS into the allowlist for enforcement.

        Unlike :meth:`_expand_variant_models` (used in ``pipes()`` for UI display),
        this helper sets ``norm_id`` to the **full suffixed ID** so that strict
        enforcement in ``pipe()`` sees the exact variant.

        Returns:
            A tuple of (expanded_models, virtual_variant_bases) where
            ``virtual_variant_bases`` maps full variant norm_id to base norm_id
            for admin-configured routing-only variants.
        """
        virtual_variant_bases: dict[str, str] = {}

        def _collect_variant_specs(raw_csv: str) -> list[tuple[str, str, bool]]:
            specs: list[tuple[str, str, bool]] = []
            if not isinstance(raw_csv, str):
                return specs
            raw_csv = raw_csv.strip()
            if not raw_csv or raw_csv.lower() == "auto":
                return specs
            for spec in raw_csv.split(","):
                spec = spec.strip()
                if not spec or spec.lower() == "auto":
                    continue
                if "@" in spec:
                    parts = spec.rsplit("@", 1)
                    base_id = parts[0].strip()
                    raw_tag = parts[1].strip()
                    is_preset = raw_tag.startswith("preset/")
                    if base_id and raw_tag:
                        specs.append((base_id, raw_tag, is_preset))
                elif ":" in spec:
                    parts = spec.rsplit(":", 1)
                    base_id = parts[0].strip()
                    variant_tag = parts[1].strip().lower()
                    if base_id and variant_tag:
                        specs.append((base_id, variant_tag, False))
            return specs

        variant_specs: list[tuple[str, str, bool]] = []
        seen_specs: set[tuple[str, str, bool]] = set()
        for spec in _collect_variant_specs(valves.VARIANT_MODELS):
            if spec in seen_specs:
                continue
            seen_specs.add(spec)
            variant_specs.append(spec)
        for spec in _collect_variant_specs(valves.MODEL_ID):
            if spec in seen_specs:
                continue
            seen_specs.add(spec)
            variant_specs.append(spec)

        if not variant_specs:
            return allowlist_models, virtual_variant_bases

        catalog_by_original: dict[str, dict[str, Any]] = {}
        catalog_by_sanitized: dict[str, dict[str, Any]] = {}
        catalog_by_norm: dict[str, dict[str, Any]] = {}
        for model in available_models:
            original_id = model.get("original_id", "")
            if original_id:
                catalog_by_original[original_id] = model
            sanitized_id = model.get("id", "")
            if sanitized_id:
                catalog_by_sanitized[sanitized_id] = model
            norm_id = model.get("norm_id", "")
            if norm_id:
                catalog_by_norm[norm_id] = model

        expanded: list[dict[str, Any]] = list(allowlist_models)
        existing_norm_ids = {
            m["norm_id"] for m in allowlist_models if isinstance(m, dict) and m.get("norm_id")
        }

        for base_id, variant_tag, is_preset in variant_specs:
            sanitized_base = sanitize_model_id(base_id)
            base_model = (
                catalog_by_original.get(base_id)
                or catalog_by_sanitized.get(sanitized_base)
                or catalog_by_sanitized.get(base_id)
            )
            base_norm_id = ModelFamily.base_model(sanitized_base)
            full_norm_id = f"{base_norm_id}:{variant_tag}" if base_norm_id else ""

            if full_norm_id and full_norm_id in existing_norm_ids:
                continue

            is_real_variant = bool(full_norm_id and not is_preset and full_norm_id in catalog_by_norm)
            if is_real_variant:
                variant_model = dict(catalog_by_norm[full_norm_id])
                expanded.append(variant_model)
                existing_norm_ids.add(full_norm_id)
                self.logger.debug(
                    "Added enforcement variant: %s (base=%s, virtual=False)",
                    full_norm_id,
                    base_norm_id,
                )
                continue

            if not base_model:
                separator = "@" if is_preset else ":"
                self.logger.warning(
                    "Variant model base not found in catalog: %s (skipping %s%s%s)",
                    base_id, base_id, separator, variant_tag,
                )
                continue
            if not full_norm_id:
                self.logger.warning("Variant model base could not be normalized: %s", base_id)
                continue

            variant_model = dict(base_model)
            base_sanitized_id = variant_model.get("id", "")
            variant_model["id"] = f"{base_sanitized_id}:{variant_tag}"
            variant_model["norm_id"] = full_norm_id
            variant_model["variant_base_norm_id"] = base_norm_id
            variant_model["variant_is_virtual"] = True

            virtual_variant_bases[full_norm_id] = base_norm_id
            expanded.append(variant_model)
            existing_norm_ids.add(full_norm_id)
            self.logger.debug(
                "Added enforcement variant: %s (base=%s, virtual=True)",
                full_norm_id, base_norm_id,
            )

        return expanded, virtual_variant_bases

    @timed
    def _model_restriction_reasons(
        self,
        model_norm_id: str,
        *,
        valves: Pipe.Valves,
        allowlist_norm_ids: set[str],
        catalog_norm_ids: set[str],
        virtual_variant_bases: dict[str, str] | None = None,
    ) -> list[str]:
        reasons: list[str] = []

        vvb = virtual_variant_bases or {}
        spec_lookup_id = vvb.get(model_norm_id, model_norm_id)
        spec_available = spec_lookup_id in catalog_norm_ids

        if (
            catalog_norm_ids and model_norm_id not in catalog_norm_ids
            and model_norm_id not in allowlist_norm_ids
        ):
            reasons.append("not_in_catalog")

        model_id_filter = valves.MODEL_ID
        if (
            model_id_filter and model_id_filter.lower() != "auto"
            and model_norm_id not in allowlist_norm_ids
        ):
            reasons.append("MODEL_ID")

        free_mode = valves.FREE_MODEL_FILTER
        if free_mode != "all" and spec_available:
            is_free = is_free_model(spec_lookup_id)
            if free_mode == "only" and not is_free:
                reasons.append("FREE_MODEL_FILTER=only")
            elif free_mode == "exclude" and is_free:
                reasons.append("FREE_MODEL_FILTER=exclude")

        tool_mode = valves.TOOL_CALLING_FILTER
        if tool_mode != "all" and spec_available:
            supports_tools = supports_tool_calling(spec_lookup_id)
            if tool_mode == "only" and not supports_tools:
                reasons.append("TOOL_CALLING_FILTER=only")
            elif tool_mode == "exclude" and supports_tools:
                reasons.append("TOOL_CALLING_FILTER=exclude")

        zdr_only = valves.ZDR_MODELS_ONLY
        if zdr_only and spec_available:
            is_zdr_capable = OpenRouterModelRegistry.is_zdr_capable(model_norm_id)
            if is_zdr_capable is False:
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
        valves: Pipe.Valves,
    ) -> None:
        """Apply provider-specific beta headers when needed.

        Currently used to opt into Claude's interleaved thinking mode when requested.
        """
        if not isinstance(headers, dict):
            return
        if not valves.ENABLE_ANTHROPIC_INTERLEAVED_THINKING:
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


    @classmethod
    @timed
    def _note_auth_failure(cls, *, ttl_seconds: int | None = None) -> None:
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
    def _resolve_openrouter_api_key(valves: Pipe.Valves) -> tuple[str | None, str | None]:
        """Return (api_key, error_message) where api_key is a usable bearer token.

        This guards against cases where `API_KEY` is stored encrypted but cannot
        be decrypted (WEBUI_SECRET_KEY mismatch / missing), which would
        otherwise be sent upstream as `Bearer encrypted:...` and cause noisy 401s.
        """
        raw_value = valves.API_KEY
        decrypted = EncryptedStr.decrypt(raw_value)
        decrypted = decrypted.strip() if isinstance(decrypted, str) else ""

        if not decrypted:
            return None, "OpenRouter API key is not configured."

        if raw_value.startswith(EncryptedStr._ENCRYPTION_PREFIX) and (
            decrypted.startswith(EncryptedStr._ENCRYPTION_PREFIX)
            or (not decrypted.startswith("sk-"))
        ):
                return (
                    None,
                    ("OpenRouter API key is encrypted but cannot be decrypted. "
                    "This usually means WEBUI_SECRET_KEY changed. Re-enter the API key in this pipe's settings."),
                )

        return decrypted, None

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
    def _merge_valves(self, global_valves, user_valves) -> Pipe.Valves:
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
                elif (
                    key == "PERSIST_REASONING_TOKENS"
                    and not hasattr(global_valves, key)
                ) or not hasattr(global_valves, target_key):
                    continue
            mapped[target_key] = value

        if not mapped:
            return global_valves

        mapped.pop("LOG_LEVEL", None)

        return global_valves.model_copy(update=mapped)


try:
    from .plugins.registry import PluginRegistry as _PluginRegistryForValves

    _ExtendedValves = _PluginRegistryForValves.build_extended_valves(Valves)
    if _ExtendedValves is not Valves:
        Pipe.Valves = _ExtendedValves  # type: ignore[misc]

    _ExtendedUserValves = _PluginRegistryForValves.build_extended_user_valves(UserValves)
    if _ExtendedUserValves is not UserValves:
        Pipe.UserValves = _ExtendedUserValves  # type: ignore[misc]
except Exception:
    logging.getLogger(__name__).warning("Plugin valve field merge failed; using base Valves/UserValves", exc_info=True)
