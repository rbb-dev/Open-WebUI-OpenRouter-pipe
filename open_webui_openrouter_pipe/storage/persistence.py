"""Database persistence and encryption subsystem.

This module provides a complete persistence layer for OpenRouter pipe artifacts:
- SQLAlchemy-based database persistence with auto-discovery of Open WebUI engine
- Fernet symmetric encryption for sensitive payloads (reasoning tokens)
- LZ4 compression for large artifacts
- Redis write-behind caching for multi-worker deployments
- Circuit breaker protection for database operations
- Background cleanup workers for old artifacts
- ULID-based artifact identifiers for monotonic sorting
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime
import functools
import hashlib
import json
import logging
import os
import random
import re
import secrets
import sys
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast

# External dependencies
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Engine,
    Index,
    MetaData,
    String,
    Table,
    func,
    text,
)
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import IdentifierError, SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..core.timing_logger import timed
from ..core.utils import (
    _await_if_needed,
    is_picture_output,
    tool_output_text_and_pictures,
)
from ..core.warn_latch import warn_level
from .owui_files import is_temporary_chat, temporary_chat_prefixes

# Optional dependencies
try:
    import lz4.frame as lz4frame
except ImportError:
    lz4frame = None

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

from ..core.config import (
    _FERNET_VERSION_HEAD,
    _ULID_TIME_MASK,
    CROCKFORD_ALPHABET,
    ULID_LENGTH,
    ULID_RANDOM_LENGTH,
    ULID_TIME_LENGTH,
)

# Persistence Constants

# Payload compression flags
_PAYLOAD_FLAG_PLAIN = 0
_PAYLOAD_FLAG_LZ4 = 1
_JSON_LEAD_BYTES = frozenset({0x7B, 0x5B, 0x20, 0x09, 0x0A, 0x0D})

# Encryption version
_ENCRYPTED_PAYLOAD_VERSION = 1
_PAYLOAD_HEADER_SIZE = 1

_REDIS_FLUSH_CHANNEL = "db-flush"

_REDIS_DELETE_MARKER_TTL_SECONDS = 86400

_REDIS_DELETE_MARKER_ALL = "owui:dropped:all"
_REDIS_DELETE_MARKER_KEEP = "owui:dropped:keep:"
_LEGACY_REDIS_DELETE_MARKER_ALL = "1"


def _delete_marker_value(keep_message_id: str | None) -> str:
    if keep_message_id:
        return f"{_REDIS_DELETE_MARKER_KEEP}{keep_message_id}"
    return _REDIS_DELETE_MARKER_ALL


def _webui_secret_key() -> str:
    return os.getenv("WEBUI_SECRET_KEY", os.getenv("WEBUI_JWT_SECRET_KEY", ""))


def _valve_column_body(raw: str) -> str:
    stripped = raw.strip()
    try:
        stored = json.loads(stripped)
    except ValueError:
        return stripped
    if isinstance(stored, str):
        return stored
    if isinstance(stored, dict):
        return ""
    return stripped


def raw_valve_column_decodes(raw: Any) -> bool:
    if not isinstance(raw, str) or not raw.strip():
        return True
    secret = _webui_secret_key()
    if not secret:
        return True
    body = _valve_column_body(raw)
    if not bool(re.fullmatch(r"[A-Za-z0-9_-]+={0,2}", body)) or not body.startswith(
        _FERNET_VERSION_HEAD
    ):
        return True
    key = secret.encode()
    if len(secret) != 44:
        key = base64.urlsafe_b64encode(hashlib.sha256(key).digest())
    try:
        Fernet(key).decrypt(body.encode())
    except (InvalidToken, ValueError, TypeError):
        return False
    return True


def _marker_spares_row(marker: Any, message_id: Any) -> bool:
    if not isinstance(marker, str) or not marker:
        return False
    if marker == _REDIS_DELETE_MARKER_ALL:
        return False
    if marker == _LEGACY_REDIS_DELETE_MARKER_ALL:
        return False
    if marker.startswith(_REDIS_DELETE_MARKER_KEEP):
        return message_id == marker[len(_REDIS_DELETE_MARKER_KEEP):]
    return message_id == marker


_CANCEL_REQUEUE_POLL_SECONDS = 0.05
_CANCEL_REQUEUE_POLL_ATTEMPTS = 40

_UNREADABLE_ARTIFACT_TABLE_KEY = "\x00artifact-key-unreadable"

_PIPE_OWNED_ROW_TYPES = frozenset({
    "session_log_segment",
    "session_log_segment_terminal",
    "session_log_lock",
    "dashboard_purge_lock",
})
_STORED_VALVE_UNREADABLE_MEMO_MAX = 32

_RETENTION_CACHE_PURGE_BATCH = 500

REPLY_MEMORY_IDLE_SECONDS = 900.0
REPLY_MEMORY_MAX_BYTES = 64 * 1024 * 1024


class ArtifactStoreUnavailable(RuntimeError):
    pass

# Type alias for Redis client
if TYPE_CHECKING:
    from redis.asyncio import Redis as _RedisClient
else:
    _RedisClient = Any


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


def _sanitize_table_fragment(value: str) -> str:
    """Normalize arbitrary identifiers into safe SQL table suffixes."""
    fragment = re.sub(r"[^a-z0-9_]", "_", (value or "").lower())
    fragment = fragment.strip("_") or "pipe"
    if len(fragment) > 62:
        fragment = fragment[:62].rstrip("_") or "pipe"
    return fragment


def _assembler_index_name(table_name: str) -> str:
    tail = table_name.rsplit("_", 1)[-1][:8]
    return f"ix_{tail}_item_type_created"


def _retention_index_name(table_name: str) -> str:
    tail = table_name.rsplit("_", 1)[-1][:8]
    return f"ix_{tail}_created_at"


@contextlib.contextmanager
def _db_session(factory: Callable[..., Session]):
    """Open a SQLAlchemy session, ensuring rollback-on-error and safe close.

    Usage::

        with _db_session(self._session_factory) as session:
            session.add(row)
            session.commit()
    """
    session: Session = factory()  # type: ignore[call-arg]
    try:
        yield session
    except Exception:
        with contextlib.suppress(Exception):
            session.rollback()
        raise
    finally:
        with contextlib.suppress(Exception):
            session.close()


def _detect_redis_config(valves: Any, logger: logging.Logger) -> tuple[str, str, str, bool]:
    """Detect Redis configuration from environment and valves.

    Returns:
        (redis_url, websocket_manager, websocket_redis_url, candidate)
    """
    redis_url = (os.getenv("REDIS_URL") or "").strip()
    websocket_manager = (os.getenv("WEBSOCKET_MANAGER") or "").strip().lower()
    websocket_redis_url = (os.getenv("WEBSOCKET_REDIS_URL") or "").strip()

    raw_uvicorn_workers = (os.getenv("UVICORN_WORKERS") or "1").strip()
    try:
        uvicorn_workers = int(raw_uvicorn_workers or "1")
    except ValueError:
        logger.warning("Invalid UVICORN_WORKERS value '%s'; defaulting to 1.", raw_uvicorn_workers)
        uvicorn_workers = 1

    multi_worker = uvicorn_workers > 1
    redis_url_configured = bool(redis_url)
    websocket_ready = websocket_manager == "redis" and bool(websocket_redis_url)
    redis_valve_enabled = valves.ENABLE_REDIS_CACHE

    if multi_worker and not redis_valve_enabled:
        logger.warning("Multiple UVicorn workers detected but ENABLE_REDIS_CACHE is disabled; Redis cache remains off.")
    if multi_worker and redis_valve_enabled:
        if not redis_url_configured:
            logger.warning("Multiple UVicorn workers detected but REDIS_URL is unset; Redis cache remains off.")
        elif websocket_manager != "redis":
            logger.warning("Multiple UVicorn workers detected but WEBSOCKET_MANAGER is not 'redis'; Redis cache remains off.")
        elif not websocket_ready:
            logger.warning("Multiple UVicorn workers detected but WEBSOCKET_REDIS_URL is unset; Redis cache remains off.")

    candidate = (
        redis_url_configured
        and multi_worker
        and websocket_ready
        and aioredis is not None
        and redis_valve_enabled
    )
    return redis_url, websocket_manager, websocket_redis_url, candidate


def _retained_bytes(value: Any) -> int:
    seen: set[int] = set()
    stack: list[Any] = [value]
    total = 0
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        total += sys.getsizeof(node)
        if isinstance(node, dict):
            for key, item in node.items():
                stack.append(key)
                stack.append(item)
        elif isinstance(node, (list, tuple, set, frozenset)):
            stack.extend(node)
    return total


def _copy_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_payload(item) for item in value)
    return value


# ArtifactStore Class


def _current_user_id() -> str:
    from open_webui_openrouter_pipe.core.logging_system import SessionLogger

    return SessionLogger.user_id.get() or ""


class ReplyMemory:
    def __init__(
        self,
        *,
        idle_seconds: float = REPLY_MEMORY_IDLE_SECONDS,
        max_bytes: int = REPLY_MEMORY_MAX_BYTES,
        clock: Callable[[], float] = time.monotonic,
        user_id: Callable[[], str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._idle_seconds = idle_seconds
        self._max_bytes = max_bytes
        self._clock = clock
        self._user_id = user_id or _current_user_id
        self._replies: OrderedDict[tuple[Any, Any, Any], tuple[float, dict[str, dict[str, Any]], int]] = OrderedDict()
        self._total_bytes: int = 0

        self._sweep: asyncio.TimerHandle | None = None
        self._sweep_loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.RLock()
        self._logger = logger
        self._evict_warn: dict[str, float] = {}
        self._evict_latch = threading.Lock()

    def _key(self, chat_id: Any, message_id: Any) -> tuple[Any, Any, Any]:
        return (self._user_id(), chat_id, message_id)

    def _expire(self) -> None:
        cutoff = self._clock() - self._idle_seconds
        replies = self._replies
        while replies:
            key, (touched, _rows, _size) = next(iter(replies.items()))
            if touched >= cutoff:
                return
            replies.pop(key, None)
            self._total_bytes -= _size


    def _arm(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._sweep is not None and self._sweep_loop is loop and not self._sweep.cancelled():
            return
        self._disarm()
        if not self._replies:
            return
        earliest = next(iter(self._replies.values()))[0]
        self._sweep = loop.call_later(max(0.0, earliest + self._idle_seconds - self._clock()), self._run_sweep_locked)
        self._sweep_loop = loop

    def _disarm(self) -> None:
        sweep, self._sweep, self._sweep_loop = self._sweep, None, None
        if sweep is not None:
            sweep.cancel()

    def _run_sweep_locked(self) -> None:
        with self._lock:
            self._run_sweep()

    def _run_sweep(self) -> None:
        self._sweep = None
        self._sweep_loop = None
        self._expire()
        self._arm()

    def rearm(self) -> None:
        with self._lock:
            self._disarm()
            self._arm()

    def _touch(self, key: tuple[Any, Any, Any]) -> None:
        _touched, rows, size = self._replies[key]
        self._replies[key] = (self._clock(), rows, size)
        self._replies.move_to_end(key)

    def open(self, chat_id: Any, message_id: Any) -> None:
        with self._lock:
            self._open(chat_id, message_id)

    def _open(self, chat_id: Any, message_id: Any) -> None:
        self._expire()
        key = self._key(chat_id, message_id)
        if key in self._replies:
            self._touch(key)
        else:
            self._replies[key] = (self._clock(), {}, 0)
        self._arm()

    def is_open(self, chat_id: Any, message_id: Any) -> bool:
        with self._lock:
            return self._is_open(chat_id, message_id)

    def _is_open(self, chat_id: Any, message_id: Any) -> bool:
        self._expire()
        return self._key(chat_id, message_id) in self._replies

    def hold(self, rows: list[dict[str, Any]]) -> list[str]:
        prepared = [(row, json.loads(json.dumps(row.get("payload"), default=str))) for row in rows]
        sized = [(row, payload, _retained_bytes(payload)) for row, payload in prepared]
        with self._lock:
            return self._hold(sized)

    def _hold(self, sized: list[tuple[dict[str, Any], Any, int]]) -> list[str]:
        self._expire()
        held: list[str] = []
        dropped = 0
        oversized = 0
        for row, payload, payload_bytes in sized:
            key = self._key(row.get("chat_id"), row.get("message_id"))
            if key not in self._replies:
                logging.getLogger(__name__).warning(
                    "A row was offered to a reply the memory no longer holds and was dropped: "
                    "chat_id=%s message_id=%s id=%s item_type=%s",
                    row.get("chat_id"),
                    row.get("message_id"),
                    row.get("id"),
                    row.get("item_type"),
                )
                continue
            item_id = row.setdefault("id", generate_item_id())
            _touched, kept, size = self._replies[key]
            replaced = _retained_bytes(kept[item_id]) if item_id in kept else 0
            charged = payload_bytes
            kept[item_id] = payload
            self._replies[key] = (self._clock(), kept, size - replaced + charged)
            self._replies.move_to_end(key)
            self._total_bytes += charged - replaced
            held.append(item_id)
        offered_keys = {self._key(row.get("chat_id"), row.get("message_id")) for row, _payload, _size in sized}
        for key in offered_keys:
            if key in self._replies and self._replies[key][2] > self._max_bytes:
                self._total_bytes -= self._replies[key][2]
                del self._replies[key]
                dropped += 1
                oversized += 1
        while self._replies and self._total_bytes > self._max_bytes:
            evicted = self._replies.popitem(last=False)[1]
            self._total_bytes -= evicted[2]
            dropped += 1
        if dropped and self._logger is not None:
            with self._evict_latch:
                level = warn_level(self._evict_warn, "reply-memory-evicted", cooldown_s=300.0)
            self._logger.log(
                level,
                "Held reply memory over its %d byte ceiling: dropped %d in-flight held repl%s "
                "(%d of them larger than the whole ceiling); an evicted reply loses its tool "
                "rounds and thinking for the rest of the stream",
                self._max_bytes, dropped, "y" if dropped == 1 else "ies", oversized,
            )
        kept_ids = {
            item_id
            for entry in (self._replies.get(key) for key in offered_keys)
            if entry is not None
            for item_id in entry[1]
        }
        self._arm()
        return [item_id for item_id in held if item_id in kept_ids]

    def read(self, chat_id: Any, message_id: Any, item_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        with self._lock:
            return self._read(chat_id, message_id, item_ids)

    def _read(self, chat_id: Any, message_id: Any, item_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        self._expire()
        key = self._key(chat_id, message_id)
        if key not in self._replies:
            return {}
        kept = self._replies[key][1]
        found = {item_id: _copy_payload(kept[item_id]) for item_id in item_ids if item_id in kept}
        if found:
            self._touch(key)
        return found

    def release(self, chat_id: Any, message_id: Any) -> None:
        with self._lock:
            self._release(chat_id, message_id)

    def _release(self, chat_id: Any, message_id: Any) -> None:
        entry = self._replies.pop(self._key(chat_id, message_id), None)
        if entry is not None:
            self._total_bytes -= entry[2]

    def holds(self, chat_id: Any) -> bool:
        with self._lock:
            return self._holds(chat_id)

    def _holds(self, chat_id: Any) -> bool:
        self._expire()
        return any(key[1] == chat_id for key in self._replies)


class ArtifactStore:
    """Manages artifact persistence, encryption, compression, and caching.

    This class encapsulates all database operations for storing and retrieving
    artifacts (reasoning traces, tool call results, etc.) with optional encryption,
    compression, and Redis caching.

    Architecture:
    - SQLAlchemy for database persistence
    - Fernet (symmetric encryption) for sensitive artifacts
    - LZ4 compression for large payloads
    - Redis write-behind cache for multi-worker deployments
    - Circuit breakers for fault tolerance
    - Background cleanup workers for old artifacts
    """

    def __init__(
        self,
        pipe_id: str,
        logger: logging.Logger,
        valves: Any,
        emit_notification_callback: Callable | None = None,
        tool_context_var: ContextVar | None = None,
        user_id_context_var: ContextVar | None = None,
        valves_owner: Any | None = None,
    ):
        """Initialize the ArtifactStore with dependencies from Pipe.

        Args:
            pipe_id: Unique identifier for this pipe instance
            logger: Logger instance for diagnostics
            valves: Pipe.Valves instance with configuration
            emit_notification_callback: Optional callback for user notifications
            tool_context_var: ContextVar for tool execution context
            user_id_context_var: ContextVar for current user ID
        """
        self.id = pipe_id
        self.logger = logger
        self._valves = valves
        self._valves_owner = valves_owner
        self._emit_notification = emit_notification_callback
        self._TOOL_CONTEXT = tool_context_var
        self._user_id_context = user_id_context_var

        self._reply_memory = ReplyMemory(
            user_id=lambda: (self._user_id_context.get() or "") if self._user_id_context else "",
            logger=self.logger,
        )
        self._api_reply_memory = ReplyMemory(
            user_id=lambda: (self._user_id_context.get() or "") if self._user_id_context else "",
            logger=self.logger,
        )
        self._initialize_database_state()
        self._initialize_encryption_state()
        self._initialize_circuit_breakers()
        self._initialize_redis_state()
        self._initialize_cleanup_state()

    @property
    def valves(self) -> Any:
        live = getattr(self._valves_owner, "valves", None)
        return self._valves if live is None else live

    def _initialize_encryption_state(self):
        """Initialize encryption and compression state."""
        from open_webui_openrouter_pipe.core.config import EncryptedStr

        self._artifact_key_warning_emitted = False
        self._write_refusal_notified = False
        self._artifact_key_unreadable = False
        self._table_key = ""
        self._stored_valve_unreadable_memo: dict[tuple[str, str, str], bool] = {}
        self._apply_artifact_encryption_key(
            EncryptedStr.read(self.valves.ARTIFACT_ENCRYPTION_KEY),
            self.valves.ARTIFACT_ENCRYPTION_KEY,
        )
        self._encrypt_all: bool = bool(self.valves.ENCRYPT_ALL)
        self._compression_min_bytes: int = self.valves.MIN_COMPRESS_BYTES
        self._compression_enabled: bool = bool(
            self.valves.ENABLE_LZ4_COMPRESSION and lz4frame is not None
        )
        self._fernet: Fernet | None = None
        self._fernet_key_source: str | None = None
        self._lz4_warning_emitted = False
        self._lz4_faulted: bool = False

    def _raw_valve_column(self) -> tuple[Any, bool]:
        session_factory = self._session_factory
        if session_factory is None:
            return None, False
        metadata = MetaData()
        table = Table("function", metadata, autoload_with=session_factory.kw["bind"])
        with _db_session(session_factory) as session:
            return session.query(table.c.valves).filter(table.c.id == self.id).scalar(), True

    def _stored_valve_row_is_unreadable(self, stored: Any) -> bool:
        secret = _webui_secret_key()
        memo_key = (str(self.id or ""), secret, str(stored or ""))
        memo = self._stored_valve_unreadable_memo
        if memo_key in memo:
            return memo[memo_key]
        try:
            raw, reached = self._raw_valve_column()
        except Exception:
            self.logger.debug(
                "Raw valve column could not be read while arming the artifact guard (pipe_id=%s)",
                self.id,
                exc_info=True,
            )
            return False
        if not reached:
            return False
        unreadable = not raw_valve_column_decodes(raw)
        if len(memo) >= _STORED_VALVE_UNREADABLE_MEMO_MAX:
            memo.clear()
        memo[memo_key] = unreadable
        return unreadable

    def _stored_key_is_unreadable(self, stored: Any) -> bool:
        if self._encryption_key:
            return False
        return bool(str(stored or "").strip()) or self._stored_valve_row_is_unreadable(stored)

    def _apply_artifact_encryption_key(self, plaintext: str | None, stored: Any) -> None:
        self._encryption_key: str = (plaintext or "").strip()
        unreadable = self._stored_key_is_unreadable(stored)
        if unreadable != self._artifact_key_unreadable:
            self._artifact_key_warning_emitted = False
            self._write_refusal_notified = False
        self._artifact_key_unreadable = unreadable
        self._table_key = _UNREADABLE_ARTIFACT_TABLE_KEY if unreadable else self._encryption_key

    def _artifact_writes_blocked(self, rows: list[dict[str, Any]]) -> bool:
        if not self._artifact_key_unreadable:
            return False
        if not self._artifact_key_warning_emitted:
            self.logger.warning(
                "ARTIFACT_ENCRYPTION_KEY is set but cannot be decrypted with the current "
                "WEBUI_SECRET_KEY; dropping %d artifact row(s) rather than storing them "
                "unencrypted. Re-enter ARTIFACT_ENCRYPTION_KEY to resume storing artifacts. "
                "The stored value looks like a ciphertext but does not decode: it may be "
                "damaged, or it may be a passphrase typed with the 'encrypted:' prefix.",
                len(rows),
            )
            self._artifact_key_warning_emitted = True
        return True

    async def _note_artifact_write_refused(self, rows: list[dict[str, Any]]) -> None:
        context = self._TOOL_CONTEXT.get() if self._TOOL_CONTEXT else None
        emitter = context.event_emitter if context else None
        if emitter is None or not self._emit_notification:
            self.logger.debug(
                "Artifact write refused with no live turn to report it on (%d row(s)).",
                len(rows),
            )
            return
        if self._write_refusal_notified:
            self.logger.debug(
                "Artifact write refused again (%d row(s) still dropped this episode).",
                len(rows),
            )
            return
        self._write_refusal_notified = True
        await self._emit_notification(
            emitter,
            "Some stored items for this conversation were not saved: ARTIFACT_ENCRYPTION_KEY "
            "cannot be decrypted with the current WEBUI_SECRET_KEY, so the pipe refused to "
            "write them rather than store them in the clear. They will be missing from later "
            "turns. Re-enter ARTIFACT_ENCRYPTION_KEY to resume storing them.",
            level="warning",
        )

    def _initialize_circuit_breakers(self):
        """Initialize circuit breaker tracking."""
        breaker_threshold = max(1, int(self.valves.BREAKER_MAX_FAILURES))
        self._breaker_threshold = breaker_threshold
        self._breaker_window_seconds = self.valves.BREAKER_WINDOW_SECONDS
        self._db_breakers: dict[str, deque[float]] = defaultdict(deque)
        self._db_sweep_after: float = 0.0

    def _sweep_db_breakers(self, now: float, *, force: bool = False) -> None:
        if not force and now < self._db_sweep_after:
            return
        self._db_sweep_after = now + self._breaker_window_seconds
        window = self._breaker_window_seconds
        for user_id, failures in list(self._db_breakers.items()):
            if not failures or now - failures[-1] > window:
                self._db_breakers.pop(user_id, None)

    def configure_breaker(self, threshold: int, window_seconds: int) -> None:
        """Update circuit breaker thresholds.

        Args:
            threshold: Maximum failures before breaker opens
            window_seconds: Time window for failure counting
        """
        self._breaker_threshold = max(1, int(threshold))
        self._breaker_window_seconds = window_seconds
        self._db_sweep_after = 0.0

    def _initialize_redis_state(self):
        """Initialize Redis caching state."""
        self._redis_url, self._websocket_manager, self._websocket_redis_url, self._redis_candidate = (
            _detect_redis_config(self.valves, self.logger)
        )

        self._redis_enabled = False
        self._redis_client: _RedisClient | None = None
        self._flush_blocked_cycles: int = 0
        self._redis_listener_task: asyncio.Task | None = None
        self._redis_flush_task: asyncio.Task | None = None
        self._redis_ready_task: asyncio.Task | None = None
        self._redis_namespace = (self.id or "openrouter").lower()
        self._redis_pending_key = f"{self._redis_namespace}:pending"
        self._redis_cache_prefix = f"{self._redis_namespace}:artifact"
        self._redis_flush_lock_key = f"{self._redis_namespace}:flush_lock"
        self._redis_valve_draining = False
        self._redis_valve_off = False
        self._redis_valve_loop: asyncio.AbstractEventLoop | None = None

    def _initialize_database_state(self):
        """Initialize SQLAlchemy state."""
        self._engine: Engine | None = None
        self._session_factory: sessionmaker | None = None
        self._item_model: type[Any] | None = None
        self._artifact_table_name: str | None = None
        self._db_executor: ThreadPoolExecutor | None = None
        self._artifact_store_signature: tuple[str, str] | None = None
        self._closed: bool = False
        self._store_lock = threading.Lock()

    def _initialize_cleanup_state(self):
        """Initialize cleanup worker state."""
        self._cleanup_task: asyncio.Task | None = None


    @timed
    def _ensure_artifact_store(self, valves: Any, pipe_identifier: str | None = None) -> None:
        """Configure encryption/compression + ensure the backing table exists."""
        from open_webui_openrouter_pipe.core.config import EncryptedStr

        if self._closed:
            return

        plaintext = EncryptedStr.read(valves.ARTIFACT_ENCRYPTION_KEY)
        encryption_key = (plaintext or "").strip()
        if encryption_key != self._encryption_key:
            self._fernet = None
            self._fernet_key_source = None
        self._apply_artifact_encryption_key(plaintext, valves.ARTIFACT_ENCRYPTION_KEY)
        self._encrypt_all = valves.ENCRYPT_ALL
        self._compression_min_bytes = valves.MIN_COMPRESS_BYTES

        wants_compression = valves.ENABLE_LZ4_COMPRESSION
        compression_enabled = wants_compression and lz4frame is not None and not self._lz4_faulted
        if wants_compression and lz4frame is None and not self._lz4_warning_emitted:
            self.logger.warning("LZ4 compression requested but the 'lz4' package is not available. Artifacts will be stored without compression.")
            self._lz4_warning_emitted = True
        self._compression_enabled = compression_enabled

        self._reconcile_redis_valve(valves)

        pipe_identifier = pipe_identifier or self.id
        if not pipe_identifier:
            raise RuntimeError("Pipe identifier is missing; Open WebUI did not assign an id to this manifold.")
        self._ensure_store_locked(pipe_identifier)

    def _ensure_store_locked(self, pipe_identifier: str) -> None:
        table_fragment = _sanitize_table_fragment(pipe_identifier)
        desired_signature = (table_fragment, self._table_key)
        with self._store_lock:
            if (
                self._artifact_store_signature == desired_signature
                and self._item_model is not None
                and self._session_factory is not None
                and self._engine is not None
                and self._db_executor is not None
            ):
                return

            self._init_artifact_store(
                pipe_identifier=pipe_identifier,
                table_fragment=table_fragment,
            )

    def _reconcile_redis_valve(self, valves: Any) -> None:
        loop = self._resolve_valve_loop()
        if loop is not None:
            self._redis_valve_loop = loop
        if bool(getattr(valves, "ENABLE_REDIS_CACHE", True)):
            if not self._redis_valve_draining:
                self._redis_valve_off = False
            return
        if not self._redis_enabled:
            return
        if self._redis_valve_draining:
            return
        self._redis_valve_draining = True
        self._redis_valve_off = True
        owner = self._valves_owner
        handler = getattr(owner, "_drain_redis_after_valve_off", None)
        if not callable(handler):
            self._redis_valve_draining = False
            return
        drain = cast(Any, handler)()
        loop = self._redis_valve_loop
        owner_loop = getattr(owner, "_redis_loop", None)
        if (loop is None or loop.is_closed()) and owner_loop is not None and not owner_loop.is_closed():
            loop = owner_loop
        self._schedule_redis_valve_drain_on(loop, drain)

    def _schedule_redis_valve_drain_on(
        self, loop: asyncio.AbstractEventLoop | None, drain: Any
    ) -> bool:
        if loop is None or loop.is_closed():
            drain.close()
            self._redis_valve_draining = False
            self.logger.warning(
                "Redis valve turned off off-loop with no usable event loop; writes are stopped "
                "now and the buffered rows drain on the next request-path reconcile"
            )
            return False
        loop.call_soon_threadsafe(
            functools.partial(loop.create_task, drain, name="openrouter-redis-valve-off")
        )
        return True

    def _resolve_valve_loop(self) -> asyncio.AbstractEventLoop | None:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    @staticmethod
    @timed
    def _discover_owui_engine_and_schema(
        owui_db: Any,
    ) -> tuple[Any | None, str | None, dict[str, str]]:
        """Best-effort discovery of OWUI SQLAlchemy engine + schema without relying on symbol names."""
        engine: Any | None = None
        schema: str | None = None
        details: dict[str, str] = {}

        try:
            base = getattr(owui_db, "Base", None)
            metadata = getattr(base, "metadata", None) if base is not None else None
            candidate = getattr(metadata, "schema", None) if metadata is not None else None
            if isinstance(candidate, str) and candidate.strip():
                schema = candidate.strip()
                details["schema_source"] = "owui_db.Base.metadata.schema"
        except (AttributeError, TypeError, RuntimeError):
            schema = None

        if schema is None:
            try:
                metadata_obj = getattr(owui_db, "metadata_obj", None)
                candidate = (
                    getattr(metadata_obj, "schema", None) if metadata_obj is not None else None
                )
                if isinstance(candidate, str) and candidate.strip():
                    schema = candidate.strip()
                    details["schema_source"] = "owui_db.metadata_obj.schema"
            except (AttributeError, TypeError, RuntimeError):
                schema = None

        if schema is None:
            try:
                import importlib

                owui_env = importlib.import_module("open_webui.env")  # type: ignore

                candidate = getattr(owui_env, "DATABASE_SCHEMA", None)
                if isinstance(candidate, str) and candidate.strip():
                    schema = candidate.strip()
                    details["schema_source"] = "open_webui.env.DATABASE_SCHEMA"
            except (ImportError, AttributeError, TypeError):
                schema = None

        if engine is None:
            for attr in ("engine", "async_engine", "ENGINE", "bind", "BIND"):
                candidate = getattr(owui_db, attr, None)
                if candidate:
                    engine = candidate
                    details["engine_source"] = f"owui_db.{attr}"
                    break

        if schema is None:
            details.setdefault("schema_source", "unavailable")
        if engine is None:
            details.setdefault("engine_source", "unavailable")

        return engine, schema, details

    @timed
    def _init_artifact_store(
        self,
        pipe_identifier: str | None = None,
        *,
        table_fragment: str | None = None,
    ) -> None:
        """Initialize the per-pipe SQLAlchemy model + executor for artifact storage."""
        engine: Any | None = None
        schema: str | None = None

        try:
            from open_webui.internal import db as owui_db  # type: ignore
        except ImportError:
            owui_db = None
        except Exception:  # pragma: no cover - open_webui present but its import raised
            logging.getLogger(__name__).warning(
                "open_webui.internal failed to import for a reason other than absence; "
                "the features that depend on it are now disabled",
                exc_info=True,
            )
            owui_db = None

        if owui_db is not None:
            engine, schema, details = self._discover_owui_engine_and_schema(owui_db)
            try:
                dialect = getattr(getattr(engine, "dialect", None), "name", None)
                driver = getattr(getattr(engine, "dialect", None), "driver", None)
                self.logger.debug(
                    "OWUI DB autodiscovery: engine_source=%s schema_source=%s dialect=%s driver=%s schema=%s",
                    details.get("engine_source", "unknown"),
                    details.get("schema_source", "unknown"),
                    dialect or "unknown",
                    driver or "unknown",
                    schema or "",
                )
            except Exception:
                self.logger.debug(
                    "OWUI DB autodiscovery: engine_source=%s schema_source=%s schema=%s",
                    details.get("engine_source", "unknown"),
                    details.get("schema_source", "unknown"),
                    schema or "",
                    exc_info=True,
                )

        if not engine:
            self.logger.warning("Artifact persistence disabled: Open WebUI database engine is unavailable.")
            self._engine = None
            self._session_factory = None
            self._item_model = None
            self._artifact_table_name = None
            self._artifact_store_signature = None
            return

        session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine,
            expire_on_commit=False,
        )
        base = declarative_base()

        pipe_identifier = pipe_identifier or self.id
        if not pipe_identifier:
            raise RuntimeError("Pipe identifier is required to initialize the artifact store.")
        table_fragment = table_fragment or _sanitize_table_fragment(pipe_identifier)
        suffix = self.table_suffix(pipe_identifier, table_fragment=table_fragment)
        table_name = f"response_items_{suffix}"
        class_name = f"ResponseItem_{table_fragment}_{suffix.rsplit('_', 1)[-1][:4]}"

        existing_table = base.metadata.tables.get(table_name)
        if existing_table is not None:
            base.metadata.remove(existing_table)

        normalized_schema: str | None = None
        if isinstance(schema, str):
            candidate = schema.strip()
            if candidate:
                normalized_schema = candidate

        table_args: dict[str, Any] = {
            "extend_existing": True,
            "sqlite_autoincrement": False,
        }
        if normalized_schema:
            table_args["schema"] = normalized_schema

        attrs: dict[str, Any] = {
            "__tablename__": table_name,
            "__table_args__": (
                Index(
                    _assembler_index_name(table_name),
                    "item_type",
                    "created_at",
                ),
                Index(
                    _retention_index_name(table_name),
                    "created_at",
                ),
                table_args,
            ),
            "id": Column(String(ULID_LENGTH), primary_key=True),
            "chat_id": Column(String(64), index=True, nullable=False),
            "message_id": Column(String(64), index=True, nullable=False),
            "model_id": Column(String(128), nullable=True),
            "item_type": Column(String(64), nullable=False),
            "payload": Column(JSON, nullable=False, default=dict),
            "is_encrypted": Column(Boolean, nullable=False, default=False),
            "created_at": Column(
                DateTime,
                nullable=False,
                default=lambda: datetime.datetime.now(datetime.UTC),
            ),
        }

        item_model = type(class_name, (base,), attrs)

        schema_name = item_model.__table__.schema
        table_exists = True
        try:
            table_exists = sa_inspect(engine).has_table(table_name, schema=schema_name)
        except Exception:
            self.logger.debug(
                "Table existence probe failed for %s; assuming present", table_name, exc_info=True
            )
            table_exists = True

        if not self._create_table_with_race_guard(item_model.__table__, engine, table_name):
            self._engine = None
            self._session_factory = None
            self._item_model = None
            self._artifact_table_name = None
            self._artifact_store_signature = None
            return

        if table_exists and not self._reconcile_artifact_schema(
            item_model.__table__, engine, table_name, schema_name
        ):
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
        self._artifact_store_signature = (table_fragment, self._table_key)
        if not table_exists:
            self.logger.info("Artifact table ready: %s (key hash: %s). Changing ARTIFACT_ENCRYPTION_KEY creates a new table; old artifacts become inaccessible.", table_name, suffix.rsplit("_", 1)[-1])
        if self._db_executor is None:
            pool_workers: int = 5
            try:
                size_attr = getattr(engine.pool, "size", None)
                if callable(size_attr):
                    val = size_attr()
                    if isinstance(val, int) and val > 0:
                        pool_workers = val
                elif isinstance(size_attr, int) and size_attr > 0:
                    pool_workers = size_attr
                try:
                    from open_webui.env import (
                        DATABASE_POOL_MAX_OVERFLOW,  # type: ignore
                    )
                    if isinstance(DATABASE_POOL_MAX_OVERFLOW, int) and DATABASE_POOL_MAX_OVERFLOW > 0:
                        pool_workers += DATABASE_POOL_MAX_OVERFLOW
                except ImportError:
                    pass
                except Exception:
                    logging.getLogger(__name__).warning(
                        "open_webui.env failed to import for a reason other than absence; "
                        "the features that depend on it are now disabled",
                        exc_info=True,
                    )
            except Exception:
                self.logger.debug("Failed to read DB pool size from engine, using default", exc_info=True)
            self._db_executor = ThreadPoolExecutor(max_workers=pool_workers, thread_name_prefix="responses-db")
            self.logger.debug("DB thread pool: max_workers=%d (pool type: %s)", pool_workers, type(engine.pool).__name__)
        self.logger.debug("Artifact table ready: %s", table_name)

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        """Return a double-quoted identifier safe for direct SQL execution."""
        value = (identifier or "").replace('"', '""')
        return f'"{value}"'

    def table_suffix(self, pipe_identifier: str | None = None, *, table_fragment: str | None = None) -> str:
        """Stable per-pipe+key table suffix shared by all pipe-owned tables.

        The single source of the ``{fragment}_{keyhash8}`` construction; the
        artifact table name is built from this so sibling tables (e.g. usage
        stats) can never diverge from it.
        """
        identifier = pipe_identifier or self.id
        if not identifier:
            raise RuntimeError("Pipe identifier is required to derive the table suffix.")
        fragment = table_fragment or _sanitize_table_fragment(identifier)
        hash_source = f"{self._table_key}{identifier}".encode("utf-8", "ignore")
        key_hash = hashlib.sha256(hash_source).hexdigest()
        return f"{fragment}_{key_hash[:8]}"

    @staticmethod
    def _is_table_exists_error(exc: Exception) -> bool:
        """True when a DDL failure means another worker created the table first."""
        return "already exists" in str(exc).lower()

    def _create_table_with_race_guard(self, table: Any, engine: Any, table_name: str) -> bool:
        """Create the table idempotently; concurrent creation counts as success.

        checkfirst=True races between its existence probe and the CREATE, so a
        losing worker sees "already exists" — the goal state, not a failure.
        """
        created = False
        try:
            created = self._create_table_best_effort(table, engine, table_name)
        finally:
            if created:
                self._create_declared_indexes(table, engine, table_name)
        return created

    def _create_table_best_effort(self, table: Any, engine: Any, table_name: str) -> bool:
        try:
            table.create(bind=engine, checkfirst=True)
            return True
        except Exception as exc:  # pragma: no cover - database-specific errors
            if self._maybe_heal_index_conflict(engine, table, exc):
                try:
                    table.create(bind=engine, checkfirst=True)
                    return True
                except Exception as retry_exc:
                    if self._is_table_exists_error(retry_exc):
                        self.logger.debug("Table %s already exists (concurrent create)", table_name)
                        return True
                    self.logger.warning(
                        "Artifact persistence disabled (table init failed after index cleanup): %s",
                        retry_exc,
                        exc_info=True,
                    )
                    return False
            if self._is_table_exists_error(exc):
                self.logger.debug("Table %s already exists (concurrent create)", table_name)
                return True
            self.logger.warning("Artifact persistence disabled (table init failed): %s", exc, exc_info=True)
            return False

    def _create_declared_indexes(self, table: Any, engine: Any, table_name: str) -> None:
        for index in sorted(getattr(table, "indexes", None) or (), key=lambda idx: str(idx.name)):
            try:
                index.create(bind=engine, checkfirst=True)
            except Exception as exc:  # noqa: BLE001 - dialect errors vary; the assembler works without the index
                log = (
                    self.logger.warning
                    if index.name == _assembler_index_name(table_name)
                    else self.logger.debug
                )
                log(
                    "Index %s on %s not created: %s: %s",
                    index.name,
                    table_name,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )

    def _reconcile_artifact_schema(
        self,
        table: Any,
        engine: Any,
        table_name: str,
        schema_name: str | None,
    ) -> bool:
        from sqlalchemy import inspect as sa_inspect
        from sqlalchemy.exc import DuplicateColumnError
        from sqlalchemy.schema import CreateColumn

        def _present() -> set[str]:
            if schema_name:
                return {c["name"] for c in sa_inspect(engine).get_columns(table_name, schema=schema_name)}
            return {c["name"] for c in sa_inspect(engine).get_columns(table_name)}

        try:
            present = _present()
            qualified = self._quote_identifier(table_name)
            if schema_name:
                qualified = f"{self._quote_identifier(schema_name)}.{qualified}"
            added: list[str] = []
            for col in table.columns:
                if col.primary_key or col.name in present:
                    continue
                was_nullable = col.nullable
                try:
                    if not col.nullable and col.server_default is None:
                        col.nullable = True
                    ddl = str(CreateColumn(col).compile(dialect=engine.dialect)).strip()
                finally:
                    col.nullable = was_nullable
                try:
                    with engine.begin() as conn:
                        conn.execute(text(f"ALTER TABLE {qualified} ADD COLUMN {ddl}"))
                except Exception as exc:
                    message = str(exc).lower()
                    if (
                        "duplicate column" in message
                        or "already exists" in message
                        or isinstance(exc, DuplicateColumnError)
                    ):
                        self.logger.debug(
                            "Artifact table column already added by another worker: %s", col.name
                        )
                        continue
                    self.logger.warning(
                        "Artifact persistence disabled: column %s could not be added to %s: %s: %s",
                        col.name,
                        table_name,
                        type(exc).__name__,
                        exc,
                        exc_info=True,
                    )
                    return False
                added.append(col.name)
            if added:
                self.logger.info(
                    "Artifact table %s reconciled with columns: %s", table_name, ", ".join(added)
                )
                self._create_declared_indexes(table, engine, table_name)
            missing_pk = [
                col.name for col in table.columns if col.primary_key and col.name not in _present()
            ]
            if missing_pk:
                self.logger.warning(
                    "Artifact persistence disabled: %s is missing its primary key: %s; "
                    "every write will fail",
                    table_name,
                    ", ".join(missing_pk),
                )
                return False
            missing = {col.name for col in table.columns if not col.primary_key} - _present()
            if missing:
                self.logger.warning(
                    "Artifact persistence disabled: %s still lacks columns after "
                    "reconciliation: %s",
                    table_name,
                    ", ".join(sorted(missing)),
                )
                return False
            return True
        except Exception:
            self.logger.warning(
                "Artifact persistence disabled: reconciling %s failed", table_name, exc_info=True
            )
            return False

    @timed
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
        if isinstance(root_exc, IdentifierError):
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

        return False


    def _get_fernet(self) -> Fernet | None:
        """Return (and cache) the Fernet helper derived from the encryption key."""
        if not self._encryption_key:
            return None
        key_source = self._encryption_key
        if self._fernet is None or self._fernet_key_source != key_source:
            digest = hashlib.sha256(key_source.encode("utf-8")).digest()
            key = base64.urlsafe_b64encode(digest)
            self._fernet = Fernet(key)
            self._fernet_key_source = key_source
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
        from open_webui.utils.misc import sanitize_data_for_db  # type: ignore

        sanitized = sanitize_data_for_db(payload)
        return json.dumps(sanitized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

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
        except Exception as exc:
            self.logger.warning("LZ4 compression failed; disabling compression for the remainder of this process: %s", exc, exc_info=True)
            self._lz4_faulted = True
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
            elif flag == _PAYLOAD_FLAG_PLAIN:
                pass
            elif flag in _JSON_LEAD_BYTES:
                body = payload_bytes
            else:
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

    def _row_readable_under_current_key(self, row: dict[str, Any]) -> bool:
        if not row.get("is_encrypted"):
            return True
        fernet = self._get_fernet()
        if fernet is None:
            return False
        payload = row.get("payload")
        if isinstance(payload, dict):
            ciphertext = payload.get("ciphertext", "") or ""
        elif isinstance(payload, str):
            ciphertext = payload
        else:
            ciphertext = ""
        try:
            fernet.decrypt(ciphertext.encode("utf-8"))
        except (InvalidToken, TypeError, ValueError):
            return False
        return True

    def _rows_readable_under_current_key(self, rows: list[dict[str, Any]]) -> list[bool]:
        return [self._row_readable_under_current_key(row) for row in rows]

    def _encrypt_if_needed(self, item_type: str, payload: dict[str, Any]) -> tuple[Any, bool]:
        """Optionally encrypt ``payload`` depending on the item type."""
        if not self._should_encrypt(item_type):
            return payload, False
        encrypted = self._encrypt_payload(payload)
        return {"ciphertext": encrypted, "enc_v": _ENCRYPTED_PAYLOAD_VERSION}, True


    def _prepare_rows_for_storage(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        form_settled: bool = False,
    ) -> None:
        """Normalize row payloads so Redis/DB always receive the stored schema."""
        if not rows:
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            payload = row.get("payload")
            if form_settled:
                if not isinstance(payload, dict):
                    continue
                if row.get("is_encrypted"):
                    stored_payload, row["is_encrypted"] = self._encrypt_if_needed("reasoning", payload)
                    row["payload"] = stored_payload
                continue
            if row.get("is_encrypted"):
                if isinstance(payload, dict) and "ciphertext" in payload:
                    payload.setdefault("enc_v", _ENCRYPTED_PAYLOAD_VERSION)
                    continue
                if not isinstance(payload, dict):
                    continue
                if self._get_fernet() is None:
                    row["is_encrypted"] = False
                    continue
                stored_payload, _ = self._encrypt_if_needed("reasoning", payload)
                row["payload"] = stored_payload
                continue
            if not isinstance(payload, dict):
                continue
            stored_payload, is_encrypted = self._encrypt_if_needed(row.get("item_type", ""), payload)
            row["payload"] = stored_payload
            row["is_encrypted"] = is_encrypted

    async def _seal_rows(self, rows: list[dict[str, Any]], *, form_settled: bool = False) -> None:
        if self._db_executor is None:
            self._prepare_rows_for_storage(rows, form_settled=form_settled)
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._db_executor,
            functools.partial(self._prepare_rows_for_storage, rows, form_settled=form_settled),
        )

    def _reply_memory_for(self, chat_id: Any) -> ReplyMemory:
        return self._api_reply_memory if not chat_id else self._reply_memory

    def _make_db_row(
        self,
        chat_id: str | None,
        message_id: str | None,
        model_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Construct a persistence-ready row dict or return ``None`` when invalid."""
        if not self._reply_memory_for(chat_id).is_open(chat_id, message_id):
            if not chat_id:
                return None
            if is_temporary_chat(chat_id):
                return None
            if not self._item_model:
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

    def _existing_ids_sync(self, identifiers: list[str]) -> list[str]:
        if not identifiers or not self._item_model or not self._session_factory:
            return []
        model = self._item_model
        with _db_session(self._session_factory) as session:
            found = session.query(model).filter(model.id.in_(identifiers)).all()
        present: list[str] = []
        for row in found:
            identifier = getattr(row, "id", None)
            if isinstance(identifier, str) and identifier:
                present.append(identifier)
        return present

    def _conflict_tolerant_insert(self, instances: list[Any]) -> Any:
        if self._engine is None:
            return None
        table = getattr(self._item_model, "__table__", None)
        if table is None:
            return None
        dialect_name = self._engine.dialect.name
        if dialect_name not in ("postgresql", "sqlite"):
            return None
        values = [
            {
                "id": instance.id,
                "chat_id": instance.chat_id,
                "message_id": instance.message_id,
                "model_id": instance.model_id,
                "item_type": instance.item_type,
                "payload": instance.payload,
                "is_encrypted": instance.is_encrypted,
                "created_at": instance.created_at,
            }
            for instance in instances
        ]
        if not values:
            return None
        if dialect_name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            return pg_insert(table).values(values).on_conflict_do_nothing(index_elements=["id"])
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        return sqlite_insert(table).values(values).on_conflict_do_nothing(index_elements=["id"])

    @timed
    def _db_persist_sync(self, rows: list[dict[str, Any]]) -> list[str]:
        """Persist prepared rows once; intentionally no automatic retry logic."""
        if not rows:
            return []
        if not self._item_model or not self._session_factory:
            raise ArtifactStoreUnavailable(
                f"artifact store was torn down mid-flight; {len(rows)} row(s) were not written"
            )

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
                now = datetime.datetime.now(datetime.UTC)
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
                    if needs_encryption and is_encrypted and isinstance(payload, str):
                        stored_payload = payload
                    elif needs_encryption:
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

                statement = self._conflict_tolerant_insert(instances)
                if statement is None:
                    with _db_session(self._session_factory) as session:
                        session.add_all(instances)
                        session.commit()
                else:
                    with _db_session(self._session_factory) as session:
                        result = session.execute(statement)
                        session.commit()
                    rowcount = getattr(result, "rowcount", -1)
                    if rowcount < len(instances):
                        self.logger.warning(
                            "Artifact persist: %d of %d row(s) were already in the artifact "
                            "table and were not written again.",
                            len(instances) - rowcount,
                            len(instances),
                        )

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

    @timed
    def _try_acquire_lock_sync(self, lock_row: dict[str, Any]) -> bool:
        """Attempt to acquire a distributed lock by inserting a lock row.

        Uses INSERT ON CONFLICT DO NOTHING to avoid exceptions when another
        worker already holds the lock. This is the standard pattern for
        distributed locking in multi-worker environments.

        Args:
            lock_row: Dict with lock data including 'id', 'chat_id', 'message_id',
                     'item_type', 'payload', etc.

        Returns:
            True if lock was acquired (row inserted)
            False if lock is held by another worker (row already exists)
        """
        if not self._item_model or not self._session_factory or not self._engine:
            return False

        dialect_name = self._engine.dialect.name

        # Prepare row data
        now = datetime.datetime.now(datetime.UTC)
        lock_id = lock_row.get("id")
        if not lock_id:
            return False

        payload = lock_row.get("payload")
        if payload is not None and not isinstance(payload, str):
            try:
                payload = json.dumps(payload)
            except (TypeError, ValueError, RecursionError):
                self.logger.debug("Lock payload could not be serialized", exc_info=True)
                return False

        values = {
            "id": lock_id,
            "chat_id": lock_row.get("chat_id"),
            "message_id": lock_row.get("message_id"),
            "model_id": lock_row.get("model_id"),
            "item_type": lock_row.get("item_type"),
            "payload": payload,
            "is_encrypted": lock_row.get("is_encrypted", False),
            "created_at": now,
        }

        with _db_session(self._session_factory) as session:
            if dialect_name == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as pg_insert
                stmt = pg_insert(self._item_model.__table__).values(**values)
                stmt = stmt.on_conflict_do_nothing(index_elements=["id"])
            elif dialect_name == "sqlite":
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert
                stmt = sqlite_insert(self._item_model.__table__).values(**values)
                stmt = stmt.on_conflict_do_nothing(index_elements=["id"])
            else:
                try:
                    instance = self._item_model(**values)  # type: ignore[call-arg]
                    session.add(instance)
                    session.commit()
                    return True
                except SQLAlchemyError as exc:
                    session.rollback()
                    # Check if it's a duplicate key error
                    exc_str = str(exc).lower()
                    if "duplicate" in exc_str or "unique" in exc_str or "constraint" in exc_str:
                        return False
                    raise

            result = session.execute(stmt)
            session.commit()

            return getattr(result, "rowcount", 0) == 1

    @timed
    async def _db_persist(self, rows: list[dict[str, Any]]) -> list[str]:
        """Persist artifacts, optionally via Redis write-behind."""
        if not rows:
            return []
        held_memory = self._reply_memory_for(rows[0].get("chat_id"))
        if held_memory.is_open(rows[0].get("chat_id"), rows[0].get("message_id")):
            held = await asyncio.to_thread(held_memory.hold, rows)
            held_memory.rearm()
            return held
        if is_temporary_chat(rows[0].get("chat_id")):
            held = await asyncio.to_thread(self._reply_memory.hold, rows)
            self._reply_memory.rearm()
            return held

        from open_webui_openrouter_pipe.core.logging_system import SessionLogger

        context = self._TOOL_CONTEXT.get() if self._TOOL_CONTEXT else None
        user_id = SessionLogger.user_id.get() or ""
        if not self._db_breaker_allows(user_id):
            self.logger.warning("DB writes disabled for user_id=%s due to repeated failures", user_id)
            if self._emit_notification:
                await self._emit_notification(
                    context.event_emitter if context else None,
                    "DB ops skipped due to repeated errors.",
                    level="warning",
                )
            return []

        for row in rows:
            row.setdefault("id", generate_item_id())

        if self._artifact_writes_blocked(rows):
            await self._note_artifact_write_refused(rows)
            return []

        try:
            await self._seal_rows(rows)
            if self._redis_active():
                queued = await self._redis_enqueue_rows(rows)
                self._reset_db_failure(user_id)
                return queued
            return await self._db_persist_direct(rows, user_id=user_id)
        except Exception:
            self._record_db_failure(user_id)
            self.logger.exception(
                "Artifact persist failed, dropping %d row(s) (types=%s)",
                len(rows),
                sorted({str(r.get("item_type")) for r in rows if isinstance(r, dict)}),
            )
            if self._emit_notification:
                await self._emit_notification(
                    context.event_emitter if context else None,
                    "Some stored items for this turn could not be written to the "
                    "database; they will be missing from later turns.",
                    level="warning",
                )
            return []

    @timed
    async def _ack_rows_present_after_duplicate(
        self, rows: list[dict[str, Any]], loop: asyncio.AbstractEventLoop
    ) -> list[str]:
        candidates = [
            identifier
            for row in rows
            for identifier in [row.get("id")]
            if isinstance(identifier, str) and identifier
        ]
        present = await loop.run_in_executor(
            self._db_executor, self._existing_ids_sync, candidates
        )
        level = logging.WARNING if len(present) < len(candidates) else logging.DEBUG
        self.logger.log(
            level,
            "Duplicate key during DB persist: %d of %d row(s) are present in the artifact "
            "table; any remainder was rolled back and is not acknowledged.",
            len(present),
            len(candidates),
        )
        return present

    async def _partition_retired_key_rows(self, rows: list[dict[str, Any]]) -> list[bool]:
        if not rows:
            return []
        partition = await asyncio.to_thread(
            functools.partial(self._rows_readable_under_current_key, rows)
        )
        discarded = [row for row, readable in zip(rows, partition) if not readable]
        if discarded:
            self.logger.warning(
                "Discarded %d artifact(s) sealed under a retired ARTIFACT_ENCRYPTION_KEY "
                "rather than writing them into this key's table, where they could never "
                "be read (item_types=%s). Markers referencing them are permanently "
                "dangling.",
                len(discarded),
                sorted({str(row.get("item_type", "unknown")) for row in discarded}),
            )
            await self._invalidate_discarded_cache_keys(discarded)
        return list(partition)

    async def _invalidate_discarded_cache_keys(self, discarded: list[dict[str, Any]]) -> None:
        keys = [
            key
            for key in (
                self._redis_cache_key(row.get("chat_id"), row.get("id"))
                for row in discarded
            )
            if key
        ]
        if not keys or not self._redis_client:
            return
        try:
            await _await_if_needed(self._redis_client.delete(*keys))
        except Exception as exc:
            self.logger.warning(
                "Redis cache invalidation of discarded artifacts failed (best-effort): %s",
                exc, exc_info=True,
            )

    async def _db_persist_direct(self, rows: list[dict[str, Any]], user_id: str = "") -> list[str]:
        if not rows:
            return []
        if self._artifact_writes_blocked(rows):
            await self._note_artifact_write_refused(rows)
            return []
        if not self._db_executor or not self._item_model or not self._session_factory:
            raise ArtifactStoreUnavailable(
                f"artifact store is not configured (table={self._artifact_table_name!r}); "
                f"{len(rows)} row(s) were not written"
            )

        partition = await self._partition_retired_key_rows(rows)
        if not all(partition):
            rows = [row for row, readable in zip(rows, partition) if readable]
            if not rows:
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
                        return await self._ack_rows_present_after_duplicate(rows, loop)
                    raise
                if self._redis_active():
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

    def _db_touch_sync(
        self,
        chat_id: str,
        message_id: str | None,
        item_ids: list[str],
    ) -> None:
        if not (self._item_model and self._session_factory):
            return
        model = self._item_model
        with _db_session(self._session_factory) as touch_session:
            now = datetime.datetime.now(datetime.UTC)
            touch_query = touch_session.query(model).filter(model.chat_id == chat_id)
            touch_query = touch_query.filter(model.id.in_(item_ids))
            if message_id:
                touch_query = touch_query.filter(model.message_id == message_id)
            touch_query.update({model.created_at: now}, synchronize_session=False)
            touch_session.commit()

    async def _touch_cached(self, chat_id: str, message_id: str | None, ids: list[str]) -> None:
        executor = self._db_executor
        if not (executor and self._item_model and self._session_factory and ids):
            return
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                executor, functools.partial(self._db_touch_sync, chat_id, message_id, ids)
            )
        except Exception as exc:
            self.logger.debug(
                "Artifact touch failed (chat_id=%s, message_id=%s): %s",
                chat_id,
                message_id or "",
                exc,
                exc_info=True,
            )

    @timed
    def _db_fetch_sync(
        self,
        chat_id: str,
        message_id: str | None,
        item_ids: list[str],
        sealed: set[str] | None = None,
        unreadable: dict[str, str] | None = None,
    ) -> dict[str, dict]:
        """Synchronously fetch persisted artifacts for ``chat_id``."""
        if not item_ids or not self._item_model or not self._session_factory:
            return {}
        model = self._item_model
        with _db_session(self._session_factory) as session:
            query = session.query(model).filter(model.chat_id == chat_id)
            if item_ids:
                query = query.filter(model.id.in_(item_ids))
            if message_id:
                query = query.filter(model.message_id == message_id)
            rows = query.all()

        if rows:
            try:
                touched_ids = [getattr(row, "id", None) for row in rows]
                touched_ids = [
                    item_id
                    for item_id in touched_ids
                    if isinstance(item_id, str) and item_id
                ]
                if touched_ids:
                    try:
                        with _db_session(self._session_factory) as touch_session:
                            now = datetime.datetime.now(datetime.UTC)
                            touch_query = touch_session.query(model).filter(model.chat_id == chat_id)
                            touch_query = touch_query.filter(model.id.in_(touched_ids))
                            if message_id:
                                touch_query = touch_query.filter(model.message_id == message_id)
                            touch_query.update({model.created_at: now}, synchronize_session=False)
                            touch_session.commit()
                    except Exception as exc:
                        self.logger.debug(
                            "Artifact touch skipped (chat_id=%s, message_id=%s, rows=%s): %s",
                            chat_id,
                            message_id or "",
                            len(touched_ids),
                            exc,
                            exc_info=True,
                        )
            except Exception as exc:
                self.logger.debug(
                    "Artifact touch failed (chat_id=%s, message_id=%s): %s",
                    chat_id,
                    message_id or "",
                    exc,
                    exc_info=True,
                )

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
                    self.logger.warning(
                        "Failed to decrypt artifact %s (item_type=%s): %s",
                        row.id, getattr(row, "item_type", "unknown"), exc, exc_info=True,
                    )
                    if unreadable is not None:
                        unreadable[row.id] = str(getattr(row, "item_type", "unknown"))
                    continue
                if sealed is not None:
                    sealed.add(row.id)
            if isinstance(payload, dict):
                results[row.id] = payload
        return results

    @timed
    async def _db_fetch(
        self,
        chat_id: str | None,
        message_id: str | None,
        item_ids: list[str],
        *,
        reply_id: str | None = None,
    ) -> dict[str, dict]:
        """Fetch artifacts with Redis cache + retries."""
        if not (chat_id and item_ids):
            return {}
        if is_temporary_chat(chat_id):
            return await asyncio.to_thread(self._reply_memory.read, chat_id, reply_id, item_ids)

        cached: dict[str, dict] = {}
        if self._redis_enabled:
            cached = await self._redis_fetch_rows(chat_id, item_ids, message_id=message_id)
            cache_hit_ids = list(cached)
            missing_ids = [item_id for item_id in item_ids if item_id not in cached]
        else:
            cache_hit_ids = []
            missing_ids = item_ids

        unreadable: dict[str, str] = {}
        if not missing_ids:
            await self._touch_cached(chat_id, message_id, list(cached))
            return cached

        if not self._artifact_store_ready():
            self.logger.warning(
                "Artifact store is not configured; %d stored item(s) could not be loaded.",
                len(missing_ids),
            )
            context = self._TOOL_CONTEXT.get() if self._TOOL_CONTEXT else None
            if self._emit_notification:
                await self._emit_notification(
                    context.event_emitter if context else None,
                    "Earlier tool results could not be loaded, so the model did not receive them.",
                    level="warning",
                )
            return cached

        from open_webui_openrouter_pipe.core.logging_system import SessionLogger

        user_id = SessionLogger.user_id.get() or ""
        if not self._db_breaker_allows(user_id):
            self.logger.warning("DB reads disabled for user_id=%s due to repeated failures", user_id)
            context = self._TOOL_CONTEXT.get() if self._TOOL_CONTEXT else None
            if self._emit_notification:
                await self._emit_notification(
                    context.event_emitter if context else None,
                    "DB ops skipped due to repeated errors.",
                    level="warning",
                )
            return cached

        try:
            sealed: set[str] = set()
            fetched = await self._db_fetch_direct(
                chat_id, message_id, missing_ids, sealed, unreadable
            )
        except Exception as exc:
            self._record_db_failure(user_id)
            self.logger.warning("Artifact fetch failed: %s", exc, exc_info=True)
            context = self._TOOL_CONTEXT.get() if self._TOOL_CONTEXT else None
            if self._emit_notification:
                try:
                    await self._emit_notification(
                        context.event_emitter if context else None,
                        "Earlier tool results could not be loaded, so the model did not receive them.",
                        level="warning",
                    )
                except Exception:
                    self.logger.debug(
                        "Artifact fetch fault notice could not be delivered", exc_info=True
                    )
        else:
            if user_id:
                self._reset_db_failure(user_id)
            cached.update(fetched)
            if fetched and self._redis_active():
                cache_rows = []
                for item_id, payload in fetched.items():
                    row = {
                        "id": item_id,
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "item_type": (payload or {}).get("type", "unknown") if isinstance(payload, dict) else "unknown",
                        "payload": payload,
                        "is_encrypted": item_id in sealed,
                    }
                    cache_rows.append(row)
                try:
                    await self._seal_rows(cache_rows, form_settled=True)
                    await self._redis_cache_rows(cache_rows, chat_id=chat_id)
                except Exception as exc:
                    self.logger.warning(
                        "Artifact read succeeded but the replay cache write failed (%d row(s)); "
                        "the rows are returned anyway and this is not charged to the database "
                        "breaker: %s", len(fetched), exc, exc_info=True,
                    )
        if unreadable:
            await self._note_unreadable_rows(unreadable)
        await self._touch_cached(chat_id, message_id, cache_hit_ids)
        return cached

    async def _note_unreadable_rows(self, unreadable: dict[str, str]) -> None:
        context = self._TOOL_CONTEXT.get() if self._TOOL_CONTEXT else None
        if self._emit_notification:
            kinds = sorted(set(unreadable.values()))
            try:
                await self._emit_notification(
                    context.event_emitter if context else None,
                    f"{len(unreadable)} stored item(s) for this conversation could not be read "
                    f"and are missing from this turn (kinds: {', '.join(kinds)}); the rest of the "
                    "stored round was replayed normally.",
                    level="warning",
                )
            except Exception:
                self.logger.debug(
                    "Unreadable artifact rows notice could not be delivered", exc_info=True
                )

    @timed
    async def _db_fetch_direct(
        self,
        chat_id: str,
        message_id: str | None,
        item_ids: list[str],
        sealed: set[str] | None = None,
        unreadable: dict[str, str] | None = None,
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
                fetch_call = functools.partial(
                    self._db_fetch_sync, chat_id, message_id, item_ids, sealed, unreadable
                )
                return await loop.run_in_executor(self._db_executor, fetch_call)
        return {}

    @timed
    def _delete_artifacts_sync(self, artifact_ids: list[str], keep_message_id: str | None = None) -> set[str]:
        """Synchronously delete artifacts by ULID."""
        if not (artifact_ids and self._session_factory and self._item_model):
            return set()
        model = self._item_model
        kept: set[str] = set()
        with _db_session(self._session_factory) as session:
            query = session.query(model).filter(model.id.in_(artifact_ids))
            if keep_message_id:
                kept = {
                    row_id
                    for (row_id,) in query.filter(model.message_id == keep_message_id).with_entities(model.id)
                }
                query = query.filter(model.message_id != keep_message_id)
            query.delete(synchronize_session=False)
            session.commit()
        return kept

    @timed
    async def _delete_artifacts(self, refs: list[tuple[str, str]], keep_message_id: str | None = None) -> bool:
        """Delete persisted artifacts (and cached copies) once they have been replayed."""
        if not refs:
            return True
        ids = sorted({artifact_id for _, artifact_id in refs if artifact_id})
        if not ids or not self._db_executor:
            return True
        owners = await self._redis_cached_owners(refs)
        await self._mark_rows_dropped(ids, keep_message_id)

        from open_webui_openrouter_pipe.core.logging_system import SessionLogger

        user_id = SessionLogger.user_id.get() or ""
        if not self._db_breaker_allows(user_id):
            self.logger.warning("DB deletes disabled for user_id=%s due to repeated failures", user_id)
            return False

        loop = asyncio.get_running_loop()
        try:
            kept = await loop.run_in_executor(
                self._db_executor, functools.partial(self._delete_artifacts_sync, ids, keep_message_id)
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record_db_failure(user_id)
            self.logger.warning(
                "Artifact delete failed; keeping %d row(s) for the next turn: %s",
                len(ids),
                exc,
                exc_info=True,
            )
            return False
        if keep_message_id:
            kept = kept | {row_id for row_id, owner in owners.items() if owner == keep_message_id}
        if self._redis_enabled and self._redis_client:
            keys = [
                self._redis_cache_key(chat_id, artifact_id)
                for chat_id, artifact_id in refs
                if artifact_id not in kept
            ]
            keys = [key for key in keys if key]
            if keys:
                try:
                    await _await_if_needed(self._redis_client.delete(*keys))
                except Exception as exc:
                    self.logger.warning("Redis cache invalidation failed (best-effort): %s", exc, exc_info=True)
        return True


    @timed
    async def _redis_pubsub_listener(self) -> None:
        """Listen for cross-worker flush wake-ups, surviving idle periods and
        transient Redis failures.

        redis-py 8.0 introduced a 5-second default socket timeout, which makes
        a bare pubsub.listen() raise TimeoutError on the first idle stretch.
        get_message(timeout=...) returns None on idle instead, and the
        reconnect loop restores the subscription after real errors (Redis
        restarts, network blips) rather than dying for the worker's lifetime.
        The periodic timer flusher remains the independent fallback.
        """
        if not self._redis_client:
            return
        backoff = 1.0
        failure_reason = ""
        last_flush_failure = ""
        while self._redis_enabled and self._redis_client:
            pubsub = None
            try:
                pubsub = self._redis_client.pubsub()
                await pubsub.subscribe(_REDIS_FLUSH_CHANNEL)
                while self._redis_enabled:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=5.0
                    )
                    if failure_reason:
                        self.logger.info("Redis pub/sub listener reconnected")
                        failure_reason = ""
                        backoff = 1.0
                    if message is None:
                        continue
                    if message.get("type") != "message":
                        continue
                    try:
                        await self._flush_redis_queue()
                    except asyncio.CancelledError:  # pragma: no cover - shutdown path
                        raise
                    except Exception as flush_exc:
                        flush_reason = f"{type(flush_exc).__name__}: {flush_exc}"
                        if flush_reason != last_flush_failure:
                            last_flush_failure = flush_reason
                            self.logger.warning(
                                "Redis flush failed after a wake-up (%s); the "
                                "subscription is healthy and the timer flush continues",
                                flush_reason,
                                exc_info=True,
                            )
                    else:
                        last_flush_failure = ""
                return
            except asyncio.CancelledError:  # pragma: no cover - shutdown path
                raise
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                if reason != failure_reason:
                    self.logger.warning(
                        "Redis pub/sub listener error: %s (reconnecting; timer flush continues)",
                        reason,
                        exc_info=True,
                    )
                    failure_reason = reason
                else:
                    self.logger.debug("Redis pub/sub listener error repeated: %s", reason)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
            finally:
                if pubsub is not None:
                    with contextlib.suppress(Exception):
                        if hasattr(pubsub, "aclose"):
                            await pubsub.aclose()
                        else:
                            await pubsub.close()

    @timed
    async def _redis_periodic_flusher(self) -> None:
        consecutive_failures = 0

        while self._redis_enabled:
            warn_threshold = self.valves.REDIS_PENDING_WARN_THRESHOLD
            failure_limit = self.valves.REDIS_FLUSH_FAILURE_LIMIT
            try:
                if not self._redis_client:
                    break

                queue_depth = await _await_if_needed(
                    self._redis_client.llen(self._redis_pending_key)
                )
                if queue_depth > warn_threshold:
                    self.logger.warning("⚠️ Redis pending queue backed up: %d items (threshold: %d)", queue_depth, warn_threshold)
                elif queue_depth > 0:
                    self.logger.debug("Redis pending queue depth: %d items", queue_depth)

                await self._flush_redis_queue()
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                self.logger.exception("Periodic flush failed (%d consecutive failures)", consecutive_failures)
                if consecutive_failures == failure_limit:
                    self.logger.critical("🚨 Writing buffered artifacts to the database has failed %d times in a row; the pipe is waiting longer between attempts and will resume when writes succeed.", failure_limit)

            if consecutive_failures:
                delay = min(10 * (2 ** min(consecutive_failures - 1, 5)), 300)
            else:
                delay = 10
            await asyncio.sleep(delay)

        self.logger.debug("Redis periodic flusher stopped")


    @timed
    def _artifact_store_ready(self) -> bool:
        return bool(self._db_executor and self._item_model and self._session_factory)

    def _note_flush_blocked(self) -> None:
        self._flush_blocked_cycles += 1
        if self._flush_blocked_cycles == 1:
            self.logger.error(
                "Artifact flush blocked: this worker has no usable artifact store, so buffered "
                "artifacts stay in the Redis pending queue (key=%s) instead of being written. "
                "Nothing is lost; a worker with a healthy store will drain the queue.",
                self._redis_pending_key,
            )
        else:
            self.logger.debug(
                "Artifact flush still blocked (%d consecutive cycles); pending queue untouched.",
                self._flush_blocked_cycles,
            )

    def _note_flush_blocked_on_key(self) -> None:
        self._flush_blocked_cycles += 1
        if self._flush_blocked_cycles == 1:
            self.logger.error(
                "Artifact flush blocked: ARTIFACT_ENCRYPTION_KEY cannot be decrypted with the "
                "current WEBUI_SECRET_KEY, so sealed artifacts stay in the Redis pending queue "
                "(key=%s) instead of being written. The queue is not draining and will not "
                "drain until the key is re-entered. Nothing is discarded.",
                self._redis_pending_key,
            )
        else:
            self.logger.debug(
                "Artifact flush still blocked (%d consecutive cycles); pending queue untouched.",
                self._flush_blocked_cycles,
            )

    def _note_flush_ready(self) -> None:
        if self._flush_blocked_cycles:
            self.logger.info(
                "Artifact flush recovered after %d blocked cycle(s); draining the pending queue.",
                self._flush_blocked_cycles,
            )
            self._flush_blocked_cycles = 0

    async def _flush_redis_queue(self) -> None:
        if not (self._redis_enabled and self._redis_client):
            return

        lock_token = secrets.token_hex(16)
        lock_acquired = False
        try:
            lock_acquired = bool(
                await _await_if_needed(
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

            if not self._artifact_store_ready():
                try:
                    self._ensure_artifact_store(self.valves, self.id)
                except Exception:
                    self.logger.debug("Artifact store re-init during flush failed", exc_info=True)
            if not self._artifact_store_ready():
                self._note_flush_blocked()
                return
            if self._artifact_key_unreadable:
                self._note_flush_blocked_on_key()
                return
            self._note_flush_ready()

            entries_by_row: list[tuple[str, dict[str, Any]]] = []
            committed: set[str] = set()
            try:
                malformed = 0
                batch_size = self.valves.DB_BATCH_SIZE
                while len(entries_by_row) < batch_size:
                    data = await _await_if_needed(self._redis_client.lpop(self._redis_pending_key))
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
                        malformed += 1
                        continue
                    try:
                        parsed = json.loads(entry)
                    except json.JSONDecodeError as exc:
                        self.logger.warning("Malformed JSON in pending queue, discarding: %s", exc)
                        malformed += 1
                        continue
                    if not isinstance(parsed, dict):
                        self.logger.warning("Pending queue entry must be an object; discarding malformed payload.")
                        malformed += 1
                        continue
                    entries_by_row.append((entry, parsed))
                if malformed:
                    self.logger.warning("Discarded %d malformed artifact(s) from Redis pending queue.", malformed)
                if not entries_by_row:
                    return

                partition = await self._partition_retired_key_rows(
                    [row for _entry, row in entries_by_row]
                )
                if not all(partition):
                    entries_by_row = [
                        (entry, row)
                        for (entry, row), readable in zip(entries_by_row, partition)
                        if readable
                    ]
                    if not entries_by_row:
                        self.logger.debug(
                            "✅ Successfully flushed 0 artifacts to DB; every entry in this "
                            "batch was unreadable under the current ARTIFACT_ENCRYPTION_KEY"
                        )
                        return

                rows = [row for _entry, row in entries_by_row]
                self.logger.debug("Flushing %d artifact(s) from Redis pending queue to DB (table: %s)", len(rows), self._artifact_table_name or "unknown")
                failure = ""
                try:
                    committed = {
                        identifier
                        for identifier in await self._db_persist_direct(rows)
                        if isinstance(identifier, str) and identifier
                    }
                except Exception as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                    self.logger.exception("❌ DB flush failed! %d artifacts could not be persisted", len(rows))
                if committed:
                    committed_rows = [row for row in rows if row.get("id") in committed]
                    dropped = await self._drop_rows_deleted_while_queued(committed_rows)
                    if dropped and self._redis_client:
                        dropped_ids = set(dropped)
                        keys = [
                            key
                            for key in (
                                self._redis_cache_key(row.get("chat_id"), row.get("id"))
                                for row in committed_rows if row.get("id") in dropped_ids
                            )
                            if key
                        ]
                        if keys:
                            try:
                                await _await_if_needed(self._redis_client.delete(*keys))
                            except Exception as exc:
                                self.logger.warning(
                                    "Redis cache invalidation of dropped rows failed (best-effort): %s",
                                    exc, exc_info=True,
                                )

                unrecoverable = [row for _entry, row in entries_by_row if row.get("payload") is None]
                uncommitted = [
                    entry
                    for entry, row in entries_by_row
                    if row.get("payload") is not None and row.get("id") not in committed
                ]
                if unrecoverable:
                    self.logger.error(
                        "Discarded %d artifact(s) with no payload that can never be persisted (ids=%s). "
                        "Markers referencing them are permanently dangling.",
                        len(unrecoverable),
                        sorted(str(row.get("id")) for row in unrecoverable),
                    )
                if uncommitted:
                    if not failure:
                        self.logger.error(
                            "DB flush reported no error but committed only %d of %d artifact(s); "
                            "returning %d to the pending queue.",
                            len(committed),
                            len(rows),
                            len(uncommitted),
                        )
                    try:
                        await self._redis_requeue_entries(uncommitted)
                        self.logger.debug(
                            "Re-queued %d artifact(s) after an incomplete flush (reason=%s)",
                            len(uncommitted),
                            failure or "uncommitted",
                        )
                    except Exception as requeue_exc:  # pragma: no cover - defensive
                        self.logger.critical(
                            "ARTIFACT LOSS: %d artifact(s) left the pending queue, were not committed, "
                            "and could not be re-queued: %s",
                            len(uncommitted),
                            requeue_exc,
                            exc_info=True,
                        )
                elif not failure:
                    self.logger.debug("✅ Successfully flushed %d artifacts to DB", len(rows))
                if failure:
                    raise RuntimeError(failure) from None
            except asyncio.CancelledError:
                await self._return_popped_entries_on_cancel(entries_by_row, committed)
                raise
        finally:
            if lock_acquired and self._redis_client:
                release_script = (
                    "if redis.call('get', KEYS[1]) == ARGV[1] then "
                    "return redis.call('del', KEYS[1]) "
                    "else return 0 end"
                )
                try:
                    released = await _await_if_needed(
                        self._redis_client.eval(
                            release_script,
                            1,
                            self._redis_flush_lock_key,
                            lock_token,
                        )
                    )
                    try:
                        released_int = int(released)
                    except (TypeError, ValueError):
                        released_int = None
                    if released_int != 1:
                        self.logger.warning(
                            "Redis flush lock was not released (result=%r, key=%s). It may have expired or been replaced.",
                            released,
                            self._redis_flush_lock_key,
                        )
                except Exception:
                    self.logger.debug("Failed to release Redis flush lock", exc_info=True)

    def _redis_cache_key(self, chat_id: str | None, row_id: str | None) -> str | None:
        if not (chat_id and row_id):
            return None
        return f"{self._redis_cache_prefix}:{chat_id}:{row_id}"

    def _redis_deleted_key(self, row_id: str) -> str:
        return f"{self._redis_namespace}:deleted:{row_id}"

    def _redis_active(self) -> bool:
        return bool(self._redis_enabled and self.valves.ENABLE_REDIS_CACHE)

    async def _mark_rows_dropped(self, row_ids: list[str], keep_message_id: str | None = None) -> None:
        if not (row_ids and self._redis_client):
            return
        try:
            pipe = self._redis_client.pipeline()
            for row_id in row_ids:
                pipe.setex(
                    self._redis_deleted_key(row_id),
                    _REDIS_DELETE_MARKER_TTL_SECONDS,
                    _delete_marker_value(keep_message_id),
                )
            await _await_if_needed(pipe.execute())
        except Exception as exc:
            self.logger.warning("Redis delete marker write failed (best-effort): %s", exc, exc_info=True)

    @timed
    async def _redis_cached_owners(self, refs: list[tuple[str, str]]) -> dict[str, Any]:
        if not (self._redis_enabled and self._redis_client):
            return {}
        keys: list[str] = []
        row_ids: list[str] = []
        for chat_id, row_id in refs:
            cache_key = self._redis_cache_key(chat_id, row_id)
            if cache_key:
                keys.append(cache_key)
                row_ids.append(row_id)
        if not keys:
            return {}
        try:
            values = await _await_if_needed(self._redis_client.mget(keys))
        except Exception as exc:
            self.logger.warning("Redis read of cached artifact owners failed (best-effort): %s", exc, exc_info=True)
            return {}
        owners: dict[str, Any] = {}
        for row_id, raw in zip(row_ids, values or []):
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if isinstance(row, dict) and row.get("message_id"):
                owners[row_id] = row["message_id"]
        return owners

    @timed
    async def _drop_rows_deleted_while_queued(self, rows: list[dict[str, Any]]) -> list[str]:
        if not (self._redis_client and self._db_executor):
            return []
        row_ids = [str(row.get("id")) for row in rows if row.get("id")]
        if not row_ids:
            return []
        try:
            markers = await _await_if_needed(
                self._redis_client.mget([self._redis_deleted_key(row_id) for row_id in row_ids])
            )
            owners = {row.get("id"): row.get("message_id") for row in rows if row.get("id")}
            deleted = [
                row_id
                for row_id, marker in zip(row_ids, markers or [])
                if marker is not None and not _marker_spares_row(marker, owners.get(row_id))
            ]
            if deleted:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    self._db_executor, functools.partial(self._delete_artifacts_sync, deleted)
                )
            return deleted
        except Exception as exc:
            self.logger.warning(
                "Removing flushed artifacts deleted while queued failed: %s", exc, exc_info=True
            )
            return []

    def _redis_admits_writes(self) -> bool:
        return not self._redis_valve_off and bool(self._redis_enabled and self._redis_client)

    @timed
    async def _redis_enqueue_rows(self, rows: list[dict[str, Any]]) -> list[str]:
        """Enqueue artifacts into Redis for asynchronous DB flushing."""
        if not rows:
            return []

        if not self._redis_admits_writes():
            return await self._db_persist_direct(rows)
        client = self._redis_client
        assert client is not None

        for row in rows:
            row.setdefault("id", generate_item_id())

        try:
            pipe = client.pipeline()
            serialized_by_id: dict[str, str] = {}
            for row in rows:
                serialized = json.dumps(row, ensure_ascii=False)
                serialized_by_id[row["id"]] = serialized
                pipe.rpush(self._redis_pending_key, serialized)
            await _await_if_needed(pipe.execute())

            await self._redis_cache_rows(rows, serialized=serialized_by_id)
            await _await_if_needed(client.publish(_REDIS_FLUSH_CHANNEL, "flush"))

            self.logger.debug("Enqueued %d artifacts to Redis pending queue", len(rows))
            return [row["id"] for row in rows]
        except Exception as exc:
            self.logger.warning(
                "Redis enqueue failed, falling back to direct DB write: %s", exc, exc_info=True
            )
            return await self._db_persist_direct(rows)

    @timed
    async def _redis_cache_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        chat_id: str | None = None,
        serialized: dict[str, str] | None = None,
    ) -> None:
        if not self._redis_admits_writes():
            return
        client = self._redis_client
        assert client is not None
        try:
            pipe = client.pipeline()
            for row in rows:
                row_payload = row if "payload" in row else {"payload": row}
                cache_key = self._redis_cache_key(row.get("chat_id") or chat_id, row.get("id"))
                if not cache_key:
                    continue
                value = (serialized or {}).get(row.get("id", "")) if "payload" in row else None
                pipe.setex(
                    cache_key, self.valves.REDIS_CACHE_TTL_SECONDS,
                    value if value is not None else json.dumps(row_payload, ensure_ascii=False),
                )
            await _await_if_needed(pipe.execute())
        except Exception as exc:
            self.logger.warning("Redis cache write failed (best-effort): %s", exc, exc_info=True)

    @timed
    async def _redis_requeue_entries(self, entries: list[str]) -> None:
        """Push raw JSON entries back onto the pending queue after a DB failure."""
        if not (entries and self._redis_client):
            return
        pipe = self._redis_client.pipeline()
        for payload in reversed(entries):
            pipe.lpush(self._redis_pending_key, payload)
        await _await_if_needed(pipe.execute())

    async def _push_entries_back(self, entries: list[str]) -> None:
        try:
            await self._redis_requeue_entries(entries)
        except Exception as exc:
            self.logger.critical(
                "ARTIFACT LOSS: %d artifact(s) left the pending queue and could not be re-queued: %s",
                len(entries),
                exc,
                exc_info=True,
            )

    async def _return_popped_entries_on_cancel(
        self,
        entries_by_row: list[tuple[str, dict[str, Any]]],
        committed: set[str],
    ) -> None:
        entries = [
            entry
            for entry, row in entries_by_row
            if row.get("payload") is not None and row.get("id") not in committed
        ]
        if not entries:
            return
        task = asyncio.ensure_future(self._push_entries_back(entries))
        for _ in range(_CANCEL_REQUEUE_POLL_ATTEMPTS):
            if task.done():
                break
            try:
                await asyncio.sleep(_CANCEL_REQUEUE_POLL_SECONDS)
            except asyncio.CancelledError:
                continue
        if task.done() and not task.cancelled():
            exc = task.exception()
            if exc is not None:
                self.logger.critical(
                    "ARTIFACT LOSS: %d artifact(s) left the pending queue and could not be re-queued: %s",
                    len(entries),
                    exc,
                )
            return
        self.logger.critical(
            "ARTIFACT LOSS: %d artifact(s) left the pending queue and their re-queue did not finish before the worker went away",
            len(entries),
        )

    @timed
    async def _redis_fetch_rows(
        self,
        chat_id: str | None,
        item_ids: list[str],
        *,
        message_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        if not (self._redis_enabled and self._redis_client and chat_id and item_ids):
            return {}
        keys: list[str] = []
        id_lookup: list[str] = []
        seen_ids: set[str] = set()
        for item_id in item_ids:
            if item_id in seen_ids:
                continue
            cache_key = self._redis_cache_key(chat_id, item_id)
            if cache_key:
                seen_ids.add(item_id)
                keys.append(cache_key)
                id_lookup.append(item_id)
        if not keys:
            return {}
        try:
            values = await _await_if_needed(self._redis_client.mget(keys))
        except Exception as exc:
            self.logger.warning("Redis read failed, falling back to DB: %s", exc, exc_info=True)
            return {}
        cached, encrypted_rows = await asyncio.to_thread(
            self._cached_rows_from_values, id_lookup, values, message_id
        )
        decrypted: dict[str, dict[str, Any]] = {}
        if encrypted_rows:
            decrypted = await asyncio.to_thread(self._decrypt_many, encrypted_rows)
        for item_id, payload in decrypted.items():
            if isinstance(payload, dict):
                cached[item_id] = payload
        return cached

    def _cached_rows_from_values(
        self,
        id_lookup: list[str],
        values: Any,
        message_id: str | None,
    ) -> tuple[dict[str, dict[str, Any]], list[tuple[str, Any]]]:
        cached: dict[str, dict[str, Any]] = {}
        encrypted_rows: list[tuple[str, Any]] = []
        for item_id, raw in zip(id_lookup, values):
            if not raw:
                continue
            try:
                row_data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if message_id is not None:
                row_message_id = row_data.get("message_id") if isinstance(row_data, dict) else None
                if row_message_id != message_id:
                    continue
            payload = row_data.get("payload", row_data) if isinstance(row_data, dict) else row_data
            recorded: Any = None
            is_encrypted = False
            if isinstance(row_data, dict):
                recorded = row_data.get("is_encrypted")
                is_encrypted = bool(recorded)
            if recorded is None and isinstance(payload, dict) and "enc_v" in payload:
                is_encrypted = "ciphertext" in payload
            if is_encrypted:
                ciphertext = ""
                if isinstance(payload, dict):
                    ciphertext = payload.get("ciphertext", "") or ""
                elif isinstance(row_data, dict) and isinstance(row_data.get("payload"), dict):
                    ciphertext = row_data["payload"].get("ciphertext", "") or ""
                encrypted_rows.append((item_id, ciphertext))
                payload = None
            if isinstance(payload, dict):
                cached[item_id] = payload
        return cached, encrypted_rows

    def _decrypt_many(
        self,
        pairs: list[tuple[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        decrypted: dict[str, dict[str, Any]] = {}
        for item_id, ciphertext in pairs:
            try:
                decrypted[item_id] = self._decrypt_payload(ciphertext)
            except Exception as exc:
                self.logger.warning("Failed to decrypt cached artifact %s: %s", item_id, exc, exc_info=True)
        return decrypted


    @timed
    async def _artifact_cleanup_worker(self) -> None:
        while True:
            try:
                await self._run_cleanup_once()
            except asyncio.CancelledError:  # pragma: no cover - shutdown
                break
            except Exception as exc:
                self.logger.warning("Artifact cleanup failed: %s", exc, exc_info=True)
            interval_hours = self.valves.ARTIFACT_CLEANUP_INTERVAL_HOURS
            interval_seconds = interval_hours * 3600
            jitter = min(600.0, interval_seconds * 0.25)
            await asyncio.sleep(interval_seconds + random.uniform(0, jitter))

    @timed
    async def _run_cleanup_once(self) -> None:
        if not (self._db_executor and self._item_model and self._session_factory):
            return
        cutoff_days = self.valves.ARTIFACT_CLEANUP_DAYS
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=cutoff_days)
        await self._purge_expired_cache_entries(cutoff)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._db_executor,
            functools.partial(self._cleanup_sync, cutoff),
        )

    @staticmethod
    def _artifact_retention_filter(model: Any, cutoff: datetime.datetime) -> Any:
        return model.created_at < cutoff

    @staticmethod
    def _artifact_retention_exclude(model: Any) -> Any:
        return ~model.item_type.in_(_PIPE_OWNED_ROW_TYPES)

    def _expired_cache_key_batch(
        self, cutoff: datetime.datetime, after_id: str | None, limit: int
    ) -> list[tuple[str, str]]:
        if not (self._session_factory and self._item_model):
            return []
        model = self._item_model
        with _db_session(self._session_factory) as session:
            query = session.query(model.id, model.chat_id).filter(model.created_at < cutoff)
            if after_id is not None:
                query = query.filter(model.id > after_id)
            return [(row[1], row[0]) for row in query.order_by(model.id).limit(limit)]

    async def _purge_expired_cache_entries(self, cutoff: datetime.datetime) -> int:
        if not (self._redis_enabled and self._redis_client):
            return 0
        loop = asyncio.get_running_loop()
        after_id: str | None = None
        purged = 0
        while True:
            batch = await loop.run_in_executor(
                self._db_executor,
                functools.partial(
                    self._expired_cache_key_batch,
                    cutoff,
                    after_id,
                    _RETENTION_CACHE_PURGE_BATCH,
                ),
            )
            if not batch:
                return purged
            keys = [key for key in (self._redis_cache_key(chat_id, row_id) for chat_id, row_id in batch) if key]
            if keys:
                try:
                    await _await_if_needed(self._redis_client.delete(*keys))
                except Exception as exc:
                    self.logger.warning(
                        "Redis cache invalidation of expired artifacts failed (best-effort): %s",
                        exc, exc_info=True,
                    )
                    return purged
            purged += len(batch)
            after_id = batch[-1][1]

    def _expired_row_id_bounds(
        self, session: Session, model: Any, cutoff: datetime.datetime
    ) -> tuple[Any, Any]:
        row = (
            session.query(func.min(model.id), func.max(model.id))
            .filter(self._artifact_retention_filter(model, cutoff))
            .filter(self._artifact_retention_exclude(model))
            .one()
        )
        return (None, None) if row is None else (row[0], row[1])

    @timed
    def _cleanup_sync(self, cutoff: datetime.datetime) -> None:
        if not (self._session_factory and self._item_model):
            return
        with _db_session(self._session_factory) as session:
            ulid_lo, ulid_hi = self._expired_row_id_bounds(session, self._item_model, cutoff)
            deleted = (
                session.query(self._item_model)
                .filter(self._artifact_retention_filter(self._item_model, cutoff))
                .filter(self._artifact_retention_exclude(self._item_model))
                .delete(synchronize_session=False)
            )
            left_by_temporary_chats = sum(
                session.query(self._item_model)
                .filter(self._item_model.chat_id.startswith(prefix))
                .delete(synchronize_session=False)
                for prefix in temporary_chat_prefixes()
            )
            session.commit()
            if left_by_temporary_chats:
                self.logger.info("Removed %s artifact row(s) left by temporary chats", left_by_temporary_chats)
            if deleted:
                self.logger.info(
                    "Retention removed %s artifact row(s) last read before %s "
                    "(ulid_range=%s..%s)",
                    deleted,
                    cutoff,
                    ulid_lo,
                    ulid_hi,
                )


    def _db_breaker_allows(self, user_id: str) -> bool:
        if not user_id:
            return True
        self._sweep_db_breakers(time.time())
        window = self._db_breakers.get(user_id)
        if window is None:
            return True
        now = time.time()
        while window and now - window[0] > self._breaker_window_seconds:
            window.popleft()
        if not window:
            self._db_breakers.pop(user_id, None)
        return len(window) < self._breaker_threshold

    def _record_db_failure(self, user_id: str) -> None:
        if user_id:
            now = time.time()
            self._sweep_db_breakers(now, force=True)
            self._db_breakers[user_id].append(now)

    def _reset_db_failure(self, user_id: str) -> None:
        if user_id:
            self._db_breakers.pop(user_id, None)

    # 7. LIFECYCLE MANAGEMENT

    def close(self) -> None:
        """Close background resources cleanly (formerly 'shutdown')."""
        with self._store_lock:
            self._closed = True
            executor = self._db_executor
            self._db_executor = None
        if executor:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
            except Exception:
                self.logger.debug(
                    "Failed to shutdown DB executor cleanly",
                    exc_info=True,
                )


# Helper Functions

def normalize_persisted_item(
    item: dict[str, Any] | None,
    generate_item_id: Callable[[], str] = generate_item_id,
) -> dict[str, Any] | None:
    """Ensure persisted response artifacts match the schema expected by the
    Responses API when replayed via the ``input`` array.

    Args:
        item: The item to normalize.
        generate_item_id: Callable that produces unique item IDs.  Defaults to
            the module-level :func:`generate_item_id`.  Callers may supply an
            alternative for deterministic testing.

    Returns:
        Normalized item dict, or ``None`` when the item is invalid.
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
        if not is_picture_output(output_value):
            normalized["output"] = tool_output_text_and_pictures(output_value)[0]
        return normalized

    if item_type == "function_call":
        name = normalized.get("name")
        arguments = normalized.get("arguments")
        if not (isinstance(name, str) and name.strip()) or arguments is None:
            return None
        if not isinstance(arguments, str):
            try:
                normalized["arguments"] = json.dumps(arguments)
            except (TypeError, ValueError):
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
