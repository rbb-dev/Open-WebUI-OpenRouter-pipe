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
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, cast

# External dependencies
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import Column, String, JSON, Boolean, DateTime, Engine, text, inspect as sa_inspect
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from ..core.timing_logger import timed

# Optional dependencies
try:
    import lz4.frame as lz4frame
except ImportError:
    lz4frame = None

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

# Import ULID constants from config
from ..core.config import (
    ULID_LENGTH,
    ULID_TIME_LENGTH,
    ULID_RANDOM_LENGTH,
    CROCKFORD_ALPHABET,
    _ULID_TIME_MASK,
)

LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Persistence Constants
# -----------------------------------------------------------------------------

# Payload compression flags
_PAYLOAD_FLAG_PLAIN = 0
_PAYLOAD_FLAG_LZ4 = 1

# Encryption version
_ENCRYPTED_PAYLOAD_VERSION = 1
_PAYLOAD_HEADER_SIZE = 1

# Redis pub/sub channel for cache invalidation
_REDIS_FLUSH_CHANNEL = "db-flush"

# Type alias for Redis client
_RedisClient = Any


# -----------------------------------------------------------------------------
# Helper Functions (Module-level)
# -----------------------------------------------------------------------------


@timed
def _encode_crockford(value: int, length: int) -> str:
    """Encode an integer into a fixed-width Crockford base32 string."""
    if value < 0:
        raise ValueError("value must be non-negative")
    chars = ["0"] * length
    for idx in range(length - 1, -1, -1):
        chars[idx] = CROCKFORD_ALPHABET[value & 0x1F]
        value >>= 5
    return "".join(chars)


@timed
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


@timed
def _sanitize_table_fragment(value: str) -> str:
    """Normalize arbitrary identifiers into safe SQL table suffixes."""
    fragment = re.sub(r"[^a-z0-9_]", "_", (value or "").lower())
    fragment = fragment.strip("_") or "pipe"
    if len(fragment) > 62:
        fragment = fragment[:62].rstrip("_") or "pipe"
    return fragment


@timed
async def _wait_for(
    value: Any,
    *,
    timeout: Optional[float] = None,
) -> Any:
    """Return value immediately when it's synchronous, otherwise await it.

    Redis' asyncio client returns synchronous fallbacks (bool/str/list) when a
    pipeline is configured for immediate execution, which caused await to be
    applied to non-awaitables. This helper centralizes the guard.
    """
    import inspect
    if inspect.isawaitable(value):
        if timeout is None:
            return await value
        return await asyncio.wait_for(value, timeout=timeout)
    return value


# -----------------------------------------------------------------------------
# ArtifactStore Class
# -----------------------------------------------------------------------------


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

    @timed
    def __init__(
        self,
        pipe_id: str,
        logger: logging.Logger,
        valves: Any,  # Pipe.Valves reference
        emit_notification_callback: Optional[Callable] = None,
        tool_context_var: Optional[ContextVar] = None,
        user_id_context_var: Optional[ContextVar] = None,
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
        self.valves = valves
        self._emit_notification = emit_notification_callback
        self._TOOL_CONTEXT = tool_context_var
        self._user_id_context = user_id_context_var

        # Initialize all instance state from valves/environment
        self._initialize_encryption_state()
        self._initialize_circuit_breakers()
        self._initialize_redis_state()
        self._initialize_database_state()
        self._initialize_cleanup_state()

    @timed
    def _initialize_encryption_state(self):
        """Initialize encryption and compression state."""
        # Import EncryptedStr locally to avoid circular dependency
        from open_webui_openrouter_pipe.core.config import EncryptedStr

        decrypted_encryption_key = EncryptedStr.decrypt(self.valves.ARTIFACT_ENCRYPTION_KEY)
        self._encryption_key: str = (decrypted_encryption_key or "").strip()
        self._encrypt_all: bool = bool(self.valves.ENCRYPT_ALL)
        self._compression_min_bytes: int = self.valves.MIN_COMPRESS_BYTES
        self._compression_enabled: bool = bool(
            self.valves.ENABLE_LZ4_COMPRESSION and lz4frame is not None
        )
        self._fernet: Fernet | None = None
        self._lz4_warning_emitted = False

    @timed
    def _initialize_circuit_breakers(self):
        """Initialize circuit breaker tracking."""
        breaker_history_size = self.valves.BREAKER_HISTORY_SIZE
        self._breaker_records: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=breaker_history_size)
        )
        self._breaker_threshold = self.valves.BREAKER_MAX_FAILURES
        self._breaker_window_seconds = self.valves.BREAKER_WINDOW_SECONDS
        self._db_breakers: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=breaker_history_size)
        )

    @timed
    def _initialize_redis_state(self):
        """Initialize Redis caching state."""
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
        self._redis_client: Optional[_RedisClient] = None
        self._redis_listener_task: asyncio.Task | None = None
        self._redis_flush_task: asyncio.Task | None = None
        self._redis_ready_task: asyncio.Task | None = None
        self._redis_namespace = (self.id or "openrouter").lower()
        self._redis_pending_key = f"{self._redis_namespace}:pending"
        self._redis_cache_prefix = f"{self._redis_namespace}:artifact"
        self._redis_flush_lock_key = f"{self._redis_namespace}:flush_lock"
        self._redis_ttl = self.valves.REDIS_CACHE_TTL_SECONDS

    @timed
    def _initialize_database_state(self):
        """Initialize SQLAlchemy state."""
        self._engine: Engine | None = None
        self._session_factory: sessionmaker | None = None
        self._item_model: Type[Any] | None = None
        self._artifact_table_name: str | None = None
        self._db_executor: ThreadPoolExecutor | None = None
        self._artifact_store_signature: tuple[str, str] | None = None

    @timed
    def _initialize_cleanup_state(self):
        """Initialize cleanup worker state."""
        self._cleanup_task: asyncio.Task | None = None

    # -----------------------------------------------------------------------------
    # 1. ARTIFACT STORE INITIALIZATION (5 methods)
    # -----------------------------------------------------------------------------

    @timed
    def _ensure_artifact_store(self, valves: Any, pipe_identifier: Optional[str] = None) -> None:
        """Configure encryption/compression + ensure the backing table exists."""
        # Import EncryptedStr locally to avoid circular dependency
        from open_webui_openrouter_pipe.core.config import EncryptedStr

        decrypted_encryption_key = EncryptedStr.decrypt(valves.ARTIFACT_ENCRYPTION_KEY)
        encryption_key = (decrypted_encryption_key or "").strip()
        self._encryption_key = encryption_key
        self._encrypt_all = valves.ENCRYPT_ALL
        self._compression_min_bytes = valves.MIN_COMPRESS_BYTES

        wants_compression = valves.ENABLE_LZ4_COMPRESSION
        compression_enabled = wants_compression and lz4frame is not None
        if wants_compression and lz4frame is None and not self._lz4_warning_emitted:
            self.logger.warning("LZ4 compression requested but the 'lz4' package is not available. Artifacts will be stored without compression.")
            self._lz4_warning_emitted = True
        self._compression_enabled = compression_enabled

        pipe_identifier = pipe_identifier or self.id
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
        except Exception:
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
            except Exception:
                schema = None

        if schema is None:
            try:
                import importlib

                owui_env = importlib.import_module("open_webui.env")  # type: ignore

                candidate = getattr(owui_env, "DATABASE_SCHEMA", None)
                if isinstance(candidate, str) and candidate.strip():
                    schema = candidate.strip()
                    details["schema_source"] = "open_webui.env.DATABASE_SCHEMA"
            except Exception:
                schema = None

        db_context = getattr(owui_db, "get_db_context", None) or getattr(owui_db, "get_db", None)
        if db_context is not None:
            try:
                cm = db_context() if callable(db_context) else db_context
                if hasattr(cm, "__enter__"):
                    with cast(contextlib.AbstractContextManager[Any], cm) as session:
                        if session is not None:
                            try:
                                engine = session.get_bind()
                                details["engine_source"] = "get_db_context.get_bind"
                            except Exception:
                                engine = getattr(session, "bind", None) or getattr(session, "engine", None)
                                if engine is not None:
                                    details["engine_source"] = "get_db_context.bind"
            except Exception:
                engine = None

        if engine is None:
            for attr in ("engine", "ENGINE", "bind", "BIND"):
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
        pipe_identifier: Optional[str] = None,
        *,
        table_fragment: Optional[str] = None,
    ) -> None:
        """Initialize the per-pipe SQLAlchemy model + executor for artifact storage."""
        engine: Any | None = None
        schema: str | None = None

        try:
            from open_webui.internal import db as owui_db  # type: ignore
        except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
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
        encryption_key = self._encryption_key
        hash_source = f"{encryption_key}{pipe_identifier}".encode("utf-8", "ignore")
        key_hash = hashlib.sha256(hash_source).hexdigest()
        table_fragment = table_fragment or _sanitize_table_fragment(pipe_identifier)
        table_name = f"response_items_{table_fragment}_{key_hash[:8]}"
        class_name = f"ResponseItem_{table_fragment}_{key_hash[:4]}"

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
            "__table_args__": table_args,
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
    @timed
    def _quote_identifier(identifier: str) -> str:
        """Return a double-quoted identifier safe for direct SQL execution."""
        value = (identifier or "").replace('"', '""')
        return f'"{value}"'

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

    # -----------------------------------------------------------------------------
    # 2. ENCRYPTION/COMPRESSION (9 methods)
    # -----------------------------------------------------------------------------

    @timed
    def _get_fernet(self) -> Fernet | None:
        """Return (and cache) the Fernet helper derived from the encryption key."""
        if not self._encryption_key:
            return None
        if self._fernet is None:
            digest = hashlib.sha256(self._encryption_key.encode("utf-8")).digest()
            key = base64.urlsafe_b64encode(digest)
            self._fernet = Fernet(key)
        return self._fernet

    @timed
    def _should_encrypt(self, item_type: str) -> bool:
        """Determine whether a payload of ``item_type`` must be encrypted."""
        if not self._encryption_key:
            return False
        if self._encrypt_all:
            return True
        return (item_type or "").lower() == "reasoning"

    @timed
    def _serialize_payload_bytes(self, payload: dict[str, Any]) -> bytes:
        """Return compact JSON bytes for ``payload``."""
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @timed
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

    @timed
    def _encode_payload_bytes(self, payload: dict[str, Any]) -> bytes:
        """Serialize payload bytes and prepend a compression flag header."""
        serialized = self._serialize_payload_bytes(payload)
        data, compressed = self._maybe_compress_payload(serialized)
        flag = _PAYLOAD_FLAG_LZ4 if compressed else _PAYLOAD_FLAG_PLAIN
        return bytes([flag]) + data

    @timed
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

    @timed
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

    @timed
    def _encrypt_payload(self, payload: dict[str, Any]) -> str:
        """Encrypt payload bytes using the configured Fernet helper."""
        fernet = self._get_fernet()
        if not fernet:
            raise RuntimeError("Encryption requested but ARTIFACT_ENCRYPTION_KEY is not configured.")
        encoded = self._encode_payload_bytes(payload)
        return fernet.encrypt(encoded).decode("utf-8")

    @timed
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

    @timed
    def _encrypt_if_needed(self, item_type: str, payload: dict[str, Any]) -> tuple[Any, bool]:
        """Optionally encrypt ``payload`` depending on the item type."""
        if not self._should_encrypt(item_type):
            return payload, False
        encrypted = self._encrypt_payload(payload)
        return {"ciphertext": encrypted, "enc_v": _ENCRYPTED_PAYLOAD_VERSION}, True

    # -----------------------------------------------------------------------------
    # 3. DATABASE OPERATIONS (11 methods)
    # -----------------------------------------------------------------------------

    @timed
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

    @timed
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

    @timed
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

    @timed
    async def _db_persist(self, rows: list[dict[str, Any]]) -> list[str]:
        """Persist artifacts, optionally via Redis write-behind."""
        if not rows:
            return []

        # Import SessionLogger locally to avoid circular dependency
        from open_webui_openrouter_pipe.core.logging_system import SessionLogger

        user_id = SessionLogger.user_id.get() or ""
        if not self._db_breaker_allows(user_id):
            self.logger.warning("DB writes disabled for user_id=%s due to repeated failures", user_id)
            context = self._TOOL_CONTEXT.get() if self._TOOL_CONTEXT else None
            if self._emit_notification:
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

    @timed
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

    @timed
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

    @timed
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

        # Best-effort "touch" for retention: update created_at on DB access (not on Redis hits).
        # This must never crash the read path.
        if rows:
            try:
                touched_ids = [getattr(row, "id", None) for row in rows]
                touched_ids = [
                    item_id
                    for item_id in touched_ids
                    if isinstance(item_id, str) and item_id
                ]
                if touched_ids:
                    touch_session: Session = self._session_factory()  # type: ignore[call-arg]
                    try:
                        now = datetime.datetime.now(datetime.UTC)
                        touch_query = touch_session.query(model).filter(model.chat_id == chat_id)
                        touch_query = touch_query.filter(model.id.in_(touched_ids))
                        if message_id:
                            touch_query = touch_query.filter(model.message_id == message_id)
                        touch_query.update({model.created_at: now}, synchronize_session=False)
                        touch_session.commit()
                    except Exception as exc:
                        with contextlib.suppress(Exception):
                            touch_session.rollback()
                        self.logger.debug(
                            "Artifact touch skipped (chat_id=%s, message_id=%s, rows=%s): %s",
                            chat_id,
                            message_id or "",
                            len(touched_ids),
                            exc,
                            exc_info=self.logger.isEnabledFor(logging.DEBUG),
                        )
                    finally:
                        touch_session.close()
            except Exception as exc:
                self.logger.debug(
                    "Artifact touch failed (chat_id=%s, message_id=%s): %s",
                    chat_id,
                    message_id or "",
                    exc,
                    exc_info=self.logger.isEnabledFor(logging.DEBUG),
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
                    self.logger.warning("Failed to decrypt artifact %s: %s", row.id, exc, exc_info=self.logger.isEnabledFor(logging.DEBUG))
                    continue
            if isinstance(payload, dict):
                results[row.id] = payload
        return results

    @timed
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

        # Import SessionLogger locally to avoid circular dependency
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

    @timed
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

    @timed
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

    @timed
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

    # -----------------------------------------------------------------------------
    # 4. REDIS CACHE (8 methods)
    # -----------------------------------------------------------------------------

    @timed
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

    @timed
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


    @timed
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
                    released = await _wait_for(
                        self._redis_client.eval(
                            release_script,
                            1,
                            self._redis_flush_lock_key,
                            lock_token,
                        )
                    )
                    try:
                        released_int = int(released)
                    except Exception:
                        released_int = None
                    if released_int != 1:
                        self.logger.warning(
                            "Redis flush lock was not released (result=%r, key=%s). It may have expired or been replaced.",
                            released,
                            self._redis_flush_lock_key,
                        )
                except Exception:
                    # Redis errors during lock release are non-fatal - continue pipe operation
                    self.logger.debug("Failed to release Redis flush lock", exc_info=self.logger.isEnabledFor(logging.DEBUG))

    @timed
    def _redis_cache_key(self, chat_id: Optional[str], row_id: Optional[str]) -> Optional[str]:
        if not (chat_id and row_id):
            return None
        return f"{self._redis_cache_prefix}:{chat_id}:{row_id}"

    @timed
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

    @timed
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

    @timed
    async def _redis_requeue_entries(self, entries: list[str]) -> None:
        """Push raw JSON entries back onto the pending queue after a DB failure."""
        if not (entries and self._redis_client):
            return
        pipe = self._redis_client.pipeline()
        for payload in reversed(entries):
            pipe.lpush(self._redis_pending_key, payload)
        pipe.expire(self._redis_pending_key, max(self._redis_ttl, 60))
        await _wait_for(pipe.execute())

    @timed
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

    # -----------------------------------------------------------------------------
    # 5. CLEANUP WORKERS (3 methods)
    # -----------------------------------------------------------------------------

    @timed
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

    @timed
    async def _run_cleanup_once(self) -> None:
        if not (self._item_model and self._session_factory):
            return
        cutoff_days = self.valves.ARTIFACT_CLEANUP_DAYS
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=cutoff_days)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._db_executor,
            functools.partial(self._cleanup_sync, cutoff),
        )

    @timed
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

    # -----------------------------------------------------------------------------
    # 6. CIRCUIT BREAKERS (3 methods)
    # -----------------------------------------------------------------------------

    @timed
    def _db_breaker_allows(self, user_id: str) -> bool:
        if not user_id:
            return True
        window = self._db_breakers[user_id]
        now = time.time()
        while window and now - window[0] > self._breaker_window_seconds:
            window.popleft()
        return len(window) < self._breaker_threshold

    @timed
    def _record_db_failure(self, user_id: str) -> None:
        if user_id:
            self._db_breakers[user_id].append(time.time())

    @timed
    def _reset_db_failure(self, user_id: str) -> None:
        if user_id and user_id in self._db_breakers:
            self._db_breakers[user_id].clear()

    @timed
    def _record_failure(self, user_id: str) -> None:
        """Record a generic failure (fallback for compatibility)."""
        if not user_id:
            return
        self._breaker_records[user_id].append(time.time())

    # -----------------------------------------------------------------------------
    # 7. LIFECYCLE MANAGEMENT
    # -----------------------------------------------------------------------------

    @timed
    def shutdown(self) -> None:
        """Shutdown background resources cleanly."""
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
                    exc_info=self.logger.isEnabledFor(logging.DEBUG),
                )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

@timed
def _normalize_persisted_item(item: Optional[Dict[str, Any]], generate_item_id: Callable[[], str]) -> Optional[Dict[str, Any]]:
    """
    Ensure persisted response artifacts match the schema expected by the
    Responses API when replayed via the `input` array.
    
    Args:
        item: The item to normalize
        generate_item_id: Function to generate unique item IDs
    """
    if not isinstance(item, dict):
        return None

    item_type = item.get("type")
    if not item_type:
        return None

    normalized = dict(item)

    @timed
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


@timed
def normalize_persisted_item(item: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convenience wrapper for _normalize_persisted_item using the module's generate_item_id.

    Ensure persisted response artifacts match the schema expected by the Responses API.

    Args:
        item: The item to normalize

    Returns:
        Normalized item dict or None
    """
    return _normalize_persisted_item(item, generate_item_id)
