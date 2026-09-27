"""Valve-gated usage-record persistence for the pipe_dashboard plugin.

One row per completed request (chat turn or task call), written by a
per-worker daemon thread through the artifact store's DB machinery. The
table name derives from the store's ``table_suffix()`` so it always sits
next to the artifact table; creation reuses the store's race-guarded DDL
path. Retention is enforced by a jittered asyncio purge task guarded by a
cross-worker DB-row lock in the artifact table.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import queue
import random
import threading
import time
from collections.abc import Callable
from typing import Any

from ...core.utils import _stable_crockford_id
from ...storage.owui_files import is_temporary_chat, temporary_chat_prefixes
from ...storage.persistence import ArtifactStore, _db_session, generate_item_id

logger = logging.getLogger(__name__)

_US_BATCH_MAX = 50
_US_QUEUE_MAX = 1000
_US_JOIN_TIMEOUT = 2.0
_US_PURGE_INTERVAL_S = 900.0
_US_PURGE_JITTER_S = 60.0
_US_LOCK_STALE_S = 600.0
_US_DROP_WARN_EVERY = 50
_US_RECONCILE_RETRY_S = 300.0

USAGE_ROW_FIELDS = (
    "ts",
    "started_at",
    "kind",
    "user_id",
    "user_name",
    "chat_id",
    "session_id",
    "model_id",
    "task_name",
    "status",
    "duration_ms",
    "tokens_in",
    "tokens_out",
    "tokens_reasoning",
    "tokens_cached",
    "tools_ok",
    "tools_failed",
    "retries",
    "cost",
    "cache_savings",
    "worker_pid",
)


def usage_ts_from_epoch(epoch: float) -> datetime.datetime:
    """An epoch second as the usage table stores it: naive, in the server's local zone.

    `ts` and `started_at` are `Column(DateTime)` -- no timezone -- so whatever frame the
    first writer picked is the frame every reader, filter and purge boundary must use.
    The obvious thing to write, `datetime.now(UTC)`, is a different instant by the UTC
    offset, and nothing raises: the purge deletes rows from a window shifted by that
    offset, and the previous-period filter reports the wrong span. Both are silent.
    """
    return datetime.datetime.fromtimestamp(epoch, tz=datetime.UTC).astimezone().replace(tzinfo=None)


def epoch_from_usage_ts(value: datetime.datetime) -> float:
    """The inverse of `usage_ts_from_epoch`.

    A naive datetime's `.timestamp()` interprets it as local time, which is right only
    because that is the frame the column holds. Paired here so the two move together.
    """
    return value.timestamp()


def _usage_model_columns() -> dict[str, Any]:
    from sqlalchemy import Column, DateTime, Float, Integer, String

    return {
        "ts": Column(DateTime, index=True, nullable=False),
        "started_at": Column(DateTime),
        "kind": Column(String(8), index=True),
        "user_id": Column(String(64), index=True),
        "user_name": Column(String(128)),
        "chat_id": Column(String(64), index=True),
        "session_id": Column(String(64)),
        "model_id": Column(String(128), index=True),
        "task_name": Column(String(32), nullable=True),
        "status": Column(String(12)),
        "duration_ms": Column(Integer),
        "tokens_in": Column(Integer),
        "tokens_out": Column(Integer),
        "tokens_reasoning": Column(Integer),
        "tokens_cached": Column(Integer),
        "tools_ok": Column(Integer),
        "tools_failed": Column(Integer),
        "retries": Column(Integer),
        "cost": Column(Float),
        "cache_savings": Column(Float),
        "worker_pid": Column(Integer),
    }


class UsageStore:
    """Per-worker usage writer mirroring the session-log manager thread pattern."""

    def __init__(self, queue_max: int = _US_QUEUE_MAX) -> None:
        self._store: Any = None
        self._model: Any = None
        self._table_name: str | None = None
        self._signature: tuple[Any, ...] | None = None
        self._reconcile_failed: tuple[Any, ...] | None = None
        self._reconcile_failed_at = 0.0
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_max)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._dropped = 0
        self._purge_task: asyncio.Task | None = None
        self._retention_days_fn: Callable[[], int] | None = None

    @property
    def enabled(self) -> bool:
        return self._model is not None

    @property
    def writer_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def table_name(self) -> str | None:
        return self._table_name

    def ensure(self, store: Any) -> bool:
        """Build the usage model and create its table; idempotent, fail-safe."""
        try:
            engine = getattr(store, "_engine", None)
            session_factory = getattr(store, "_session_factory", None)
            if engine is None or session_factory is None:
                return False
            suffix = store.table_suffix()
            signature = (id(engine), suffix)
            if self._signature == signature:
                if self._model is not None:
                    return True
                if (
                    self._reconcile_failed == signature
                    and time.monotonic() - self._reconcile_failed_at < _US_RECONCILE_RETRY_S
                ):
                    return False
            else:
                self._reconcile_failed = None
                self._reconcile_failed_at = 0.0

            from sqlalchemy import Column, String
            from sqlalchemy.orm import declarative_base

            table_name = f"dashboard_{suffix}"
            item_table = getattr(getattr(store, "_item_model", None), "__table__", None)
            schema_name = getattr(item_table, "schema", None)
            table_args: dict[str, Any] = {"extend_existing": True}
            if schema_name:
                table_args["schema"] = schema_name
            base = declarative_base()
            attrs: dict[str, Any] = {
                "__tablename__": table_name,
                "__table_args__": table_args,
                "id": Column(String(26), primary_key=True),
            }
            for _name, _column in _usage_model_columns().items():
                attrs[_name] = _column
            model = type(f"PipeUsage_{suffix[:12]}", (base,), attrs)
            if not store._create_table_with_race_guard(model.__table__, engine, table_name):
                return False
            if not self._reconcile_schema(model.__table__, engine, table_name, schema_name, store):
                self._signature = signature
                self._reconcile_failed = signature
                self._reconcile_failed_at = time.monotonic()
                return False
            self._store = store
            self._model = model
            self._table_name = table_name
            self._signature = signature
            self._reconcile_failed = None
            self._reconcile_failed_at = 0.0
            return True
        except Exception:
            logger.debug("usage store ensure failed", exc_info=True)
            return False

    def _reconcile_schema(
        self,
        table: Any,
        engine: Any,
        table_name: str,
        schema_name: str | None,
        store: Any = None,
    ) -> bool:
        from sqlalchemy import inspect as sa_inspect
        from sqlalchemy import text
        from sqlalchemy.exc import DuplicateColumnError
        from sqlalchemy.schema import CreateColumn

        def _present() -> set[str]:
            if schema_name:
                return {c["name"] for c in sa_inspect(engine).get_columns(table_name, schema=schema_name)}
            return {c["name"] for c in sa_inspect(engine).get_columns(table_name)}

        try:
            present = _present()
            qualified = ArtifactStore._quote_identifier(table_name)
            if schema_name:
                qualified = f"{ArtifactStore._quote_identifier(schema_name)}.{qualified}"
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
                except Exception as exc:  # noqa: BLE001 - any other DDL failure is a gate failure with a stated reason
                    message = str(exc).lower()
                    if (
                        "duplicate column" in message
                        or "already exists" in message
                        or isinstance(exc, DuplicateColumnError)
                    ):
                        logger.debug("usage table column already added by another worker: %s", col.name)
                        continue
                    logger.warning("usage table column could not be added: %s: %s", col.name, exc)
                    return False
                added.append(col.name)
            if added:
                logger.info("usage table %s reconciled with columns: %s", table_name, ", ".join(added))
                if store is not None:
                    store._create_declared_indexes(table, engine, table_name)
            missing_pk = [
                col.name for col in table.columns if col.primary_key and col.name not in _present()
            ]
            if missing_pk:
                logger.warning(
                    "usage table %s is missing its primary key: %s; every write will fail",
                    table_name,
                    ", ".join(missing_pk),
                )
                return False
            missing = {col.name for col in table.columns if not col.primary_key} - _present()
            if missing:
                logger.warning(
                    "usage table %s still lacks columns after reconciliation: %s",
                    table_name,
                    ", ".join(sorted(missing)),
                )
                return False
            return True
        except Exception:
            logger.debug("usage table reconciliation failed", exc_info=True)
            return False

    def record(self, row: dict[str, Any]) -> None:
        """Enqueue one usage row; non-blocking, drop-oldest under overload."""
        if self._model is None:
            return
        self._start_thread()
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(row)
            except queue.Full:
                pass
            self._dropped += 1
            if self._dropped % _US_DROP_WARN_EVERY == 1:
                logger.warning("usage queue overloaded; %d rows dropped so far", self._dropped)

    def _start_thread(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="openrouter-usage-writer",
            daemon=True,
        )
        self._thread.start()

    def _writer_loop(self) -> None:
        while True:
            batch: list[dict[str, Any]] = []
            try:
                item = self._queue.get(timeout=0.5)
                if item is not None:
                    batch.append(item)
            except queue.Empty:
                pass
            try:
                while len(batch) < _US_BATCH_MAX:
                    extra = self._queue.get_nowait()
                    if extra is not None:
                        batch.append(extra)
            except queue.Empty:
                pass
            if batch:
                try:
                    self._persist_sync(batch)
                except Exception:
                    logger.debug("usage batch persist failed", exc_info=True)
            if self._stop_event.is_set() and self._queue.qsize() == 0:
                break

    def _persist_sync(self, rows: list[dict[str, Any]]) -> None:
        store = self._store
        model = self._model
        if store is None or model is None:
            return
        session_factory = getattr(store, "_session_factory", None)
        if session_factory is None:
            return
        instances = []
        for row in rows:
            data = {key: row.get(key) for key in USAGE_ROW_FIELDS}
            if is_temporary_chat(data["chat_id"]):
                data["chat_id"] = data["session_id"] = ""
            data["id"] = row.get("id") or generate_item_id()
            instances.append(model(**data))
        with _db_session(session_factory) as session:
            session.add_all(instances)
            session.commit()

    def start_purge_task(self, retention_days_fn: Callable[[], int]) -> None:
        """Start the jittered retention purge loop on the running loop."""
        self._retention_days_fn = retention_days_fn
        task = self._purge_task
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._purge_task = loop.create_task(self._purge_loop(), name="openrouter-usage-purge")

    async def _purge_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(_US_PURGE_INTERVAL_S + random.uniform(0, _US_PURGE_JITTER_S))
                await self._run_purge_once()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("usage purge iteration failed", exc_info=True)

    async def _run_purge_once(self) -> None:
        store = self._store
        if store is None or self._model is None:
            return
        cutoff = self._purge_cutoff()
        executor = getattr(store, "_db_executor", None)
        if executor is None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, self._purge_sync, cutoff)

    def _purge_cutoff(self) -> datetime.datetime:
        days = 30
        fn = self._retention_days_fn
        if fn is not None:
            try:
                days = int(fn())
            except Exception:
                logger.warning(
                    "usage store: retention-days valve is unreadable; falling back to %d days",
                    days,
                    exc_info=True,
                )
        days = max(1, days)
        return usage_ts_from_epoch(time.time() - days * 86400.0)

    def _purge_sync(self, cutoff: datetime.datetime) -> None:
        store = self._store
        model = self._model
        if store is None or model is None or self._table_name is None:
            return
        lock_id = _stable_crockford_id(f"{self._table_name}:purge")
        item_model = getattr(store, "_item_model", None)
        acquired = True
        if item_model is not None:
            acquired = self._acquire_purge_lock(store, item_model, lock_id)
            if not acquired:
                return
        try:
            session_factory = getattr(store, "_session_factory", None)
            if session_factory is None:
                return
            with _db_session(session_factory) as session:
                session.query(model).filter(model.ts < cutoff).delete(synchronize_session=False)
                for prefix in temporary_chat_prefixes():
                    session.query(model).filter(model.chat_id.startswith(prefix)).update(
                        {"chat_id": "", "session_id": ""}, synchronize_session=False
                    )
                session.commit()
        finally:
            if item_model is not None and acquired:
                try:
                    store._delete_artifacts_sync([lock_id])
                except Exception:
                    logger.debug("purge lock release failed", exc_info=True)

    def _acquire_purge_lock(self, store: Any, item_model: Any, lock_id: str) -> bool:
        try:
            self._reap_stale_lock(store, item_model, lock_id)
            lock_row = {
                "id": lock_id,
                "chat_id": "dashboard",
                "message_id": "purge",
                "model_id": "",
                "item_type": "dashboard_purge_lock",
                "payload": {"pid": os.getpid(), "claimed_at": time.time()},
                "is_encrypted": False,
                "created_at": datetime.datetime.now(datetime.UTC),
            }
            return bool(store._try_acquire_lock_sync(lock_row))
        except Exception:
            logger.debug("purge lock acquire failed; proceeding unlocked", exc_info=True)
            return True

    def _reap_stale_lock(self, store: Any, item_model: Any, lock_id: str) -> None:
        session_factory = getattr(store, "_session_factory", None)
        if session_factory is None:
            return
        stale_before = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=_US_LOCK_STALE_S)
        try:
            with _db_session(session_factory) as session:
                session.query(item_model).filter(
                    item_model.id == lock_id,
                    item_model.created_at < stale_before,
                ).delete(synchronize_session=False)
                session.commit()
        except Exception:
            logger.debug("purge lock reap failed", exc_info=True)

    def _table_info_sync(self) -> dict[str, Any]:
        """Record count + approximate on-disk size for the usage table."""
        info: dict[str, Any] = {"records": None, "approx_bytes": None}
        store = self._store
        model = self._model
        if store is None or model is None or self._table_name is None:
            return info
        session_factory = getattr(store, "_session_factory", None)
        engine = getattr(store, "_engine", None)
        if session_factory is None or engine is None:
            return info
        try:
            from sqlalchemy import func, text

            with _db_session(session_factory) as session:
                info["records"] = int(session.query(func.count(model.id)).scalar() or 0)
                dialect = str(getattr(engine, "dialect", None) and engine.dialect.name or "")
                try:
                    if dialect == "postgresql":
                        qualified = model.__table__.fullname
                        size = session.execute(
                            text("SELECT pg_total_relation_size(:tbl)"), {"tbl": qualified}
                        ).scalar()
                        info["approx_bytes"] = int(size) if size is not None else None
                    elif dialect == "sqlite":
                        size = session.execute(
                            text("SELECT SUM(pgsize) FROM dbstat WHERE name = :tbl"),
                            {"tbl": self._table_name},
                        ).scalar()
                        info["approx_bytes"] = int(size) if size is not None else None
                except Exception:
                    logger.debug("usage table size query failed", exc_info=True)
                    info["approx_bytes"] = None
        except Exception:
            logger.debug("usage table info failed", exc_info=True)
        return info

    async def table_info(self) -> dict[str, Any]:
        store = self._store
        executor = getattr(store, "_db_executor", None) if store is not None else None
        if executor is None:
            return {"records": None, "approx_bytes": None}
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self._table_info_sync)

    def signal_stop(self) -> Any:
        """Signal the writer to drain-and-exit and cancel the purge task.

        Loop-safe and fast: sets the stop flag, wakes the writer, cancels the
        purge task. Does NOT join the writer thread — call ``join_writer`` off
        the event loop for that. Returns the cancelled purge task (awaitable)
        or ``None``.
        """
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        task = self._purge_task
        self._purge_task = None
        if task is not None and not task.done():
            task.cancel()
            return task
        return None

    def join_writer(self, timeout: float = _US_JOIN_TIMEOUT) -> None:
        """Block until the writer thread drains and exits (call OFF the loop)."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(
                    "usage writer did not drain within %.1fs; %d rows may be lost",
                    timeout, self._queue.qsize(),
                )
        self._thread = None

    def stop(self) -> Any:
        """Synchronous stop (signal + in-line join). Prefer signal_stop +
        an off-loop join_writer from an async caller to avoid stalling the loop.
        """
        task = self.signal_stop()
        self.join_writer()
        return task
