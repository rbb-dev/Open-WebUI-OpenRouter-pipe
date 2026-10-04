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
import inspect
import logging
import os
import queue
import random
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any, NamedTuple

from ...core.utils import _stable_crockford_id
from ...core.warn_latch import warn_level
from ...storage.owui_files import is_temporary_chat, temporary_chat_prefixes
from ...storage.persistence import (
    ArtifactStore,
    _db_session,
    _other_installed_fragments,
    generate_item_id,
    raw_valve_column_decodes,
)

logger = logging.getLogger(__name__)

_warned_usage_table_create: dict[str, float] = {}

_US_BATCH_MAX = 50
_US_QUEUE_MAX = 1000
_US_JOIN_TIMEOUT = 2.0
_US_PURGE_INTERVAL_S = 900.0
_US_PURGE_JITTER_S = 60.0
_US_LOCK_STALE_S = 600.0
_warned_purge_lock: dict[str, float] = {}
_US_DROP_WARN_EVERY = 50
_US_RECONCILE_RETRY_S = 300.0
_US_POLL_INTERVAL_S = 0.5
_US_PERSIST_WARN_COOLDOWN_S = 300.0
_USAGE_TABLE_PREFIX = "dashboard_"
_US_SHARED_FRAGMENT_COOLDOWN_S = 3600.0

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
    "tools_skipped",
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
        "tools_skipped": Column(Integer),
        "retries": Column(Integer),
        "cost": Column(Float),
        "cache_savings": Column(Float),
        "worker_pid": Column(Integer),
    }


def _declared_lengths() -> dict[str, int | None]:
    from sqlalchemy import String

    out: dict[str, int | None] = {}
    for name, column in _usage_model_columns().items():
        col_type = column.type
        out[name] = col_type.length if isinstance(col_type, String) else None
    return out


class _Published(NamedTuple):
    store: Any = None
    model: Any = None
    table_name: str | None = None
    signature: tuple[Any, ...] | None = None
    model_signature: tuple[Any, ...] | None = None


_EMPTY_PUBLISHED = _Published()


class UsageStore:
    """Per-worker usage writer mirroring the session-log manager thread pattern."""

    def __init__(self, queue_max: int = _US_QUEUE_MAX) -> None:
        self._published = _EMPTY_PUBLISHED
        self._ensure_lock = threading.Lock()
        self._reconcile_failed: tuple[Any, ...] | None = None
        self._reconcile_failed_at = 0.0
        self._held: list[dict[str, Any]] = []
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_max)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stopped = False
        self._dropped = 0
        self._persist_failed = 0
        self._warned: dict[str, float] = {}
        self._declared_lengths = _declared_lengths()
        self._effective_widths: dict[str, int | None] = dict(self._declared_lengths)
        self._width_warned: set[str] = set()
        self._purge_task: asyncio.Task | None = None
        self._retention_days_fn: Callable[[], int | Awaitable[int]] | None = None
        self._table_absent: tuple[Any, ...] | None = None

    @property
    def _store(self) -> Any:
        return self._published.store

    @property
    def _model(self) -> Any:
        return self._published.model

    @property
    def _table_name(self) -> str | None:
        return self._published.table_name

    @property
    def _signature(self) -> tuple[Any, ...] | None:
        return self._published.signature

    @property
    def _model_signature(self) -> tuple[Any, ...] | None:
        return self._published.model_signature

    def _publish(self, published: _Published) -> None:
        self._published = published

    def _retract_model(self) -> None:
        self._publish(self._published._replace(model=None, model_signature=None))

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
    def persist_failed(self) -> int:
        return self._persist_failed

    def _fit_row(self, data: dict[str, Any]) -> dict[str, Any]:
        fitted = dict(data)
        for name, width in self._effective_widths.items():
            if width is None:
                continue
            value = fitted.get(name)
            if isinstance(value, str) and len(value) > width:
                fitted[name] = value[:width]
        return fitted

    @property
    def table_name(self) -> str | None:
        return self._table_name

    def _reflected_widths(self, engine: Any, table_name: str, schema_name: str | None) -> dict[str, int | None]:
        from sqlalchemy import String
        from sqlalchemy import inspect as sa_inspect

        if schema_name:
            columns = sa_inspect(engine).get_columns(table_name, schema=schema_name)
        else:
            columns = sa_inspect(engine).get_columns(table_name)
        out: dict[str, int | None] = {}
        for column in columns:
            col_type = column.get("type")
            out[column["name"]] = col_type.length if isinstance(col_type, String) else None
        return out

    def _reconcile_widths(
        self,
        engine: Any,
        table_name: str,
        schema_name: str | None,
        reflected: dict[str, int | None],
    ) -> dict[str, int | None]:
        effective = dict(self._declared_lengths)
        for name, declared in self._declared_lengths.items():
            actual = reflected.get(name, declared)
            if declared is None or actual is None or actual >= declared:
                continue
            if self._widen_column(engine, table_name, schema_name, name, actual, declared):
                effective[name] = declared
            else:
                if name not in self._width_warned:
                    self._width_warned.add(name)
                    logger.warning(
                        "usage table %s column %s is VARCHAR(%s) on disk but this release "
                        "declares VARCHAR(%s); the widen was refused, so usage rows will be "
                        "fitted to %s characters. Widen it with: ALTER TABLE %s ALTER COLUMN "
                        "%s TYPE VARCHAR(%s)",
                        table_name, name, actual, declared, actual,
                        ArtifactStore._quote_identifier(table_name), name, declared,
                    )
                else:
                    logger.debug(
                        "usage table %s column %s is still VARCHAR(%s) against a declared "
                        "VARCHAR(%s); usage rows keep being fitted to %s characters",
                        table_name, name, actual, declared, actual,
                    )
                effective[name] = actual
        self._effective_widths = effective
        return effective

    def _widen_column(
        self,
        engine: Any,
        table_name: str,
        schema_name: str | None,
        name: str,
        actual: int | None,
        declared: int | None,
    ) -> bool:
        from sqlalchemy import text

        qualified = ArtifactStore._quote_identifier(table_name)
        if schema_name:
            qualified = f"{ArtifactStore._quote_identifier(schema_name)}.{qualified}"
        column = ArtifactStore._quote_identifier(name)
        statements = [
            f"ALTER TABLE {qualified} ALTER COLUMN {column} TYPE VARCHAR({declared})",
            f"ALTER TABLE {qualified} MODIFY COLUMN {column} VARCHAR({declared})",
        ]
        for statement in statements:
            try:
                with engine.begin() as conn:
                    conn.execute(text(statement))
            except Exception as exc:  # noqa: BLE001 - any refusal leaves the width as it is
                logger.debug(
                    "usage table widen not accepted on this dialect: %s", type(exc).__name__
                )
                continue
            try:
                check = self._reflected_widths(engine, table_name, schema_name).get(name)
            except Exception:  # noqa: BLE001 - an uninspectable table is not a widened one
                check = None
            if check is not None and actual is not None and check > actual:
                return True
        return False

    def _clear_reconcile_backoff(self) -> None:
        self._reconcile_failed = None
        self._reconcile_failed_at = 0.0

    def _arm_reconcile_backoff(self, signature: tuple[Any, ...]) -> None:
        self._reconcile_failed = signature
        self._reconcile_failed_at = time.monotonic()

    def ensure(self, store: Any) -> bool:
        """Build the usage model and create its table; idempotent, fail-safe."""
        with self._ensure_lock:
            return self._ensure_locked(store)

    def _ensure_locked(self, store: Any) -> bool:
        try:
            engine = getattr(store, "_engine", None)
            session_factory = getattr(store, "_session_factory", None)
            if engine is None or session_factory is None:
                return False
            suffix = store.table_suffix()
        except Exception:
            logger.debug("usage store ensure failed", exc_info=True)
            return False
        signature = (id(engine), suffix)
        try:
            if self._signature == signature:
                if self._model is not None and self._model_signature == signature:
                    return True
                if (
                    self._reconcile_failed == signature
                    and time.monotonic() - self._reconcile_failed_at < _US_RECONCILE_RETRY_S
                ):
                    return False
            else:
                self._clear_reconcile_backoff()

            from sqlalchemy import Column, String
            from sqlalchemy.orm import declarative_base

            table_name = f"{_USAGE_TABLE_PREFIX}{suffix}"
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
            self._declared_lengths = _declared_lengths()
            self._effective_widths = dict(self._declared_lengths)
            if not store._create_table_with_race_guard(model.__table__, engine, table_name):
                self._publish(
                    _Published(
                        store=self._published.store,
                        model=None,
                        table_name=None,
                        signature=signature,
                        model_signature=self._published.model_signature,
                    )
                )
                self._arm_reconcile_backoff(signature)
                _level = warn_level(
                    _warned_usage_table_create, table_name, cooldown_s=_US_RECONCILE_RETRY_S
                )
                logger.log(
                    _level,
                    "usage table %s could not be created; usage rows are not being "
                    "written until the retry interval elapses",
                    table_name,
                )
                return False
            if not self._reconcile_schema(model.__table__, engine, table_name, schema_name, store):
                self._publish(
                    _Published(
                        store=None,
                        model=None,
                        table_name=None,
                        signature=signature,
                        model_signature=self._published.model_signature,
                    )
                )
                self._arm_reconcile_backoff(signature)
                return False
            self._publish(
                _Published(
                    store=store,
                    model=model,
                    table_name=table_name,
                    signature=signature,
                    model_signature=signature,
                )
            )
            self._reconcile_failed = None
            self._reconcile_failed_at = 0.0
            return True
        except Exception:
            self._publish(
                _Published(
                    store=None,
                    model=None,
                    table_name=None,
                    signature=signature,
                    model_signature=self._published.model_signature,
                )
            )
            self._arm_reconcile_backoff(signature)
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
            self._reconcile_widths(
                engine, table_name, schema_name, self._reflected_widths(engine, table_name, schema_name)
            )
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
        if not self._stopped:
            self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="openrouter-usage-writer",
            daemon=True,
        )
        self._thread.start()

    def _writer_loop(self) -> None:
        while self._writer_pass():
            pass

    def _writer_pass(self) -> bool:
        batch: list[dict[str, Any]] = []
        try:
            item = self._queue.get(timeout=_US_POLL_INTERVAL_S)
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
        failed = False
        if batch:
            failed = not self._write_batch(batch)
        if failed:
            time.sleep(_US_POLL_INTERVAL_S)
        if self._stop_event.is_set() and self._queue.qsize() == 0:
            self._release_held()
            return False
        return True

    def _write_batch(self, batch: list[dict[str, Any]]) -> bool:
        held, self._held = self._held, []
        combined = held + batch
        if len(combined) > _US_BATCH_MAX:
            self._dropped += len(combined) - _US_BATCH_MAX
            combined = combined[-_US_BATCH_MAX:]
        if self._write_now(combined):
            return True
        self._held = combined
        return False

    def _release_held(self) -> None:
        held, self._held = self._held, []
        if not held:
            return
        self._clear_reconcile_backoff()
        if self._write_now(held):
            return
        self._dropped += len(held)
        logger.warning(
            "usage writer stopped with %d rows it could not write; the usage table was not writable",
            len(held),
        )

    def _write_now(self, rows: list[dict[str, Any]]) -> bool:
        try:
            return self._persist_sync(rows)
        except Exception:
            logger.log(
                warn_level(self._warned, "persist", cooldown_s=_US_PERSIST_WARN_COOLDOWN_S),
                "usage batch persist failed; %d row(s) held for retry, not written (%d dropped so far)",
                len(rows), self._dropped, exc_info=True,
            )
            return False

    def _stored_collect_flag(self, store: Any) -> tuple[bool, bool]:
        session_factory = getattr(store, "_session_factory", None)
        if session_factory is None:
            return False, False
        try:
            from open_webui.models.functions import Function
            from open_webui.utils.valves import decrypt_valves
            from sqlalchemy import select

            with _db_session(session_factory) as session:
                raw = session.execute(
                    select(Function.valves).filter_by(id=getattr(store, "id", "") or "")
                ).scalar_one_or_none()
            stored = decrypt_valves(raw)
        except Exception:
            logger.debug("usage store could not read the persisted collect valve", exc_info=True)
            return False, False
        if stored == {} and not raw_valve_column_decodes(raw):
            return False, False
        if not isinstance(stored, dict):
            return False, False
        if not bool(stored.get("ENABLE_PLUGIN_SYSTEM", False)):
            return False, True
        return bool(stored.get("PIPE_DASHBOARD_USAGE_COLLECT", False)), True

    def _persist_sync(self, rows: list[dict[str, Any]]) -> bool:
        store = self._store
        if store is None:
            return False
        if not self.ensure(store):
            return False
        model = self._model
        if model is None:
            return False
        _collect_on, _read_ok = self._stored_collect_flag(store)
        if not _collect_on:
            if not _read_ok:
                logger.log(
                    warn_level(
                        self._warned,
                        "collect_valve_unreadable",
                        cooldown_s=_US_PERSIST_WARN_COOLDOWN_S,
                    ),
                    "usage store: the persisted PIPE_DASHBOARD_USAGE_COLLECT valve "
                    "could not be read (a rotated WEBUI_SECRET_KEY with valve encryption "
                    "on does this); no usage rows are being written",
                )
            return True
        session_factory = getattr(store, "_session_factory", None)
        if session_factory is None:
            return False
        instances = []
        for row in rows:
            data = {key: row.get(key) for key in USAGE_ROW_FIELDS}
            if is_temporary_chat(data["chat_id"]):
                data["chat_id"] = data["session_id"] = ""
            data["id"] = row.get("id") or generate_item_id()
            instances.append(model(**self._fit_row(data)))
        rejected: list[tuple[Any, Exception]] = []
        with _db_session(session_factory) as session:
            for instance in instances:
                try:
                    with session.begin_nested():
                        session.add(instance)
                        session.flush()
                except Exception as exc:  # noqa: BLE001 - one row's failure costs one row
                    rejected.append((instance, exc))
            session.commit()
        if rejected:
            self._persist_failed += len(rejected)
            for instance, exc in rejected:
                logger.warning(
                    "usage row rejected by the store and dropped; the rest of the batch "
                    "was written. id=%s chat_id=%r user_id=%r model_id=%r cost=%s: %s: %s",
                    getattr(instance, "id", None),
                    getattr(instance, "chat_id", None),
                    getattr(instance, "user_id", None),
                    getattr(instance, "model_id", None),
                    getattr(instance, "cost", None),
                    type(exc).__name__,
                    exc,
                )
        return True

    def start_purge_task(
        self, retention_days_fn: Callable[[], int | Awaitable[int]]
    ) -> None:
        """Start the jittered retention purge loop on the running loop."""
        if self._stopped:
            return
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
        days = await self._resolve_retention_days()
        cutoff = self._purge_cutoff(days)
        executor = getattr(store, "_db_executor", None)
        if executor is None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, self._purge_sync, cutoff)

    async def _resolve_retention_days(self) -> int:
        fn = self._retention_days_fn
        if fn is None:
            return 30
        try:
            resolved = fn()
            if inspect.isawaitable(resolved):
                resolved = await resolved
            return int(resolved)
        except Exception:
            logger.warning(
                "usage store: the retention-days provider is unreadable; falling back to "
                "%d days",
                30,
                exc_info=True,
            )
            return 30

    def _purge_cutoff(self, days: int | None = None) -> datetime.datetime:
        if days is None:
            days = 30
            fn = self._retention_days_fn
            if fn is not None:
                try:
                    value = fn()
                    if not isinstance(value, int):
                        raise TypeError(
                            f"the retention provider answered {type(value).__name__}, not an int"
                        )
                    days = value
                except Exception:
                    logger.warning(
                        "usage store: retention-days valve is unreadable; falling back to "
                        "%d days",
                        days,
                        exc_info=True,
                    )
        days = max(1, days)
        return usage_ts_from_epoch(time.time() - days * 86400.0)

    def _usage_table_fragment(self) -> str:
        table_name = self._table_name or ""
        suffix = table_name.removeprefix(_USAGE_TABLE_PREFIX)
        head, separator, key_hash = suffix.rpartition("_")
        if not separator or not key_hash:
            return ""
        return head

    def _published_table_present(self, store: Any) -> bool:
        from sqlalchemy import inspect as sa_inspect

        engine = getattr(store, "_engine", None)
        suffix_of = getattr(store, "table_suffix", None)
        if engine is None or not callable(suffix_of):
            return False
        try:
            signature = (id(engine), str(suffix_of()))
        except Exception:
            logger.debug("usage table suffix could not be derived", exc_info=True)
            return False
        if self._table_absent == signature:
            return False
        try:
            present = bool(sa_inspect(engine).has_table(f"{_USAGE_TABLE_PREFIX}{signature[1]}"))
        except Exception:
            logger.debug("usage table presence probe failed", exc_info=True)
            return False
        if not present:
            self._table_absent = signature
        return present

    def start_retention_purge(
        self, store: Any, retention_days_fn: Callable[[], int | Awaitable[int]]
    ) -> bool:
        task = self._purge_task
        if task is not None and not task.done():
            return True
        if self._model is None:
            if not self._published_table_present(store):
                return False
            if not self.ensure(store):
                return False
        self.start_purge_task(retention_days_fn)
        return True

    def _retired_usage_table_names(self, store: Any) -> list[str]:
        from sqlalchemy import inspect as sa_inspect

        current = self._table_name
        fragment = self._usage_table_fragment()
        engine = getattr(store, "_engine", None)
        if current is None or engine is None or not fragment:
            return []
        try:
            names = sa_inspect(engine).get_table_names()
        except Exception:
            logger.debug("usage table discovery failed", exc_info=True)
            return []
        prefix = f"{_USAGE_TABLE_PREFIX}{fragment}_"
        candidates = [
            name for name in sorted(names) if name != current and name.startswith(prefix)
        ]
        if not candidates:
            return []
        other_fragments = _other_installed_fragments(store)
        if other_fragments is None:
            for name in candidates:
                logger.log(
                    warn_level(
                        self._warned, f"usage_retired_unverified:{name}",
                        cooldown_s=_US_SHARED_FRAGMENT_COOLDOWN_S,
                    ),
                    "usage table %s was not swept: the installed function ids could not be "
                    "read, so the pipe cannot tell its own retired tables apart from "
                    "another installed copy's",
                    name,
                )
            return []
        if fragment in other_fragments:
            for name in candidates:
                logger.log(
                    warn_level(
                        self._warned, f"usage_retired_shared:{name}",
                        cooldown_s=_US_SHARED_FRAGMENT_COOLDOWN_S,
                    ),
                    "usage table %s is not swept: another installed function id sanitizes "
                    "to the same table fragment (%s), so the pipe cannot tell its own "
                    "retired tables apart from the ones that copy is still writing into",
                    name, fragment,
                )
            return []
        return candidates

    def _purge_table_sync(
        self,
        store: Any,
        name: str,
        cutoff: datetime.datetime,
        model: Any,
    ) -> None:
        from sqlalchemy import text

        session_factory = getattr(store, "_session_factory", None)
        if session_factory is None:
            return
        with _db_session(session_factory) as session:
            if model is not None:
                session.query(model).filter(model.ts < cutoff).delete(synchronize_session=False)
                for prefix in temporary_chat_prefixes():
                    session.query(model).filter(model.chat_id.startswith(prefix)).update(
                        {"chat_id": "", "session_id": ""}, synchronize_session=False
                    )
            else:
                quoted = ArtifactStore._quote_identifier(name)
                session.execute(
                    text(f"DELETE FROM {quoted} WHERE ts < :cutoff"), {"cutoff": cutoff}
                )
                for prefix in temporary_chat_prefixes():
                    session.execute(
                        text(
                            f"UPDATE {quoted} SET chat_id = '', session_id = '' "
                            "WHERE chat_id LIKE :pattern"
                        ),
                        {"pattern": f"{prefix}%"},
                    )
            session.commit()

    def _purge_sync(self, cutoff: datetime.datetime) -> None:
        published = self._published
        store = published.store
        model = published.model
        table_name = published.table_name
        if store is None or model is None or table_name is None:
            return
        session_factory = getattr(store, "_session_factory", None)
        if session_factory is None:
            return
        targets: list[tuple[str, Any]] = [(table_name, model)]
        targets.extend((name, None) for name in self._retired_usage_table_names(store))
        item_model = getattr(store, "_item_model", None)
        for name, target_model in targets:
            lock_id = _stable_crockford_id(f"{name}:purge")
            acquired = True
            if item_model is not None:
                acquired = self._acquire_purge_lock(store, item_model, lock_id)
                if not acquired:
                    continue
            try:
                self._purge_table_sync(store, name, cutoff, target_model)
            except Exception:
                logger.debug("usage table %s could not be purged", name, exc_info=True)
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
            logger.log(
                warn_level(_warned_purge_lock, f"purge_lock:{self._table_name}", cooldown_s=3600.0),
                "usage store: the retention purge could not take its cross-worker lock, so this pass "
                "was skipped and the lock was left alone (usage_table=%s).", self._table_name,
                exc_info=True,
            )
            return False

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
        info: dict[str, Any] = {
            "records": None,
            "approx_bytes": None,
            "persist_failed": self._persist_failed,
        }
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
            return {"records": None, "approx_bytes": None, "persist_failed": self._persist_failed}
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self._table_info_sync)

    def signal_stop(self) -> Any:
        """Signal the writer to drain-and-exit and cancel the purge task.

        Loop-safe and fast: sets the stop flag, wakes the writer, cancels the
        purge task. Does NOT join the writer thread — call ``join_writer`` off
        the event loop for that. Returns the cancelled purge task (awaitable)
        or ``None``.
        """
        self._stopped = True
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
                return
        self._thread = None

    def stop(self) -> Any:
        """Synchronous stop (signal + in-line join). Prefer signal_stop +
        an off-loop join_writer from an async caller to avoid stalling the loop.
        """
        task = self.signal_stop()
        self.join_writer()
        return task
