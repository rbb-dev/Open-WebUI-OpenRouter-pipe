"""Session log management with worker threads and archive handling.

This module provides the SessionLogManager class which handles:
- Background worker threads for log archival (writer, cleanup, assembler)
- Session log queue management
- DB-backed segment persistence and assembly
- Archive file management and retention cleanup

The manager coordinates between in-memory session logs (from SessionLogger)
and persistent storage (encrypted zip archives via pyzipper).
"""

from __future__ import annotations

import contextlib
import datetime
import json
import logging
import os
import queue
import random
import threading
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..core.timing_logger import timed

# Optional pyzipper support for session log encryption
try:
    import pyzipper  # pyright: ignore[reportMissingImports]
except ImportError:
    pyzipper = None  # type: ignore

if TYPE_CHECKING:
    from ..pipe import Pipe
    from ..storage.persistence import ArtifactStore


class SessionLogManager:
    """Manages session log background workers and archive operations.

    This class owns:
    - Worker threads: writer, cleanup, and assembler
    - Queue for archive jobs
    - Lock for thread-safe state access
    - Archive settings and directories

    It coordinates with:
    - ArtifactStore for DB persistence
    - SessionLogger/write_session_log_archive for actual archive writing
    - Pipe instance for live valve configuration access
    """

    def __init__(
        self,
        logger: logging.Logger,
        pipe: "Pipe",
        artifact_store: "ArtifactStore | None" = None,
    ) -> None:
        """Initialize the session log manager.

        Args:
            logger: Logger instance for debug/warning messages
            pipe: Pipe instance for accessing live valves configuration
            artifact_store: Optional ArtifactStore for DB operations
        """
        from ..core.logging_system import _SessionLogArchiveJob

        self.logger = logger
        self._pipe = pipe
        self._artifact_store: "ArtifactStore | None" = artifact_store

        # Worker thread state
        self._queue: queue.Queue[_SessionLogArchiveJob] | None = None
        self._stop_event: threading.Event | None = None
        self._worker_thread: threading.Thread | None = None
        self._cleanup_thread: threading.Thread | None = None
        self._assembler_thread: threading.Thread | None = None

        # Thread-safe configuration access
        self._lock = threading.Lock()
        self._cleanup_interval_seconds = getattr(self.valves, "SESSION_LOG_CLEANUP_INTERVAL_SECONDS", 3600)
        self._retention_days = getattr(self.valves, "SESSION_LOG_RETENTION_DAYS", 30)
        self._dirs: set[str] = set()
        self._warning_emitted = False

    def set_artifact_store(self, artifact_store: "ArtifactStore") -> None:
        """Set the artifact store reference."""
        self._artifact_store = artifact_store

    @property
    def valves(self) -> Any:
        """Access live valves from the Pipe instance.

        This ensures we always read current valve values, not a stale snapshot
        captured at initialization time. Open WebUI replaces the valves object
        when settings are saved, so we must access via pipe reference.
        """
        return self._pipe.valves

    @property
    def queue(self) -> "queue.Queue | None":
        """Access the archive job queue (for tests)."""
        return self._queue

    @property
    def stop_event(self) -> threading.Event | None:
        """Access the stop event (for tests)."""
        return self._stop_event

    @property
    def worker_thread(self) -> threading.Thread | None:
        """Access the writer thread (for tests)."""
        return self._worker_thread

    @property
    def cleanup_thread(self) -> threading.Thread | None:
        """Access the cleanup thread (for tests)."""
        return self._cleanup_thread

    @property
    def assembler_thread(self) -> threading.Thread | None:
        """Access the assembler thread (for tests)."""
        return self._assembler_thread

    @property
    def dirs(self) -> set[str]:
        """Access the tracked log directories."""
        return self._dirs

    @property
    def retention_days(self) -> int:
        """Access the retention days setting."""
        return self._retention_days

    @property
    def warning_emitted(self) -> bool:
        """Access the warning emitted flag."""
        return self._warning_emitted

    @warning_emitted.setter
    def warning_emitted(self, value: bool) -> None:
        """Set the warning emitted flag."""
        self._warning_emitted = value

    # =========================================================================
    # Worker Thread Management
    # =========================================================================

    @timed
    def stop_workers(self) -> None:
        """Stop session log background threads (best effort)."""
        if self._stop_event:
            with contextlib.suppress(Exception):
                self._stop_event.set()
        if self._queue:
            with contextlib.suppress(Exception):
                self._queue.put_nowait(None)  # type: ignore[arg-type]
        for thread in (self._worker_thread, self._cleanup_thread, self._assembler_thread):
            if thread and thread.is_alive():
                with contextlib.suppress(Exception):
                    thread.join(timeout=2.0)
        self._worker_thread = None
        self._cleanup_thread = None
        self._assembler_thread = None

    @timed
    def start_workers(self) -> None:
        """Start session log writer + cleanup threads if not already running."""
        from ..core.logging_system import _SessionLogArchiveJob

        if self._worker_thread and self._worker_thread.is_alive():
            return
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        if self._queue is None:
            self._queue = queue.Queue(maxsize=500)
        if self._stop_event is None:
            self._stop_event = threading.Event()

        def _writer_loop() -> None:
            while True:
                if self._stop_event and self._stop_event.is_set():
                    break
                item: _SessionLogArchiveJob | None = None
                try:
                    item = self._queue.get(timeout=0.5) if self._queue else None
                except queue.Empty:
                    continue
                except Exception:
                    continue
                if item is None:
                    with contextlib.suppress(Exception):
                        if self._queue:
                            self._queue.task_done()
                    continue
                try:
                    self._write_archive(item)
                except Exception:
                    self.logger.debug("Session log writer failed", exc_info=True)
                finally:
                    with contextlib.suppress(Exception):
                        if self._queue:
                            self._queue.task_done()

        def _cleanup_loop() -> None:
            while True:
                if self._stop_event and self._stop_event.is_set():
                    break
                try:
                    self.cleanup_archives()
                except Exception:
                    self.logger.debug("Session log cleanup failed", exc_info=True)
                interval = 3600
                with contextlib.suppress(Exception):
                    with self._lock:
                        interval = self._cleanup_interval_seconds
                try:
                    time.sleep(interval)
                except Exception:
                    time.sleep(60)

        self._worker_thread = threading.Thread(
            target=_writer_loop,
            name="openrouter-session-log-writer",
            daemon=True,
        )
        self._worker_thread.start()

        self._cleanup_thread = threading.Thread(
            target=_cleanup_loop,
            name="openrouter-session-log-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

    @timed
    def start_assembler_worker(self) -> None:
        """Start the DB-backed session log assembler thread (multi-worker safe)."""
        if self._assembler_thread and self._assembler_thread.is_alive():
            return
        if self._stop_event is None:
            self._stop_event = threading.Event()

        def _wait(stop_event: threading.Event, seconds: float) -> bool:
            try:
                return stop_event.wait(timeout=max(0.0, float(seconds)))
            except Exception:
                time.sleep(max(0.0, float(seconds)))
                return stop_event.is_set()

        def _assembler_loop() -> None:
            stop_event = self._stop_event
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
                    self.run_assembler_once()
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

        self._assembler_thread = threading.Thread(
            target=_assembler_loop,
            name="openrouter-session-log-assembler",
            daemon=True,
        )
        self._assembler_thread.start()

    # =========================================================================
    # Archive Writing
    # =========================================================================

    def _write_archive(self, job: Any) -> None:
        from ..core.logging_system import write_session_log_archive
        return write_session_log_archive(job)

    # =========================================================================
    # Archive Settings Resolution
    # =========================================================================

    @timed
    def resolve_archive_settings(
        self,
        valves: Any,
    ) -> tuple[str, bytes, str, int | None] | None:
        """Resolve the session log archive settings required for eventual zip writing."""
        from ..core.config import EncryptedStr

        if not valves.SESSION_LOG_STORE_ENABLED:
            return None
        if pyzipper is None:
            if not self._warning_emitted:
                self.logger.warning(
                    "Session log storage is enabled but the 'pyzipper' package is not available; skipping persistence."
                )
                self._warning_emitted = True
            return None

        base_dir = valves.SESSION_LOG_DIR.strip()
        if not base_dir:
            if not self._warning_emitted:
                self.logger.warning(
                    "Session log storage is enabled but SESSION_LOG_DIR is empty; skipping persistence."
                )
                self._warning_emitted = True
            return None

        decrypted = EncryptedStr.decrypt(valves.SESSION_LOG_ZIP_PASSWORD)
        password = (decrypted or "").strip()
        if not password:
            if not self._warning_emitted:
                self.logger.warning(
                    "Session log storage is enabled but SESSION_LOG_ZIP_PASSWORD is not configured; skipping persistence."
                )
                self._warning_emitted = True
            return None

        zip_compression = valves.SESSION_LOG_ZIP_COMPRESSION
        zip_compresslevel = valves.SESSION_LOG_ZIP_COMPRESSLEVEL
        if zip_compression in {"stored", "lzma"}:
            zip_compresslevel = None

        with contextlib.suppress(Exception):
            with self._lock:
                self._cleanup_interval_seconds = valves.SESSION_LOG_CLEANUP_INTERVAL_SECONDS
                self._retention_days = valves.SESSION_LOG_RETENTION_DAYS
                self._dirs.add(base_dir)

        return base_dir, password.encode("utf-8"), zip_compression, zip_compresslevel

    # =========================================================================
    # Enqueue Archive Job
    # =========================================================================

    @timed
    def enqueue_archive(
        self,
        valves: Any,
        *,
        user_id: str,
        session_id: str,
        chat_id: str,
        message_id: str,
        request_id: str,
        log_events: list[dict[str, Any]],
    ) -> None:
        """Queue the current request's session logs for encrypted zip persistence."""
        from ..core.config import EncryptedStr
        from ..core.logging_system import _SessionLogArchiveJob

        if not valves.SESSION_LOG_STORE_ENABLED:
            return
        if not (user_id and session_id and chat_id and message_id):
            return
        if not log_events:
            return
        if pyzipper is None:
            if not self._warning_emitted:
                self.logger.warning("Session log storage is enabled but the 'pyzipper' package is not available; skipping persistence.")
                self._warning_emitted = True
            return

        base_dir = valves.SESSION_LOG_DIR.strip()
        if not base_dir:
            if not self._warning_emitted:
                self.logger.warning("Session log storage is enabled but SESSION_LOG_DIR is empty; skipping persistence.")
                self._warning_emitted = True
            return

        decrypted = EncryptedStr.decrypt(valves.SESSION_LOG_ZIP_PASSWORD)
        password = (decrypted or "").strip()
        if not password:
            if not self._warning_emitted:
                self.logger.warning("Session log storage is enabled but SESSION_LOG_ZIP_PASSWORD is not configured; skipping persistence.")
                self._warning_emitted = True
            return

        zip_compression = valves.SESSION_LOG_ZIP_COMPRESSION
        zip_compresslevel = valves.SESSION_LOG_ZIP_COMPRESSLEVEL
        if zip_compression in {"stored", "lzma"}:
            zip_compresslevel = None

        with contextlib.suppress(Exception):
            with self._lock:
                self._cleanup_interval_seconds = valves.SESSION_LOG_CLEANUP_INTERVAL_SECONDS
                self._retention_days = valves.SESSION_LOG_RETENTION_DAYS
                self._dirs.add(base_dir)

        if self._queue is None:
            self._queue = queue.Queue(maxsize=500)
        if self._queue.full():
            self.logger.warning("Session log archive queue is full; dropping archive for chat_id=%s message_id=%s", chat_id, message_id)
            return

        self.start_workers()
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
            self._queue.put_nowait(job)
        except Exception:
            self.logger.debug("Failed to enqueue session log archive job", exc_info=True)

    # =========================================================================
    # DB Segment Persistence
    # =========================================================================

    @timed
    async def persist_segment_to_db(
        self,
        valves: Any,
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
        from ..storage.persistence import generate_item_id
        from ..core.logging_system import _SessionLogArchiveJob, write_session_log_archive

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
        archive_settings = self.resolve_archive_settings(valves)
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
        if self._artifact_store is not None:
            with contextlib.suppress(Exception):
                self._artifact_store._ensure_artifact_store(valves, pipe_identifier)

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
            self.start_workers()
            self.start_assembler_worker()
            persisted = await self._artifact_store._db_persist([row]) if self._artifact_store else []
            if not persisted:
                # Survivability guarantee: if DB staging isn't available (or breaker blocks writes),
                # fall back to direct zip persistence so operators still get logs.
                self.logger.warning(
                    "Session log DB staging returned no ids; falling back to direct zip write (chat_id=%s message_id=%s request_id=%s).",
                    chat_id,
                    message_id,
                    request_id,
                )
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
        except Exception:
            self.logger.debug(
                "Failed to persist session log segment (chat_id=%s message_id=%s request_id=%s terminal=%s)",
                chat_id,
                message_id,
                request_id,
                terminal,
                exc_info=True,
            )

    # =========================================================================
    # DB Handles
    # =========================================================================

    def _db_handles(self) -> tuple[Any | None, Any | None]:
        """Return (model, session_factory) for direct DB queries (best effort)."""
        if self._artifact_store is None:
            return None, None
        model = getattr(self._artifact_store, "_item_model", None)
        session_factory = getattr(self._artifact_store, "_session_factory", None)
        return model, session_factory

    # =========================================================================
    # Assembler Logic
    # =========================================================================

    @timed
    def run_assembler_once(self) -> None:
        """One assembler tick: cleanup stale locks, assemble terminal + stale bundles."""
        if not getattr(self.valves, "SESSION_LOG_STORE_ENABLED", False):
            return
        model, session_factory = self._db_handles()
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

        self._cleanup_stale_locks(model, session_factory, lock_stale_seconds)

        terminals = self._list_terminal_messages(model, session_factory, limit=batch_size)
        for chat_id, message_id in terminals:
            self._assemble_and_write_bundle(chat_id, message_id, terminal=True)

        stale = self._list_stale_messages(
            model,
            session_factory,
            stale_finalize_seconds=stale_finalize_seconds,
            limit=batch_size,
        )
        for chat_id, message_id in stale:
            self._assemble_and_write_bundle(chat_id, message_id, terminal=False, stale_finalize_seconds=stale_finalize_seconds)

    @timed
    def _cleanup_stale_locks(
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
        if ids and self._artifact_store:
            with contextlib.suppress(Exception):
                self._artifact_store._delete_artifacts_sync(ids)

    @timed
    def _list_terminal_messages(
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
    def _list_stale_messages(
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

    # =========================================================================
    # Archive Event Helpers
    # =========================================================================

    def read_archive_events(
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

    def dedupe_events(
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

    # =========================================================================
    # Bundle Assembly
    # =========================================================================

    @timed
    def _assemble_and_write_bundle(
        self,
        chat_id: str,
        message_id: str,
        *,
        terminal: bool,
        stale_finalize_seconds: float = 0.0,
        archive_settings: tuple[str, bytes, str, int | None] | None = None,
    ) -> bool:
        """Assemble all segments for one message into a single zip, then delete DB rows."""
        from ..core.utils import _stable_crockford_id, _sanitize_path_component
        from ..core.logging_system import _SessionLogArchiveJob

        if not (chat_id and message_id):
            return False
        model, session_factory = self._db_handles()
        if not model or not session_factory:
            return False
        if self._artifact_store is None:
            return False

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
        if not self._artifact_store._try_acquire_lock_sync(lock_row):
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
                self._artifact_store._delete_artifacts_sync([lock_id])
            return False

        payloads = self._artifact_store._db_fetch_sync(chat_id, message_id, ids)
        segments: list[dict[str, Any]] = []
        for item_id in ids:
            payload = payloads.get(item_id)
            if not isinstance(payload, dict):
                continue
            if payload.get("type") in {"session_log_segment", "session_log_segment_terminal"}:
                segments.append(payload)

        if not segments:
            with contextlib.suppress(Exception):
                self._artifact_store._delete_artifacts_sync(ids + [lock_id])
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
                    "func": "_assemble_and_write_bundle",
                    "lineno": 0,
                    "message": msg,
                }
            )

        settings = archive_settings or self.resolve_archive_settings(self.valves)
        if settings is None:
            with contextlib.suppress(Exception):
                self._artifact_store._delete_artifacts_sync([lock_id])
            return False
        base_dir, zip_password, zip_compression, zip_compresslevel = settings

        out_dir = Path(base_dir).expanduser() / _sanitize_path_component(resolved_user_id, fallback="user") / _sanitize_path_component(chat_id, fallback="chat")
        out_path = out_dir / f"{_sanitize_path_component(message_id, fallback='message')}.zip"
        before_stat = None
        with contextlib.suppress(Exception):
            before_stat = out_path.stat()

        # Merge with existing archive events if the zip already exists.
        if out_path.exists():
            try:
                existing_events = self.read_archive_events(out_path, settings)
                if existing_events:
                    merged_events = existing_events + merged_events
                    merged_events = self.dedupe_events(merged_events)
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
        self._write_archive(job)

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
                self._artifact_store._delete_artifacts_sync(ids + [lock_id])
            return True

        # If writing failed, keep segments for retry and allow lock reaping.
        with contextlib.suppress(Exception):
            self._artifact_store._delete_artifacts_sync([lock_id])
        return False

    # =========================================================================
    # Archive Cleanup
    # =========================================================================

    @timed
    def cleanup_archives(self) -> None:
        """Delete expired session log archives and prune empty directories."""
        with self._lock:
            dirs = set(self._dirs)
            retention_days = self._retention_days
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
                    with contextlib.suppress(Exception):
                        if any(Path(dirpath).iterdir()):
                            continue
                        os.rmdir(dirpath)
            except Exception:
                continue
