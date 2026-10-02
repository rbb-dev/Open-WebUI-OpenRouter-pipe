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
import weakref
from collections.abc import Callable, Collection, MutableMapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import case, func, not_

from ..core.logging_system import _prune_archive_dir
from ..core.timing_logger import timed
from ..core.warn_latch import warn_level
from ..integrations.video_intent_prompts import INTENT_SCHEMA_NAME
from ..storage.persistence import _db_session

logger = logging.getLogger(__name__)

_THREAD_WAKE_SLICE_S = 30.0


def _wait_alive(mgr_ref: Any, stop_event: threading.Event, seconds: float) -> bool:
    remaining = max(0.0, float(seconds))
    while remaining > 0.0:
        if mgr_ref() is None or stop_event.is_set():
            return True
        if stop_event.wait(timeout=min(_THREAD_WAKE_SLICE_S, remaining)):
            return True
        remaining -= _THREAD_WAKE_SLICE_S
    return mgr_ref() is None or stop_event.is_set()


def _truncate_latch(latch: MutableMapping[str, float] | set[str], keep: int) -> None:
    excess = len(latch) - int(keep)
    if excess <= 0:
        return
    if isinstance(latch, set):
        for stale_key in list(latch)[:excess]:
            latch.discard(stale_key)
        return
    for stale_key in sorted(latch, key=lambda k: latch[k])[:excess]:
        latch.pop(stale_key, None)


def _unwritten_archive_count(job_queue: queue.Queue) -> int:
    with job_queue.mutex:
        return sum(1 for entry in job_queue.queue if entry is not None)


def _report_drain_incomplete(
    mgr_ref: Any, job_queue: queue.Queue, dropped: int = 0
) -> None:
    residual = _unwritten_archive_count(job_queue) + int(dropped)
    if residual <= 0:
        return
    mgr = mgr_ref()
    if mgr is not None:
        _truncate_latch(_warned_drain_incomplete, _MAX_DRAIN_LATCH_KEYS)
        mgr.logger.log(
            warn_level(
                _warned_drain_incomplete,
                f"session_log_shutdown_drain_incomplete:{time.monotonic_ns()}",
                cooldown_s=3600.0,
            ),
            "Session log writer stopped with %d queued archive(s) it could not write.",
            residual,
        )
        return
    _truncate_latch(_DEAD_MANAGER_DRAIN_WARNINGS, _MAX_DRAIN_LATCH_KEYS)
    logger.log(
        warn_level(
            _DEAD_MANAGER_DRAIN_WARNINGS,
            f"session_log_shutdown_drain_incomplete:{time.monotonic_ns()}",
            cooldown_s=3600.0,
        ),
        "Session log writer stopped with %d queued archive(s) it could not write.",
        residual,
    )


def _writer_loop(mgr_ref: Any, stop_event: threading.Event, job_queue: queue.Queue) -> None:
    drain_deadline: float | None = None
    dropped = 0

    def _should_stop(deadline: float | None) -> bool:
        return deadline is not None or stop_event.is_set()

    while True:
        manager_gone = mgr_ref() is None
        item: Any = None
        if manager_gone or stop_event.is_set():
            if drain_deadline is None:
                drain_deadline = time.monotonic() + _WRITER_DRAIN_SECONDS
            if drain_deadline is not None and time.monotonic() >= drain_deadline and not job_queue.empty():
                _report_drain_incomplete(mgr_ref, job_queue, dropped)
                break
            try:
                item = job_queue.get_nowait()
            except queue.Empty:
                _report_drain_incomplete(mgr_ref, job_queue, dropped)
                break
        else:
            try:
                item = job_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception:
                logger.debug("Session log writer queue.get failed", exc_info=True)
                continue
        mgr = mgr_ref()
        if mgr is None:
            if item is not None:
                dropped += 1
            with contextlib.suppress(Exception):
                job_queue.task_done()
            if item is None:
                _report_drain_incomplete(mgr_ref, job_queue, dropped)
                break
            continue
        if item is None:
            with contextlib.suppress(Exception):
                job_queue.task_done()
            if drain_deadline is None and _should_stop(None):
                break
            continue
        try:
            mgr._write_archive(item)
        except Exception:
            logger.debug("Session log writer failed", exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                job_queue.task_done()
        mgr = None


def _cleanup_loop(mgr_ref: Any, stop_event: threading.Event) -> None:
    while True:
        mgr = mgr_ref()
        if mgr is None:
            break
        try:
            mgr.cleanup_archives()
        except Exception:
            logger.debug("Session log cleanup failed", exc_info=True)
        interval = 3600
        with contextlib.suppress(Exception):
            interval = mgr.valves.SESSION_LOG_CLEANUP_INTERVAL_SECONDS
        mgr = None
        if _wait_alive(mgr_ref, stop_event, interval):
            break


def _assembler_loop(mgr_ref: Any, stop_event: threading.Event) -> None:
    mgr = mgr_ref()
    if mgr is None:
        return

    try:
        jitter = mgr.valves.SESSION_LOG_ASSEMBLER_JITTER_SECONDS
    except Exception:
        logger.debug("Session log assembler exiting: jitter valve unavailable", exc_info=True)
        return
    mgr = None
    if jitter and _wait_alive(mgr_ref, stop_event, random.uniform(0.0, jitter)):
        return

    while True:
        mgr = mgr_ref()
        if mgr is None or stop_event.is_set():
            break
        try:
            mgr.run_assembler_once()
        except Exception:
            logger.debug("Session log assembler failed", exc_info=True)
        if stop_event.is_set():
            break
        interval = 30
        extra = 0.0
        mgr = None
        try:
            valves = mgr_ref()
            if valves is None:
                break
            interval = valves.valves.SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS
            extra = valves.valves.SESSION_LOG_ASSEMBLER_JITTER_SECONDS
        except Exception:
            logger.debug("Session log assembler exiting: interval valves unavailable", exc_info=True)
            break
        valves = None
        delay = interval + (random.uniform(0.0, extra) if extra else 0.0)
        if _wait_alive(mgr_ref, stop_event, delay):
            break

_UNREADABLE_ARCHIVE_CAPTURE_AFTER = 3

_WRITER_DRAIN_SECONDS = 1.0

_MAX_EXCLUDED_TURNS = 1000

_EXCLUDED_KEY_SEP = "\x1f"

_MAX_DRAIN_LATCH_KEYS = 32

_DEAD_MANAGER_DRAIN_WARNINGS: dict[str, float] = {}
_warned_drain_incomplete: dict[str, float] = {}

_INCOMPLETE_MARKER_PREFIX = "Session log finalized as incomplete"
_INCOMPLETE_MARKER_FUNC = "_assemble_and_write_bundle"

_TEMPORARY_CHAT_PROCESS_SCOPE = "process"


def _excluded_key(model: Any) -> Any:
    return model.chat_id + _EXCLUDED_KEY_SEP + model.message_id


def _set_aside_predicate(model: Any, exclude: Collection[tuple[str, str]]) -> Any:
    return not_(
        _excluded_key(model).in_(
            [f"{chat_id}{_EXCLUDED_KEY_SEP}{message_id}" for chat_id, message_id in exclude]
        )
    )


def _window_rows(
    rows: Sequence[Any],
    exclude: Collection[tuple[str, str]],
    refusable: Callable[[str], bool],
    limit: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    offered: list[tuple[str, str]] = []
    held: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chat_id, message_id in rows:
        if not (isinstance(chat_id, str) and isinstance(message_id, str)):
            continue
        key = (chat_id, message_id)
        if key in seen or key in exclude:
            continue
        seen.add(key)
        if refusable(chat_id):
            if len(held) < 1:
                held.append(key)
            continue
        offered.append(key)
        if len(offered) >= int(limit):
            break
    return offered, held


class _LockContended:

    __slots__ = ()


_LOCK_CONTENDED = _LockContended()


def _is_incomplete_marker(evt: Any) -> bool:
    if not isinstance(evt, dict):
        return False
    if evt.get("func") != _INCOMPLETE_MARKER_FUNC:
        return False
    if not str(evt.get("message", "")).startswith(_INCOMPLETE_MARKER_PREFIX):
        return False
    return evt.get("lineno") == 0 and evt.get("level") == "WARNING"


def _incomplete_marker(
    request_id: str,
    session_id: str,
    user_id: str,
    stale_finalize_seconds: float,
) -> dict[str, Any]:
    message = _INCOMPLETE_MARKER_PREFIX
    if stale_finalize_seconds:
        message = f"{message} (no terminal segment after {int(stale_finalize_seconds)}s)"
    return {
        "created": time.time(),
        "level": "WARNING",
        "logger": __name__,
        "request_id": request_id or "",
        "session_id": session_id or "",
        "user_id": user_id or "",
        "event_type": "pipe",
        "module": __name__,
        "func": _INCOMPLETE_MARKER_FUNC,
        "lineno": 0,
        "message": message,
    }


def _inherited_message_id(metadata: dict[str, Any]) -> str:
    mid = metadata.get("message_id")
    if mid:
        return str(mid)
    if not metadata.get("task"):
        return ""
    task_body = metadata.get("task_body") or {}
    if isinstance(task_body, dict):
        messages = task_body.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                candidate = last.get("id")
                if candidate:
                    return str(candidate)
    user_message = metadata.get("user_message") or {}
    if isinstance(user_message, dict):
        children = user_message.get("childrenIds")
        if isinstance(children, list) and children:
            candidate = children[0]
            if candidate:
                return str(candidate)
    return ""


_TASK_QUALIFIERS = frozenset(
    {
        "title_generation",
        "follow_up_generation",
        "tags_generation",
        "emoji_generation",
        "query_generation",
        "image_prompt_generation",
        "autocomplete_generation",
        "function_calling",
        "moa_response_generation",
        "context_compaction",
        "memory_review",
        "context_summary",
        INTENT_SCHEMA_NAME,
    }
)


def _known_task_names() -> frozenset[str]:
    return _TASK_QUALIFIERS


def _split_archive_key(message_id: str) -> tuple[str, str]:
    head, dot, tail = message_id.rpartition(".")
    if dot and tail in _TASK_QUALIFIERS:
        return head, tail
    return message_id, ""


def _preferred_request_id(segments: list[dict[str, Any]]) -> str:
    for seg in segments:
        if seg.get("type") == "session_log_segment_terminal":
            rid = seg.get("request_id")
            if isinstance(rid, str) and rid.strip():
                return rid.strip()
    for seg in segments:
        rid = seg.get("request_id")
        if isinstance(rid, str) and rid.strip():
            return rid.strip()
    return ""


_MAX_KEY_CHARS = 64
_MIN_HEAD_CHARS = 16
_HEAD_DIGEST_CHARS = 10


def _bounded_head(resolved: str, budget: int) -> str:
    if len(resolved) <= budget:
        return resolved
    from ..core.utils import _stable_crockford_id
    keep = max(0, budget - _HEAD_DIGEST_CHARS - 1)
    return f"{resolved[:keep]}-{_stable_crockford_id(resolved, length=_HEAD_DIGEST_CHARS)}"


def resolve_message_id(metadata: Any) -> str:
    if not isinstance(metadata, dict):
        return ""
    try:
        resolved = _inherited_message_id(metadata)
        task = metadata.get("task")
        if not task:
            return resolved
        if not resolved:
            return ""
        qualifier = str(task)
        if len(qualifier) > _MAX_KEY_CHARS - _MIN_HEAD_CHARS - 1:
            qualifier = qualifier[: _MAX_KEY_CHARS - _MIN_HEAD_CHARS - 1]
        return f"{_bounded_head(resolved, _MAX_KEY_CHARS - len(qualifier) - 1)}.{qualifier}"
    except Exception:
        logger.debug("resolve_message_id failed to parse metadata", exc_info=True)
    return ""


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

    _FAULT_LATCHES = (
        "_unreadable_archive_warnings",
        "_stale_filter_warnings",
        "_read_fault_warnings",
        "_captured_turns",
        "_captured_row_ids",
        "_capture_exempt",
        "_rescue_pending",
    )

    def __init__(
        self,
        logger: logging.Logger,
        pipe: Pipe,
        artifact_store: ArtifactStore | None = None,
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
        self._artifact_store: ArtifactStore | None = artifact_store

        # Worker thread state
        self._queue: queue.Queue[_SessionLogArchiveJob] | None = None
        self._stop_event: threading.Event | None = None
        self._worker_thread: threading.Thread | None = None
        self._cleanup_thread: threading.Thread | None = None
        self._assembler_thread: threading.Thread | None = None

        # Thread-safe configuration access
        self._lock = threading.Lock()
        self._dirs: set[str] = set()
        self._assembler_recent_failures: dict[tuple[str, str], float] = {}
        self._rescue_pending: dict[tuple[str, str], float] = {}
        self._assembly_failure_stale_arm: set[tuple[str, str]] = set()
        self._warned: set[str] = set()
        self._unreadable_archive_warnings: dict[str, float] = {}
        self._unreadable_archive_attempts: dict[str, int] = {}
        self._stale_filter_warnings: dict[str, float] = {}
        self._read_fault_warnings: dict[str, float] = {}
        self._captured_turns: set[str] = set()
        self._captured_row_ids: dict[str, str] = {}
        self._capture_exempt: set[tuple[str, str]] = set()
        self._skip_info_emitted: set[str] = set()
        self._warned_temporary_chat: dict[str, float] = {}
        self._archive_queue_full_warnings: dict[str, float] = {}
        self._archive_queue_drops: int = 0
        self._temporary_chat_sweep_at: float = 0.0
        self._ownership_skips: dict[str, float] = {}

    @property
    def _assembly_failures(self) -> dict[tuple[str, str], float]:
        return self._assembler_recent_failures

    def set_artifact_store(self, artifact_store: ArtifactStore) -> None:
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
    def queue(self) -> queue.Queue | None:
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
        return int(self.valves.SESSION_LOG_RETENTION_DAYS)

    @property
    def warning_emitted(self) -> bool:
        """Access the warning emitted flag."""
        return self._warning_emitted

    @warning_emitted.setter
    def warning_emitted(self, value: bool) -> None:
        if not value:
            self._warned.clear()

    def _warn_once(self, cause: str, message: str) -> None:
        with self._lock:
            self.logger.log(warn_level(self._warned, cause), message)

    def _warn_temporary_chat_skip(
        self, site: str, scope: str, message: str, *args: object
    ) -> None:
        cooldown_s = 300.0
        now = time.monotonic()
        with self._lock:
            if now - self._temporary_chat_sweep_at >= cooldown_s:
                self._temporary_chat_sweep_at = now
                for key in [
                    k
                    for k, armed_at in self._warned_temporary_chat.items()
                    if now - armed_at >= cooldown_s
                ]:
                    self._warned_temporary_chat.pop(key, None)
            self.logger.log(
                warn_level(
                    self._warned_temporary_chat,
                    f"temporary_chat:{site}:{scope}",
                    cooldown_s=cooldown_s,
                ),
                message,
                *args,
            )

    async def _caller_owns_chat(self, chat_id: str, user_id: str) -> bool:
        try:
            from open_webui.models.chats import Chats

            return bool(await Chats.is_chat_owner(chat_id, user_id))
        except Exception:
            self.logger.log(
                warn_level(
                    self._ownership_skips,
                    f"session_log_ownership_unreadable:{chat_id}:{user_id}",
                    cooldown_s=3600.0,
                ),
                "Session log segment not staged: Open WebUI's chat ownership check is "
                "unavailable, so the archive path cannot be trusted (user_id=%s "
                "chat_id=%s). Refusing rather than writing under an unverified id.",
                user_id,
                chat_id,
                exc_info=True,
            )
            return False

    async def _caller_is_admin(self, user_id: str) -> bool:
        try:
            from open_webui.models.users import Users

            user = await Users.get_user_by_id(user_id)
            return getattr(user, "role", None) == "admin"
        except Exception:
            self.logger.debug(
                "Session log admin bypass lookup failed for user_id=%s", user_id, exc_info=True,
            )
            return False

    def _latch_ownership_skip(
        self, user_id: str, chat_id: str, message_id: str, request_id: str
    ) -> None:
        self.logger.log(
            warn_level(
                self._ownership_skips,
                f"session_log_not_owned:{chat_id}:{user_id}",
                cooldown_s=3600.0,
            ),
            "Session log segment not staged (chat is not the caller's): user_id=%s "
            "chat_id=%s message_id=%s request_id=%s",
            user_id,
            chat_id,
            message_id,
            request_id,
        )

    def _skip_task(self, task: str, chat_id: str, request_id: str) -> None:
        _truncate_latch(self._skip_info_emitted, _MAX_DRAIN_LATCH_KEYS - 1)
        self.logger.log(
            warn_level(self._skip_info_emitted, f"task:{task}"),
            "Session log segment skipped (task resolves to no message id): "
            "task=%s chat_id=%s request_id=%s",
            task,
            chat_id or "(none)",
            request_id,
        )

    def _warn_archive_queue_full(self, job: Any) -> None:
        with self._lock:
            self._archive_queue_drops += 1
            dropped = self._archive_queue_drops
            level = warn_level(
                self._archive_queue_full_warnings,
                "session_log_archive_queue_full",
                cooldown_s=300.0,
            )
        self.logger.log(
            level,
            "Session log archive queue is full; dropping archive for chat_id=%s "
            "message_id=%s (%d dropped since this worker started).",
            job.chat_id,
            job.message_id,
            dropped,
        )

    @property
    def _warning_emitted(self) -> bool:
        return bool(self._warned)

    @_warning_emitted.setter
    def _warning_emitted(self, value: bool) -> None:
        if not value:
            if getattr(self, "_warned", None) is None:
                self._warned = set()
            else:
                self._warned.clear()

    # =========================================================================
    # Worker Thread Management
    # =========================================================================

    def stop_workers(self) -> None:
        """Stop session log background threads (best effort)."""
        signalled = self._stop_event
        if signalled:
            with contextlib.suppress(Exception):
                signalled.set()
        if self._queue:
            with contextlib.suppress(Exception):
                self._queue.put_nowait(None)  # type: ignore[arg-type]
        threads = (self._worker_thread, self._cleanup_thread, self._assembler_thread)
        for thread in threads:
            if thread and thread.is_alive():
                with contextlib.suppress(Exception):
                    thread.join(timeout=2.0)
        for _name in ("_worker_thread", "_cleanup_thread", "_assembler_thread"):
            _thread = getattr(self, _name)
            if _thread is None or not _thread.is_alive():
                setattr(self, _name, None)
        if self._stop_event is signalled and not any(
            thread is not None and thread.is_alive() for thread in threads
        ):
            self._stop_event = None

    @timed
    def start_workers(self) -> None:
        """Start session log writer + cleanup threads if not already running."""
        with self._lock:
            if self._queue is None:
                self._queue = queue.Queue(maxsize=500)
            writer_live = bool(self._worker_thread and self._worker_thread.is_alive())
            cleanup_live = bool(self._cleanup_thread and self._cleanup_thread.is_alive())
            cleanup_on_a_set_event = bool(
                self._stop_event is not None and self._stop_event.is_set() and cleanup_live
            )
            if self._stop_event is None or self._stop_event.is_set():
                self._stop_event = threading.Event()
            stop_event = self._stop_event

            mgr_ref = weakref.ref(self)
            if not writer_live:
                self._worker_thread = threading.Thread(
                    target=_writer_loop,
                    args=(mgr_ref, stop_event, self._queue),
                    name="openrouter-session-log-writer",
                    daemon=True,
                )
                self._worker_thread.start()
            if not cleanup_live or cleanup_on_a_set_event:
                self._cleanup_thread = threading.Thread(
                    target=_cleanup_loop,
                    args=(mgr_ref, stop_event),
                    name="openrouter-session-log-cleanup",
                    daemon=True,
                )
                self._cleanup_thread.start()

    @timed
    def start_assembler_worker(self) -> None:
        """Start the DB-backed session log assembler thread (multi-worker safe)."""
        with self._lock:
            stop_was_set = bool(self._stop_event is not None and self._stop_event.is_set())
            if self._assembler_thread and self._assembler_thread.is_alive() and not stop_was_set:
                return
            if self._stop_event is None or stop_was_set:
                self._stop_event = threading.Event()
            stop_event = self._stop_event

            self._assembler_thread = threading.Thread(
                target=_assembler_loop,
                args=(weakref.ref(self), stop_event),
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
            self._warn_once(
                "pyzipper",
                "Session log storage is enabled but the 'pyzipper' package is not available; skipping persistence.",
            )
            return None

        base_dir = str(valves.SESSION_LOG_DIR or "").strip()
        if not base_dir:
            self._warn_once(
                "dir",
                "Session log storage is enabled but SESSION_LOG_DIR is empty; skipping persistence.",
            )
            return None

        decrypted = EncryptedStr.read(valves.SESSION_LOG_ZIP_PASSWORD)
        password = (decrypted or "").strip()
        if not password:
            self._warn_once(
                "password",
                "Session log storage is enabled but SESSION_LOG_ZIP_PASSWORD is not configured or cannot be decrypted with the current WEBUI_SECRET_KEY; skipping persistence. The stored value looks like a ciphertext but does not decode: it may be damaged, or it may be a passphrase typed with the 'encrypted:' prefix.",
            )
            return None

        zip_compression = valves.SESSION_LOG_ZIP_COMPRESSION
        zip_compresslevel = valves.SESSION_LOG_ZIP_COMPRESSLEVEL
        if zip_compression in {"stored", "lzma"}:
            zip_compresslevel = None

        with contextlib.suppress(Exception), self._lock:
            self._dirs.add(base_dir)

        return base_dir, password.encode("utf-8"), zip_compression, zip_compresslevel

    def _enqueue_archive_job(self, job: Any) -> None:
        with self._lock:
            if self._queue is None:
                self._queue = queue.Queue(maxsize=500)
        if self._queue.full():
            self._warn_archive_queue_full(job)
            return
        try:
            self._queue.put_nowait(job)
            self.start_workers()
        except queue.Full:
            self._warn_archive_queue_full(job)
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
        task: str = "",
    ) -> None:
        """Persist one invocation's session log events into the DB for later assembly.

        The assembler thread merges all segments for a (chat_id, message_id) into a
        single `<SESSION_LOG_DIR>/<user_id>/<chat_id>/<message_id>.zip`.
        """
        from ..core.logging_system import _SessionLogArchiveJob
        from ..storage.owui_files import is_linkable_chat, is_temporary_chat
        from ..storage.persistence import generate_item_id

        temporary = is_temporary_chat(chat_id)
        if not valves.SESSION_LOG_STORE_ENABLED:
            if self.logger.isEnabledFor(logging.DEBUG):
                if temporary:
                    self.logger.debug(
                        "Session log segment skipped (SESSION_LOG_STORE_ENABLED=false): request_id=%s",
                        request_id,
                    )
                else:
                    self.logger.debug(
                        "Session log segment skipped (SESSION_LOG_STORE_ENABLED=false): chat_id=%s message_id=%s request_id=%s",
                        chat_id,
                        message_id,
                        request_id,
                    )
            return
        if not (user_id and request_id):
            self.logger.log(
                warn_level(self._skip_info_emitted, "ids"),
                "Session log segment skipped (missing user_id or request_id): user_id=%s request_id=%s",
                bool(user_id),
                bool(request_id),
            )
            return
        if temporary:
            self._warn_temporary_chat_skip(
                "segment", user_id, "Session log segment skipped (temporary chat): request_id=%s", request_id,
            )
            return
        surrogate_in_play = False
        if not (chat_id and message_id):
            if task and not message_id:
                self._skip_task(task, chat_id, request_id)
                return
            if not getattr(valves, "SESSION_LOG_ARCHIVE_API_CALLS", True):
                self.logger.log(
                    warn_level(self._skip_info_emitted, "valve"),
                    "Session log segment skipped (no chat/message id and SESSION_LOG_ARCHIVE_API_CALLS is off): request_id=%s",
                    request_id,
                )
                return
            chat_id = "api"
            message_id = f"api-{request_id}"
            surrogate_in_play = True
        if not log_events:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Session log segment skipped (no events): chat_id=%s message_id=%s request_id=%s",
                    chat_id,
                    message_id,
                    request_id,
                )
            return
        if (
            not surrogate_in_play
            and is_linkable_chat(chat_id)
            and not await self._caller_owns_chat(chat_id, user_id)
            and not await self._caller_is_admin(user_id)
        ):
            self._latch_ownership_skip(user_id, chat_id, message_id, request_id)
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
            "session_id": "" if surrogate_in_play else str(session_id or ""),
            "chat_id": str(chat_id or ""),
            "message_id": str(message_id or ""),
            "request_id": str(request_id or ""),
            "created_at": time.time(),
            "log_format": valves.SESSION_LOG_FORMAT,
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

        persisted: list[str] = []
        try:
            # Best-effort: background workers may not be long-lived in some OWUI deployment
            # modes, so we also attempt to assemble terminal bundles inline (below).
            self.start_workers()
            self.start_assembler_worker()
            persisted = await self._artifact_store._db_persist([row]) if self._artifact_store else []
        except Exception:
            self.logger.debug(
                "Failed to persist session log segment (chat_id=%s message_id=%s request_id=%s terminal=%s)",
                chat_id,
                message_id,
                request_id,
                terminal,
                exc_info=True,
            )

        if not persisted and getattr(self._artifact_store, "_artifact_key_unreadable", False):
            self.logger.debug(
                "Session log segment not staged (artifact storage refusing writes): "
                "chat_id=%s message_id=%s request_id=%s",
                chat_id,
                message_id,
                request_id,
            )
            return
        if not persisted:
            # Survivability guarantee: if DB staging isn't available (or breaker blocks writes),
            # fall back to direct zip persistence so operators still get logs.
            self.logger.warning(
                "Session log DB staging returned no staged segment; falling back to a queued zip write (chat_id=%s message_id=%s request_id=%s).",
                chat_id,
                message_id,
                request_id,
            )
            base_dir, zip_password, zip_compression, zip_compresslevel = archive_settings
            fallback_message_id = message_id if surrogate_in_play else f"{message_id}.{request_id}"
            meta_message_id, meta_task = _split_archive_key(message_id)
            self._enqueue_archive_job(
                _SessionLogArchiveJob(
                    base_dir=base_dir,
                    zip_password=zip_password,
                    zip_compression=zip_compression,
                    zip_compresslevel=zip_compresslevel,
                    user_id=user_id,
                    session_id="" if surrogate_in_play else session_id,
                    chat_id=chat_id,
                    message_id=fallback_message_id,
                    request_id=request_id,
                    created_at=time.time(),
                    log_format=valves.SESSION_LOG_FORMAT,
                    log_events=log_events,
                    meta_message_id=meta_message_id,
                    meta_task=meta_task,
                    terminal=bool(terminal),
                    status=str(status or "").strip(),
                    reason=str(reason or "").strip(),
                )
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
        if not self.valves.SESSION_LOG_STORE_ENABLED:
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

        batch_size = self.valves.SESSION_LOG_ASSEMBLER_BATCH_SIZE
        lock_stale_seconds = self.valves.SESSION_LOG_LOCK_STALE_SECONDS
        stale_finalize_seconds = float(self.valves.SESSION_LOG_STALE_FINALIZE_SECONDS)
        budget = float(self.valves.SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS)

        with self._lock:
            for _turn in [
                key
                for key in self._rescue_pending
                if self._unreadable_archive_attempts.get(f"{key[0]}:{key[1]}", 0)
                >= _UNREADABLE_ARCHIVE_CAPTURE_AFTER
            ]:
                self._rescue_pending.pop(_turn, None)
            for _latch_name in self._FAULT_LATCHES:
                _truncate_latch(getattr(self, _latch_name), _MAX_DRAIN_LATCH_KEYS)

        self._cleanup_stale_locks(model, session_factory, lock_stale_seconds)
        self._cleanup_stale_segments(
            model, session_factory, int(self.valves.SESSION_LOG_RETENTION_DAYS)
        )

        backed_off = self._backoff_exclusion(lock_stale_seconds)

        contended: list[tuple[str, str]] = []

        def _record(key: tuple[str, str], assembled: bool | Any, *, stale_arm: bool = False) -> None:
            from ..storage.owui_files import is_temporary_chat

            if is_temporary_chat(key[0]):
                return
            if assembled is _LOCK_CONTENDED:
                contended.append(key)
                return
            if assembled is not True and assembled is not False:
                return
            with self._lock:
                if assembled:
                    self._rescue_pending.pop(key, None)
                capture_exempt = key in self._capture_exempt
                if capture_exempt:
                    self._capture_exempt.discard(key)
                if assembled or self._rescue_exempt(key) or capture_exempt:
                    self._assembler_recent_failures.pop(key, None)
                    self._assembly_failure_stale_arm.discard(key)
                else:
                    self._assembler_recent_failures[key] = time.monotonic() + float(lock_stale_seconds)
                    if stale_arm:
                        self._assembly_failure_stale_arm.add(key)
                    else:
                        self._assembly_failure_stale_arm.discard(key)

        candidates = self._candidate_turns(
            model, session_factory, batch_size, stale_finalize_seconds, backed_off
        )
        deadline = time.monotonic() + budget
        set_aside: list[tuple[str, str]] = []
        while True:
            del contended[:]
            over_budget = False
            for index, (terminal, turns) in enumerate(candidates):
                if time.monotonic() >= deadline:
                    self.logger.debug(
                        "Session log assembler pass stopped at %d of %d candidate turn(s) at its "
                        "%.1fs wall-clock budget; the rest are offered again by the next pass",
                        index,
                        len(candidates),
                        budget,
                    )
                    over_budget = True
                    break
                try:
                    if terminal:
                        assembled = self._assemble_and_write_bundle(turns[0], turns[1], terminal=True)
                    else:
                        assembled = self._assemble_and_write_bundle(
                            turns[0], turns[1], terminal=False,
                            stale_finalize_seconds=stale_finalize_seconds,
                        )
                except Exception:
                    self.logger.log(
                        warn_level(self._unreadable_archive_warnings,
                                   f"session_log_assemble_failed:{turns[0]}:{turns[1]}", cooldown_s=3600.0),
                        "Session log assembly raised for chat_id=%s message_id=%s; the assembly lock and the "
                        "staged segments' age are handed back and the pass continues.",
                        turns[0], turns[1], exc_info=True,
                    )
                    assembled = False
                _record(turns, assembled, stale_arm=not terminal)
            if over_budget or not contended:
                break
            set_aside.extend(contended)
            candidates = self._candidate_turns(
                model,
                session_factory,
                batch_size,
                stale_finalize_seconds,
                (*backed_off, *set_aside),
            )

    def _rescue_exempt(self, key: tuple[str, str]) -> bool:
        if key not in self._rescue_pending:
            return False
        turn = f"{key[0]}:{key[1]}"
        return self._unreadable_archive_attempts.get(turn, 0) < _UNREADABLE_ARCHIVE_CAPTURE_AFTER

    def _backoff_exclusion(
        self, lock_stale_seconds: float
    ) -> tuple[tuple[str, str], ...]:
        now = time.monotonic()
        with self._lock:
            for key in [k for k, deadline in self._assembler_recent_failures.items() if deadline <= now]:
                self._assembler_recent_failures.pop(key, None)
                self._assembly_failure_stale_arm.discard(key)
            live = sorted(
                (
                    key
                    for key, deadline in self._assembler_recent_failures.items()
                    if deadline > now
                ),
                key=lambda key: self._assembler_recent_failures[key],
                reverse=True,
            )
        if len(live) > _MAX_EXCLUDED_TURNS:
            self.logger.debug(
                "Session log assembly excluded %d set-aside turn(s) beyond the %d bound",
                len(live) - _MAX_EXCLUDED_TURNS,
                _MAX_EXCLUDED_TURNS,
            )
        return tuple(live[:_MAX_EXCLUDED_TURNS])

    def _candidate_turns(
        self,
        model: Any,
        session_factory: Any,
        batch_size: int,
        stale_finalize_seconds: float,
        backed_off: Sequence[tuple[str, str]],
    ) -> list[tuple[bool, tuple[str, str]]]:
        terminal_excluded = tuple(
            key for key in backed_off if key not in self._assembly_failure_stale_arm
        )
        turns = [(True, turn) for turn in self._list_terminal_messages(
            model, session_factory, limit=batch_size, exclude=terminal_excluded
        )]
        turns += [(False, turn) for turn in self._list_stale_messages(
            model,
            session_factory,
            stale_finalize_seconds=stale_finalize_seconds,
            limit=batch_size,
            exclude=backed_off,
        )]
        return turns

    @timed
    def _cleanup_stale_segments(
        self,
        model: Any,
        session_factory: Any,
        retention_days: int,
    ) -> None:
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=float(retention_days))
        ids: list[str] = []
        try:
            with _db_session(session_factory) as session:
                rows = (
                    session.query(model.id)  # type: ignore[attr-defined]
                    .filter(  # type: ignore[attr-defined]
                        model.item_type.in_(["session_log_segment", "session_log_segment_terminal"])
                    )
                    .filter(model.created_at < cutoff)  # type: ignore[attr-defined]
                    .limit(500)
                    .all()
                )
                ids = [row[0] for row in rows if row and isinstance(row[0], str)]
        except Exception as exc:
            self.logger.debug(
                "Stale segment cleanup skipped — %s: %s", type(exc).__name__, exc, exc_info=True
            )
            return
        if ids and self._artifact_store:
            with contextlib.suppress(Exception):
                self._artifact_store._delete_artifacts_sync(ids)

    def _cleanup_stale_locks(
        self,
        model: Any,
        session_factory: Any,
        lock_stale_seconds: float,
    ) -> None:
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=float(lock_stale_seconds))
        ids: list[str] = []
        try:
            with _db_session(session_factory) as session:
                rows = (
                    session.query(model.id)  # type: ignore[attr-defined]
                    .filter(model.item_type == "session_log_lock")  # type: ignore[attr-defined]
                    .filter(model.created_at < cutoff)  # type: ignore[attr-defined]
                    .limit(500)
                    .all()
                )
                ids = [row[0] for row in rows if row and isinstance(row[0], str)]
        except Exception as exc:
            self.logger.debug("Stale lock cleanup skipped — %s: %s", type(exc).__name__, exc, exc_info=True)
            return
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
        exclude: Collection[tuple[str, str]] = (),
    ) -> list[tuple[str, str]]:
        from ..storage.owui_files import is_temporary_chat

        rows: list[Any] = []
        offered: list[tuple[str, str]] = []
        held: list[tuple[str, str]] = []
        _page = int(limit) + len(exclude)
        while True:
            try:
                with _db_session(session_factory) as session:
                    query = session.query(model.chat_id, model.message_id)  # type: ignore[attr-defined]
                    query = query.filter(model.item_type == "session_log_segment_terminal")  # type: ignore[attr-defined]
                    if exclude:
                        query = query.filter(_set_aside_predicate(model, exclude))
                    rows = (
                        query
                        .group_by(model.chat_id, model.message_id)  # type: ignore[attr-defined]
                        .order_by(func.min(model.created_at).asc())  # type: ignore[attr-defined]
                        .limit(_page)
                        .all()
                    )
                offered, held = _window_rows(rows, exclude, is_temporary_chat, limit)
            except Exception as exc:
                self.logger.debug("Terminal message listing skipped — %s: %s", type(exc).__name__, exc, exc_info=True)
                return []
            if len(rows) < _page or len(offered) >= int(limit):
                break
            _page *= 2
            if _page > 100_000:
                break
        return [*offered, *held]

    @timed
    def _list_stale_messages(
        self,
        model: Any,
        session_factory: Any,
        *,
        stale_finalize_seconds: float,
        limit: int,
        exclude: Collection[tuple[str, str]] = (),
    ) -> list[tuple[str, str]]:
        from ..storage.owui_files import is_temporary_chat

        cutoff = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - datetime.timedelta(seconds=float(stale_finalize_seconds))
        rows: list[Any] = []
        offered: list[tuple[str, str]] = []
        held: list[tuple[str, str]] = []
        _page = int(limit)
        while True:
            try:
                # Candidates (best effort): any message that has at least one segment.
                with _db_session(session_factory) as session:
                    terminal_count = func.sum(
                        case((model.item_type == "session_log_segment_terminal", 1), else_=0)  # type: ignore[attr-defined]
                    )
                    query = session.query(model.chat_id, model.message_id)  # type: ignore[attr-defined]
                    query = query.filter(model.item_type.in_(["session_log_segment", "session_log_segment_terminal"]))  # type: ignore[attr-defined]
                    query = (
                        query
                        .group_by(model.chat_id, model.message_id)  # type: ignore[attr-defined]
                        .having(func.max(model.created_at) < cutoff)  # type: ignore[attr-defined]
                        .having(terminal_count == 0)  # type: ignore[attr-defined]
                    )
                    if exclude:
                        query = query.having(_set_aside_predicate(model, exclude))
                    rows = (
                        query
                        .order_by(func.max(model.created_at).asc())  # type: ignore[attr-defined]
                        .limit(_page)
                        .all()
                    )
                offered, held = _window_rows(rows, exclude, is_temporary_chat, limit)
            except Exception as exc:
                self.logger.debug("Stale message listing skipped — %s: %s", type(exc).__name__, exc, exc_info=True)
                return []
            if len(rows) < _page or len(offered) >= int(limit):
                break
            _page *= 2
            if _page > 100_000:
                break
        return [*offered, *held]

    # =========================================================================
    # Archive Event Helpers
    # =========================================================================

    def read_archive(
        self,
        zip_path: Path,
        settings: tuple[str, bytes, str, int | None],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        import pyzipper

        _, zip_password, _, _ = settings
        meta: dict[str, Any] = {}
        events: list[dict[str, Any]] = []

        with pyzipper.AESZipFile(zip_path, "r") as zf:
            zf.setpassword(zip_password)
            if "meta.json" in zf.namelist():
                try:
                    loaded = json.loads(zf.read("meta.json").decode("utf-8"))
                except (ValueError, RecursionError):
                    loaded = None
                if isinstance(loaded, dict):
                    meta = loaded
            if "logs.jsonl" in zf.namelist():
                content = zf.read("logs.jsonl").decode("utf-8")
                for line in content.strip().split("\n"):
                    if line.strip():
                        try:
                            evt = json.loads(line)
                        except (ValueError, RecursionError):
                            continue  # Skip malformed lines (JSONDecodeError ⊂ ValueError)
                        if isinstance(evt, dict):
                            events.append(self._convert_jsonl_to_internal(evt))
        return meta, events

    def _convert_jsonl_to_internal(self, evt: dict[str, Any]) -> dict[str, Any]:
        """Convert JSONL archive format back to internal event format.

        JSONL uses 'ts' (ISO timestamp), internal uses 'created' (epoch float).
        """
        internal = dict(evt)
        if "ts" in internal and "created" not in internal:
            try:
                ts_str = internal.pop("ts")
                dt = datetime.datetime.fromisoformat(ts_str)
                internal["created"] = dt.timestamp()
            except (AttributeError, TypeError, ValueError, OSError, OverflowError):
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
            # Key: timestamp + request_id + lineno + message hash.
            # Normalise `created` to the archive's millisecond-ISO form — the
            # SAME serialization the writer uses (logging_system.py:
            # isoformat(timespec="milliseconds"), which TRUNCATES to ms). DB
            # events carry full float precision; archive-round-tripped events
            # are ms-truncated. Re-serializing both guarantees identical keys;
            # rounding would not (round vs truncate disagree for ~half of all
            # timestamps), so the same event from DB and archive would fail to
            # dedup and duplicate the log line.
            created_raw = evt.get("created", 0)
            try:
                created_key: Any = datetime.datetime.fromtimestamp(
                    float(created_raw), tz=datetime.UTC
                ).isoformat(timespec="milliseconds")
            except (TypeError, ValueError, OSError, OverflowError):
                created_key = created_raw
            key = "{}:{}:{}:{}".format(
                created_key,
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

    def _capture_unassemblable_turn(
        self,
        chat_id: str,
        message_id: str,
        out_path: Path,
        base_dir: str,
        zip_password: bytes,
        zip_compression: str,
        zip_compresslevel: int | None,
        user_id: str,
        session_id: str,
        segments: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> bool:

        from ..core.logging_system import _archive_file_path, _SessionLogArchiveJob
        from ..core.utils import _stable_crockford_id

        key = f"{chat_id}:{message_id}"
        already = key in self._captured_turns and all(
            i <= self._captured_row_ids.get(key, "") for i in (ids or ())
        )
        self._capture_exempt.discard((chat_id, message_id))
        if already:
            return True
        if not self.valves.SESSION_LOG_STORE_ENABLED:
            self.logger.log(
                warn_level(
                    self._unreadable_archive_warnings,
                    f"session_log_capture_skipped_store_disabled:{key}",
                    cooldown_s=3600.0,
                ),
                "Skipped the separate archive for a stranded session log turn "
                "(chat_id=%s message_id=%s): `Enable session log storage` is off, so the rescue "
                "was not published and the staged segments stay in the database for a pass "
                "after it is back.",
                chat_id,
                message_id,
            )
            return False
        attempts = self._unreadable_archive_attempts.get(key, 0) + 1
        self._unreadable_archive_attempts[key] = attempts
        if attempts < _UNREADABLE_ARCHIVE_CAPTURE_AFTER:
            return False

        events: list[dict[str, Any]] = []
        request_id = _preferred_request_id(segments)
        resolved_status = ""
        resolved_reason = ""
        saw_terminal_segment = False
        for seg in segments:
            if seg.get("type") == "session_log_segment_terminal":
                saw_terminal_segment = True
                raw_status = seg.get("status")
                if isinstance(raw_status, str) and raw_status.strip():
                    resolved_status = raw_status.strip()
                    raw_reason = seg.get("reason")
                    resolved_reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
            seg_events = seg.get("events")
            if isinstance(seg_events, list):
                events.extend(e for e in seg_events if isinstance(e, dict))
        if not events or not request_id:
            return False

        meta_message_id, meta_task = _split_archive_key(message_id)
        fallback_message_id = f"{message_id}.{request_id}"
        rescue_path = _archive_file_path(
            base_dir, user_id=user_id, chat_id=chat_id, message_id=fallback_message_id
        )
        before_stat = None
        with contextlib.suppress(Exception):
            before_stat = rescue_path.stat()
        if before_stat is not None and rescue_path.is_file():
            try:
                _prior_meta, prior_events = self.read_archive(
                    rescue_path, (base_dir, zip_password, zip_compression, zip_compresslevel)
                )
            except Exception:
                self.logger.log(
                    warn_level(self._unreadable_archive_warnings,
                               f"session_log_rescue_unreadable:{rescue_path}", cooldown_s=3600.0),
                    "Refusing to re-capture stranded session log turn chat_id=%s message_id=%s over an "
                    "unreadable rescue archive; the existing file and the newly staged segments are left "
                    "intact (path=%s).", chat_id, message_id, str(rescue_path), exc_info=True,
                )
                return False
            if prior_events:
                events = self.dedupe_events(prior_events + events)
                events.sort(key=lambda evt: float(evt.get("created") or 0.0))
        try:
            self._write_archive(
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
                    log_format=self.valves.SESSION_LOG_FORMAT,
                    log_events=events,
                    meta_message_id=meta_message_id,
                    meta_task=meta_task,
                    terminal=saw_terminal_segment,
                    status=resolved_status,
                    reason=resolved_reason,
                )
            )
        except Exception:
            self.logger.warning(
                "Could not capture stranded session log turn chat_id=%s message_id=%s; "
                "its staged segments stay in the database until session log retention reaps them.",
                chat_id,
                message_id,
                exc_info=True,
            )
            return False

        after_stat = None
        with contextlib.suppress(Exception):
            after_stat = rescue_path.stat()
        wrote = after_stat is not None and (
            before_stat is None
            or after_stat.st_mtime_ns != before_stat.st_mtime_ns
            or after_stat.st_size != before_stat.st_size
        )
        if not wrote:
            self._unreadable_archive_attempts[key] = attempts
            _truncate_latch(self._unreadable_archive_warnings, _MAX_DRAIN_LATCH_KEYS)
            self.logger.log(
                warn_level(
                    self._unreadable_archive_warnings,
                    f"session_log_capture_failed:{key}",
                    cooldown_s=3600.0,
                ),
                "Could not capture stranded session log turn chat_id=%s message_id=%s; "
                "its staged segments stay in the database until session log retention reaps them.",
                chat_id,
                message_id,
            )
            return False

        self._unreadable_archive_attempts.pop(key, None)
        self._captured_turns.add(key)
        self._captured_row_ids[key] = max([self._captured_row_ids.get(key, ""), *list(ids or [])])
        self._capture_exempt.add((chat_id, message_id))
        self._rescue_pending.pop((chat_id, message_id), None)
        self._release_assembly_lock(
            _stable_crockford_id(f"{chat_id}:{message_id}:session_log_lock"), list(ids or [])
        )

        _truncate_latch(self._unreadable_archive_warnings, _MAX_DRAIN_LATCH_KEYS)
        self.logger.log(
            warn_level(
                self._unreadable_archive_warnings,
                f"session_log_turn_captured:{key}",
                cooldown_s=3600.0,
            ),
            "Wrote stranded session log turn to a SEPARATE archive after %d refused attempts: "
            "the existing archive at %s cannot be read, so this turn can never be merged into it. "
            "Those staged segments have now been removed from the database "
            "(chat_id=%s message_id=%s, captured as %s), and the turn's own archive still does not "
            "contain them. That separate file ages like any other archive and is reaped by the retention window. "
            "The usual cause is rotating SESSION_LOG_ZIP_PASSWORD while a turn was mid-assembly; "
            "rotate between turns, not during one.",
                attempts,
                str(out_path),
                chat_id,
                message_id,
                rescue_path.name,
        )
        return True

    def _restore_touched_stamps(
        self,
        model: Any,
        session_factory: Any,
        stamps: dict[str, Any],
        touched_ids: list[str],
        chat_id: str = "",
        message_id: str = "",
    ) -> None:

        if not touched_ids:
            return
        try:
            with _db_session(session_factory) as session:
                for item_id in touched_ids:
                    original = stamps.get(item_id)
                    if original is None:
                        continue
                    session.query(model).filter(model.id == item_id).update(  # type: ignore[attr-defined]
                        {model.created_at: original},  # type: ignore[attr-defined]
                        synchronize_session=False,
                    )
                session.commit()
        except Exception:
            _truncate_latch(self._stale_filter_warnings, _MAX_DRAIN_LATCH_KEYS)
            self.logger.log(
                warn_level(
                    self._stale_filter_warnings,
                    f"session_log_stamp_restore_failed:{chat_id}:{message_id}",
                    cooldown_s=3600.0,
                ),
                "Restoring staged-segment staleness stamps failed for chat_id=%s message_id=%s; "
                "the turn may no longer be surfaced as stale, so its staged segments can be stranded.",
                chat_id,
                message_id,
                exc_info=True,
            )

    def _release_assembly_lock(
        self,
        lock_id: str,
        ids: list[str] | None = None,
    ) -> None:

        with contextlib.suppress(Exception):
            self._artifact_store._delete_artifacts_sync([*(ids or []), lock_id])  # type: ignore[union-attr]

    @timed
    def _assemble_and_write_bundle(
        self,
        chat_id: str,
        message_id: str,
        *,
        terminal: bool,
        stale_finalize_seconds: float = 0.0,
        archive_settings: tuple[str, bytes, str, int | None] | None = None,
    ) -> bool | _LockContended:
        """Assemble all segments for one message into a single zip, then delete DB rows."""
        from ..core.logging_system import (
            _archive_file_path,
            _archive_publish_changed_file,
            _SessionLogArchiveJob,
        )
        from ..core.utils import _stable_crockford_id
        from ..storage.owui_files import is_temporary_chat

        if not (chat_id and message_id):
            return False
        if is_temporary_chat(chat_id):
            self._warn_temporary_chat_skip(
                "assembly", _TEMPORARY_CHAT_PROCESS_SCOPE,
                "Session log assembly skipped (temporary chat): message_id=%s", message_id,
            )
            return _LOCK_CONTENDED
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
            return _LOCK_CONTENDED

        ids: list[str] = []
        stamps: dict[str, Any] = {}
        try:
            # Fetch all segment ids for this message (including any terminal markers).
            with _db_session(session_factory) as session:
                rows = (
                    session.query(model.id, model.created_at)  # type: ignore[attr-defined]
                    .filter(model.chat_id == chat_id)  # type: ignore[attr-defined]
                    .filter(model.message_id == message_id)  # type: ignore[attr-defined]
                    .filter(model.item_type.in_(["session_log_segment", "session_log_segment_terminal"]))  # type: ignore[attr-defined]
                    .order_by(model.created_at.asc())  # type: ignore[attr-defined]
                    .all()
                )
                ids = [row[0] for row in rows if row and isinstance(row[0], str)]
                stamps = {row[0]: row[1] for row in rows if row and isinstance(row[0], str)}

            if not ids:
                self._release_assembly_lock(lock_id)
                return False

            try:
                payloads = self._artifact_store._db_fetch_sync(chat_id, message_id, ids)
            except Exception:
                _truncate_latch(self._read_fault_warnings, _MAX_DRAIN_LATCH_KEYS)
                self.logger.log(
                    warn_level(
                        self._read_fault_warnings,
                        f"session_log_fetch_failed:{chat_id}:{message_id}",
                        cooldown_s=3600.0,
                    ),
                    "Session log segment fetch failed for chat_id=%s message_id=%s; keeping staged segments for retry.",
                    chat_id,
                    message_id,
                    exc_info=True,
                )
                self._release_assembly_lock(lock_id)
                return False

            segments: list[dict[str, Any]] = []
            readable_ids: list[str] = []
            for item_id in ids:
                payload = payloads.get(item_id) if isinstance(payloads, dict) else None
                if not isinstance(payload, dict):
                    continue
                if payload.get("type") in {"session_log_segment", "session_log_segment_terminal"}:
                    segments.append(payload)
                    readable_ids.append(item_id)

            if len(readable_ids) != len(ids):
                _truncate_latch(self._read_fault_warnings, _MAX_DRAIN_LATCH_KEYS)
                self.logger.log(
                    warn_level(
                        self._read_fault_warnings,
                        f"session_log_partial_fetch:{chat_id}:{message_id}",
                        cooldown_s=3600.0,
                    ),
                    "Session log fetch returned %d of %d staged segments for chat_id=%s message_id=%s; "
                    "keeping every staged segment for retry.",
                    len(readable_ids),
                    len(ids),
                    chat_id,
                    message_id,
                )
                self._restore_touched_stamps(model, session_factory, stamps, list(ids), chat_id, message_id)
                self._release_assembly_lock(lock_id)
                return False

            resolved_user_id = ""
            distinct_user_ids = sorted({
                str(seg.get("user_id") or "").strip()
                for seg in segments
                if str(seg.get("user_id") or "").strip()
            })
            if len(distinct_user_ids) > 1:
                self.logger.log(
                    warn_level(
                        self._unreadable_archive_warnings,
                        f"session_log_mixed_user:{chat_id}:{message_id}",
                        cooldown_s=3600.0,
                    ),
                    "Refusing to assemble a session log bundle whose segments name %d "
                    "different users (chat_id=%s message_id=%s users=%s); no archive is "
                    "written under any of them and every staged segment is kept for retry.",
                    len(distinct_user_ids),
                    chat_id,
                    message_id,
                    ",".join(distinct_user_ids),
                )
                self._restore_touched_stamps(model, session_factory, stamps, list(ids), chat_id, message_id)
                self._release_assembly_lock(lock_id)
                return False
            if distinct_user_ids:
                resolved_user_id = distinct_user_ids[0]
            resolved_session_id = ""
            preferred_request_id = _preferred_request_id(segments)
            resolved_status = ""
            resolved_reason = ""
            merged_events: list[dict[str, Any]] = []

            for seg in segments:
                if not resolved_session_id:
                    raw_sid = seg.get("session_id")
                    if isinstance(raw_sid, str) and raw_sid.strip():
                        resolved_session_id = raw_sid.strip()
                if seg.get("type") == "session_log_segment_terminal":
                    raw_status = seg.get("status")
                    if isinstance(raw_status, str) and raw_status.strip():
                        resolved_status = raw_status.strip()
                        raw_reason = seg.get("reason")
                        resolved_reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
                events = seg.get("events")
                if isinstance(events, list):
                    for evt in events:
                        if isinstance(evt, dict):
                            merged_events.append(evt)

            def _event_ts(evt: dict[str, Any]) -> float:
                created = evt.get("created")
                try:
                    return float(created) if created is not None else 0.0
                except (TypeError, ValueError):
                    return 0.0

            merged_events.sort(key=_event_ts)

            if not terminal and any(seg.get("type") == "session_log_segment_terminal" for seg in segments):
                terminal = True

            settings = archive_settings or self.resolve_archive_settings(self.valves)
            if settings is None:
                _truncate_latch(self._unreadable_archive_warnings, _MAX_DRAIN_LATCH_KEYS)
                self.logger.log(
                    warn_level(
                        self._unreadable_archive_warnings,
                        f"session_log_settings_unresolved:{chat_id}:{message_id}",
                        cooldown_s=3600.0,
                    ),
                    "Session log archive settings did not resolve for chat_id=%s message_id=%s; "
                    "keeping staged segments for retry.",
                    chat_id,
                    message_id,
                )
                self._restore_touched_stamps(model, session_factory, stamps, list(ids), chat_id, message_id)
                self._release_assembly_lock(lock_id)
                return False
            base_dir, zip_password, zip_compression, zip_compresslevel = settings

            out_path = _archive_file_path(
                base_dir,
                user_id=resolved_user_id,
                chat_id=chat_id,
                message_id=message_id,
            )
            before_stat = None
            with contextlib.suppress(Exception):
                before_stat = out_path.stat()

            # Merge with existing archive events if the zip already exists.
            existing_raw: list[dict[str, Any]] = []
            existing_meta: dict[str, Any] = {}
            read_failed = False
            if out_path.exists():
                try:
                    existing_meta, existing_raw = self.read_archive(out_path, settings)
                    existing_events = existing_raw
                    existing_count = len(existing_events)
                    db_count = len(merged_events)
                    if existing_events:
                        existing_events = [evt for evt in existing_events if not _is_incomplete_marker(evt)]
                        if existing_events:
                            merged_events = existing_events + merged_events
                            merged_events = self.dedupe_events(merged_events)
                            merged_events.sort(key=_event_ts)
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(
                                "Merged %d existing archive events with %d DB events "
                                "(%d written; chat_id=%s message_id=%s)",
                                existing_count,
                                db_count,
                                len(merged_events),
                                chat_id,
                                message_id,
                            )
                except Exception:
                    read_failed = True
                    _truncate_latch(self._unreadable_archive_warnings, _MAX_DRAIN_LATCH_KEYS)
                    self.logger.log(
                        warn_level(
                            self._unreadable_archive_warnings,
                            f"session_log_archive_unreadable:{out_path}",
                            cooldown_s=3600.0,
                        ),
                        "Refusing to assemble over an unreadable session log archive; "
                        "the existing file and the staged segments are left intact (path=%s chat_id=%s message_id=%s).",
                        str(out_path),
                        chat_id,
                        message_id,
                        exc_info=True,
                    )

            if not terminal and existing_raw and not any(_is_incomplete_marker(evt) for evt in existing_raw):
                terminal = True
                if not resolved_status:
                    resolved_status = str(existing_meta.get("status") or "")
                    resolved_reason = str(existing_meta.get("reason") or "")
            if not terminal and not existing_meta.get("terminal"):
                merged_events.append(
                    _incomplete_marker(
                        preferred_request_id, resolved_session_id, resolved_user_id, stale_finalize_seconds
                    )
                )

            if read_failed:
                self._restore_touched_stamps(model, session_factory, stamps, list(ids), chat_id, message_id)
                self._release_assembly_lock(lock_id)
                captured = self._capture_unassemblable_turn(
                    chat_id,
                    message_id,
                    out_path,
                    base_dir,
                    zip_password,
                    zip_compression,
                    zip_compresslevel,
                    resolved_user_id,
                    resolved_session_id,
                    segments,
                    list(ids),
                )
                if not captured:
                    self._rescue_pending[(chat_id, message_id)] = time.monotonic()
                return False

            if existing_meta.get("terminal"):
                terminal = True

            meta_message_id, meta_task = _split_archive_key(message_id)
            job = _SessionLogArchiveJob(
                base_dir=base_dir,
                zip_password=zip_password,
                zip_compression=zip_compression,
                zip_compresslevel=zip_compresslevel,
                user_id=resolved_user_id,
                session_id="" if chat_id == "api" else (resolved_session_id or ""),
                chat_id=chat_id,
                message_id=message_id,
                request_id=preferred_request_id or "",
                created_at=time.time(),
                log_format=self.valves.SESSION_LOG_FORMAT,
                log_events=merged_events,
                meta_message_id=meta_message_id,
                meta_task=meta_task,
                terminal=terminal,
                status=resolved_status,
                reason=resolved_reason,
            )
            if not self.valves.SESSION_LOG_STORE_ENABLED:
                self.logger.log(
                    warn_level(
                        self._unreadable_archive_warnings,
                        f"session_log_write_skipped_store_disabled:{chat_id}:{message_id}",
                        cooldown_s=3600.0,
                    ),
                    "Session log archive was not written for chat_id=%s message_id=%s: "
                    "`Enable session log storage` went off inside this pass, so the staged "
                    "segments stay in the database for a later pass.",
                    chat_id,
                    message_id,
                )
                self._restore_touched_stamps(model, session_factory, stamps, list(ids), chat_id, message_id)
                self._release_assembly_lock(lock_id)
                return False
            try:
                self._write_archive(job)
            except Exception:
                self._restore_touched_stamps(model, session_factory, stamps, list(ids), chat_id, message_id)
                self._release_assembly_lock(lock_id)
                self.logger.log(
                    warn_level(
                        self._unreadable_archive_warnings,
                        f"session_log_write_failed:{chat_id}:{message_id}",
                        cooldown_s=3600.0,
                    ),
                    "Session log archive write failed for chat_id=%s message_id=%s path=%s; keeping staged segments for retry.",
                    chat_id, message_id, str(out_path),
                    exc_info=True,
                )
                return False

            wrote = _archive_publish_changed_file(out_path, before_stat)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Session log archive write attempted (chat_id=%s message_id=%s terminal=%s wrote=%s)",
                    chat_id,
                    message_id,
                    terminal,
                    wrote,
                )

            if wrote:
                try:
                    self._artifact_store._delete_artifacts_sync(ids + [lock_id])  # type: ignore[union-attr]
                except Exception:
                    _truncate_latch(self._unreadable_archive_warnings, _MAX_DRAIN_LATCH_KEYS)
                    self.logger.log(
                        warn_level(
                            self._unreadable_archive_warnings,
                            f"session_log_published_rows_not_deleted:{chat_id}:{message_id}",
                            cooldown_s=3600.0,
                        ),
                        "Session log archive published for chat_id=%s message_id=%s but its staged "
                        "rows and assembly lock could not be deleted; they stay in the database, so the "
                        "turn is offered again and can hold the assembler's window until a later pass "
                        "deletes them.",
                        chat_id,
                        message_id,
                        exc_info=True,
                    )
                return True

            # If writing failed, keep segments for retry and allow lock reaping.
            self._restore_touched_stamps(model, session_factory, stamps, list(ids), chat_id, message_id)
            self._release_assembly_lock(lock_id)
            _truncate_latch(self._unreadable_archive_warnings, _MAX_DRAIN_LATCH_KEYS)
            self.logger.log(
                warn_level(
                    self._unreadable_archive_warnings,
                    f"session_log_write_failed:{chat_id}:{message_id}",
                    cooldown_s=3600.0,
                ),
                "Session log archive write failed for chat_id=%s message_id=%s path=%s; keeping staged segments for retry.",
                chat_id, message_id, str(out_path),
            )
            return False
        except BaseException:
            self._restore_touched_stamps(model, session_factory, stamps, list(ids), chat_id, message_id)
            self._release_assembly_lock(lock_id)
            raise

    # =========================================================================
    # Archive Cleanup
    # =========================================================================

    @timed
    def _sweep_enabled(self) -> bool:
        return bool(self.valves.SESSION_LOG_STORE_ENABLED)

    def cleanup_archives(self) -> None:
        """Delete expired session log archives and prune empty directories."""
        if not self._sweep_enabled():
            return
        with self._lock:
            dirs = set(self._dirs)
        configured = str(getattr(self.valves, "SESSION_LOG_DIR", "") or "").strip()
        if configured:
            dirs.add(configured)
        if not dirs:
            return
        retention_days = int(self.valves.SESSION_LOG_RETENTION_DAYS)
        cutoff = time.time() - retention_days * 86400

        for base_dir in dirs:
            base_dir = (base_dir or "").strip()
            if not base_dir:
                continue
            root = Path(base_dir).expanduser()
            if not root.exists():
                continue
            try:
                for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
                    for name in filenames:
                        if not name.endswith(".zip"):
                            continue
                        with contextlib.suppress(Exception):
                            path = Path(dirpath) / name
                            stat = path.stat()
                            if stat.st_mtime < cutoff:
                                path.unlink(missing_ok=True)  # type: ignore[arg-type]
                    with contextlib.suppress(Exception):
                        _prune_archive_dir(dirpath)
            except OSError:
                self.logger.debug("Session log cleanup: archive scan failed for %s", base_dir, exc_info=True)
                continue
