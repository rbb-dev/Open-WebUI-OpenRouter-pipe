from __future__ import annotations

import asyncio
import atexit
import contextlib
import contextvars
import logging
import math
import os
import sys
import tempfile
import threading
import time
import traceback
import weakref
from typing import Any, NamedTuple

from .process_state import _upgrade_process_state
from .warn_latch import warn_level

try:
    import faulthandler
except ImportError:
    faulthandler = None  # type: ignore

logger = logging.getLogger(__name__)

_STALL_ALARM_HOLDER_KEY = "_openrouter_pipe_stall_alarm"

_STALL_TICK_SECONDS: float = 1.0

_STALL_BEAT_SECONDS: float = 0.5

_STALL_THREAD_NAME = "openrouter-pipe-stall-alarm"

_STALL_DUMP_PREFIX = "openrouter-pipe-stall-"

_STALL_DUMP_MAX_BYTES = 1 << 20

_STALL_DUMP_LOG_BYTES = 16 << 10

_STALL_STACK_FRAMES = 40

_STALL_DEFAULT_LOOP_SECONDS = 5.0

_STALL_DEFAULT_GIL_SECONDS = 3.0

_STALL_DEFAULT_REPEAT_SECONDS = 60.0

_warned_stall_alarm: set[str] = set()


class _StallSettings(NamedTuple):
    loop_seconds: float
    gil_seconds: float
    repeat_seconds: float
    dump: bool


class _StallAttachment:
    def __init__(self, pipe: Any, state: Any) -> None:
        self.pipe_ref = weakref.ref(pipe)
        self.logger = getattr(pipe, "logger", None) or logger
        self.state = state
        self.closed = False
        self.beat_at = 0.0
        self.beat_task: asyncio.Task[None] | None = None
        self.loop_ref: weakref.ref[asyncio.AbstractEventLoop] | None = None
        self.loop_thread: int | None = None


class _StallAlarmState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.attachments: list[_StallAttachment] = []
        self.watcher: threading.Thread | None = None
        self.armed = False
        self.stopped = False
        self.pid = 0
        self.dump_fd: int | None = None
        self.dump_path: str | None = None
        self.dump_seen = 0
        self.last_tick = 0.0
        self.stalls: dict[int, float] = {}
        self.loop_warned_at: float | None = None
        self.pause_warned_at: float | None = None
        self.conflicts: set[tuple[_StallSettings, _StallSettings]] = set()


class _StallAlarmHolder:
    def __init__(self) -> None:
        self.state = _StallAlarmState()

    def adopt_from(self, previous: Any) -> None:
        carried = getattr(previous, "state", None)
        if carried is not None:
            self.state = carried


def _stall_seconds(valves: Any, name: str, default: float) -> float:
    try:
        value = float(getattr(valves, name, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0 else default


def _stall_settings(attachment: Any) -> _StallSettings | None:
    pipe = attachment.pipe_ref()
    if attachment.closed or pipe is None:
        return None
    valves = getattr(pipe, "valves", None)
    if getattr(valves, "STALL_ALARM_ENABLED", False) is not True:
        return None
    return _StallSettings(
        _stall_seconds(valves, "STALL_ALARM_LOOP_SECONDS", _STALL_DEFAULT_LOOP_SECONDS),
        _stall_seconds(valves, "STALL_ALARM_GIL_SECONDS", _STALL_DEFAULT_GIL_SECONDS),
        _stall_seconds(valves, "STALL_ALARM_REPEAT_SECONDS", _STALL_DEFAULT_REPEAT_SECONDS),
        getattr(valves, "STALL_ALARM_DUMP_ENABLED", False) is True,
    )


def _stall_live_locked(state: Any) -> list[tuple[Any, _StallSettings]]:
    live: list[tuple[Any, _StallSettings]] = []
    for attachment in list(state.attachments):
        if attachment.closed or attachment.pipe_ref() is None:
            attachment.closed = True
            state.attachments.remove(attachment)
            continue
        settings = _stall_settings(attachment)
        if settings is not None:
            live.append((attachment, settings))
    return live


def _stall_latch_open(state: Any, field: str, now: float, window: float) -> bool:
    last = getattr(state, field)
    if last is not None and now - last < window:
        return False
    setattr(state, field, now)
    return True


def _stall_thread_stack(thread: int) -> str:
    frame = sys._current_frames().get(thread)
    if frame is None:
        return "<no frame available>\n"
    return "".join(traceback.format_stack(frame, limit=_STALL_STACK_FRAMES))


def _stall_arm(state: Any, gil_seconds: float) -> tuple[str | None, Exception | None]:
    if faulthandler is None:
        return None, None
    created = None
    try:
        if state.dump_fd is None:
            fd, path = tempfile.mkstemp(
                prefix=f"{_STALL_DUMP_PREFIX}{os.getpid()}-", suffix=".txt"
            )
            state.dump_fd = fd
            state.dump_path = path
            state.dump_seen = 0
            created = path
        elif os.fstat(state.dump_fd).st_size > _STALL_DUMP_MAX_BYTES:
            os.ftruncate(state.dump_fd, 0)
            os.lseek(state.dump_fd, 0, os.SEEK_SET)
            state.dump_seen = 0
        faulthandler.dump_traceback_later(
            gil_seconds, repeat=False, file=state.dump_fd, exit=False
        )
        state.armed = True
    except (OSError, RuntimeError, ValueError, OverflowError) as exc:
        return created, exc
    return created, None


def _stall_read_new_dump(state: Any) -> tuple[str | None, Exception | None]:
    fd = state.dump_fd
    if fd is None:
        return None, None
    try:
        size = os.fstat(fd).st_size
        if size <= state.dump_seen:
            state.dump_seen = min(state.dump_seen, size)
            return None, None
        start = state.dump_seen
        state.dump_seen = size
        wanted = min(size - start, _STALL_DUMP_LOG_BYTES)
        text = os.pread(fd, wanted, start).decode("utf-8", "replace")
    except (OSError, AttributeError) as exc:
        return None, exc
    if size - start > wanted:
        text += "\n[dump truncated]\n"
    return text, None


def _stall_release_dump_locked(state: Any) -> None:
    if state.armed and faulthandler is not None:
        with contextlib.suppress(Exception):
            faulthandler.cancel_dump_traceback_later()
    state.armed = False
    fd, path = state.dump_fd, state.dump_path
    state.dump_fd = None
    state.dump_path = None
    state.dump_seen = 0
    if fd is None:
        return
    empty = False
    with contextlib.suppress(OSError):
        empty = os.fstat(fd).st_size == 0
    with contextlib.suppress(OSError):
        os.close(fd)
    if empty and path is not None:
        with contextlib.suppress(OSError):
            os.unlink(path)


def _stall_release_locked(state: Any) -> None:
    _stall_release_dump_locked(state)
    atexit.unregister(_stall_stop_at_exit)
    state.last_tick = 0.0
    state.stalls.clear()
    state.watcher = None


def _stall_stop_at_exit(state: Any) -> None:
    if state.pid != os.getpid() or not state.lock.acquire(timeout=1.0):
        return
    try:
        state.stopped = True
        _stall_release_locked(state)
    finally:
        state.lock.release()


def _stall_report_pause(
    state: Any,
    log: Any,
    settings: _StallSettings,
    now: float,
    gap: float,
    dump: str | None,
) -> None:
    if not _stall_latch_open(state, "pause_warned_at", now, settings.repeat_seconds):
        return
    causes = (
        "Typical causes are one long call that holds Python's global interpreter lock "
        "(a very large JSON parse or encode, a regex, a compression or a string join), "
        "memory pressure or swapping, or the host pausing the process."
    )
    if dump is None:
        log.warning(
            "Stall alarm: the watcher thread did not run for %.1fs (it checks every %.1fs; "
            "STALL_ALARM_GIL_SECONDS is %.1fs), so the whole process was unresponsive, "
            "the event loop included. %s %s",
            gap,
            _STALL_TICK_SECONDS,
            settings.gil_seconds,
            "No thread dump was written."
            if settings.dump
            else "No thread dump was taken because STALL_ALARM_DUMP_ENABLED is off; switch it "
            "on to capture the stack of a call that holds the process.",
            causes,
        )
        return
    log.warning(
        "Stall alarm: the watcher thread did not run for %.1fs (it checks every %.1fs; "
        "STALL_ALARM_GIL_SECONDS is %.1fs), so the whole process was unresponsive, the "
        "event loop included. %s The thread dump taken during the pause was written to "
        "%s and follows; a thread whose latest frame is an ordinary call rather than a "
        "wait is a candidate for the one that held the process.\n%s",
        gap,
        _STALL_TICK_SECONDS,
        settings.gil_seconds,
        causes,
        state.dump_path,
        dump,
    )


def _stall_check_loops(
    state: Any, live: list[tuple[Any, _StallSettings]], settings: _StallSettings, now: float
) -> None:
    newest: dict[int, float] = {}
    loggers: dict[int, Any] = {}
    for attachment, _ in live:
        loop = attachment.loop_ref() if attachment.loop_ref is not None else None
        thread = attachment.loop_thread
        if thread is None or loop is None or loop.is_closed() or not loop.is_running():
            continue
        if attachment.beat_at >= newest.get(thread, 0.0):
            newest[thread] = attachment.beat_at
            loggers[thread] = attachment.logger
    for thread, beat_at in newest.items():
        age = now - beat_at
        if age >= settings.loop_seconds:
            if _stall_latch_open(state, "loop_warned_at", now, settings.repeat_seconds):
                state.stalls.setdefault(thread, beat_at)
                loggers[thread].warning(
                    "Stall alarm: the event loop has not run for %.1fs (limit %.1fs, "
                    "STALL_ALARM_LOOP_SECONDS). The loop thread is currently inside:\n%s"
                    "A call that holds Python's global interpreter lock keeps this check "
                    "from running at all; that case is reported as a watcher pause instead.",
                    age,
                    settings.loop_seconds,
                    _stall_thread_stack(thread),
                )
        elif thread in state.stalls:
            since = state.stalls.pop(thread)
            loggers[thread].warning(
                "Stall alarm: the event loop is running again after %.1fs without a heartbeat.",
                beat_at - since,
            )
    for thread in [t for t in state.stalls if t not in newest]:
        del state.stalls[thread]


def _stall_note_conflicts(
    state: Any, log: Any, live: list[tuple[Any, _StallSettings]], settings: _StallSettings
) -> None:
    for _, other in live[1:]:
        key = (settings, other)
        if other != settings and key not in state.conflicts:
            state.conflicts.add(key)
            log.warning(
                "Stall alarm: pipe instances in this process configure it differently. "
                "The first instance's values apply (STALL_ALARM_LOOP_SECONDS %.1f, "
                "STALL_ALARM_GIL_SECONDS %.1f, STALL_ALARM_REPEAT_SECONDS %.1f, "
                "STALL_ALARM_DUMP_ENABLED %s) and another's (%.1f, %.1f, %.1f, %s) is ignored.",
                *settings,
                *other,
            )


def _stall_tick(state: Any) -> bool:
    now = time.monotonic()
    with state.lock:
        state.wake.clear()
        live = _stall_live_locked(state)
        if state.stopped or not live:
            _stall_release_locked(state)
            return False
        gap = now - state.last_tick if state.last_tick else 0.0
        state.last_tick = now
        attachment, settings = live[0]
        dump, read_failure = _stall_read_new_dump(state)
        created, arm_failure = None, None
        if settings.dump:
            created, arm_failure = _stall_arm(state, settings.gil_seconds)
        elif dump is None and (state.armed or state.dump_fd is not None):
            _stall_release_dump_locked(state)
    if created is not None:
        logger.info(
            "Stall alarm: the thread dump is armed. If this worker stalls as a whole, the dump is "
            "written to %s, which stays empty while nothing stalls.",
            created,
        )
    if read_failure is not None:
        logger.debug("Stall alarm could not read its thread dump", exc_info=read_failure)
    if arm_failure is not None:
        level = warn_level(_warned_stall_alarm, "arm")
        logger.log(level, "Stall alarm could not arm its thread dump", exc_info=arm_failure)
    if gap >= settings.gil_seconds or dump is not None:
        _stall_report_pause(state, attachment.logger, settings, now, gap, dump)
    else:
        _stall_check_loops(state, live, settings, now)
    _stall_note_conflicts(state, attachment.logger, live, settings)
    return True


def _stall_watch(state: Any) -> None:
    try:
        while True:
            try:
                if not _stall_tick(state):
                    return
            except Exception:
                level = warn_level(_warned_stall_alarm, "tick")
                logger.log(level, "Stall alarm check failed", exc_info=True)
            state.wake.wait(_STALL_TICK_SECONDS)
    finally:
        with state.lock:
            if state.watcher is threading.current_thread():
                _stall_release_locked(state)


async def _stall_beat(attachment: Any) -> None:
    task = asyncio.current_task()
    while attachment.beat_task is task and _stall_settings(attachment) is not None:
        attachment.beat_at = time.monotonic()
        await asyncio.sleep(_STALL_BEAT_SECONDS)


def _stall_ensure_beat(attachment: Any) -> None:
    if _stall_settings(attachment) is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = attachment.beat_task
    if task is not None and not task.done() and task.get_loop() is loop:
        return
    attachment.loop_ref = weakref.ref(loop)
    attachment.loop_thread = threading.get_ident()
    attachment.beat_at = time.monotonic()
    attachment.beat_task = loop.create_task(
        _stall_beat(attachment),
        name="openrouter-stall-beat",
        context=contextvars.Context(),
    )


def ensure_stall_alarm(attachment: Any) -> None:
    if attachment is None:
        return
    _stall_ensure_beat(attachment)
    state = attachment.state
    if state.watcher is not None:
        return
    with state.lock:
        if state.stopped or state.watcher is not None or _stall_settings(attachment) is None:
            return
        watcher = threading.Thread(
            target=_stall_watch, args=(state,), name=_STALL_THREAD_NAME, daemon=True
        )
        state.watcher = watcher
        state.pid = os.getpid()
        try:
            watcher.start()
        except RuntimeError:
            state.watcher = None
            raise
        atexit.register(_stall_stop_at_exit, state)


def attach_stall_alarm(pipe: Any) -> Any:
    holder = _upgrade_process_state(_STALL_ALARM_HOLDER_KEY, _StallAlarmHolder)
    state = holder.state
    attachment = _StallAttachment(pipe, state)
    with state.lock:
        state.attachments.append(attachment)
    return attachment


def detach_stall_alarm(attachment: Any) -> asyncio.Task[None] | None:
    if attachment is None:
        return None
    state = attachment.state
    with state.lock:
        attachment.closed = True
        with contextlib.suppress(ValueError):
            state.attachments.remove(attachment)
    state.wake.set()
    return attachment.beat_task
