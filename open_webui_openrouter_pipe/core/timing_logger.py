"""Function timing instrumentation with direct file output.

Provides:
- @timed decorator for automatic function entrance/exit logging
- timing_scope() context manager for code block timing
- timing_mark() for point-in-time events
- Direct JSONL file output (configured via TIMING_LOG_FILE valve)

Usage:
    from .core.timing_logger import timed, timing_scope, timing_mark

    @timed
    async def my_function():
        with timing_scope("database_query"):
            await db.fetch(...)
        timing_mark("checkpoint_reached")

Enable via valve: ENABLE_TIMING_LOG=True
Output file: TIMING_LOG_FILE (default: logs/timing.jsonl)
"""

from __future__ import annotations

import asyncio
import datetime
import functools
import json
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, TypeVar

# -----------------------------------------------------------------------------
# Global file output state
# -----------------------------------------------------------------------------

_timing_file_lock = threading.Lock()
_timing_file_path: Optional[Path] = None
_timing_file_handle: Optional[Any] = None  # File object when open

# Per-request timing buffer (kept for session log integration if needed)
_timing_events: Dict[str, Deque[Dict[str, Any]]] = {}
_timing_lock = threading.Lock()

# Context variables for per-request state
_timing_enabled: ContextVar[bool] = ContextVar("timing_enabled", default=False)
_timing_request_id: ContextVar[Optional[str]] = ContextVar(
    "timing_request_id", default=None
)

# Maximum events per request to prevent unbounded growth
MAX_TIMING_EVENTS = 10000


# -----------------------------------------------------------------------------
# TimingEvent dataclass
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class TimingEvent:
    """Single timing event for entrance or exit."""

    ts: float  # time.perf_counter() for high-precision relative timing
    wall_ts: float  # time.time() for absolute/ISO timestamp
    event: str  # "enter" or "exit"
    label: str  # function/scope name
    elapsed_ms: Optional[float] = None  # Only on exit events


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _format_iso_utc(wall_ts: float) -> str:
    """Format wall clock time as ISO 8601 UTC string."""
    try:
        dt = datetime.datetime.fromtimestamp(wall_ts, tz=datetime.timezone.utc)
        return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except Exception:
        return datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")


def _record_event(event: TimingEvent) -> None:
    """Write timing event directly to the configured file.

    Events are written immediately in JSONL format. Thread-safe via file lock.
    """
    global _timing_file_handle

    if not _timing_enabled.get():
        return
    request_id = _timing_request_id.get()
    if not request_id:
        return

    record: Dict[str, Any] = {
        "ts": _format_iso_utc(event.wall_ts),
        "perf_ts": round(event.ts, 6),
        "event": event.event,
        "label": event.label,
        "request_id": request_id,
    }
    if event.elapsed_ms is not None:
        record["elapsed_ms"] = round(event.elapsed_ms, 3)

    # Write directly to file (thread-safe)
    with _timing_file_lock:
        if _timing_file_handle is not None:
            try:
                line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                _timing_file_handle.write(line + "\n")
                _timing_file_handle.flush()  # Ensure immediate write
            except Exception:
                pass  # Silently ignore write errors to avoid disrupting request flow

    # Also store in per-request buffer for potential session log integration
    with _timing_lock:
        if request_id not in _timing_events:
            _timing_events[request_id] = deque(maxlen=MAX_TIMING_EVENTS)
        _timing_events[request_id].append(record)


# -----------------------------------------------------------------------------
# Public API: File configuration
# -----------------------------------------------------------------------------


def configure_timing_file(file_path: str) -> bool:
    """Configure the timing log file path and open it for writing.

    Call this once at startup (e.g., in Pipe.__init__) when ENABLE_TIMING_LOG is True.
    Creates parent directories automatically if needed.

    Args:
        file_path: Path to the timing log file (e.g., "logs/timing.jsonl")

    Returns:
        True if file was successfully opened, False otherwise
    """
    global _timing_file_path, _timing_file_handle

    with _timing_file_lock:
        # Close existing file if open
        if _timing_file_handle is not None:
            try:
                _timing_file_handle.close()
            except Exception:
                pass
            _timing_file_handle = None

        try:
            path = Path(file_path)
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            # Open file in append mode
            _timing_file_handle = open(path, "a", encoding="utf-8")
            _timing_file_path = path
            return True
        except Exception:
            _timing_file_path = None
            _timing_file_handle = None
            return False


def close_timing_file() -> None:
    """Close the timing log file.

    Call this during shutdown if needed. Safe to call multiple times.
    """
    global _timing_file_handle, _timing_file_path

    with _timing_file_lock:
        if _timing_file_handle is not None:
            try:
                _timing_file_handle.close()
            except Exception:
                pass
            _timing_file_handle = None
            _timing_file_path = None


def ensure_timing_file_configured(file_path: str) -> bool:
    """Ensure the timing log file is configured, opening it if needed.

    This is a lazy configuration function that should be called when timing
    is enabled at runtime. It only opens the file if it hasn't been opened yet
    or if the path has changed.

    This solves the problem where ENABLE_TIMING_LOG is False at pipe load time
    but enabled later via the UI - the file would never get configured because
    configure_timing_file() only ran in __init__.

    Args:
        file_path: Path to the timing log file (e.g., "logs/timing.jsonl")

    Returns:
        True if file is ready for writing (already open or newly opened),
        False if configuration failed
    """
    global _timing_file_path, _timing_file_handle

    with _timing_file_lock:
        # Check if already configured with the same path
        if _timing_file_handle is not None and _timing_file_path is not None:
            if str(_timing_file_path) == str(Path(file_path)):
                return True
            # Path changed, close old file
            try:
                _timing_file_handle.close()
            except Exception:
                pass
            _timing_file_handle = None

    # Not configured or path changed, configure now
    return configure_timing_file(file_path)


# -----------------------------------------------------------------------------
# Public API: Context management
# -----------------------------------------------------------------------------


def set_timing_context(request_id: str, enabled: bool) -> None:
    """Set timing context for the current request.

    Call this at the start of request handling to enable/disable timing
    based on the ENABLE_TIMING_LOG valve setting.

    Args:
        request_id: Unique identifier for the request (same as SessionLogger uses)
        enabled: Whether timing is enabled for this request
    """
    _timing_request_id.set(request_id)
    _timing_enabled.set(enabled)


def clear_timing_context() -> None:
    """Clear timing context for the current request."""
    _timing_request_id.set(None)
    _timing_enabled.set(False)


def get_timing_events(request_id: str) -> List[Dict[str, Any]]:
    """Retrieve timing events for a request (for session log archival).

    Args:
        request_id: The request ID to get events for

    Returns:
        List of timing event dictionaries, ready for JSON serialization
    """
    with _timing_lock:
        buffer = _timing_events.get(request_id)
        return list(buffer) if buffer else []


def clear_timing_events(request_id: str) -> None:
    """Clear timing events for a request after archival.

    Args:
        request_id: The request ID to clear events for
    """
    with _timing_lock:
        _timing_events.pop(request_id, None)


def format_timing_jsonl(request_id: str) -> str:
    """Format timing events as JSONL string for archive inclusion.

    Args:
        request_id: The request ID to format events for

    Returns:
        JSONL-formatted string of timing events, or empty string if none
    """
    events = get_timing_events(request_id)
    if not events:
        return ""
    lines = []
    for evt in events:
        try:
            lines.append(json.dumps(evt, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            # Skip malformed events
            pass
    result = "\n".join(lines)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


def timing_mark(label: str) -> None:
    """Record a single point-in-time timing event.

    Use this for marking specific moments during execution without needing
    a scope/duration measurement. Useful for tracking:
    - HTTP request start
    - First chunk received
    - Stream completion

    Args:
        label: Descriptive label for this timing point

    Usage:
        timing_mark("http_request_start")
        async with session.post(...) as resp:
            timing_mark("http_first_response")
    """
    # Early exit if timing disabled - zero overhead
    if not _timing_enabled.get():
        return
    now_perf = time.perf_counter()
    now_wall = time.time()
    _record_event(TimingEvent(
        ts=now_perf,
        wall_ts=now_wall,
        event="mark",
        label=label,
    ))


# -----------------------------------------------------------------------------
# Public API: timing_scope context manager
# -----------------------------------------------------------------------------


@contextmanager
def timing_scope(label: str):
    """Context manager for timing a code block.

    Records enter/exit events with elapsed time.

    Usage:
        with timing_scope("expensive_operation"):
            do_work()

        with timing_scope("http_post"):
            async with session.post(...) as resp:
                ...

    Args:
        label: Human-readable label for this timing scope
    """
    # Early exit if timing disabled - zero overhead
    if not _timing_enabled.get():
        yield
        return
    start_perf = time.perf_counter()
    start_wall = time.time()
    _record_event(
        TimingEvent(
            ts=start_perf,
            wall_ts=start_wall,
            event="enter",
            label=label,
        )
    )
    try:
        yield
    finally:
        end_perf = time.perf_counter()
        elapsed_ms = (end_perf - start_perf) * 1000
        _record_event(
            TimingEvent(
                ts=end_perf,
                wall_ts=time.time(),
                event="exit",
                label=label,
                elapsed_ms=elapsed_ms,
            )
        )


# -----------------------------------------------------------------------------
# Public API: @timed decorator
# -----------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def timed(func: F) -> F:
    """Decorator for timing function entrance/exit.

    Works with both sync and async functions. Records the fully-qualified
    function name as the label.

    Usage:
        @timed
        def sync_function():
            ...

        @timed
        async def async_function():
            ...

    Args:
        func: The function to wrap with timing instrumentation

    Returns:
        Wrapped function that records timing events
    """
    # Build a readable label from module + qualified name
    module = getattr(func, "__module__", "") or ""
    qualname = getattr(func, "__qualname__", "") or getattr(func, "__name__", "unknown")

    # Shorten common prefixes for readability
    if module.startswith("open_webui_openrouter_pipe."):
        module = module[len("open_webui_openrouter_pipe.") :]

    label = f"{module}.{qualname}" if module else qualname

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Early exit if timing disabled - zero overhead
            if not _timing_enabled.get():
                return await func(*args, **kwargs)
            start_perf = time.perf_counter()
            start_wall = time.time()
            _record_event(
                TimingEvent(
                    ts=start_perf,
                    wall_ts=start_wall,
                    event="enter",
                    label=label,
                )
            )
            try:
                return await func(*args, **kwargs)
            finally:
                end_perf = time.perf_counter()
                elapsed_ms = (end_perf - start_perf) * 1000
                _record_event(
                    TimingEvent(
                        ts=end_perf,
                        wall_ts=time.time(),
                        event="exit",
                        label=label,
                        elapsed_ms=elapsed_ms,
                    )
                )

        return async_wrapper  # type: ignore[return-value]
    else:

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Early exit if timing disabled - zero overhead
            if not _timing_enabled.get():
                return func(*args, **kwargs)
            start_perf = time.perf_counter()
            start_wall = time.time()
            _record_event(
                TimingEvent(
                    ts=start_perf,
                    wall_ts=start_wall,
                    event="enter",
                    label=label,
                )
            )
            try:
                return func(*args, **kwargs)
            finally:
                end_perf = time.perf_counter()
                elapsed_ms = (end_perf - start_perf) * 1000
                _record_event(
                    TimingEvent(
                        ts=end_perf,
                        wall_ts=time.time(),
                        event="exit",
                        label=label,
                        elapsed_ms=elapsed_ms,
                    )
                )

        return sync_wrapper  # type: ignore[return-value]
