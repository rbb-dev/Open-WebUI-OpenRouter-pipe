"""Shared per-process stats collectors for pipe_dashboard.

Provides a single source of truth for reading concurrency, queue,
rate-limit, and session metrics from a Pipe instance.  Both
``runtime_metrics.collect_fast_stats`` (single-worker SSE) and
``dashboard_publisher._collect_worker_payload`` (multi-worker Redis)
delegate to these functions.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ...core.circuit_breaker import counted_tool_failures
from ...core.warn_latch import warn_level

logger = logging.getLogger(__name__)

PROCESS_START = time.monotonic()

_warned_collectors: set[str] = set()


TRANSPORT_SESSION_STATES = ("none", "active", "closed")


def collect_transport_session_state(pipe: Any) -> str:
    reader = getattr(getattr(pipe, "_multimodal_handler", None), "transport_session_state", None)
    if not callable(reader):
        return "none"
    try:
        state = reader()
        if state not in TRANSPORT_SESSION_STATES:
            raise ValueError(state)
    except ValueError:
        _level = warn_level(_warned_collectors, 'transport_session_domain')
        logger.log(
            _level,
            "pipe_dashboard: the vetted transport reported a session state this "
            "dashboard cannot render; it will be reported as not yet opened",
            exc_info=True,
        )
        return "none"
    except (AttributeError, TypeError, RuntimeError):
        _level = warn_level(_warned_collectors, 'transport_session_state')
        logger.log(
            _level,
            "pipe_dashboard: cannot read the vetted transport's session state; the "
            "dashboard will report it as not yet opened",
            exc_info=True,
        )
        return "none"
    return state


def _limits(pipe: Any) -> Any:
    from ...pipe import _get_process_limits

    return _get_process_limits().for_id(getattr(pipe, "id", ""))


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (OverflowError, TypeError, ValueError):
        _level = warn_level(_warned_collectors, 'safe_int')
        logger.log(
            _level,
            "pipe_dashboard: a collected metric was not an integer; reporting 0 "
            "for it until this is fixed",
            exc_info=True,
        )
        return 0


def _waiter_count(sem: Any) -> int:
    try:
        waiters = getattr(sem, "_waiters", None)
        return len(waiters) if waiters is not None else 0
    except (AttributeError, TypeError):
        _level = warn_level(_warned_collectors, 'waiter_count')
        logger.log(
            _level,
            "pipe_dashboard: cannot read semaphore waiters; the dashboard will "
            "report an empty wait queue",
            exc_info=True,
        )
        return 0


def _semaphore_active(sem: Any, limit: int) -> int:
    """Live usage of a semaphore, or 0 when its internals cannot be read."""
    if sem is None or not limit:
        return 0
    try:
        return max(0, limit - int(sem._value)) + _safe_int(getattr(sem, "_debt", 0))
    except (AttributeError, TypeError, ValueError):
        _level = warn_level(_warned_collectors, 'semaphore_active')
        logger.log(
            _level,
            "pipe_dashboard: cannot read semaphore usage; the dashboard will "
            "report 0 active for it",
            exc_info=True,
        )
        return 0


def collect_concurrency(pipe: Any) -> dict[str, int]:
    """Read concurrency semaphore state from the pipe."""
    slots = _limits(pipe)
    sem = slots.request_semaphore
    sem_limit = slots.request_limit or 0
    tool_sem = slots.tool_semaphore
    tool_limit = slots.tool_limit or 0
    # Fall back to valve config when semaphores aren't materialized yet
    if not sem_limit:
        valves = getattr(pipe, "valves", None)
        sem_limit = getattr(valves, "MAX_CONCURRENT_REQUESTS", 0) or 0
    if not tool_limit:
        valves = getattr(pipe, "valves", None)
        tool_limit = getattr(valves, "MAX_PARALLEL_TOOLS_GLOBAL", 0) or 0
    return {
        "active_requests": _semaphore_active(sem, sem_limit),
        "max_requests": sem_limit,
        "active_tools": _semaphore_active(tool_sem, tool_limit),
        "max_tools": tool_limit,
    }


def collect_queues(pipe: Any) -> dict[str, int]:
    """Read queue depths, bounds, and semaphore wait backlogs from the pipe."""
    rq = getattr(pipe, "_request_queue", None)
    lq = getattr(pipe, "_log_queue", None)
    slm = getattr(pipe, "_session_log_manager", None)
    archive_q = getattr(slm, "_queue", None) if slm else None
    return {
        "requests": rq.qsize() if rq else 0,
        "requests_max": _safe_int(getattr(pipe, "_QUEUE_MAXSIZE", 1000)) or 1000,
        "waiting": _waiter_count(_limits(pipe).request_semaphore),
        "tool_waiting": _waiter_count(_limits(pipe).tool_semaphore),
        "logs": lq.qsize() if lq else 0,
        "logs_max": _safe_int(getattr(lq, "maxsize", 0)) if lq else 0,
        "archive": archive_q.qsize() if archive_q else 0,
        "archive_max": _safe_int(getattr(archive_q, "maxsize", 0)) if archive_q else 0,
    }


def collect_video_pool(pipe: Any) -> dict[str, int]:
    """Read the video-generation concurrency pool (limit + live usage)."""
    slots = _limits(pipe)
    limit = _safe_int(slots.video_limit)
    sem = slots.video_semaphore
    if sem is not None and limit:
        active = _semaphore_active(sem, limit)
    else:
        try:
            active = len(getattr(pipe, "_video_active_tasks", {}) or {})
        except TypeError:
            _level = warn_level(_warned_collectors, 'video_active')
            logger.log(
                _level,
                "pipe_dashboard: cannot count active video tasks; reporting 0",
                exc_info=True,
            )
            active = 0
    return {"active": active, "max": limit}


def collect_rate_limits(pipe: Any) -> dict[str, Any]:
    """Read circuit breaker / rate limit state from the pipe."""
    cb = getattr(pipe, "_circuit_breaker", None)
    if not cb:
        return {
            "tracked_users": 0,
            "users_with_failures": 0,
            "tripped_users": 0,
            "threshold": 0,
            "window_s": 0,
            "tool_tracked": 0,
            "tool_with_failures": 0,
            "tool_tripped": 0,
            "auth_failures_active": 0,
        }

    threshold = getattr(cb, "_threshold", 0)
    window = getattr(cb, "_window_seconds", 0.0)
    now = time.time()
    cutoff = now - window if window else 0

    # Request breakers
    records = getattr(cb, "_breaker_records", {})
    tracked = 0
    with_failures = 0
    tripped = 0
    for dq in list(records.values()):
        if dq and (not cutoff or dq[-1] > cutoff):
            tracked += 1
        if cutoff:
            recent = 0
            try:
                for recent, ts in enumerate(reversed(dq), start=1):
                    if ts <= cutoff:
                        recent -= 1
                        break
                    if recent >= threshold:
                        break
            except RuntimeError:
                recent = 0
                for recent, ts in enumerate(reversed(list(dq)), start=1):
                    if ts <= cutoff:
                        recent -= 1
                        break
                    if recent >= threshold:
                        break
        else:
            recent = len(dq)
        if recent > 0:
            with_failures += 1
        if recent >= threshold:
            tripped += 1

    # Tool breakers
    tool_breakers = getattr(cb, "_tool_breakers", {})
    tool_tracked = 0
    tool_tripped = 0
    tool_with_failures = 0
    for user_tools in list(tool_breakers.values()):
        for dq in list(user_tools.values()):
            tool_count = counted_tool_failures(dq, now, window)
            if tool_count > 0:
                tool_tracked += 1
                tool_with_failures += 1
            if tool_count >= threshold:
                tool_tripped += 1

    # Auth failures (class-level)
    auth_active = 0
    try:
        from ...core.circuit_breaker import CircuitBreaker
        with CircuitBreaker._AUTH_FAILURE_LOCK:
            deadlines = tuple(CircuitBreaker._AUTH_FAILURE_UNTIL.values())
        counted = sum(1 for until in deadlines if now < until)
        auth_active = counted
    except (AttributeError, ImportError, RuntimeError, TypeError):
        _level = warn_level(_warned_collectors, 'auth_failures')
        logger.log(
            _level,
            "pipe_dashboard: cannot read active auth failures; the dashboard will "
            "report 0 for them",
            exc_info=True,
        )

    return {
        "tracked_users": tracked,
        "users_with_failures": with_failures,
        "tripped_users": tripped,
        "threshold": threshold,
        "window_s": round(window, 1),
        "tool_tracked": tool_tracked,
        "tool_with_failures": tool_with_failures,
        "tool_tripped": tool_tripped,
        "auth_failures_active": auth_active,
    }


def collect_sessions(pipe: Any) -> dict[str, int]:
    """Read the true top-level in-flight call gauge."""
    return {"in_flight": _safe_int(getattr(pipe, "_active_pipes_calls", 0))}
