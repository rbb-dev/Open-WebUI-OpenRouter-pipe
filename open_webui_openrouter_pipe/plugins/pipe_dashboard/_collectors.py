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

logger = logging.getLogger(__name__)

PROCESS_START = time.monotonic()

_warned_collectors: set[str] = set()


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (OverflowError, TypeError, ValueError):
        if "safe_int" not in _warned_collectors:
            _warned_collectors.add("safe_int")
            logger.warning(
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
        if "waiter_count" not in _warned_collectors:
            _warned_collectors.add("waiter_count")
            logger.warning(
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
        return max(0, limit - int(sem._value))
    except (AttributeError, TypeError, ValueError):
        if "semaphore_active" not in _warned_collectors:
            _warned_collectors.add("semaphore_active")
            logger.warning(
                "pipe_dashboard: cannot read semaphore usage; the dashboard will "
                "report 0 active for it",
                exc_info=True,
            )
        return 0


def collect_concurrency(pipe: Any) -> dict[str, int]:
    """Read concurrency semaphore state from the pipe."""
    sem = getattr(pipe, "_global_semaphore", None)
    sem_limit = getattr(pipe, "_semaphore_limit", 0) or 0
    tool_sem = getattr(pipe, "_tool_global_semaphore", None)
    tool_limit = getattr(pipe, "_tool_global_limit", 0) or 0
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
        "waiting": _waiter_count(getattr(pipe, "_global_semaphore", None)),
        "tool_waiting": _waiter_count(getattr(pipe, "_tool_global_semaphore", None)),
        "logs": lq.qsize() if lq else 0,
        "logs_max": _safe_int(getattr(lq, "maxsize", 0)) if lq else 0,
        "archive": archive_q.qsize() if archive_q else 0,
        "archive_max": _safe_int(getattr(archive_q, "maxsize", 0)) if archive_q else 0,
    }


def collect_video_pool(pipe: Any) -> dict[str, int]:
    """Read the video-generation concurrency pool (limit + live usage)."""
    limit = _safe_int(getattr(pipe, "_video_global_limit", 0))
    sem = getattr(pipe, "_video_global_semaphore", None)
    if sem is not None and limit:
        active = _semaphore_active(sem, limit)
    else:
        try:
            active = len(getattr(pipe, "_video_active_tasks", {}) or {})
        except TypeError:
            if "video_active" not in _warned_collectors:
                _warned_collectors.add("video_active")
                logger.warning(
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
    tracked = len(records)
    with_failures = 0
    tripped = 0
    for dq in records.values():
        recent = sum(1 for ts in dq if ts > cutoff) if cutoff else len(dq)
        if recent > 0:
            with_failures += 1
        if recent >= threshold:
            tripped += 1

    # Tool breakers
    tool_breakers = getattr(cb, "_tool_breakers", {})
    tool_tracked = 0
    tool_tripped = 0
    tool_with_failures = 0
    for user_tools in tool_breakers.values():
        for dq in user_tools.values():
            recent = sum(1 for ts in dq if ts > cutoff) if cutoff else len(dq)
            tool_tracked += 1
            if recent > 0:
                tool_with_failures += 1
            if recent >= threshold:
                tool_tripped += 1

    # Auth failures (class-level)
    auth_active = 0
    try:
        from ...core.circuit_breaker import CircuitBreaker
        counted = 0
        with CircuitBreaker._AUTH_FAILURE_LOCK:
            for until in CircuitBreaker._AUTH_FAILURE_UNTIL.values():
                if now < until:
                    counted += 1
        auth_active = counted
    except (AttributeError, ImportError, RuntimeError, TypeError):
        if "auth_failures" not in _warned_collectors:
            _warned_collectors.add("auth_failures")
            logger.warning(
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
