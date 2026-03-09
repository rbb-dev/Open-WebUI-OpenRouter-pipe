"""Shared per-process stats collectors for pipe_stats.

Provides a single source of truth for reading concurrency, queue,
rate-limit, and session metrics from a Pipe instance.  Both
``runtime_stats.collect_fast_stats`` (single-worker SSE) and
``stats_publisher._collect_worker_payload`` (multi-worker Redis)
delegate to these functions.
"""

from __future__ import annotations

import time
from typing import Any


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
        "active_requests": max(0, sem_limit - sem._value) if sem else 0,
        "max_requests": sem_limit,
        "active_tools": max(0, tool_limit - tool_sem._value) if tool_sem else 0,
        "max_tools": tool_limit,
    }


def collect_queues(pipe: Any) -> dict[str, int]:
    """Read queue sizes from the pipe."""
    rq = getattr(pipe, "_request_queue", None)
    lq = getattr(pipe, "_log_queue", None)
    slm = getattr(pipe, "_session_log_manager", None)
    archive_q = getattr(slm, "_queue", None) if slm else None
    return {
        "requests": rq.qsize() if rq else 0,
        "requests_max": getattr(pipe, "_QUEUE_MAXSIZE", 1000),
        "logs": lq.qsize() if lq else 0,
        "archive": archive_q.qsize() if archive_q else 0,
    }


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
    for user_tools in tool_breakers.values():
        for dq in user_tools.values():
            recent = sum(1 for ts in dq if ts > cutoff) if cutoff else len(dq)
            tool_tracked += 1
            if recent >= threshold:
                tool_tripped += 1

    # Auth failures (class-level)
    auth_active = 0
    try:
        from ...core.circuit_breaker import CircuitBreaker
        with CircuitBreaker._AUTH_FAILURE_LOCK:
            for until in CircuitBreaker._AUTH_FAILURE_UNTIL.values():
                if now < until:
                    auth_active += 1
    except Exception:
        pass  # Only lose auth failure count

    return {
        "tracked_users": tracked,
        "users_with_failures": with_failures,
        "tripped_users": tripped,
        "threshold": threshold,
        "window_s": round(window, 1),
        "tool_tracked": tool_tracked,
        "tool_tripped": tool_tripped,
        "auth_failures_active": auth_active,
    }


def collect_sessions() -> dict[str, int]:
    """Read active session count."""
    try:
        from ...core.logging_system import SessionLogger
        return {"active": len(SessionLogger.logs)}
    except Exception:
        return {"active": 0}
