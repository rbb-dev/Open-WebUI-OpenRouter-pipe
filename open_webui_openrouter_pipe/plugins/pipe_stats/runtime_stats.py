"""Tiered runtime metrics collection for the SSE stats dashboard.

Provides four collector functions at different frequencies:

- **identity** (once): version, pipe_id, worker_count
- **fast** (every 2 s): concurrency, queues, rate limits, sessions, uptime
- **medium** (~16 s): model catalog health, system health
- **slow** (~60 s): storage DB queries, config, plugins
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from .formatters import (
    build_model_name_map,
    format_ago,
    format_bytes,
    format_datetime,
    format_number,
    humanize_type,
    resolve_model_name,
)

if TYPE_CHECKING:
    from ...pipe import Pipe

# Captured once at import time — closest proxy for "process start".
_PS_RT_PROCESS_START = time.monotonic()


# ---------------------------------------------------------------------------
# Identity (once)
# ---------------------------------------------------------------------------

def collect_identity(pipe: Pipe, *, worker_count: int = 1) -> dict[str, Any]:
    """Static identity data — sent on the first SSE event only.

    *worker_count* is overridden by the SSE reader when Redis
    aggregation is active (number of live ``stats:worker:*`` keys).
    """
    try:
        from open_webui_openrouter_pipe import __version__
        version = __version__
    except Exception:
        version = "unknown"

    return {
        "identity": {
            "version": version,
            "pipe_id": getattr(pipe, "id", "unknown"),
            "worker_count": worker_count,
        },
    }


# ---------------------------------------------------------------------------
# Fast tier (every 2 s) — no DB, no heavy imports
# ---------------------------------------------------------------------------

def collect_fast_stats(pipe: Pipe) -> dict[str, Any]:
    """Lightweight metrics collected every tick."""
    stats: dict[str, Any] = {
        "uptime_s": round(time.monotonic() - _PS_RT_PROCESS_START, 1),
        "pid": os.getpid(),
    }

    # ── Concurrency ──
    sem = getattr(pipe, "_global_semaphore", None)
    sem_limit = getattr(pipe, "_semaphore_limit", 0) or 0
    tool_sem = getattr(pipe, "_tool_global_semaphore", None)
    tool_limit = getattr(pipe, "_tool_global_limit", 0) or 0
    # Fall back to valve config when semaphores aren't materialized yet
    # (lazy init on first request — avoids misleading "0 / 0" display)
    if not sem_limit:
        valves = getattr(pipe, "valves", None)
        sem_limit = getattr(valves, "MAX_CONCURRENT_REQUESTS", 0) or 0
    if not tool_limit:
        valves = getattr(pipe, "valves", None)
        tool_limit = getattr(valves, "MAX_PARALLEL_TOOLS_GLOBAL", 0) or 0
    stats["concurrency"] = {
        "active_requests": max(0, sem_limit - sem._value) if sem else 0,
        "max_requests": sem_limit,
        "active_tools": max(0, tool_limit - tool_sem._value) if tool_sem else 0,
        "max_tools": tool_limit,
    }

    # ── Queues ──
    rq = getattr(pipe, "_request_queue", None)
    lq = getattr(pipe, "_log_queue", None)
    slm = getattr(pipe, "_session_log_manager", None)
    archive_q = getattr(slm, "_queue", None) if slm else None
    stats["queues"] = {
        "requests": rq.qsize() if rq else 0,
        "requests_max": getattr(pipe, "_QUEUE_MAXSIZE", 1000),
        "logs": lq.qsize() if lq else 0,
        "archive": archive_q.qsize() if archive_q else 0,
    }

    # ── Rate Limits (summary — no per-user data) ──
    cb = getattr(pipe, "_circuit_breaker", None)
    if cb:
        threshold = getattr(cb, "_threshold", 0)
        window = getattr(cb, "_window_seconds", 0.0)
        now = time.time()
        cutoff = now - window if window else 0

        # Request breakers
        records = getattr(cb, "_breaker_records", {})
        tracked = 0
        with_failures = 0
        tripped = 0
        for dq in records.values():
            recent = sum(1 for ts in dq if ts > cutoff) if cutoff else len(dq)
            if recent > 0:
                tracked += 1
                with_failures += 1
            else:
                tracked += 1
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
        from ...core.circuit_breaker import CircuitBreaker
        auth_active = 0
        with CircuitBreaker._AUTH_FAILURE_LOCK:
            for until in CircuitBreaker._AUTH_FAILURE_UNTIL.values():
                if now < until:
                    auth_active += 1

        stats["rate_limits"] = {
            "tracked_users": tracked,
            "users_with_failures": with_failures,
            "tripped_users": tripped,
            "threshold": threshold,
            "window_s": round(window, 1),
            "tool_tracked": tool_tracked,
            "tool_tripped": tool_tripped,
            "auth_failures_active": auth_active,
        }
    else:
        stats["rate_limits"] = {
            "tracked_users": 0,
            "users_with_failures": 0,
            "tripped_users": 0,
            "threshold": 0,
            "window_s": 0,
            "tool_tracked": 0,
            "tool_tripped": 0,
            "auth_failures_active": 0,
        }

    # ── Sessions ──
    try:
        from ...core.logging_system import SessionLogger
        stats["sessions"] = {"active": len(SessionLogger.logs)}
    except Exception:
        stats["sessions"] = {"active": 0}

    return stats


# ---------------------------------------------------------------------------
# Medium tier (~16 s) — light reads, no DB queries
# ---------------------------------------------------------------------------

def collect_medium_stats(pipe: Pipe) -> dict[str, Any]:
    """Model catalog health + system health indicators."""
    stats: dict[str, Any] = {}

    # ── Models ──
    try:
        from ...models.registry import OpenRouterModelRegistry as Reg
        loaded = len(getattr(Reg, "_models", []) or [])
        failures = getattr(Reg, "_consecutive_failures", 0) or 0
        last_fetch = getattr(Reg, "_last_fetch", 0.0) or 0.0
        last_error = getattr(Reg, "_last_error", None)

        if loaded == 0 and failures == 0 and not last_fetch:
            status = "pending"
        elif failures >= 3 or loaded == 0:
            status = "failing"
        elif failures >= 1:
            status = "degraded"
        else:
            status = "healthy"

        stats["models"] = {
            "loaded": loaded,
            "specs_cached": len(getattr(Reg, "_specs", {}) or {}),
            "last_fetch_ago": format_ago(last_fetch) if last_fetch else "never",
            "failures": failures,
            "last_error": (str(last_error)[:80] + "...") if last_error and len(str(last_error)) > 80 else (str(last_error) if last_error else None),
            "status": status,
        }
    except Exception:
        stats["models"] = {
            "loaded": 0, "specs_cached": 0,
            "last_fetch_ago": "never", "failures": 0,
            "last_error": None, "status": "unknown",
        }

    # ── System Health ──
    valves = getattr(pipe, "valves", None)
    slm = getattr(pipe, "_session_log_manager", None)
    worker = getattr(slm, "_worker_thread", None) if slm else None
    http = getattr(pipe, "_http_session", None)

    logging_enabled = getattr(valves, "SESSION_LOG_STORE_ENABLED", False) if valves else False

    stats["health"] = {
        "initialized": getattr(pipe, "_initialized", False),
        "startup_complete": getattr(pipe, "_startup_checks_complete", False),
        "warmup_failed": getattr(pipe, "_warmup_failed", False),
        "http_session": "active" if (http and not http.closed) else "closed" if http else "none",
        "logging_enabled": logging_enabled,
        "log_worker_alive": worker.is_alive() if worker else False,
        "log_retention_days": getattr(slm, "_retention_days", None) if slm else None,
        "redis_enabled": getattr(pipe, "_redis_enabled", False),
        "redis_connected": getattr(pipe, "_redis_client", None) is not None,
    }

    return stats


# ---------------------------------------------------------------------------
# Slow tier (~60 s) — DB queries, file I/O
# ---------------------------------------------------------------------------

def collect_slow_stats(pipe: Pipe) -> dict[str, Any]:
    """Heavy data: storage DB queries, config, plugins."""
    stats: dict[str, Any] = {}

    # ── Storage ──
    store = getattr(pipe, "_artifact_store", None)
    if store is not None:
        sf = getattr(store, "_session_factory", None)
        model = getattr(store, "_item_model", None)
        db_connected = sf is not None and model is not None

        enc_on = bool(getattr(store, "_encryption_key", ""))
        enc_all = getattr(store, "_encrypt_all", False)
        comp = getattr(store, "_compression_enabled", False)

        storage: dict[str, Any] = {
            "connected": db_connected,
            "table": getattr(store, "_artifact_table_name", None),
            "total_items": "-",
            "total_size": "-",
            "db_file_size": "-",
            "encrypted_count": "-",
            "encryption_mode": "All items" if enc_all else ("Sensitive only" if enc_on else "Disabled"),
            "compression_mode": "LZ4" if comp else "Disabled",
            "compress_min_bytes": format_number(getattr(store, "_compression_min_bytes", 0)),
            "redis_cache": getattr(store, "_redis_enabled", False),
            "by_type": [],
            "by_model": [],
        }

        if db_connected:
            assert sf is not None  # guaranteed by db_connected check
            assert model is not None  # guaranteed by db_connected check
            try:
                from ...storage.persistence import _db_session
                from sqlalchemy import String, func
                from sqlalchemy.sql.expression import cast

                name_map = build_model_name_map()

                with _db_session(sf) as session:
                    # Total count
                    total_count = session.query(func.count(model.id)).scalar() or 0
                    storage["total_items"] = format_number(total_count)

                    # Total payload size
                    payload_col = getattr(model, "payload", None)
                    if payload_col is not None:
                        total_size = session.query(
                            func.sum(func.length(cast(payload_col, String)))
                        ).scalar() or 0
                        storage["total_size"] = format_bytes(total_size)

                    # Encrypted count
                    enc_col = getattr(model, "is_encrypted", None)
                    if enc_col is not None:
                        enc_count = session.query(func.count()).filter(enc_col.is_(True)).scalar() or 0
                        pct = enc_count / total_count * 100 if total_count > 0 else 0
                        storage["encrypted_count"] = f"{format_number(enc_count)} ({pct:.0f}%)"

                    # DB file size (SQLite only)
                    engine = getattr(store, "_engine", None)
                    if engine is not None:
                        try:
                            url_str = str(engine.url)
                            if "sqlite" in url_str:
                                path = url_str.split("///", 1)[-1] if "///" in url_str else ""
                                if path and os.path.isfile(path):
                                    storage["db_file_size"] = format_bytes(os.path.getsize(path))
                        except Exception:
                            pass

                    # By-type breakdown
                    type_col = getattr(model, "item_type", None)
                    created_col = getattr(model, "created_at", None)
                    if type_col is not None and payload_col is not None and created_col is not None:
                        type_rows = (
                            session.query(
                                type_col,
                                func.count(model.id),
                                func.sum(func.length(cast(payload_col, String))),
                                func.min(created_col),
                                func.max(created_col),
                            )
                            .group_by(type_col)
                            .order_by(func.count(model.id).desc())
                            .all()
                        )
                        storage["by_type"] = [
                            {
                                "type": humanize_type(str(r[0] or "unknown")),
                                "count": format_number(int(r[1])),
                                "size": format_bytes(int(r[2] or 0)),
                                "oldest": format_datetime(r[3]),
                                "newest": format_datetime(r[4]),
                            }
                            for r in type_rows
                        ]

                    # By-model breakdown (top 15)
                    model_col = getattr(model, "model_id", None)
                    chat_col = getattr(model, "chat_id", None)
                    if model_col is not None and payload_col is not None and created_col is not None:
                        model_rows = (
                            session.query(
                                model_col,
                                func.count(model.id),
                                func.sum(func.length(cast(payload_col, String))),
                                func.count(func.distinct(chat_col)) if chat_col is not None else func.count(model.id),
                                func.min(created_col),
                                func.max(created_col),
                            )
                            .group_by(model_col)
                            .order_by(func.count(model.id).desc())
                            .limit(15)
                            .all()
                        )
                        storage["by_model"] = [
                            {
                                "name": resolve_model_name(str(r[0] or "unknown"), name_map),
                                "model_id": str(r[0] or "unknown"),
                                "count": format_number(int(r[1])),
                                "size": format_bytes(int(r[2] or 0)),
                                "chats": format_number(int(r[3])),
                                "oldest": format_datetime(r[4]),
                                "newest": format_datetime(r[5]),
                            }
                            for r in model_rows
                        ]

            except Exception:
                pass  # Keep defaults

        stats["storage"] = storage
    else:
        stats["storage"] = {"connected": False}

    # ── Configuration ──
    valves = getattr(pipe, "valves", None)
    if valves is not None:
        stats["config"] = {
            "endpoint": str(getattr(valves, "DEFAULT_LLM_ENDPOINT", "?")),
            "breaker": f"{getattr(valves, 'BREAKER_MAX_FAILURES', '?')} failures / {getattr(valves, 'BREAKER_WINDOW_SECONDS', '?')}s",
            "timing_log": bool(getattr(valves, "ENABLE_TIMING_LOG", False)),
            "artifact_cleanup": f"every {getattr(valves, 'ARTIFACT_CLEANUP_INTERVAL_HOURS', '?')}h, keep {getattr(valves, 'ARTIFACT_CLEANUP_DAYS', '?')}d",
            "log_retention": f"{getattr(valves, 'SESSION_LOG_RETENTION_DAYS', '?')} days",
            "redis_ttl": f"{getattr(valves, 'REDIS_CACHE_TTL_SECONDS', '?')}s",
            "stream_idle_flush": f"{getattr(valves, 'STREAMING_IDLE_FLUSH_MS', '?')}ms",
        }
    else:
        stats["config"] = {}

    # ── Plugins ──
    pr = getattr(pipe, "_plugin_registry", None)
    if pr is not None:
        plugins = getattr(pr, "_plugins", [])
        stats["plugins"] = [
            {
                "name": getattr(p, "plugin_name", "?"),
                "id": getattr(p, "plugin_id", "?"),
                "version": getattr(p, "plugin_version", "?"),
            }
            for p in plugins
        ]
    else:
        stats["plugins"] = []

    return stats
