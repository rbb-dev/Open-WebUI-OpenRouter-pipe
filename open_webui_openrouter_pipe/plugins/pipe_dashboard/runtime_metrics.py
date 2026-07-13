"""Tiered runtime metrics collection for the SSE dashboard.

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

from ._collectors import (
    PROCESS_START,
    _safe_int,
    collect_concurrency,
    collect_queues,
    collect_rate_limits,
    collect_sessions,
    collect_video_pool,
)
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


# ---------------------------------------------------------------------------
# Identity (once)
# ---------------------------------------------------------------------------

def collect_identity(pipe: Pipe, *, worker_count: int = 1) -> dict[str, Any]:
    """Static identity data — sent on the first SSE event only.

    *worker_count* is overridden by the SSE reader when Redis
    aggregation is active (number of live ``dashboard:worker:*`` keys).
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
    """Lightweight metrics collected every tick.

    Delegates to shared collectors in ``_collectors`` for concurrency,
    queues, rate limits, and sessions.
    """
    return {
        "uptime_s": round(time.monotonic() - PROCESS_START, 1),
        "pid": os.getpid(),
        "concurrency": collect_concurrency(pipe),
        "queues": collect_queues(pipe),
        "videos": collect_video_pool(pipe),
        "rate_limits": collect_rate_limits(pipe),
        "sessions": collect_sessions(pipe),
    }


# ---------------------------------------------------------------------------
# Medium tier (~16 s) — light reads, no DB queries
# ---------------------------------------------------------------------------

def collect_model_registry() -> dict[str, Any]:
    """Raw model-registry state shared by the dashboard and the health command.

    Classifies the unified catalog into text/image/video using the same
    feature rules the registry itself applies, so the type counts always
    sum to the loaded total.
    """
    from ...models.registry import OpenRouterModelRegistry as Reg

    specs = getattr(Reg, "_specs", {}) or {}
    text_n = image_n = video_n = 0
    for m in getattr(Reg, "_models", []) or []:
        spec = (specs.get(m.get("norm_id")) or {}) if isinstance(m, dict) else {}
        feats = set(spec.get("features") or ())
        out_mods = (spec.get("architecture") or {}).get("output_modalities") or []
        if "video_generation" in feats:
            video_n += 1
        elif "image_output" in feats and "text" not in out_mods:
            image_n += 1
        else:
            text_n += 1
    loaded = text_n + image_n + video_n
    failures = getattr(Reg, "_consecutive_failures", 0) or 0
    last_fetch = getattr(Reg, "_last_fetch", 0.0) or 0.0
    last_error = getattr(Reg, "_last_error", None)
    zdr = getattr(Reg, "_zdr_model_ids", None)

    if loaded == 0 and failures == 0 and not last_fetch:
        status = "pending"
    elif failures >= 3 or loaded == 0:
        status = "failing"
    elif failures >= 1:
        status = "degraded"
    else:
        status = "healthy"

    return {
        "loaded": loaded,
        "text": text_n,
        "image": image_n,
        "video": video_n,
        "zdr": len(zdr) if zdr is not None else None,
        "specs_cached": len(specs),
        "failures": failures,
        "last_fetch": last_fetch,
        "video_fetch": getattr(Reg, "_last_video_fetch", 0.0) or 0.0,
        "image_fetch": getattr(Reg, "_last_image_fetch", 0.0) or 0.0,
        "video_attempt": getattr(Reg, "_last_video_attempt", 0.0) or 0.0,
        "image_attempt": getattr(Reg, "_last_image_attempt", 0.0) or 0.0,
        "last_error": str(last_error) if last_error else None,
        "last_error_time": getattr(Reg, "_last_error_time", 0.0) or 0.0,
        "status": status,
    }


def collect_medium_stats(pipe: Pipe) -> dict[str, Any]:
    """Model catalog health + system health indicators."""
    stats: dict[str, Any] = {}

    # ── Models ──
    try:
        reg = collect_model_registry()
        last_error = reg["last_error"]
        stats["models"] = {
            "loaded": reg["loaded"],
            "text": reg["text"],
            "image": reg["image"],
            "video": reg["video"],
            "zdr": reg["zdr"],
            "specs_cached": reg["specs_cached"],
            "last_fetch_ago": format_ago(reg["last_fetch"]) if reg["last_fetch"] else "never",
            "video_fetch_ago": format_ago(reg["video_fetch"]) if reg["video_fetch"] else None,
            "image_fetch_ago": format_ago(reg["image_fetch"]) if reg["image_fetch"] else None,
            "video_attempt_ago": format_ago(reg["video_attempt"]) if reg["video_attempt"] and reg["video_attempt"] > reg["video_fetch"] else None,
            "image_attempt_ago": format_ago(reg["image_attempt"]) if reg["image_attempt"] and reg["image_attempt"] > reg["image_fetch"] else None,
            "failures": reg["failures"],
            "last_error": (last_error[:80] + "...") if last_error and len(last_error) > 80 else last_error,
            "last_error_ago": format_ago(reg["last_error_time"]) if reg["last_error_time"] else None,
            "status": reg["status"],
        }
    except Exception:
        stats["models"] = {
            "loaded": 0, "text": 0, "image": 0, "video": 0, "zdr": None,
            "specs_cached": 0, "last_fetch_ago": "never",
            "video_fetch_ago": None, "image_fetch_ago": None,
            "video_attempt_ago": None, "image_attempt_ago": None,
            "failures": 0, "last_error": None, "last_error_ago": None,
            "status": "unknown",
        }

    # ── System Health ──
    valves = getattr(pipe, "valves", None)
    slm = getattr(pipe, "_session_log_manager", None)
    worker = getattr(slm, "_worker_thread", None) if slm else None
    http = getattr(pipe, "_http_session", None)

    logging_enabled = getattr(valves, "SESSION_LOG_STORE_ENABLED", False) if valves else False
    if not logging_enabled:
        log_worker = "disabled"
    elif worker is None:
        log_worker = "idle"
    elif worker.is_alive():
        log_worker = "active"
    else:
        log_worker = "stopped"

    log_buffers = 0
    log_events = 0
    try:
        from ...core.logging_system import SessionLogger

        log_buffers = len(SessionLogger.logs)
        log_events = sum(len(dq) for dq in SessionLogger.logs.values())
    except Exception:
        pass

    stats["health"] = {
        "initialized": getattr(pipe, "_initialized", False),
        "startup_complete": getattr(pipe, "_startup_checks_complete", False),
        "warmup_failed": getattr(pipe, "_warmup_failed", False),
        "http_session": "active" if (http and not http.closed) else "closed" if http else "none",
        "logging_enabled": logging_enabled,
        "log_worker": log_worker,
        "log_buffers": log_buffers,
        "log_events_buffered": log_events,
        "redis_enabled": getattr(pipe, "_redis_enabled", False),
        "redis_connected": getattr(pipe, "_redis_client", None) is not None,
    }

    stats["db"] = _collect_db_stats(pipe)
    stats["system"] = collect_system_resources()

    return stats


def collect_system_resources() -> dict[str, Any]:
    """Host CPU/memory + free space on the data-dir volume (psutil-optional)."""
    out: dict[str, Any] = {}
    try:
        import psutil

        cpu_raw: Any = psutil.cpu_percent(interval=None)
        if isinstance(cpu_raw, (int, float)):
            out["cpu_pct"] = float(cpu_raw)
        memory = psutil.virtual_memory()
        out["mem_used_pct"] = float(memory.percent)
        out["mem_total"] = int(memory.total)
        out["cores"] = int(psutil.cpu_count() or 0)
    except Exception:
        pass
    try:
        out["load1"] = float(os.getloadavg()[0])
        out.setdefault("cores", int(os.cpu_count() or 0))
    except Exception:
        pass
    try:
        import shutil

        try:
            from open_webui.env import DATA_DIR

            data_path = str(DATA_DIR)
        except Exception:
            data_path = os.getcwd()
        usage = shutil.disk_usage(data_path)
        out["disk_total"] = int(usage.total)
        out["disk_free"] = int(usage.free)
        out["disk_path"] = data_path
    except Exception:
        pass
    return out


def _collect_db_stats(pipe: Pipe) -> dict[str, Any]:
    """Artifact-store DB circuit breaker + write-pool backlog."""
    store = getattr(pipe, "_artifact_store", None)
    breakers = tripped = pending = workers = 0
    if store is not None:
        try:
            records = getattr(store, "_db_breakers", {}) or {}
            threshold = int(getattr(store, "_breaker_threshold", 0) or 0)
            window = float(getattr(store, "_breaker_window_seconds", 0.0) or 0.0)
            cutoff = time.time() - window if window else 0
            breakers = len(records)
            for dq in records.values():
                recent = sum(1 for ts in dq if ts > cutoff) if cutoff else len(dq)
                if threshold and recent >= threshold:
                    tripped += 1
        except Exception:
            breakers = tripped = 0
        try:
            executor = getattr(store, "_db_executor", None)
            if executor is not None:
                pending = _safe_int(executor._work_queue.qsize())
                workers = _safe_int(getattr(executor, "_max_workers", 0))
        except Exception:
            pending = workers = 0
    return {
        "breakers_tracked": breakers,
        "breakers_tripped": tripped,
        "pool_pending": pending,
        "pool_workers": workers,
    }


# ---------------------------------------------------------------------------
# Slow tier (~60 s) — DB queries, file I/O
# ---------------------------------------------------------------------------

def collect_slow_stats(pipe: Pipe) -> dict[str, Any]:
    """Heavy data: storage DB queries, config, plugins."""
    stats: dict[str, Any] = {}

    # ── Storage ──
    store = getattr(pipe, "_artifact_store", None)
    if store is not None:
        ensure_error: str | None = None
        sf = getattr(store, "_session_factory", None)
        model = getattr(store, "_item_model", None)
        if sf is None or model is None:
            try:
                store._ensure_artifact_store(getattr(pipe, "valves", None), getattr(pipe, "id", "") or "")
            except Exception as exc:
                ensure_error = type(exc).__name__
            sf = getattr(store, "_session_factory", None)
            model = getattr(store, "_item_model", None)
        db_connected = sf is not None and model is not None

        enc_key_set = bool(getattr(store, "_encryption_key", "") or "")
        enc_all = getattr(store, "_encrypt_all", False)
        comp = getattr(store, "_compression_enabled", False)

        storage: dict[str, Any] = {
            "connected": db_connected,
            "state": "connected" if db_connected else "unavailable",
            "error": ensure_error,
            "as_of": int(time.time()),
            "table": getattr(store, "_artifact_table_name", None),
            "total_items": "-",
            "total_size": "-",
            "encrypted_count": "-",
            "encryption_mode": "All items" if (enc_key_set and enc_all) else ("Sensitive only" if enc_key_set else "Disabled"),
            "compression_mode": "LZ4" if comp else "Disabled",
            "compress_min_bytes": format_number(getattr(store, "_compression_min_bytes", 0)),
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
                    total_count = 0
                    try:
                        total_count = session.query(func.count(model.id)).scalar() or 0
                        storage["total_items"] = format_number(total_count)
                    except Exception:
                        pass

                    payload_col = getattr(model, "payload", None)
                    if payload_col is not None:
                        try:
                            total_size = session.query(
                                func.sum(func.length(cast(payload_col, String)))
                            ).scalar() or 0
                            storage["total_size"] = format_bytes(total_size)
                        except Exception:
                            pass

                    enc_col = getattr(model, "is_encrypted", None)
                    if enc_col is not None:
                        try:
                            enc_count = session.query(func.count()).filter(enc_col.is_(True)).scalar() or 0
                            pct = enc_count / total_count * 100 if total_count > 0 else 0
                            storage["encrypted_count"] = f"{format_number(enc_count)} ({pct:.0f}%)"
                        except Exception:
                            pass

                    type_col = getattr(model, "item_type", None)
                    created_col = getattr(model, "created_at", None)
                    if type_col is not None and payload_col is not None and created_col is not None:
                        try:
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
                        except Exception:
                            pass

                    model_col = getattr(model, "model_id", None)
                    chat_col = getattr(model, "chat_id", None)
                    if model_col is not None and payload_col is not None and created_col is not None:
                        try:
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
                            pass

            except Exception as exc:
                storage["state"] = "degraded"
                storage["error"] = type(exc).__name__

        stats["storage"] = storage
    else:
        stats["storage"] = {"connected": False, "state": "unavailable", "error": "no store"}

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
    plugins: list[dict[str, str]] = []
    try:
        from ..registry import PluginRegistry

        for cls in list(getattr(PluginRegistry, "_plugin_classes", []) or []):
            plugins.append({
                "name": getattr(cls, "plugin_name", "?"),
                "id": getattr(cls, "plugin_id", "?"),
                "version": getattr(cls, "plugin_version", "?"),
            })
    except Exception:
        plugins = []
    if not plugins:
        pr = getattr(pipe, "_plugin_registry", None)
        if pr is not None:
            plugins = [
                {
                    "name": getattr(p, "plugin_name", "?"),
                    "id": getattr(p, "plugin_id", "?"),
                    "version": getattr(p, "plugin_version", "?"),
                }
                for p in getattr(pr, "_plugins", [])
            ]
    stats["plugins"] = plugins

    return stats
