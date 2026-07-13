"""Per-worker background stats task: Redis slice publishing + socket.io emits.

Each uvicorn worker runs one async task with three modes:

1. **Emitting** — this worker has local members in the ``pipe_dashboard_viewers``
   socket.io room. It renews the ``{ns}:dashboard:active`` flag, writes its own
   slice, reads all workers' slices from Redis, aggregates, merges the tiered
   collectors, and emits the payload to the room (local delivery only).
2. **Publishing** — no local viewers, but another worker set the active flag:
   write this worker's slice to ``{ns}:dashboard:worker:{pid}`` every
   ``_PD_PUBLISH_INTERVAL`` seconds so the emitting worker can aggregate it.
3. **Idle** — no viewers anywhere: one Redis EXISTS per ``_PD_POLL_INTERVAL``,
   woken instantly via the ``{ns}:dashboard:wake`` pub/sub channel.

Without Redis a single worker emits directly from its local collectors.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from typing import Any

from ._collectors import (
    PROCESS_START,
    collect_concurrency,
    collect_queues,
    collect_rate_limits,
    collect_sessions,
    collect_video_pool,
)
from .dashboard_socket import (
    consume_resync,
    emit_dashboard,
    local_viewer_sids,
    reauthorize_local_viewers,
    read_config_rev,
)
from .runtime_metrics import (
    collect_fast_stats,
    collect_identity,
    collect_medium_stats,
    collect_slow_stats,
)

_pd_pub_log = logging.getLogger(__name__)

_pd_snapshot_getter: Any = None


def set_snapshot_getter(getter: Any) -> None:
    global _pd_snapshot_getter
    _pd_snapshot_getter = getter


def _snapshot_safe() -> tuple[list[dict[str, Any]], dict[str, float]]:
    getter = _pd_snapshot_getter
    if getter is None:
        return [], {}
    try:
        rows, tc = getter()
        return (rows if isinstance(rows, list) else []), (tc if isinstance(tc, dict) else {})
    except Exception:
        _pd_pub_log.debug("live snapshot failed", exc_info=True)
        return [], {}


def _fold_task_costs(rows: list[dict[str, Any]], task_costs: dict[str, float]) -> list[dict[str, Any]]:
    if task_costs:
        by_chat: dict[str, dict[str, Any]] = {}
        for r in rows:
            cid = r.get("chat_id")
            if cid and r.get("kind") != "task":
                cur = by_chat.get(cid)
                if cur is None or (r.get("started") or 0.0) > (cur.get("started") or 0.0):
                    by_chat[cid] = r
        for cid, cost in task_costs.items():
            parent = by_chat.get(cid)
            if parent is not None and cost:
                parent["cost"] = round((parent.get("cost") or 0.0) + cost, 6)
                parent["task_cost"] = round((parent.get("task_cost") or 0.0) + cost, 6)
    for r in rows:
        r.pop("chat_id", None)
    return rows


_PD_POLL_INTERVAL = 5.0
_PD_PUBLISH_INTERVAL = 2.0
_PD_KEY_TTL = 10
_PD_ACTIVE_FLAG_TTL = 30
_PD_SESSIONS_CAP = 300
_PD_SETTLE_DELAY = 1.0
_PD_MEDIUM_EVERY = 8
_PD_SLOW_EVERY = 30
_PD_SLOW_MIN_INTERVAL = 30.0
_PD_REAUTH_EVERY = 15


def _worker_health(pipe: Any) -> dict[str, int]:
    http = getattr(pipe, "_http_session", None)
    try:
        http_ok = 1 if (http is not None and not http.closed) else 0
    except Exception:
        http_ok = 0
    rss = 0
    try:
        import psutil

        rss = int(psutil.Process().memory_info().rss)
    except Exception:
        rss = 0
    return {
        "init": 1 if getattr(pipe, "_initialized", False) else 0,
        "wf": 1 if getattr(pipe, "_warmup_failed", False) else 0,
        "http": http_ok,
        "r": 1 if getattr(pipe, "_redis_client", None) is not None else 0,
        "rss": rss,
    }


def _collect_worker_payload(pipe: Any) -> dict[str, Any]:
    """Collect compact per-worker stats for Redis publishing.

    Only includes data that is *unique* to this process — shared data
    (models, storage, config, plugins) is collected locally by the
    emitting worker and doesn't need Redis sync.
    """
    c = collect_concurrency(pipe)
    q = collect_queues(pipe)
    rl = collect_rate_limits(pipe)
    s = collect_sessions(pipe)
    v = collect_video_pool(pipe)
    snap = _snapshot_safe()

    return {
        "pid": os.getpid(),
        "up": round(time.monotonic() - PROCESS_START, 1),
        "ls": int(time.time()),
        "c": {
            "ar": c["active_requests"],
            "mr": c["max_requests"],
            "at": c["active_tools"],
            "mt": c["max_tools"],
        },
        "q": {
            "rq": q["requests"],
            "rm": q["requests_max"],
            "w": q["waiting"],
            "tw": q["tool_waiting"],
            "lq": q["logs"],
            "lm": q["logs_max"],
            "aq": q["archive"],
            "am": q["archive_max"],
        },
        "rl": {
            "tu": rl["tracked_users"],
            "fu": rl["users_with_failures"],
            "tr": rl["tripped_users"],
            "th": rl["threshold"],
            "ws": rl["window_s"],
            "tt": rl["tool_tracked"],
            "tf": rl["tool_with_failures"],
            "tp": rl["tool_tripped"],
            "aa": rl["auth_failures_active"],
        },
        "v": {"a": v["active"], "m": v["max"]},
        "s": s["in_flight"],
        "h": _worker_health(pipe),
        "sl": snap[0],
        "tc": snap[1],
    }


_PD_CONCURRENCY_MAP = {
    "ar": "active_requests",
    "mr": "max_requests",
    "at": "active_tools",
    "mt": "max_tools",
}

_PD_QUEUE_MAP = {
    "rq": "requests",
    "rm": "requests_max",
    "w": "waiting",
    "tw": "tool_waiting",
    "lq": "logs",
    "lm": "logs_max",
    "aq": "archive",
    "am": "archive_max",
}

_PD_RATE_LIMIT_MAP = {
    "tu": "tracked_users",
    "fu": "users_with_failures",
    "tr": "tripped_users",
    "th": "threshold",
    "ws": "window_s",
    "tt": "tool_tracked",
    "tf": "tool_with_failures",
    "tp": "tool_tripped",
    "aa": "auth_failures_active",
}

_PD_VIDEO_MAP = {
    "a": "active",
    "m": "max",
}


def expand_worker_payload(compact: dict[str, Any]) -> dict[str, Any]:
    """Expand a compact Redis payload into the full dashboard field names."""
    health = compact.get("h")
    return {
        "pid": compact.get("pid", 0),
        "uptime_s": compact.get("up", 0),
        "last_seen": compact.get("ls", 0),
        "concurrency": {_PD_CONCURRENCY_MAP[k]: v for k, v in compact.get("c", {}).items() if k in _PD_CONCURRENCY_MAP},
        "queues": {_PD_QUEUE_MAP[k]: v for k, v in compact.get("q", {}).items() if k in _PD_QUEUE_MAP},
        "videos": {_PD_VIDEO_MAP[k]: v for k, v in compact.get("v", {}).items() if k in _PD_VIDEO_MAP},
        "rate_limits": {_PD_RATE_LIMIT_MAP[k]: v for k, v in compact.get("rl", {}).items() if k in _PD_RATE_LIMIT_MAP},
        "sessions": {"in_flight": compact.get("s", 0)},
        "sessions_live": compact.get("sl") if isinstance(compact.get("sl"), list) else [],
        "task_costs": compact.get("tc") if isinstance(compact.get("tc"), dict) else {},
        "worker_health": health if isinstance(health, dict) else {},
    }


def aggregate_worker_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate expanded payloads from multiple workers into a single payload.

    Numeric fields (active values AND per-process capacity limits) are summed
    across workers; workers whose slice lacks a section are skipped rather
    than counted as zero, and breaker config takes the max so a slice from an
    older schema can't zero the cluster values.
    """
    if not payloads:
        return {}

    conc = [c for p in payloads if isinstance((c := p.get("concurrency")), dict)]
    qs = [q for p in payloads if isinstance((q := p.get("queues")), dict)]
    rls = [r for p in payloads if isinstance((r := p.get("rate_limits")), dict)]
    vids = [v for p in payloads if isinstance((v := p.get("videos")), dict)]

    agg_concurrency = {
        "active_requests": sum(c.get("active_requests", 0) for c in conc),
        "max_requests": sum(c.get("max_requests", 0) for c in conc),
        "active_tools": sum(c.get("active_tools", 0) for c in conc),
        "max_tools": sum(c.get("max_tools", 0) for c in conc),
    }

    agg_queues = {
        "requests": sum(q.get("requests", 0) for q in qs),
        "requests_max": sum(q.get("requests_max", 0) for q in qs),
        "waiting": sum(q.get("waiting", 0) for q in qs),
        "tool_waiting": sum(q.get("tool_waiting", 0) for q in qs),
        "logs": sum(q.get("logs", 0) for q in qs),
        "logs_max": sum(q.get("logs_max", 0) for q in qs),
        "archive": sum(q.get("archive", 0) for q in qs),
        "archive_max": sum(q.get("archive_max", 0) for q in qs),
    }

    agg_rate_limits = {
        "tracked_users": sum(r.get("tracked_users", 0) for r in rls),
        "users_with_failures": sum(r.get("users_with_failures", 0) for r in rls),
        "tripped_users": sum(r.get("tripped_users", 0) for r in rls),
        "threshold": max((r.get("threshold", 0) for r in rls), default=0),
        "window_s": max((r.get("window_s", 0) for r in rls), default=0),
        "tool_tracked": sum(r.get("tool_tracked", 0) for r in rls),
        "tool_with_failures": sum(r.get("tool_with_failures", 0) for r in rls),
        "tool_tripped": sum(r.get("tool_tripped", 0) for r in rls),
        "auth_failures_active": sum(r.get("auth_failures_active", 0) for r in rls),
    }

    agg_videos = {
        "active": sum(v.get("active", 0) for v in vids),
        "max": sum(v.get("max", 0) for v in vids),
    }

    agg_sessions = {
        "in_flight": sum(p.get("sessions", {}).get("in_flight", 0) for p in payloads),
    }

    sessions_live: list[dict[str, Any]] = []
    for p in payloads:
        rows = p.get("sessions_live")
        if isinstance(rows, list):
            sessions_live.extend(row for row in rows if isinstance(row, dict))
    task_costs: dict[str, float] = {}
    for p in payloads:
        tc = p.get("task_costs")
        if isinstance(tc, dict):
            for cid, c in tc.items():
                task_costs[cid] = task_costs.get(cid, 0.0) + (c or 0.0)
    sessions_live = _fold_task_costs(sessions_live, task_costs)
    sessions_live.sort(key=lambda row: (1 if row.get("done") else 0, -(row.get("started") or 0.0)))
    sessions_live = sessions_live[:_PD_SESSIONS_CAP]

    now = time.time()
    workers = []
    for p in payloads:
        last_seen = p.get("last_seen", 0) or 0
        workers.append({
            "pid": p.get("pid", 0),
            "uptime_s": p.get("uptime_s", 0),
            "last_seen_age": round(max(0.0, now - last_seen), 1) if last_seen else None,
            "active_requests": p.get("concurrency", {}).get("active_requests", 0) if isinstance(p.get("concurrency"), dict) else 0,
            "health": p.get("worker_health", {}) if isinstance(p.get("worker_health"), dict) else {},
        })

    max_uptime = max((p.get("uptime_s", 0) for p in payloads), default=0)

    workers_rss = 0
    for p in payloads:
        health = p.get("worker_health")
        if isinstance(health, dict):
            try:
                workers_rss += int(health.get("rss") or 0)
            except Exception:
                pass

    return {
        "uptime_s": max_uptime,
        "workers_rss": workers_rss,
        "pid": payloads[0].get("pid", 0),
        "concurrency": agg_concurrency,
        "queues": agg_queues,
        "videos": agg_videos,
        "rate_limits": agg_rate_limits,
        "sessions": agg_sessions,
        "sessions_live": sessions_live,
        "workers": sorted(workers, key=lambda w: w["pid"]),
    }


async def _read_redis_workers(client: Any, namespace: str) -> list[dict[str, Any]] | None:
    """Read and expand all live worker slices from Redis.

    Returns ``None`` on a read error (so the caller can substitute the last
    known worker set instead of collapsing to single-worker), and ``[]`` only
    when the scan legitimately finds no slices.
    """
    pattern = f"{namespace}:dashboard:worker:*"
    try:
        keys = []
        async for key in client.scan_iter(match=pattern, count=50):
            keys.append(key)
        if not keys:
            return []
        values = await client.mget(*keys)
        payloads = []
        for raw in values:
            if raw is None:
                continue
            try:
                compact = json.loads(raw)
                payloads.append(expand_worker_payload(compact))
            except (json.JSONDecodeError, TypeError):
                continue
        return payloads
    except Exception:
        _pd_pub_log.debug("Redis worker read failed", exc_info=True)
        return None


async def _set_active_flag(client: Any, namespace: str, *, wake: bool) -> None:
    try:
        await client.set(f"{namespace}:dashboard:active", "1", ex=_PD_ACTIVE_FLAG_TTL)
    except Exception:
        _pd_pub_log.debug("Failed to set stats active flag", exc_info=True)
    if wake:
        try:
            await client.publish(f"{namespace}:dashboard:wake", "wake")
        except Exception:
            pass


async def _write_own_slice(client: Any, worker_key: str, pipe: Any) -> None:
    try:
        payload = _collect_worker_payload(pipe)
        await client.set(
            worker_key,
            json.dumps(payload, separators=(",", ":")),
            ex=_PD_KEY_TTL,
        )
    except Exception:
        _pd_pub_log.debug("Dashboard publish failed (pid=%d)", os.getpid(), exc_info=True)


async def _redis_alive(pipe: Any) -> bool:
    """Liveness probe: a bounded real ping, not client-object existence."""
    client = getattr(pipe, "_redis_client", None)
    if client is None:
        return False
    try:
        result = client.ping()
        if inspect.isawaitable(result):
            result = await asyncio.wait_for(result, timeout=0.25)
        return bool(result)
    except Exception:
        return False


def _collect_fast_safe(pipe: Any) -> dict[str, Any]:
    try:
        return collect_fast_stats(pipe)
    except Exception:
        _pd_pub_log.debug("Fast stats collect error", exc_info=True)
        return {}


async def _build_emit_payload(
    pipe: Any,
    client: Any,
    namespace: str,
    worker_key: str,
    tick: int,
    slow_state: dict[str, Any],
    agg_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one dashboard emit: fast aggregate + tiered collectors."""
    pid = os.getpid()
    payload: dict[str, Any] = {"tick": tick}
    worker_count = 1
    agg_state = agg_state if agg_state is not None else {}

    if client is not None:
        await _write_own_slice(client, worker_key, pipe)
        worker_payloads = await _read_redis_workers(client, namespace)
        degraded = False
        if worker_payloads is None:
            misses = agg_state.get("misses", 0) + 1
            agg_state["misses"] = misses
            cached = agg_state.get("workers") or []
            if cached and misses <= 2:
                worker_payloads = list(cached)
                degraded = True
            else:
                worker_payloads = []
        else:
            agg_state["misses"] = 0
        local_pids = {p.get("pid", 0) for p in worker_payloads}
        if pid not in local_pids:
            try:
                worker_payloads.append(expand_worker_payload(_collect_worker_payload(pipe)))
            except Exception:
                _pd_pub_log.debug("Local worker payload collect error", exc_info=True)
        if not degraded:
            agg_state["workers"] = list(worker_payloads)
        if worker_payloads:
            payload.update(aggregate_worker_payloads(worker_payloads))
            worker_count = len(worker_payloads)
        else:
            payload.update(_collect_fast_safe(pipe))
        if degraded:
            payload["degraded"] = True
    else:
        payload.update(_collect_fast_safe(pipe))
        _snap = _snapshot_safe()
        payload["sessions_live"] = _fold_task_costs(_snap[0], _snap[1])
        payload["workers_rss"] = _worker_health(pipe).get("rss", 0)
        payload["workers"] = [{
            "pid": pid,
            "uptime_s": payload.get("uptime_s", 0),
            "last_seen_age": 0.0,
            "active_requests": payload.get("concurrency", {}).get("active_requests", 0),
            "health": _worker_health(pipe),
        }]

    payload["worker_count"] = worker_count

    now = time.monotonic()
    medium_due = tick == 0 or now - slow_state.get("medium_sent_at", 0.0) >= _PD_MEDIUM_EVERY * _PD_PUBLISH_INTERVAL
    if medium_due:
        slow_state["medium_sent_at"] = now
        try:
            payload.update(collect_identity(pipe, worker_count=worker_count))
        except Exception:
            _pd_pub_log.debug("Identity collect error", exc_info=True)
        try:
            payload.update(collect_medium_stats(pipe))
        except Exception:
            _pd_pub_log.debug("Medium stats collect error", exc_info=True)
        health = payload.get("health")
        if isinstance(health, dict):
            health["redis_connected"] = await _redis_alive(pipe)
        slow_state["cfg_rev"] = await read_config_rev(getattr(pipe, "id", ""))

    slow_due = tick == 0 or now - slow_state.get("slow_sent_at", 0.0) >= _PD_SLOW_EVERY * _PD_PUBLISH_INTERVAL
    if slow_due:
        slow_state["slow_sent_at"] = now
        if not slow_state.get("cache") or now - slow_state.get("at", 0.0) >= _PD_SLOW_MIN_INTERVAL:
            try:
                slow_state["cache"] = collect_slow_stats(pipe)
                slow_state["at"] = now
            except Exception:
                _pd_pub_log.debug("Slow stats collect error", exc_info=True)
        cached = slow_state.get("cache")
        if cached:
            payload.update(cached)

    payload["cfgRev"] = slow_state.get("cfg_rev")
    return payload


async def run_dashboard_publisher(
    get_pipe: Any,
    get_redis: Any,
    namespace: str,
) -> None:
    """Long-running background task: slice publishing + viewer emits.

    Parameters
    ----------
    get_pipe:
        Callable returning the current pipe instance (or None).
    get_redis:
        Callable returning ``(redis_client, redis_enabled)`` tuple.
    namespace:
        Redis key namespace prefix (e.g. ``"openrouter"``).
    """
    pid = os.getpid()
    active_key = f"{namespace}:dashboard:active"
    worker_key = f"{namespace}:dashboard:worker:{pid}"
    wake_channel = f"{namespace}:dashboard:wake"

    await asyncio.sleep(2.0)

    _pd_pub_log.debug("Dashboard publisher started (pid=%d, ns=%s)", pid, namespace)

    pubsub = None
    tick = 0
    emitting = False
    slow_state: dict[str, Any] = {}
    agg_state: dict[str, Any] = {}

    try:
        while True:
            client, enabled = get_redis()
            redis_ok = bool(enabled) and client is not None
            pipe = get_pipe()

            if redis_ok and pubsub is None:
                try:
                    pubsub = client.pubsub()
                    await pubsub.subscribe(wake_channel)
                except Exception:
                    _pd_pub_log.debug("Pub/sub subscribe failed", exc_info=True)
                    pubsub = None

            if pipe is None:
                emitting = False
                await asyncio.sleep(_PD_POLL_INTERVAL)
                continue

            if local_viewer_sids():
                if consume_resync() or not emitting:
                    tick = 0
                if redis_ok:
                    await _set_active_flag(client, namespace, wake=not emitting)
                    if not emitting:
                        await asyncio.sleep(_PD_SETTLE_DELAY)
                emitting = True
                try:
                    payload = await _build_emit_payload(
                        pipe,
                        client if redis_ok else None,
                        namespace,
                        worker_key,
                        tick,
                        slow_state,
                        agg_state,
                    )
                    await emit_dashboard(payload)
                except Exception:
                    _pd_pub_log.debug("Dashboard emit iteration failed", exc_info=True)
                if tick % _PD_REAUTH_EVERY == 0:
                    try:
                        await reauthorize_local_viewers()
                    except Exception:
                        _pd_pub_log.debug("viewer re-auth failed", exc_info=True)
                tick += 1
                await asyncio.sleep(_PD_PUBLISH_INTERVAL)
                continue

            emitting = False

            if not redis_ok:
                await asyncio.sleep(_PD_POLL_INTERVAL)
                continue

            try:
                is_active = await client.exists(active_key)
            except Exception:
                await asyncio.sleep(_PD_POLL_INTERVAL)
                continue

            if not is_active:
                if pubsub is not None:
                    try:
                        msg = await asyncio.wait_for(
                            pubsub.get_message(ignore_subscribe_messages=True, timeout=_PD_POLL_INTERVAL),
                            timeout=_PD_POLL_INTERVAL + 1.0,
                        )
                        if msg and msg.get("type") == "message":
                            continue
                    except Exception:
                        pubsub = None
                        await asyncio.sleep(_PD_POLL_INTERVAL)
                else:
                    await asyncio.sleep(_PD_POLL_INTERVAL)
                continue

            await _write_own_slice(client, worker_key, pipe)
            await asyncio.sleep(_PD_PUBLISH_INTERVAL)

    except asyncio.CancelledError:
        _pd_pub_log.debug("Dashboard publisher cancelled (pid=%d)", pid)
    finally:
        if pubsub is not None:
            try:
                await pubsub.unsubscribe(wake_channel)
                await pubsub.close()
            except Exception:
                pass
        try:
            client, enabled = get_redis()
            if enabled and client is not None:
                await client.delete(worker_key)
        except Exception:
            pass
