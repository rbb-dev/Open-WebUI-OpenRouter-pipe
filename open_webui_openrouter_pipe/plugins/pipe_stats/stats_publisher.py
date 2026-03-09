"""Per-worker background stats publisher for multi-worker Redis aggregation.

Each uvicorn worker runs a lightweight async task that:

1. **Idle mode** — checks every ``_PS_POLL_INTERVAL`` seconds whether the
   ``{ns}:stats:active`` flag exists in Redis.  Cost: one EXISTS call.
2. **Publish mode** — when the flag is present (set by the SSE endpoint),
   writes this worker's per-process stats to
   ``{ns}:stats:worker:{pid}`` every ``_PS_PUBLISH_INTERVAL`` seconds with a
   short TTL.  When the flag disappears (dashboard closed, TTL expired),
   the task returns to idle mode.

The SSE endpoint also publishes a wake-up message on
``{ns}:stats:wake`` for instant activation via pub/sub; otherwise the
worst-case latency before a worker starts publishing is
``_PS_POLL_INTERVAL`` seconds.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from ._collectors import collect_concurrency, collect_queues, collect_rate_limits, collect_sessions

_ps_pub_log = logging.getLogger(__name__)

# Timing constants
_PS_POLL_INTERVAL = 5.0   # seconds between flag checks when idle
_PS_PUBLISH_INTERVAL = 2.0  # seconds between stats writes when active
_PS_KEY_TTL = 10           # TTL on per-worker key (seconds)

# Captured once — closest proxy for process start time.
_PS_PUB_PROCESS_START = time.monotonic()


def _collect_worker_payload(pipe: Any) -> dict[str, Any]:
    """Collect compact per-worker stats for Redis publishing.

    Only includes data that is *unique* to this process — shared data
    (models, storage, config, plugins) is read locally by the SSE
    reader and doesn't need Redis sync.

    Delegates to shared collectors and maps full keys to compact Redis keys.
    """
    c = collect_concurrency(pipe)
    q = collect_queues(pipe)
    rl = collect_rate_limits(pipe)
    s = collect_sessions()

    return {
        "pid": os.getpid(),
        "up": round(time.monotonic() - _PS_PUB_PROCESS_START, 1),
        "c": {
            "ar": c["active_requests"],
            "mr": c["max_requests"],
            "at": c["active_tools"],
            "mt": c["max_tools"],
        },
        "q": {
            "rq": q["requests"],
            "rm": q["requests_max"],
            "lq": q["logs"],
            "aq": q["archive"],
        },
        "rl": {
            "tu": rl["tracked_users"],
            "fu": rl["users_with_failures"],
            "tr": rl["tripped_users"],
            "th": rl["threshold"],
            "ws": rl["window_s"],
            "tt": rl["tool_tracked"],
            "tp": rl["tool_tripped"],
            "aa": rl["auth_failures_active"],
        },
        "s": s["active"],
    }


# Field mapping: compact Redis keys -> full SSE keys
_PS_CONCURRENCY_MAP = {
    "ar": "active_requests",
    "mr": "max_requests",
    "at": "active_tools",
    "mt": "max_tools",
}

_PS_QUEUE_MAP = {
    "rq": "requests",
    "rm": "requests_max",
    "lq": "logs",
    "aq": "archive",
}

_PS_RATE_LIMIT_MAP = {
    "tu": "tracked_users",
    "fu": "users_with_failures",
    "tr": "tripped_users",
    "th": "threshold",
    "ws": "window_s",
    "tt": "tool_tracked",
    "tp": "tool_tripped",
    "aa": "auth_failures_active",
}


def expand_worker_payload(compact: dict[str, Any]) -> dict[str, Any]:
    """Expand a compact Redis payload into the full SSE field names.

    Used by the SSE reader to reconstitute per-worker data after
    reading from Redis.
    """
    return {
        "pid": compact.get("pid", 0),
        "uptime_s": compact.get("up", 0),
        "concurrency": {_PS_CONCURRENCY_MAP[k]: v for k, v in compact.get("c", {}).items() if k in _PS_CONCURRENCY_MAP},
        "queues": {_PS_QUEUE_MAP[k]: v for k, v in compact.get("q", {}).items() if k in _PS_QUEUE_MAP},
        "rate_limits": {_PS_RATE_LIMIT_MAP[k]: v for k, v in compact.get("rl", {}).items() if k in _PS_RATE_LIMIT_MAP},
        "sessions": {"active": compact.get("s", 0)},
    }


def aggregate_worker_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate expanded payloads from multiple workers into a single SSE payload.

    Numeric fields are summed across workers.  Per-worker max values
    (semaphore limits, queue max) are multiplied by worker count so
    that summed active values are comparable against total system capacity.
    """
    if not payloads:
        return {}

    # Per-worker limits are identical; multiply by worker count for totals.
    first = payloads[0]
    n = len(payloads)

    # Sum concurrency — limits × worker count
    agg_concurrency = {
        "active_requests": sum(p.get("concurrency", {}).get("active_requests", 0) for p in payloads),
        "max_requests": first.get("concurrency", {}).get("max_requests", 0) * n,
        "active_tools": sum(p.get("concurrency", {}).get("active_tools", 0) for p in payloads),
        "max_tools": first.get("concurrency", {}).get("max_tools", 0) * n,
    }

    # Sum queues — limits × worker count
    agg_queues = {
        "requests": sum(p.get("queues", {}).get("requests", 0) for p in payloads),
        "requests_max": first.get("queues", {}).get("requests_max", 1000) * n,
        "logs": sum(p.get("queues", {}).get("logs", 0) for p in payloads),
        "archive": sum(p.get("queues", {}).get("archive", 0) for p in payloads),
    }

    # Sum rate limits (tracked users may overlap, but counts are per-process)
    agg_rate_limits = {
        "tracked_users": sum(p.get("rate_limits", {}).get("tracked_users", 0) for p in payloads),
        "users_with_failures": sum(p.get("rate_limits", {}).get("users_with_failures", 0) for p in payloads),
        "tripped_users": sum(p.get("rate_limits", {}).get("tripped_users", 0) for p in payloads),
        "threshold": first.get("rate_limits", {}).get("threshold", 0),
        "window_s": first.get("rate_limits", {}).get("window_s", 0),
        "tool_tracked": sum(p.get("rate_limits", {}).get("tool_tracked", 0) for p in payloads),
        "tool_tripped": sum(p.get("rate_limits", {}).get("tool_tripped", 0) for p in payloads),
        "auth_failures_active": sum(p.get("rate_limits", {}).get("auth_failures_active", 0) for p in payloads),
    }

    # Sum sessions
    agg_sessions = {
        "active": sum(p.get("sessions", {}).get("active", 0) for p in payloads),
    }

    # Workers table: pid + uptime for each worker
    workers = [
        {"pid": p.get("pid", 0), "uptime_s": p.get("uptime_s", 0)}
        for p in payloads
    ]

    # Use max uptime across workers (longest-running = system uptime)
    max_uptime = max((p.get("uptime_s", 0) for p in payloads), default=0)

    return {
        "uptime_s": max_uptime,
        "pid": payloads[0].get("pid", 0),
        "concurrency": agg_concurrency,
        "queues": agg_queues,
        "rate_limits": agg_rate_limits,
        "sessions": agg_sessions,
        "workers": sorted(workers, key=lambda w: w["pid"]),
    }


async def run_stats_publisher(
    get_pipe: Any,
    get_redis: Any,
    namespace: str,
) -> None:
    """Long-running background task: publish per-worker stats to Redis.

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
    active_key = f"{namespace}:stats:active"
    worker_key = f"{namespace}:stats:worker:{pid}"
    wake_channel = f"{namespace}:stats:wake"

    # Small initial delay to let the pipe finish initialization
    await asyncio.sleep(2.0)

    _ps_pub_log.debug("Stats publisher started (pid=%d, ns=%s)", pid, namespace)

    pubsub = None

    try:
        while True:
            client, enabled = get_redis()
            if not enabled or client is None:
                await asyncio.sleep(_PS_POLL_INTERVAL)
                continue

            # Set up pub/sub listener for instant wake-up
            if pubsub is None:
                try:
                    pubsub = client.pubsub()
                    await pubsub.subscribe(wake_channel)
                except Exception:
                    _ps_pub_log.debug("Pub/sub subscribe failed", exc_info=True)
                    pubsub = None

            # Check if anyone is watching
            try:
                is_active = await client.exists(active_key)
            except Exception:
                await asyncio.sleep(_PS_POLL_INTERVAL)
                continue

            if not is_active:
                # Idle mode — wait for flag or pub/sub wake
                if pubsub is not None:
                    try:
                        # Non-blocking check for wake message
                        msg = await asyncio.wait_for(
                            pubsub.get_message(ignore_subscribe_messages=True, timeout=_PS_POLL_INTERVAL),
                            timeout=_PS_POLL_INTERVAL + 1.0,
                        )
                        if msg and msg.get("type") == "message":
                            continue  # Re-check the active flag immediately
                    except (asyncio.TimeoutError, Exception):
                        pass
                else:
                    await asyncio.sleep(_PS_POLL_INTERVAL)
                continue

            # Publish mode — write stats until flag disappears
            pipe = get_pipe()
            if pipe is None:
                await asyncio.sleep(_PS_PUBLISH_INTERVAL)
                continue

            try:
                payload = _collect_worker_payload(pipe)
                await client.set(
                    worker_key,
                    json.dumps(payload, separators=(",", ":")),
                    ex=_PS_KEY_TTL,
                )
            except Exception:
                _ps_pub_log.debug("Stats publish failed (pid=%d)", pid, exc_info=True)

            await asyncio.sleep(_PS_PUBLISH_INTERVAL)

    except asyncio.CancelledError:
        _ps_pub_log.debug("Stats publisher cancelled (pid=%d)", pid)
    finally:
        # Clean up pub/sub
        if pubsub is not None:
            try:
                await pubsub.unsubscribe(wake_channel)
                await pubsub.close()
            except Exception:
                pass
        # Remove our worker key
        try:
            client, enabled = get_redis()
            if enabled and client is not None:
                await client.delete(worker_key)
        except Exception:
            pass
