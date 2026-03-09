"""SSE endpoint for the fully dynamic stats dashboard.

Dynamically registers a lightweight Server-Sent Events route on OWUI's
FastAPI app.  The dashboard iframe connects with a single
``new EventSource(url)`` call — zero external dependencies.

Events carry tiered JSON payloads:
- **fast** (every 2 s): concurrency, queues, rate limits, sessions, uptime
- **medium** (~16 s): model catalog, system health
- **slow** (~60 s): storage DB queries, config, plugins
- **once** (tick 0): identity (version, pipe_id, worker_count)

When Redis is available, the fast tier is **aggregated** from all
workers via ``{ns}:stats:worker:*`` keys.  The SSE endpoint sets a
``{ns}:stats:active`` flag to signal workers to start publishing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Any

from .runtime_stats import (
    collect_fast_stats,
    collect_identity,
    collect_medium_stats,
    collect_slow_stats,
)
from .stats_publisher import (
    _collect_worker_payload,
    aggregate_worker_payloads,
    expand_worker_payload,
)

if TYPE_CHECKING:
    from .._utils import EphemeralKeyStore

_ps_sse_log = logging.getLogger(__name__)


# How long the "active" flag lives in Redis (seconds).
# Renewed every tick by the SSE generator.
_PS_ACTIVE_FLAG_TTL = 600  # 10 minutes


async def _read_redis_workers(client: Any, namespace: str) -> list[dict[str, Any]]:
    """Read and expand all live worker payloads from Redis.

    Returns a list of expanded (full-field-name) worker dicts, or an
    empty list if Redis is unavailable or no workers are publishing.
    """
    pattern = f"{namespace}:stats:worker:*"
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
        _ps_sse_log.debug("Redis worker read failed", exc_info=True)
        return []


async def _activate_workers(client: Any, namespace: str) -> None:
    """Set the active flag and publish a wake-up message."""
    active_key = f"{namespace}:stats:active"
    wake_channel = f"{namespace}:stats:wake"
    try:
        await client.set(active_key, "1", ex=_PS_ACTIVE_FLAG_TTL)
    except Exception:
        _ps_sse_log.debug("Failed to set stats active flag", exc_info=True)
    try:
        await client.publish(wake_channel, "wake")
    except Exception:
        pass  # Pub/sub publish failure is non-critical


async def _renew_active_flag(client: Any, namespace: str) -> None:
    """Renew the active flag TTL (called every tick)."""
    active_key = f"{namespace}:stats:active"
    try:
        await client.set(active_key, "1", ex=_PS_ACTIVE_FLAG_TTL)
    except Exception:
        pass


def register_sse_route(
    key_store: EphemeralKeyStore,
    get_pipe: Any,
) -> bool:
    """Register ``GET /api/pipe/stats/{key}`` on OWUI's FastAPI app.

    Uses a closure to capture *key_store* and *get_pipe* (a callable
    returning the current pipe instance).  The route is registered once;
    subsequent calls are no-ops.

    Returns ``True`` if the route was registered (or already exists),
    ``False`` if the OWUI app could not be imported.
    """
    from .._utils import register_sse_endpoint

    async def _pipe_stats_sse(key: str) -> Any:
        from starlette.responses import StreamingResponse

        async def _generate():  # type: ignore[return-type]
            tick = 0
            redis_mode = False
            namespace = ""

            while await key_store.async_validate(key):
                pipe = get_pipe()
                if pipe is None:
                    break

                payload: dict[str, Any] = {"tick": tick}

                # Detect Redis availability on first tick
                if tick == 0:
                    redis_client = getattr(pipe, "_redis_client", None)
                    redis_enabled = getattr(pipe, "_redis_enabled", False)
                    if redis_enabled and redis_client is not None:
                        redis_mode = True
                        namespace = getattr(pipe, "_redis_namespace", "openrouter")
                        # Signal all workers to start publishing
                        await _activate_workers(redis_client, namespace)
                        # Small delay to let workers publish their first payload
                        await asyncio.sleep(1.0)

                # ── Fast tier: every tick ──
                local_pid = os.getpid()
                if redis_mode:
                    redis_client = getattr(pipe, "_redis_client", None)
                    if redis_client is not None:
                        # Renew active flag
                        await _renew_active_flag(redis_client, namespace)
                        # Read all worker payloads
                        worker_payloads = await _read_redis_workers(redis_client, namespace)
                        # Guarantee the local worker is included — its publisher
                        # might not have published yet on early ticks.
                        local_pids = {p.get("pid", 0) for p in worker_payloads}
                        if local_pid not in local_pids:
                            try:
                                local_compact = _collect_worker_payload(pipe)
                                worker_payloads.append(expand_worker_payload(local_compact))
                            except Exception:
                                _ps_sse_log.debug("Local worker payload collect error", exc_info=True)
                        if worker_payloads:
                            aggregated = aggregate_worker_payloads(worker_payloads)
                            payload.update(aggregated)
                            worker_count = len(worker_payloads)
                        else:
                            # No workers publishing yet — fall back to local
                            try:
                                payload.update(collect_fast_stats(pipe))
                            except Exception:
                                _ps_sse_log.debug("Fast stats collect error", exc_info=True)
                            worker_count = 1
                    else:
                        # Redis went away — fall back to local
                        redis_mode = False
                        try:
                            payload.update(collect_fast_stats(pipe))
                        except Exception:
                            _ps_sse_log.debug("Fast stats collect error", exc_info=True)
                        worker_count = 1
                else:
                    try:
                        payload.update(collect_fast_stats(pipe))
                    except Exception:
                        _ps_sse_log.debug("Fast stats collect error", exc_info=True)
                    worker_count = 1

                # ── Once tier: first event only ──
                if tick == 0:
                    try:
                        payload.update(collect_identity(pipe, worker_count=worker_count))
                    except Exception:
                        _ps_sse_log.debug("Identity collect error", exc_info=True)

                # ── Medium tier: ~16s (every 8th tick) ──
                if tick == 0 or tick % 8 == 0:
                    try:
                        payload.update(collect_medium_stats(pipe))
                    except Exception:
                        _ps_sse_log.debug("Medium stats collect error", exc_info=True)

                # ── Slow tier: ~60s (every 30th tick) ──
                if tick == 0 or tick % 30 == 0:
                    try:
                        payload.update(collect_slow_stats(pipe))
                    except Exception:
                        _ps_sse_log.debug("Slow stats collect error", exc_info=True)

                # Send worker_count updates periodically in Redis mode
                # (NOT inside identity — that would clobber cached version/pipe_id)
                if redis_mode and tick > 0 and tick % 4 == 0:
                    payload["worker_count"] = worker_count

                yield f"data: {json.dumps(payload)}\n\n"
                tick += 1
                await asyncio.sleep(2)

            # Terminal event — tells the client the session is over
            yield f"data: {json.dumps({'status': 'expired'})}\n\n"

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return register_sse_endpoint(
        "/api/pipe/stats/{key}",
        _pipe_stats_sse,
        logger=_ps_sse_log,
    )
