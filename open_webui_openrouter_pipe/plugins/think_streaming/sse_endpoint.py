"""SSE endpoint for Think Streaming sessions.

Registers ``GET /api/pipe/think_streaming/{key}`` on OWUI's FastAPI app.
The iframe's ``EventSource`` connects to this route to receive thinking
events in real time.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..pipe_stats.ephemeral_keys import EphemeralKeyStore
    from .session import SessionRegistry

_ts_sse_log = logging.getLogger(__name__)

_ts_registered = False


def register_think_streaming_route(
    key_store: EphemeralKeyStore,
    session_registry: SessionRegistry,
) -> bool:
    """Register the Think Streaming SSE endpoint on OWUI's FastAPI app.

    Returns ``True`` if registered (or already exists), ``False`` if
    the OWUI app could not be imported.
    """
    global _ts_registered
    if _ts_registered:
        return True

    try:
        from starlette.responses import PlainTextResponse, StreamingResponse

        from .._utils import ensure_route_before_spa, get_owui_app
    except ImportError:
        _ts_sse_log.debug("Dependencies not available — Think Streaming endpoint not registered")
        return False

    app = get_owui_app()
    if app is None:
        _ts_sse_log.debug("OWUI app not available — Think Streaming endpoint not registered")
        return False

    @app.get("/api/pipe/think_streaming/{key}")
    async def _think_streaming_sse(key: str) -> Any:
        if not key_store.validate(key):
            return PlainTextResponse("Invalid or expired key", status_code=403)
        session = session_registry.get(key)
        if session is None:
            return PlainTextResponse("No session for key", status_code=404)
        return StreamingResponse(
            _generate(session, key_store, key),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    ensure_route_before_spa(app)
    _ts_registered = True
    _ts_sse_log.debug("Think Streaming SSE endpoint registered at /api/pipe/think_streaming/{key}")
    return True


async def _generate(
    session: Any,
    key_store: Any,
    key: str,
) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE events from the session queue.

    Yields thinking deltas, completion markers, and heartbeats.  On
    generator exit (client disconnect), sets ``consumer_alive = False``
    to trigger graceful fallback in the emitter wrapper.
    """
    try:
        while session.consumer_alive:
            try:
                item = await asyncio.wait_for(session.queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # Send heartbeat to keep the connection alive
                yield ": heartbeat\n\n"
                # Check if the key is still valid
                if not key_store.validate(key):
                    break
                continue

            if item is None:
                # Completion sentinel — stream is done
                yield 'data: {"type":"done"}\n\n'
                break

            yield f"data: {item}\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        session.consumer_alive = False
