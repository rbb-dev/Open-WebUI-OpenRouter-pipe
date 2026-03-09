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
    from .._utils import EphemeralKeyStore
    from .session import SessionRegistry

_ts_sse_log = logging.getLogger(__name__)


def register_think_streaming_route(
    key_store: EphemeralKeyStore,
    session_registry: SessionRegistry,
) -> bool:
    """Register the Think Streaming SSE endpoint on OWUI's FastAPI app.

    Returns ``True`` if registered (or already exists), ``False`` if
    the OWUI app could not be imported.
    """
    from .._utils import register_sse_endpoint

    async def _think_streaming_sse(key: str) -> Any:
        from starlette.responses import PlainTextResponse, StreamingResponse

        if not await key_store.async_validate(key):
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

    return register_sse_endpoint(
        "/api/pipe/think_streaming/{key}",
        _think_streaming_sse,
        logger=_ts_sse_log,
    )


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
                if not await key_store.async_validate(key):
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
