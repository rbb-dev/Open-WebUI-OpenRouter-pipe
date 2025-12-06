"""OpenRouter Responses API streaming adapter.

This module provides SSE (Server-Sent Events) streaming integration with OpenRouter:
- Producer/consumer pipeline architecture
- Multi-worker SSE chunk parsing
- Delta batching with configurable thresholds
- Sequence-ordered event delivery
- Idle flush watchdog for responsive UI
- Retry logic with exponential backoff
- Circuit breaker integration

Layer: adapters (integrates with external OpenRouter service)

Dependencies:
- aiohttp: HTTP client for SSE streaming
- tenacity: Retry logic with exponential backoff
- .models: build_openrouter_api_error, debug_print_error_response

Architecture:
    HTTP Stream → Producer → Chunk Queue → Workers → Event Queue → Ordered Output

    1. Producer: Reads SSE stream, splits by newlines, enqueues raw JSON blobs
    2. Workers (4x): Parse JSON blobs in parallel, forward to event queue
    3. Consumer: Reorders events by sequence number, batches deltas, yields to caller
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional

import aiohttp
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .models import build_openrouter_api_error, debug_print_error_response

# OpenRouter metadata
_OPENROUTER_TITLE = "OpenRouter Responses Pipe"

# Module logger
LOGGER = logging.getLogger(__name__)


async def send_streaming_request(
    session: aiohttp.ClientSession,
    request_body: dict[str, Any],
    api_key: str,
    base_url: str,
    *,
    workers: int = 4,
    breaker_allows_fn: Optional[callable] = None,
    record_failure_fn: Optional[callable] = None,
    delta_char_limit: int = 50,
    idle_flush_ms: int = 0,
    chunk_queue_maxsize: int = 100,
    event_queue_maxsize: int = 100,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream OpenRouter Responses API via producer/worker SSE pipeline.

    Args:
        session: aiohttp client session
        request_body: JSON payload for /responses endpoint
        api_key: OpenRouter API key
        base_url: OpenRouter base URL (e.g., https://openrouter.ai/api/v1)
        workers: Number of parallel JSON parsing workers (1-8)
        breaker_allows_fn: Optional circuit breaker check function() -> bool
        record_failure_fn: Optional failure recorder function() for breaker
        delta_char_limit: Batch deltas until reaching this character count
        idle_flush_ms: Force flush after this many ms without new events (0=disabled)
        chunk_queue_maxsize: Raw chunk buffer size before backpressure
        event_queue_maxsize: Parsed event buffer size before backpressure

    Yields:
        Parsed SSE events as dicts, with deltas batched for efficiency

    Raises:
        OpenRouterAPIError: On HTTP 400 errors (user/provider issues)
        RuntimeError: On non-retryable HTTP errors
        aiohttp.ClientError: On network failures after retries
        asyncio.TimeoutError: On timeout after retries

    Pipeline Architecture:
        ```
        HTTP SSE Stream
            ↓
        Producer (parses SSE format, enqueues raw JSON)
            ↓
        Chunk Queue (bounded buffer)
            ↓
        Workers × N (parallel JSON.loads, enqueues events)
            ↓
        Event Queue (bounded buffer)
            ↓
        Consumer (reorders by sequence, batches deltas, yields)
        ```

    Delta Batching:
        - response.output_text.delta events are accumulated
        - Flushed when: char limit reached, non-delta event, idle timeout, stream end
        - Reduces event volume by ~10-50x for chatty models

    Sequence Ordering:
        - Producer assigns monotonic sequence numbers
        - Consumer buffers out-of-order events
        - Guarantees in-order delivery despite parallel parsing

    Circuit Breaker:
        - Checks breaker before starting and during stream
        - Records failures on HTTP errors
        - Caller responsible for breaker logic

    Example:
        >>> async with aiohttp.ClientSession() as session:
        ...     async for event in send_streaming_request(
        ...         session,
        ...         {"model": "openai/gpt-4", "input": "Hello", "stream": True},
        ...         api_key="sk-...",
        ...         base_url="https://openrouter.ai/api/v1",
        ...         workers=4,
        ...         delta_char_limit=50,
        ...     ):
        ...         print(event)
    """
    from .models import debug_print_request

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Title": _OPENROUTER_TITLE,
    }
    debug_print_request(headers, request_body)
    url = base_url.rstrip("/") + "/responses"

    workers = max(1, min(int(workers or 1), 8))
    chunk_queue_size = max(0, int(chunk_queue_maxsize))
    event_queue_size = max(0, int(event_queue_maxsize))
    chunk_queue: asyncio.Queue[tuple[Optional[int], bytes]] = asyncio.Queue(maxsize=chunk_queue_size)
    event_queue: asyncio.Queue[tuple[Optional[int], Optional[dict[str, Any]]]] = asyncio.Queue(maxsize=event_queue_size)
    chunk_sentinel = (None, b"")
    delta_batch_threshold = max(1, int(delta_char_limit))
    idle_flush_seconds = float(idle_flush_ms) / 1000 if idle_flush_ms > 0 else None
    producer_error: BaseException | None = None

    async def _producer() -> None:
        """Read SSE stream, parse SSE format, enqueue raw JSON blobs."""
        nonlocal producer_error
        seq = 0
        retryer = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
            reraise=True,
        )
        try:
            async for attempt in retryer:
                if breaker_allows_fn and not breaker_allows_fn():
                    raise RuntimeError("Circuit breaker open during stream")
                with attempt:
                    buf = bytearray()
                    event_data_parts: list[bytes] = []
                    stream_complete = False
                    try:
                        async with session.post(url, json=request_body, headers=headers) as resp:
                            if resp.status >= 400:
                                error_body = await debug_print_error_response(resp)
                                if record_failure_fn:
                                    record_failure_fn()
                                if resp.status == 400:
                                    producer_error = build_openrouter_api_error(
                                        resp.status,
                                        resp.reason or "Bad Request",
                                        error_body,
                                        requested_model=request_body.get("model"),
                                    )
                                    return
                                if resp.status < 500:
                                    raise RuntimeError(
                                        f"OpenRouter request failed ({resp.status}): {resp.reason}"
                                    )
                            resp.raise_for_status()

                            async for chunk in resp.content.iter_chunked(4096):
                                view = memoryview(chunk)
                                if breaker_allows_fn and not breaker_allows_fn():
                                    raise RuntimeError("Circuit breaker open during stream")
                                buf.extend(view)
                                start_idx = 0
                                while True:
                                    newline_idx = buf.find(b"\n", start_idx)
                                    if newline_idx == -1:
                                        break
                                    line = buf[start_idx:newline_idx]
                                    start_idx = newline_idx + 1
                                    stripped = line.strip()
                                    if not stripped:
                                        if event_data_parts:
                                            data_blob = b"\n".join(event_data_parts).strip()
                                            event_data_parts.clear()
                                            if not data_blob:
                                                continue
                                            if data_blob == b"[DONE]":
                                                stream_complete = True
                                                break
                                            await chunk_queue.put((seq, data_blob))
                                            seq += 1
                                        continue
                                    if stripped.startswith(b":"):
                                        continue
                                    if stripped.startswith(b"data:"):
                                        event_data_parts.append(stripped[5:].lstrip())
                                        continue
                                if start_idx > 0:
                                    del buf[:start_idx]
                                if stream_complete:
                                    break

                            if event_data_parts and not stream_complete:
                                data_blob = b"\n".join(event_data_parts).strip()
                                event_data_parts.clear()
                                if data_blob and data_blob != b"[DONE]":
                                    await chunk_queue.put((seq, data_blob))
                                    seq += 1
                    except Exception:
                        if record_failure_fn:
                            record_failure_fn()
                        raise
                    if stream_complete:
                        break
        finally:
            for _ in range(workers):
                with contextlib.suppress(asyncio.CancelledError):
                    await chunk_queue.put(chunk_sentinel)

    async def _worker(worker_idx: int) -> None:
        """Parse JSON blobs from chunk queue, forward to event queue."""
        try:
            while True:
                seq, data = await chunk_queue.get()
                try:
                    if seq is None:
                        break
                    if data == b"[DONE]":
                        continue
                    try:
                        event = json.loads(data.decode("utf-8"))
                    except json.JSONDecodeError as exc:
                        LOGGER.warning("Chunk parse failed (seq=%s): %s", seq, exc)
                        continue
                    await event_queue.put((seq, event))
                finally:
                    chunk_queue.task_done()
        finally:
            with contextlib.suppress(asyncio.CancelledError):
                await event_queue.put((None, None))

    producer_task = asyncio.create_task(_producer(), name="openrouter-sse-producer")
    worker_tasks = [
        asyncio.create_task(_worker(idx), name=f"openrouter-sse-worker-{idx}")
        for idx in range(workers)
    ]

    pending_events: dict[int, dict[str, Any]] = {}
    next_seq = 0
    done_workers = 0
    delta_buffer: list[str] = []
    delta_template: Optional[dict[str, Any]] = None
    delta_length = 0

    def flush_delta(force: bool = False) -> Optional[dict[str, Any]]:
        """Flush accumulated deltas when threshold reached or forced."""
        nonlocal delta_buffer, delta_template, delta_length
        if delta_buffer and (force or delta_length >= delta_batch_threshold):
            combined = "".join(delta_buffer)
            base = dict(delta_template or {"type": "response.output_text.delta"})
            base["delta"] = combined
            delta_buffer = []
            delta_template = None
            delta_length = 0
            return base
        return None

    try:
        while True:
            timeout = idle_flush_seconds if (idle_flush_seconds and delta_buffer) else None
            timed_out = False
            if timeout is not None:
                try:
                    seq, event = await asyncio.wait_for(event_queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    timed_out = True
            else:
                seq, event = await event_queue.get()

            if timed_out:
                batched = flush_delta(force=True)
                if batched:
                    yield batched
                continue

            event_queue.task_done()
            if seq is None:
                done_workers += 1
                if done_workers >= workers and not pending_events:
                    break
                continue
            pending_events[seq] = event
            while next_seq in pending_events:
                current = pending_events.pop(next_seq)
                next_seq += 1
                etype = current.get("type")
                if etype == "response.output_text.delta":
                    delta_chunk = current.get("delta", "")
                    if delta_chunk:
                        delta_buffer.append(delta_chunk)
                        delta_length += len(delta_chunk)
                        if delta_template is None:
                            delta_template = {k: v for k, v in current.items() if k != "delta"}
                    batched = flush_delta()
                    if batched:
                        yield batched
                    continue

                batched = flush_delta(force=True)
                if batched:
                    yield batched
                yield current

        final_delta = flush_delta(force=True)
        if final_delta:
            yield final_delta

        await producer_task
        if producer_error is not None:
            raise producer_error
    finally:
        if not producer_task.done():
            producer_task.cancel()
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(producer_task, *worker_tasks, return_exceptions=True)


__all__ = ["send_streaming_request"]
