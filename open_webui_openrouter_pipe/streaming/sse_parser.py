"""Server-Sent Events (SSE) parsing.

This module handles SSE stream parsing with a producer/worker/distributor pipeline:
- Event line parsing (data:, event:, id:, retry:)
- Multi-line event accumulation
- JSON delta parsing
- Event type classification
- Delta batching optimization for performance
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any, AsyncGenerator, Optional, Callable, Awaitable

import aiohttp
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

LOGGER = logging.getLogger(__name__)


class SSEParser:
    """Server-Sent Events parser with async producer/worker/distributor pipeline.

    This class implements a high-performance SSE parsing pipeline that:
    - Reads HTTP chunks in a dedicated producer task
    - Parses JSON in parallel worker tasks
    - Distributes events in order with delta batching optimization

    The pipeline ensures events are processed in order while maintaining high throughput
    through parallel JSON parsing.
    """

    def __init__(
        self,
        *,
        workers: int = 4,
        delta_char_limit: int = 256,
        idle_flush_ms: int = 30,
        chunk_queue_size: int = 32,
        event_queue_size: int = 32,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the SSE parser.

        Args:
            workers: Number of parallel JSON parsing workers (default: 4)
            delta_char_limit: Character threshold for batching delta events (default: 256)
            idle_flush_ms: Milliseconds to wait before flushing buffered deltas (default: 30)
            chunk_queue_size: Maximum size of chunk queue (default: 32)
            event_queue_size: Maximum size of event queue (default: 32)
            logger: Logger instance for diagnostic output (default: module logger)
        """
        self.workers = max(1, workers)
        self.delta_char_limit = max(0, delta_char_limit)
        self.idle_flush_ms = max(0, idle_flush_ms)
        self.chunk_queue_size = max(1, chunk_queue_size)
        self.event_queue_size = max(1, event_queue_size)
        self.logger = logger or LOGGER

    async def parse_sse_stream(
        self,
        session: aiohttp.ClientSession,
        url: str,
        *,
        request_body: dict[str, Any],
        headers: dict[str, str],
        breaker_check: Optional[Callable[[], bool]] = None,
        record_failure: Optional[Callable[[], None]] = None,
        extract_error: Optional[Callable[[dict[str, Any]], Optional[Exception]]] = None,
        event_queue_warn_size: int = 100,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Parse SSE byte stream from HTTP response into ordered JSON events.

        This method implements a three-stage pipeline:
        1. Producer: Reads HTTP chunks and extracts SSE events
        2. Workers: Parse JSON in parallel
        3. Distributor: Ensures order and batches deltas

        Args:
            session: aiohttp ClientSession for making HTTP requests
            url: URL to send request to
            request_body: JSON request body
            headers: HTTP headers
            breaker_check: Optional callback to check if circuit breaker is open
            record_failure: Optional callback to record failures
            extract_error: Optional callback to extract errors from events
            event_queue_warn_size: Queue size threshold for warnings

        Yields:
            Ordered dict events with delta batching applied

        Raises:
            Exception: Any errors from the HTTP request or parsing
        """
        chunk_queue: asyncio.Queue[tuple[Optional[int], Optional[bytes]]] = asyncio.Queue(
            maxsize=self.chunk_queue_size
        )
        event_queue: asyncio.Queue[tuple[Optional[int], Optional[dict[str, Any]]]] = asyncio.Queue(
            maxsize=self.event_queue_size
        )

        chunk_sentinel = (None, None)
        requested_model = request_body.get("model")

        # Launch producer task
        producer_task = asyncio.create_task(
            self._producer(
                session=session,
                url=url,
                request_body=request_body,
                headers=headers,
                chunk_queue=chunk_queue,
                breaker_check=breaker_check,
                record_failure=record_failure,
                requested_model=requested_model,
            ),
            name="sse-producer",
        )

        # Launch worker tasks
        worker_tasks = [
            asyncio.create_task(
                self._worker(
                    worker_id=idx,
                    chunk_queue=chunk_queue,
                    event_queue=event_queue,
                ),
                name=f"sse-worker-{idx}",
            )
            for idx in range(self.workers)
        ]

        # Distribute events with ordering and batching
        try:
            async for event in self._distributor(
                event_queue=event_queue,
                extract_error=extract_error,
                requested_model=requested_model,
                event_queue_warn_size=event_queue_warn_size,
            ):
                yield event

            # Wait for producer to complete
            await producer_task
        finally:
            # Cleanup tasks
            if not producer_task.done():
                producer_task.cancel()
            for task in worker_tasks:
                if not task.done():
                    task.cancel()

            # Gather results and log any errors
            results = await asyncio.gather(producer_task, *worker_tasks, return_exceptions=True)
            for idx, result in enumerate(results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    task_name = "producer" if idx == 0 else f"worker-{idx - 1}"
                    self.logger.error(
                        "SSE %s task failed during cleanup: %s",
                        task_name,
                        result,
                        exc_info=result,
                    )

    async def _producer(
        self,
        *,
        session: aiohttp.ClientSession,
        url: str,
        request_body: dict[str, Any],
        headers: dict[str, str],
        chunk_queue: asyncio.Queue,
        breaker_check: Optional[Callable[[], bool]],
        record_failure: Optional[Callable[[], None]],
        requested_model: Optional[str],
    ) -> None:
        """Producer task: Read HTTP stream, parse SSE lines, emit complete events.

        This task:
        - Makes the HTTP POST request
        - Reads chunks from the response stream (4KB buffer)
        - Parses SSE format (data: lines, comment lines, empty line delimiters)
        - Detects [DONE] termination signal
        - Emits complete SSE events to chunk queue for worker processing

        Args:
            session: aiohttp ClientSession
            url: Request URL
            request_body: JSON request body
            headers: HTTP headers
            chunk_queue: Queue to send parsed SSE events to
            breaker_check: Optional circuit breaker check callback
            record_failure: Optional failure recording callback
            requested_model: Model ID for error reporting
        """
        seq = 0
        retryer = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
            reraise=True,
        )

        try:
            async for attempt in retryer:
                if breaker_check and not breaker_check():
                    raise RuntimeError("Circuit breaker open for user during stream")

                with attempt:
                    buf = bytearray()
                    event_data_parts: list[bytes] = []
                    stream_complete = False

                    try:
                        async with session.post(url, json=request_body, headers=headers) as resp:
                            # Handle HTTP errors
                            if resp.status >= 400:
                                if record_failure:
                                    record_failure()
                                # Re-raise - let caller handle error response
                                resp.raise_for_status()

                            resp.raise_for_status()

                            # Read chunks from response stream
                            async for chunk in resp.content.iter_chunked(4096):
                                view = memoryview(chunk)
                                if breaker_check and not breaker_check():
                                    raise RuntimeError("Circuit breaker open during stream")

                                buf.extend(view)
                                start_idx = 0

                                # Parse lines
                                while True:
                                    newline_idx = buf.find(b"\n", start_idx)
                                    if newline_idx == -1:
                                        break

                                    line = buf[start_idx:newline_idx]
                                    start_idx = newline_idx + 1
                                    stripped = line.strip()

                                    # Empty line = event boundary
                                    if not stripped:
                                        if event_data_parts:
                                            data_blob = b"\n".join(event_data_parts).strip()
                                            event_data_parts.clear()
                                            if not data_blob:
                                                continue

                                            # [DONE] signal
                                            if data_blob == b"[DONE]":
                                                stream_complete = True
                                                break

                                            # Enqueue complete SSE event
                                            await chunk_queue.put((seq, data_blob))
                                            seq += 1
                                        continue

                                    # Skip comment lines
                                    if stripped.startswith(b":"):
                                        continue

                                    # Parse data: lines
                                    if stripped.startswith(b"data:"):
                                        event_data_parts.append(bytes(stripped[5:].lstrip()))
                                        continue

                                # Clear processed bytes from buffer
                                if start_idx > 0:
                                    del buf[:start_idx]

                                if stream_complete:
                                    break

                            # Handle any remaining event data
                            if event_data_parts and not stream_complete:
                                data_blob = b"\n".join(event_data_parts).strip()
                                event_data_parts.clear()
                                if data_blob and data_blob != b"[DONE]":
                                    await chunk_queue.put((seq, data_blob))
                                    seq += 1

                    except Exception as producer_exc:
                        # Import here to avoid circular dependency
                        from ..core.errors import OpenRouterAPIError

                        is_auth_failure = (
                            isinstance(producer_exc, OpenRouterAPIError)
                            and getattr(producer_exc, "status", None) in {401, 403}
                        )

                        if is_auth_failure:
                            self.logger.warning(
                                "Producer encountered auth error while streaming: %s",
                                producer_exc,
                            )
                        else:
                            self.logger.error(
                                "Producer encountered error while streaming: %s",
                                producer_exc,
                                exc_info=True,
                            )

                        if record_failure:
                            record_failure()
                        raise

                    if stream_complete:
                        break
        finally:
            # Send sentinel to workers
            for _ in range(self.workers):
                with contextlib.suppress(asyncio.CancelledError):
                    await chunk_queue.put((None, None))

    async def _worker(
        self,
        *,
        worker_id: int,
        chunk_queue: asyncio.Queue,
        event_queue: asyncio.Queue,
    ) -> None:
        """Worker task: Parse JSON from SSE event data.

        This task:
        - Reads SSE events from chunk queue
        - Decodes UTF-8 and parses JSON
        - Handles parse errors gracefully
        - Emits parsed events to event queue for distribution

        Args:
            worker_id: Worker index for logging
            chunk_queue: Queue to read raw SSE events from
            event_queue: Queue to send parsed JSON events to
        """
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
                        self.logger.warning("Chunk parse failed (seq=%s): %s", seq, exc)
                        continue

                    await event_queue.put((seq, event))
                finally:
                    chunk_queue.task_done()
        finally:
            # Send sentinel to distributor
            with contextlib.suppress(asyncio.CancelledError):
                await event_queue.put((None, None))

    async def _distributor(
        self,
        *,
        event_queue: asyncio.Queue,
        extract_error: Optional[Callable[[dict[str, Any]], Optional[Exception]]],
        requested_model: Optional[str],
        event_queue_warn_size: int,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Distributor task: Order events and batch deltas for efficiency.

        This task:
        - Ensures events are yielded in sequence order
        - Batches delta events to reduce downstream processing
        - Implements timeout-based flushing of buffered deltas
        - Monitors queue backlog and warns if high

        Args:
            event_queue: Queue to read parsed events from
            extract_error: Optional callback to extract errors from events
            requested_model: Model ID for error context
            event_queue_warn_size: Queue size threshold for warnings

        Yields:
            Ordered events with delta batching applied
        """
        delta_batch_threshold = max(1, self.delta_char_limit)
        idle_flush_seconds = float(self.idle_flush_ms) / 1000 if self.idle_flush_ms > 0 else None
        passthrough_deltas = self.delta_char_limit <= 0 and self.idle_flush_ms <= 0

        pending_events: dict[int, dict[str, Any]] = {}
        next_seq = 0
        done_workers = 0
        delta_buffer: list[str] = []
        delta_template: Optional[dict[str, Any]] = None
        delta_length = 0
        event_queue_warn_last_ts: float = 0.0

        def flush_delta(force: bool = False) -> Optional[dict[str, Any]]:
            """Flush buffered delta events into a single batched event."""
            if passthrough_deltas:
                return None

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
                # Use timeout if we have buffered deltas
                timeout = idle_flush_seconds if (idle_flush_seconds and delta_buffer) else None
                timed_out = False
                seq: int | None = None
                event: dict[str, Any] | None = None

                # Wait for next event with optional timeout
                if timeout is not None:
                    try:
                        seq, event = await asyncio.wait_for(event_queue.get(), timeout=timeout)
                    except asyncio.TimeoutError:
                        timed_out = True
                else:
                    seq, event = await event_queue.get()

                # Timeout: flush buffered deltas
                if timed_out:
                    batched = flush_delta(force=True)
                    if batched:
                        yield batched
                    continue

                event_queue.task_done()

                # Monitor queue backlog
                now = time.perf_counter()
                qsize = event_queue.qsize()
                if qsize >= event_queue_warn_size and (now - event_queue_warn_last_ts) > 5.0:
                    self.logger.warning("Event queue backlog high: %d items", qsize)
                    event_queue_warn_last_ts = now

                # Worker done sentinel
                if seq is None:
                    done_workers += 1
                    if done_workers >= self.workers and not pending_events:
                        break
                    continue

                # Skip empty events
                if event is None:
                    self.logger.debug("Skipping empty SSE event (seq=%s)", seq)
                    continue

                # Buffer event until we can emit in order
                pending_events[seq] = event

                # Emit events in sequence order
                while next_seq in pending_events:
                    current = pending_events.pop(next_seq)
                    next_seq += 1

                    # Check for error events
                    if extract_error:
                        streaming_error = extract_error(current)
                        if streaming_error is not None:
                            raise streaming_error

                    etype = current.get("type")

                    # Handle delta batching
                    if etype == "response.output_text.delta":
                        delta_chunk = current.get("delta", "")
                        if passthrough_deltas:
                            if delta_chunk:
                                yield current
                            continue

                        if delta_chunk:
                            delta_buffer.append(delta_chunk)
                            delta_length += len(delta_chunk)
                            if delta_template is None:
                                delta_template = {k: v for k, v in current.items() if k != "delta"}

                        batched = flush_delta()
                        if batched:
                            yield batched
                        continue

                    # Non-delta event: flush any buffered deltas first
                    batched = flush_delta(force=True)
                    if batched:
                        yield batched
                    yield current

            # Flush final buffered deltas
            final_delta = flush_delta(force=True)
            if final_delta:
                yield final_delta

        except Exception:
            # Ensure we flush any buffered deltas before propagating error
            final_delta = flush_delta(force=True)
            if final_delta:
                yield final_delta
            raise
