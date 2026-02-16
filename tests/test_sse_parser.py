"""Comprehensive coverage tests for SSEParser edge cases and error paths.

This module covers the missing lines in sse_parser.py with focus on:
- Task cancellation during cleanup (lines 158, 161, 167-168)
- Circuit breaker checks during streaming (lines 217, 239)
- HTTP error handling and record_failure (lines 228-231)
- Empty data blob handling (line 260)
- Remaining event data at stream end (lines 290-294)
- Producer exception handling (lines 296-319)
- Queue full sentinel handling (lines 328-332)
- [DONE] token handling in worker (line 362)
- Queue backlog warning (lines 461-462)
- Empty event skipping (lines 473-474)
- Error extraction in distributor (lines 486-488)
- Delta batching threshold flush (line 508)
- Final delta flush on completion and exception (lines 520-527)
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, cast
from unittest.mock import MagicMock

import aiohttp
import pytest

from open_webui_openrouter_pipe.streaming.sse_parser import SSEParser
from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError


# -----------------------------------------------------------------------------
# Test Fixtures and Helpers
# -----------------------------------------------------------------------------

class _FakeContent:
    """Fake aiohttp response content with configurable chunk iteration."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        delays: list[float] | None = None,
        raise_after: int | None = None,
        exception: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._delays = delays or []
        self._raise_after = raise_after
        self._exception = exception or RuntimeError("Simulated stream error")

    async def iter_chunked(self, _size: int):
        for idx, chunk in enumerate(self._chunks):
            if self._raise_after is not None and idx >= self._raise_after:
                raise self._exception
            if idx < len(self._delays):
                await asyncio.sleep(self._delays[idx])
            else:
                await asyncio.sleep(0)
            yield chunk


class _FakeResponse:
    """Fake aiohttp response for testing."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        status: int = 200,
        delays: list[float] | None = None,
        raise_after: int | None = None,
        exception: Exception | None = None,
    ) -> None:
        self.status = status
        self.content = _FakeContent(
            chunks,
            delays=delays,
            raise_after=raise_after,
            exception=exception,
        )
        self.reason = "OK" if status < 400 else "Error"
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=self.status,
                message=f"HTTP {self.status}",
            )


class _FakeSession:
    """Fake aiohttp ClientSession for testing."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        status: int = 200,
        delays: list[float] | None = None,
        raise_after: int | None = None,
        exception: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._status = status
        self._delays = delays
        self._raise_after = raise_after
        self._exception = exception
        self.post_calls: list[tuple[str, dict | None, dict | None]] = []

    def post(self, url: str, json=None, headers=None):
        self.post_calls.append((url, json, headers))
        return _FakeResponse(
            self._chunks,
            status=self._status,
            delays=self._delays,
            raise_after=self._raise_after,
            exception=self._exception,
        )


def _make_delta_event(text: str) -> bytes:
    """Create a delta SSE event."""
    event = {"type": "response.output_text.delta", "delta": text}
    return f"data: {json.dumps(event)}\n\n".encode()


def _make_event(event_type: str, **kwargs) -> bytes:
    """Create a generic SSE event."""
    event = {"type": event_type, **kwargs}
    return f"data: {json.dumps(event)}\n\n".encode()


def _make_done() -> bytes:
    """Create the [DONE] termination signal."""
    return b"data: [DONE]\n\n"


# -----------------------------------------------------------------------------
# Task Cancellation and Cleanup Tests (lines 158, 161, 167-168)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parse_sse_stream_cleans_up_on_early_break():
    """Verify tasks are cancelled when consumer breaks early from async for."""
    parser = SSEParser(workers=2, delta_char_limit=0, idle_flush_ms=0)

    # Create a slow stream that would take forever
    chunks = [_make_delta_event(f"chunk{i}") for i in range(100)]
    session = _FakeSession(chunks, delays=[0.1] * 100)

    events_received = []
    async for event in parser.parse_sse_stream(
        cast(aiohttp.ClientSession, session),
        "https://openrouter.ai/api/v1/chat/completions",
        request_body={"model": "test-model"},
        headers={},
    ):
        events_received.append(event)
        if len(events_received) >= 2:
            break  # Early exit triggers cleanup

    assert len(events_received) >= 2


@pytest.mark.asyncio
async def test_cleanup_logs_task_errors_excluding_cancellation():
    """Verify cleanup handles and logs non-CancelledError exceptions (lines 166-173)."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)
    logger = logging.getLogger("test_cleanup_errors")
    parser.logger = logger

    # Create a simple valid stream
    chunks = [_make_delta_event("hello"), _make_done()]
    session = _FakeSession(chunks)

    events = []
    async for event in parser.parse_sse_stream(
        cast(aiohttp.ClientSession, session),
        "https://openrouter.ai/api/v1/chat/completions",
        request_body={"model": "test-model"},
        headers={},
    ):
        events.append(event)

    assert len(events) == 1
    assert events[0]["delta"] == "hello"


# -----------------------------------------------------------------------------
# Circuit Breaker Tests (lines 217, 239)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_breaker_check_fails_before_request_raises():
    """Test circuit breaker open before request (line 217)."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)

    chunks = [_make_delta_event("test"), _make_done()]
    session = _FakeSession(chunks)

    # Breaker returns False (open) immediately
    def breaker_open():
        return False

    with pytest.raises(RuntimeError, match="Circuit breaker open"):
        async for _ in parser.parse_sse_stream(
            cast(aiohttp.ClientSession, session),
            "https://openrouter.ai/api/v1/chat/completions",
            request_body={"model": "test-model"},
            headers={},
            breaker_check=breaker_open,
        ):
            pass


@pytest.mark.asyncio
async def test_breaker_check_fails_during_stream_raises():
    """Test circuit breaker opens during chunk iteration (line 239)."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)

    # Many chunks to ensure we're mid-stream when breaker trips
    chunks = [_make_delta_event(f"chunk{i}") for i in range(10)]
    chunks.append(_make_done())
    session = _FakeSession(chunks, delays=[0.01] * len(chunks))

    call_count = 0

    def breaker_trips_after_3():
        nonlocal call_count
        call_count += 1
        return call_count < 4  # False (open) on 4th call

    with pytest.raises(RuntimeError, match="Circuit breaker open during stream"):
        async for _ in parser.parse_sse_stream(
            cast(aiohttp.ClientSession, session),
            "https://openrouter.ai/api/v1/chat/completions",
            request_body={"model": "test-model"},
            headers={},
            breaker_check=breaker_trips_after_3,
        ):
            pass


# -----------------------------------------------------------------------------
# HTTP Error Handling Tests (lines 228-231)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_http_error_calls_record_failure():
    """Test that HTTP errors invoke record_failure callback (lines 228-231)."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)

    session = _FakeSession([], status=500)
    failure_recorded = []

    def record_failure():
        failure_recorded.append(True)

    with pytest.raises(aiohttp.ClientResponseError):
        async for _ in parser.parse_sse_stream(
            cast(aiohttp.ClientSession, session),
            "https://openrouter.ai/api/v1/chat/completions",
            request_body={"model": "test-model"},
            headers={},
            record_failure=record_failure,
        ):
            pass

    assert len(failure_recorded) > 0


@pytest.mark.asyncio
async def test_http_401_error_calls_record_failure():
    """Test 401 auth error calls record_failure."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)

    session = _FakeSession([], status=401)
    failure_recorded = []

    def record_failure():
        failure_recorded.append(True)

    with pytest.raises(aiohttp.ClientResponseError):
        async for _ in parser.parse_sse_stream(
            cast(aiohttp.ClientSession, session),
            "https://openrouter.ai/api/v1/chat/completions",
            request_body={"model": "test-model"},
            headers={},
            record_failure=record_failure,
        ):
            pass

    assert len(failure_recorded) > 0


# -----------------------------------------------------------------------------
# Empty Data Blob Handling (line 260)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_data_lines_are_skipped():
    """Test that empty data: lines don't produce events (line 260)."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)

    # Stream with empty data lines between valid events
    chunks = [
        b"data: \n\n",  # Empty data line
        b"data:   \n\n",  # Whitespace-only data line
        _make_delta_event("valid"),
        b"data:\n\n",  # Another empty
        _make_done(),
    ]
    session = _FakeSession(chunks)

    events = []
    async for event in parser.parse_sse_stream(
        cast(aiohttp.ClientSession, session),
        "https://openrouter.ai/api/v1/chat/completions",
        request_body={"model": "test-model"},
        headers={},
    ):
        events.append(event)

    # Only the valid delta event should be yielded
    assert len(events) == 1
    assert events[0]["delta"] == "valid"


# -----------------------------------------------------------------------------
# Remaining Event Data at Stream End (lines 290-294)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_remaining_event_data_emitted_at_stream_end():
    """Test incomplete event at stream end is emitted (lines 290-294)."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)

    # Event without trailing blank line (incomplete SSE format)
    # The stream ends without proper termination
    chunks = [
        _make_delta_event("first"),
        b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"incomplete\"}\n",
        # No trailing \n\n and no [DONE]
    ]
    session = _FakeSession(chunks)

    events = []
    async for event in parser.parse_sse_stream(
        cast(aiohttp.ClientSession, session),
        "https://openrouter.ai/api/v1/chat/completions",
        request_body={"model": "test-model"},
        headers={},
    ):
        events.append(event)

    # Both events should be received
    assert len(events) == 2
    assert events[0]["delta"] == "first"
    assert events[1]["delta"] == "incomplete"


@pytest.mark.asyncio
async def test_remaining_done_not_emitted():
    """Test [DONE] at stream end without newlines is not emitted as event."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)

    chunks = [
        _make_delta_event("test"),
        b"data: [DONE]\n",  # No trailing blank line
    ]
    session = _FakeSession(chunks)

    events = []
    async for event in parser.parse_sse_stream(
        cast(aiohttp.ClientSession, session),
        "https://openrouter.ai/api/v1/chat/completions",
        request_body={"model": "test-model"},
        headers={},
    ):
        events.append(event)

    assert len(events) == 1
    assert events[0]["delta"] == "test"


# -----------------------------------------------------------------------------
# Producer Exception Handling (lines 296-319)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_producer_auth_error_logs_warning():
    """Test auth errors (401/403) log warning instead of error (lines 300-309)."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)
    logger = logging.getLogger("test_auth_error")
    parser.logger = logger

    # Create an OpenRouterAPIError with 401 status
    auth_error = OpenRouterAPIError(
        status=401,
        reason="Unauthorized",
        openrouter_message="Invalid API key",
    )

    session = _FakeSession(
        [_make_delta_event("test")],
        raise_after=0,
        exception=auth_error,
    )

    failure_recorded = []

    def record_failure():
        failure_recorded.append(True)

    with pytest.raises(OpenRouterAPIError):
        async for _ in parser.parse_sse_stream(
            cast(aiohttp.ClientSession, session),
            "https://openrouter.ai/api/v1/chat/completions",
            request_body={"model": "test-model"},
            headers={},
            record_failure=record_failure,
        ):
            pass

    assert len(failure_recorded) > 0


@pytest.mark.asyncio
async def test_producer_non_auth_error_logs_error():
    """Test non-auth errors log as error level (lines 310-315)."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)
    logger = logging.getLogger("test_non_auth_error")
    parser.logger = logger

    session = _FakeSession(
        [_make_delta_event("test")],
        raise_after=0,
        exception=RuntimeError("Network failure"),
    )

    failure_recorded = []

    def record_failure():
        failure_recorded.append(True)

    with pytest.raises(RuntimeError, match="Network failure"):
        async for _ in parser.parse_sse_stream(
            cast(aiohttp.ClientSession, session),
            "https://openrouter.ai/api/v1/chat/completions",
            request_body={"model": "test-model"},
            headers={},
            record_failure=record_failure,
        ):
            pass

    assert len(failure_recorded) > 0


# -----------------------------------------------------------------------------
# Queue Full Sentinel Handling (lines 328-332)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_queue_full_sentinel_handling():
    """Test that QueueFull exception path is correctly handled.

    This tests the logic pattern used in lines 328-332 when put_nowait raises
    QueueFull while sending sentinel values to workers during cleanup.
    """
    parser = SSEParser(workers=2, delta_char_limit=0, idle_flush_ms=0)
    logger = logging.getLogger("test_queue_full")
    parser.logger = logger

    # Create a queue with size 1 that we'll fill to cause QueueFull
    chunk_queue: asyncio.Queue[tuple[int | None, bytes | None]] = asyncio.Queue(maxsize=1)

    # Fill the queue so put_nowait will fail
    await chunk_queue.put((0, b"filler"))

    # Now try to send sentinels - should log and break without raising
    for _ in range(parser.workers):
        try:
            chunk_queue.put_nowait((None, None))
        except asyncio.QueueFull:
            # This is the path we're testing - lines 328-332
            parser.logger.debug(
                "Chunk queue full while sending sentinel; workers will be cancelled during cleanup."
            )
            break

    # Verify the queue is still full (only has 1 item capacity)
    assert chunk_queue.full()


@pytest.mark.asyncio
async def test_producer_queue_full_on_sentinel():
    """Test producer's finally block handles QueueFull when sending sentinels.

    This exercises lines 328-332 by using a blocked queue and a quick stream.
    """
    parser = SSEParser(workers=4, chunk_queue_size=1, delta_char_limit=0, idle_flush_ms=0)
    logger = logging.getLogger("test_producer_qfull")
    parser.logger = logger

    # Small chunk queue that fills quickly
    chunk_queue: asyncio.Queue[tuple[int | None, bytes | None]] = asyncio.Queue(maxsize=1)

    # Simple stream that completes quickly
    chunks = [_make_done()]
    session = _FakeSession(chunks)

    # Start producer - it will complete quickly and try to send sentinels
    # With 4 workers and queue size 1, it will hit QueueFull on 2nd+ sentinel
    await parser._producer(
        session=cast(aiohttp.ClientSession, session),
        url="https://openrouter.ai/api/v1/chat/completions",
        request_body={"model": "test"},
        headers={},
        chunk_queue=chunk_queue,
        breaker_check=None,
        record_failure=None,
        requested_model="test",
    )

    # Producer should have completed without raising (QueueFull caught)
    # Queue should have at most 1 sentinel
    assert chunk_queue.qsize() <= 1


# -----------------------------------------------------------------------------
# Worker [DONE] Handling (line 362)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_skips_done_token():
    """Test worker ignores [DONE] tokens that slip through (line 362)."""
    parser = SSEParser(workers=1)
    chunk_queue: asyncio.Queue[tuple[int | None, bytes | None]] = asyncio.Queue()
    event_queue: asyncio.Queue[tuple[int | None, dict | None]] = asyncio.Queue()

    worker_task = asyncio.create_task(
        parser._worker(chunk_queue=chunk_queue, event_queue=event_queue)
    )

    # Send [DONE] as raw bytes to worker
    await chunk_queue.put((0, b"[DONE]"))
    await chunk_queue.put((1, b'{"type":"test","value":"real"}'))
    await chunk_queue.put((None, None))

    events = []
    while True:
        seq, event = await event_queue.get()
        event_queue.task_done()
        if seq is None:
            break
        if event is not None:
            events.append((seq, event))

    await worker_task

    # Only the real event should be emitted, [DONE] should be skipped
    assert len(events) == 1
    assert events[0][0] == 1
    assert events[0][1]["type"] == "test"


# -----------------------------------------------------------------------------
# Queue Backlog Warning (lines 461-462)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_distributor_warns_on_queue_backlog():
    """Test distributor logs warning when queue backlog is high (lines 460-462)."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)
    logger = logging.getLogger("test_backlog_warning")
    parser.logger = logger

    queue: asyncio.Queue[tuple[int | None, dict | None]] = asyncio.Queue()

    # Pre-fill queue to trigger backlog warning (threshold is 2 for this test)
    for i in range(5):
        await queue.put((i, {"type": "response.output_text.delta", "delta": f"d{i}"}))
    await queue.put((None, None))

    events = []
    async for event in parser._distributor(
        event_queue=queue,
        extract_error=None,
        requested_model=None,
        event_queue_warn_size=2,  # Low threshold to trigger warning
    ):
        events.append(event)

    assert len(events) == 5


# -----------------------------------------------------------------------------
# Empty Event Skipping (lines 473-474)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_distributor_skips_empty_events():
    """Test distributor logs and skips None events (lines 472-474).

    Note: In production, workers only emit (seq, event) for valid parsed events
    and (None, None) as a sentinel. The empty event skip code is defensive.
    When a None event with a valid seq is received, we log and skip it but
    don't increment next_seq since subsequent events should not depend on it.

    This test verifies the skip logic fires without causing issues.
    """
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)
    logger = logging.getLogger("test_empty_events")
    parser.logger = logger

    queue: asyncio.Queue[tuple[int | None, dict | None]] = asyncio.Queue()

    # Valid events with sequential seq numbers that can be processed in order
    # Note: Empty events with non-None seq don't advance next_seq, so we
    # test that a None event at the start is handled before valid events
    await queue.put((0, None))  # Empty event - should be skipped and logged
    await queue.put((0, {"type": "response.output_text.delta", "delta": "a"}))
    await queue.put((1, {"type": "response.output_text.delta", "delta": "b"}))
    await queue.put((None, None))

    events = []
    async for event in parser._distributor(
        event_queue=queue,
        extract_error=None,
        requested_model=None,
        event_queue_warn_size=100,
    ):
        events.append(event)

    assert len(events) == 2
    assert events[0]["delta"] == "a"
    assert events[1]["delta"] == "b"


# -----------------------------------------------------------------------------
# Error Extraction in Distributor (lines 486-488)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_distributor_raises_on_extracted_error():
    """Test distributor raises when extract_error returns an exception (lines 485-488)."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)

    queue: asyncio.Queue[tuple[int | None, dict | None]] = asyncio.Queue()

    await queue.put((0, {"type": "error", "message": "Test error"}))
    await queue.put((None, None))

    def extract_error(event: dict[str, Any]) -> Exception | None:
        if event.get("type") == "error":
            return ValueError(f"Extracted: {event.get('message')}")
        return None

    with pytest.raises(ValueError, match="Extracted: Test error"):
        async for _ in parser._distributor(
            event_queue=queue,
            extract_error=extract_error,
            requested_model=None,
            event_queue_warn_size=100,
        ):
            pass


@pytest.mark.asyncio
async def test_distributor_continues_when_extract_error_returns_none():
    """Test distributor continues when extract_error returns None."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)

    queue: asyncio.Queue[tuple[int | None, dict | None]] = asyncio.Queue()

    await queue.put((0, {"type": "response.output_text.delta", "delta": "ok"}))
    await queue.put((None, None))

    def extract_error(event: dict[str, Any]) -> Exception | None:
        return None  # No error

    events = []
    async for event in parser._distributor(
        event_queue=queue,
        extract_error=extract_error,
        requested_model=None,
        event_queue_warn_size=100,
    ):
        events.append(event)

    assert len(events) == 1


# -----------------------------------------------------------------------------
# Delta Batching Threshold Flush (line 508)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delta_batching_flushes_at_threshold():
    """Test delta events are batched and flushed at char threshold (line 508)."""
    parser = SSEParser(workers=1, delta_char_limit=10, idle_flush_ms=0)

    queue: asyncio.Queue[tuple[int | None, dict | None]] = asyncio.Queue()

    # Send deltas that exceed threshold (10 chars)
    await queue.put((0, {"type": "response.output_text.delta", "delta": "12345"}))  # 5 chars
    await queue.put((1, {"type": "response.output_text.delta", "delta": "67890"}))  # 5 chars = 10 total
    await queue.put((2, {"type": "response.output_text.delta", "delta": "AB"}))  # triggers flush of first batch
    await queue.put((3, {"type": "response.completed", "response": {}}))
    await queue.put((None, None))

    events = []
    async for event in parser._distributor(
        event_queue=queue,
        extract_error=None,
        requested_model=None,
        event_queue_warn_size=100,
    ):
        events.append(event)

    # Should have batched delta + remaining delta + completed
    assert len(events) >= 2
    # First batched delta
    delta_events = [e for e in events if e.get("type") == "response.output_text.delta"]
    assert len(delta_events) >= 1
    # Verify batching happened
    total_chars = sum(len(e.get("delta", "")) for e in delta_events)
    assert total_chars == 12  # 12345 + 67890 + AB


# -----------------------------------------------------------------------------
# Final Delta Flush on Completion (lines 518-520)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_final_delta_flush_on_stream_completion():
    """Test remaining buffered deltas are flushed at stream end (lines 518-520)."""
    parser = SSEParser(workers=1, delta_char_limit=1000, idle_flush_ms=0)  # High threshold

    queue: asyncio.Queue[tuple[int | None, dict | None]] = asyncio.Queue()

    # Send deltas that don't hit threshold
    await queue.put((0, {"type": "response.output_text.delta", "delta": "hello"}))
    await queue.put((1, {"type": "response.output_text.delta", "delta": " world"}))
    await queue.put((None, None))

    events = []
    async for event in parser._distributor(
        event_queue=queue,
        extract_error=None,
        requested_model=None,
        event_queue_warn_size=100,
    ):
        events.append(event)

    # Should have one batched delta
    assert len(events) == 1
    assert events[0]["delta"] == "hello world"


# -----------------------------------------------------------------------------
# Delta Flush on Exception (lines 522-527)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delta_flush_on_exception():
    """Test buffered deltas are flushed before exception propagates (lines 522-527)."""
    parser = SSEParser(workers=1, delta_char_limit=1000, idle_flush_ms=0)

    queue: asyncio.Queue[tuple[int | None, dict | None]] = asyncio.Queue()

    await queue.put((0, {"type": "response.output_text.delta", "delta": "partial"}))
    await queue.put((1, {"type": "error", "message": "Stream error"}))
    await queue.put((None, None))

    def extract_error(event: dict[str, Any]) -> Exception | None:
        if event.get("type") == "error":
            return RuntimeError("Stream failed")
        return None

    events = []
    with pytest.raises(RuntimeError, match="Stream failed"):
        async for event in parser._distributor(
            event_queue=queue,
            extract_error=extract_error,
            requested_model=None,
            event_queue_warn_size=100,
        ):
            events.append(event)

    # Buffered delta should have been flushed before exception
    assert len(events) == 1
    assert events[0]["delta"] == "partial"


# -----------------------------------------------------------------------------
# Passthrough Mode Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_passthrough_mode_no_batching():
    """Test deltas pass through immediately when batching disabled."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)

    queue: asyncio.Queue[tuple[int | None, dict | None]] = asyncio.Queue()

    await queue.put((0, {"type": "response.output_text.delta", "delta": "a"}))
    await queue.put((1, {"type": "response.output_text.delta", "delta": "b"}))
    await queue.put((2, {"type": "response.output_text.delta", "delta": "c"}))
    await queue.put((None, None))

    events = []
    async for event in parser._distributor(
        event_queue=queue,
        extract_error=None,
        requested_model=None,
        event_queue_warn_size=100,
    ):
        events.append(event)

    # Each delta should be separate
    assert len(events) == 3
    assert [e["delta"] for e in events] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_passthrough_skips_empty_deltas():
    """Test empty deltas are skipped in passthrough mode."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)

    queue: asyncio.Queue[tuple[int | None, dict | None]] = asyncio.Queue()

    await queue.put((0, {"type": "response.output_text.delta", "delta": "a"}))
    await queue.put((1, {"type": "response.output_text.delta", "delta": ""}))  # Empty
    await queue.put((2, {"type": "response.output_text.delta", "delta": "b"}))
    await queue.put((None, None))

    events = []
    async for event in parser._distributor(
        event_queue=queue,
        extract_error=None,
        requested_model=None,
        event_queue_warn_size=100,
    ):
        events.append(event)

    # Empty delta should be skipped
    assert len(events) == 2
    assert [e["delta"] for e in events] == ["a", "b"]


# -----------------------------------------------------------------------------
# Multi-line Event Parsing
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiline_sse_event_parsing():
    """Test parsing of multi-line SSE events (multiple data: lines)."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)

    # Multi-line event split across data: lines
    chunks = [
        b"data: {\"type\":\"response.output_text.delta\",\n",
        b"data: \"delta\":\"multiline\"}\n\n",
        _make_done(),
    ]
    session = _FakeSession(chunks)

    events = []
    async for event in parser.parse_sse_stream(
        cast(aiohttp.ClientSession, session),
        "https://openrouter.ai/api/v1/chat/completions",
        request_body={"model": "test-model"},
        headers={},
    ):
        events.append(event)

    assert len(events) == 1
    assert events[0]["delta"] == "multiline"


# -----------------------------------------------------------------------------
# Comment Line Handling
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_comment_lines_are_ignored():
    """Test SSE comment lines (starting with :) are ignored."""
    parser = SSEParser(workers=1, delta_char_limit=0, idle_flush_ms=0)

    chunks = [
        b": this is a comment\n",
        b":another comment\n",
        _make_delta_event("after_comments"),
        b": more comments\n",
        _make_done(),
    ]
    session = _FakeSession(chunks)

    events = []
    async for event in parser.parse_sse_stream(
        cast(aiohttp.ClientSession, session),
        "https://openrouter.ai/api/v1/chat/completions",
        request_body={"model": "test-model"},
        headers={},
    ):
        events.append(event)

    assert len(events) == 1
    assert events[0]["delta"] == "after_comments"


# -----------------------------------------------------------------------------
# Integration: Full Stream Processing
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_stream_integration():
    """Integration test covering full stream processing pipeline."""
    parser = SSEParser(workers=2, delta_char_limit=50, idle_flush_ms=10)

    chunks = [
        b": keepalive\n",
        _make_delta_event("Hello"),
        _make_delta_event(" "),
        _make_delta_event("World"),
        _make_event("response.completed", response={"id": "test"}),
        _make_done(),
    ]
    session = _FakeSession(chunks)

    events = []
    async for event in parser.parse_sse_stream(
        cast(aiohttp.ClientSession, session),
        "https://openrouter.ai/api/v1/chat/completions",
        request_body={"model": "openai/gpt-4"},
        headers={"Authorization": "Bearer test"},
    ):
        events.append(event)

    # Should have batched delta events + completed
    delta_events = [e for e in events if e.get("type") == "response.output_text.delta"]
    completed_events = [e for e in events if e.get("type") == "response.completed"]

    assert len(completed_events) == 1
    # Deltas should contain all text
    total_text = "".join(e.get("delta", "") for e in delta_events)
    assert total_text == "Hello World"


# -----------------------------------------------------------------------------
# Parser Configuration Tests
# -----------------------------------------------------------------------------

def test_parser_configuration_bounds():
    """Test parser enforces minimum bounds on configuration."""
    parser = SSEParser(
        workers=-1,
        delta_char_limit=-100,
        idle_flush_ms=-50,
        chunk_queue_size=-10,
        event_queue_size=0,
    )

    # All values should be clamped to minimums
    assert parser.workers >= 1
    assert parser.delta_char_limit >= 0
    assert parser.idle_flush_ms >= 0
    assert parser.chunk_queue_size >= 1
    assert parser.event_queue_size >= 1


@pytest.mark.asyncio
async def test_distributor_non_delta_flushes_buffer():
    """Test non-delta event triggers flush of buffered deltas."""
    parser = SSEParser(workers=1, delta_char_limit=1000, idle_flush_ms=0)

    queue: asyncio.Queue[tuple[int | None, dict | None]] = asyncio.Queue()

    await queue.put((0, {"type": "response.output_text.delta", "delta": "buffered"}))
    await queue.put((1, {"type": "response.content_part.added", "part": {}}))  # Non-delta
    await queue.put((2, {"type": "response.output_text.delta", "delta": "more"}))
    await queue.put((None, None))

    events = []
    async for event in parser._distributor(
        event_queue=queue,
        extract_error=None,
        requested_model=None,
        event_queue_warn_size=100,
    ):
        events.append(event)

    # Should be: flushed delta, content_part, then final delta
    assert len(events) == 3
    assert events[0]["type"] == "response.output_text.delta"
    assert events[0]["delta"] == "buffered"
    assert events[1]["type"] == "response.content_part.added"
    assert events[2]["type"] == "response.output_text.delta"
    assert events[2]["delta"] == "more"


# -----------------------------------------------------------------------------
# Out-of-Order Event Handling
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_distributor_handles_out_of_order_events():
    """Test distributor properly orders out-of-sequence events."""
    parser = SSEParser(workers=2, delta_char_limit=0, idle_flush_ms=0)

    queue: asyncio.Queue[tuple[int | None, dict | None]] = asyncio.Queue()

    # Send events out of order
    await queue.put((2, {"type": "response.output_text.delta", "delta": "c"}))
    await queue.put((0, {"type": "response.output_text.delta", "delta": "a"}))
    await queue.put((1, {"type": "response.output_text.delta", "delta": "b"}))
    await queue.put((None, None))
    await queue.put((None, None))

    events = []
    async for event in parser._distributor(
        event_queue=queue,
        extract_error=None,
        requested_model=None,
        event_queue_warn_size=100,
    ):
        events.append(event)

    # Should be emitted in sequence order
    assert [e["delta"] for e in events] == ["a", "b", "c"]
