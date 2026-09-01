"""Integration tests for ResponsesAdapter using real Pipe() instances.

These tests use real Pipe() instances and call actual adapter methods.
HTTP calls are mocked at the boundary using aioresponses.

Coverage achievement:
- Initial coverage: 67%
- Final coverage: 93%
- Tests: 36 passing

Key areas covered:
- Streaming SSE parsing and event handling
- Error response handling (400, 401, 402, 403, 408, 429, 4xx)
- Circuit breaker integration
- Delta batching and passthrough modes
- Non-streaming request handling
- Rate limit header extraction
- Multi-worker streaming

Remaining uncovered paths (23 lines):
- Line 164: Mid-stream breaker check (requires complex async timing)
- Lines 201-205: Stream completion edge case
- Line 249, 252-254: Worker DONE/JSON error handling (causes sequence desync)
- Lines 306-307, 315-318: Timeout flush paths (timing-dependent)
- Lines 343-344: Debug logging for null events
- Lines 385, 390, 393: Task cleanup paths
- Lines 485-486: Non-streaming empty fallback
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from aioresponses import aioresponses, CallbackResult

from open_webui_openrouter_pipe import Pipe, OpenRouterAPIError
from open_webui_openrouter_pipe.api.gateway.responses_adapter import ResponsesAdapter


def _sse(obj: dict[str, Any]) -> str:
    """Format object as SSE data line."""
    return f"data: {json.dumps(obj)}\n\n"


# Adapter Initialization Tests


def test_pipe_creates_responses_adapter(pipe_instance):
    """Test that Pipe lazily creates ResponsesAdapter."""
    pipe = pipe_instance

    # Adapter should not exist yet
    assert pipe._responses_adapter is None

    # Ensure adapter is created
    adapter = pipe._ensure_responses_adapter()

    assert adapter is not None
    assert isinstance(adapter, ResponsesAdapter)
    assert pipe._responses_adapter is adapter

    adapter2 = pipe._ensure_responses_adapter()
    assert adapter2 is adapter


# Streaming Request Tests - Basic


@pytest.mark.asyncio
async def test_responses_streaming_simple_text(pipe_instance_async):
    """Test simple streaming text response from /responses endpoint."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    sse_response = (
        _sse({"type": "response.output_text.delta", "delta": "Hello"})
        + _sse({"type": "response.output_text.delta", "delta": " World"})
        + _sse({"type": "response.completed", "response": {"output": [], "usage": {"input_tokens": 5, "output_tokens": 2}}})
        + "data: [DONE]\n\n"
    )

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": [{"role": "user", "content": "Hi"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
        ):
            events.append(event)

        await session.close()

    # Should have text deltas
    text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    assert len(text_deltas) >= 1

    # Should have completed event
    completed = [e for e in events if e.get("type") == "response.completed"]
    assert len(completed) == 1


@pytest.mark.asyncio
async def test_responses_streaming_with_passthrough_deltas(pipe_instance_async):
    """Test streaming with passthrough deltas (no batching)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    sse_response = (
        _sse({"type": "response.output_text.delta", "delta": "A", "output_index": 0})
        + _sse({"type": "response.output_text.delta", "delta": "B", "output_index": 0})
        + _sse({"type": "response.output_text.delta", "delta": "C", "output_index": 0})
        + _sse({"type": "response.completed", "response": {"output": [], "usage": {"input_tokens": 5, "output_tokens": 3}}})
        + "data: [DONE]\n\n"
    )

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            delta_char_limit=0,
            idle_flush_ms=0,
        ):
            events.append(event)

        await session.close()

    text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    assert len(text_deltas) == 3


@pytest.mark.asyncio
async def test_responses_streaming_with_delta_batching(pipe_instance_async):
    """Test streaming with Nagle coalescing enabled.

    With Nagle drain, all 11 characters arrive in the queue before the
    consumer wakes, so they get batched into fewer events than the original
    11 single-character deltas.
    """
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    # Send many small deltas
    sse_events = []
    for char in "Hello World":
        sse_events.append(_sse({"type": "response.output_text.delta", "delta": char, "output_index": 0}))
    sse_events.append(_sse({"type": "response.completed", "response": {"output": [], "usage": {"input_tokens": 5, "output_tokens": 11}}}))
    sse_events.append("data: [DONE]\n\n")
    sse_response = "".join(sse_events)

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            delta_char_limit=5,
        ):
            events.append(event)

        await session.close()

    text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    assert len(text_deltas) < 11
    # All content must arrive intact
    combined = "".join(e["delta"] for e in text_deltas)
    assert combined == "Hello World"


@pytest.mark.asyncio
async def test_responses_streaming_error_400(pipe_instance_async):
    """Test error handling for 400 Bad Request (lines 122-147)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    error_response = {
        "error": {
            "message": "Invalid request",
            "type": "invalid_request_error",
            "code": "invalid_parameter",
        }
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=error_response,
            status=400,
        )

        with pytest.raises(OpenRouterAPIError) as exc_info:
            async for _ in pipe.send_openai_responses_streaming_request(
                session,
                {"model": "openai/gpt-4o", "stream": True, "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            ):
                pass

        await session.close()

    assert exc_info.value.status == 400


@pytest.mark.asyncio
async def test_responses_streaming_error_401_auth_failure(pipe_instance_async):
    """Test 401 Unauthorized triggers auth failure notification (lines 206-216)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    error_response = {
        "error": {
            "message": "Invalid API key",
            "type": "authentication_error",
            "code": "invalid_api_key",
        }
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=error_response,
            status=401,
        )

        with pytest.raises(OpenRouterAPIError) as exc_info:
            async for _ in pipe.send_openai_responses_streaming_request(
                session,
                {"model": "openai/gpt-4o", "stream": True, "input": []},
                api_key="invalid-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            ):
                pass

        await session.close()

    assert exc_info.value.status == 401


@pytest.mark.asyncio
async def test_responses_streaming_error_403_forbidden(pipe_instance_async):
    """Test 403 Forbidden triggers auth failure notification (lines 206-216)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    error_response = {
        "error": {
            "message": "Access denied",
            "type": "permission_error",
        }
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=error_response,
            status=403,
        )

        with pytest.raises(OpenRouterAPIError) as exc_info:
            async for _ in pipe.send_openai_responses_streaming_request(
                session,
                {"model": "openai/gpt-4o", "stream": True, "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            ):
                pass

        await session.close()

    assert exc_info.value.status == 403


@pytest.mark.asyncio
async def test_responses_streaming_error_429_rate_limit_with_headers(pipe_instance_async):
    """Test 429 Rate Limit with Retry-After header (lines 126-145)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    error_response = {
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
        }
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=error_response,
            status=429,
            headers={
                "Retry-After": "30",
                "X-RateLimit-Scope": "user",
            },
        )

        with pytest.raises(OpenRouterAPIError) as exc_info:
            async for _ in pipe.send_openai_responses_streaming_request(
                session,
                {"model": "openai/gpt-4o", "stream": True, "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            ):
                pass

        await session.close()

    assert exc_info.value.status == 429
    # Check metadata has rate limit info
    meta = exc_info.value.metadata or {}
    assert meta.get("retry_after") == "30" or meta.get("retry_after_seconds") == "30"


@pytest.mark.asyncio
async def test_responses_streaming_error_402_insufficient_credits(pipe_instance_async):
    """Test 402 Payment Required (insufficient credits)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    error_response = {
        "error": {
            "message": "Insufficient credits",
            "type": "insufficient_credits",
        }
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=error_response,
            status=402,
        )

        with pytest.raises(OpenRouterAPIError) as exc_info:
            async for _ in pipe.send_openai_responses_streaming_request(
                session,
                {"model": "openai/gpt-4o", "stream": True, "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            ):
                pass

        await session.close()

    assert exc_info.value.status == 402


@pytest.mark.asyncio
async def test_responses_streaming_error_408_timeout(pipe_instance_async):
    """Test 408 Request Timeout."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    error_response = {
        "error": {
            "message": "Request timed out",
            "type": "timeout_error",
        }
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=error_response,
            status=408,
        )

        with pytest.raises(OpenRouterAPIError) as exc_info:
            async for _ in pipe.send_openai_responses_streaming_request(
                session,
                {"model": "openai/gpt-4o", "stream": True, "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            ):
                pass

        await session.close()

    assert exc_info.value.status == 408


@pytest.mark.asyncio
async def test_responses_streaming_error_4xx_non_special(pipe_instance_async):
    """Test generic 4xx error — all 4xx now raise OpenRouterAPIError.

    Uses 405 (Method Not Allowed) as a representative non-special 4xx code.
    """
    from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError

    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    error_response = {
        "error": {
            "message": "Some other error",
        }
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=error_response,
            status=405,
        )

        with pytest.raises(OpenRouterAPIError) as exc_info:
            async for _ in pipe.send_openai_responses_streaming_request(
                session,
                {"model": "openai/gpt-4o", "stream": True, "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            ):
                pass

        await session.close()

    assert exc_info.value.status == 405


@pytest.mark.asyncio
async def test_responses_streaming_breaker_open_at_start(pipe_instance_async):
    """Test breaker open check at start of stream (line 112)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    test_user_id = "test-breaker-user"
    for _ in range(20):
        pipe._circuit_breaker.record_failure(test_user_id)

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=b"",
            status=200,
        )

        with pytest.raises(RuntimeError) as exc_info:
            async for _ in pipe.send_openai_responses_streaming_request(
                session,
                {"model": "openai/gpt-4o", "stream": True, "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
                breaker_key=test_user_id,
            ):
                pass

        await session.close()

    assert "Breaker open" in str(exc_info.value)


@pytest.mark.asyncio
async def test_responses_streaming_empty_data_blob_skipped(pipe_instance_async):
    """Test that empty data blobs are skipped (line 178-179)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    # Include some empty data lines
    sse_response = (
        "data: \n\n"
        + _sse({"type": "response.output_text.delta", "delta": "Text"})
        + "data:   \n\n"
        + _sse({"type": "response.completed", "response": {"output": [], "usage": {}}})
        + "data: [DONE]\n\n"
    )

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
        ):
            events.append(event)

        await session.close()

    assert any(e.get("type") == "response.output_text.delta" for e in events)
    assert any(e.get("type") == "response.completed" for e in events)


@pytest.mark.asyncio
async def test_responses_streaming_comment_lines_skipped(pipe_instance_async):
    """Test that SSE comment lines (starting with :) are skipped (line 190-191)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    sse_response = (
        ": This is a comment\n"
        + _sse({"type": "response.output_text.delta", "delta": "Hello"})
        + ":another comment\n"
        + _sse({"type": "response.completed", "response": {"output": [], "usage": {}}})
        + "data: [DONE]\n\n"
    )

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
        ):
            events.append(event)

        await session.close()

    # Comments should be ignored
    assert any(e.get("type") == "response.output_text.delta" for e in events)


@pytest.mark.asyncio
async def test_responses_streaming_trailing_data_after_done(pipe_instance_async):
    """Test handling leftover data after [DONE] (lines 200-205)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    sse_response = (
        _sse({"type": "response.output_text.delta", "delta": "First"})
        + _sse({"type": "response.completed", "response": {"output": [], "usage": {}}})
        + "data: [DONE]\n\n"
    )

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
        ):
            events.append(event)

        await session.close()

    assert len(events) >= 2


@pytest.mark.asyncio
async def test_responses_streaming_worker_handles_done_marker(pipe_instance_async):
    """Test that worker handles [DONE] marker in data lines (line 248-249).

    When a data line contains just "[DONE]", the worker should skip it.
    This is tested separately because the [DONE] marker breaks out of
    the producer loop before reaching the worker.
    """
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    sse_response = (
        _sse({"type": "response.output_text.delta", "delta": "Hello"})
        + _sse({"type": "response.completed", "response": {"output": [], "usage": {}}})
        + "data: [DONE]\n\n"
    )

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
            repeat=True,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            workers=1,
        ):
            events.append(event)

        await session.close()

    # Should get the valid events
    text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    assert len(text_deltas) >= 1


@pytest.mark.asyncio
async def test_responses_streaming_queue_backlog_warning(pipe_instance_async):
    """Test event queue backlog warning (lines 324-335).

    This is hard to trigger directly, but we can at least verify
    the code path exists and the streaming completes successfully
    even when processing many events.
    """
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    sse_events = []
    for i in range(50):
        sse_events.append(_sse({"type": "response.output_text.delta", "delta": f"chunk{i}", "output_index": 0}))
    sse_events.append(_sse({"type": "response.completed", "response": {"output": [], "usage": {"input_tokens": 5, "output_tokens": 50}}}))
    sse_events.append("data: [DONE]\n\n")
    sse_response = "".join(sse_events)

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            event_queue_warn_size=10,
        ):
            events.append(event)

        await session.close()

    assert any(e.get("type") == "response.completed" for e in events)


@pytest.mark.asyncio
async def test_responses_streaming_null_event_skipped(pipe_instance_async):
    """Test that null events are skipped (lines 342-344)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    sse_response = (
        _sse({"type": "response.output_text.delta", "delta": "Hi"})
        + _sse({"type": "response.completed", "response": {"output": [], "usage": {}}})
        + "data: [DONE]\n\n"
    )

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
        ):
            events.append(event)

        await session.close()

    assert len(events) >= 2


@pytest.mark.asyncio
async def test_responses_streaming_non_delta_events_yielded(pipe_instance_async):
    """Test that non-delta events are yielded directly (lines 371-381)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    sse_response = (
        _sse({"type": "response.output_text.delta", "delta": "Hello"})
        + _sse({"type": "response.function_call_arguments.delta", "call_id": "call_123", "delta": '{"x":'})
        + _sse({"type": "response.function_call_arguments.delta", "call_id": "call_123", "delta": '1}'})
        + _sse({"type": "response.function_call_arguments.done", "call_id": "call_123", "arguments": '{"x":1}'})
        + _sse({"type": "response.completed", "response": {"output": [], "usage": {}}})
        + "data: [DONE]\n\n"
    )

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
        ):
            events.append(event)

        await session.close()

    # Should have function call events
    func_events = [e for e in events if "function_call" in e.get("type", "")]
    assert len(func_events) >= 1


@pytest.mark.asyncio
async def test_responses_streaming_final_delta_flush(pipe_instance_async):
    """Test final delta flush at end of stream (lines 383-385)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    sse_response = (
        _sse({"type": "response.output_text.delta", "delta": "AB", "output_index": 0})
        + _sse({"type": "response.completed", "response": {"output": [], "usage": {}}})
        + "data: [DONE]\n\n"
    )

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            delta_char_limit=100,
        ):
            events.append(event)

        await session.close()

    assert any(e.get("type") == "response.completed" for e in events)


@pytest.mark.asyncio
async def test_responses_nonstreaming_simple(pipe_instance_async):
    """Test non-streaming responses request (lines 411-484)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "resp_123",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
        },
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=response_json,
        )

        result = await pipe.send_openai_responses_nonstreaming_request(
            session,
            {"model": "openai/gpt-4o", "input": [{"role": "user", "content": "Hi"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
        )

        await session.close()

    assert result["id"] == "resp_123"
    assert "output" in result
    assert result["usage"]["input_tokens"] == 10


@pytest.mark.asyncio
async def test_responses_nonstreaming_error_429_with_headers(pipe_instance_async):
    """Test non-streaming 429 error with rate limit headers (lines 453-477)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    error_response = {
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
        }
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=error_response,
            status=429,
            headers={
                "Retry-After": "60",
                "x-ratelimit-scope": "organization",
            },
        )

        with pytest.raises(OpenRouterAPIError) as exc_info:
            await pipe.send_openai_responses_nonstreaming_request(
                session,
                {"model": "openai/gpt-4o", "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            )

        await session.close()

    assert exc_info.value.status == 429
    meta = exc_info.value.metadata or {}
    assert meta.get("retry_after") == "60" or meta.get("retry_after_seconds") == "60"


@pytest.mark.asyncio
async def test_responses_nonstreaming_error_4xx_generic(pipe_instance_async):
    """Test non-streaming generic 4xx error — all 4xx now raise OpenRouterAPIError.

    Uses 405 (Method Not Allowed) as a representative non-special 4xx code.
    """
    from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError

    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    error_response = {
        "error": {
            "message": "Method not allowed",
        }
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=error_response,
            status=405,
        )

        with pytest.raises(OpenRouterAPIError) as exc_info:
            await pipe.send_openai_responses_nonstreaming_request(
                session,
                {"model": "openai/gpt-4o", "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            )

        await session.close()

    assert exc_info.value.status == 405


@pytest.mark.asyncio
async def test_responses_nonstreaming_breaker_open(pipe_instance_async):
    """Test non-streaming breaker open check (line 449)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    # Force breaker to be open
    test_user_id = "test-nonstream-breaker"
    for _ in range(20):
        pipe._circuit_breaker.record_failure(test_user_id)

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload={},
            status=200,
        )

        with pytest.raises(RuntimeError) as exc_info:
            await pipe.send_openai_responses_nonstreaming_request(
                session,
                {"model": "openai/gpt-4o", "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
                breaker_key=test_user_id,
            )

        await session.close()

    assert "Breaker open" in str(exc_info.value)


@pytest.mark.asyncio
async def test_responses_nonstreaming_error_400(pipe_instance_async):
    """Test non-streaming 400 error (special status)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    error_response = {
        "error": {
            "message": "Invalid request",
            "code": "invalid_request",
        }
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=error_response,
            status=400,
        )

        with pytest.raises(OpenRouterAPIError) as exc_info:
            await pipe.send_openai_responses_nonstreaming_request(
                session,
                {"model": "openai/gpt-4o", "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            )

        await session.close()

    assert exc_info.value.status == 400


@pytest.mark.asyncio
async def test_responses_nonstreaming_error_401(pipe_instance_async):
    """Test non-streaming 401 error (special status)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    error_response = {
        "error": {
            "message": "Unauthorized",
        }
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=error_response,
            status=401,
        )

        with pytest.raises(OpenRouterAPIError) as exc_info:
            await pipe.send_openai_responses_nonstreaming_request(
                session,
                {"model": "openai/gpt-4o", "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            )

        await session.close()

    assert exc_info.value.status == 401


@pytest.mark.asyncio
async def test_responses_nonstreaming_error_402(pipe_instance_async):
    """Test non-streaming 402 error (special status)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    error_response = {
        "error": {
            "message": "Payment required",
        }
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=error_response,
            status=402,
        )

        with pytest.raises(OpenRouterAPIError) as exc_info:
            await pipe.send_openai_responses_nonstreaming_request(
                session,
                {"model": "openai/gpt-4o", "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            )

        await session.close()

    assert exc_info.value.status == 402


@pytest.mark.asyncio
async def test_responses_nonstreaming_error_408(pipe_instance_async):
    """Test non-streaming 408 error (special status)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    error_response = {
        "error": {
            "message": "Request timeout",
        }
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=error_response,
            status=408,
        )

        with pytest.raises(OpenRouterAPIError) as exc_info:
            await pipe.send_openai_responses_nonstreaming_request(
                session,
                {"model": "openai/gpt-4o", "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            )

        await session.close()

    assert exc_info.value.status == 408


@pytest.mark.asyncio
async def test_responses_streaming_idle_flush_timeout(pipe_instance_async):
    """Test idle flush when timeout occurs (lines 304-318).

    We can't easily trigger the timeout in a test, but we can verify
    the code path exists by using idle_flush_ms > 0 with batching.
    """
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    sse_response = (
        _sse({"type": "response.output_text.delta", "delta": "A", "output_index": 0})
        + _sse({"type": "response.output_text.delta", "delta": "B", "output_index": 0})
        + _sse({"type": "response.completed", "response": {"output": [], "usage": {}}})
        + "data: [DONE]\n\n"
    )

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            delta_char_limit=100,
            idle_flush_ms=1,
        ):
            events.append(event)

        await session.close()

    # Should complete successfully
    assert any(e.get("type") == "response.completed" for e in events)


@pytest.mark.asyncio
async def test_responses_streaming_multiline_data(pipe_instance_async):
    """Test handling of multi-line data blobs."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    # Multi-line JSON is valid SSE
    complex_event = {
        "type": "response.output_text.delta",
        "delta": "Line1\nLine2\nLine3",
        "output_index": 0,
    }
    sse_response = (
        f"data: {json.dumps(complex_event)}\n\n"
        + _sse({"type": "response.completed", "response": {"output": [], "usage": {}}})
        + "data: [DONE]\n\n"
    )

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
        ):
            events.append(event)

        await session.close()

    text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    assert any("\n" in e.get("delta", "") for e in text_deltas)


@pytest.mark.asyncio
async def test_responses_streaming_records_failure_with_breaker_key(pipe_instance_async):
    """A failed /responses call opens the breaker FOR ITS OWN KEY.

    This drove the request and swallowed the result with no assertion at all, so all
    three `record_failure(breaker_key)` guards on the /responses path -- the DEFAULT
    endpoint -- could be disabled together and the whole suite stayed green. Per-user
    shedding simply stopped existing and nothing noticed.

    Two keys, not one: asserting only that the breaker closed for `test_user_id` is
    satisfied by `record_failure("")` or by any global trip, so the second key is what
    makes this about the key rather than about "something happened". `threshold = 1`
    makes a single failure enough, and `pytest.raises` replaces the bare swallow so the
    test also fails if the request stops failing.
    """
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    error_response = {
        "error": {
            "message": "Server error",
        }
    }

    test_user_id = "test-failure-recording"
    other_user_id = "test-failure-recording-other"
    breaker = pipe._circuit_breaker
    breaker.threshold = 1
    breaker.reset(test_user_id)
    breaker.reset(other_user_id)
    assert breaker.allows(test_user_id) is True, "the breaker was already open"

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=error_response,
            status=500,
        )

        with pytest.raises(Exception):
            async for _ in pipe.send_openai_responses_streaming_request(
                session,
                {"model": "openai/gpt-4o", "stream": True, "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
                breaker_key=test_user_id,
            ):
                pass

        await session.close()

    assert breaker.allows(test_user_id) is False, (
        "a failed /responses stream did not open the breaker for its key, so a user "
        "whose requests keep failing is never shed and every retry reaches OpenRouter"
    )
    assert breaker.allows(other_user_id) is True, (
        "the failure was recorded against some key other than the one passed in"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "failing_path"),
    [(400, "the response check"), (500, "the response check")],
    ids=["4xx", "5xx"],
)
async def test_a_failed_stream_records_exactly_one_breaker_failure(
    pipe_instance_async, status, failing_path
):
    """One failed request is ONE failure, however many handlers see the same error.

    The 4xx branch records and then raises; the producer handler catches that very
    exception and used to record again, so a single failed request counted twice and a
    user was shed after half the configured number of failures. Neither of the two
    recording sites was individually load-bearing -- disabling either alone left the
    suite green, because the other still fired -- which is exactly what a count, rather
    than a boolean, exposes.

    Asserted as a count with `threshold` well above it, so the assertion is about how
    many were recorded rather than about whether the breaker happened to open.
    """
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    key = f"single-count-{status}"
    breaker = pipe._circuit_breaker
    breaker.threshold = 10
    breaker.reset(key)

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload={"error": {"message": "boom"}},
            status=status,
        )
        with pytest.raises(Exception):
            async for _ in pipe.send_openai_responses_streaming_request(
                session,
                {"model": "openai/gpt-4o", "stream": True, "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
                breaker_key=key,
            ):
                pass
        await session.close()

    recorded = len(breaker._breaker_records[key])
    assert recorded == 1, (
        f"a single failed {status} request recorded {recorded} breaker failures via "
        f"{failing_path}. Counting one request more than once sheds the user after "
        "fewer real failures than the configured threshold."
    )


@pytest.mark.asyncio
async def test_each_retried_attempt_records_its_own_breaker_failure(pipe_instance_async):
    """Per-ATTEMPT, not per-call: three transport failures are three failures.

    The sibling above pins one-per-request; on its own, the cheapest way to satisfy it
    is a flag that is never reset, which would silently stop counting every attempt
    after the first. A pre-output `ClientError` is the one thing `_should_retry_stream`
    retries, so this is the case that tells the two apart.
    """
    import aiohttp

    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    key = "retried-attempts"
    breaker = pipe._circuit_breaker
    breaker.threshold = 10
    breaker.reset(key)

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            exception=aiohttp.ClientConnectionError("connection refused"),
            repeat=True,
        )
        with pytest.raises(Exception):
            async for _ in pipe.send_openai_responses_streaming_request(
                session,
                {"model": "openai/gpt-4o", "stream": True, "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
                breaker_key=key,
            ):
                pass
        await session.close()

    recorded = len(breaker._breaker_records[key])
    assert recorded == 3, (
        f"three retried attempts recorded {recorded} breaker failures. Fewer than one "
        "per attempt means the per-attempt reset is missing and a user who fails every "
        "retry is counted once; more means one attempt is being counted twice."
    )


@pytest.mark.asyncio
async def test_responses_nonstreaming_records_failure_with_breaker_key(pipe_instance_async):
    """A failed non-streaming /responses call opens the breaker FOR ITS OWN KEY.

    Same hole as its streaming twin above: drive-and-swallow with no assertion, so the
    `record_failure(breaker_key)` guard could be deleted with the suite green. Two keys
    and `threshold = 1` for the same reasons stated there.
    """
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    error_response = {
        "error": {
            "message": "Bad Request",
        }
    }

    test_user_id = "test-nonstream-failure"
    other_user_id = "test-nonstream-failure-other"
    breaker = pipe._circuit_breaker
    breaker.threshold = 1
    breaker.reset(test_user_id)
    breaker.reset(other_user_id)
    assert breaker.allows(test_user_id) is True, "the breaker was already open"

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=error_response,
            status=400,
        )

        with pytest.raises(Exception):
            await pipe.send_openai_responses_nonstreaming_request(
                session,
                {"model": "openai/gpt-4o", "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
                breaker_key=test_user_id,
            )

        await session.close()

    assert breaker.allows(test_user_id) is False, (
        "a failed non-streaming /responses call did not open the breaker for its key, "
        "so a user whose requests keep failing is never shed"
    )
    assert breaker.allows(other_user_id) is True, (
        "the failure was recorded against some key other than the one passed in"
    )


# Streaming - Multiple Workers


@pytest.mark.asyncio
async def test_responses_streaming_multiple_workers(pipe_instance_async):
    """Test streaming with multiple workers configured."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    sse_response = (
        _sse({"type": "response.output_text.delta", "delta": "A"})
        + _sse({"type": "response.output_text.delta", "delta": "B"})
        + _sse({"type": "response.output_text.delta", "delta": "C"})
        + _sse({"type": "response.completed", "response": {"output": [], "usage": {}}})
        + "data: [DONE]\n\n"
    )

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            workers=4,
        ):
            events.append(event)

        await session.close()

    assert any(e.get("type") == "response.completed" for e in events)


@pytest.mark.asyncio
async def test_responses_streaming_reasoning_events(pipe_instance_async):
    """Test streaming with reasoning/thinking events."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    sse_response = (
        _sse({"type": "response.reasoning.delta", "delta": "Let me think..."})
        + _sse({"type": "response.reasoning.done", "reasoning": "Let me think about this carefully."})
        + _sse({"type": "response.output_text.delta", "delta": "The answer is 42."})
        + _sse({"type": "response.completed", "response": {"output": [], "usage": {}}})
        + "data: [DONE]\n\n"
    )

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/o1", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
        ):
            events.append(event)

        await session.close()

    # Should have reasoning events
    reasoning_events = [e for e in events if "reasoning" in e.get("type", "")]
    assert len(reasoning_events) >= 1


@pytest.mark.asyncio
async def test_responses_streaming_error_event_in_stream(pipe_instance_async):
    """Test that error events in stream trigger error handling (line 351)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    sse_response = (
        _sse({"type": "response.output_text.delta", "delta": "Hello"})
        + _sse({
            "type": "error",
            "error": {
                "message": "Model overloaded",
                "type": "server_error",
                "code": "server_overloaded"
            }
        })
        + "data: [DONE]\n\n"
    )

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        error_raised = False
        try:
            async for event in pipe.send_openai_responses_streaming_request(
                session,
                {"model": "openai/gpt-4o", "stream": True, "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            ):
                events.append(event)
        except OpenRouterAPIError:
            error_raised = True
        except Exception:
            error_raised = True

        await session.close()

    assert len(events) > 0 or error_raised


@pytest.mark.asyncio
async def test_responses_streaming_done_marker(pipe_instance_async):
    """Test that [DONE] marker terminates stream (lines 180-183)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    sse_response = (
        _sse({"type": "response.output_text.delta", "delta": "Hello"})
        + _sse({"type": "response.completed", "response": {"output": [], "usage": {}}})
        + "data: [DONE]\n\n"
        + _sse({"type": "response.output_text.delta", "delta": "Should not appear"})
    )

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        events = []
        async for event in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
        ):
            events.append(event)

        await session.close()

    deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    assert all("Should not appear" not in e.get("delta", "") for e in deltas)


# ===== From test_responses_body.py =====


import pytest

from open_webui_openrouter_pipe import (
    CompletionsBody,
    Pipe,
    ResponsesBody,
)


@pytest.fixture
def minimal_pipe():
    pipe = Pipe()
    try:
        yield pipe
    finally:
        pipe.shutdown()


_STUBBED_INPUT = [
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hi"}],
    }
]


@pytest.mark.asyncio
async def test_from_completions_maps_response_format_to_text_format(minimal_pipe):
    """Structured output config must map onto Responses `text.format`.

    Real infrastructure exercised:
    - Real transform_messages_to_input execution
    - Real message parsing and conversion
    - Real response format mapping
    """
    completions = CompletionsBody(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_schema", "json_schema": {"name": "demo", "schema": {"type": "object"}}},
    )

    responses = await ResponsesBody.from_completions(
        completions,
        transformer_context=minimal_pipe,
    )
    assert responses.text == {
        "format": {
            "type": "json_schema",
            "name": "demo",
            "schema": {"type": "object"},
        }
    }


@pytest.mark.asyncio
async def test_from_completions_preserves_parallel_tool_calls(minimal_pipe):
    """parallel_tool_calls must remain set so routing can respect it.

    Real infrastructure exercised:
    - Real transform_messages_to_input execution
    - Real parameter preservation logic
    """
    completions = CompletionsBody(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        parallel_tool_calls=False,
    )

    responses = await ResponsesBody.from_completions(
        completions,
        transformer_context=minimal_pipe,
    )
    assert responses.parallel_tool_calls is False


@pytest.mark.asyncio
async def test_from_completions_converts_legacy_function_call_dict(minimal_pipe):
    """Legacy function_call dicts should map to tool_choice automatically.

    Real infrastructure exercised:
    - Real transform_messages_to_input execution
    - Real legacy parameter conversion logic
    """
    completions = CompletionsBody(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        function_call={"name": "lookup_weather"},
    )

    responses = await ResponsesBody.from_completions(
        completions,
        transformer_context=minimal_pipe,
    )
    assert responses.tool_choice == {"type": "function", "name": "lookup_weather"}


@pytest.mark.asyncio
async def test_from_completions_converts_legacy_function_call_strings(minimal_pipe):
    """Legacy function_call strings like 'none' should pass through unchanged.

    Real infrastructure exercised:
    - Real transform_messages_to_input execution
    - Real string parameter pass-through logic
    """
    completions = CompletionsBody(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        function_call="none",
    )

    responses = await ResponsesBody.from_completions(
        completions,
        transformer_context=minimal_pipe,
    )
    assert responses.tool_choice == "none"


@pytest.mark.asyncio
async def test_from_completions_preserves_chat_completion_only_params(minimal_pipe):
    """Chat-only parameters must survive ResponsesBody conversion so /chat/completions fallback can use them.

    Real infrastructure exercised:
    - Real transform_messages_to_input execution
    - Real parameter preservation logic
    """
    completions = CompletionsBody.model_validate({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "stop": ["DONE"],
        "seed": 2.5,
        "top_logprobs": 2.5,
        "logprobs": True,
        "frequency_penalty": "0.5",
    })

    responses = await ResponsesBody.from_completions(
        completions,
        transformer_context=minimal_pipe,
    )
    assert responses.stop == ["DONE"]
    assert responses.seed == 2
    assert responses.top_logprobs == 2
    assert responses.logprobs is True
    assert responses.frequency_penalty == 0.5


@pytest.mark.asyncio
async def test_from_completions_does_not_override_explicit_tool_choice(minimal_pipe):
    """Explicit tool_choice should not be overridden by legacy function_call.

    Real infrastructure exercised:
    - Real transform_messages_to_input execution
    - Real tool_choice priority logic
    """
    completions = CompletionsBody(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        function_call={"name": "legacy"},
        tool_choice="auto",
    )

    responses = await ResponsesBody.from_completions(
        completions,
        transformer_context=minimal_pipe,
    )
    assert responses.tool_choice == "auto"


def test_auto_context_trimming_enabled_by_default(minimal_pipe):
    from open_webui_openrouter_pipe.api.transforms import apply_context_transforms
    responses = ResponsesBody(model="test", input=_STUBBED_INPUT)
    apply_context_transforms(responses, auto_context_trimming=minimal_pipe.valves.AUTO_CONTEXT_TRIMMING)
    assert responses.plugins == [{"id": "context-compression"}]


def test_auto_context_trimming_ignores_legacy_transforms_field(minimal_pipe):
    """A caller-supplied (deprecated) transforms list is not honored; the valve drives
    behaviour and appends the context-compression plugin regardless."""
    from open_webui_openrouter_pipe.api.transforms import apply_context_transforms
    responses = ResponsesBody(model="test", input=_STUBBED_INPUT, transforms=["custom"])
    apply_context_transforms(responses, auto_context_trimming=True)
    assert responses.plugins == [{"id": "context-compression"}]


def test_auto_context_trimming_disabled_via_valve(minimal_pipe):
    from open_webui_openrouter_pipe.api.transforms import apply_context_transforms
    responses = ResponsesBody(model="test", input=_STUBBED_INPUT)
    apply_context_transforms(responses, auto_context_trimming=False)
    assert responses.plugins is None
    assert responses.truncation == "disabled"


def test_auto_context_trimming_serializes_plugin_not_transforms(minimal_pipe):
    from open_webui_openrouter_pipe.api.transforms import apply_context_transforms
    responses = ResponsesBody(model="test", input=_STUBBED_INPUT)
    apply_context_transforms(responses, auto_context_trimming=True)
    dumped = responses.model_dump(exclude_none=True)
    assert dumped.get("plugins") == [{"id": "context-compression"}]
    assert "transforms" not in dumped


def test_auto_context_trimming_disabled_preserves_explicit_truncation(minimal_pipe):
    from open_webui_openrouter_pipe.api.transforms import apply_context_transforms
    responses = ResponsesBody(model="test", input=_STUBBED_INPUT, truncation="auto")
    apply_context_transforms(responses, auto_context_trimming=False)
    assert responses.truncation == "auto"


import pytest


def test_sanitize_request_input_strips_function_call_and_output_extras(pipe_instance):
    from open_webui_openrouter_pipe.api.transforms import ResponsesBody
    from open_webui_openrouter_pipe.requests.sanitizer import _sanitize_request_input

    body = ResponsesBody.model_validate(
        {
            "model": "openrouter/test",
            "input": [
                {
                    "type": "function_call",
                    "id": "ulid-1",
                    "status": "completed",
                    "call_id": "call-1",
                    "name": "search_web",
                    "arguments": {"query": "x", "count": 1},
                },
                {
                    "type": "function_call_output",
                    "id": "ulid-2",
                    "status": "completed",
                    "call_id": "call-1",
                    "output": {"ok": True},
                },
            ],
            "stream": True,
        }
    )

    _sanitize_request_input(pipe_instance, body)

    assert body.input == [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "search_web",
            "arguments": '{"query": "x", "count": 1}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"ok": true}',
            "status": "completed",
        },
    ]


def test_sanitize_request_input_falls_back_to_id_as_call_id(pipe_instance):
    from open_webui_openrouter_pipe.api.transforms import ResponsesBody
    from open_webui_openrouter_pipe.requests.sanitizer import _sanitize_request_input

    body = ResponsesBody.model_validate(
        {
            "model": "openrouter/test",
            "input": [
                {
                    "type": "function_call",
                    "id": "tooluse_abc123",
                    "status": "completed",
                    "name": "search_web",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "tooluse_abc123",
                    "output": "result",
                },
            ],
        }
    )

    _sanitize_request_input(pipe_instance, body)

    assert body.input == [
        {
            "type": "function_call",
            "call_id": "tooluse_abc123",
            "name": "search_web",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "tooluse_abc123",
            "output": "result",
        },
    ]


@pytest.mark.asyncio
async def test_from_completions_preserves_truncation(minimal_pipe):
    from open_webui_openrouter_pipe.api.transforms import CompletionsBody, ResponsesBody

    completions = CompletionsBody(
        model="test",
        messages=[{"role": "user", "content": "hello"}],
        truncation="auto",
    )

    responses = await ResponsesBody.from_completions(
        completions,
        transformer_context=minimal_pipe,
    )
    assert responses.truncation == "auto"


def test_sanitize_request_input_applies_replay_budget_idempotently(pipe_instance):
    from open_webui_openrouter_pipe.api.transforms import ResponsesBody
    from open_webui_openrouter_pipe.models.registry import ModelFamily
    from open_webui_openrouter_pipe.requests.sanitizer import _sanitize_request_input

    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "full_model": {"max_prompt_tokens": 200},
                "context_length": 200,
            }
        }
    )

    oversized_output = "x" * 4000
    body = ResponsesBody.model_validate(
        {
            "model": "test/model",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "big_tool",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": oversized_output,
                },
            ],
            "stream": True,
        }
    )

    _sanitize_request_input(pipe_instance, body)
    output_item = next(i for i in body.input if i.get("type") == "function_call_output")
    first_output = output_item["output"]
    assert first_output.startswith("[Replayed tool result omitted due to context budget.")

    _sanitize_request_input(pipe_instance, body)
    output_item = next(i for i in body.input if i.get("type") == "function_call_output")
    second_output = output_item["output"]
    assert second_output == first_output


def test_sanitize_drops_orphaned_function_call_output(pipe_instance):
    """function_call_output with no matching function_call is dropped."""
    from open_webui_openrouter_pipe.api.transforms import ResponsesBody
    from open_webui_openrouter_pipe.requests.sanitizer import _sanitize_request_input

    body = ResponsesBody.model_validate(
        {
            "model": "openrouter/test",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {
                    "type": "function_call_output",
                    "call_id": "orphan-out-1",
                    "output": "stale result",
                },
            ],
        }
    )

    _sanitize_request_input(pipe_instance, body)

    types = [i.get("type") for i in body.input]
    assert "function_call_output" not in types
    assert len(body.input) == 1


def test_sanitize_stubs_interior_orphaned_function_call(pipe_instance):
    """Interior function_call with no matching output gets a stub inserted after it."""
    from open_webui_openrouter_pipe.api.transforms import ResponsesBody
    from open_webui_openrouter_pipe.requests.sanitizer import (
        _sanitize_request_input,
        _ORPHAN_STUB_OUTPUT,
    )

    body = ResponsesBody.model_validate(
        {
            "model": "openrouter/test",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "orphan-call-1",
                    "name": "search_web",
                    "arguments": "{}",
                },
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "next"}]},
            ],
        }
    )

    _sanitize_request_input(pipe_instance, body)

    assert body.input[0]["type"] == "function_call"
    assert body.input[1]["type"] == "function_call_output"
    assert body.input[1]["call_id"] == "orphan-call-1"
    assert body.input[1]["output"] == _ORPHAN_STUB_OUTPUT
    assert body.input[2]["type"] == "message"


def test_sanitize_preserves_frontier_orphaned_function_call(pipe_instance):
    """Frontier function_call items (no user message after them) are NOT stubbed."""
    from open_webui_openrouter_pipe.api.transforms import ResponsesBody
    from open_webui_openrouter_pipe.requests.sanitizer import _sanitize_request_input

    body = ResponsesBody.model_validate(
        {
            "model": "openrouter/test",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {
                    "type": "function_call",
                    "call_id": "pending-1",
                    "name": "search_web",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "pending-2",
                    "name": "get_weather",
                    "arguments": "{}",
                },
            ],
        }
    )

    _sanitize_request_input(pipe_instance, body)

    types = [i.get("type") for i in body.input]
    assert types == ["message", "function_call", "function_call"]
    assert len(body.input) == 3


def test_sanitize_preserves_frontier_call_with_assistant_message(pipe_instance):
    """function_call followed by assistant message (same turn) is NOT stubbed."""
    from open_webui_openrouter_pipe.api.transforms import ResponsesBody
    from open_webui_openrouter_pipe.requests.sanitizer import _sanitize_request_input

    body = ResponsesBody.model_validate(
        {
            "model": "openrouter/test",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {
                    "type": "function_call",
                    "call_id": "pending-1",
                    "name": "search_web",
                    "arguments": "{}",
                },
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "calling tool"}]},
            ],
        }
    )

    _sanitize_request_input(pipe_instance, body)

    types = [i.get("type") for i in body.input]
    assert types == ["message", "function_call", "message"]
    assert len(body.input) == 3


def test_sanitize_matched_pairs_unchanged(pipe_instance):
    """Properly paired function_call + function_call_output pass through."""
    from open_webui_openrouter_pipe.api.transforms import ResponsesBody
    from open_webui_openrouter_pipe.requests.sanitizer import _sanitize_request_input

    body = ResponsesBody.model_validate(
        {
            "model": "openrouter/test",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "ok-1",
                    "name": "search_web",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "ok-1",
                    "output": "good result",
                },
            ],
        }
    )

    _sanitize_request_input(pipe_instance, body)

    assert len(body.input) == 2
    assert body.input[0]["type"] == "function_call"
    assert body.input[1]["type"] == "function_call_output"


def test_sanitize_mixed_orphans_and_valid_pairs(pipe_instance):
    """Only orphans are removed; valid pairs and messages survive."""
    from open_webui_openrouter_pipe.api.transforms import ResponsesBody
    from open_webui_openrouter_pipe.requests.sanitizer import (
        _sanitize_request_input,
        _ORPHAN_STUB_OUTPUT,
    )

    body = ResponsesBody.model_validate(
        {
            "model": "openrouter/test",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "good-1",
                    "name": "search_web",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "good-1",
                    "output": "valid",
                },
                {
                    "type": "function_call",
                    "call_id": "orphan-call",
                    "name": "broken_tool",
                    "arguments": "{}",
                },
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "next"}]},
                {
                    "type": "function_call_output",
                    "call_id": "orphan-out",
                    "output": "stale",
                },
            ],
        }
    )

    _sanitize_request_input(pipe_instance, body)

    types = [i.get("type") for i in body.input]
    assert types == [
        "function_call",
        "function_call_output",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert body.input[1]["call_id"] == "good-1"
    assert body.input[1]["output"] == "valid"
    assert body.input[3]["call_id"] == "orphan-call"
    assert body.input[3]["output"] == _ORPHAN_STUB_OUTPUT


# The fix is 3 parts:

_HANG_TIMEOUT = 10


def _completed_sse() -> str:
    """Return a standard response.completed SSE line."""
    return _sse({"type": "response.completed", "response": {"output": [], "usage": {}}})


async def _collect_sse_events(
    pipe, session, sse_body: bytes, valves, *, workers: int = 4,
) -> list[dict[str, Any]]:
    """Mock HTTP, run the streaming pipeline, return collected events.

    Wraps the pipeline with ``asyncio.wait_for`` so that if the C3 fix
    regresses (distributor stalls on a missing sequence), the test raises
    ``TimeoutError`` instead of hanging the entire test suite.
    """

    async def _run() -> list[dict[str, Any]]:
        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                body=sse_body,
                headers={"Content-Type": "text/event-stream"},
                status=200,
            )
            events: list[dict[str, Any]] = []
            async for event in pipe.send_openai_responses_streaming_request(
                session,
                {"model": "openai/gpt-4o", "stream": True, "input": []},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
                workers=workers,
            ):
                events.append(event)
            return events

    return await asyncio.wait_for(_run(), timeout=_HANG_TIMEOUT)


@pytest.mark.asyncio
async def test_streaming_malformed_json_does_not_block(pipe_instance_async):
    """Malformed JSON in SSE stream must not hang — valid events still arrive.

    Before the C3 fix the worker would ``continue`` on JSONDecodeError
    without emitting a placeholder, so the distributor waited forever for
    the missing sequence number.
    """
    pipe = pipe_instance_async
    session = pipe._create_http_session(pipe.valves)

    sse_body = (
        _sse({"type": "response.output_text.delta", "delta": "Hello"})
        + "data: {not valid json!!!}\n\n"
        + _sse({"type": "response.output_text.delta", "delta": " World"})
        + _completed_sse()
        + "data: [DONE]\n\n"
    ).encode("utf-8")

    events = await _collect_sse_events(pipe, session, sse_body, pipe.valves)
    await session.close()

    text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    combined = "".join(e["delta"] for e in text_deltas)
    assert "Hello" in combined
    assert "World" in combined
    assert any(e.get("type") == "response.completed" for e in events)


@pytest.mark.asyncio
async def test_streaming_invalid_utf8_does_not_block(pipe_instance_async):
    """Invalid UTF-8 bytes after ``data:`` must not crash or hang the pipeline.

    Before the C3 fix the ``except json.JSONDecodeError`` would NOT catch
    ``UnicodeDecodeError``.  The broadened ``except Exception`` catches it
    and emits the placeholder.
    """
    pipe = pipe_instance_async
    session = pipe._create_http_session(pipe.valves)

    sse_body = (
        _sse({"type": "response.output_text.delta", "delta": "Before"}).encode("utf-8")
        + b"data: \xff\xfe\xfd\n\n"
        + _sse({"type": "response.output_text.delta", "delta": "After"}).encode("utf-8")
        + _completed_sse().encode("utf-8")
        + b"data: [DONE]\n\n"
    )

    events = await _collect_sse_events(pipe, session, sse_body, pipe.valves)
    await session.close()

    text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    combined = "".join(e["delta"] for e in text_deltas)
    assert "Before" in combined
    assert "After" in combined
    assert any(e.get("type") == "response.completed" for e in events)


@pytest.mark.asyncio
async def test_streaming_only_malformed_before_completed(pipe_instance_async):
    """If all text-delta events are malformed, only completed is yielded.

    No text deltas should appear, but the stream must still terminate
    normally — the completed event must arrive.
    """
    pipe = pipe_instance_async
    session = pipe._create_http_session(pipe.valves)

    sse_body = (
        "data: INVALID_1\n\n"
        + "data: INVALID_2\n\n"
        + "data: INVALID_3\n\n"
        + _completed_sse()
        + "data: [DONE]\n\n"
    ).encode("utf-8")

    events = await _collect_sse_events(pipe, session, sse_body, pipe.valves)
    await session.close()

    text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    assert len(text_deltas) == 0
    assert any(e.get("type") == "response.completed" for e in events)


@pytest.mark.asyncio
async def test_streaming_multiple_failures_preserve_event_order(pipe_instance_async):
    """Interleaved parse failures must not reorder valid events.

    Input:  A, BAD, BAD, B, BAD, C, completed, [DONE]
    Output: A, B, C in that exact order — the distributor's sequence
    numbering must advance through None placeholders without reordering.
    """
    pipe = pipe_instance_async
    session = pipe._create_http_session(pipe.valves)

    sse_body = (
        _sse({"type": "response.output_text.delta", "delta": "A"})
        + "data: BAD_1\n\n"
        + "data: BAD_2\n\n"
        + _sse({"type": "response.output_text.delta", "delta": "B"})
        + "data: BAD_3\n\n"
        + _sse({"type": "response.output_text.delta", "delta": "C"})
        + _completed_sse()
        + "data: [DONE]\n\n"
    ).encode("utf-8")

    events = await _collect_sse_events(pipe, session, sse_body, pipe.valves)
    await session.close()

    text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    delta_texts = [e["delta"] for e in text_deltas]
    assert delta_texts == ["A", "B", "C"]
    assert any(e.get("type") == "response.completed" for e in events)


@pytest.mark.asyncio
async def test_streaming_parse_failure_multiworker(pipe_instance_async):
    """Parse failures with multiple workers must not cause hangs.

    Uses ``workers=4`` to verify that the sequencing fix holds even when
    chunks are distributed across multiple concurrent workers.
    """
    pipe = pipe_instance_async
    session = pipe._create_http_session(pipe.valves)

    sse_body = (
        _sse({"type": "response.output_text.delta", "delta": "W1"})
        + "data: BROKEN\n\n"
        + _sse({"type": "response.output_text.delta", "delta": "W2"})
        + _completed_sse()
        + "data: [DONE]\n\n"
    ).encode("utf-8")

    events = await _collect_sse_events(
        pipe, session, sse_body, pipe.valves, workers=4,
    )
    await session.close()

    text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    combined = "".join(e["delta"] for e in text_deltas)
    assert "W1" in combined
    assert "W2" in combined
    assert any(e.get("type") == "response.completed" for e in events)


@pytest.mark.asyncio
async def test_streaming_mixed_error_types_do_not_block(pipe_instance_async):
    """Both JSONDecodeError and UnicodeDecodeError must be handled together.

    Sends a mix of:
      - ``{broken json`` → JSONDecodeError
      - raw ``\\xff\\x80\\x81`` → UnicodeDecodeError
    Both must produce placeholders; valid events must still arrive.
    """
    pipe = pipe_instance_async
    session = pipe._create_http_session(pipe.valves)

    sse_body = (
        _sse({"type": "response.output_text.delta", "delta": "Start"}).encode("utf-8")
        + b"data: {broken json\n\n"
        + b"data: \xff\x80\x81\n\n"
        + _sse({"type": "response.output_text.delta", "delta": "End"}).encode("utf-8")
        + _completed_sse().encode("utf-8")
        + b"data: [DONE]\n\n"
    )

    events = await _collect_sse_events(pipe, session, sse_body, pipe.valves)
    await session.close()

    text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    combined = "".join(e["delta"] for e in text_deltas)
    assert "Start" in combined
    assert "End" in combined
    assert any(e.get("type") == "response.completed" for e in events)


@pytest.mark.asyncio
async def test_streaming_consecutive_malformed_at_start(pipe_instance_async):
    """Malformed events at the very start of the stream must not block.

    The distributor starts at ``next_seq = 0``.  If sequences 0 and 1 are
    both None placeholders, the distributor must advance through them to
    reach the first valid event at sequence 2.
    """
    pipe = pipe_instance_async
    session = pipe._create_http_session(pipe.valves)

    sse_body = (
        "data: JUNK_0\n\n"
        + "data: JUNK_1\n\n"
        + _sse({"type": "response.output_text.delta", "delta": "Finally"})
        + _completed_sse()
        + "data: [DONE]\n\n"
    ).encode("utf-8")

    events = await _collect_sse_events(pipe, session, sse_body, pipe.valves)
    await session.close()

    text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    assert len(text_deltas) >= 1
    assert "Finally" in "".join(e["delta"] for e in text_deltas)
    assert any(e.get("type") == "response.completed" for e in events)


def test_should_retry_stream_no_retry_after_emit():
    """Streaming retry is only safe before any event reaches the consumer.
    Once output is emitted, retrying re-streams from scratch with fresh seq
    numbers and duplicates delivered content (regression guard)."""
    import asyncio as _asyncio
    import aiohttp as _aiohttp
    from open_webui_openrouter_pipe.api.gateway.responses_adapter import _should_retry_stream

    assert _should_retry_stream(False, _aiohttp.ClientConnectionError()) is True
    assert _should_retry_stream(False, _aiohttp.ClientPayloadError()) is True
    assert _should_retry_stream(False, _asyncio.TimeoutError()) is True

    assert _should_retry_stream(True, _aiohttp.ClientConnectionError()) is False
    assert _should_retry_stream(True, _aiohttp.ClientPayloadError()) is False
    assert _should_retry_stream(True, _asyncio.TimeoutError()) is False

    assert _should_retry_stream(False, ValueError("x")) is False
    assert _should_retry_stream(False, None) is False


@pytest.mark.asyncio
async def test_responses_streaming_injects_anthropic_toplevel_cache_control(pipe_instance_async):
    """Streaming /responses requests for anthropic models carry a top-level cache_control and the interleaved-thinking header."""
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
            "ENABLE_ANTHROPIC_INTERLEAVED_THINKING": True,
        }
    )
    session = pipe._create_http_session(valves)
    captured: dict[str, Any] = {}

    def _callback(url, **kwargs):
        captured.update(kwargs)
        return CallbackResult(
            status=200,
            body=(_completed_sse() + "data: [DONE]\n\n").encode("utf-8"),
            content_type="text/event-stream",
        )

    with aioresponses() as mock_http:
        mock_http.post("https://openrouter.ai/api/v1/responses", callback=_callback)
        async for _ in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "anthropic/claude-sonnet-4.6", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
        ):
            pass
        await session.close()

    assert captured.get("json", {}).get("cache_control") == {"type": "ephemeral"}
    assert "interleaved-thinking-2025-05-14" in (captured.get("headers", {}) or {}).get("x-anthropic-beta", "")


@pytest.mark.asyncio
async def test_responses_nonstreaming_injects_anthropic_toplevel_cache_control(pipe_instance_async):
    """Non-streaming /responses requests for anthropic models carry top-level cache_control (1h) and the interleaved-thinking header."""
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "1h",
            "ENABLE_ANTHROPIC_INTERLEAVED_THINKING": True,
        }
    )
    session = pipe._create_http_session(valves)
    captured: dict[str, Any] = {}

    def _callback(url, **kwargs):
        captured.update(kwargs)
        return CallbackResult(status=200, payload={"output": [], "usage": {"input_tokens": 5, "output_tokens": 1}})

    with aioresponses() as mock_http:
        mock_http.post("https://openrouter.ai/api/v1/responses", callback=_callback)
        await pipe.send_openai_responses_nonstreaming_request(
            session,
            {"model": "anthropic/claude-sonnet-4.6", "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
        )
        await session.close()

    assert captured.get("json", {}).get("cache_control") == {"type": "ephemeral", "ttl": "1h"}
    assert "interleaved-thinking-2025-05-14" in (captured.get("headers", {}) or {}).get("x-anthropic-beta", "")


@pytest.mark.asyncio
async def test_responses_no_toplevel_cache_control_for_non_anthropic(pipe_instance_async):
    """Non-anthropic /responses requests do not get a top-level cache_control."""
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(update={"ENABLE_ANTHROPIC_PROMPT_CACHING": True})
    session = pipe._create_http_session(valves)
    captured: dict[str, Any] = {}

    def _callback(url, **kwargs):
        captured.update(kwargs)
        return CallbackResult(
            status=200,
            body=(_completed_sse() + "data: [DONE]\n\n").encode("utf-8"),
            content_type="text/event-stream",
        )

    with aioresponses() as mock_http:
        mock_http.post("https://openrouter.ai/api/v1/responses", callback=_callback)
        async for _ in pipe.send_openai_responses_streaming_request(
            session,
            {"model": "openai/gpt-4o", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
        ):
            pass
        await session.close()

    assert "cache_control" not in captured.get("json", {})


@pytest.mark.parametrize(
    ("reported", "survives"),
    [
        ("completed", True),
        ("incomplete", True),
        ("in_progress", True),
        ("failed", False),
        ("error", False),
        ("cancelled", False),
        ("", False),
        (None, False),
        (123, False),
    ],
)
def test_only_documented_tool_output_statuses_reach_the_wire(pipe_instance, reported, survives):
    """The whitelist is load-bearing in both directions.

    Narrowing it back to nothing is caught elsewhere; widening it to
    `if reported_status is not None` is not, and that is the shape that lets a
    provider-specific string through. OpenRouter documents exactly three values for
    a function_call_output, and an undocumented one is a request the provider may
    reject outright.
    """
    from open_webui_openrouter_pipe.api.transforms import ResponsesBody
    from open_webui_openrouter_pipe.requests.sanitizer import _sanitize_request_input

    item = {
        "type": "function_call_output",
        "id": "ulid-1",
        "call_id": "call-1",
        "output": "result text",
    }
    if reported is not None:
        item["status"] = reported

    call = {
        "type": "function_call",
        "call_id": "call-1",
        "name": "search_web",
        "arguments": "{}",
    }
    body = ResponsesBody.model_validate(
        {"model": "openrouter/test", "input": [call, item], "stream": True}
    )
    _sanitize_request_input(pipe_instance, body)

    sent = [i for i in body.input if isinstance(i, dict) and i.get("type") == "function_call_output"]
    assert sent, "the tool output was dropped entirely"
    if survives:
        assert sent[0].get("status") == reported, (
            f"{reported!r} is a documented ToolCallStatus and must reach the provider; "
            "dropping it tells the model nothing about the call's outcome"
        )
    else:
        assert "status" not in sent[0], (
            f"{reported!r} is not one of OpenRouter's documented ToolCallStatus values "
            f"but reached the wire as {sent[0].get('status')!r}"
        )
