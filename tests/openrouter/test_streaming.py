import asyncio
import pytest
from typing import Any, AsyncGenerator, cast
from unittest.mock import AsyncMock, Mock

from fastapi.responses import JSONResponse

from open_webui_openrouter_pipe import (
    ModelFamily,
    OpenRouterAPIError,
    Pipe,
    ResponsesBody,
)


@pytest.mark.asyncio
async def test_completion_events_preserve_streamed_text(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    body = ResponsesBody(model="openrouter/test", input=[], stream=True)
    valves = pipe.valves

    events = [
        {"type": "response.output_text.delta", "delta": "Hello"},
        {"type": "response.output_text.delta", "delta": " world"},
        {
            "type": "response.completed",
            "response": {
                "output": [],
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        },
    ]

    async def fake_stream(self, session, request_body, **_kwargs):
        for event in events:
            yield event

    monkeypatch.setattr(
        Pipe, "send_openai_responses_streaming_request", fake_stream
    )

    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    output = await pipe._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "sandbox"}},
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )

    assert output == "Hello world"

    completion_events = [event for event in emitted if event.get("type") == "chat:completion"]
    assert completion_events, "Expected at least one completion event"

    for event in completion_events:
        assert event["data"]["content"] == "Hello world"
        assert event["data"].get("usage") == {"input_tokens": 5, "output_tokens": 2, "turn_count": 1, "function_call_count": 0}


@pytest.mark.asyncio
async def test_streaming_loop_handles_openrouter_errors(monkeypatch, pipe_instance_async):
    """Test that OpenRouter errors are reported via real _report_openrouter_error method.

    FIX: Removed inappropriate spy mock on _report_openrouter_error.
    Now exercises the real error reporting method which emits status events.
    """
    pipe = pipe_instance_async
    body = ResponsesBody(model="openrouter/test", input=[], stream=True)
    valves = pipe.valves

    error = OpenRouterAPIError(status=400, reason="Bad Request", provider="Test")

    async def fake_stream(self, session, request_body, **_kwargs):
        raise error
        if False:  # pragma: no cover
            yield {}

    monkeypatch.setattr(
        Pipe, "send_openai_responses_streaming_request", fake_stream
    )

    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    result = await pipe._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "sandbox"}},
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )

    assert result == ""

    # Verify that error reporting happened by checking status events
    status_events = [event for event in emitted if event.get("type") == "status"]
    assert status_events, "Expected provider error status event"
    assert status_events[-1]["data"]["done"] is True
    assert "provider error" in status_events[-1]["data"]["description"].lower()


@pytest.mark.asyncio
async def test_streaming_loop_reasoning_status_and_tools(monkeypatch, pipe_instance_async):
    """Test reasoning status updates and tool execution.

    FIX: Removed inappropriate mocks on _db_persist, _execute_function_calls, and _make_db_row.
    Now exercises real methods with proper mocking of the ArtifactStore boundary.
    """
    pipe = pipe_instance_async
    body = ResponsesBody(
        model="openrouter/test-rich",
        input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        stream=True,
    )
    valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "status"})

    # Set real model capabilities using ModelFamily.set_dynamic_specs
    ModelFamily.set_dynamic_specs({
        "openrouter.test-rich": {
            "features": {"function_calling": True}
        }
    })

    # Mock ArtifactStore boundary methods instead of Pipe internal methods
    persisted_rows: list[dict] = []

    async def mock_artifact_persist(rows):
        """Mock the ArtifactStore._db_persist boundary."""
        persisted_rows.extend(rows)
        ulids: list[str] = []
        for idx, row in enumerate(rows):
            row_id = row.get("id")
            if not row_id:
                row_id = f"fake-ulid-{idx}"
            ulids.append(row_id)
        return ulids

    def mock_artifact_make_db_row(chat_id, message_id, model_id, payload):
        """Mock the ArtifactStore._make_db_row boundary."""
        return {
            "id": payload.get("id") or f"row-{payload.get('type')}",
            "chat_id": chat_id,
            "message_id": message_id,
            "model_id": model_id,
            "item_type": payload.get("type", "unknown"),
            "payload": payload,
        }

    # Mock at the ArtifactStore boundary, not Pipe internal methods
    monkeypatch.setattr(pipe._artifact_store, "_db_persist", mock_artifact_persist)
    monkeypatch.setattr(pipe._artifact_store, "_make_db_row", mock_artifact_make_db_row)

    # Mock the tool callable to return a simple result
    async def mock_tool_callable(**kwargs):
        return "ok"

    events = [
        {"type": "response.reasoning.delta", "delta": "Analyzing context."},
        {"type": "response.reasoning_summary_text.done", "text": "**Plan** Use cached data."},
        {
            "type": "response.output_text.annotation.added",
            "annotation": {
                "type": "url_citation",
                "url": "https://example.com/data?utm_source=openai",
                "title": "Example Data",
            },
        },
        {"type": "response.output_text.delta", "delta": "All set."},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "id": "reason-1",
                "status": "completed",
                "content": [{"type": "reasoning_text", "text": "Finished reasoning."}],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "All set."}],
                    },
                    {
                        "type": "function_call",
                        "id": "call-1",
                        "call_id": "call-1",
                        "name": "lookup",
                        "arguments": '{"foo": 1}',
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 6},
            },
        },
    ]

    async def fake_stream(self, session, request_body, **_kwargs):
        assert request_body["model"] == body.model
        for event in events:
            yield event

    monkeypatch.setattr(Pipe, "send_openai_responses_streaming_request", fake_stream)

    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    result = await pipe._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "sandbox"}, "chat_id": "chat-1", "message_id": "msg-1"},
        tools={"lookup": {"type": "function", "spec": {"name": "lookup"}, "callable": mock_tool_callable}},
        session=cast(Any, object()),
        user_id="user-123",
    )

    assert result.startswith("All set.")
    assert any(event["type"] == "source" for event in emitted), "Expected source event"
    status_texts = [event["data"]["description"] for event in emitted if event["type"] == "status"]
    assert any("Plan" in text for text in status_texts), "Reasoning status update missing"
    completion_events = [event for event in emitted if event["type"] == "chat:completion"]
    assert completion_events and "turn_count" in completion_events[-1]["data"]["usage"]
    assert persisted_rows, "Expected reasoning payload persistence"


@pytest.mark.asyncio
async def test_pipe_stream_mode_outputs_openai_reasoning_chunks(monkeypatch, pipe_instance_async):
    """Test that reasoning events are emitted in open_webui mode.

    When THINKING_OUTPUT_MODE is "open_webui" (default), reasoning events
    are emitted as reasoning:delta and reasoning:completed events for the
    Open WebUI thinking box feature.
    """
    pipe = pipe_instance_async
    body = ResponsesBody(
        model="openrouter/test",
        input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        stream=True,
    )
    valves = pipe.valves  # Use default valves (open_webui mode)

    # Prepare SSE events with reasoning to return
    events = [
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs-1"}},
        {"type": "response.reasoning_text.delta", "delta": "Analysing..."},
        {"type": "response.output_text.delta", "delta": "Hello"},
        {"type": "response.reasoning_text.delta", "delta": "Late reasoning."},
        {"type": "response.completed", "response": {"output": [], "usage": {"input_tokens": 1}}},
    ]

    # Mock the SSE streaming method to return our events
    async def fake_stream(self, session, request_body, **_kwargs):
        for event in events:
            yield event

    monkeypatch.setattr(Pipe, "send_openai_responses_streaming_request", fake_stream)

    # Collect emitted items
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    # Call _run_streaming_loop to test event emission
    output = await pipe._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "sandbox"}},
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )

    assert output == "Hello"

    # In open_webui mode, reasoning events should be emitted as reasoning:delta
    reasoning_deltas = [
        item
        for item in emitted
        if isinstance(item, dict) and item.get("type") == "reasoning:delta"
    ]
    assert reasoning_deltas, "Expected reasoning:delta events in open_webui mode"

    # Verify reasoning content contains expected text
    reasoning_text = "".join(
        item.get("data", {}).get("delta", "")
        for item in reasoning_deltas
    )
    assert "Analysing" in reasoning_text, f"Expected 'Analysing' in reasoning deltas, got: {reasoning_text}"
    assert "Late reasoning" in reasoning_text, f"Expected 'Late reasoning' in reasoning deltas, got: {reasoning_text}"

    # Verify regular content message exists
    chat_messages = [
        item
        for item in emitted
        if isinstance(item, dict) and item.get("type") == "chat:message"
    ]
    assert any(
        item.get("data", {}).get("content") == "Hello"
        for item in chat_messages
    ), "Expected chat:message with 'Hello'"

    # Verify reasoning:completed event is emitted at the end
    assert any(
        isinstance(item, dict) and item.get("type") == "reasoning:completed"
        for item in emitted
    ), "Expected reasoning:completed event"

    # Verify reasoning is NOT in status messages (that's "status" mode behavior)
    status_msgs = [
        item.get("data", {}).get("description", "")
        for item in emitted
        if isinstance(item, dict) and item.get("type") == "status"
    ]
    assert not any(
        "Analysing" in msg or "Late reasoning" in msg
        for msg in status_msgs
    ), "Reasoning text should not appear in status messages in open_webui mode"


@pytest.mark.asyncio
async def test_thinking_output_mode_open_webui_suppresses_thinking_status(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    body = ResponsesBody(
        model="openrouter/test",
        input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        stream=True,
    )
    valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})

    events = [
        {"type": "response.reasoning_text.delta", "delta": "Building a plan."},
        {"type": "response.reasoning_summary_text.done", "text": "**Building a plan…**\nDrafting steps."},
        {"type": "response.completed", "response": {"output": [], "usage": {}}},
    ]

    async def fake_stream(self, session, request_body, **_kwargs):
        for event in events:
            yield event

    monkeypatch.setattr(Pipe, "send_openai_responses_streaming_request", fake_stream)

    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "sandbox"}},
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )

    assert any(event.get("type") == "reasoning:delta" for event in emitted)
    status_texts = [event.get("data", {}).get("description", "") for event in emitted if event.get("type") == "status"]
    assert any("Thinking" in text for text in status_texts)
    assert not any("Building a plan" in text for text in status_texts)


@pytest.mark.asyncio
async def test_thinking_output_mode_status_suppresses_reasoning_events(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    body = ResponsesBody(
        model="openrouter/test",
        input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        stream=True,
    )
    valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "status"})

    events = [
        {"type": "response.reasoning_text.delta", "delta": "Building a plan."},
        {"type": "response.reasoning_summary_text.done", "text": "**Building a plan…**\nDrafting steps."},
        {"type": "response.completed", "response": {"output": [], "usage": {}}},
    ]

    async def fake_stream(self, session, request_body, **_kwargs):
        for event in events:
            yield event

    monkeypatch.setattr(Pipe, "send_openai_responses_streaming_request", fake_stream)

    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "sandbox"}},
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )

    assert not any(event.get("type") == "reasoning:delta" for event in emitted)
    status_texts = [event.get("data", {}).get("description", "") for event in emitted if event.get("type") == "status"]
    assert any("Building a plan" in text for text in status_texts)


@pytest.mark.asyncio
async def test_reasoning_summary_only_streams_to_reasoning_box_in_open_webui_mode(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    body = ResponsesBody(
        model="openrouter/test",
        input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        stream=True,
    )
    valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})

    events = [
        {"type": "response.reasoning_summary_text.done", "text": "**Thinking…**\nSummary only."},
        {"type": "response.completed", "response": {"output": [], "usage": {}}},
    ]

    async def fake_stream(self, session, request_body, **_kwargs):
        for event in events:
            yield event

    monkeypatch.setattr(Pipe, "send_openai_responses_streaming_request", fake_stream)

    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "sandbox"}},
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )

    assert any(event.get("type") == "reasoning:delta" for event in emitted)
    assert any(event.get("type") == "reasoning:completed" for event in emitted)
    status_texts = [event.get("data", {}).get("description", "") for event in emitted if event.get("type") == "status"]
    assert any("Thinking" in text for text in status_texts)
    assert not any("Summary only" in text for text in status_texts)


@pytest.mark.asyncio
async def test_reasoning_summary_part_done_does_not_replay_after_incremental(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    body = ResponsesBody(
        model="openrouter/test",
        input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        stream=True,
    )
    valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})

    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "reasoning", "id": "rs-1", "summary": []},
        },
        {
            "type": "response.reasoning_summary_part.added",
            "output_index": 0,
            "item_id": "rs-1",
            "summary_index": 0,
            "part": {"type": "summary_text", "text": "Hello"},
        },
        {"type": "response.output_text.delta", "delta": "Answer"},
        {
            "type": "response.reasoning_summary_part.done",
            "output_index": 0,
            "item_id": "rs-1",
            "summary_index": 0,
            "part": {"type": "summary_text", "text": "Hello"},
        },
        {
            "type": "response.reasoning_summary_text.done",
            "output_index": 0,
            "item_id": "rs-1",
            "summary_index": 0,
            "text": "**Thinking…**\nHello",
        },
        {"type": "response.completed", "response": {"output": [], "usage": {}}},
    ]

    async def fake_stream(self, session, request_body, **_kwargs):
        for event in events:
            yield event

    monkeypatch.setattr(Pipe, "send_openai_responses_streaming_request", fake_stream)

    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "sandbox"}},
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )

    deltas = [event for event in emitted if event.get("type") == "reasoning:delta"]
    assert [event.get("data", {}).get("delta") for event in deltas] == ["Hello"]


@pytest.mark.asyncio
async def test_reasoning_done_snapshots_do_not_replay_after_delta(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    body = ResponsesBody(
        model="openrouter/test",
        input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        stream=True,
    )
    valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})

    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "reasoning", "id": "rs-1", "summary": []},
        },
        {"type": "response.reasoning_text.delta", "delta": "Step 1."},
        {"type": "response.output_text.delta", "delta": "Answer"},
        {
            "type": "response.reasoning_text.done",
            "output_index": 0,
            "item_id": "rs-1",
            "content_index": 0,
            "text": "Step 1.",
        },
        {
            "type": "response.content_part.done",
            "item_id": "rs-1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "reasoning_text", "text": "Step 1."},
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "rs-1",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "Step 1."}],
            },
        },
        {"type": "response.completed", "response": {"output": [], "usage": {}}},
    ]

    async def fake_stream(self, session, request_body, **_kwargs):
        for event in events:
            yield event

    monkeypatch.setattr(Pipe, "send_openai_responses_streaming_request", fake_stream)

    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "sandbox"}},
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )

    deltas = [event for event in emitted if event.get("type") == "reasoning:delta"]
    assert [event.get("data", {}).get("delta") for event in deltas] == ["Step 1."]


@pytest.mark.asyncio
async def test_function_call_status_invalid_json_arguments_does_not_crash(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    body = ResponsesBody(
        model="openrouter/test",
        input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        stream=True,
    )
    valves = pipe.valves

    events = [
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "call-1",
                "call_id": "call-1",
                "name": "get_weather_forecast_forecast_get",
                "arguments": "{",
            },
        },
        {"type": "response.output_text.delta", "delta": "OK"},
        {"type": "response.completed", "response": {"output": [], "usage": {}}},
    ]

    async def fake_stream(self, session, request_body, **_kwargs):
        for event in events:
            yield event

    monkeypatch.setattr(Pipe, "send_openai_responses_streaming_request", fake_stream)

    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    result = await pipe._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "sandbox"}},
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )

    assert result == "OK"
    assert not any(event.get("type") == "chat:completion" and event.get("data", {}).get("error") for event in emitted)


@pytest.mark.asyncio
async def test_legacy_tool_execution_invalid_arguments_returns_failed_output(pipe_instance_async):
    pipe = pipe_instance_async
    calls = [{"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{"}]
    tools = {"lookup": {"type": "function", "spec": {"name": "lookup"}, "callable": lambda: None}}

    outputs = await pipe._execute_function_calls_legacy(calls, tools)

    assert outputs and outputs[0]["type"] == "function_call_output"
    assert "Invalid arguments" in outputs[0]["output"]


def test_anthropic_interleaved_thinking_header_applied(pipe_instance):
    pipe = pipe_instance
    valves = pipe.valves.model_copy(update={"ENABLE_ANTHROPIC_INTERLEAVED_THINKING": True})

    headers: dict[str, str] = {}
    pipe._maybe_apply_anthropic_beta_headers(headers, "anthropic/claude-sonnet-4", valves=valves)
    assert headers.get("x-anthropic-beta") == "interleaved-thinking-2025-05-14"

    headers = {"x-anthropic-beta": "fine-grained-tool-streaming-2025-05-14"}
    pipe._maybe_apply_anthropic_beta_headers(headers, "anthropic/claude-sonnet-4", valves=valves)
    assert headers.get("x-anthropic-beta") == (
        "fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14"
    )

    valves_disabled = pipe.valves.model_copy(update={"ENABLE_ANTHROPIC_INTERLEAVED_THINKING": False})
    headers = {}
    pipe._maybe_apply_anthropic_beta_headers(headers, "anthropic/claude-sonnet-4", valves=valves_disabled)
    assert "x-anthropic-beta" not in headers


@pytest.mark.asyncio
async def test_function_call_loop_limit_emits_warning(monkeypatch, pipe_instance_async):
    """Test function call loop limit warning.

    FIX: Removed inappropriate mock on _execute_function_calls.
    Now exercises real tool execution path with proper tool callable.
    """
    pipe = pipe_instance_async
    body = ResponsesBody(
        model="openrouter/test-loops",
        input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        stream=True,
    )
    valves = pipe.valves.model_copy(update={"MAX_FUNCTION_CALL_LOOPS": 1})

    # Set real model capabilities using ModelFamily.set_dynamic_specs
    ModelFamily.set_dynamic_specs({
        "openrouter.test-loops": {
            "features": {"function_calling": True}
        }
    })

    events = [
        {
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "type": "function_call",
                        "id": "call-1",
                        "call_id": "call-1",
                        "name": "lookup",
                        "arguments": "{}",
                    },
                ],
                "usage": {},
            },
        },
    ]

    async def fake_stream(self, session, request_body, **_kwargs):
        assert request_body["model"] == body.model
        for event in events:
            yield event

    monkeypatch.setattr(Pipe, "send_openai_responses_streaming_request", fake_stream)

    # Provide a real tool callable instead of mocking _execute_function_calls
    async def mock_lookup_tool(**kwargs):
        return "ok"

    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "sandbox"}},
        tools={"lookup": {"type": "function", "spec": {"name": "lookup"}, "callable": mock_lookup_tool}},
        session=cast(Any, object()),
        user_id="user-123",
    )

    notifications = [e for e in emitted if e.get("type") == "notification"]
    assert notifications, "Expected a notification when MAX_FUNCTION_CALL_LOOPS is reached"
    assert "MAX_FUNCTION_CALL_LOOPS" in notifications[-1]["data"]["content"]
    chat_messages = [e for e in emitted if e.get("type") == "chat:message"]
    assert any("Tool step limit reached" in (m.get("data", {}).get("content") or "") for m in chat_messages)
