import pytest
from typing import Any, cast

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import (
    ModelFamily,
    OpenRouterAPIError,
    Pipe,
    ResponsesBody,
)


@pytest.mark.asyncio
async def test_completion_events_preserve_streamed_text(monkeypatch):
    pipe = Pipe()
    body = ResponsesBody(model="openrouter/test", input=[])
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
async def test_streaming_loop_handles_openrouter_errors(monkeypatch):
    pipe = Pipe()
    body = ResponsesBody(model="openrouter/test", input=[])
    valves = pipe.valves

    error = OpenRouterAPIError(status=400, reason="Bad Request", provider="Test")

    async def fake_stream(self, session, request_body, **_kwargs):
        raise error
        if False:  # pragma: no cover
            yield {}

    monkeypatch.setattr(
        Pipe, "send_openai_responses_streaming_request", fake_stream
    )

    reported: list[tuple[OpenRouterAPIError, dict]] = []

    original_report = Pipe._report_openrouter_error

    async def spy_report(self, exc, **kwargs):
        reported.append((exc, kwargs))
        await original_report(self, exc, **kwargs)

    monkeypatch.setattr(Pipe, "_report_openrouter_error", spy_report)

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
    assert reported and reported[0][0] is error
    report_kwargs = reported[0][1]
    assert report_kwargs["normalized_model_id"] == body.model
    assert report_kwargs["api_model_id"] == getattr(body, "api_model", None)
    status_events = [event for event in emitted if event.get("type") == "status"]
    assert status_events, "Expected provider error status event"
    assert status_events[-1]["data"]["done"] is True
    assert "provider error" in status_events[-1]["data"]["description"].lower()


@pytest.mark.asyncio
async def test_streaming_loop_reasoning_status_and_tools(monkeypatch):
    pipe = Pipe()
    body = ResponsesBody(
        model="openrouter/test-rich",
        input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        stream=True,
    )
    valves = pipe.valves

    def fake_supports(cls, feature, _model_id):
        return feature == "function_calling"

    monkeypatch.setattr(ModelFamily, "supports", classmethod(fake_supports))

    persisted_rows: list[dict] = []

    async def fake_db_persist(self, rows):
        persisted_rows.extend(rows)
        ulids: list[str] = []
        for idx, row in enumerate(rows):
            row_id = row.get("id")
            if not row_id:
                row_id = f"fake-ulid-{idx}"
            ulids.append(row_id)
        return ulids

    async def fake_execute(self, calls, _tools):
        outputs = []
        for call in calls:
            outputs.append(
                {
                    "type": "function_call_output",
                    "status": "completed",
                    "call_id": call.get("call_id"),
                    "output": f"ran {call.get('name')}",
                }
            )
        return outputs

    def fake_make_db_row(self, chat_id, message_id, model_id, payload):
        return {
            "id": payload.get("id") or f"row-{payload.get('type')}",
            "chat_id": chat_id,
            "message_id": message_id,
            "model_id": model_id,
            "item_type": payload.get("type", "unknown"),
            "payload": payload,
        }

    monkeypatch.setattr(Pipe, "_db_persist", fake_db_persist)
    monkeypatch.setattr(Pipe, "_execute_function_calls", fake_execute)
    monkeypatch.setattr(Pipe, "_make_db_row", fake_make_db_row)

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
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )

    assert result.startswith("All set.")
    assert any(event["type"] == "citation" for event in emitted), "Expected citation event"
    status_texts = [event["data"]["description"] for event in emitted if event["type"] == "status"]
    assert any("Plan" in text for text in status_texts), "Reasoning status update missing"
    completion_events = [event for event in emitted if event["type"] == "chat:completion"]
    assert completion_events and "turn_count" in completion_events[-1]["data"]["usage"]
    assert persisted_rows, "Expected reasoning payload persistence"
