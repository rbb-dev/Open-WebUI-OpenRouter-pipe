import pytest

from openrouter_responses_pipe.openrouter_responses_pipe import (
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
        session=object(),
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

    async def fake_report(self, exc, **kwargs):
        reported.append((exc, kwargs))

    monkeypatch.setattr(Pipe, "_report_openrouter_error", fake_report)

    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    result = await pipe._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "sandbox"}},
        tools={},
        session=object(),
        user_id="user-123",
    )

    assert result == ""
    assert reported and reported[0][0] is error
    report_kwargs = reported[0][1]
    assert report_kwargs["normalized_model_id"] == body.model
    assert report_kwargs["api_model_id"] == getattr(body, "api_model", None)
