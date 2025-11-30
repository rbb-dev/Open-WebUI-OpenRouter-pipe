from __future__ import annotations

import pytest

from openrouter_responses_pipe.openrouter_responses_pipe import (
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
async def test_from_completions_preserves_response_format(monkeypatch, minimal_pipe):
    """response_format must flow through to the Responses body."""
    completions = CompletionsBody(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_schema", "json_schema": {"name": "demo", "schema": {"type": "object"}}},
    )

    async def fake_transform(_transformer, *_args, **_kwargs):
        return list(_STUBBED_INPUT)

    monkeypatch.setattr(ResponsesBody, "transform_messages_to_input", fake_transform)

    responses = await ResponsesBody.from_completions(
        completions,
        transformer_context=minimal_pipe,
    )
    assert responses.response_format == completions.response_format


@pytest.mark.asyncio
async def test_from_completions_preserves_parallel_tool_calls(monkeypatch, minimal_pipe):
    """parallel_tool_calls must remain set so routing can respect it."""
    completions = CompletionsBody(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        parallel_tool_calls=False,
    )

    async def fake_transform(_transformer, *_args, **_kwargs):
        return list(_STUBBED_INPUT)

    monkeypatch.setattr(ResponsesBody, "transform_messages_to_input", fake_transform)

    responses = await ResponsesBody.from_completions(
        completions,
        transformer_context=minimal_pipe,
    )
    assert responses.parallel_tool_calls is False
