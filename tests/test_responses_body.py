from __future__ import annotations

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

    # No mocking - let real transform_messages_to_input run
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

    # No mocking - let real transform_messages_to_input run
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

    # No mocking - let real transform_messages_to_input run
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

    # No mocking - let real transform_messages_to_input run
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

    # No mocking - let real transform_messages_to_input run
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

    # No mocking - let real transform_messages_to_input run
    responses = await ResponsesBody.from_completions(
        completions,
        transformer_context=minimal_pipe,
    )
    assert responses.tool_choice == "auto"


def test_auto_context_trimming_enabled_by_default(minimal_pipe):
    responses = ResponsesBody(model="test", input=_STUBBED_INPUT)
    minimal_pipe._apply_context_transforms(responses, minimal_pipe.valves)
    assert responses.transforms == ["middle-out"]


def test_auto_context_trimming_respects_explicit_transforms(minimal_pipe):
    responses = ResponsesBody(model="test", input=_STUBBED_INPUT, transforms=["custom"])
    minimal_pipe._apply_context_transforms(responses, minimal_pipe.valves)
    assert responses.transforms == ["custom"]


def test_auto_context_trimming_disabled_via_valve(minimal_pipe):
    responses = ResponsesBody(model="test", input=_STUBBED_INPUT)
    valves = minimal_pipe.valves.model_copy(update={"AUTO_CONTEXT_TRIMMING": False})
    minimal_pipe._apply_context_transforms(responses, valves)
    assert responses.transforms is None
