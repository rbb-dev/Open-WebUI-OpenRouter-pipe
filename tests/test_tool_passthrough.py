from __future__ import annotations

from typing import Any, AsyncGenerator, cast

import pytest

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import Pipe, ResponsesBody, build_tools


def _fake_responses_events_with_tool_call(*, tool_name: str = "my_tool") -> AsyncGenerator[dict[str, Any], None]:
    async def _gen() -> AsyncGenerator[dict[str, Any], None]:
        yield {"type": "response.output_text.delta", "delta": "Hello"}
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "id": "call_1",
                "name": tool_name,
                "arguments": "",
            },
        }
        yield {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": tool_name,
                "arguments": "",
            },
        }
        yield {
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": tool_name,
                        "arguments": {"a": 1},
                    },
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        }

    return _gen()


@pytest.mark.asyncio
async def test_tool_passthrough_nonstreaming_returns_tool_calls(monkeypatch) -> None:
    pipe = Pipe()
    valves = pipe.valves.model_copy(update={"TOOL_EXECUTION_MODE": "Open-WebUI"})

    async def fake_send_events(*_args: Any, **_kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        async for event in _fake_responses_events_with_tool_call():
            yield event

    async def fail_execute(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("_execute_function_calls should not run in Open-WebUI mode")

    monkeypatch.setattr(Pipe, "send_openrouter_nonstreaming_request_as_events", fake_send_events)
    monkeypatch.setattr(Pipe, "_execute_function_calls", fail_execute)

    try:
        body = ResponsesBody.model_validate(
            {
                "model": "pipe.model",
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
                "stream": False,
            }
        )
        result = await pipe._run_nonstreaming_loop(
            body,
            valves,
            event_emitter=None,
            metadata={"model": {"id": "pipe.model"}},
            tools={},
            session=cast(Any, object()),
            user_id="u1",
        )

        assert isinstance(result, dict)
        choice0 = (result.get("choices") or [{}])[0]
        assert choice0.get("finish_reason") == "tool_calls"
        message = choice0.get("message") or {}
        assert message.get("role") == "assistant"
        assert message.get("content") == "Hello"
        tool_calls = message.get("tool_calls") or []
        assert tool_calls and tool_calls[0]["function"]["name"] == "my_tool"
    finally:
        await pipe.close()


@pytest.mark.asyncio
async def test_tool_passthrough_streaming_emits_tool_calls_event(monkeypatch) -> None:
    pipe = Pipe()
    valves = pipe.valves.model_copy(update={"TOOL_EXECUTION_MODE": "Open-WebUI"})
    captured: list[dict[str, Any]] = []

    async def fake_send_stream(*_args: Any, **_kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        async for event in _fake_responses_events_with_tool_call():
            yield event

    async def fail_execute(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("_execute_function_calls should not run in Open-WebUI mode")

    async def capture_emitter(event: dict[str, Any]) -> None:
        captured.append(event)

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", fake_send_stream)
    monkeypatch.setattr(Pipe, "_execute_function_calls", fail_execute)

    try:
        body = ResponsesBody.model_validate(
            {
                "model": "pipe.model",
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
                "stream": True,
            }
        )
        await pipe._run_streaming_loop(
            body,
            valves,
            event_emitter=capture_emitter,
            metadata={"model": {"id": "pipe.model"}},
            tools={},
            session=cast(Any, object()),
            user_id="u1",
        )

        tool_events = [e for e in captured if e.get("type") == "chat:tool_calls"]
        assert tool_events, "Expected chat:tool_calls to be emitted in streaming pass-through mode"
        payload = tool_events[-1].get("data") or {}
        tool_calls = payload.get("tool_calls") or []
        assert tool_calls and tool_calls[0]["function"]["name"] == "my_tool"
        assert isinstance(tool_calls[0]["function"].get("arguments"), str)
        assert tool_calls[0]["function"]["arguments"].strip()
        assert '"a"' in tool_calls[0]["function"]["arguments"]
    finally:
        await pipe.close()


@pytest.mark.asyncio
async def test_transform_messages_to_input_replays_owui_tool_results() -> None:
    pipe = Pipe()
    try:
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "my_tool", "arguments": "{\"a\":1}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "OK"},
        ]
        replayed: list[tuple[str, str]] = []
        input_items = await pipe.transform_messages_to_input(
            messages,
            chat_id=None,
            openwebui_model_id=None,
            artifact_loader=None,
            pruning_turns=0,
            replayed_reasoning_refs=replayed,
            __request__=None,
            user_obj=None,
            event_emitter=None,
            model_id="pipe.model",
            valves=pipe.valves,
        )

        function_calls = [i for i in input_items if i.get("type") == "function_call"]
        function_outputs = [i for i in input_items if i.get("type") == "function_call_output"]
        assert function_calls and function_calls[0]["call_id"] == "call_1"
        assert function_calls[0]["name"] == "my_tool"
        assert function_calls[0]["arguments"] == "{\"a\":1}"
        assert function_outputs and function_outputs[0]["call_id"] == "call_1"
        assert function_outputs[0]["output"] == "OK"
    finally:
        await pipe.close()


def test_build_tools_openwebui_mode_keeps_tools_and_does_not_strictify(monkeypatch) -> None:
    pipe = Pipe()
    valves = pipe.valves.model_copy(
        update={"TOOL_EXECUTION_MODE": "Open-WebUI", "ENABLE_STRICT_TOOL_CALLING": True}
    )

    # Even if ModelFamily thinks tools are unsupported, pass-through mode should forward them.
    from open_webui_openrouter_pipe import open_webui_openrouter_pipe as mod

    monkeypatch.setattr(mod.ModelFamily, "supports", lambda *_args, **_kwargs: False)

    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    tools = build_tools(
        ResponsesBody.model_validate({"model": "pipe.model", "input": [], "stream": False}),
        valves,
        __tools__={"my_tool": {"spec": {"name": "my_tool", "parameters": schema}}},
    )
    assert tools and tools[0]["name"] == "my_tool"
    assert "strict" not in tools[0]
    assert tools[0]["parameters"] == schema


@pytest.mark.asyncio
async def test_tool_passthrough_streaming_does_not_repeat_function_name(monkeypatch) -> None:
    pipe = Pipe()
    valves = pipe.valves.model_copy(update={"TOOL_EXECUTION_MODE": "Open-WebUI"})
    captured: list[dict[str, Any]] = []

    async def fake_send_stream(*_args: Any, **_kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        async def _gen() -> AsyncGenerator[dict[str, Any], None]:
            yield {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "id": "call_1",
                    "name": "my_tool",
                    "arguments": "",
                },
            }
            yield {
                "type": "response.function_call_arguments.delta",
                "item_id": "call_1",
                "name": "my_tool",
                "delta": "{\"a\":",
            }
            yield {
                "type": "response.function_call_arguments.delta",
                "item_id": "call_1",
                "name": "my_tool",
                "delta": "1}",
            }
            yield {
                "type": "response.completed",
                "response": {
                    "output": [
                        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": ""}]},
                        {"type": "function_call", "call_id": "call_1", "name": "my_tool", "arguments": "{\"a\":1}"},
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            }

        async for event in _gen():
            yield event

    async def fail_execute(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("_execute_function_calls should not run in Open-WebUI mode")

    async def capture_emitter(event: dict[str, Any]) -> None:
        captured.append(event)

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", fake_send_stream)
    monkeypatch.setattr(Pipe, "_execute_function_calls", fail_execute)

    try:
        body = ResponsesBody.model_validate(
            {
                "model": "pipe.model",
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
                "stream": True,
            }
        )
        await pipe._run_streaming_loop(
            body,
            valves,
            event_emitter=capture_emitter,
            metadata={"model": {"id": "pipe.model"}},
            tools={},
            session=cast(Any, object()),
            user_id="u1",
        )

        tool_events = [e for e in captured if e.get("type") == "chat:tool_calls"]
        assert len(tool_events) == 2
        data0 = tool_events[0].get("data")
        assert isinstance(data0, dict)
        calls0 = data0.get("tool_calls")
        assert isinstance(calls0, list) and calls0
        first_fn = calls0[0].get("function")
        assert isinstance(first_fn, dict)

        data1 = tool_events[1].get("data")
        assert isinstance(data1, dict)
        calls1 = data1.get("tool_calls")
        assert isinstance(calls1, list) and calls1
        second_fn = calls1[0].get("function")
        assert isinstance(second_fn, dict)
        assert first_fn.get("name") == "my_tool"
        assert "name" not in second_fn
        assert f"{first_fn.get('arguments', '')}{second_fn.get('arguments', '')}" == "{\"a\":1}"
    finally:
        await pipe.close()
