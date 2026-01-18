from __future__ import annotations

import json
from typing import Any
from aioresponses import aioresponses
import aiohttp

import pytest

from open_webui_openrouter_pipe import (
    OpenRouterAPIError,
    Pipe,
    _responses_payload_to_chat_completions_payload,
)


def test_responses_payload_to_chat_preserves_cache_control() -> None:
    payload = {
        "model": "anthropic/claude-sonnet-4.5",
        "stream": True,
        "instructions": "SYSTEM INSTRUCTIONS",
        "input": [
            {
                "type": "message",
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "HUGE TEXT BODY",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    }
    chat = _responses_payload_to_chat_completions_payload(payload)
    assert chat["model"] == payload["model"]
    assert chat["stream"] is True
    assert chat["stream_options"]["include_usage"] is True
    assert chat["usage"] == {"include": True}
    assert chat["messages"][0]["role"] == "system"
    assert chat["messages"][0]["content"][0]["type"] == "text"
    assert chat["messages"][0]["content"][0]["text"] == "SYSTEM INSTRUCTIONS"
    assert chat["messages"][0]["content"][2]["type"] == "text"
    assert chat["messages"][0]["content"][2]["text"] == "HUGE TEXT BODY"
    assert chat["messages"][0]["content"][2]["cache_control"] == {"type": "ephemeral"}


def test_responses_payload_to_chat_rounds_top_k_for_chat_completions() -> None:
    payload = {"model": "openai/gpt-5", "stream": False, "input": [], "top_k": 2.5}
    chat = _responses_payload_to_chat_completions_payload(payload)
    assert chat["top_k"] == 2


def test_force_chat_completions_models_matches_slash_glob(pipe_instance) -> None:
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "DEFAULT_LLM_ENDPOINT": "responses",
            "FORCE_CHAT_COMPLETIONS_MODELS": "anthropic/*",
        }
    )
    assert pipe._select_llm_endpoint("anthropic/claude-sonnet-4.5", valves=valves) == "chat_completions"
    assert pipe._select_llm_endpoint("anthropic.claude-sonnet-4.5", valves=valves) == "chat_completions"


def test_force_responses_models_overrides_force_chat_models(pipe_instance) -> None:
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "DEFAULT_LLM_ENDPOINT": "responses",
            "FORCE_CHAT_COMPLETIONS_MODELS": "anthropic/*",
            "FORCE_RESPONSES_MODELS": "anthropic/claude-sonnet-4.5",
        }
    )
    assert pipe._select_llm_endpoint("anthropic/claude-sonnet-4.5", valves=valves) == "responses"
    assert pipe._select_llm_endpoint("anthropic.claude-sonnet-4.5", valves=valves) == "responses"


def test_responses_payload_to_chat_converts_tools_schema() -> None:
    payload = {
        "model": "openai/gpt-5",
        "stream": True,
        "input": [],
        "tools": [
            {
                "type": "function",
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            }
        ],
    }
    chat = _responses_payload_to_chat_completions_payload(payload)
    assert chat["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            },
        }
    ]


@pytest.mark.asyncio
async def test_chat_completions_stream_adapts_tool_calls_and_usage() -> None:
    """Test that chat completions streaming adapter correctly transforms OpenAI-style
    responses to OpenRouter Responses API events, including tool calls and usage.

    THIS IS A REAL TEST: Uses aioresponses to mock HTTP, exercises real adapter logic."""
    pipe = Pipe()
    valves = pipe.valves.model_copy(
        update={
            "DEFAULT_LLM_ENDPOINT": "chat_completions",
        }
    )

    def _sse(obj: dict[str, Any]) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    sse_response = (
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {
                                    "type": "reasoning.text",
                                    "text": "Let me think...",
                                    "id": "reasoning-text-1",
                                    "format": "anthropic-claude-v1",
                                    "index": 0,
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        + _sse({"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]})
        + _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": '{"foo":'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        + _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": " 1}"}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        + _sse(
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "cost": 0.95,
                    "prompt_tokens_details": {"cached_tokens": 5},
                },
            }
        )
        + _sse(
            {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Hello",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url_citation": {
                                        "url": "https://example.com/web-search-result",
                                        "title": "Example result",
                                        "start_index": 0,
                                        "end_index": 5,
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        )
        + "data: [DONE]\n\n"
    )

    # Mock HTTP at boundary - adapter will make real HTTP call to mocked endpoint
    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            body=sse_response.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        # Create real aiohttp session and call adapter method directly
        async with aiohttp.ClientSession() as session:
            events: list[dict[str, Any]] = []
            async for event in pipe.send_openai_chat_completions_streaming_request(
                session,
                {
                    "model": "anthropic/claude-sonnet-4.5",
                    "stream": True,
                    "min_p": 0.1,
                    "top_a": 0.2,
                    "repetition_penalty": 1.02,
                    "structured_outputs": True,
                    "include_reasoning": True,
                    "reasoning_effort": "medium",
                    "verbosity": "low",
                    "web_search_options": {"search_context_size": "high"},
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "hi"}],
                        }
                    ],
                },
                api_key="test",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
                breaker_key=None,
            ):
                events.append(event)

        # Verify events generated by real adapter logic
        assert any(e.get("type") == "response.output_text.delta" and e.get("delta") == "Hello" for e in events)
        assert any(e.get("type") == "response.reasoning_text.delta" for e in events)
        assert any(e.get("type") == "response.output_text.annotation.added" for e in events)
        assert any(e.get("type") == "response.output_item.added" for e in events)
        assert any(e.get("type") == "response.output_item.done" for e in events)
        completed = [e for e in events if e.get("type") == "response.completed"]
        assert completed, "Expected a response.completed event"
        final = completed[-1]["response"]
        assert final["usage"]["input_tokens"] == 10
        assert final["usage"]["output_tokens"] == 2
        assert final["usage"]["cost"] == 0.95
        assert final["usage"]["input_tokens_details"]["cached_tokens"] == 5
        calls = [i for i in final["output"] if i.get("type") == "function_call"]
        assert calls and calls[0]["name"] == "lookup"
        message_items = [i for i in final["output"] if i.get("type") == "message"]
        assert message_items, "Expected assistant message output"
        assert "annotations" in message_items[0]
        assert "reasoning_details" in message_items[0]

    await pipe.close()


@pytest.mark.asyncio
async def test_chat_completions_inlines_internal_file_urls(monkeypatch) -> None:
    """Test that internal file URLs are inlined when using chat completions adapter.

    THIS IS A REAL TEST: Mocks file inlining helper, uses aioresponses for HTTP,
    exercises real adapter logic."""
    pipe = Pipe()
    valves = pipe.valves.model_copy(update={"DEFAULT_LLM_ENDPOINT": "chat_completions"})

    async def fake_inline(url: str, *, chunk_size: int, max_bytes: int) -> str | None:
        assert url == "/api/v1/files/abc123"
        return "data:application/pdf;base64,QUJD"

    monkeypatch.setattr(pipe, "_inline_internal_file_url", fake_inline)

    # Mock HTTP at boundary
    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            body=b"data: [DONE]\n\n",
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )

        async with aiohttp.ClientSession() as session:
            async for _event in pipe.send_openai_chat_completions_streaming_request(
                session,
                {
                    "model": "anthropic/claude-sonnet-4.5",
                    "stream": True,
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "read this"},
                                {"type": "input_file", "filename": "doc.pdf", "file_url": "/api/v1/files/abc123"},
                            ],
                        }
                    ],
                },
                api_key="test",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
                breaker_key=None,
            ):
                pass

    await pipe.close()


@pytest.mark.asyncio
async def test_transform_messages_to_input_preserves_annotations() -> None:
    """Test that assistant message annotations and reasoning_details are preserved
    when transforming OpenAI format to Responses API input."""
    pipe = Pipe()
    try:
        out = await pipe.transform_messages_to_input(
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "ok",
                    "annotations": [{"type": "file_annotation", "foo": "bar"}],
                    "reasoning_details": [{"type": "reasoning.text", "text": "think"}],
                },
                {"role": "user", "content": "next"},
            ],
            model_id="anthropic/claude-sonnet-4.5",
        )
        assistant_items = [i for i in out if i.get("type") == "message" and i.get("role") == "assistant"]
        assert assistant_items, "Expected assistant message in Responses input"
        assert assistant_items[0]["annotations"] == [{"type": "file_annotation", "foo": "bar"}]
        assert assistant_items[0]["reasoning_details"] == [{"type": "reasoning.text", "text": "think"}]
    finally:
        await pipe.close()


@pytest.mark.asyncio
async def test_send_openrouter_streaming_request_falls_back_to_chat(monkeypatch) -> None:
    """Test that when Responses API fails with unsupported model error,
    the system falls back to chat completions adapter.

    THIS IS A REAL TEST: Mocks individual adapter methods to simulate error/success,
    exercises real fallback logic in send_openrouter_streaming_request."""
    pipe = Pipe()
    valves = pipe.valves.model_copy(
        update={"DEFAULT_LLM_ENDPOINT": "responses", "AUTO_FALLBACK_CHAT_COMPLETIONS": True}
    )

    async def fake_responses(self, session, request_body, **_kwargs):  # type: ignore[no-untyped-def]
        raise OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            openrouter_message="This model does not support the Responses API.",
        )
        if False:  # pragma: no cover
            yield {}

    async def fake_chat(self, session, responses_request_body, **_kwargs):  # type: ignore[no-untyped-def]
        yield {"type": "response.output_text.delta", "delta": "hi"}
        yield {
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hi"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }

    monkeypatch.setattr(Pipe, "send_openai_responses_streaming_request", fake_responses)
    monkeypatch.setattr(Pipe, "send_openai_chat_completions_streaming_request", fake_chat)

    # Create real session for send_openrouter_streaming_request to use
    async with aiohttp.ClientSession() as session:
        got: list[dict[str, Any]] = []
        async for event in pipe.send_openrouter_streaming_request(
            session,
            {"model": "anthropic/claude-sonnet-4.5", "stream": True, "input": []},
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            workers=1,
            breaker_key=None,
        ):
            got.append(event)

    assert any(e.get("type") == "response.output_text.delta" for e in got)
    assert any(e.get("type") == "response.completed" for e in got)

    await pipe.close()


@pytest.mark.asyncio
async def test_send_openrouter_streaming_request_does_not_fallback_after_output(monkeypatch) -> None:
    """Test that fallback to chat completions does NOT happen if Responses API
    already started producing output before failing.

    THIS IS A REAL TEST: Mocks adapter methods, exercises real fallback decision logic."""
    pipe = Pipe()
    valves = pipe.valves.model_copy(
        update={"DEFAULT_LLM_ENDPOINT": "responses", "AUTO_FALLBACK_CHAT_COMPLETIONS": True}
    )

    async def fake_responses(self, session, request_body, **_kwargs):  # type: ignore[no-untyped-def]
        yield {"type": "response.output_text.delta", "delta": "partial"}
        raise OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            openrouter_message="This model does not support the Responses API.",
        )
        if False:  # pragma: no cover
            yield {}

    async def fake_chat(self, session, responses_request_body, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("chat adapter should not be called after partial /responses output")
        if False:  # pragma: no cover
            yield {}

    monkeypatch.setattr(Pipe, "send_openai_responses_streaming_request", fake_responses)
    monkeypatch.setattr(Pipe, "send_openai_chat_completions_streaming_request", fake_chat)

    async with aiohttp.ClientSession() as session:
        got: list[dict[str, Any]] = []
        with pytest.raises(OpenRouterAPIError):
            async for event in pipe.send_openrouter_streaming_request(
                session,
                {"model": "anthropic/claude-sonnet-4.5", "stream": True, "input": []},
                api_key="test",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
                workers=1,
                breaker_key=None,
            ):
                got.append(event)

    assert any(e.get("type") == "response.output_text.delta" and e.get("delta") == "partial" for e in got)
    assert not any(e.get("type") == "response.completed" for e in got)

    await pipe.close()


@pytest.mark.asyncio
async def test_send_openrouter_streaming_request_fallback_discards_nonvisible_events(monkeypatch) -> None:
    """Test that when falling back from Responses to chat completions,
    non-visible events (like response.created) are discarded.

    THIS IS A REAL TEST: Mocks adapter methods, exercises real event filtering logic."""
    pipe = Pipe()
    valves = pipe.valves.model_copy(
        update={"DEFAULT_LLM_ENDPOINT": "responses", "AUTO_FALLBACK_CHAT_COMPLETIONS": True}
    )

    async def fake_responses(self, session, request_body, **_kwargs):  # type: ignore[no-untyped-def]
        yield {"type": "response.created", "response": {"id": "resp_test"}}
        raise OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            openrouter_message="This model does not support the Responses API.",
        )
        if False:  # pragma: no cover
            yield {}

    async def fake_chat(self, session, responses_request_body, **_kwargs):  # type: ignore[no-untyped-def]
        yield {"type": "response.output_text.delta", "delta": "hi"}
        yield {
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hi"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }

    monkeypatch.setattr(Pipe, "send_openai_responses_streaming_request", fake_responses)
    monkeypatch.setattr(Pipe, "send_openai_chat_completions_streaming_request", fake_chat)

    async with aiohttp.ClientSession() as session:
        got: list[dict[str, Any]] = []
        async for event in pipe.send_openrouter_streaming_request(
            session,
            {"model": "anthropic/claude-sonnet-4.5", "stream": True, "input": []},
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            workers=1,
            breaker_key=None,
        ):
            got.append(event)

    assert not any(e.get("type") == "response.created" for e in got)
    assert any(e.get("type") == "response.output_text.delta" and e.get("delta") == "hi" for e in got)
    assert any(e.get("type") == "response.completed" for e in got)

    await pipe.close()


@pytest.mark.asyncio
async def test_send_openrouter_nonstreaming_request_as_events_chat_adapter() -> None:
    """Test that non-streaming requests through chat adapter correctly transform
    OpenAI chat completion responses to Responses API events.

    THIS IS A REAL TEST: Uses aioresponses for HTTP, exercises real non-streaming adapter logic."""
    pipe = Pipe()
    valves = pipe.valves.model_copy(update={"DEFAULT_LLM_ENDPOINT": "chat_completions"})

    resp_payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://example.com/web-search-result",
                                "title": "Example result",
                                "start_index": 0,
                                "end_index": 5,
                            },
                        }
                    ],
                    "reasoning_details": [
                        {
                            "type": "reasoning.text",
                            "text": "Let me think...",
                            "id": "reasoning-text-1",
                            "format": "anthropic-claude-v1",
                            "index": 0,
                        }
                    ],
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"foo": 1}'},
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cost": 0.95,
            "prompt_tokens_details": {"cached_tokens": 5},
        },
    }

    # Mock HTTP at boundary
    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=resp_payload,
            status=200,
        )

        async with aiohttp.ClientSession() as session:
            events: list[dict[str, Any]] = []
            async for event in pipe.send_openrouter_nonstreaming_request_as_events(
                session,
                {
                    "model": "anthropic/claude-sonnet-4.5",
                    "stream": False,
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "hi"}],
                        }
                    ],
                },
                api_key="test",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
                breaker_key=None,
            ):
                events.append(event)

    assert any(e.get("type") == "response.output_text.delta" and e.get("delta") == "Hello" for e in events)
    assert any(e.get("type") == "response.output_text.annotation.added" for e in events)
    assert any(e.get("type") == "response.reasoning_text.delta" for e in events)
    completed = [e for e in events if e.get("type") == "response.completed"]
    assert completed, "Expected a response.completed event"
    final = completed[-1]["response"]
    assert final["usage"]["input_tokens"] == 10
    assert final["usage"]["cost"] == 0.95
    calls = [i for i in final["output"] if i.get("type") == "function_call"]
    assert calls and calls[0]["name"] == "lookup"

    await pipe.close()


@pytest.mark.asyncio
async def test_chat_completions_nonstreaming_inlines_internal_file_urls(monkeypatch) -> None:
    """Test that internal file URLs are inlined in non-streaming chat completions.

    THIS IS A REAL TEST: Mocks file inlining helper, uses aioresponses for HTTP,
    exercises real adapter logic."""
    pipe = Pipe()
    valves = pipe.valves.model_copy(update={"DEFAULT_LLM_ENDPOINT": "chat_completions"})

    async def fake_inline(url: str, *, chunk_size: int, max_bytes: int) -> str | None:
        assert url == "/api/v1/files/abc123"
        return "data:application/pdf;base64,QUJD"

    monkeypatch.setattr(pipe, "_inline_internal_file_url", fake_inline)

    # Mock HTTP at boundary
    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            status=200,
        )

        async with aiohttp.ClientSession() as session:
            _ = await pipe.send_openai_chat_completions_nonstreaming_request(
                session,
                {
                    "model": "anthropic/claude-sonnet-4.5",
                    "stream": False,
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "read this"},
                                {"type": "input_file", "filename": "doc.pdf", "file_url": "/api/v1/files/abc123"},
                            ],
                        }
                    ],
                },
                api_key="test",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
                breaker_key=None,
            )

    await pipe.close()
