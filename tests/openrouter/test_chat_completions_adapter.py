from __future__ import annotations

import json
from typing import Any, AsyncIterator, cast

import pytest

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import (
    OpenRouterAPIError,
    Pipe,
    _responses_payload_to_chat_completions_payload,
)


class _FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        status: int = 200,
        reason: str = "OK",
        json_payload: dict[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.reason = reason
        self.headers: dict[str, str] = {}
        self.content = _FakeContent(chunks)
        self._json_payload = json_payload

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def json(self) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        return self._json_payload or {}

    async def text(self) -> str:  # type: ignore[no-untyped-def]
        return json.dumps(self._json_payload or {})


class _FakeRequestCtx:
    def __init__(self, resp: _FakeResponse) -> None:
        self._resp = resp

    async def __aenter__(self) -> _FakeResponse:
        return self._resp

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None


class _FakeSession:
    def __init__(self, resp: _FakeResponse) -> None:
        self._resp = resp
        self.posts: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> _FakeRequestCtx:
        self.posts.append((url, json, headers))
        return _FakeRequestCtx(self._resp)


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


def test_force_chat_completions_models_matches_slash_glob() -> None:
    pipe = Pipe()
    valves = pipe.valves.model_copy(
        update={
            "DEFAULT_LLM_ENDPOINT": "responses",
            "FORCE_CHAT_COMPLETIONS_MODELS": "anthropic/*",
        }
    )
    assert pipe._select_llm_endpoint("anthropic/claude-sonnet-4.5", valves=valves) == "chat_completions"
    assert pipe._select_llm_endpoint("anthropic.claude-sonnet-4.5", valves=valves) == "chat_completions"


def test_force_responses_models_overrides_force_chat_models() -> None:
    pipe = Pipe()
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
    pipe = Pipe()
    valves = pipe.valves.model_copy(
        update={
            "DEFAULT_LLM_ENDPOINT": "chat_completions",
        }
    )

    def _sse(obj: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(obj)}\n\n".encode("utf-8")

    sse = [
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
        ),
        _sse({"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}),
        _sse(
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
        ),
        _sse(
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
        ),
        _sse(
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
        ),
        _sse(
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
        ),
        b"data: [DONE]\n\n",
    ]
    fake_resp = _FakeResponse(sse)
    fake_session = _FakeSession(fake_resp)

    events: list[dict[str, Any]] = []
    async for event in pipe.send_openai_chat_completions_streaming_request(
        cast(Any, fake_session),
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

    assert fake_session.posts, "Expected /chat/completions HTTP call"
    url, body, _headers = fake_session.posts[0]
    assert url.endswith("/chat/completions")
    assert body["stream"] is True
    assert body["stream_options"]["include_usage"] is True
    assert body["usage"] == {"include": True}
    assert body["min_p"] == 0.1
    assert body["top_a"] == 0.2
    assert body["repetition_penalty"] == 1.02
    assert body["structured_outputs"] is True
    assert body["include_reasoning"] is True
    assert body["reasoning_effort"] == "medium"
    assert body["verbosity"] == "low"
    assert body["web_search_options"] == {"search_context_size": "high"}


@pytest.mark.asyncio
async def test_chat_completions_inlines_internal_file_urls(monkeypatch) -> None:
    pipe = Pipe()
    valves = pipe.valves.model_copy(update={"DEFAULT_LLM_ENDPOINT": "chat_completions"})

    async def fake_inline(url: str, *, chunk_size: int, max_bytes: int) -> str | None:  # type: ignore[no-untyped-def]
        assert url == "/api/v1/files/abc123"
        return "data:application/pdf;base64,QUJD"

    monkeypatch.setattr(pipe, "_inline_internal_file_url", fake_inline)

    fake_resp = _FakeResponse([b"data: [DONE]\n\n"])
    fake_session = _FakeSession(fake_resp)

    async for _event in pipe.send_openai_chat_completions_streaming_request(
        cast(Any, fake_session),
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

    assert fake_session.posts, "Expected /chat/completions HTTP call"
    _url, body, _headers = fake_session.posts[0]
    blocks = body["messages"][0]["content"]
    file_blocks = [b for b in blocks if isinstance(b, dict) and b.get("type") == "file"]
    assert file_blocks, "Expected chat file content block"
    assert file_blocks[0]["file"]["file_data"] == "data:application/pdf;base64,QUJD"


@pytest.mark.asyncio
async def test_transform_messages_to_input_preserves_annotations() -> None:
    pipe = Pipe()
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

@pytest.mark.asyncio
async def test_send_openrouter_streaming_request_falls_back_to_chat(monkeypatch) -> None:
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

    got: list[dict[str, Any]] = []
    async for event in pipe.send_openrouter_streaming_request(
        cast(Any, object()),
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


@pytest.mark.asyncio
async def test_send_openrouter_streaming_request_does_not_fallback_after_output(monkeypatch) -> None:
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

    got: list[dict[str, Any]] = []
    with pytest.raises(OpenRouterAPIError):
        async for event in pipe.send_openrouter_streaming_request(
            cast(Any, object()),
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


@pytest.mark.asyncio
async def test_send_openrouter_streaming_request_fallback_discards_nonvisible_events(monkeypatch) -> None:
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

    got: list[dict[str, Any]] = []
    async for event in pipe.send_openrouter_streaming_request(
        cast(Any, object()),
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


@pytest.mark.asyncio
async def test_send_openrouter_nonstreaming_request_as_events_chat_adapter(monkeypatch) -> None:
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
                            "function": {"name": "lookup", "arguments": "{\"foo\": 1}"},
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
    fake_resp = _FakeResponse([], json_payload=resp_payload)
    fake_session = _FakeSession(fake_resp)

    events: list[dict[str, Any]] = []
    async for event in pipe.send_openrouter_nonstreaming_request_as_events(
        cast(Any, fake_session),
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

    assert fake_session.posts, "Expected /chat/completions HTTP call"
    _url, body, _headers = fake_session.posts[0]
    assert body["stream"] is False
    assert body["usage"] == {"include": True}


@pytest.mark.asyncio
async def test_chat_completions_nonstreaming_inlines_internal_file_urls(monkeypatch) -> None:
    pipe = Pipe()
    valves = pipe.valves.model_copy(update={"DEFAULT_LLM_ENDPOINT": "chat_completions"})

    async def fake_inline(url: str, *, chunk_size: int, max_bytes: int) -> str | None:  # type: ignore[no-untyped-def]
        assert url == "/api/v1/files/abc123"
        return "data:application/pdf;base64,QUJD"

    monkeypatch.setattr(pipe, "_inline_internal_file_url", fake_inline)

    fake_resp = _FakeResponse([], json_payload={"choices": [{"message": {"role": "assistant", "content": ""}}]})
    fake_session = _FakeSession(fake_resp)

    _ = await pipe.send_openai_chat_completions_nonstreaming_request(
        cast(Any, fake_session),
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

    assert fake_session.posts, "Expected /chat/completions HTTP call"
    _url, body, _headers = fake_session.posts[0]
    blocks = body["messages"][0]["content"]
    file_blocks = [b for b in blocks if isinstance(b, dict) and b.get("type") == "file"]
    assert file_blocks, "Expected chat file content block"
    assert file_blocks[0]["file"]["file_data"] == "data:application/pdf;base64,QUJD"
