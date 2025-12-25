from __future__ import annotations

from typing import Any, Awaitable, Callable, cast

import pytest

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import Pipe


_STUBBED_INPUT = [
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hi"}],
    }
]


@pytest.mark.asyncio
async def test_direct_tool_servers_are_advertised_and_executable(monkeypatch):
    pipe = Pipe()
    captured: dict[str, Any] = {"event_payloads": []}

    async def fake_transform(_self, *_args, **_kwargs):
        return list(_STUBBED_INPUT)

    async def fake_event_call(payload: dict[str, Any]):
        captured["event_payloads"].append(payload)
        return [{"ok": True}, None]

    async def fake_run_nonstreaming_loop(
        self,
        body,
        valves,
        event_emitter,
        metadata,
        tools,
        *,
        session,
        user_id,
        **_kwargs,
    ):
        assert body.tools is not None
        assert any(t.get("type") == "function" and t.get("name") == "getWeather" for t in body.tools)

        assert isinstance(tools, dict)
        assert "getWeather" in tools
        tool_cfg = tools["getWeather"]
        assert callable(tool_cfg.get("callable"))

        result = await tool_cfg["callable"](location="Paris", ignored=123)
        assert result == [{"ok": True}, None]

        payload = captured["event_payloads"][-1]
        assert payload.get("type") == "execute:tool"
        data = payload.get("data") or {}
        assert data.get("name") == "getWeather"
        assert data.get("session_id") == "s1"
        assert data.get("params") == {"location": "Paris"}
        assert (data.get("server") or {}).get("url") == "https://example.com"
        return "ok"

    def fake_supports(cls, feature: str, _model_id: str | None = None):
        return feature == "function_calling"

    monkeypatch.setattr(Pipe, "transform_messages_to_input", fake_transform)
    monkeypatch.setattr(Pipe, "_run_nonstreaming_loop", fake_run_nonstreaming_loop)
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelFamily.supports",
        classmethod(fake_supports),
    )

    try:
        body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": False}
        metadata = {
            "chat_id": "c1",
            "message_id": "m1",
            "session_id": "s1",
            "model": {"id": "test-model"},
            "tool_servers": [
                {
                    "url": "https://example.com",
                    "specs": [
                        {
                            "name": "getWeather",
                            "description": "Fetch weather.",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        }
                    ],
                }
            ],
        }

        result = await pipe._process_transformed_request(
            body,
            __user__={"id": "u1", "valves": {}},
            __request__=None,
            __event_emitter__=None,
            __event_call__=cast(Callable[[dict[str, Any]], Awaitable[Any]], fake_event_call),
            __metadata__=metadata,
            __tools__={},
            __task__=None,
            __task_body__=None,
            valves=pipe.valves,
            session=cast(Any, object()),
            openwebui_model_id="test-model",
            pipe_identifier="pipe.test",
            pipe_identifier_for_artifacts="pipe.test",
            allowed_norm_ids=set(),
            features={},
        )
        assert result == "ok"
    finally:
        await pipe.close()


@pytest.mark.asyncio
async def test_direct_tool_servers_skipped_without_event_call(monkeypatch):
    pipe = Pipe()

    async def fake_transform(_self, *_args, **_kwargs):
        return list(_STUBBED_INPUT)

    async def fake_run_nonstreaming_loop(
        self,
        body,
        valves,
        event_emitter,
        metadata,
        tools,
        *,
        session,
        user_id,
        **_kwargs,
    ):
        # No event_call => direct tools should not be advertised.
        assert not (body.tools and any(t.get("name") == "getWeather" for t in body.tools))
        return "ok"

    def fake_supports(cls, feature: str, _model_id: str | None = None):
        return feature == "function_calling"

    monkeypatch.setattr(Pipe, "transform_messages_to_input", fake_transform)
    monkeypatch.setattr(Pipe, "_run_nonstreaming_loop", fake_run_nonstreaming_loop)
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelFamily.supports",
        classmethod(fake_supports),
    )

    try:
        body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": False}
        metadata = {
            "chat_id": "c1",
            "message_id": "m1",
            "session_id": "s1",
            "model": {"id": "test-model"},
            "tool_servers": [
                {
                    "url": "https://example.com",
                    "specs": [
                        {
                            "name": "getWeather",
                            "description": "Fetch weather.",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        }
                    ],
                }
            ],
        }

        result = await pipe._process_transformed_request(
            body,
            __user__={"id": "u1", "valves": {}},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__={},
            __task__=None,
            __task_body__=None,
            valves=pipe.valves,
            session=cast(Any, object()),
            openwebui_model_id="test-model",
            pipe_identifier="pipe.test",
            pipe_identifier_for_artifacts="pipe.test",
            allowed_norm_ids=set(),
            features={},
        )
        assert result == "ok"
    finally:
        await pipe.close()

