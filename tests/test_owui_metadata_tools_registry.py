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
async def test_owui_metadata_tools_registry_is_used_for_native_tools(monkeypatch):
    """OWUI 0.7.x puts builtin tool executors in __metadata__["tools"], not __tools__."""
    pipe = Pipe()

    async def fake_transform(_self, *_args, **_kwargs):
        return list(_STUBBED_INPUT)

    async def builtin_search_web(query: str):
        return {"ok": True, "query": query}

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
        assert any(t.get("type") == "function" and t.get("name") == "search_web" for t in body.tools)

        assert isinstance(tools, dict)
        assert "search_web" in tools
        tool_cfg = tools["search_web"]
        assert callable(tool_cfg.get("callable"))

        result = await tool_cfg["callable"](query="hello")
        assert result == {"ok": True, "query": "hello"}
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
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            # Native tools are sent in OpenAI-style chat tool schema by OWUI middleware.
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "Search the web.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
        }
        metadata = {
            "chat_id": "c1",
            "message_id": "m1",
            "session_id": "s1",
            "model": {"id": "test-model"},
            # In OWUI 0.7.x, middleware attaches the tool registry to metadata["tools"].
            "tools": {
                "search_web": {
                    "tool_id": "builtin:search_web",
                    "type": "builtin",
                    "spec": {
                        "name": "search_web",
                        "description": "Search the web.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                    "callable": builtin_search_web,
                }
            },
        }

        result = await pipe._process_transformed_request(
            body,
            __user__={"id": "u1", "valves": {}},
            __request__=None,
            __event_emitter__=None,
            __event_call__=cast(Callable[[dict[str, Any]], Awaitable[Any]], None),
            __metadata__=metadata,
            __tools__={},
            __task__=None,
            __task_body__=None,
            valves=pipe.valves,
            session=cast(Any, object()),
            openwebui_model_id="test-model",
            pipe_identifier="pipe.test",
            allowlist_norm_ids=set(),
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )
        assert result == "ok"
    finally:
        await pipe.close()

