from __future__ import annotations

import json
from typing import Any

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import OpenRouterModelRegistry, Pipe


@pytest.mark.asyncio
async def test_owui_metadata_tools_registry_is_used_for_native_tools(monkeypatch):
    """OWUI 0.7.x puts builtin tool executors in __metadata__["tools"], not __tools__.

    This test verifies that:
    1. Tools from __metadata__["tools"] are properly extracted
    2. They are passed to the OpenRouter API request
    3. The real infrastructure processes tool calling correctly
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key-12345")

    pipe = Pipe()
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    # Clear registry to ensure we fetch from mocked endpoint
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._last_fetch = 0.0

    # Track if tool was called
    tool_called = False
    tool_call_args = None

    async def builtin_search_web(query: str):
        nonlocal tool_called, tool_call_args
        tool_called = True
        tool_call_args = {"query": query}
        return {"ok": True, "query": query, "results": ["result1", "result2"]}

    # Build metadata with tool registry (OWUI 0.7.x style)
    metadata = {
        "chat_id": "c1",
        "message_id": "m1",
        "session_id": "s1",
        "model": {"id": "openai/gpt-4o-mini"},
        # In OWUI 0.7.x, middleware attaches the tool registry to metadata["tools"]
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

    # Mock HTTP at boundary
    with aioresponses() as mock_http:
        # Mock catalog endpoint
        catalog_response = {
            "data": [
                {
                    "id": "openai/gpt-4o-mini",
                    "name": "GPT-4o Mini",
                    "context_length": 128000,
                    "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
                    "supported_parameters": ["tools", "tool_choice"],
                }
            ]
        }
        mock_http.get(
            "https://openrouter.ai/api/v1/models",
            payload=catalog_response,
            headers={"Content-Type": "application/json"},
        )

        # Mock the API response with a tool call (using /responses endpoint)
        # In Responses API, function calls are separate items with type:"function_call"
        api_response = {
            "id": "gen-123",
            "model": "openai/gpt-4o-mini",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_abc123",
                    "name": "search_web",
                    "arguments": json.dumps({"query": "hello world"}),
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=api_response,
            headers={"Content-Type": "application/json"},
        )

        # Mock the follow-up response after tool execution
        final_response = {
            "id": "gen-124",
            "model": "openai/gpt-4o-mini",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I found some results for you!"}],
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        }
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=final_response,
            headers={"Content-Type": "application/json"},
        )

        try:
            body = {
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "search for hello world"}],
                "stream": False,
                # Native tools are sent in OpenAI-style chat tool schema by OWUI middleware
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

            result = await pipe.pipe(
                body=body,
                __user__={"id": "u1"},
                __request__=None,
                __event_emitter__=None,
                __event_call__=None,
                __metadata__=metadata,
                __tools__={},  # Empty - tools come from metadata["tools"]
            )

            # Verify tool was actually called through the real infrastructure
            # This is the KEY assertion: tools from metadata["tools"] must be used
            assert tool_called, "Tool should have been called through real execution"
            assert tool_call_args is not None
            assert tool_call_args["query"] == "hello world"

            # Verify the result is valid (non-streaming with event_emitter=None returns string)
            assert result is not None
            assert isinstance(result, str)

        finally:
            await session.close()
            await pipe.close()
            # Clear the registry to avoid polluting other tests
            OpenRouterModelRegistry._models = []
            OpenRouterModelRegistry._specs = {}
            OpenRouterModelRegistry._id_map = {}
            OpenRouterModelRegistry._last_fetch = 0.0
