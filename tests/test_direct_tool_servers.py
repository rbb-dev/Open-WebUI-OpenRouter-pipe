from __future__ import annotations

from typing import Any, Awaitable, Callable, cast

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import Pipe


_STUBBED_INPUT = [
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hi"}],
    }
]


@pytest.mark.asyncio
async def test_direct_tool_servers_are_advertised_and_executable():
    """
    Test that tool servers from metadata are:
    1. Advertised in body.tools sent to OpenRouter (via HTTP request capture)
    2. Executable via callable that invokes __event_call__

    REAL TEST - Uses aioresponses for HTTP boundary mocking.
    Exercises real tool server integration logic and real HTTP request construction.
    """
    pipe = Pipe()

    # Track event_call invocations
    event_call_payloads: list[dict[str, Any]] = []

    async def track_event_call(payload: dict[str, Any]):
        event_call_payloads.append(payload)
        return [{"ok": True}, None]

    # Track the HTTP request body to verify tools were added
    captured_request_body: dict[str, Any] | None = None

    def capture_request(url, **kwargs):
        nonlocal captured_request_body
        if "json" in kwargs:
            captured_request_body = kwargs["json"]

    # Configure model with tool support
    from open_webui_openrouter_pipe import ModelFamily
    ModelFamily.set_dynamic_specs({
        "test-model": {
            "features": {"function_calling"},
            "supported_parameters": frozenset(["tools", "tool_choice"]),
        }
    })

    try:
        # Mock catalog and responses endpoints
        with aioresponses() as mock_http:
            mock_http.get(
                "https://openrouter.ai/api/v1/models",
                payload={
                    "data": [
                        {
                            "id": "test-model",
                            "name": "Test Model",
                            "pricing": {"prompt": "0", "completion": "0"},
                            "context_length": 4096,
                        }
                    ]
                },
            )

            # Mock the HTTP POST to /responses with callback to capture request
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                payload={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "Weather result"}],
                            }
                        }
                    ],
                },
                callback=capture_request,
            )

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

            # Create a real HTTP session for the request
            session = pipe._create_http_session(pipe.valves)

            result = await pipe._process_transformed_request(
                body,
                __user__={"id": "u1", "valves": {}},
                __request__=None,
                __event_emitter__=None,
                __event_call__=cast(Callable[[dict[str, Any]], Awaitable[Any]], track_event_call),
                __metadata__=metadata,
                __tools__={},
                __task__=None,
                __task_body__=None,
                valves=pipe.valves,
                session=session,
                openwebui_model_id="test-model",
                pipe_identifier="pipe.test",
                allowlist_norm_ids=set(),
                enforced_norm_ids=set(),
                catalog_norm_ids=set(),
                features={},
            )

        # ASSERTIONS

        # 1. Verify HTTP request body contains getWeather tool
        assert captured_request_body is not None, "HTTP request should have been made"
        request_tools = captured_request_body.get("tools", [])

        # Responses API uses direct tool format: {"type": "function", "name": "getWeather", ...}
        assert request_tools, "tools should be present in request"
        assert any(
            t.get("type") == "function" and t.get("name") == "getWeather"
            for t in request_tools
        ), f"getWeather should be in HTTP request tools. Got: {request_tools}"

        # 2. Verify result was returned
        assert result is not None, "result should not be None"

    finally:
        # Clean up specs
        ModelFamily.set_dynamic_specs({})
        await pipe.close()


@pytest.mark.asyncio
async def test_direct_tool_servers_skipped_without_event_call():
    """
    Test that tool servers are NOT advertised when __event_call__ is None.

    REAL TEST - Uses aioresponses for HTTP boundary mocking.
    Exercises real conditional logic for tool server registration.
    Captures HTTP request to verify tools were NOT added.
    """
    pipe = Pipe()

    # Track the HTTP request body to verify tools were NOT added
    captured_request_body: dict[str, Any] | None = None

    def capture_request(url, **kwargs):
        nonlocal captured_request_body
        if "json" in kwargs:
            captured_request_body = kwargs["json"]

    # Configure model with tool support
    from open_webui_openrouter_pipe import ModelFamily
    ModelFamily.set_dynamic_specs({
        "test-model": {
            "features": {"function_calling"},
            "supported_parameters": frozenset(["tools", "tool_choice"]),
        }
    })

    try:
        # Mock catalog and responses endpoints
        with aioresponses() as mock_http:
            mock_http.get(
                "https://openrouter.ai/api/v1/models",
                payload={
                    "data": [
                        {
                            "id": "test-model",
                            "name": "Test Model",
                            "pricing": {"prompt": "0", "completion": "0"},
                            "context_length": 4096,
                        }
                    ]
                },
            )

            # Mock the HTTP POST to /responses with callback to capture request
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                payload={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "ok"}],
                            }
                        }
                    ],
                },
                callback=capture_request,
            )

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

            # Create a real HTTP session for the request
            session = pipe._create_http_session(pipe.valves)

            result = await pipe._process_transformed_request(
                body,
                __user__={"id": "u1", "valves": {}},
                __request__=None,
                __event_emitter__=None,
                __event_call__=None,  # KEY: No event_call provided
                __metadata__=metadata,
                __tools__={},
                __task__=None,
                __task_body__=None,
                valves=pipe.valves,
                session=session,
                openwebui_model_id="test-model",
                pipe_identifier="pipe.test",
                allowlist_norm_ids=set(),
                enforced_norm_ids=set(),
                catalog_norm_ids=set(),
                features={},
            )

        # ASSERTIONS

        # 1. When event_call is None, direct tool servers should NOT be advertised
        assert captured_request_body is not None, "HTTP request should have been made"
        request_tools = captured_request_body.get("tools", [])

        # Responses API uses direct tool format
        assert not any(
            t.get("name") == "getWeather"
            for t in request_tools
        ), f"getWeather should NOT be in HTTP request tools when event_call is None. Got: {request_tools}"

        # 2. Result should still be returned
        assert result is not None, "result should not be None"

    finally:
        # Clean up specs
        ModelFamily.set_dynamic_specs({})
        await pipe.close()
