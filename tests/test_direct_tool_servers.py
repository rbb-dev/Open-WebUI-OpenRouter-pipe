from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import EncryptedStr, Pipe
from tests.test_tools import _as_open_webui_resolves_them


_STUBBED_INPUT = [
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hi"}],
    }
]


_WEATHER_SPEC: dict[str, Any] = {
    "name": "getWeather",
    "description": "Fetch weather.",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}


def _offered_names(captured: dict[str, Any]) -> list[str | None]:
    return [t.get("name") for t in (captured.get("tools") or [])]


def _metadata_with_a_browser_tool(*, session_id: str | None, request_params: dict[str, Any] | None = None):
    """The metadata Open WebUI 0.11.4 builds for a chat whose tool picker holds one browser tool server.

    `session_id` is the only thing that decides whether the pipe gets an `__event_call__`
    (`middleware.py:3281-3292`): a chat replayed by an API caller, a background task or a title
    generation has none, and then the pipe owns the call with no browser to make it.
    """
    metadata: dict[str, Any] = {
        "chat_id": "c1",
        "message_id": "m1",
        "model": {"id": "test-model"},
        "tool_servers": [{"url": "https://example.com", "specs": [dict(_WEATHER_SPEC)]}],
    }
    if session_id is not None:
        metadata["session_id"] = session_id
    if request_params is not None:
        metadata["params"] = request_params
    metadata = _as_open_webui_resolves_them(metadata)
    # A second, non-direct resolved tool the pipe CAN run: an `extra_tools` spec sharing this name
    # is offered, and it is what separates "the new name-scoped drop removed the browser tool" from
    # "an extra tool with no runnable name behind it is withheld", which is the rule that was
    # already true before this fix.
    resolved = metadata["tools"]
    resolved["server_side"] = {
        "spec": {"name": "server_side", "description": "Runs in this process.",
                 "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}},
        "callable": _always_runs,
    }
    return metadata


async def _always_runs(**_kwargs):
    return [{"answer": 1}, {"content-type": "application/json"}]


async def _send_one_request(*, event_call, metadata, body, valves, extra_tools=None):
    """One real `Pipe._process_transformed_request` with only OpenRouter's HTTP seam stubbed.

    Returns the tools the request carried to OpenRouter, by name. Everything between the pipe and
    the wire -- the orchestrator, the tool registry, the collision-safe builder -- is the real code.
    """
    pipe = Pipe()
    if valves is not None:
        pipe.valves.TOOL_EXECUTION_MODE = valves
    session = None
    captured: dict[str, Any] = {}

    def capture_request(url, **kwargs):
        if "json" in kwargs:
            captured.update(kwargs["json"])

    from open_webui_openrouter_pipe import ModelFamily

    ModelFamily.set_dynamic_specs({
        "test-model": {
            "features": {"function_calling"},
            "supported_parameters": frozenset(["tools", "tool_choice"]),
        }
    })

    try:
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
            mock_http.get("https://openrouter.ai/api/v1/endpoints/zdr", payload={"data": []}, repeat=True)
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                payload={
                    "choices": [
                        {"message": {"role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}}
                    ]
                },
                callback=capture_request,
            )

            sent_body = dict(body)
            if extra_tools is not None:
                sent_body["extra_tools"] = extra_tools
            session = pipe._create_http_session(pipe.valves)
            result = await pipe._process_transformed_request(
                sent_body,
                __user__={"id": "u1", "valves": {}},
                __request__=None,
                __event_emitter__=None,
                __event_call__=event_call,
                __metadata__=metadata,
                __tools__={},
                __task__=None,
                __task_body__=None,
                session=session,
                openwebui_model_id="test-model",
                pipe_identifier="pipe.test",
                allowlist_norm_ids=set(),
                enforced_norm_ids=set(),
                catalog_norm_ids=set(),
                valves=pipe.valves,
                features={},
            )
        assert result is not None, "result should not be None"
        assert captured, "HTTP request should have been made"
        return _offered_names(captured)
    finally:
        ModelFamily.set_dynamic_specs({})
        if session is not None:
            await session.close()
        await pipe.close()


def _body(*, with_request_tools: bool, extra_names: tuple[str, ...] = ()) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    if with_request_tools:
        # `form_data['tools']` as middleware.py:3189 builds it: the same resolved specs, in the
        # chat shape. This is the half that Open WebUI writes alongside `metadata['tools']` and the
        # half the shipped fixture was missing, which is what made its assertion vacuous.
        body["tools"] = [
            {"type": "function", "function": dict(_WEATHER_SPEC)},
            *[
                {"type": "function", "function": {"name": name, "description": f"Call {name}."}}
                for name in extra_names
            ],
        ]
    return body


@pytest.mark.asyncio
async def test_direct_tool_servers_skipped_without_event_call():
    """A browser tool the pipe cannot run is not offered to the model.

    The fixture here is the shape Open WebUI actually produces and the shipped one was not: the
    resolved tools land in `form_data['tools']` as well as in `metadata['tools']`
    (`middleware.py:3186-3193`), and `session_id` is not what carries a browser channel. A fixture
    with a `session_id` and no `tools` key in the body is a shape Open WebUI never produces, and
    asserting on it proved nothing.
    """
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false
    offered = await _send_one_request(
        event_call=None,
        metadata=_metadata_with_a_browser_tool(session_id="s1"),
        body=_body(with_request_tools=True),
        valves=None,
    )
    assert "getWeather" not in offered, (
        f"getWeather should NOT be in HTTP request tools when event_call is None. Got: {offered}"
    )
