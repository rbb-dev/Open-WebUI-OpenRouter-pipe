"""What a person reads when the connection to OpenRouter times out or fails before any of the answer arrives.

The user's decision (2026-09-17): a timeout shows the admin's network-timeout message; a connection that fails, or a
stream OpenRouter closes without sending anything, shows the connection-failure message. The unexpected-error card
stays for genuinely unexpected errors. An answer cut off after its text began is covered next to the other streamed
cards, in ``tests/test_streamed_error_card_delivery.py``.

Every message template is replaced by its own marker, so each arm shows which of the admin's messages was chosen, and
the requests go through ``pipe.pipe()`` and the real transports, retries included.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import Pipe

_RESPONSES_URL = "https://openrouter.ai/api/v1/responses"
_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
_MARKERS = {
    "NETWORK_TIMEOUT_TEMPLATE": "### TIMEOUT-CARD",
    "CONNECTION_ERROR_TEMPLATE": "### CONNECTION-CARD",
    "STREAM_INTERRUPTED_TEMPLATE": "### INTERRUPTED-NOTICE",
    "INTERNAL_ERROR_TEMPLATE": "### UNEXPECTED-CARD",
    "SERVICE_ERROR_TEMPLATE": "### SERVICE-CARD",
}
_TRANSPORTS = [("responses", True), ("responses", False), ("chat_completions", True), ("chat_completions", False)]
_TRANSPORT_IDS = ["responses-streaming", "responses-not-streaming", "chat-streaming", "chat-not-streaming"]


def _pipe_reaching_the_model(monkeypatch, endpoint: str) -> Pipe:
    import open_webui_openrouter_pipe.pipe as pipe_mod

    async def loaded(*_args: Any, **_kwargs: Any) -> None:
        return None

    pipe = Pipe()
    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: ("sk-test-key", None))
    monkeypatch.setattr(pipe._artifact_store, "_ensure_artifact_store", lambda *_a, **_k: None)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", loaded)
    monkeypatch.setattr(
        pipe_mod.OpenRouterModelRegistry, "list_models", lambda: [{"id": "m1", "name": "Model m1", "norm_id": "m1"}]
    )
    pipe.valves = pipe.valves.model_copy(update={"DEFAULT_LLM_ENDPOINT": endpoint, **_MARKERS})
    return pipe


async def _turn(pipe: Pipe, *, stream: bool, tools: dict[str, Any] | None = None) -> str:
    result = await pipe.pipe(
        body={"model": "m1", "messages": [{"role": "user", "content": "Look it up."}], "stream": stream},
        __user__={"id": "user-1", "role": "user"},
        __request__=None,
        __event_emitter__=None,
        __event_call__=None,
        __metadata__={"chat_id": "chat-1", "message_id": "message-1", "model": {"id": "m1"}},
        __tools__=tools,
    )
    if isinstance(result, AsyncIterator):
        return "".join([str(chunk) async for chunk in result])
    return str(result)


def _markers_shown(reply: str) -> list[str]:
    return [name for name, marker in _MARKERS.items() if marker in reply]


_BEFORE_ANY_TEXT = {
    "timed-out": (asyncio.TimeoutError, (), "NETWORK_TIMEOUT_TEMPLATE"),
    "socket-read-timed-out": (aiohttp.ServerTimeoutError, ("Timeout on reading data from socket",), "NETWORK_TIMEOUT_TEMPLATE"),
    "connection-refused": (aiohttp.ClientConnectionError, ("Cannot connect to host openrouter.ai:443",), "CONNECTION_ERROR_TEMPLATE"),
    "server-disconnected": (aiohttp.ServerDisconnectedError, (), "CONNECTION_ERROR_TEMPLATE"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", list(_BEFORE_ANY_TEXT))
@pytest.mark.parametrize(("endpoint", "stream"), _TRANSPORTS, ids=_TRANSPORT_IDS)
async def test_a_call_that_fails_before_any_answer_arrives_shows_the_admins_message_for_that_failure(
    monkeypatch, endpoint, stream, failure
):
    error_type, args, template = _BEFORE_ANY_TEXT[failure]
    pipe = _pipe_reaching_the_model(monkeypatch, endpoint)
    try:
        with aioresponses() as mock_http:
            mock_http.post(_RESPONSES_URL if endpoint == "responses" else _CHAT_URL, exception=error_type(*args), repeat=True)
            reply = await _turn(pipe, stream=stream)
            recorded = len(pipe._circuit_breaker._breaker_records["user-1"])
    finally:
        await pipe.close()

    assert _markers_shown(reply) == [template], reply
    assert recorded >= 1, (recorded, reply)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "limits", "seconds"),
    [
        (aiohttp.ConnectionTimeoutError("Connection timeout to host"), {"HTTP_CONNECT_TIMEOUT_SECONDS": 7}, "7"),
        (aiohttp.SocketTimeoutError("Timeout on reading data from socket"), {"HTTP_SOCK_READ_SECONDS": 45}, "45"),
        (asyncio.TimeoutError(), {"HTTP_TOTAL_TIMEOUT_SECONDS": 90}, "90"),
    ],
    ids=["while-connecting", "while-reading", "whole-request"],
)
async def test_the_timeout_message_gives_the_seconds_of_the_limit_that_ran_out(monkeypatch, error, limits, seconds):
    pipe = _pipe_reaching_the_model(monkeypatch, "responses")
    pipe.valves = pipe.valves.model_copy(
        update={"NETWORK_TIMEOUT_TEMPLATE": "### TIMEOUT-CARD after {timeout_seconds} seconds", **limits}
    )
    try:
        with aioresponses() as mock_http:
            mock_http.post(_RESPONSES_URL, exception=error, repeat=True)
            reply = await _turn(pipe, stream=True)
    finally:
        await pipe.close()

    assert f"TIMEOUT-CARD after {seconds} seconds" in reply, reply


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_body", [b"", b"data: [DONE]\n\n"], ids=["nothing", "only-the-end-marker"])
@pytest.mark.parametrize("endpoint", ["responses", "chat_completions"])
async def test_a_stream_openrouter_closes_without_sending_anything_shows_the_connection_failure_message(
    monkeypatch, endpoint, empty_body
):
    pipe = _pipe_reaching_the_model(monkeypatch, endpoint)
    try:
        with aioresponses() as mock_http:
            mock_http.post(_RESPONSES_URL if endpoint == "responses" else _CHAT_URL, status=200, body=empty_body, repeat=True)
            reply = await _turn(pipe, stream=True)
            posts = sum(len(calls) for (method, _url), calls in mock_http.requests.items() if method == "POST")
    finally:
        await pipe.close()

    assert posts > 1, (posts, reply)
    assert _markers_shown(reply) == ["CONNECTION_ERROR_TEMPLATE"], reply


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "template"),
    [(aiohttp.ClientConnectionError("dropped"), "CONNECTION_ERROR_TEMPLATE"), (asyncio.TimeoutError(), "NETWORK_TIMEOUT_TEMPLATE")],
    ids=["connection-dropped", "timed-out"],
)
async def test_a_tool_round_that_produced_no_text_then_a_failed_call_shows_the_admins_message(monkeypatch, failure, template):
    """Nothing of the answer arrived, only a tool round, so there is no text to keep and no interrupted notice."""
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager
    from open_webui_openrouter_pipe.models.registry import ModelFamily

    async def no_web_tools(*_args: Any, **_kwargs: Any) -> None:
        return None

    pipe = _pipe_reaching_the_model(monkeypatch, "responses")
    monkeypatch.setattr(FilterManager, "collect_installed_web_tools_config", no_web_tools)
    monkeypatch.setattr(ModelFamily, "supports", classmethod(lambda cls, capability, model: capability == "function_calling"))
    ran: list[str] = []

    async def lookup(**_kwargs: Any) -> str:
        ran.append("lookup")
        return "found"

    async def model_that_calls_then_loses_the_connection(self, session, request_body, **_kwargs):
        if any(isinstance(item, dict) and item.get("type") == "function_call_output" for item in request_body.get("input") or []):
            raise failure
        call = {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}", "status": "completed"}
        yield {"type": "response.output_item.done", "item": call}
        yield {"type": "response.completed", "response": {"output": [call], "usage": {}}}

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", model_that_calls_then_loses_the_connection)
    tools = {
        "lookup": {
            "type": "function",
            "callable": lookup,
            "spec": {"name": "lookup", "description": "Look it up", "parameters": {"type": "object", "properties": {}}},
        }
    }
    try:
        reply = await _turn(pipe, stream=True, tools=tools)
    finally:
        await pipe.close()

    assert ran == ["lookup"], (ran, reply)
    assert _markers_shown(reply) == [template], reply


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["responses", "chat_completions"])
async def test_the_built_in_connection_failure_message_covers_a_reply_openrouter_never_sent(monkeypatch, endpoint):
    """The user's decision (2026-09-17): the built-in text also fits a stream OpenRouter closed without sending anything."""
    pipe = _pipe_reaching_the_model(monkeypatch, endpoint)
    pipe.valves = pipe.valves.model_copy(
        update={"CONNECTION_ERROR_TEMPLATE": Pipe.Valves.model_fields["CONNECTION_ERROR_TEMPLATE"].get_default()}
    )
    try:
        with aioresponses() as mock_http:
            mock_http.post(_RESPONSES_URL if endpoint == "responses" else _CHAT_URL, status=200, body=b"", repeat=True)
            reply = await _turn(pipe, stream=True)
    finally:
        await pipe.close()

    assert "### 🔌 Connection Failed" in reply, reply
    assert "The connection to OpenRouter failed, or ended before a reply arrived." in reply, reply
    assert "- OpenRouter closed the connection without sending a reply" in reply, reply
    assert "1. Try the message again" in reply, reply
    assert "Unable to reach OpenRouter's servers" not in reply, reply
