"""Tests for the OWUI user-identity header forwarding helper.

The helper mirrors what a native Open WebUI OpenAI connection stamps on an
outbound request: it delegates to OWUI's own ``include_user_info_headers`` for
the user-identity headers (inheriting custom names / JWT mode) and adds the
session Chat-Id header. It is a no-op unless the operator enabled OWUI's
``ENABLE_FORWARD_USER_INFO_HEADERS``.

In the test environment ``open_webui`` is stubbed, so the real OWUI function is
not importable; the helper's guarded import falls back to "off". These tests
monkeypatch the module symbols to exercise the enabled path with a fake that
matches OWUI's verified contract (``include_user_info_headers(headers, user)``
returns ``{**headers, <identity headers>}``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe.core import config


def _fake_plain_include(headers, user):
    """Mimic OWUI plain-header mode."""
    return {
        **headers,
        "X-OpenWebUI-User-Name": user.name,
        "X-OpenWebUI-User-Id": user.id,
        "X-OpenWebUI-User-Email": user.email,
        "X-OpenWebUI-User-Role": user.role,
    }


def _fake_jwt_include(headers, user):
    """Mimic OWUI JWT mode (four user headers collapsed to one signed token)."""
    return {**headers, "X-OpenWebUI-User-Jwt": "signed.jwt.token"}


@pytest.fixture
def enabled(monkeypatch):
    """Enable the forwarding path with OWUI's plain-header contract."""
    monkeypatch.setattr(
        config,
        "_owui_env",
        SimpleNamespace(
            ENABLE_FORWARD_USER_INFO_HEADERS=True,
            FORWARD_USER_INFO_HEADER_USER_NAME="X-OpenWebUI-User-Name",
            FORWARD_USER_INFO_HEADER_USER_ID="X-OpenWebUI-User-Id",
            FORWARD_USER_INFO_HEADER_USER_EMAIL="X-OpenWebUI-User-Email",
            FORWARD_USER_INFO_HEADER_USER_ROLE="X-OpenWebUI-User-Role",
            FORWARD_SESSION_INFO_HEADER_CHAT_ID="X-OpenWebUI-Chat-Id",
            FORWARD_USER_INFO_HEADER_JWT="X-OpenWebUI-User-Jwt",
        ),
    )
    monkeypatch.setattr(config, "_owui_include_user_info_headers", _fake_plain_include)


@pytest.fixture
def user():
    return SimpleNamespace(id="uuid-123", email="alice@example.com", name="Alice", role="user")


def test_noop_when_forwarding_unavailable(monkeypatch, user):
    # A working fn is present but OWUI's env module was not importable -> no-op.
    monkeypatch.setattr(config, "_owui_env", None)
    monkeypatch.setattr(config, "_owui_include_user_info_headers", _fake_plain_include)
    base = {"Authorization": "Bearer x"}
    out = config._apply_owui_forward_user_headers(base, user, chat_id="c1")
    assert out == {"Authorization": "Bearer x"}


def test_noop_when_enable_flag_false(monkeypatch, user):
    monkeypatch.setattr(config, "_owui_env", SimpleNamespace(ENABLE_FORWARD_USER_INFO_HEADERS=False))
    monkeypatch.setattr(config, "_owui_include_user_info_headers", _fake_plain_include)
    out = config._apply_owui_forward_user_headers({"Authorization": "Bearer x"}, user, chat_id="c1")
    assert out == {"Authorization": "Bearer x"}


def test_stamps_user_headers_when_enabled(enabled, user):
    out = config._apply_owui_forward_user_headers({"Authorization": "Bearer x"}, user)
    assert out["Authorization"] == "Bearer x"
    assert out["X-OpenWebUI-User-Id"] == "uuid-123"
    assert out["X-OpenWebUI-User-Email"] == "alice@example.com"
    assert out["X-OpenWebUI-User-Name"] == "Alice"
    assert out["X-OpenWebUI-User-Role"] == "user"


def test_adds_chat_id_with_configured_name(monkeypatch, enabled, user):
    # Custom header name (e.g. AWS Bedrock AgentCore prefix) must be respected.
    monkeypatch.setattr(
        config._owui_env,
        "FORWARD_SESSION_INFO_HEADER_CHAT_ID",
        "X-Amzn-Bedrock-AgentCore-Runtime-Custom-Chat-Id",
    )
    out = config._apply_owui_forward_user_headers({}, user, chat_id="chat-42")
    assert out["X-Amzn-Bedrock-AgentCore-Runtime-Custom-Chat-Id"] == "chat-42"
    assert "X-OpenWebUI-Chat-Id" not in out


def test_omits_chat_id_when_none(enabled, user):
    out = config._apply_owui_forward_user_headers({}, user, chat_id=None)
    assert "X-OpenWebUI-Chat-Id" not in out


def test_none_user_is_noop(enabled):
    out = config._apply_owui_forward_user_headers({"Authorization": "Bearer x"}, None, chat_id="c1")
    assert out == {"Authorization": "Bearer x"}


def test_dict_user_is_noop_no_exception(enabled):
    # Proven: OWUI's function does getattr(user, "name") -> AttributeError on a dict.
    dict_user = {"id": "uuid-123", "email": "a@b.com", "name": "Alice", "role": "user"}
    out = config._apply_owui_forward_user_headers({"Authorization": "Bearer x"}, dict_user, chat_id="c1")
    assert out == {"Authorization": "Bearer x"}


def test_jwt_mode_passthrough(monkeypatch, enabled, user):
    monkeypatch.setattr(config, "_owui_include_user_info_headers", _fake_jwt_include)
    out = config._apply_owui_forward_user_headers({"Authorization": "Bearer x"}, user, chat_id="c1")
    assert out["X-OpenWebUI-User-Jwt"] == "signed.jwt.token"
    assert "X-OpenWebUI-User-Id" not in out  # collapsed into the JWT
    assert out["X-OpenWebUI-Chat-Id"] == "c1"


def test_swallows_owui_exception(monkeypatch, enabled, user):
    def _boom(headers, user):
        raise RuntimeError("owui blew up")

    monkeypatch.setattr(config, "_owui_include_user_info_headers", _boom)
    base = {"Authorization": "Bearer x"}
    out = config._apply_owui_forward_user_headers(base, user, chat_id="c1")
    assert out == {"Authorization": "Bearer x"}


def test_does_not_mutate_caller_headers(enabled, user):
    base = {"Authorization": "Bearer x"}
    config._apply_owui_forward_user_headers(base, user, chat_id="c1")
    assert base == {"Authorization": "Bearer x"}  # original untouched


# ============================================================================
# Adapter integration: the four send_* functions stamp the headers on the
# actual outbound request (captured at the aiohttp boundary via aioresponses).
# ============================================================================

import json  # noqa: E402

from aioresponses import CallbackResult, aioresponses  # noqa: E402


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def _capture(store, body, ctype):
    def cb(url, **kwargs):
        store["headers"] = dict(kwargs.get("headers") or {})
        return CallbackResult(status=200, body=body, headers={"Content-Type": ctype})

    return cb


_RESP_SSE = (_sse({"type": "response.completed", "response": {"output": [], "usage": {}}}) + "data: [DONE]\n\n").encode()
_RESP_JSON = json.dumps({"output": [], "usage": {}}).encode()
_CHAT_SSE = (
    _sse({"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]})
    + _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    + "data: [DONE]\n\n"
).encode()
_CHAT_JSON = json.dumps({"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}], "usage": {}}).encode()


@pytest.mark.asyncio
async def test_responses_streaming_stamps_five_headers(pipe_instance_async, enabled, user):
    pipe = pipe_instance_async
    session = pipe._create_http_session(pipe.valves)
    store: dict = {}
    with aioresponses() as m:
        m.post("https://openrouter.ai/api/v1/responses", callback=_capture(store, _RESP_SSE, "text/event-stream"))
        async for _ in pipe.send_openai_responses_streaming_request(
            session, {"model": "openai/gpt-4o", "input": []}, api_key="k",
            base_url="https://openrouter.ai/api/v1", valves=pipe.valves, user=user, owui_chat_id="chat-1",
        ):
            pass
        await session.close()
    assert store["headers"]["X-OpenWebUI-User-Id"] == "uuid-123"
    assert store["headers"]["X-OpenWebUI-User-Email"] == "alice@example.com"
    assert store["headers"]["X-OpenWebUI-Chat-Id"] == "chat-1"


@pytest.mark.asyncio
async def test_responses_nonstreaming_stamps_headers(pipe_instance_async, enabled, user):
    pipe = pipe_instance_async
    session = pipe._create_http_session(pipe.valves)
    store: dict = {}
    with aioresponses() as m:
        m.post("https://openrouter.ai/api/v1/responses", callback=_capture(store, _RESP_JSON, "application/json"))
        await pipe.send_openai_responses_nonstreaming_request(
            session, {"model": "openai/gpt-4o", "input": []}, api_key="k",
            base_url="https://openrouter.ai/api/v1", valves=pipe.valves, user=user, owui_chat_id="chat-2",
        )
        await session.close()
    assert store["headers"]["X-OpenWebUI-User-Id"] == "uuid-123"
    assert store["headers"]["X-OpenWebUI-Chat-Id"] == "chat-2"


@pytest.mark.asyncio
async def test_chat_streaming_stamps_headers(pipe_instance_async, enabled, user):
    pipe = pipe_instance_async
    session = pipe._create_http_session(pipe.valves)
    store: dict = {}
    with aioresponses() as m:
        m.post("https://openrouter.ai/api/v1/chat/completions", callback=_capture(store, _CHAT_SSE, "text/event-stream"))
        async for _ in pipe.send_openai_chat_completions_streaming_request(
            session, {"model": "openai/gpt-4o", "input": []}, api_key="k",
            base_url="https://openrouter.ai/api/v1", valves=pipe.valves, user=user, owui_chat_id="chat-3",
        ):
            pass
        await session.close()
    assert store["headers"]["X-OpenWebUI-User-Id"] == "uuid-123"
    assert store["headers"]["X-OpenWebUI-Chat-Id"] == "chat-3"


@pytest.mark.asyncio
async def test_chat_nonstreaming_stamps_headers(pipe_instance_async, enabled, user):
    pipe = pipe_instance_async
    session = pipe._create_http_session(pipe.valves)
    store: dict = {}
    with aioresponses() as m:
        m.post("https://openrouter.ai/api/v1/chat/completions", callback=_capture(store, _CHAT_JSON, "application/json"))
        await pipe.send_openai_chat_completions_nonstreaming_request(
            session, {"model": "openai/gpt-4o", "input": []}, api_key="k",
            base_url="https://openrouter.ai/api/v1", valves=pipe.valves, user=user, owui_chat_id="chat-4",
        )
        await session.close()
    assert store["headers"]["X-OpenWebUI-User-Id"] == "uuid-123"
    assert store["headers"]["X-OpenWebUI-Chat-Id"] == "chat-4"


@pytest.mark.asyncio
async def test_no_headers_when_forwarding_disabled(pipe_instance_async, user):
    # No `enabled` fixture -> forwarding off by default -> nothing stamped.
    pipe = pipe_instance_async
    session = pipe._create_http_session(pipe.valves)
    store: dict = {}
    with aioresponses() as m:
        m.post("https://openrouter.ai/api/v1/responses", callback=_capture(store, _RESP_SSE, "text/event-stream"))
        async for _ in pipe.send_openai_responses_streaming_request(
            session, {"model": "openai/gpt-4o", "input": []}, api_key="k",
            base_url="https://openrouter.ai/api/v1", valves=pipe.valves, user=user, owui_chat_id="chat-x",
        ):
            pass
        await session.close()
    assert not any(k.startswith("X-OpenWebUI") for k in store["headers"])


# ============================================================================
# Video client: separate header builder (submit/status/list + content download)
# must forward identity too, since video gen doesn't go through the text adapters.
# ============================================================================

import logging  # noqa: E402

from open_webui_openrouter_pipe.integrations.video_client import OpenRouterVideoClient  # noqa: E402

_VLOG = logging.getLogger("test_video_headers")


def _video_client(user=None, chat_id=None):
    return OpenRouterVideoClient(
        cast(Any, None), base_url="https://openrouter.ai/api/v1", api_key="k",
        logger=_VLOG, user=user, owui_chat_id=chat_id,
    )


def test_video_client_headers_stamp_identity(enabled, user):
    h = _video_client(user=user, chat_id="vid-1")._headers()
    assert h["X-OpenWebUI-User-Id"] == "uuid-123"
    assert h["X-OpenWebUI-Chat-Id"] == "vid-1"


def test_video_bearer_header_stamps_identity(enabled, user):
    h = _video_client(user=user, chat_id="vid-1").bearer_header()
    assert h["X-OpenWebUI-User-Id"] == "uuid-123"


def test_video_client_headers_disabled_no_identity(user):
    h = _video_client(user=user, chat_id="vid-1")._headers()
    assert not any(k.startswith("X-OpenWebUI") for k in h)


def test_video_client_headers_noop_without_user(enabled):
    h = _video_client()._headers()
    assert not any(k.startswith("X-OpenWebUI") for k in h)


# ============================================================================
# End-to-end: chat_id threaded through the streaming dispatcher reaches the header.
# ============================================================================


@pytest.mark.asyncio
async def test_dispatcher_threads_chat_id_to_headers(pipe_instance_async, enabled, user):
    pipe = pipe_instance_async
    session = pipe._create_http_session(pipe.valves)
    store: dict = {}
    with aioresponses() as m:
        m.post("https://openrouter.ai/api/v1/chat/completions", callback=_capture(store, _CHAT_SSE, "text/event-stream"))
        async for _ in pipe.send_openrouter_streaming_request(
            session, {"model": "openai/gpt-4o", "stream": True, "input": []}, api_key="k",
            base_url="https://openrouter.ai/api/v1", valves=pipe.valves,
            endpoint_override="chat_completions", user=user, owui_chat_id="e2e-chat",
        ):
            pass
        await session.close()
    assert store["headers"]["X-OpenWebUI-User-Id"] == "uuid-123"
    assert store["headers"]["X-OpenWebUI-Chat-Id"] == "e2e-chat"


# ============================================================================
# Fix 1: debug/session logging must not leak forwarded PII or the signed JWT.
# ============================================================================


class _CaptureLogger:
    def __init__(self):
        self.messages: list = []

    def isEnabledFor(self, level):
        return True

    def debug(self, fmt, *args):
        self.messages.append(fmt % args if args else fmt)


def test_debug_redacts_default_identity_headers(enabled):
    from open_webui_openrouter_pipe.requests.debug import _debug_print_request

    log = _CaptureLogger()
    headers = {
        "Authorization": "Bearer sk-supersecretkey",
        "X-OpenWebUI-User-Email": "alice@example.com",
        "X-OpenWebUI-User-Name": "Alice",
        "X-OpenWebUI-User-Jwt": "sig.jwt.tok",
        "X-OpenRouter-Title": "keep-me",
    }
    _debug_print_request(headers, {}, logger=cast(Any, log))
    blob = "\n".join(log.messages)
    assert "alice@example.com" not in blob
    assert "sig.jwt.tok" not in blob
    assert "Bearer sk-supersecretkey" not in blob  # Authorization still redacted
    assert "keep-me" in blob  # non-identity header preserved
    assert "X-OpenWebUI-User-Email" in blob  # key retained, value redacted


def test_debug_redacts_custom_prefixed_identity_headers(monkeypatch):
    from open_webui_openrouter_pipe.requests.debug import _debug_print_request

    monkeypatch.setattr(
        config,
        "_owui_env",
        SimpleNamespace(FORWARD_USER_INFO_HEADER_USER_EMAIL="X-Amzn-Bedrock-AgentCore-Runtime-Custom-User-Email"),
    )
    log = _CaptureLogger()
    headers = {"X-Amzn-Bedrock-AgentCore-Runtime-Custom-User-Email": "bob@example.com"}
    _debug_print_request(headers, {}, logger=cast(Any, log))
    blob = "\n".join(log.messages)
    assert "bob@example.com" not in blob
    assert "X-Amzn-Bedrock-AgentCore-Runtime-Custom-User-Email" in blob
