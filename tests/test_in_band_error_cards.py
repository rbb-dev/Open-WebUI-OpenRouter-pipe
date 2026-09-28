"""Which message a person reads when OpenRouter reports a failure inside a reply it has already started.

OpenRouter commits `200 OK` as soon as a provider accepts a request, so every later failure — a rate limit, an
overloaded or unreachable provider, a provider timeout, an exhausted balance — arrives inside the reply carrying its
own code (`api_reference/errors-and-debugging.md`). The user's decision (2026-09-18): the pipe reads that code and
shows the message for that kind of failure, keeping the rejected-request message for codes it does not recognise.

Every message template is replaced by its own marker, so each arm shows which of the admin's messages was chosen, and
the requests go through ``pipe.pipe()`` and the real transports.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import Pipe

_RESPONSES_URL = "https://openrouter.ai/api/v1/responses"
_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
_MARKERS = {
    "OPENROUTER_ERROR_TEMPLATE": "### REJECTED-CARD {status_code}",
    "RATE_LIMIT_TEMPLATE": "### RATE-LIMIT-CARD {status_code}",
    "SERVICE_ERROR_TEMPLATE": "### SERVICE-CARD {status_code}",
    "SERVER_TIMEOUT_TEMPLATE": "### SERVER-TIMEOUT-CARD {status_code}",
    "AUTHENTICATION_ERROR_TEMPLATE": "### AUTH-CARD {status_code}",
    "INSUFFICIENT_CREDITS_TEMPLATE": "### CREDITS-CARD {status_code}",
    "PAYLOAD_TOO_LARGE_TEMPLATE": "### PAYLOAD-CARD {status_code}",
    "INTERNAL_ERROR_TEMPLATE": "### UNEXPECTED-CARD",
}


def _pipe_reaching_the_model(monkeypatch, endpoint: str, templates: dict[str, str] | None = None) -> Pipe:
    """``templates`` replaces the marker set; an empty dict leaves every factory template in place."""
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
    update = {"DEFAULT_LLM_ENDPOINT": endpoint, **(_MARKERS if templates is None else templates)}
    pipe.valves = pipe.valves.model_copy(update=update)
    return pipe


async def _reply(pipe: Pipe, *, stream: bool) -> Any:
    return await pipe.pipe(
        body={"model": "m1", "messages": [{"role": "user", "content": "Look it up."}], "stream": stream},
        __user__={"id": "user-1", "role": "user"},
        __request__=None,
        __event_emitter__=None,
        __event_call__=None,
        __metadata__={"chat_id": "chat-1", "message_id": "message-1", "model": {"id": "m1"}},
        __tools__=None,
    )


async def _turn(pipe: Pipe, *, stream: bool) -> str:
    result = await _reply(pipe, stream=stream)
    if isinstance(result, AsyncIterator):
        return "".join([str(chunk) async for chunk in result])
    return str(result)


def _cards(reply: str) -> list[str]:
    return [name for name, marker in _MARKERS.items() if marker.split(" {", 1)[0] in reply]


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _chat_stream(code: Any, error_type: str, *, chunk_id: Any = "gen-1") -> bytes:
    chunk = {
        "id": chunk_id, "object": "chat.completion.chunk", "created": 1, "model": "m1", "provider": "P",
        "error": {"code": code, "message": "reported inside the reply", "metadata": {"error_type": error_type}},
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
    }
    return (_sse(chunk) + "data: [DONE]\n\n").encode("utf-8")


def _chat_body(code: Any, error_type: str, *, body_id: str | None = None) -> bytes:
    body: dict[str, Any] = {} if body_id is None else {"id": body_id}
    body["error"] = {"code": code, "message": "reported inside the reply", "metadata": {"error_type": error_type}}
    return json.dumps(body).encode("utf-8")


def _responses_stream(code: str, error_type: str, *, response_id: Any = "resp-1") -> bytes:
    event = {
        "type": "response.failed",
        "response": {"id": response_id, "status": "failed", "error": {"code": code, "message": "reported inside the reply"},
                     "error_type": error_type},
    }
    return (_sse(event) + "data: [DONE]\n\n").encode("utf-8")


def _chat_stream_native(code: str) -> bytes:
    """The shape `api_reference/streaming.md` documents: a native code and no typed kind."""
    chunk = {
        "id": "cmpl-abc123", "object": "chat.completion.chunk", "created": 1, "model": "m1", "provider": "openai",
        "error": {"code": code, "message": "Provider disconnected unexpectedly"},
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
    }
    return (_sse(chunk) + "data: [DONE]\n\n").encode("utf-8")


def _responses_error_event(event_type: str, code: str) -> bytes:
    event = {"type": event_type, "error": {"code": code, "message": "reported inside the reply"}}
    return (_sse(event) + "data: [DONE]\n\n").encode("utf-8")


def _responses_body(code: str, error_type: str, *, response_id: str = "resp-1") -> bytes:
    body = {"id": response_id, "status": "failed", "error": {"code": code, "message": "reported inside the reply"},
            "error_type": error_type}
    return json.dumps(body).encode("utf-8")


# Each arm: the endpoint, whether the turn streams, the reply body, and the message the person must read.
_ARMS: dict[str, tuple[str, bool, bytes, str, int]] = {
    "chat-stream-rate-limit": ("chat_completions", True, _chat_stream(429, "rate_limit_exceeded"), "RATE_LIMIT_TEMPLATE", 429),
    "chat-stream-provider-unavailable": ("chat_completions", True, _chat_stream(502, "provider_unavailable"), "SERVICE_ERROR_TEMPLATE", 502),
    "chat-stream-provider-overloaded": ("chat_completions", True, _chat_stream(503, "provider_overloaded"), "SERVICE_ERROR_TEMPLATE", 503),
    "chat-stream-provider-timed-out": ("chat_completions", True, _chat_stream(504, "timeout"), "SERVICE_ERROR_TEMPLATE", 504),
    "chat-stream-invalid-request": ("chat_completions", True, _chat_stream(400, "invalid_request"), "OPENROUTER_ERROR_TEMPLATE", 400),
    "chat-body-out-of-credits": ("chat_completions", False, _chat_body(402, "payment_required"), "INSUFFICIENT_CREDITS_TEMPLATE", 402),
    "responses-stream-rate-limit": ("responses", True, _responses_stream("rate_limit_exceeded", "rate_limit_exceeded"), "RATE_LIMIT_TEMPLATE", 429),
    "responses-stream-server-error": ("responses", True, _responses_stream("server_error", "server"), "SERVICE_ERROR_TEMPLATE", 500),
    "responses-stream-invalid-prompt": ("responses", True, _responses_stream("invalid_prompt", "invalid_request"), "OPENROUTER_ERROR_TEMPLATE", 400),
    "responses-body-authentication": ("responses", False, _responses_body("server_error", "authentication"), "AUTHENTICATION_ERROR_TEMPLATE", 401),
    "chat-stream-provider-disconnected": ("chat_completions", True, _chat_stream_native("server_error"), "SERVICE_ERROR_TEMPLATE", 500),
    "chat-stream-kind-the-pipe-does-not-know": ("chat_completions", True, _chat_stream(429, "quota_exhausted"), "RATE_LIMIT_TEMPLATE", 429),
    "responses-stream-error-event-rate-limit": ("responses", True, _responses_error_event("response.error", "rate_limit_exceeded"), "RATE_LIMIT_TEMPLATE", 429),
    "responses-stream-error-event-invalid-key": ("responses", True, _responses_error_event("error", "invalid_api_key"), "AUTHENTICATION_ERROR_TEMPLATE", 401),
    "responses-stream-error-event-blocked-image": ("responses", True, _responses_error_event("response.error", "image_content_policy_violation"), "OPENROUTER_ERROR_TEMPLATE", 403),
    "responses-stream-error-event-unknown-code": ("responses", True, _responses_error_event("error", "wolves_ate_the_response"), "OPENROUTER_ERROR_TEMPLATE", 400),
    # the remaining 4xx/5xx rows of _IN_BAND_STATUS_BY_ERROR_TYPE that the chat skin can carry, sent with a
    # numeric code of 400 so the arm fails if the kind stops deciding the status. Not every row of the table:
    # the eleven kinds OpenRouter documents at 400 are covered on the rejection path instead, by
    # test_a_rejection_and_a_mid_reply_failure_read_the_same_way.py::test_every_documented_400_kind_is_read_as_a_bad_request
    "chat-stream-permission-denied": ("chat_completions", True, _chat_stream(400, "permission_denied"), "OPENROUTER_ERROR_TEMPLATE", 403),
    "chat-stream-content-policy": ("chat_completions", True, _chat_stream(400, "content_policy_violation"), "OPENROUTER_ERROR_TEMPLATE", 403),
    "chat-stream-refusal": ("chat_completions", True, _chat_stream(400, "refusal"), "OPENROUTER_ERROR_TEMPLATE", 403),
    "chat-stream-not-found": ("chat_completions", True, _chat_stream(400, "not_found"), "OPENROUTER_ERROR_TEMPLATE", 404),
    "chat-stream-image-not-found": ("chat_completions", True, _chat_stream(400, "image_not_found"), "OPENROUTER_ERROR_TEMPLATE", 404),
    "chat-stream-precondition-failed": ("chat_completions", True, _chat_stream(400, "precondition_failed"), "OPENROUTER_ERROR_TEMPLATE", 412),
    "chat-stream-payload-too-large": ("chat_completions", True, _chat_stream(400, "payload_too_large"), "PAYLOAD_TOO_LARGE_TEMPLATE", 413),
    "chat-stream-unprocessable": ("chat_completions", True, _chat_stream(400, "unprocessable"), "OPENROUTER_ERROR_TEMPLATE", 422),
    "chat-stream-unmapped": ("chat_completions", True, _chat_stream(400, "unmapped"), "SERVICE_ERROR_TEMPLATE", 500),
    "chat-stream-timed-out-in-band": ("chat_completions", True, _chat_stream(408, "request_timeout"), "SERVER_TIMEOUT_TEMPLATE", 408),
}


# The service card prints the status the pipe resolved, and the built-in template explains that number to the
# reader, so the number itself is part of what the person is told.
@pytest.mark.asyncio
@pytest.mark.parametrize("arm", list(_ARMS))
async def test_a_failure_reported_inside_a_started_reply_shows_the_message_for_that_failure(monkeypatch, arm):
    endpoint, stream, body, expected, expected_status = _ARMS[arm]
    pipe = _pipe_reaching_the_model(monkeypatch, endpoint)
    try:
        with aioresponses() as mock_http:
            mock_http.post(_RESPONSES_URL if endpoint == "responses" else _CHAT_URL, status=200, body=body, repeat=True)
            reply = await _turn(pipe, stream=stream)
    finally:
        await pipe.close()

    assert _cards(reply) == [expected], reply
    assert _MARKERS[expected].format(status_code=expected_status) in reply, reply


# ---------------------------------------------------------------------------
# what the card carries, and what the failure does to the rest of the session
# ---------------------------------------------------------------------------


_MODERATION_METADATA = {
    "error_type": "content_policy_violation",
    "reasons": ["hate", "violence"],
    "flagged_input": "the exact words that were flagged",
    "provider_name": "Google",
    "model_slug": "google/gemini-3-pro",
}
_RATE_LIMIT_METADATA = {
    "error_type": "rate_limit_exceeded",
    "rate_limit_type": "per-minute",
    "provider_name": "OpenAI",
}


def _rendered_values(error: Any) -> dict[str, Any]:
    from open_webui_openrouter_pipe.core.errors import _build_error_template_values

    return _build_error_template_values(
        error,
        heading="Provider: m1",
        diagnostics=[],
        metrics={},
        model_identifier="m1",
        normalized_model_id="m1",
        api_model_id="m1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata, shown",
    [
        pytest.param(
            _MODERATION_METADATA,
            {
                "moderation_reasons": "- hate\n- violence",
                "flagged_excerpt": "```\nthe exact words that were flagged\n```",
                "provider": "Google",
            },
            id="a-content-block-says-why",
        ),
        pytest.param(
            _RATE_LIMIT_METADATA,
            {"rate_limit_type": "per-minute", "provider": "OpenAI"},
            id="a-rate-limit-says-which-limit",
        ),
    ],
)
async def test_a_card_says_the_same_thing_whether_the_failure_arrived_as_a_status_or_inside_a_reply(metadata, shown):
    """OpenRouter sends the same `error.metadata` both ways, so the card must not lose it one way.

    A provider reports a content block after OpenRouter has already committed `200 OK`, which is the
    delivery the built-in rejected-request message's moderation rows exist for. Reading that block only
    for its typed kind and discarding the rest left those rows empty exactly when they matter, while the
    same body returned as a rejection filled them.

    Each arm asserts the rendered values, not the presence of the words: the whole event is also dumped
    into the raw-JSON placeholder, so a substring check would pass against the defect.
    """
    from open_webui_openrouter_pipe.core.errors import _build_openrouter_api_error

    body = {"error": {"code": 403, "message": "This request was declined.", "metadata": metadata}}
    rejected = _build_openrouter_api_error(403, "Forbidden", json.dumps(body))
    pipe = Pipe()
    try:
        in_band = pipe._ensure_error_formatter()._build_streaming_openrouter_error(body, requested_model="m1")
    finally:
        await pipe.close()

    rejected_values, in_band_values = _rendered_values(rejected), _rendered_values(in_band)
    for name, expected in shown.items():
        assert rejected_values[name] == expected, (name, "as a rejection")
        assert in_band_values[name] == expected, (name, "inside a reply")


# --- what the service card tells an admin a 503 means ------------------------------------------------------

# Prescription 122 split the `503` line, which had stated routing constraints as THE cause, and added the
# missing `504` row. The wording was fixed but nothing observed it: round 32's verify lens measured both
# reverts SURVIVING the whole suite. These assert the factory template, not a rendered card -- a rendered card
# carries OpenRouter's own `{reason}`, so a substring check there passes against the defect.
def test_the_service_card_offers_both_meanings_of_a_503_rather_than_only_routing():
    from open_webui_openrouter_pipe.core.config import DEFAULT_SERVICE_ERROR_TEMPLATE as template

    line = next(ln for ln in template.splitlines() if ln.startswith("- `503`"))
    assert "overloaded" in line, line
    assert " or " in line, f"the 503 line states one cause as though it were the only one: {line}"


def test_the_service_card_explains_the_504_it_is_rendered_for():
    from open_webui_openrouter_pipe.core.config import DEFAULT_SERVICE_ERROR_TEMPLATE as template
    from open_webui_openrouter_pipe.core.error_formatter import _IN_BAND_STATUS_BY_ERROR_TYPE

    assert _IN_BAND_STATUS_BY_ERROR_TYPE["timeout"] == 504
    assert any(ln.startswith("- `504`") for ln in template.splitlines()), template


def test_the_service_cards_routing_advice_is_offered_as_a_possibility_not_a_diagnosis():
    from open_webui_openrouter_pipe.core.config import DEFAULT_SERVICE_ERROR_TEMPLATE as template

    advice = next(ln for ln in template.splitlines() if "routing constraints" in ln)
    assert advice.lstrip("- ").startswith("If a `503` keeps repeating"), advice
    assert "may be" in advice, f"the advice names routing as the cause rather than a possibility: {advice}"


def _rejection(code: Any, message: str, **metadata: Any) -> bytes:
    error: dict[str, Any] = {"code": code, "message": message}
    if metadata:
        error["metadata"] = metadata
    return json.dumps({"error": error}).encode("utf-8")
