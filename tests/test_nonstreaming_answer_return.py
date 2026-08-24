"""With streaming off, the only answer Open WebUI keeps is the one the pipe RETURNS.

Open WebUI's ``non_streaming_chat_response_handler`` gates the whole tail of the turn on
``if choices and (content or response_output)``. Returning ``""`` makes that false, and
the branch it skips is not just the message write: it is also ``ctx['assistant_message']``,
and with it the outlet filters and every background task -- title, tags, follow-ups. The
socket emitter that puts the card on screen writes ``status``, ``message``, ``replace``,
``embeds``, ``files`` and ``source``/``citation`` to the database and nothing else, so a
``chat:message`` card is shown and never stored. Net effect: the user reads a perfectly
good error card, reloads, and the assistant turn is blank with no title.

Every test here drives the NON-STREAMING leg (``stream`` absent or false) and asserts the
returned value against the text that actually reached the emitter, rather than against a
literal written into the test. Where a seam is stubbed it is stubbed one level BELOW the
function under test, and parametrised over two distinct values, so ``return "<constant>"``
in production cannot satisfy both rows.

The task-model calls -- title, tags, follow-ups, query generation -- travel through the
same two functions but are not chat turns: Open WebUI parses their return value as data.
A Markdown card returned there lands in the chat's title or its tag list, so those rows
assert the JSON fallback instead of the card.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import httpx
import pytest

import open_webui_openrouter_pipe.pipe as pipe_mod
from open_webui_openrouter_pipe import EncryptedStr, Pipe
from open_webui_openrouter_pipe.core.errors import (
    OpenRouterAPIError,
    RequiredInternalFileError,
)


class _NoSession:
    """Stand-in for the aiohttp session; nothing under test ever calls it."""


def _catalog_entry(model_id: str) -> dict[str, str]:
    return {"id": model_id, "name": f"Model {model_id}", "norm_id": model_id}


def _shown_card(events: list[dict[str, Any]]) -> str:
    """The markdown Open WebUI put on screen, taken from the wire and not from a literal."""
    cards = [
        event["data"]["content"]
        for event in events
        if isinstance(event, dict)
        and event.get("type") == "chat:message"
        and isinstance(event.get("data"), dict)
        and isinstance(event["data"].get("content"), str)
    ]
    assert cards, f"nothing was shown to the user; events were {events}"
    return cards[-1]


def _shown_error(events: list[dict[str, Any]]) -> str:
    """The message Open WebUI put in the error banner, taken from the wire."""
    errors = [
        event["data"]["error"]["message"]
        for event in events
        if isinstance(event, dict)
        and event.get("type") == "chat:completion"
        and isinstance(event.get("data"), dict)
        and isinstance(event["data"].get("error"), dict)
        and isinstance(event["data"]["error"].get("message"), str)
    ]
    assert errors, f"nothing was shown to the user; events were {events}"
    return errors[-1]


def _prepared(pipe: Pipe, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: ("sk-test", None))
    monkeypatch.setattr(pipe._artifact_store, "_ensure_artifact_store", lambda *_a, **_k: None)

    async def _loaded(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", _loaded)
    monkeypatch.setattr(
        pipe_mod.OpenRouterModelRegistry, "list_models", lambda: [_catalog_entry("m1")]
    )


async def _call(
    pipe: Pipe,
    events: list[dict[str, Any]],
    *,
    task: Any = None,
    stream: bool = False,
) -> Any:
    async def emitter(event: dict[str, Any]) -> None:
        events.append(event)

    return await pipe._handle_pipe_call(
        {"stream": stream, "model": "m1"},
        {},
        None,
        emitter,
        None,
        {},
        None,
        task,
        None,
        valves=pipe.valves,
        session=cast(Any, _NoSession()),
    )


# ---------------------------------------------------------------------------
# Templated cards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised", "marker"),
    [
        (httpx.TimeoutException("slow"), "NETWORK_TIMEOUT_TEMPLATE"),
        (httpx.ConnectError("refused"), "CONNECTION_ERROR_TEMPLATE"),
    ],
)
async def test_a_templated_card_is_returned_as_well_as_shown(
    monkeypatch, pipe_instance_async, raised, marker
) -> None:
    """Two different templates, so a card hardcoded in production fails one of them."""
    pipe = pipe_instance_async
    _prepared(pipe, monkeypatch)
    pipe.valves.NETWORK_TIMEOUT_TEMPLATE = "### Timed out\n\nWaited too long for {endpoint}."
    pipe.valves.CONNECTION_ERROR_TEMPLATE = "### Unreachable\n\nCould not open {endpoint}."

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise raised

    monkeypatch.setattr(pipe, "_process_transformed_request", _raise)

    events: list[dict[str, Any]] = []
    result = await _call(pipe, events)

    shown = _shown_card(events)
    assert result == shown
    expected_heading = "Timed out" if marker == "NETWORK_TIMEOUT_TEMPLATE" else "Unreachable"
    assert expected_heading in shown


@pytest.mark.asyncio
@pytest.mark.parametrize("template", ["### Boom\n\nInternal.", "### Kaput\n\nAlso internal."])
async def test_the_generic_card_is_returned_as_well_as_shown(
    monkeypatch, pipe_instance_async, template
) -> None:
    pipe = pipe_instance_async
    _prepared(pipe, monkeypatch)
    pipe.valves.INTERNAL_ERROR_TEMPLATE = template

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("unexpected")

    monkeypatch.setattr(pipe, "_process_transformed_request", _raise)

    events: list[dict[str, Any]] = []
    result = await _call(pipe, events)

    assert result == _shown_card(events)
    assert result == template


# ---------------------------------------------------------------------------
# OpenRouter rejection cards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "heading"),
    [(402, "Out of credits"), (429, "Too many requests")],
)
async def test_an_openrouter_rejection_card_is_returned_as_well_as_shown(
    monkeypatch, pipe_instance_async, status, heading
) -> None:
    """Two statuses select two different templates, so one constant cannot serve both."""
    pipe = pipe_instance_async
    _prepared(pipe, monkeypatch)
    pipe.valves.INSUFFICIENT_CREDITS_TEMPLATE = "### Out of credits\n\n{openrouter_message}"
    pipe.valves.RATE_LIMIT_TEMPLATE = "### Too many requests\n\n{openrouter_message}"

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise OpenRouterAPIError(
            status=status,
            reason="rejected",
            openrouter_message=f"provider said {status}",
        )

    monkeypatch.setattr(pipe, "_process_transformed_request", _raise)

    events: list[dict[str, Any]] = []
    result = await _call(pipe, events)

    shown = _shown_card(events)
    assert result == shown
    assert heading in shown
    assert f"provider said {status}" in shown


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "heading"), [(502, "Upstream down"), (503, "Upstream down")])
async def test_an_http_status_error_card_is_returned_as_well_as_shown(
    monkeypatch, pipe_instance_async, status, heading
) -> None:
    pipe = pipe_instance_async
    _prepared(pipe, monkeypatch)
    pipe.valves.SERVICE_ERROR_TEMPLATE = "### Upstream down\n\nHTTP {status_code} {reason}."

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/responses")
    response = httpx.Response(status, request=request)

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise httpx.HTTPStatusError("boom", request=request, response=response)

    monkeypatch.setattr(pipe, "_process_transformed_request", _raise)

    events: list[dict[str, Any]] = []
    result = await _call(pipe, events)

    shown = _shown_card(events)
    assert result == shown
    assert heading in shown
    assert str(status) in shown


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "heading"),
    [(429, "Too many requests"), (413, "Too large")],
)
async def test_a_sub_500_http_status_error_card_is_returned_as_well_as_shown(
    monkeypatch, pipe_instance_async, status, heading
) -> None:
    """Below 500 the handler rebuilds the failure as an OpenRouter rejection instead.

    Two statuses select two different templates, so this cannot be satisfied by a
    constant, and neither row overlaps the 5xx branch above.
    """
    pipe = pipe_instance_async
    _prepared(pipe, monkeypatch)
    pipe.valves.RATE_LIMIT_TEMPLATE = "### Too many requests\n\nHTTP {openrouter_code}."
    pipe.valves.PAYLOAD_TOO_LARGE_TEMPLATE = "### Too large\n\nHTTP {openrouter_code}."

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/responses")
    response = httpx.Response(status, request=request, content=b'{"error": {"message": "no"}}')

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise httpx.HTTPStatusError("boom", request=request, response=response)

    monkeypatch.setattr(pipe, "_process_transformed_request", _raise)

    events: list[dict[str, Any]] = []
    result = await _call(pipe, events)

    shown = _shown_card(events)
    assert result == shown
    assert heading in shown


# ---------------------------------------------------------------------------
# Error-block frames
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "denial",
    ["You do not have access to that file.", "That attachment could not be prepared."],
)
async def test_an_error_frame_is_returned_as_well_as_shown(
    monkeypatch, pipe_instance_async, denial
) -> None:
    pipe = pipe_instance_async
    _prepared(pipe, monkeypatch)

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise RequiredInternalFileError(denial, denied=True)

    monkeypatch.setattr(pipe, "_process_transformed_request", _raise)

    events: list[dict[str, Any]] = []
    result = await _call(pipe, events)

    assert result == _shown_error(events)
    assert result == denial


@pytest.mark.asyncio
@pytest.mark.parametrize("detail", ["bad base url", "malformed key"])
async def test_a_catalog_configuration_error_is_returned_as_well_as_shown(
    monkeypatch, pipe_instance_async, detail
) -> None:
    pipe = pipe_instance_async
    _prepared(pipe, monkeypatch)

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError(detail)

    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", _raise)

    events: list[dict[str, Any]] = []
    result = await _call(pipe, events)

    assert result == _shown_error(events)
    assert detail in result


@pytest.mark.asyncio
@pytest.mark.parametrize("boom", [RuntimeError("catalog down"), OSError("dns")])
async def test_an_unusable_catalog_is_returned_as_well_as_shown(
    monkeypatch, pipe_instance_async, boom
) -> None:
    pipe = pipe_instance_async
    _prepared(pipe, monkeypatch)

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise boom

    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", _raise)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "list_models", lambda: [])

    events: list[dict[str, Any]] = []
    result = await _call(pipe, events)

    assert result == _shown_error(events)
    assert result


# ---------------------------------------------------------------------------
# Task-model calls must NOT receive a card
# ---------------------------------------------------------------------------


_TASK_ROWS = [
    ("title_generation", "title", "Chat"),
    ("tags_generation", "tags", ["General"]),
    ("follow_up_generation", "follow_ups", []),
]

_TASK_FAILURES = [
    httpx.TimeoutException("slow"),
    httpx.ConnectError("refused"),
    OpenRouterAPIError(status=402, reason="rejected", openrouter_message="no credit left"),
    RequiredInternalFileError("that attachment is not yours", denied=True),
    ValueError("unexpected"),
]

_TASK_FAILURE_IDS = ["timeout", "connect", "rejection", "internal-file", "unexpected"]

_VISIBLE_MESSAGE_FRAMES = (
    "chat:message",
    "chat:message:delta",
    "chat:completion",
    "status",
)


def _visible_message_frames(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every frame Open WebUI applies to the assistant message the task was launched from.

    ``chat:message`` replaces ``message.content`` outright, ``chat:message:delta`` appends
    to it, ``chat:completion`` is read for ``content``, ``error`` and ``usage`` and then
    marks the message done, and ``status`` is appended to ``statusHistory`` and written to
    the database under the same message id. A background task shares the parent turn's
    ``message_id``, so any of the four lands on an answer the user has already read.

    ``notification`` is deliberately not here: Open WebUI shows it as a toast and never
    attaches it to a message, so it is the one channel a background task may still use.
    """
    return [
        event
        for event in events
        if isinstance(event, dict) and event.get("type") in _VISIBLE_MESSAGE_FRAMES
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("task", "key", "expected"), _TASK_ROWS)
@pytest.mark.parametrize("raised", _TASK_FAILURES, ids=_TASK_FAILURE_IDS)
async def test_a_failed_task_call_returns_parseable_data_not_a_card(
    monkeypatch, pipe_instance_async, task, key, expected, raised
) -> None:
    """The same failure that yields a card on a chat turn must not title a chat with one.

    Five failures are driven because five different branches build the reply, and each
    reaches a different emitter helper: two templated cards, an OpenRouter rejection
    card, an error frame and the generic catch-all. A guard added to one of them cannot
    satisfy the set. Three tasks name three different objects, so a constant returned by
    the fallback builder cannot satisfy the set either.
    """
    pipe = pipe_instance_async
    _prepared(pipe, monkeypatch)
    pipe.valves.INTERNAL_ERROR_TEMPLATE = "### Boom\n\nSomething broke."

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise raised

    monkeypatch.setattr(pipe, "_process_transformed_request", _raise)

    events: list[dict[str, Any]] = []
    result = await _call(pipe, events, task=task)

    assert isinstance(result, str)
    assert json.loads(result) == {key: expected}
    assert "Boom" not in result
    assert _visible_message_frames(events) == [], (
        "a background task overwrote the answer the user was already reading"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("task", "key", "expected"), _TASK_ROWS)
@pytest.mark.parametrize("raised", _TASK_FAILURES, ids=_TASK_FAILURE_IDS)
async def test_a_failed_task_call_keeps_the_toast_and_drops_the_rest(
    monkeypatch, pipe_instance_async, task, key, expected, raised
) -> None:
    """The toast channel stays open; every channel that writes to the message closes.

    The frames are emitted from inside the stubbed request processor, which is the seam
    every downstream emitter reaches the socket through -- the orchestrator and the image
    and video adapters included -- so this covers far more than the branches in
    ``_handle_pipe_call`` itself. A guard that closed everything would pass the second
    assertion and fail the first, and one that closed nothing would do the reverse.
    """
    pipe = pipe_instance_async
    _prepared(pipe, monkeypatch)

    async def _emit_then_raise(*args: Any, **_kwargs: Any) -> None:
        downstream = args[3]
        await downstream({"type": "status", "data": {"description": f"working on {task}"}})
        await downstream(
            {"type": "notification", "data": {"type": "warning", "content": f"heads up {task}"}}
        )
        await downstream({"type": "chat:message", "data": {"content": "clobber"}})
        raise raised

    monkeypatch.setattr(pipe, "_process_transformed_request", _emit_then_raise)

    events: list[dict[str, Any]] = []
    result = await _call(pipe, events, task=task)

    assert json.loads(result) == {key: expected}
    assert [
        event["data"].get("content") for event in events if event["type"] == "notification"
    ] == [f"heads up {task}"]
    assert _visible_message_frames(events) == [], (
        "a background task wrote to the answer the user was already reading"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("task", "key", "expected"), _TASK_ROWS)
async def test_a_task_rejected_before_it_is_queued_leaves_the_answer_alone(
    monkeypatch, pipe_instance_async, task, key, expected
) -> None:
    """``pipe`` refuses some work before the job exists, and that leg has its own emitter.

    Driving it through ``pipe`` rather than ``_handle_pipe_call`` is the point: the guard
    that closes the visible channel has to sit above the enqueue, because the refusal
    never reaches the job. The task rows differ, so the returned string is checked to be
    the refusal itself rather than any one literal.
    """
    pipe = pipe_instance_async
    pipe._warmup_failed = True
    pipe._startup_task = None
    pipe._request_queue = asyncio.Queue()

    async def _noop(_valves: Any) -> None:
        return None

    monkeypatch.setattr(pipe, "_ensure_concurrency_controls", _noop)

    events: list[dict[str, Any]] = []

    async def emitter(event: dict[str, Any]) -> None:
        events.append(event)

    result = await pipe.pipe(
        {"stream": False, "model": "m1"},
        {"id": "user"},
        None,
        emitter,
        None,
        {},
        None,
        task,
        None,
    )

    assert "Service unavailable" in str(result)
    assert _visible_message_frames(events) == [], (
        "a refusal that never reached the queue still painted over the answer"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task", "key", "expected"),
    [
        ("title_generation", "title", "Chat"),
        ("tags_generation", "tags", ["General"]),
        ("follow_up_generation", "follow_ups", []),
    ],
)
@pytest.mark.parametrize(
    "boom",
    [ValueError("bad base url"), RuntimeError("catalog down")],
    ids=["configuration", "unavailable"],
)
async def test_a_task_call_that_hits_an_unusable_catalog_returns_data_not_a_card(
    monkeypatch, pipe_instance_async, task, key, expected, boom
) -> None:
    """Both catalog exits are driven: the configuration one and the unavailable one.

    Each row names the whole object its task must produce. A key-subset check would have
    accepted ``{}``, and ``{}`` is the failure: Open WebUI reads ``title`` out of the
    returned JSON to name the chat, so an empty object leaves the chat untitled exactly as
    a Markdown card would. The three tasks expect three different objects, so a constant
    returned from the fallback builder cannot satisfy the set either.
    """
    pipe = pipe_instance_async
    _prepared(pipe, monkeypatch)

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise boom

    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", _raise)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "list_models", lambda: [])

    events: list[dict[str, Any]] = []
    result = await _call(pipe, events, task=task)

    assert isinstance(result, str)
    assert json.loads(result) == {key: expected}


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["query_generation", "function_calling"])
async def test_a_task_with_no_json_fallback_returns_nothing_rather_than_a_card(
    monkeypatch, pipe_instance_async, task
) -> None:
    """No shape is known for these, so the pipe must stay silent rather than invent one."""
    pipe = pipe_instance_async
    _prepared(pipe, monkeypatch)
    pipe.valves.INTERNAL_ERROR_TEMPLATE = "### Boom\n\nSomething broke."

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("unexpected")

    monkeypatch.setattr(pipe, "_process_transformed_request", _raise)

    events: list[dict[str, Any]] = []
    result = await _call(pipe, events, task=task)

    assert result == ""


# ---------------------------------------------------------------------------
# The streaming leg is unaffected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("template", ["### Boom\n\nInternal.", "### Kaput\n\nAlso internal."])
async def test_the_streaming_leg_still_shows_the_card(
    monkeypatch, pipe_instance_async, template
) -> None:
    """With streaming on the return value is discarded, so only the emission matters."""
    pipe = pipe_instance_async
    _prepared(pipe, monkeypatch)
    pipe.valves.INTERNAL_ERROR_TEMPLATE = template

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("unexpected")

    monkeypatch.setattr(pipe, "_process_transformed_request", _raise)

    events: list[dict[str, Any]] = []
    await _call(pipe, events, stream=True)

    assert _shown_card(events) == template


# ---------------------------------------------------------------------------
# The orchestrator's own exits
# ---------------------------------------------------------------------------


def _orchestrator_pipe() -> Pipe:
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
    return pipe


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("template", "marker"),
    [
        ("### Blocked\n\n{requested_model} is off limits.", "Blocked"),
        ("### Not allowed\n\n{requested_model} cannot be used.", "Not allowed"),
    ],
)
async def test_a_restricted_model_card_is_returned_as_well_as_shown(
    monkeypatch, template, marker
) -> None:
    """Straight through the orchestrator, with the real template renderer in the path."""
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

    pipe = _orchestrator_pipe()
    pipe.valves.MODEL_RESTRICTED_TEMPLATE = template
    events: list[dict[str, Any]] = []

    async def emitter(event: dict[str, Any]) -> None:
        events.append(event)

    monkeypatch.setattr(OpenRouterModelRegistry, "is_zdr_capable", staticmethod(lambda _m: True))
    try:
        orchestrator = pipe._ensure_request_orchestrator()
        result = await orchestrator.process_request(
            body={"model": "blocked/model", "messages": [{"role": "user", "content": "hi"}],
                  "stream": False},
            __user__={"id": "user-1"},
            __request__=None,
            __event_emitter__=emitter,
            __event_call__=None,
            __metadata__={},
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=pipe.valves,
            session=cast(Any, _NoSession()),
            openwebui_model_id="blocked/model",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"other/model"},
            enforced_norm_ids={"other/model"},
            catalog_norm_ids={"other/model", "blocked/model"},
            features={},
        )

        shown = _shown_card(events)
        assert result == shown
        assert marker in shown
    finally:
        await pipe.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task", "wants_card"),
    [(None, True), ("title_generation", False), ("tags_generation", False)],
)
async def test_a_preset_endpoint_conflict_never_hands_a_card_to_a_task(
    monkeypatch, task, wants_card
) -> None:
    """A preset on a /responses-forced model aborts the turn before any model is chosen.

    Both arms of the discriminator are driven: a chat turn must get the card back, and a
    task call must get its JSON fallback instead, because Open WebUI parses a task's
    return value as data and would otherwise title the chat with Markdown.
    """
    pipe = _orchestrator_pipe()
    pipe.valves.FORCE_RESPONSES_MODELS = "forced/model"
    pipe.valves.ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE = (
        "### Endpoint conflict\n\n{requested_model} needs {required_endpoint}."
    )
    events: list[dict[str, Any]] = []

    async def emitter(event: dict[str, Any]) -> None:
        events.append(event)

    try:
        orchestrator = pipe._ensure_request_orchestrator()
        result = await orchestrator.process_request(
            body={
                "model": "forced/model",
                "messages": [{"role": "user", "content": "hi"}],
                "preset": "some-preset",
                "stream": False,
            },
            __user__={"id": "user-1"},
            __request__=None,
            __event_emitter__=emitter,
            __event_call__=None,
            __metadata__={},
            __tools__=None,
            __task__=task,
            __task_body__=None,
            valves=pipe.valves,
            session=cast(Any, _NoSession()),
            openwebui_model_id="forced/model",
            pipe_identifier="test-pipe",
            allowlist_norm_ids=set(),
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        shown = _shown_card(events)
        assert "Endpoint conflict" in shown
        if wants_card:
            assert result == shown
        else:
            assert result != shown
            assert isinstance(result, str)
            assert json.loads(result) in ({"title": "Chat"}, {"tags": ["General"]})
    finally:
        await pipe.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("template", "marker"),
    [
        ("### Privacy hold\n\n{restriction_reasons}", "Privacy hold"),
        ("### Cannot verify\n\n{restriction_reasons}", "Cannot verify"),
    ],
)
async def test_an_unverifiable_zdr_card_is_returned_as_well_as_shown(
    monkeypatch, template, marker
) -> None:
    """ZDR enforcement fails closed when the endpoint list cannot be read.

    The card that says so is the whole turn, so with streaming off it has to be the
    return value too or the refusal is not recorded anywhere.
    """
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

    pipe = _orchestrator_pipe()
    pipe.valves.ZDR_ENFORCE = True
    pipe.valves.MODEL_RESTRICTED_TEMPLATE = template
    events: list[dict[str, Any]] = []

    async def emitter(event: dict[str, Any]) -> None:
        events.append(event)

    monkeypatch.setattr(OpenRouterModelRegistry, "is_zdr_capable", staticmethod(lambda _m: None))
    try:
        orchestrator = pipe._ensure_request_orchestrator()
        result = await orchestrator.process_request(
            body={"model": "some/model", "messages": [{"role": "user", "content": "hi"}],
                  "stream": False},
            __user__={"id": "user-1"},
            __request__=None,
            __event_emitter__=emitter,
            __event_call__=None,
            __metadata__={},
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=pipe.valves,
            session=cast(Any, _NoSession()),
            openwebui_model_id="some/model",
            pipe_identifier="test-pipe",
            allowlist_norm_ids=set(),
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        from open_webui_openrouter_pipe.pipe import _RESTRICTION_REASON_PHRASES

        unreadable = _RESTRICTION_REASON_PHRASES["ZDR_LIST_UNAVAILABLE"]
        shown = _shown_card(events)
        assert result == shown
        assert marker in shown
        assert unreadable in shown, (
            "this is the arm where the ZDR list could not be read, and the card has to say "
            f"so rather than name a model that has no ZDR endpoint:\n{shown}"
        )
    finally:
        await pipe.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("template", "marker"),
    [
        ("### Wrong endpoint\n\n{reason}", "Wrong endpoint"),
        ("### Cannot route\n\n{reason}", "Cannot route"),
    ],
)
async def test_a_direct_upload_endpoint_conflict_card_is_returned_as_well_as_shown(
    monkeypatch, template, marker
) -> None:
    """A video attachment needs /chat/completions; the valve pins the model to /responses."""
    from open_webui_openrouter_pipe.core.config import _PIPE_METADATA_KEY

    pipe = _orchestrator_pipe()
    pipe.valves.FORCE_RESPONSES_MODELS = "forced/model"
    pipe.valves.ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE = template
    events: list[dict[str, Any]] = []

    async def emitter(event: dict[str, Any]) -> None:
        events.append(event)

    class _FileRecord:
        id = "vid-1"
        filename = "v.mp4"
        meta = {"content_type": "video/mp4"}

    async def _get_file(*_args: Any, **_kwargs: Any) -> Any:
        return _FileRecord()

    async def _read_b64(*_args: Any, **_kwargs: Any) -> str:
        return "AAAA"

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.requests.orchestrator.get_file_by_id",
        _get_file,
    )
    monkeypatch.setattr(pipe._file_gateway, "read_file_record_base64", _read_b64)
    try:
        orchestrator = pipe._ensure_request_orchestrator()
        result = await orchestrator.process_request(
            body={"model": "forced/model", "messages": [{"role": "user", "content": "hi"}],
                  "stream": False},
            __user__={"id": "user-1"},
            __request__=None,
            __event_emitter__=emitter,
            __event_call__=None,
            __metadata__={
                _PIPE_METADATA_KEY: {
                    "direct_uploads": {"video": [{"id": "vid-1", "url": "https://x/v.mp4"}]}
                }
            },
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=pipe.valves,
            session=cast(Any, _NoSession()),
            openwebui_model_id="forced/model",
            pipe_identifier="test-pipe",
            allowlist_norm_ids=set(),
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        shown = _shown_card(events)
        assert result == shown
        assert marker in shown
    finally:
        await pipe.close()


# ---------------------------------------------------------------------------
# A truncated answer must say why it stopped
# ---------------------------------------------------------------------------


def _streamed(events: list[dict[str, Any]]) -> str:
    """What a browser would have accumulated from the frames that actually went out."""
    return "".join(
        event["data"]["content"]
        for event in events
        if isinstance(event, dict)
        and event.get("type") == "chat:message:delta"
        and isinstance(event.get("data"), dict)
        and isinstance(event["data"].get("content"), str)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("partial", "raised"),
    [
        ("the opening half of one answer", RuntimeError),
        ("a quite different opening half", ValueError),
    ],
)
async def test_a_truncated_answer_comes_back_carrying_the_reason_it_stopped(
    pipe_instance_async, partial, raised
) -> None:
    """The response loop's catch-all showed a card and returned the half-answer alone.

    With streaming off the card never reaches the browser at all: the non-streaming leg
    reuses the streaming loop behind an emitter that drops every ``chat:message``, so the
    only channel left is the return value -- and the handler assigned it nothing. The user
    was handed a sentence that stops mid-thought with no indication anything had failed.

    Both halves are asserted. Returning only the card would throw away text the user has
    already been billed for, and returning only the half-answer is the defect itself.

    The stream is stubbed one seam below the loop, at the event source it iterates, and the
    rows differ in BOTH the text and the exception type, so neither half can come from a
    constant.
    """
    from open_webui_openrouter_pipe import ResponsesBody

    pipe = pipe_instance_async

    async def source():
        yield {"type": "response.created", "response": {"model": "m1"}}
        yield {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}}
        yield {"type": "response.output_text.delta", "output_index": 0, "delta": partial}
        raise raised("the upstream connection died")

    events: list[dict[str, Any]] = []

    async def emitter(event: dict[str, Any]) -> None:
        events.append(event)

    body = ResponsesBody(model="m1", input=[], stream=False)
    result = await pipe._streaming_handler._run_nonstreaming_loop(
        body,
        pipe.valves,
        emitter,
        metadata={},
        tools={},
        session=cast(Any, _NoSession()),
        user_id="u",
        event_source=source(),
    )

    assert not _streamed(events), (
        "this leg is only interesting because nothing is streamed on it; if frames went out "
        f"the suppression is gone and this proves nothing. got {events!r}"
    )
    assert isinstance(result, str)
    assert partial in result, (
        "the model produced this much before the failure and the account was charged for "
        f"it; replacing it with a card throws it away. got {result!r}"
    )
    assert raised.__name__ in result, (
        "the answer stops mid-sentence and the return value is the only thing Open WebUI "
        f"keeps, so it has to carry the reason. got {result!r}"
    )
    assert result.index(partial) < result.index(raised.__name__), (
        f"the reason must follow the answer it interrupted, not precede it. got {result!r}"
    )


# ---------------------------------------------------------------------------
# A caller with no chat still gets the card
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("template", "variables", "expected"),
    [
        ("### Rejected\n\nThe endpoint refused {reason}.", {"reason": "the payload"}, "the payload"),
        ("### Blocked\n\nNothing served {reason}.", {"reason": "this model"}, "this model"),
    ],
)
async def test_a_templated_card_is_built_even_with_nobody_to_show_it_to(
    pipe_instance_async, template, variables, expected
) -> None:
    """Open WebUI supplies no event emitter to a caller with no chat context.

    The API surface still has to answer that caller with something. The helper returned the
    empty string before it rendered anything, and every call site returns what it hands
    back, so an API caller's failure came out as empty content.

    Two templates and two substitutions, so a constant satisfies neither row.
    """
    pipe = pipe_instance_async
    card = await pipe._ensure_error_formatter()._emit_templated_error(
        None,
        template=template,
        variables=variables,
        log_message="probe",
    )
    assert isinstance(card, str) and card.strip(), (
        "an unshowable failure still has to be answerable; empty content is not an answer"
    )
    assert expected in card, f"the card was not rendered from the template it was given: {card!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "message"), [(402, "out of credits"), (429, "too many")])
async def test_an_openrouter_rejection_is_rendered_even_with_nobody_to_show_it_to(
    pipe_instance_async, status, message
) -> None:
    """Same property on the OpenRouter path, which selects its template from the status.

    Two statuses select two different templates, so one rendered constant cannot pass both.
    The heading each row expects is taken from the selector rather than written out here,
    so this cannot drift from whatever the templates say.
    """
    pipe = pipe_instance_async
    formatter = pipe._ensure_error_formatter()
    other = 429 if status == 402 else 402
    heading = formatter._select_openrouter_template(status).splitlines()[0]
    other_heading = formatter._select_openrouter_template(other).splitlines()[0]
    assert "{" not in heading, f"this row's anchor is not a fixed line: {heading!r}"
    assert heading != other_heading, (
        f"{status} and {other} select the same card, so this row cannot tell which was used"
    )

    card = await formatter._report_openrouter_error(
        OpenRouterAPIError(status=status, reason="rejected", openrouter_message=message),
        event_emitter=None,
        normalized_model_id="m1",
        api_model_id="m1",
    )
    assert isinstance(card, str) and card.strip(), (
        "an unshowable rejection still has to be answerable; empty content is not an answer"
    )
    assert heading in card, (
        f"a {status} did not render the template its status selects: {card!r}"
    )
    assert other_heading not in card, (
        f"a {status} rendered the card that belongs to a {other}: {card!r}"
    )


# ---------------------------------------------------------------------------
# Every branch, driven with nobody to show it to
# ---------------------------------------------------------------------------


async def _call_without_emitter(pipe: Pipe, *, task: Any = None, stream: bool = False) -> Any:
    """The same entry point, for the caller Open WebUI hands no event emitter."""
    return await pipe._handle_pipe_call(
        {"stream": stream, "model": "m1"},
        {},
        None,
        None,
        None,
        {},
        None,
        task,
        None,
        valves=pipe.valves,
        session=cast(Any, _NoSession()),
    )


def _arrange_failure(pipe: Pipe, monkeypatch, branch: str, variant: str) -> str:
    """Make one branch of ``_handle_pipe_call`` fire, and return the words it must carry."""
    _prepared(pipe, monkeypatch)
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/responses")

    def _from_request(exc: BaseException):
        async def _raise(*_args: Any, **_kwargs: Any) -> None:
            raise exc

        monkeypatch.setattr(pipe, "_process_transformed_request", _raise)

    if branch == "network timeout":
        pipe.valves.NETWORK_TIMEOUT_TEMPLATE = f"### {variant}\n\nWaited too long for {{endpoint}}."
        _from_request(httpx.TimeoutException("slow"))
    elif branch == "connection refused":
        pipe.valves.CONNECTION_ERROR_TEMPLATE = f"### {variant}\n\nCould not open {{endpoint}}."
        _from_request(httpx.ConnectError("refused"))
    elif branch == "service error":
        pipe.valves.SERVICE_ERROR_TEMPLATE = f"### {variant}\n\nHTTP {{status_code}}."
        _from_request(
            httpx.HTTPStatusError("boom", request=request, response=httpx.Response(503, request=request))
        )
    elif branch == "openrouter rejection":
        pipe.valves.INSUFFICIENT_CREDITS_TEMPLATE = "### Out of credits\n\n{openrouter_message}"
        _from_request(
            OpenRouterAPIError(status=402, reason="rejected", openrouter_message=variant)
        )
    elif branch == "unexpected":
        pipe.valves.INTERNAL_ERROR_TEMPLATE = f"### {variant}\n\nA {{error_type}} ended it."
        _from_request(ValueError("unexpected"))
    elif branch == "required file":
        _from_request(RequiredInternalFileError(variant, denied=True))
    elif branch == "catalog misconfigured":

        async def _bad_config(*_args: Any, **_kwargs: Any) -> None:
            raise ValueError(variant)

        monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", _bad_config)
    elif branch == "catalog unusable":

        async def _down(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(variant)

        monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", _down)
        monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "list_models", lambda: [])
        return "catalog"
    else:  # pragma: no cover - a branch name with no arrangement is a test bug
        raise AssertionError(f"no arrangement for branch {branch!r}")
    return variant


_EMITTERLESS_BRANCHES = [
    "network timeout",
    "connection refused",
    "service error",
    "openrouter rejection",
    "unexpected",
    "required file",
    "catalog misconfigured",
    "catalog unusable",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", _EMITTERLESS_BRANCHES)
@pytest.mark.parametrize("variant", ["Herring", "Mackerel"], ids=["first", "second"])
async def test_every_failure_branch_answers_a_caller_with_no_emitter(
    monkeypatch, pipe_instance_async, branch, variant
) -> None:
    """What the pipe RETURNS with nobody to show it to is what it would have shown.

    Three of these branches route through ``_emit_error``, whose return value was assigned
    inside ``if show_error_message and event_emitter:`` -- so with no emitter they answered an
    API caller with the empty string while their siblings rendered a card. The previous
    coverage tested the two siblings BY NAME, which is exactly how the third slipped; this
    walks the branches of ``_handle_pipe_call`` instead, so a new branch that forgets to
    return is a failure here rather than an untested path.

    Each branch is driven twice under one arrangement: once with an emitter and once without.
    Asserting the two surfaces are equal rather than asserting against a literal is what makes
    ``return "<constant>"`` in production fail -- the emitter still receives the real card.
    Two variants per branch carry two distinct strings through the render.
    """
    pipe = pipe_instance_async
    words = _arrange_failure(pipe, monkeypatch, branch, variant)

    events: list[dict[str, Any]] = []
    with_emitter = await _call(pipe, events)
    without_emitter = await _call_without_emitter(pipe)

    assert isinstance(without_emitter, str) and without_emitter.strip(), (
        f"the {branch!r} branch answered a caller with no chat context with "
        f"{without_emitter!r}; empty content is not an answer"
    )
    assert without_emitter == with_emitter, (
        f"the {branch!r} branch returns something different when nobody is watching: "
        f"{without_emitter!r} vs {with_emitter!r}"
    )
    assert words in without_emitter, (
        f"the {branch!r} branch did not carry its own failure through: {without_emitter!r}"
    )
    shown = [
        event["data"]["content"]
        for event in events
        if isinstance(event, dict)
        and event.get("type") == "chat:message"
        and isinstance(event.get("data"), dict)
        and isinstance(event["data"].get("content"), str)
    ] or [
        event["data"]["error"]["message"]
        for event in events
        if isinstance(event, dict)
        and event.get("type") == "chat:completion"
        and isinstance(event.get("data"), dict)
        and isinstance(event["data"].get("error"), dict)
    ]
    assert shown and shown[-1] == without_emitter, (
        "the returned text has to be the text the emitter was given, not a second rendering: "
        f"{shown!r} vs {without_emitter!r}"
    )
