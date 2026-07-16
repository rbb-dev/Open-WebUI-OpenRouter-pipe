"""Issue #48 -- Anthropic thinking-signature handling across both endpoints.

Covers the coordinated fixes:
  C1/C2 -- the streaming /chat reverse adapter keys reasoning_details by `index`
           (so one block's deltas consolidate and distinct blocks stay separate)
           and carries the final delta's signature.
  R1    -- /responses back-fills the signature from final_response.output onto the
           persisted reasoning row, in the one loop that BOTH streaming and
           non-streaming execute.
  S1    -- a thinking-signature 400 strips replayed reasoning from the input and
           retries (rather than bricking the conversation).
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportCallIssue=false, reportOptionalIterable=false
from __future__ import annotations

import json
from typing import Any, cast

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import Pipe, ResponsesBody
from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError
from open_webui_openrouter_pipe.api.transforms import _responses_payload_to_chat_completions_payload
from open_webui_openrouter_pipe.requests.sanitizer import (
    _sanitize_request_input,
    _strip_unreplayable_anthropic_reasoning,
)


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _make_fake_stream(events: list[dict[str, Any]]):
    async def fake_stream(self, session, request_body, **_kwargs):
        for event in events:
            yield event
    return fake_stream


def _completed_output(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for event in events:
        if event.get("type") == "response.completed":
            return event.get("response", {}).get("output", []) or []
    return []


def _message_reasoning_details(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for item in _completed_output(events):
        if item.get("type") == "message":
            return item.get("reasoning_details") or []
    return []


async def _drive_chat_stream(pipe: Pipe, sse_body: str) -> list[dict[str, Any]]:
    session = pipe._create_http_session(pipe.valves)
    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            body=sse_body.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status=200,
        )
        events: list[dict[str, Any]] = []
        async for event in pipe.send_openai_chat_completions_streaming_request(
            session,
            {"model": "anthropic/claude-sonnet-4.6", "stream": True, "input": []},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=pipe.valves,
        ):
            events.append(event)
        await session.close()
    return events


# --------------------------------------------------------------------------- #
# C1/C2 -- streaming chat reverse adapter: key on index, final signature wins
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_chat_idless_deltas_same_index_consolidate_with_final_signature(pipe_instance_async):
    """Three reasoning.text deltas of ONE block (id=null, index=0, signature on the
    last) consolidate into ONE entry with the full text and the final signature --
    not three signature-less fragments."""
    pipe = pipe_instance_async
    sse = (
        _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 0, "text": "Let me "}]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 0, "text": "think "}]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 0, "text": "carefully.",
             "signature": "SIG_FINAL", "format": "anthropic-claude-v1"}]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"content": "Answer"}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        + "data: [DONE]\n\n"
    )
    events = await _drive_chat_stream(pipe, sse)
    details = [d for d in _message_reasoning_details(events) if d.get("type") == "reasoning.text"]
    assert len(details) == 1, f"expected one consolidated block, got {details}"
    assert details[0]["text"] == "Let me think carefully."
    assert details[0]["signature"] == "SIG_FINAL"


@pytest.mark.asyncio
async def test_chat_distinct_index_blocks_not_merged(pipe_instance_async):
    """Two distinct thinking blocks (index 0 and 1, both id=null) stay SEPARATE with
    their own signatures -- never collapsed into one block under a positional key."""
    pipe = pipe_instance_async
    sse = (
        _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 0, "text": "Block zero", "signature": "SIG0"}]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 1, "text": "Block one", "signature": "SIG1"}]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"content": "Done"}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        + "data: [DONE]\n\n"
    )
    events = await _drive_chat_stream(pipe, sse)
    details = [d for d in _message_reasoning_details(events) if d.get("type") == "reasoning.text"]
    assert len(details) == 2, f"distinct blocks must not merge: {details}"
    by_text = {d["text"]: d.get("signature") for d in details}
    assert by_text == {"Block zero": "SIG0", "Block one": "SIG1"}


@pytest.mark.asyncio
async def test_chat_interleaved_distinct_index_blocks_each_consolidate(pipe_instance_async):
    """Two distinct blocks (index 0 and 1), EACH split across interleaved deltas
    (0,1,0,1), consolidate BY INDEX into two separate signed blocks. A positional key
    would split each block's fragments into four entries and detach each signature --
    so this distinguishes index-keying from positional."""
    pipe = pipe_instance_async
    sse = (
        _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 0, "text": "zero-a "}]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 1, "text": "one-a "}]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 0, "text": "zero-b", "signature": "SIG0"}]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 1, "text": "one-b", "signature": "SIG1"}]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"content": "Done"}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        + "data: [DONE]\n\n"
    )
    events = await _drive_chat_stream(pipe, sse)
    details = [d for d in _message_reasoning_details(events) if d.get("type") == "reasoning.text"]
    assert len(details) == 2, f"interleaved distinct blocks must stay two: {details}"
    by_text = {d["text"]: d.get("signature") for d in details}
    assert by_text == {"zero-a zero-b": "SIG0", "one-a one-b": "SIG1"}


@pytest.mark.asyncio
async def test_chat_stable_id_deltas_still_merge(pipe_instance_async):
    """Regression: when deltas carry a stable id (no index), they still merge by id."""
    pipe = pipe_instance_async
    sse = (
        _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "id": "r1", "text": "first "}]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "id": "r1", "text": "second", "signature": "SIG_R1"}]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"content": "Ok"}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        + "data: [DONE]\n\n"
    )
    events = await _drive_chat_stream(pipe, sse)
    details = [d for d in _message_reasoning_details(events) if d.get("type") == "reasoning.text"]
    assert len(details) == 1, f"same-id deltas must still merge: {details}"
    assert details[0]["text"] == "first second"
    assert details[0]["signature"] == "SIG_R1"


@pytest.mark.asyncio
async def test_fragmented_reasoning_round_trips_as_one_signed_block_on_chat(pipe_instance_async):
    """Round-trip: fragmented reasoning.text deltas consolidate into ONE signed block on
    emission, and replaying that block as a prior assistant turn serializes to the
    /chat/completions wire as ONE signed reasoning.text detail -- not re-fragmented, signature
    intact. Guards the Channel-B chat replay that Anthropic verifies across both halves."""
    pipe = pipe_instance_async
    sse = (
        _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 0, "text": "Let me "}]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 0, "text": "think "}]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 0, "text": "carefully.",
             "signature": "SIG_FINAL", "format": "anthropic-claude-v1"}]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"content": "Answer"}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        + "data: [DONE]\n\n"
    )
    events = await _drive_chat_stream(pipe, sse)
    consolidated = [d for d in _message_reasoning_details(events) if d.get("type") == "reasoning.text"]
    assert len(consolidated) == 1, f"turn 1 must consolidate fragments into one block: {consolidated}"
    assert consolidated[0]["text"] == "Let me think carefully."
    assert consolidated[0]["signature"] == "SIG_FINAL"

    payload = {
        "model": "anthropic/claude-sonnet-4.6",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "Answer"}],
             "reasoning_details": [dict(consolidated[0])]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "again"}]},
        ],
    }
    chat = _responses_payload_to_chat_completions_payload(payload)
    assistant_msgs = [m for m in chat["messages"] if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1, f"expected one replayed assistant message: {chat['messages']}"
    replayed = assistant_msgs[0].get("reasoning_details")
    assert isinstance(replayed, list) and len(replayed) == 1, \
        f"replayed reasoning must stay ONE block, not re-fragment: {replayed}"
    assert replayed[0]["text"] == "Let me think carefully.", "full reasoning text must survive replay"
    assert replayed[0]["signature"] == "SIG_FINAL", "signature must survive replay to the chat wire"


# --------------------------------------------------------------------------- #
# R1 -- /responses back-fills the signature onto the persisted reasoning row
# --------------------------------------------------------------------------- #
def _responses_events_with_signed_final(rs_id: str, signature: str) -> list[dict[str, Any]]:
    """The per-item `output_item.done` reasoning has NO signature (as observed in the
    pipe's traffic); the assembled `response.completed` output carries it."""
    reasoning_item = {
        "type": "reasoning", "id": rs_id, "status": "completed",
        "content": [{"type": "reasoning_text", "text": "weighing options"}], "summary": [],
    }
    assistant_message = {"type": "message", "role": "assistant",
                         "content": [{"type": "output_text", "text": "Answer"}]}
    return [
        {"type": "response.output_item.done", "item": dict(reasoning_item)},
        {"type": "response.output_item.done", "item": dict(assistant_message)},
        {"type": "response.completed", "response": {"output": [
            dict(reasoning_item, signature=signature, format="anthropic-claude-v1"),
            dict(assistant_message),
        ], "usage": {}}},
    ]


def _capture_persisted_reasoning(pipe: Pipe, monkeypatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    original = pipe._artifact_store._make_db_row

    def capture(chat_id, message_id, model_id, payload):
        if isinstance(payload, dict) and payload.get("type") == "reasoning":
            captured.append(payload)  # same object the R1 back-fill mutates by reference
        return original(chat_id, message_id, model_id, payload)

    monkeypatch.setattr(pipe._artifact_store, "_make_db_row", capture)
    return captured


@pytest.mark.asyncio
async def test_responses_signature_backfilled_onto_persisted_reasoning(pipe_instance_async, monkeypatch):
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(update={"PERSIST_REASONING_TOKENS": "conversation"})
    captured = _capture_persisted_reasoning(pipe, monkeypatch)
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request",
                        _make_fake_stream(_responses_events_with_signed_final("rs_abc", "SIG_BACKFILL")))
    body = ResponsesBody(model="anthropic/claude-sonnet-4.6", input=[], stream=True)

    await pipe._streaming_handler._run_streaming_loop(
        body, valves, None, metadata={"chat_id": "chat-1", "message_id": "msg-1"},
        tools={}, session=cast(Any, object()), user_id="u1",
    )

    assert captured, "no reasoning artifact was persisted"
    assert captured[0].get("signature") == "SIG_BACKFILL", \
        f"signature not back-filled onto persisted row: keys={sorted(captured[0])}"


@pytest.mark.asyncio
async def test_nonstreaming_loop_delegates_to_streaming_loop(pipe_instance_async, monkeypatch):
    """Non-streaming reuses the streaming loop verbatim, so R1's back-fill (and R2's
    gate) apply to both modes without duplication."""
    pipe = pipe_instance_async
    captured: dict[str, Any] = {}

    async def fake_streaming_loop(body, *args, **kwargs):
        captured["delegated"] = True
        captured["stream_flag"] = body.stream
        return "DELEGATED_RESULT"

    monkeypatch.setattr(pipe._streaming_handler, "_run_streaming_loop", fake_streaming_loop)
    body = ResponsesBody(model="anthropic/claude-sonnet-4.6", input=[], stream=False)

    result = await pipe._streaming_handler._run_nonstreaming_loop(
        body, pipe.valves, None, metadata={}, tools={},
        session=cast(Any, object()), user_id="u1",
    )

    assert result == "DELEGATED_RESULT"
    assert captured.get("delegated") is True, "_run_nonstreaming_loop must delegate to _run_streaming_loop"


# --------------------------------------------------------------------------- #
# S1 -- a thinking-signature 400 strips replayed reasoning and retries
# --------------------------------------------------------------------------- #
def _signature_error() -> OpenRouterAPIError:
    return OpenRouterAPIError(
        status=400, reason="Bad Request",
        upstream_message="messages.1.content.0: Invalid `signature` in `thinking` block",
    )


def test_signature_error_strips_reasoning_items_and_details_then_retries(pipe_instance):
    mgr = pipe_instance._ensure_reasoning_config_manager()
    body = ResponsesBody(model="anthropic/claude-sonnet-4.6", input=[
        {"type": "reasoning", "id": "r1",
         "content": [{"type": "reasoning_text", "text": "x"}], "signature": "S"},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "hi"}],
         "reasoning_details": [{"type": "reasoning.text", "text": "y", "signature": "S2"}]},
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "next"}]},
    ])

    assert mgr._should_retry_dropping_signed_reasoning(_signature_error(), body) is True
    types = [it.get("type") for it in body.input]
    assert "reasoning" not in types, f"reasoning item not stripped: {types}"
    assert all("reasoning_details" not in it for it in body.input), "reasoning_details not stripped"
    assert any(it.get("role") == "user" for it in body.input), "user message must survive the strip"


def test_non_signature_error_does_not_strip_or_retry(pipe_instance):
    mgr = pipe_instance._ensure_reasoning_config_manager()
    body = ResponsesBody(model="anthropic/claude-sonnet-4.6", input=[
        {"type": "reasoning", "id": "r1", "content": [], "signature": "S"},
    ])
    err = OpenRouterAPIError(status=429, reason="Too Many Requests", upstream_message="rate limited")

    assert mgr._should_retry_dropping_signed_reasoning(err, body) is False
    assert any(it.get("type") == "reasoning" for it in body.input), "must not strip on an unrelated error"


def test_signature_error_with_nothing_to_strip_does_not_retry(pipe_instance):
    mgr = pipe_instance._ensure_reasoning_config_manager()
    body = ResponsesBody(model="anthropic/claude-sonnet-4.6", input=[
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
    ])

    assert mgr._should_retry_dropping_signed_reasoning(_signature_error(), body) is False


def test_signature_error_non_anthropic_model_does_not_strip(pipe_instance):
    """D3 gate: a signature-shaped 400 on a NON-Anthropic model must not trigger the
    reactive strip/retry -- the backstop is Anthropic-only."""
    mgr = pipe_instance._ensure_reasoning_config_manager()
    body = ResponsesBody(model="google/gemini-2.5-pro", api_model="google/gemini-2.5-pro", input=[
        {"type": "reasoning", "id": "r1", "content": [], "signature": "S"},
    ])
    assert mgr._should_retry_dropping_signed_reasoning(_signature_error(), body) is False
    assert any(it.get("type") == "reasoning" for it in body.input), "non-Anthropic input must be untouched"


def test_signature_error_tilde_anthropic_alias_strips_and_retries(pipe_instance):
    """~anthropic router aliases take the S1 reactive strip path like plain slugs (issue #53)."""
    mgr = pipe_instance._ensure_reasoning_config_manager()
    body = ResponsesBody(model="~anthropic/claude-fable-latest", input=[
        {"type": "reasoning", "id": "r1",
         "content": [{"type": "reasoning_text", "text": "x"}], "signature": "S"},
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "next"}]},
    ])
    assert mgr._should_retry_dropping_signed_reasoning(_signature_error(), body) is True
    assert all(it.get("type") != "reasoning" for it in body.input), "reasoning must be stripped for the retry"


def test_proactive_strip_applies_to_tilde_anthropic_alias(pipe_instance):
    """The pre-send sanitizer treats ~anthropic aliases as Anthropic: unsigned
    replayed reasoning is stripped before it can 400 upstream (issue #53)."""
    from open_webui_openrouter_pipe.requests.sanitizer import _sanitize_request_input

    body = ResponsesBody(model="~anthropic/claude-fable-latest", input=[
        {"type": "reasoning", "id": "r1",
         "content": [{"type": "reasoning_text", "text": "x"}]},
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "hi"}]},
    ])
    _sanitize_request_input(pipe_instance, body)
    assert all(it.get("type") != "reasoning" for it in body.input), "unsigned reasoning must be stripped for ~anthropic"


# --------------------------------------------------------------------------- #
# Proactive strip (the streaming heal). The sanitizer removes unreplayable
# thinking BEFORE every send, so a legacy conversation never 400s. S1's reactive
# retry is the backstop for a present-but-rejected signature -- now reachable on
# streaming via the pre-emission re-raise.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_proactive_strip_removes_unsigned_reasoning_on_streaming(pipe_instance_async, monkeypatch):
    """On streaming, the sanitizer strips unsigned/unencrypted reasoning before the
    send, so a legacy conversation never 400s (the path S1 can't reach). The signed
    block lives in a SEPARATE turn (after a user message) so it is kept."""
    pipe = pipe_instance_async
    captured: list[Any] = []

    async def capturing_stream(self, session, request_body, **_kwargs):
        captured.append(request_body)
        yield {"type": "response.completed", "response": {"output": [], "usage": {}}}

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", capturing_stream)
    body = ResponsesBody(
        model="anthropic/claude-sonnet-4.6", api_model="anthropic/claude-sonnet-4.6", stream=True,
        input=[
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "reasoning", "id": "rs_unsigned", "status": "completed",
             "content": [{"type": "reasoning_text", "text": "stale legacy thinking"}], "summary": []},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "more"}]},
            {"type": "reasoning", "id": "rs_signed", "status": "completed",
             "content": [{"type": "reasoning_text", "text": "valid"}], "summary": [], "signature": "SIG"},
        ],
    )
    await pipe._streaming_handler._run_streaming_loop(
        body, pipe.valves, None, metadata={}, tools={}, session=cast(Any, object()), user_id="u1",
    )
    assert captured, "request was never sent"
    sent = captured[0]
    sent_input = sent.get("input") if isinstance(sent, dict) else getattr(sent, "input", None)
    ids = [it.get("id") for it in (sent_input or []) if isinstance(it, dict) and it.get("type") == "reasoning"]
    assert "rs_unsigned" not in ids, f"unsigned reasoning not stripped before send: {ids}"
    assert "rs_signed" in ids, f"signed reasoning wrongly stripped: {ids}"


@pytest.mark.asyncio
async def test_proactive_strip_drops_whole_interleaved_turn_on_streaming(pipe_instance_async, monkeypatch):
    """End-to-end on the real send path: an interleaved turn whose reasoning is split by
    tool items, with one block unsigned, has its ENTIRE reasoning sequence stripped before
    the send. A per-item strip would keep the signed pre-tool block and ship a partial
    turn-sequence that still 400s."""
    pipe = pipe_instance_async
    captured: list[Any] = []

    async def capturing_stream(self, session, request_body, **_kwargs):
        captured.append(request_body)
        yield {"type": "response.completed", "response": {"output": [], "usage": {}}}

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", capturing_stream)
    body = ResponsesBody(
        model="anthropic/claude-sonnet-4.6", api_model="anthropic/claude-sonnet-4.6", stream=True,
        input=[
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "reasoning", "id": "rs_signed", "status": "completed",
             "content": [{"type": "reasoning_text", "text": "pre-tool"}], "summary": [], "signature": "SIG"},
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            {"type": "reasoning", "id": "rs_unsigned", "status": "completed",
             "content": [{"type": "reasoning_text", "text": "post-tool stale"}], "summary": []},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
        ],
    )
    await pipe._streaming_handler._run_streaming_loop(
        body, pipe.valves, None, metadata={}, tools={}, session=cast(Any, object()), user_id="u1",
    )
    assert captured, "request was never sent"
    sent = captured[0]
    sent_input = sent.get("input") if isinstance(sent, dict) else getattr(sent, "input", None)
    ids = [it.get("id") for it in (sent_input or []) if isinstance(it, dict) and it.get("type") == "reasoning"]
    assert ids == [], f"the whole interleaved turn's reasoning must be stripped, not just the unsigned block: {ids}"


def test_strip_items_drops_whole_consecutive_run_when_any_unsigned():
    """Block-aware: a maximal run of consecutive /responses reasoning items is dropped
    whole when ANY item is unreplayable -- a partial run is a modified sequence the
    provider rejects (reasoning-tokens.md: 'you cannot rearrange or modify the sequence')."""
    items = [
        {"type": "message", "role": "user", "content": []},
        {"type": "reasoning", "id": "u", "content": [{"type": "reasoning_text", "text": "x"}], "summary": []},
        {"type": "reasoning", "id": "s", "content": [{"type": "reasoning_text", "text": "y"}], "summary": [], "signature": "S"},
        {"type": "reasoning", "id": "e", "content": [], "summary": [], "encrypted_content": "blob"},
        {"type": "reasoning", "id": "sum", "content": [], "summary": [{"type": "summary_text", "text": "z"}]},
    ]
    out = _strip_unreplayable_anthropic_reasoning(items)
    kept = [it.get("id") for it in out if it.get("type") == "reasoning"]
    assert kept == [], f"the whole run must drop when any item is unsigned: {kept}"
    assert any(it.get("type") == "message" for it in out), "the surrounding message must survive"


def test_strip_items_keeps_run_with_no_unsigned():
    """A consecutive run with no unsigned reasoning.text is preserved intact -- signed,
    encrypted, and summary items are all replayable."""
    items = [
        {"type": "message", "role": "user", "content": []},
        {"type": "reasoning", "id": "s", "content": [{"type": "reasoning_text", "text": "y"}], "summary": [], "signature": "S"},
        {"type": "reasoning", "id": "e", "content": [], "summary": [], "encrypted_content": "blob"},
        {"type": "reasoning", "id": "sum", "content": [], "summary": [{"type": "summary_text", "text": "z"}]},
    ]
    out = _strip_unreplayable_anthropic_reasoning(items)
    kept = [it.get("id") for it in out if it.get("type") == "reasoning"]
    assert kept == ["s", "e", "sum"], kept


def test_strip_items_independent_runs_isolated_by_non_reasoning():
    """Reasoning runs separated by a non-reasoning item are independent: an unsigned run
    is dropped without affecting a signed run elsewhere in the input."""
    items = [
        {"type": "reasoning", "id": "u1", "content": [{"type": "reasoning_text", "text": "x"}], "summary": []},
        {"type": "message", "role": "user", "content": []},
        {"type": "reasoning", "id": "s2", "content": [{"type": "reasoning_text", "text": "y"}], "summary": [], "signature": "S"},
    ]
    out = _strip_unreplayable_anthropic_reasoning(items)
    kept = [it.get("id") for it in out if it.get("type") == "reasoning"]
    assert kept == ["s2"], f"only the unsigned run drops; the signed run survives: {kept}"


def test_strip_items_interleaved_turn_split_by_tools_drops_whole_turn():
    """Interleaved thinking: one turn's reasoning is split across function_call /
    function_call_output items. Tool items are transparent (not turn separators), so an
    unreplayable block taints the WHOLE turn -- dropping every reasoning block in it,
    never leaving a partial same-turn sequence that still 400s. Tool plumbing survives."""
    items = [
        {"type": "message", "role": "user", "content": []},
        {"type": "reasoning", "id": "r1", "content": [], "summary": [], "encrypted_content": "blob"},
        {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        {"type": "reasoning", "id": "r2", "content": [{"type": "reasoning_text", "text": "plain"}], "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
    ]
    out = _strip_unreplayable_anthropic_reasoning(items)
    kept = [it.get("id") for it in out if it.get("type") == "reasoning"]
    assert kept == [], f"a partial turn-sequence must not survive a tool split: {kept}"
    assert [it.get("type") for it in out] == [
        "message", "function_call", "function_call_output", "message",
    ], "tool plumbing must be preserved when the turn's reasoning is dropped"


def test_strip_items_interleaved_turn_all_replayable_kept():
    """An interleaved turn whose reasoning blocks are all replayable (signed + encrypted)
    is preserved intact across the tool split."""
    items = [
        {"type": "message", "role": "user", "content": []},
        {"type": "reasoning", "id": "r1", "content": [{"type": "reasoning_text", "text": "a"}], "summary": [], "signature": "S1"},
        {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        {"type": "reasoning", "id": "r2", "content": [], "summary": [], "encrypted_content": "blob"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
    ]
    out = _strip_unreplayable_anthropic_reasoning(items)
    kept = [it.get("id") for it in out if it.get("type") == "reasoning"]
    assert kept == ["r1", "r2"], f"a fully-replayable interleaved turn must be preserved: {kept}"


def test_strip_items_turn_split_by_assistant_text_chunk_drops_whole_turn():
    """The transformer emits one assistant turn's visible text as type:message,role:assistant
    items, which can sit BETWEEN that turn's reasoning blocks. Only USER messages are turn
    boundaries, so an unsigned block taints the whole turn -- the signed block on the other
    side of the assistant text chunk is dropped too, never leaving a partial sequence."""
    items = [
        {"type": "message", "role": "user", "content": []},
        {"type": "reasoning", "id": "r1", "content": [{"type": "reasoning_text", "text": "a"}], "summary": [], "signature": "S1"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "partial"}]},
        {"type": "reasoning", "id": "r2", "content": [{"type": "reasoning_text", "text": "b"}], "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "final"}]},
    ]
    out = _strip_unreplayable_anthropic_reasoning(items)
    kept = [it.get("id") for it in out if it.get("type") == "reasoning"]
    assert kept == [], f"an assistant text chunk must NOT split a turn: {kept}"
    assert sum(1 for it in out if it.get("type") == "message") == 3, "all messages survive"


def test_strip_reasoning_details_drops_whole_array_when_any_unsigned():
    """Block-aware: when a message's reasoning_details mixes an unsigned reasoning.text
    with signed/encrypted/summary entries, the WHOLE reasoning_details is dropped. The
    provider requires the consecutive-block sequence intact and rejects a partially
    modified one (reasoning-tokens.md: 'you cannot rearrange or modify the sequence')."""
    items = [{"type": "message", "role": "assistant", "content": "hi", "reasoning_details": [
        {"type": "reasoning.text", "text": "unsigned"},
        {"type": "reasoning.text", "text": "signed", "signature": "S"},
        {"type": "reasoning.encrypted", "data": "blob"},
        {"type": "reasoning.summary", "summary": "s"},
    ]}]
    out = _strip_unreplayable_anthropic_reasoning(items)
    assert "reasoning_details" not in out[0], "a partial sequence must not survive"
    assert out[0]["content"] == "hi", "the assistant message itself must survive"


def test_strip_reasoning_details_kept_when_all_replayable():
    """No unsigned reasoning.text -> the sequence is preserved intact (signed text,
    encrypted, and summary are all replayable)."""
    details = [
        {"type": "reasoning.text", "text": "signed", "signature": "S"},
        {"type": "reasoning.encrypted", "data": "blob"},
        {"type": "reasoning.summary", "summary": "s"},
    ]
    items = [{"type": "message", "role": "assistant", "content": "hi",
              "reasoning_details": [dict(d) for d in details]}]
    out = _strip_unreplayable_anthropic_reasoning(items)
    assert out[0]["reasoning_details"] == details, "an all-replayable sequence must be preserved"


# --------------------------------------------------------------------------- #
# A4 -- signature / encrypted_content "present" means a non-blank STRING. A
# whitespace-only or non-string value is corrupt and unreplayable, so it counts
# as absent: the reasoning is stripped rather than replayed into a 400. Plain
# truthiness wrongly read such garbage as a valid signature and kept it.
# --------------------------------------------------------------------------- #
def test_strip_item_whitespace_signature_treated_as_unsigned():
    """A reasoning item whose signature is whitespace-only is unsigned, not signed:
    its turn is dropped. Truthiness would read '   ' as valid and replay into a 400."""
    items = [
        {"type": "message", "role": "user", "content": []},
        {"type": "reasoning", "id": "ws", "summary": [],
         "content": [{"type": "reasoning_text", "text": "x"}], "signature": "   "},
    ]
    out = _strip_unreplayable_anthropic_reasoning(items)
    kept = [it.get("id") for it in out if it.get("type") == "reasoning"]
    assert kept == [], f"whitespace signature must read as unsigned: {kept}"


def test_strip_item_whitespace_encrypted_content_treated_as_absent():
    """encrypted_content of whitespace is not a real payload: the item reads as
    unreplayable and is dropped (truthiness would wrongly keep it)."""
    items = [
        {"type": "message", "role": "user", "content": []},
        {"type": "reasoning", "id": "we", "summary": [],
         "content": [{"type": "reasoning_text", "text": "x"}], "encrypted_content": "  "},
    ]
    out = _strip_unreplayable_anthropic_reasoning(items)
    kept = [it.get("id") for it in out if it.get("type") == "reasoning"]
    assert kept == [], f"whitespace encrypted_content must read as absent: {kept}"


def test_strip_content_part_whitespace_signature_treated_as_unsigned():
    """The per-content-part check is string-strict too: a reasoning_text part whose
    signature is whitespace reads as unsigned, so the item is dropped. Isolates the
    part-level site, which truthiness left keeping a corrupt block."""
    items = [
        {"type": "message", "role": "user", "content": []},
        {"type": "reasoning", "id": "wp", "summary": [],
         "content": [{"type": "reasoning_text", "text": "thinking", "signature": "  "}]},
    ]
    out = _strip_unreplayable_anthropic_reasoning(items)
    kept = [it.get("id") for it in out if it.get("type") == "reasoning"]
    assert kept == [], f"whitespace part signature must read as unsigned: {kept}"


def test_strip_content_part_whitespace_encrypted_content_treated_as_absent():
    """The per-content-part encrypted_content check is string-strict too: a part whose
    encrypted_content is whitespace reads as absent, so the item is dropped. Isolates the
    part-level encrypted_content operand, which truthiness left keeping a corrupt block."""
    items = [
        {"type": "message", "role": "user", "content": []},
        {"type": "reasoning", "id": "wpe", "summary": [],
         "content": [{"type": "reasoning_text", "text": "thinking", "encrypted_content": "  "}]},
    ]
    out = _strip_unreplayable_anthropic_reasoning(items)
    kept = [it.get("id") for it in out if it.get("type") == "reasoning"]
    assert kept == [], f"whitespace part encrypted_content must read as absent: {kept}"


def test_strip_content_part_real_signature_kept():
    """Regression: a real (non-blank) signature on a content part keeps the item -- the
    tightening must not strip genuinely-signed reasoning."""
    items = [
        {"type": "message", "role": "user", "content": []},
        {"type": "reasoning", "id": "sp", "summary": [],
         "content": [{"type": "reasoning_text", "text": "thinking", "signature": "REALSIG"}]},
    ]
    out = _strip_unreplayable_anthropic_reasoning(items)
    kept = [it.get("id") for it in out if it.get("type") == "reasoning"]
    assert kept == ["sp"], f"a real part signature must be kept: {kept}"


def test_strip_detail_whitespace_signature_treated_as_unsigned():
    """reasoning_details: a reasoning.text whose signature is whitespace reads as unsigned,
    so the whole array is dropped all-or-nothing."""
    items = [{"type": "message", "role": "assistant", "content": "hi", "reasoning_details": [
        {"type": "reasoning.text", "text": "t", "signature": "   "}]}]
    out = _strip_unreplayable_anthropic_reasoning(items)
    assert "reasoning_details" not in out[0], "whitespace detail signature must read as unsigned"
    assert out[0]["content"] == "hi", "the assistant message itself must survive"


def test_strip_detail_nonstring_signature_treated_as_unsigned():
    """reasoning_details: a non-string signature (an int) is not a real signature; the
    detail reads as unsigned and the array is dropped."""
    items = [{"type": "message", "role": "assistant", "content": "hi", "reasoning_details": [
        {"type": "reasoning.text", "text": "t", "signature": 123}]}]
    out = _strip_unreplayable_anthropic_reasoning(items)
    assert "reasoning_details" not in out[0], "non-string detail signature must read as unsigned"


def test_strip_detail_empty_signature_still_unsigned_characterization():
    """Characterization (unchanged by A4): an empty-string signature was already falsy and
    stays unsigned, so the array is dropped. A4 only adds the whitespace/non-string cases."""
    items = [{"type": "message", "role": "assistant", "content": "hi", "reasoning_details": [
        {"type": "reasoning.text", "text": "t", "signature": ""}]}]
    out = _strip_unreplayable_anthropic_reasoning(items)
    assert "reasoning_details" not in out[0], "empty-string detail signature is unsigned (unchanged)"


def test_sanitize_does_not_strip_reasoning_for_non_anthropic(pipe_instance):
    body = ResponsesBody(model="google/gemini-2.5-pro", api_model="google/gemini-2.5-pro", input=[
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]},
        {"type": "reasoning", "id": "g", "content": [{"type": "reasoning_text", "text": "x"}], "summary": []},
    ])
    _sanitize_request_input(pipe_instance, body)
    assert any(it.get("type") == "reasoning" for it in body.input), "non-Anthropic reasoning must not be stripped"


def test_sanitize_strips_unsigned_reasoning_for_anthropic(pipe_instance):
    body = ResponsesBody(model="anthropic/claude-sonnet-4.6", api_model="anthropic/claude-sonnet-4.6", input=[
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]},
        {"type": "reasoning", "id": "u", "content": [{"type": "reasoning_text", "text": "x"}], "summary": []},
    ])
    _sanitize_request_input(pipe_instance, body)
    assert not any(it.get("type") == "reasoning" for it in body.input), "unsigned Anthropic reasoning must be stripped"


@pytest.mark.asyncio
async def test_responses_format_backfilled_completed_wins(pipe_instance_async, monkeypatch):
    """R1 also back-fills `format`, and the completed-output value wins over a stale
    streamed one."""
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(update={"PERSIST_REASONING_TOKENS": "conversation"})
    captured = _capture_persisted_reasoning(pipe, monkeypatch)
    rs_id = "rs_fmt"
    reasoning_item = {"type": "reasoning", "id": rs_id, "status": "completed",
                      "content": [{"type": "reasoning_text", "text": "t"}], "summary": []}
    msg = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "a"}]}
    events = [
        {"type": "response.output_item.done", "item": dict(reasoning_item, format="stale-format")},
        {"type": "response.output_item.done", "item": dict(msg)},
        {"type": "response.completed", "response": {"output": [
            dict(reasoning_item, signature="SIG", format="anthropic-claude-v1"), dict(msg),
        ], "usage": {}}},
    ]
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _make_fake_stream(events))
    body = ResponsesBody(model="anthropic/claude-sonnet-4.6", input=[], stream=True)
    await pipe._streaming_handler._run_streaming_loop(
        body, valves, None, metadata={"chat_id": "c", "message_id": "m"},
        tools={}, session=cast(Any, object()), user_id="u",
    )
    assert captured
    assert captured[0].get("signature") == "SIG"
    assert captured[0].get("format") == "anthropic-claude-v1", \
        f"completed-output format must win: {captured[0].get('format')}"


@pytest.mark.asyncio
async def test_chat_empty_signature_dropped(pipe_instance_async):
    """A delta carrying signature='' must not leak an empty signature onto the final
    consolidated reasoning detail."""
    pipe = pipe_instance_async
    sse = (
        _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 0, "text": "think", "signature": ""}]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"content": "ans"}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        + "data: [DONE]\n\n"
    )
    events = await _drive_chat_stream(pipe, sse)
    details = [d for d in _message_reasoning_details(events) if d.get("type") == "reasoning.text"]
    assert len(details) == 1
    assert "signature" not in details[0], f"empty signature leaked: {details[0]}"


# --------------------------------------------------------------------------- #
# Streaming retry barrier. A pre-emission error re-raises (so the orchestrator can
# retry); once content has streamed, the same error is reported, never re-raised
# (no re-run / duplicated output).
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_streaming_error_after_content_is_reported_not_reraised(pipe_instance_async, monkeypatch):
    pipe = pipe_instance_async
    err = OpenRouterAPIError(
        status=400, reason="Bad Request",
        upstream_message="messages.1.content.0: Invalid `signature` in `thinking` block",
    )

    async def fake_stream(self, session, request_body, **_kwargs):
        yield {"type": "response.output_text.delta", "delta": "partial answer"}
        raise err

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", fake_stream)
    body = ResponsesBody(model="anthropic/claude-sonnet-4.6", input=[], stream=True)
    emitted: list[Any] = []

    async def emitter(event):
        emitted.append(event)

    result = await pipe._streaming_handler._run_streaming_loop(
        body, pipe.valves, emitter, metadata={}, tools={},
        session=cast(Any, object()), user_id="u",
    )
    # Post-content error: reported to the user, NOT re-raised (re-raising would let the
    # orchestrator re-run and duplicate the already-streamed content).
    assert isinstance(result, str)
    joined = "".join(
        e.get("data", {}).get("content") or ""
        for e in emitted if e.get("type") == "chat:message:delta"
    )
    assert joined == "partial answer", f"content must stream exactly once (no duplicate): {joined!r}"
    reported = [
        e for e in emitted
        if e.get("type") == "status"
        and "provider error" in str(e.get("data", {}).get("description", "")).lower()
    ]
    assert reported, f"the post-content error must be reported to the user: {emitted}"


@pytest.mark.asyncio
async def test_streaming_pre_emission_error_reraises(pipe_instance_async, monkeypatch):
    """A pre-emission OpenRouterAPIError (nothing streamed yet) re-raises so the
    orchestrator's retry loop can handle it -- without a spurious done=True completion."""
    pipe = pipe_instance_async
    err = OpenRouterAPIError(status=400, reason="Bad Request",
                             upstream_message="Invalid `signature` in `thinking` block")

    async def fake_stream(self, session, request_body, **_kwargs):
        raise err
        if False:  # pragma: no cover
            yield {}

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", fake_stream)
    body = ResponsesBody(model="anthropic/claude-sonnet-4.6", input=[], stream=True)
    emitted: list[Any] = []

    async def emitter(event):
        emitted.append(event)

    with pytest.raises(OpenRouterAPIError):
        await pipe._streaming_handler._run_streaming_loop(
            body, pipe.valves, emitter, metadata={}, tools={},
            session=cast(Any, object()), user_id="u",
        )
    done = [e for e in emitted if e.get("type") == "status" and e.get("data", {}).get("done") is True]
    assert not done, f"pre-emission re-raise must not emit a completion: {done}"
