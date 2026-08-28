# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportGeneralTypeIssues=false, reportCallIssue=false
"""The /chat/completions adapter must report generation order to the persist side.

Production 400 (five providers in succession, byte-identical text):

    messages.1.content.1: `thinking` or `redacted_thinking` blocks in the latest
    assistant message cannot be modified.

The call-ordinal anchor (commit 99788bc) is derived in `_run_streaming_loop` by
matching reasoning ids against `response.completed.output`, falling back to the
STREAM POSITION of the reasoning `response.output_item.done` relative to the
`function_call` ones. The chat-completions adapter folds reasoning into
`message.reasoning_details`, so no reasoning item is ever in `output` and the
fallback always runs -- and the adapter used to flush every `function_call`
`.done` BEFORE the reasoning `.done`. Reasoning the model generated *before* its
tool call was therefore stamped `_anchor_preceding_call_ordinal` (replay it after
the tool result) instead of `_anchor_following_call_ordinal` (replay it before
the call), which is exactly the block-reordering Anthropic rejects.

These tests drive the REAL adapter over REAL SSE bytes through the REAL streaming
loop; unit-testing `_reinterleave_region` cannot see this (its inputs are the
already-stamped anchors, and those unit tests passed while the bug was live).
"""
from __future__ import annotations

import json
from typing import Any, cast

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import Pipe, ResponsesBody, generate_item_id, _serialize_marker
from open_webui_openrouter_pipe.core.utils import (
    REASONING_FOLLOWING_ORDINAL_KEY,
    REASONING_PRECEDING_ORDINAL_KEY,
)
from open_webui_openrouter_pipe.requests.transformer import transform_messages_to_input

MODEL = "anthropic/claude-opus-4.8"


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _chat_round_with_call(text: str, call_id: str) -> str:
    """One /chat/completions SSE response: reasoning FIRST, then a tool call."""
    return (
        _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 0, "text": text,
             "signature": f"sig-{text}", "format": "anthropic-claude-v1"},
        ]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": call_id, "type": "function",
             "function": {"name": "f", "arguments": "{}"}},
        ]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        + "data: [DONE]\n\n"
    )


def _chat_final_round(text: str) -> str:
    return (
        _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 0, "text": text,
             "signature": f"sig-{text}", "format": "anthropic-claude-v1"},
        ]}, "finish_reason": None}]})
        + _sse({"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]})
        + "data: [DONE]\n\n"
    )


def _responses_round_with_call(text: str, call_id: str) -> list[dict[str, Any]]:
    """The same generation order as `_chat_round_with_call`, in /responses events."""
    block = {"type": "reasoning", "id": f"rs-{text}", "signature": f"sig-{text}",
             "content": [{"type": "reasoning_text", "text": text}]}
    call = {"type": "function_call", "call_id": call_id, "name": "f", "arguments": "{}"}
    return [
        {"type": "response.output_item.done", "item": {**block, "status": "completed", "summary": []}},
        {"type": "response.output_item.done", "item": {**call, "status": "completed"}},
        {"type": "response.completed", "response": {"output": [block, call], "usage": {}}},
    ]


def _responses_final_round(text: str) -> list[dict[str, Any]]:
    block = {"type": "reasoning", "id": f"rs-{text}", "signature": f"sig-{text}",
             "content": [{"type": "reasoning_text", "text": text}]}
    return [
        {"type": "response.output_item.done", "item": {**block, "status": "completed", "summary": []}},
        {"type": "response.output_text.delta", "delta": "done"},
        {"type": "response.completed", "response": {"output": [block], "usage": {}}},
    ]


def _valves(pipe: Pipe, endpoint: str, n_rounds: int):
    return pipe.valves.model_copy(update={
        "DEFAULT_LLM_ENDPOINT": endpoint,
        "TOOL_EXECUTION_MODE": "Pipeline",
        "MAX_FUNCTION_CALL_LOOPS": n_rounds + 2,
        "PERSIST_REASONING_TOKENS": "conversation",
        "PERSIST_TOOL_RESULTS": False,
        "API_KEY": "sk-test",
    })


def _install_capture(pipe: Pipe, monkeypatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    async def mock_exec(calls, registry):
        return [{"type": "function_call_output", "call_id": c.get("call_id"), "output": "out"}
                for c in calls]

    def fake_row(chat_id, message_id, model_id, payload):
        captured.append(payload)
        return {"payload": payload, "item_type": payload.get("type")}

    async def fake_persist(rows):
        return [generate_item_id() for _ in rows]

    monkeypatch.setattr(pipe._ensure_tool_executor(), "_execute_function_calls", mock_exec)
    monkeypatch.setattr(pipe._artifact_store, "_make_db_row", fake_row)
    monkeypatch.setattr(pipe._artifact_store, "_db_persist", fake_persist)
    return captured


def _call_ids(n_rounds: int) -> list[str]:
    return [f"toolu-{i}" for i in range(n_rounds)]


async def _persist_via_chat(pipe: Pipe, monkeypatch, n_rounds: int) -> list[dict[str, Any]]:
    """Drive the REAL chat-completions adapter over REAL SSE bytes."""
    valves = _valves(pipe, "chat_completions", n_rounds)
    captured = _install_capture(pipe, monkeypatch)
    bodies = [_chat_round_with_call(f"PRE{i}", cid) for i, cid in enumerate(_call_ids(n_rounds))]
    bodies.append(_chat_final_round("FINAL"))
    session = pipe._create_http_session(valves)
    try:
        with aioresponses() as mock_http:
            for sse_body in bodies:
                mock_http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    body=sse_body.encode("utf-8"),
                    headers={"Content-Type": "text/event-stream"},
                    status=200,
                )
            await pipe._streaming_handler._run_streaming_loop(
                ResponsesBody(model=MODEL, input=[], stream=True), valves, None,
                metadata={"model": {"id": MODEL}, "chat_id": "c1", "message_id": "m1"},
                tools={"f": {"callable": lambda **_k: "ok"}},
                session=cast(Any, session), user_id="u1",
            )
    finally:
        await session.close()
    return [p for p in captured if p.get("type") == "reasoning"]


async def _persist_via_responses(pipe: Pipe, monkeypatch, n_rounds: int) -> list[dict[str, Any]]:
    """Reference transport: the same generation order as native /responses events."""
    valves = _valves(pipe, "responses", n_rounds)
    captured = _install_capture(pipe, monkeypatch)
    rounds = [_responses_round_with_call(f"PRE{i}", cid) for i, cid in enumerate(_call_ids(n_rounds))]
    rounds.append(_responses_final_round("FINAL"))
    seen = [0]

    async def streaming(self, session, request_body, **_k):
        events = rounds[min(seen[0], len(rounds) - 1)]
        seen[0] += 1
        for event in events:
            yield event

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", streaming)
    await pipe._streaming_handler._run_streaming_loop(
        ResponsesBody(model=MODEL, input=[], stream=True), valves, None,
        metadata={"model": {"id": MODEL}, "chat_id": "c1", "message_id": "m1"},
        tools={"f": {"callable": lambda **_k: "ok"}},
        session=cast(Any, object()), user_id="u1",
    )
    return [p for p in captured if p.get("type") == "reasoning"]


def _anchors(payloads: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """(mode, ordinal) per persisted reasoning item, in persist order."""
    out: list[tuple[str, int]] = []
    for p in payloads:
        if REASONING_FOLLOWING_ORDINAL_KEY in p:
            out.append(("following", p[REASONING_FOLLOWING_ORDINAL_KEY]))
        elif REASONING_PRECEDING_ORDINAL_KEY in p:
            out.append(("preceding", p[REASONING_PRECEDING_ORDINAL_KEY]))
        else:
            out.append(("none", -1))
    return out


# --------------------------------------------------------------------------- #
# 1. The adapter's own event order IS the contract the persist side reads.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("n_calls", [1, 2, 3])
async def test_chat_adapter_emits_reasoning_done_before_every_function_call_done(
    pipe_instance_async, n_calls
):
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(update={"API_KEY": "sk-test"})
    sse_body = (
        _sse({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "index": 0, "text": "PRE", "signature": "sig"},
        ]}, "finish_reason": None}]})
        + "".join(
            _sse({"choices": [{"delta": {"tool_calls": [
                {"index": i, "id": f"toolu-{i}", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}},
            ]}, "finish_reason": None}]})
            for i in range(n_calls)
        )
        + _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        + "data: [DONE]\n\n"
    )
    session = pipe._create_http_session(valves)
    done_types: list[str] = []
    try:
        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                body=sse_body.encode("utf-8"),
                headers={"Content-Type": "text/event-stream"},
                status=200,
            )
            async for event in pipe.send_openai_chat_completions_streaming_request(
                session, {"model": MODEL, "stream": True, "input": []},
                api_key="sk-test", base_url="https://openrouter.ai/api/v1", valves=valves,
            ):
                if event.get("type") == "response.output_item.done":
                    done_types.append(event["item"].get("type"))
    finally:
        await session.close()

    assert done_types.count("function_call") == n_calls, done_types
    assert done_types.count("reasoning") == 1, done_types
    assert done_types.index("reasoning") < done_types.index("function_call"), (
        f"reasoning generated before the tool calls must be flushed first: {done_types}"
    )


# --------------------------------------------------------------------------- #
# 2. Pre-call reasoning is anchored to the call that FOLLOWS it.
#    Parametrised over round count so the expected ordinals differ per case
#    ([0] vs [0,1] vs [0,1,2]); no constant return value satisfies all three.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("n_rounds", [1, 2, 3])
async def test_chat_adapter_anchors_pre_call_reasoning_to_the_following_call(
    pipe_instance_async, monkeypatch, n_rounds
):
    payloads = await _persist_via_chat(pipe_instance_async, monkeypatch, n_rounds)
    assert len(payloads) == n_rounds + 1, _anchors(payloads)
    expected = [("following", i) for i in range(n_rounds)] + [("preceding", n_rounds - 1)]
    assert _anchors(payloads) == expected


# --------------------------------------------------------------------------- #
# 3. Transport equivalence: the same generation order must anchor identically
#    whichever endpoint carried it.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("n_rounds", [1, 2, 3])
async def test_chat_and_responses_transports_agree_on_anchors(
    pipe_instance_async, monkeypatch, n_rounds
):
    via_chat = _anchors(await _persist_via_chat(pipe_instance_async, monkeypatch, n_rounds))
    via_responses = _anchors(await _persist_via_responses(pipe_instance_async, monkeypatch, n_rounds))
    assert via_responses == [("following", i) for i in range(n_rounds)] + [("preceding", n_rounds - 1)]
    assert via_chat == via_responses


# --------------------------------------------------------------------------- #
# 4. Round trip: replaying a chat-completions turn must not move a thinking
#    block across the tool boundary (the shape the provider 400s on).
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("n_rounds", [1, 2, 3])
async def test_chat_persisted_reasoning_replays_before_its_own_call(
    pipe_instance_async, monkeypatch, n_rounds
):
    pipe = pipe_instance_async
    payloads = await _persist_via_chat(pipe, monkeypatch, n_rounds)
    call_ids = _call_ids(n_rounds)
    markers = [generate_item_id() for _ in payloads]
    artifacts = dict(zip(markers, payloads))

    messages: list[dict[str, Any]] = [{"role": "user", "content": "q1"}]
    for rnd, cid in enumerate(call_ids, 1):
        messages.append({"role": "assistant", "content": f"t{rnd}", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "f", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": cid, "content": "out"})
    messages.append({"role": "assistant", "message_id": "final",
                     "content": "done\n\n" + "\n\n".join(_serialize_marker(m) for m in markers)})
    messages.append({"role": "user", "content": "q2"})

    async def loader(_chat_id, _message_id, ulids):
        return {u: artifacts[u] for u in ulids if u in artifacts}

    result = await transform_messages_to_input(
        pipe, messages, chat_id="c1", openwebui_model_id="owui",
        artifact_loader=loader, model_id=MODEL,
        valves=_valves(pipe, "chat_completions", n_rounds),
    )
    types = [it.get("type") for it in result]
    texts = [
        (i, it["content"][0]["text"]) for i, it in enumerate(result)
        if it.get("type") == "reasoning" and it.get("content")
    ]
    fc_idx = [i for i, it in enumerate(result) if it.get("type") == "function_call"]

    assert len(texts) == n_rounds + 1, f"reasoning lost on replay: {types}"
    assert not any(b == a + 1 for a, b in zip(
        [i for i, t in enumerate(types) if t == "reasoning"],
        [i for i, t in enumerate(types) if t == "reasoning"][1:],
    )), f"400-trigger shape (consecutive reasoning): {types}"
    for ordinal in range(n_rounds):
        pos = next(i for i, txt in texts if txt == f"PRE{ordinal}")
        assert pos < fc_idx[ordinal], (
            f"PRE{ordinal} replayed after its own call {ordinal}: {types}"
        )
    assert next(i for i, txt in texts if txt == "FINAL") > fc_idx[-1], types
    assert not any(k.startswith("_anchor") for it in result for k in it), types
