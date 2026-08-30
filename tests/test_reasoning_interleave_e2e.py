# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportGeneralTypeIssues=false, reportCallIssue=false, reportUnusedFunction=false
"""End-to-end proof for the interleaved-reasoning ordering fix.

Drives the REAL pipe round trip:
  Stage A - real `_run_streaming_loop` generates a turn (one reasoning + a tool
            call per round, then a closing reasoning) and persists reasoning
            artifacts (call-ordinal anchors stamped).
  Stage B - the persisted artifacts + an OWUI-shaped turn-2 history are fed to the
            REAL `transform_messages_to_input`, which rebuilds the provider input.

Invariant (the thing that 400s in production): reasoning generated on opposite
sides of a tool round must never come out consecutive. Also: anchors never leak
to the wire, and reasoning is never silently lost. Anchors are call ORDINALS, so
the round-trip holds even when a provider reuses the same call_id every round.
"""
from __future__ import annotations

from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import Pipe, ResponsesBody, generate_item_id, _serialize_marker
from open_webui_openrouter_pipe.requests.transformer import transform_messages_to_input


# --------------------------------------------------------------------------- #
# Round-trip helpers
# --------------------------------------------------------------------------- #
async def _stage_a_persist(pipe, monkeypatch, valves, round_call_ids: list[str]) -> list[dict]:
    """Drive the real streaming loop: one (reasoning -> tool call) per entry in
    `round_call_ids`, then a closing reasoning + text. Returns the persisted
    reasoning payloads (with whatever anchors production stamped)."""
    body = ResponsesBody(model="anthropic/claude-opus-4.8", input=[], stream=True)
    n = len(round_call_ids)
    call_count = [0]

    async def streaming(self, session, request_body, **_k):
        call_count[0] += 1
        rnd = call_count[0]
        if rnd <= n:
            cid = round_call_ids[rnd - 1]
            rid = f"rs-pre-{rnd}"
            block = {"type": "reasoning", "id": rid,
                     "content": [{"type": "reasoning_text", "text": f"PRE{rnd}"}],
                     "signature": f"S{rnd}"}
            yield {"type": "response.output_item.done",
                   "item": {**block, "status": "completed", "summary": []}}
            # Emit the function_call as a stream item too (real Anthropic does),
            # so generation-order call tracking sees it.
            yield {"type": "response.output_item.done", "item": {
                "type": "function_call", "call_id": cid, "name": "f",
                "arguments": "{}", "status": "completed"}}
            yield {"type": "response.completed", "response": {"output": [
                block,
                {"type": "function_call", "call_id": cid, "name": "f", "arguments": "{}"},
            ], "usage": {}}}
        else:
            yield {"type": "response.output_item.done", "item": {
                "id": "rs-final", "type": "reasoning", "status": "completed",
                "content": [{"type": "reasoning_text", "text": "FINAL"}],
                "summary": [], "signature": "SF"}}
            yield {"type": "response.output_text.delta", "delta": "done"}
            yield {"type": "response.completed", "response": {"output": [], "usage": {}}}

    async def mock_exec(calls, registry):
        return [{"type": "function_call_output", "call_id": c.get("call_id"), "output": "out"}
                for c in calls]

    captured: list[dict] = []

    def fake_row(chat_id, message_id, model_id, payload):
        captured.append({"item_type": payload.get("type"), "payload": payload})
        return {"payload": payload, "item_type": payload.get("type")}

    async def fake_persist(rows):
        return [generate_item_id() for _ in rows]

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", streaming)
    monkeypatch.setattr(pipe._ensure_tool_executor(), "_execute_function_calls", mock_exec)
    monkeypatch.setattr(pipe._artifact_store, "_make_db_row", fake_row)
    monkeypatch.setattr(pipe._artifact_store, "_db_persist", fake_persist)

    await pipe._streaming_handler._run_streaming_loop(
        body, valves, None,
        metadata={"model": {"id": "anthropic/claude-opus-4.8"}, "chat_id": "c1", "message_id": "m1"},
        tools={"f": {"callable": lambda **_k: "ok"}},
        session=cast(Any, object()), user_id="u1",
    )
    return [r["payload"] for r in captured if r["item_type"] == "reasoning"]


async def _stage_b_replay(pipe, reasoning_payloads, valves, round_call_ids: list[str]) -> list[dict]:
    """Rebuild an OWUI-shaped turn-2 history (one assistant+tool_calls message per
    round, all reasoning markers clustered on the final continuation message) and
    run the real transformer."""
    markers = [generate_item_id() for _ in reasoning_payloads]
    artifacts = dict(zip(markers, reasoning_payloads))
    messages: list[dict] = [{"role": "user", "content": "q1"}]
    for rnd, cid in enumerate(round_call_ids, 1):
        messages.append({"role": "assistant", "content": f"t{rnd}",
                         "tool_calls": [{"id": cid, "type": "function",
                                         "function": {"name": "f", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": cid, "content": "out"})
    marker_block = "\n\n".join(_serialize_marker(m) for m in markers)
    messages.append({"role": "assistant", "message_id": "final", "content": "done\n\n" + marker_block})
    messages.append({"role": "user", "content": "q2"})

    async def loader(_chat_id, _message_id, ulids):
        return {u: artifacts[u] for u in ulids if u in artifacts}

    return await transform_messages_to_input(
        pipe, messages, chat_id="c1", openwebui_model_id="owui",
        artifact_loader=loader, model_id="anthropic/claude-opus-4.8", valves=valves,
    )


def _has_consecutive_reasoning(items: list[dict]) -> bool:
    r_idx = [i for i, it in enumerate(items) if it.get("type") == "reasoning"]
    return any(b == a + 1 for a, b in zip(r_idx, r_idx[1:]))


def _anchor_leaked(items: list[dict]) -> bool:
    return any(any(str(k).startswith("_anchor") for k in it)
               for it in items if isinstance(it, dict))


# --------------------------------------------------------------------------- #
# Core end-to-end proof (single round)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_e2e_real_round_trip_no_consecutive_reasoning(monkeypatch, pipe_instance_async, capsys):
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(update={
        "TOOL_EXECUTION_MODE": "Pipeline", "MAX_FUNCTION_CALL_LOOPS": 3,
        "PERSIST_REASONING_TOKENS": "conversation", "PERSIST_TOOL_RESULTS": False,
    })
    rounds = ["toolu-A"]
    payloads = await _stage_a_persist(pipe, monkeypatch, valves, rounds)
    result = await _stage_b_replay(pipe, payloads, valves, rounds)
    types = [it.get("type") for it in result]

    print("\n[E2E] Stage-A anchors:",
          [{k: v for k, v in p.items() if k.startswith("_anchor")} for p in payloads])
    print("[E2E] Stage-B wire types:", types)

    assert len(payloads) == 2, "turn should persist pre + final reasoning"
    assert types.count("reasoning") == 2, f"reasoning lost on replay: {types}"
    assert not _has_consecutive_reasoning(result), f"400-trigger shape present: {types}"
    assert not _anchor_leaked(result), "internal anchor key leaked to the wire"
    r_idx = [i for i, t in enumerate(types) if t == "reasoning"]
    fc_idx = [i for i, t in enumerate(types) if t == "function_call"]
    fco_idx = [i for i, t in enumerate(types) if t == "function_call_output"]
    assert min(r_idx) < min(fc_idx), f"first reasoning not before tool calls: {types}"
    assert max(r_idx) > max(fco_idx), f"final reasoning not after tool outputs: {types}"


# --------------------------------------------------------------------------- #
# Bug B: same call_id reused across rounds must NOT collapse reasoning.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_e2e_colliding_call_id_across_rounds(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(update={
        "TOOL_EXECUTION_MODE": "Pipeline", "MAX_FUNCTION_CALL_LOOPS": 5,
        "PERSIST_REASONING_TOKENS": "conversation", "PERSIST_TOOL_RESULTS": False,
    })
    # Three rounds, all emitting the SAME call_id (the chat-completions gateway's
    # `toolcall-{model}-0` reset-per-round behaviour).
    rounds = ["tc-0", "tc-0", "tc-0"]
    payloads = await _stage_a_persist(pipe, monkeypatch, valves, rounds)

    # Persist side: the pre-call reasoning of each round got DISTINCT ordinals
    # even though every call_id is identical.
    pre_ords = sorted(
        p.get("_anchor_following_call_ordinal")
        for p in payloads if "_anchor_following_call_ordinal" in p
    )
    assert pre_ords == [0, 1, 2], f"colliding call_ids collapsed ordinals: {payloads}"

    # Replay side: the round-trip produces no consecutive reasoning and no leak.
    result = await _stage_b_replay(pipe, payloads, valves, rounds)
    types = [it.get("type") for it in result]
    assert types.count("reasoning") == 4, f"reasoning lost: {types}"
    assert not _has_consecutive_reasoning(result), f"collapse with colliding ids: {types}"
    assert not _anchor_leaked(result)


# --------------------------------------------------------------------------- #
# Prompt caching must never decorate (or leak via) a reasoning item.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_e2e_prompt_caching_does_not_touch_or_leak_reasoning(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(update={
        "TOOL_EXECUTION_MODE": "Pipeline", "MAX_FUNCTION_CALL_LOOPS": 3,
        "PERSIST_REASONING_TOKENS": "conversation", "PERSIST_TOOL_RESULTS": False,
        "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
    })
    rounds = ["toolu-A"]
    payloads = await _stage_a_persist(pipe, monkeypatch, valves, rounds)
    result = await _stage_b_replay(pipe, payloads, valves, rounds)
    assert not _anchor_leaked(result)

    def _has_cache_control(item: dict) -> bool:
        if "cache_control" in item:
            return True
        content = item.get("content")
        if isinstance(content, list):
            return any(isinstance(p, dict) and "cache_control" in p for p in content)
        return False

    # Caching must actually be ACTIVE for this test to mean anything: prove at
    # least one non-reasoning block got decorated, then prove no reasoning did.
    assert any(_has_cache_control(it) for it in result if it.get("type") != "reasoning"), \
        "prompt caching did not run -- the no-leak assertion would be vacuous"
    for it in result:
        if it.get("type") == "reasoning":
            assert not _has_cache_control(it), "caching must not decorate a reasoning block"


# --------------------------------------------------------------------------- #
# Meaningful matrix: the invariant holds across the dimensions that affect the
# path (reasoning persistence x tool-round count x execution mode).
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("persist_reasoning", ["disabled", "next_reply", "conversation"])
@pytest.mark.parametrize("n_rounds", [1, 2, 3])
@pytest.mark.parametrize("tool_mode", ["Pipeline", "Open-WebUI"])
async def test_e2e_invariant_matrix(monkeypatch, pipe_instance_async,
                                    persist_reasoning, n_rounds, tool_mode):
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(update={
        "PERSIST_REASONING_TOKENS": persist_reasoning,
        "TOOL_EXECUTION_MODE": tool_mode,
        "MAX_FUNCTION_CALL_LOOPS": n_rounds + 1,
    })
    rounds = [f"tc-{i}" for i in range(n_rounds)]
    payloads = await _stage_a_persist(pipe, monkeypatch, valves, rounds)
    result = await _stage_b_replay(pipe, payloads, valves, rounds)
    types = [it.get("type") for it in result]
    ctx = f"persist={persist_reasoning} rounds={n_rounds} mode={tool_mode} -> {types}"

    assert not _has_consecutive_reasoning(result), f"consecutive reasoning! {ctx}"
    assert not _anchor_leaked(result), f"anchor leaked! {ctx}"
    if persist_reasoning in {"next_reply", "conversation"} and payloads:
        replayed = types.count("reasoning")
        assert replayed == len(payloads), f"reasoning lost ({replayed}/{len(payloads)})! {ctx}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reasoning_first", "expected"),
    [
        (True, ["message", "reasoning", "message", "message"]),
        (False, ["message", "message", "reasoning", "message"]),
    ],
    ids=["reasoning-before-the-answer", "reasoning-after-the-answer"],
)
async def test_e2e_a_no_tool_turn_replays_reasoning_where_it_was_generated(
    monkeypatch, pipe_instance_async, reasoning_first, expected
):
    """Every marker is appended at the tail, so position is lost unless it is recorded.

    On a turn with no tool call there is no ordinal to anchor to, and both shapes store
    byte-identical content: the answer, then the markers. The text ordinal stamped at
    persist time is the only thing that separates them, and the two rows here differ
    ONLY in when the reasoning arrived -- so a constant answer cannot satisfy both.
    """
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(update={"PERSIST_REASONING_TOKENS": "conversation"})
    reasoning_event = {
        "type": "response.output_item.done",
        "item": {
            "id": "rs-1", "type": "reasoning", "status": "completed",
            "content": [{"type": "reasoning_text", "text": "THOUGHT"}],
            "summary": [], "signature": "SIG-1",
        },
    }
    text_event = {"type": "response.output_text.delta", "delta": "The answer."}
    ordered = [reasoning_event, text_event] if reasoning_first else [text_event, reasoning_event]

    async def streaming(self, session, request_body, **_kwargs):
        for event in ordered:
            yield event
        yield {"type": "response.completed", "response": {"output": [], "usage": {}}}

    captured: list[dict] = []

    def fake_row(chat_id, message_id, model_id, payload):
        captured.append(payload)
        return {"payload": payload, "item_type": payload.get("type")}

    async def fake_persist(rows):
        return [generate_item_id() for _ in rows]

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", streaming)
    monkeypatch.setattr(pipe._artifact_store, "_make_db_row", fake_row)
    monkeypatch.setattr(pipe._artifact_store, "_db_persist", fake_persist)

    await pipe._streaming_handler._run_streaming_loop(
        ResponsesBody(model="anthropic/claude-opus-4.8", input=[], stream=True),
        valves, None,
        metadata={"model": {"id": "anthropic/claude-opus-4.8"}, "chat_id": "c1", "message_id": "m1"},
        tools={}, session=cast(Any, object()), user_id="u1",
    )

    payloads = [p for p in captured if p.get("type") == "reasoning"]
    assert len(payloads) == 1

    marker = generate_item_id()
    artifacts = {marker: payloads[0]}

    async def loader(_chat_id, _message_id, ulids):
        return {u: artifacts[u] for u in ulids if u in artifacts}

    replayed = await transform_messages_to_input(
        pipe,
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "message_id": "m1",
             "content": "The answer.\n\n" + _serialize_marker(marker)},
            {"role": "user", "content": "q2"},
        ],
        chat_id="c1", openwebui_model_id="owui", artifact_loader=loader,
        model_id="anthropic/claude-opus-4.8", valves=valves,
    )

    assert [i.get("type") for i in replayed] == expected
    assert not _anchor_leaked(replayed)
    assert replayed[expected.index("reasoning")].get("signature") == "SIG-1"


@pytest.mark.asyncio
async def test_e2e_hoisted_reasoning_never_jumps_the_system_prompt(
    monkeypatch, pipe_instance_async
):
    """Regions are delimited by USER messages, so a system prompt shares the region.

    Anchoring on "the first message item" rather than "the first ASSISTANT message
    item" places the turn's reasoning ahead of the system prompt -- a different
    conversation from the one the model was given.
    """
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(update={"PERSIST_REASONING_TOKENS": "conversation"})

    async def streaming(self, session, request_body, **_kwargs):
        yield {"type": "response.output_item.done", "item": {
            "id": "rs-1", "type": "reasoning", "status": "completed",
            "content": [{"type": "reasoning_text", "text": "THOUGHT"}],
            "summary": [], "signature": "SIG-1"}}
        yield {"type": "response.output_text.delta", "delta": "The answer."}
        yield {"type": "response.completed", "response": {"output": [], "usage": {}}}

    captured: list[dict] = []

    def fake_row(chat_id, message_id, model_id, payload):
        captured.append(payload)
        return {"payload": payload, "item_type": payload.get("type")}

    async def fake_persist(rows):
        return [generate_item_id() for _ in rows]

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", streaming)
    monkeypatch.setattr(pipe._artifact_store, "_make_db_row", fake_row)
    monkeypatch.setattr(pipe._artifact_store, "_db_persist", fake_persist)

    await pipe._streaming_handler._run_streaming_loop(
        ResponsesBody(model="anthropic/claude-opus-4.8", input=[], stream=True),
        valves, None,
        metadata={"model": {"id": "anthropic/claude-opus-4.8"}, "chat_id": "c1", "message_id": "m1"},
        tools={}, session=cast(Any, object()), user_id="u1",
    )

    payload = next(p for p in captured if p.get("type") == "reasoning")
    marker = generate_item_id()

    async def loader(_chat_id, _message_id, ulids):
        return {marker: payload} if marker in ulids else {}

    replayed = await transform_messages_to_input(
        pipe,
        [
            {"role": "system", "content": "SYSTEM PROMPT"},
            {"role": "assistant", "message_id": "m1",
             "content": "The answer.\n\n" + _serialize_marker(marker)},
        ],
        chat_id="c1", openwebui_model_id="owui", artifact_loader=loader,
        model_id="anthropic/claude-opus-4.8", valves=valves,
    )

    roles = [(i.get("type"), i.get("role")) for i in replayed]
    assert roles[0] == ("message", "system"), (
        f"reasoning was placed ahead of the system prompt: {roles}"
    )
    assert [t for t, _ in roles] == ["message", "reasoning", "message"]
