# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportGeneralTypeIssues=false, reportCallIssue=false
"""Replay-side tests for the interleaved-reasoning ordering fix.

Production failure (OpenRouter -> Anthropic, interleaved thinking + tool calls):

    400 messages.N.content.X: `thinking`/`redacted_thinking` blocks in the latest
    assistant message cannot be modified.

The model emits reasoning interleaved around tool calls (thinking before a call,
more thinking after the result). On replay the persisted reasoning must be put
back on the correct side of each call, never collapsed into a consecutive run.
Each reasoning carries the ORDINAL (0,1,2...) of the call it sits next to; the
transformer places it relative to the Nth call in its turn. Ordinals stay unique
within a turn even when a provider reuses call_ids across rounds.

These tests drive the REAL transformer (only the DB fetch is a double).
"""
from __future__ import annotations

import asyncio
import json

from open_webui_openrouter_pipe import (
    Pipe,
    _responses_payload_to_chat_completions_payload,
    _serialize_marker,
    generate_item_id,
)
from open_webui_openrouter_pipe.requests.transformer import (
    transform_messages_to_input,
    _reinterleave_region,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _reasoning_artifact(ulid: str, text, signature: str, *,
                        seq: int | None = None,
                        following_call_ordinal: int | None = None,
                        preceding_call_ordinal: int | None = None,
                        encrypted_content: str | None = None) -> dict:
    """The shape `normalize_persisted_item` stores/returns for a reasoning block.

    Ordering anchors live as internal `_anchor_*` keys in the payload; the
    transformer reads them to re-interleave, then strips them before send.
    `text=None` produces an empty-content block (the redacted_thinking shape).
    """
    payload = {
        "id": ulid,
        "type": "reasoning",
        "status": "completed",
        "content": [{"type": "reasoning_text", "text": text}] if text is not None else [],
        "summary": [],
        "signature": signature,
        "format": "anthropic-claude-v1",
    }
    if encrypted_content is not None:
        payload["encrypted_content"] = encrypted_content
    if seq is not None:
        payload["_anchor_seq"] = seq
    if following_call_ordinal is not None:
        payload["_anchor_following_call_ordinal"] = following_call_ordinal
    if preceding_call_ordinal is not None:
        payload["_anchor_preceding_call_ordinal"] = preceding_call_ordinal
    return payload


def _run_transform(messages, artifacts, *, model_id="anthropic/claude-opus-4.8",
                   persist="conversation"):
    """Drive the REAL transform; only the DB fetch (loader) is a test double."""
    pipe = Pipe()
    valves = pipe.valves.model_copy(update={"PERSIST_REASONING_TOKENS": persist})

    async def loader(chat_id, message_id, ulids):
        _ = (chat_id, message_id)  # caller passes (chat_id, message_id, ulids); unused here
        return {u: artifacts[u] for u in ulids if u in artifacts}

    async def _run():
        try:
            return await transform_messages_to_input(
                pipe, messages,
                chat_id="chat-1", openwebui_model_id="owui-model",
                artifact_loader=loader, model_id=model_id, valves=valves,
            )
        finally:
            await pipe.close()

    return asyncio.run(_run())


def _types(items):
    return [it.get("type") for it in items]


def _reasoning_text(item):
    parts = item.get("content") or []
    return " ".join(p.get("text", "") for p in parts if isinstance(p, dict))


def _no_anchor_keys(items) -> bool:
    return not any(
        str(k).startswith("_anchor") for it in items if isinstance(it, dict) for k in it
    )


def _faithful_turn2_history(r1_ulid: str, r2_ulid: str):
    """The EXACT turn-2 history OWUI hands the pipe (verified from the live log):
    msg[1] has native `tool_calls` (no markers); the continuation message carries
    both reasoning markers at its tail."""
    return [
        {"role": "user", "content": "Please give me instructions for a relational systems map"},
        {
            "role": "assistant",
            "content": "Welcome. Let me look.",
            "tool_calls": [
                {"id": "toolu_01T3", "type": "function",
                 "function": {"name": "query_knowledge_files", "arguments": "{}"}},
                {"id": "toolu_01Xc", "type": "function",
                 "function": {"name": "query_knowledge_files", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_01T3", "content": "[{\"content\": \"...\"}]"},
        {"role": "tool", "tool_call_id": "toolu_01Xc", "content": "[{\"content\": \"...\"}]"},
        {
            "role": "assistant",
            "message_id": "asst-final",
            "content": "There's something important here.\n\n"
                       + _serialize_marker(r1_ulid) + "\n\n" + _serialize_marker(r2_ulid),
        },
        {"role": "user", "content": "I want a whole picture of the systems aliveness"},
    ]


# --------------------------------------------------------------------------- #
# chat_completions drops reasoning entirely, so it cannot carry this bug.
# --------------------------------------------------------------------------- #
def test_chat_completions_conversion_drops_reasoning_items():
    """The Responses->ChatCompletions conversion drops standalone `reasoning`
    items, so the chat endpoint cannot carry the interleaved thinking blocks that
    trigger this 400 (observed only on /responses). Scope: the fix is Responses-only."""
    payload = {
        "model": "anthropic/claude-opus-4.8",
        "input": [
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "PLAN_TOKEN"}],
             "signature": "SIG1"},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "hi"}]},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "REFLECT_TOKEN"}],
             "signature": "SIG2"},
        ],
    }
    messages = _responses_payload_to_chat_completions_payload(payload)["messages"]
    blob = json.dumps(messages)
    assert "PLAN_TOKEN" not in blob and "REFLECT_TOKEN" not in blob
    assert all(m.get("role") != "reasoning" for m in messages)
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages)
    assert any(m.get("role") == "tool" for m in messages)


# --------------------------------------------------------------------------- #
# Anchored reasoning yields the valid, interleaved wire order.
# --------------------------------------------------------------------------- #
def test_anchored_reasoning_exact_wire_shape():
    """Full item-type sequence: reasoning on the correct side of the tool round,
    never consecutive (the inverse of the messages.N.content.X collapse)."""
    r1, r2 = generate_item_id(), generate_item_id()
    messages = _faithful_turn2_history(r1, r2)
    artifacts = {
        r1: _reasoning_artifact(r1, "PLAN: check the library", "SIG_PLAN",
                                seq=0, following_call_ordinal=0),
        r2: _reasoning_artifact(r2, "REFLECT: synthesize", "SIG_REFLECT",
                                seq=1, preceding_call_ordinal=1),
    }
    types = _types(_run_transform(messages, artifacts))
    assert types == [
        "message", "message", "reasoning",
        "function_call", "function_call",
        "function_call_output", "function_call_output",
        "reasoning", "message", "message",
    ], f"unexpected wire shape: {types}"


def test_interleaved_reasoning_replayed_in_generation_order():
    """PLAN (pre-tool) precedes the function_calls, REFLECT (post-tool) follows the
    tool outputs, the two are NOT consecutive, anchors stripped, signatures intact."""
    r1, r2 = generate_item_id(), generate_item_id()
    messages = _faithful_turn2_history(r1, r2)
    artifacts = {
        r1: _reasoning_artifact(r1, "PLAN: check the library", "SIG_PLAN",
                                seq=0, following_call_ordinal=0),
        r2: _reasoning_artifact(r2, "REFLECT: synthesize", "SIG_REFLECT",
                                seq=1, preceding_call_ordinal=1),
    }
    result = _run_transform(messages, artifacts)
    types = _types(result)
    assert types.count("reasoning") == 2
    r_idx = [i for i, t in enumerate(types) if t == "reasoning"]
    fc_idx = [i for i, t in enumerate(types) if t == "function_call"]
    fco_idx = [i for i, t in enumerate(types) if t == "function_call_output"]
    assert r_idx[1] > r_idx[0] + 1, f"reasoning must not be consecutive: {types}"
    plan_idx = next(i for i, it in enumerate(result)
                    if it.get("type") == "reasoning" and "PLAN" in _reasoning_text(it))
    reflect_idx = next(i for i, it in enumerate(result)
                       if it.get("type") == "reasoning" and "REFLECT" in _reasoning_text(it))
    assert plan_idx < min(fc_idx), f"PLAN must precede the function_calls: {types}"
    assert reflect_idx > max(fco_idx), f"REFLECT must follow the tool outputs: {types}"
    assert _no_anchor_keys(result), "anchor key leaked to the wire"
    sigs = {_reasoning_text(it).split(":")[0]: it.get("signature")
            for it in result if it.get("type") == "reasoning"}
    assert sigs.get("PLAN") == "SIG_PLAN" and sigs.get("REFLECT") == "SIG_REFLECT"


def test_unanchored_reasoning_is_preserved_not_dropped():
    """Reasoning with no anchor (legacy artifacts) is left untouched, never dropped."""
    r1, r2 = generate_item_id(), generate_item_id()
    messages = _faithful_turn2_history(r1, r2)
    artifacts = {
        r1: _reasoning_artifact(r1, "PLAN", "SIG_PLAN"),
        r2: _reasoning_artifact(r2, "REFLECT", "SIG_REFLECT"),
    }
    result = _run_transform(messages, artifacts)
    assert _types(result).count("reasoning") == 2
    assert _no_anchor_keys(result)


def test_consecutive_reasoning_in_no_tool_turn_is_preserved():
    """Two genuinely-consecutive reasoning blocks in a turn with NO tool call (e.g.
    redacted_thinking immediately followed by thinking) must both survive replay."""
    r1, r2 = generate_item_id(), generate_item_id()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "message_id": "a1",
         "content": "Answer.\n\n" + _serialize_marker(r1) + "\n\n" + _serialize_marker(r2)},
        {"role": "user", "content": "next"},
    ]
    artifacts = {
        r1: _reasoning_artifact(r1, None, "SIG1", encrypted_content="enc"),  # redacted shape
        r2: _reasoning_artifact(r2, "THINKING", "SIG2"),
    }
    result = _run_transform(messages, artifacts)
    assert _types(result).count("reasoning") == 2, f"both must survive: {_types(result)}"


def test_redacted_thinking_shape_anchored_before_call():
    """A redacted block (empty content + encrypted_content) anchored before a call
    is replayed on the correct side and survives (not dropped)."""
    r1 = generate_item_id()
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "thinking...",
         "tool_calls": [{"id": "c0", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "out"},
        {"role": "assistant", "message_id": "a1", "content": "done\n\n" + _serialize_marker(r1)},
        {"role": "user", "content": "q2"},
    ]
    artifacts = {r1: _reasoning_artifact(r1, None, "SIGR", seq=0,
                                         following_call_ordinal=0, encrypted_content="enc-blob")}
    result = _run_transform(messages, artifacts)
    types = _types(result)
    assert types.count("reasoning") == 1, f"redacted reasoning lost: {types}"
    r_idx = types.index("reasoning")
    assert r_idx < types.index("function_call"), f"redacted block must precede its call: {types}"
    red = next(it for it in result if it.get("type") == "reasoning")
    assert red.get("encrypted_content") == "enc-blob"  # preserved
    assert _no_anchor_keys(result)


def test_anchor_does_not_bind_across_turns():
    """Ordinals are scoped per assistant turn (regions split on user messages), so
    turn-2 reasoning cannot bind to turn-1's calls even when call_ids collide and
    ordinals repeat. Turn 1 carries TWO reasoning so a scope leak would be visible
    as a consecutive pair."""
    r1a, r1b, r2 = generate_item_id(), generate_item_id(), generate_item_id()
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "t1", "tool_calls": [
            {"id": "tc-0", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            {"id": "tc-1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "tc-0", "content": "o"},
        {"role": "tool", "tool_call_id": "tc-1", "content": "o"},
        {"role": "assistant", "message_id": "a1",
         "content": "done1\n\n" + _serialize_marker(r1a) + "\n\n" + _serialize_marker(r1b)},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "t2", "tool_calls": [
            {"id": "tc-0", "type": "function", "function": {"name": "f", "arguments": "{}"}},  # collides w/ turn 1
        ]},
        {"role": "tool", "tool_call_id": "tc-0", "content": "o"},
        {"role": "assistant", "message_id": "a2", "content": "done2\n\n" + _serialize_marker(r2)},
        {"role": "user", "content": "q3"},
    ]
    artifacts = {
        r1a: _reasoning_artifact(r1a, "T1-A", "S", seq=0, following_call_ordinal=0),
        r1b: _reasoning_artifact(r1b, "T1-B", "S", seq=1, following_call_ordinal=1),
        r2: _reasoning_artifact(r2, "T2", "S", seq=0, following_call_ordinal=0),
    }
    result = _run_transform(messages, artifacts)
    types = _types(result)
    assert types.count("reasoning") == 3, f"all reasoning must survive: {types}"
    r_idx = [i for i, t in enumerate(types) if t == "reasoning"]
    assert all(b > a + 1 for a, b in zip(r_idx, r_idx[1:])), f"reasoning collapsed: {types}"
    t2_idx = next(i for i, it in enumerate(result)
                  if it.get("type") == "reasoning" and "T2" in _reasoning_text(it))
    q2_idx = next(i for i, it in enumerate(result)
                  if it.get("type") == "message" and "whole" not in _reasoning_text(it)
                  and it.get("role") == "user"
                  and "q2" in json.dumps(it.get("content")))
    assert t2_idx > q2_idx, f"turn-2 reasoning leaked into an earlier turn: {types}"


# --------------------------------------------------------------------------- #
# Direct _reinterleave_region tests for the fallback/edge branches.
# --------------------------------------------------------------------------- #
def _msg(text):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _fc(cid):
    return {"type": "function_call", "call_id": cid, "name": "f", "arguments": "{}"}


def _fco(cid):
    return {"type": "function_call_output", "call_id": cid, "output": "out"}


def _reason(text, **anchors):
    return _reasoning_artifact("rid-" + text, text, "S", **anchors)


def test_region_preceding_falls_back_to_function_call_without_output():
    """A preceding-ordinal block whose tool round has no output lands right after
    the matching function_call (the no-output fallback)."""
    region = [_msg("x"), _fc("z"), _msg("done"), _reason("R", seq=0, preceding_call_ordinal=0)]
    out = _reinterleave_region(region)
    types = [x.get("type") for x in out]
    assert types == ["message", "function_call", "reasoning", "message"], types
    assert _no_anchor_keys(out)


def test_region_orphaned_ordinal_kept_in_place_no_leak():
    """A reasoning anchored to an ordinal beyond the available calls (e.g. its call
    was pruned) is kept, never dropped, and leaks no anchor key."""
    region = [
        _msg("x"), _fc("a"), _fco("a"), _msg("done"),
        _reason("VALID", seq=0, following_call_ordinal=0),   # resolvable -> before fc[0]
        _reason("ORPHAN", seq=1, following_call_ordinal=9),  # out of range -> kept in place
    ]
    out = _reinterleave_region(region)
    types = [x.get("type") for x in out]
    assert types.count("reasoning") == 2, f"orphaned reasoning dropped: {types}"
    r_idx = [i for i, t in enumerate(types) if t == "reasoning"]
    assert all(b > a + 1 for a, b in zip(r_idx, r_idx[1:])), f"unexpected adjacency: {types}"
    valid_idx = next(i for i, x in enumerate(out)
                     if x.get("type") == "reasoning" and _reasoning_text(x) == "VALID")
    assert valid_idx < types.index("function_call"), f"VALID not before its call: {types}"
    assert _no_anchor_keys(out)


def test_region_preceding_with_missing_output_no_wedge():
    """A failed tool (call 'a') produces no output, but call 'b' has one. A
    preceding block for call 1 ('b') must land after b's OWN output (matched by
    call_id), NEVER wedged between b's tool_use and tool_result."""
    region = [
        _fc("a"), _fc("b"), _fco("b"), _msg("done"),
        _reason("POST_B", seq=0, preceding_call_ordinal=1),
    ]
    out = _reinterleave_region(region)
    types = [x.get("type") for x in out]
    assert types == ["function_call", "function_call", "function_call_output",
                     "reasoning", "message"], types
    # Explicit: no reasoning sits between a function_call and a function_call_output.
    for i, t in enumerate(types[:-2]):
        if t == "function_call" and types[i + 1] == "reasoning" and types[i + 2] == "function_call_output":
            raise AssertionError(f"reasoning wedged between tool_use and tool_result: {types}")
    assert _no_anchor_keys(out)


def test_region_preceding_colliding_call_ids_pair_in_order():
    """Repeated call_ids across rounds pair to their own output in order, so a
    preceding block lands after the correct round's output, not the first match."""
    region = [
        _fc("tc0"), _fco("tc0"), _fc("tc0"), _fco("tc0"), _msg("done"),
        _reason("AFTER_R2", seq=0, preceding_call_ordinal=1),
    ]
    out = _reinterleave_region(region)
    types = [x.get("type") for x in out]
    assert types == ["function_call", "function_call_output", "function_call",
                     "function_call_output", "reasoning", "message"], types
    assert _no_anchor_keys(out)
