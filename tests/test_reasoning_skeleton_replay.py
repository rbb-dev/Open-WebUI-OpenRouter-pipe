"""With tool results not kept, the next turn must still replay each tool round's structure around its reasoning.

A reasoning model that thinks before and after a tool call produces thinking blocks with a tool round between them.
When the round is missing, replay puts the blocks next to each other, and Anthropic rejects exactly that ("thinking
blocks ... cannot be modified", recorded live in tests/fixtures/anthropic_reasoning_replay_probe.json). So the pipe
keeps its own copy of every round it runs -- the call and its result, committed where the round happened -- and
replay uses it whenever Open WebUI does not hand the same round back; while results are not kept, the model is
handed `{}` and a placeholder in their place. The thinking blocks stay apart. The copy is a record of the round,
not of the reasoning: it survives every path that drops reasoning (measured accepted without it, t233a).

Stage A is the real streaming loop. Stage B feeds the next request what Open WebUI would: its saved message when the
turn published one, otherwise the content stage A produced (text plus hidden markers), through the real transformer.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from open_webui_openrouter_pipe import Pipe, ResponsesBody, _ToolExecutionContext, generate_item_id
from open_webui_openrouter_pipe.api.transforms import (
    _filter_openrouter_request,
    _responses_payload_to_chat_completions_payload,
)
from open_webui_openrouter_pipe.core.utils import PIPE_ONLY_TOOL_ROUND_KEY, TOOL_ROUND_SKELETON_KEY, contains_marker
from open_webui_openrouter_pipe.models.reasoning_config import ReasoningConfigManager
from open_webui_openrouter_pipe.requests.sanitizer import _sanitize_request_input
from open_webui_openrouter_pipe.requests.transformer import transform_messages_to_input
from tests.test_continue_stores_once import _open_webui_convert_output_to_messages

MODEL = "anthropic/claude-opus-4.8"
RESULT_CANARY = "SECRET-TOOL-RESULT-7f3a"
ARGUMENT_CANARY = "SECRET-ARGUMENT-19c2"
FIXTURE = Path(__file__).parent / "fixtures" / "anthropic_reasoning_replay_probe.json"

SEQUENTIAL_TWO_ROUNDS = [("calls", ["toolu-1"]), ("calls", ["toolu-2"]), ("answer", False)]
PARALLEL_WITH_FINAL_REASONING = [("calls", ["toolu-a", "toolu-b"]), ("answer", True)]


def _recorded(group: str, arm: str) -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))[group]["arms"][arm]


def _shape(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("type")) if item.get("type") != "message" else f"message:{item.get('role')}" for item in items]


def _valves(pipe, **changes):
    settings = {
        "TOOL_EXECUTION_MODE": "Pipeline",
        "PERSIST_REASONING_TOKENS": "conversation",
        "PERSIST_TOOL_RESULTS": False,
        "SHOW_TOOL_CARDS": False,
        "MAX_FUNCTION_CALL_LOOPS": 5,
    }
    settings.update(changes)
    return pipe.valves.model_copy(update=settings)


async def _stage_a(pipe, monkeypatch, valves, rounds, *, stream=True, emitter=None, tool_status="completed",
                   real_executor=False, signed: bool | Literal["at-completion"] = True, message_id: str | None = "m1",
                   real_row_builder=False, real_store=False,
                   chat_id: str = "c1", tool_name: str = "lookup", stop_in_round: int | None = None,
                   rows: dict[str, dict[str, Any]] | None = None, tool_result: Any = RESULT_CANARY,
                   continues_after_marker: bool = False, builtin_ask_user: bool = False):
    """Run one turn. Each round is ("calls", [call ids]), which reasons, writes and calls; ("quiet-calls", [call ids]),
    which writes and calls without reasoning; ("silent-calls", [call ids]), which only calls; ("silent-search-then-calls",
    [call ids]) and ("think-silent-search-then-calls", [call ids]), which have OpenRouter run a web search and then call,
    writing nothing, the second reasoning first; ("search-then-calls", [call ids]), which reasons, has OpenRouter run a web
    search (item id "ws-<round>"), writes and calls; ("advise-then-calls", [call ids]) and ("search-think-then-calls",
    [call ids]), which reason, have OpenRouter consult its advisor (item id "adv-<round>") or run a web search, reason
    again ("THOUGHT-<round>-AFTER"), write and call; or ("answer", reasons_first); or ("answer-then-reasoning", None),
    which writes and only then reports its reasoning, a provider that reasons after its own message item; or
    ("call-then-thought", [call ids]), ("call-then-thought-done", [call ids]) and
    ("call-thought-call-thought", [two call ids]), which call first and only then report their reasoning, the first
    streamed as text and the second closed by its own `output_item.done` -- the shape whose box belongs under its call.
    ``signed=False`` streams reasoning with
    no signature, which Anthropic cannot take back; ``signed="at-completion"`` streams it unsigned and signs it only in
    the completed response, as Anthropic does. ``message_id=None`` sends no message id, as an API request does;
    ``real_row_builder`` builds rows with the store's own `_make_db_row` instead of a stand-in, and ``real_store`` keeps
    the store's own writer, which holds a temporary chat's reply in memory. ``stop_in_round``
    cancels the turn as Stop does, when that round's model call starts; pass ``rows`` to see what was stored by then.
    Rows are copied when they are written, as the database stores them: a change made to an item afterwards is not in
    its row. ``builtin_ask_user`` registers the round's tool as Open WebUI's real builtin ``ask_user`` rather than a
    user's tool of the same name, which is the shape that earns the privacy exemption.

    Returns the content the loop produced, the rows it persisted (ulid -> payload) and the events it emitted.
    """
    step = [0]

    async def model(self, session, request_body, **_kwargs):
        kind, value = rounds[step[0]]
        step[0] += 1
        index = step[0]
        if stop_in_round == index:
            raise asyncio.CancelledError()
        output: list[dict[str, Any]] = []

        def thought(suffix: str) -> dict[str, Any]:
            block = {"type": "reasoning", "id": f"rs-{index}{suffix}", "status": "completed",
                     "content": [{"type": "reasoning_text", "text": f"THOUGHT-{index}{suffix.upper()}"}], "summary": []}
            if signed:
                block["signature"] = f"sig-{index}{suffix}"
            output.append(block)
            streamed = {k: v for k, v in block.items() if k != "signature"} if signed == "at-completion" else block
            return {"type": "response.output_item.done", "item": streamed}

        if kind == "answer-then-reasoning":
            # A provider that returns its reasoning *after* its own message item. The
            # input order is the point, so it is written out here rather than reused from
            # the "answer" shape: a future edit that reasons first would neuter the row
            # that depends on it, and the test asserts this order for that reason.
            yield {"type": "response.output_text.delta", "delta": f"text {index} "}
            yield thought("")
            yield {"type": "response.completed", "response": {"output": output, "usage": {}}}
            return

        if kind in ("delta-calls", "summary-calls"):
            # Reasoning the provider streams as text and never closes with its own
            # `output_item.done` -- the common shape, and the one the pipe can only
            # publish at the end of the round. No text before the call, so the call is
            # what the box has to precede. The block still reaches the completed response,
            # so the round is a replayable one.
            reasoning_id = f"rs-{index}"
            yield {"type": "response.output_item.added", "output_index": 0,
                   "item": {"type": "reasoning", "id": reasoning_id, "status": "in_progress"}}
            yield {"type": "response.reasoning_text.delta", "item_id": reasoning_id,
                   "delta": f"THOUGHT-{index} "}
            if kind == "summary-calls":
                yield {"type": "response.reasoning_summary_text.done", "item_id": reasoning_id,
                       "text": f"**THOUGHT-{index}**"}
            output.append({"type": "reasoning", "id": reasoning_id, "status": "completed",
                           "content": [{"type": "reasoning_text", "text": f"THOUGHT-{index}"}], "summary": []})
            for call_id in value:
                call = {"type": "function_call", "call_id": call_id, "name": tool_name,
                        "arguments": json.dumps({"q": ARGUMENT_CANARY}), "status": "completed"}
                yield {"type": "response.output_item.done", "item": call}
                output.append(call)
            yield {"type": "response.completed", "response": {"output": output, "usage": {}}}
            return

        if kind in ("call-then-thought", "call-then-thought-done", "call-thought-call-thought"):
            # A provider that reports its reasoning *after* its own tool call. The order
            # is the point, so it is written out here rather than reused from the
            # "delta-calls" shape: no other kind has a call in front of the thought, and a
            # future edit that reasoned first would neuter the rows that depend on it, so
            # the tests assert this order off the record rather than assume it. Each call
            # id is distinct per round, so a round's card is its own.
            for slot, call_id in enumerate(value):
                call = {"type": "function_call", "call_id": call_id, "name": tool_name,
                        "arguments": json.dumps({"q": ARGUMENT_CANARY}), "status": "completed"}
                yield {"type": "response.output_item.done", "item": call}
                output.append(call)
                if kind == "call-thought-call-thought" and slot == 0:
                    between = f"rs-{index}-between"
                    if signed:
                        between_block = {"type": "reasoning", "id": between, "status": "completed",
                                         "content": [{"type": "reasoning_text", "text": f"BETWEEN-{index}"}],
                                         "summary": [], "signature": f"sig-{index}-between"}
                        yield {"type": "response.output_item.done", "item": between_block}
                    else:
                        yield {"type": "response.output_item.added", "output_index": 0,
                               "item": {"type": "reasoning", "id": between, "status": "in_progress"}}
                        yield {"type": "response.reasoning_text.delta", "item_id": between,
                               "delta": f"BETWEEN-{index}"}
                    output.append({"type": "reasoning", "id": between, "status": "completed",
                                   "content": [{"type": "reasoning_text", "text": f"BETWEEN-{index}"}], "summary": []})
            reasoning_id = f"rs-{index}"
            if kind == "call-then-thought-done":
                yield thought("")
            else:
                yield {"type": "response.output_item.added", "output_index": 0,
                       "item": {"type": "reasoning", "id": reasoning_id, "status": "in_progress"}}
                yield {"type": "response.reasoning_text.delta", "item_id": reasoning_id,
                       "delta": f"THOUGHT-{index} "}
                output.append({"type": "reasoning", "id": reasoning_id, "status": "completed",
                               "content": [{"type": "reasoning_text", "text": f"THOUGHT-{index}"}], "summary": []})
            yield {"type": "response.completed", "response": {"output": output, "usage": {}}}
            return

        consulting = kind in ("advise-then-calls", "search-think-then-calls")
        silent = kind in ("silent-calls", "silent-search-then-calls", "think-silent-search-then-calls")
        if kind in ("calls", "search-then-calls", "think-silent-search-then-calls") or consulting or (
            kind == "answer" and value
        ):
            yield thought("")
        if kind in ("search-then-calls", "search-think-then-calls", "silent-search-then-calls",
                    "think-silent-search-then-calls"):
            search = {"type": "openrouter:web_search", "id": f"ws-{index}", "status": "completed",
                      "action": {"sources": []}}
            yield {"type": "response.output_item.done", "item": search}
            output.append(search)
        if kind == "advise-then-calls":
            advice = {"type": "openrouter:advisor", "id": f"adv-{index}", "status": "completed",
                      "advice": f"ADVICE-{index}"}
            yield {"type": "response.output_item.done", "item": advice}
            output.append(advice)
        if consulting:
            yield thought("-after")
        if not silent:
            yield {"type": "response.output_text.delta", "delta": f"text {index} "}
        if kind in ("calls", "quiet-calls", "search-then-calls") or consulting or silent:
            for call_id in value:
                call = {"type": "function_call", "call_id": call_id, "name": tool_name,
                        "arguments": json.dumps({"q": ARGUMENT_CANARY}), "status": "completed"}
                yield {"type": "response.output_item.done", "item": call}
                output.append(call)
        yield {"type": "response.completed", "response": {"output": output, "usage": {}}}

    async def lookup(**_kwargs):
        return tool_result

    async def run_tools(calls, _registry):
        return [{"type": "function_call_output", "call_id": c.get("call_id"), "output": tool_result,
                 "status": tool_status} for c in calls]

    persisted: dict[str, dict[str, Any]] = {} if rows is None else rows

    def make_row(_chat_id, _message_id, _model_id, payload):
        return {"payload": payload}

    async def persist(rows):
        ulids = [generate_item_id() for _ in rows]
        persisted.update(zip(ulids, (copy.deepcopy(row["payload"]) for row in rows)))
        return ulids

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", model)
    monkeypatch.setattr(Pipe, "send_openrouter_nonstreaming_request_as_events", model)
    if not real_executor:
        monkeypatch.setattr(pipe._ensure_tool_executor(), "_execute_function_calls", run_tools)
    if real_row_builder:
        monkeypatch.setattr(pipe._artifact_store, "_item_model", object())
    else:
        monkeypatch.setattr(pipe._artifact_store, "_make_db_row", make_row)
    if not real_store:
        monkeypatch.setattr(pipe._artifact_store, "_db_persist", persist)

    emitted: list[dict[str, Any]] = []

    async def capture(event):
        emitted.append(event)

    body = ResponsesBody(model=MODEL, input=[], stream=stream)
    body._continues_after_marker = continues_after_marker
    registry = {"lookup": {"type": "function", "callable": lookup,
                           "spec": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}}
    if builtin_ask_user:
        registry[tool_name] = {"type": "builtin", "tool_id": "builtin:ask_user",
                               "spec": {"name": tool_name, "parameters": {"type": "object", "properties": {}}}}
    handler = pipe._streaming_handler
    context = token = None
    if real_executor:
        context = _ToolExecutionContext(queue=asyncio.Queue(maxsize=50), per_request_semaphore=asyncio.Semaphore(1),
                                        global_semaphore=None, timeout=5.0, batch_timeout=5.0, idle_timeout=None,
                                        user_id="u1", event_emitter=None, batch_cap=1)
        context.workers.append(asyncio.create_task(pipe._ensure_tool_executor()._tool_worker_loop(context)))
        token = pipe._TOOL_CONTEXT.set(context)
    try:
        runner = handler._run_streaming_loop if stream else handler._run_nonstreaming_loop
        content = await runner(
            body, valves, capture if emitter is None else emitter,
            metadata={"model": {"id": MODEL}, "chat_id": chat_id, **({"message_id": message_id} if message_id else {}),
                      **({"assistant_message_id": message_id} if continues_after_marker else {})},
            tools=registry, session=cast(Any, object()), user_id="u1",
        )
    finally:
        if context is not None:
            pipe._TOOL_CONTEXT.reset(token)
            for worker in context.workers:
                worker.cancel()
            await asyncio.gather(*context.workers, return_exceptions=True)
    return content, persisted, emitted


def _recorded_output(emitted):
    """The output the turn published, which is what Open WebUI stores as the message's record."""
    for event in reversed(emitted):
        if event.get("type") == "response.completed":
            return (event.get("response") or {}).get("output") or []
    return []


async def _stage_b(pipe, valves, content, persisted, *, refs=None, recorded=None, ask_user_names=frozenset()):
    """Feed the turn back as the next request.

    When the turn published output items, Open WebUI does NOT hand the pipe the assistant message as text:
    `process_messages_with_output` replaces any assistant message carrying `output` with the messages
    `convert_output_to_messages(output, raw=True, flatten_tool_images=True)` builds from it, and those
    carry no message id. Passing `recorded` reproduces that. Turns that publish nothing -- non-streaming,
    or no event emitter -- still arrive as a plain assistant message, which is the `recorded=None` path.
    """
    async def loader(_chat_id, _message_id, ulids):
        return {u: persisted[u] for u in ulids if u in persisted}

    if recorded:
        assistant_turn = _open_webui_convert_output_to_messages()(
            recorded, raw=True, flatten_tool_images=True
        )
    else:
        assistant_turn = [{"role": "assistant", "message_id": "m1", "content": content}]

    return await transform_messages_to_input(
        pipe,
        [{"role": "user", "content": "q1"}, *assistant_turn, {"role": "user", "content": "q2"}],
        chat_id="c1", openwebui_model_id="owui", artifact_loader=loader, model_id=MODEL, valves=valves,
        replayed_reasoning_refs=refs, ask_user_names=ask_user_names,
    )


def _has_consecutive_reasoning(items):
    positions = [i for i, item in enumerate(items) if item.get("type") == "reasoning"]
    return any(b == a + 1 for a, b in zip(positions, positions[1:]))


# --- the replay is a shape Anthropic accepted --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("cards", [True, False], ids=["cards-on", "cards-off"])
@pytest.mark.parametrize(
    ("rounds", "group", "arm"),
    [
        # The round is replayed where it happened, text before the reasoning and calls of its round, and parallel
        # calls grouped. With cards on it comes back in Open WebUI's saved message; with cards off from the pipe's
        # own storage -- the same turn either way. Every arm is a live-measured 200.
        (SEQUENTIAL_TWO_ROUNDS, "sequential_rounds", "V8_STUB"),
        (PARALLEL_WITH_FINAL_REASONING, "parallel_calls_with_final_round_reasoning", "REC_GROUPED"),
    ],
    ids=["two-sequential-rounds", "parallel-calls-then-reasoning"],
)
async def test_an_unretained_tool_turn_replays_as_a_shape_anthropic_accepted(
    monkeypatch, pipe_instance_async, rounds, group, arm, cards
):
    pipe = pipe_instance_async
    valves = _valves(pipe, SHOW_TOOL_CARDS=cards)
    accepted = _recorded(group, arm)
    assert accepted["status"] == 200

    content, persisted, emitted = await _stage_a(pipe, monkeypatch, valves, rounds)
    replay = await _stage_b(pipe, valves, content, persisted, recorded=_recorded_output(emitted))

    assert _shape(replay) == accepted["shape"]
    assert _shape(replay) != _recorded("sequential_rounds", "T52")["shape"]

# --- every default-settings path keeps the rounds apart ----------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("rounds", [1, 2])
@pytest.mark.parametrize(
    ("stream", "cards"), [(True, False), (False, True), (False, False)],
    ids=["streaming-cards-off", "not-streaming-cards-on", "not-streaming-cards-off"],
)
async def test_every_executed_round_keeps_its_structure_when_no_card_holds_it(
    monkeypatch, pipe_instance_async, rounds, stream, cards
):
    # Cards are only shown while streaming, so a non-streaming turn with the cards setting on still needs the skeleton.
    pipe = pipe_instance_async
    valves = _valves(pipe, SHOW_TOOL_CARDS=cards)
    turn = [("calls", [f"toolu-{i}"]) for i in range(rounds)] + [("answer", True)]

    content, persisted, emitted = await _stage_a(pipe, monkeypatch, valves, turn, stream=stream)
    replay = await _stage_b(pipe, valves, content, persisted, recorded=_recorded_output(emitted))

    assert not _has_consecutive_reasoning(replay), _shape(replay)
    calls = [item.get("call_id") for item in replay if item.get("type") == "function_call"]
    results = [item.get("call_id") for item in replay if item.get("type") == "function_call_output"]
    assert calls == results == [f"toolu-{i}" for i in range(rounds)]


# --- what the copy holds, and where it never goes -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_skeleton_is_never_published_as_turn_output(monkeypatch, pipe_instance_async):
    # A published call without its result is one Open WebUI would run again.
    pipe = pipe_instance_async
    valves = _valves(pipe)

    _, persisted, emitted = await _stage_a(pipe, monkeypatch, valves, SEQUENTIAL_TWO_ROUNDS, stream=False)
    assert [p.get("type") for p in persisted.values()].count("function_call") == 2, persisted

    published = [
        item
        for event in emitted if event.get("type") == "response.completed"
        for item in (event.get("response") or {}).get("output") or []
    ]
    assert [item.get("type") for item in published if item.get("type") in ("function_call", "function_call_output")] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_status", ["completed", "incomplete"])
async def test_a_skeleton_result_keeps_the_real_status(monkeypatch, pipe_instance_async, tool_status):
    pipe = pipe_instance_async
    valves = _valves(pipe)

    _, persisted, emitted = await _stage_a(
        pipe, monkeypatch, valves, [("calls", ["toolu-1"]), ("answer", True)], tool_status=tool_status,
        stream=False,
    )

    results = [p for p in persisted.values() if p.get("type") == "function_call_output"]
    assert [r.get("status") for r in results] == [tool_status]



@pytest.mark.asyncio
@pytest.mark.parametrize(("retention", "warns"), [("disabled", False), ("conversation", True)])
async def test_a_request_without_a_message_id_warns_only_when_it_had_something_to_keep(
    monkeypatch, pipe_instance_async, caplog, retention, warns
):
    """A request without a message id, as an API call sends it, cannot store anything. With reasoning not kept it had
    nothing to store, so a warning that its artifacts were skipped would be false."""
    pipe = pipe_instance_async
    valves = _valves(pipe, PERSIST_REASONING_TOKENS=retention)
    caplog.set_level(logging.WARNING)

    content, persisted, emitted = await _stage_a(
        pipe, monkeypatch, valves, SEQUENTIAL_TWO_ROUNDS, message_id=None, real_row_builder=True
    )

    assert "text 3" in content
    assert persisted == {}
    assert any("missing message_id" in record.getMessage() for record in caplog.records) is warns


# --- every stage that drops reasoning keeps the rounds ----------------------------------------------------------------


async def _outgoing_body(pipe, valves, content, persisted, *, recorded=None):
    """The next turn's request as the streaming loop holds it: the real replay, then the real sanitizer."""
    body = ResponsesBody(
        model=MODEL, input=await _stage_b(pipe, valves, content, persisted, recorded=recorded), stream=True
    )
    _sanitize_request_input(pipe, body)
    return body


def _internal_keys(items: list[Any]) -> list[str]:
    return [str(key) for item in items if isinstance(item, dict) for key in item if str(key).startswith("_anchor")]


@pytest.mark.asyncio
@pytest.mark.parametrize("cards", [True, False], ids=["cards-on", "cards-off"])
@pytest.mark.parametrize(
    ("rounds", "group", "arm"),
    [
        # The round is replayed where it happened, text before the reasoning and calls of its round, and parallel
        # calls grouped. With cards on it comes back in Open WebUI's saved message; with cards off from the pipe's
        # own storage -- the same turn either way. Every arm is a live-measured 200.
        (SEQUENTIAL_TWO_ROUNDS, "sequential_rounds", "V8_STUB"),
        (PARALLEL_WITH_FINAL_REASONING, "parallel_calls_with_final_round_reasoning", "REC_GROUPED"),
    ],
    ids=["two-sequential-rounds", "parallel-calls-then-reasoning"],
)
async def test_the_responses_request_on_the_wire_is_the_accepted_shape_with_no_internal_keys(
    monkeypatch, pipe_instance_async, rounds, group, arm, cards
):
    pipe = pipe_instance_async
    valves = _valves(pipe, SHOW_TOOL_CARDS=cards)
    content, persisted, emitted = await _stage_a(pipe, monkeypatch, valves, rounds)
    body = await _outgoing_body(pipe, valves, content, persisted, recorded=_recorded_output(emitted))

    wire = _filter_openrouter_request(body.model_dump(exclude_none=True))

    assert _shape(wire["input"]) == _recorded(group, arm)["shape"]
    assert _internal_keys(wire["input"]) == []

# --- a Continue must still end on something the model accepts ----------------------------------------------------

# A Continue sends the history stopping INSIDE the turn being written, so there is no trailing user message to
# close the region. Removing that turn's skeleton rounds leaves the request ending on an assistant message.
# Measured live 2026-09-21 on claude-opus-5, claude-sonnet-5, claude-fable-5.1, claude-opus-4.8 and
# claude-sonnet-4.6: that shape is 400 "This model does not support assistant message prefill. The conversation
# must end with a user message.", while the same history WITH its rounds kept is 200 on all five. The 4.5
# generation accepts both. Every existing strip test feeds a list closed by a user message, which is why none of
# them saw this.
def _continue_shaped_items():
    from open_webui_openrouter_pipe.core.utils import TOOL_ROUND_SKELETON_KEY

    return [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Part one."}]},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}",
         TOOL_ROUND_SKELETON_KEY: True},
        {"type": "function_call_output", "call_id": "c1", "output": "[tool result not retained]",
         TOOL_ROUND_SKELETON_KEY: True},
    ]


def _ends_on(items):
    last = items[-1]
    return f"{last['type']}:{last.get('role', '')}".rstrip(":")


def test_the_unsigned_strip_leaves_a_continue_ending_on_a_tool_result():
    from open_webui_openrouter_pipe.requests.sanitizer import _strip_unreplayable_anthropic_reasoning

    items = _continue_shaped_items()
    items[1] = {**items[1], "reasoning_details": [{"type": "reasoning.text", "text": "unsigned"}]}

    out = _strip_unreplayable_anthropic_reasoning(items)

    assert _ends_on(out) != "message:assistant", [_ends_on(out), out]
    assert _ends_on(out) == "function_call_output", _ends_on(out)


def test_the_signature_retry_leaves_a_continue_ending_on_a_tool_result():
    from open_webui_openrouter_pipe.api.transforms import ResponsesBody

    body = ResponsesBody(model="anthropic/claude-opus-4.8", input=_continue_shaped_items())
    assert isinstance(body.input, list)
    body.input[1] = {**body.input[1],
                     "reasoning_details": [{"type": "reasoning.encrypted", "data": "SIGNED", "format": "x"}]}

    Pipe()._ensure_reasoning_config_manager()._strip_replayed_reasoning(body)

    assert isinstance(body.input, list)
    assert _ends_on(body.input) != "message:assistant", [_ends_on(body.input), body.input]
    assert _ends_on(body.input) == "function_call_output", _ends_on(body.input)
