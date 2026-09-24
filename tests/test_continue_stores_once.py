"""A continued answer must be stored with each earlier item exactly once, on every Open WebUI the pipe supports.

Open WebUI stores what it folds together from the pipe's events, not what the pipe publishes, so these tests fold the
published `response.completed` output the way Open WebUI's middleware does and check the stored array:

- A continued turn (the request carries `assistant_message_id`) sets the message's stored output aside as
  `prior_output`, lets `response.completed` replace `output`, and saves `prior_output + output`. Open WebUI keeps
  the stored items itself, so the pipe must not republish them.
- A tool-loop re-call sets everything accumulated so far aside (`full_output()`), streams the re-call into a fresh
  `output`, and puts the set-aside items back in front. Such a re-call carries no `assistant_message_id`, because
  the frontend sends that only when continuing, so it is on that path that the pipe reads the stored output.

The read finds nothing there, because Open WebUI writes a message's output once, when the message finishes - measured
on a live server across a two-round turn, where the row held no output at all until it was marked done. A stored
output with no `assistant_message_id` reaches the pipe only from a client posting to the completions endpoint itself,
and for that caller republishing is right, so the read simply seeds whatever it finds.
"""

from __future__ import annotations

import __future__
import ast
import copy
import importlib.metadata
import json
import sysconfig
from pathlib import Path
from typing import Any, cast

import pytest

import open_webui_openrouter_pipe.streaming.streaming_core as streaming_core_mod
from open_webui_openrouter_pipe import Pipe, ResponsesBody, generate_item_id
from open_webui_openrouter_pipe.requests.transformer import transform_messages_to_input
from tests.test_reasoning_native_items import _events_of, _install_clock, _make_timed_stream

STORED = [
    {"type": "function_call", "id": "fc-old", "call_id": "old-1", "name": "lookup", "arguments": "{}",
     "status": "completed"},
    {"type": "function_call_output", "id": "fco-old", "call_id": "old-1",
     "output": [{"type": "input_text", "text": "x"}], "status": "completed"},
    {"type": "message", "id": "msg-old", "role": "assistant", "status": "completed",
     "content": [{"type": "output_text", "text": "Part one."}]},
]
STORED_IDS = ["fc-old", "fco-old", "msg-old"]


def _select_open_webui(monkeypatch, *, stored: list[dict[str, Any]]) -> None:
    class _Chats:
        @staticmethod
        async def get_message_by_id_and_message_id(_chat_id, _message_id):
            return {"output": copy.deepcopy(stored)}

    monkeypatch.setattr(streaming_core_mod, "Chats", _Chats)


def _user(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _assistant(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _answer_steps(text: str) -> list[tuple[float, dict[str, Any]]]:
    return [
        (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-1"}}),
        (1.0, {"type": "response.reasoning_text.delta", "item_id": "rs-1", "delta": "Thinking. "}),
        (0.2, {"type": "response.output_text.delta", "delta": text}),
        (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
    ]


async def _published(
    pipe, monkeypatch, *, continued: bool, steps, body_input, open_webui_runs_tools=False, on_event=None,
    handing_back=False,
):
    clock = _install_clock(monkeypatch)
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _make_timed_stream(steps, clock))
    emitted: list[dict[str, Any]] = []

    async def emitter(event):
        emitted.append(event)
        if on_event is not None:
            on_event(event)

    metadata: dict[str, Any] = {"model": {"id": "test"}, "chat_id": "c-1", "message_id": "m-1"}
    if continued:
        metadata["assistant_message_id"] = "m-1"
    valves = pipe.valves.model_copy(
        update={
            "THINKING_OUTPUT_MODE": "open_webui",
            "TOOL_EXECUTION_MODE": "Open-WebUI" if open_webui_runs_tools else "Pipeline",
        }
    )
    await pipe._streaming_handler._run_streaming_loop(
        ResponsesBody(model="test/model", input=body_input, stream=True),
        valves,
        emitter,
        metadata=metadata,
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )
    completions = _events_of(emitted, "response.completed")
    if handing_back:
        assert not completions, "a response that hands its calls to Open WebUI published a closing record"
        return emitted
    assert completions, "the turn published no terminal output"
    return (completions[-1].get("response") or {}).get("output") or []


def _stored_after_one_call(existing, published, *, continued: bool):
    if continued:
        prior_output, output = list(existing), []
    else:
        prior_output, output = [], list(existing)
    output = published or output
    return prior_output + output


def _held_after_hand_back(events):
    """What Open WebUI holds after a response that handed it calls: the items it folded in from the stream, with the
    calls it built from the tool-call chunks settled once the stream ended."""
    from tests.test_open_webui_mode_keeps_the_calls_it_runs import _open_webui_backend

    return _open_webui_backend(events)


def _stored_after_a_tool_round(existing, hand_back_events, tool_results, second_published):
    return list(existing) + _held_after_hand_back(hand_back_events) + tool_results + (second_published or [])


def _texts(items) -> list[str]:
    return [
        str(part.get("text"))
        for item in items
        if item.get("type") == "message"
        for part in item.get("content") or []
        if isinstance(part, dict)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("continued", [True, False], ids=["continue", "no-continue"])
async def test_the_stored_turn_holds_each_earlier_item_exactly_once(monkeypatch, pipe_instance_async, continued):
    _select_open_webui(monkeypatch, stored=STORED)
    published = await _published(
        pipe_instance_async, monkeypatch, continued=continued, steps=_answer_steps("Part two."),
        body_input=[_user("hi"), _assistant("Part one.")],
    )

    stored = _stored_after_one_call(STORED, published, continued=continued)
    ids = [item.get("id") for item in stored]

    assert sorted(i for i in ids if i in STORED_IDS) == sorted(STORED_IDS), ids
    assert len(ids) == len(set(ids)), ids
    assert "Part two." in "".join(_texts(stored))


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_tool_round", [False, True], ids=["answer-only", "after-a-stored-tool-round"])
async def test_an_open_webui_tool_round_inside_a_continued_turn_stores_the_earlier_answer_once(
    monkeypatch, pipe_instance_async, stored_tool_round
):
    stored_answer = list(STORED) if stored_tool_round else [STORED[2]]
    stored_ids = [item["id"] for item in stored_answer]
    _select_open_webui(monkeypatch, stored=stored_answer)
    history = [_user("hi")]
    if stored_tool_round:
        history += [
            {"type": "function_call", "call_id": "old-1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "old-1", "output": "x"},
        ]
    history.append(_assistant("Part one."))
    call = {"type": "function_call", "id": "fc_A", "call_id": "call_A", "name": "lookup", "arguments": "{}",
            "status": "completed"}
    first = await _published(
        pipe_instance_async, monkeypatch, continued=True, open_webui_runs_tools=True, handing_back=True,
        steps=[
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-1"}}),
            (1.0, {"type": "response.reasoning_text.delta", "item_id": "rs-1", "delta": "Thinking. "}),
            (0.2, {"type": "response.output_item.done", "item": call}),
            (0.0, {"type": "response.completed", "response": {"output": [call], "usage": {}}}),
        ],
        body_input=history,
    )
    result = {"type": "function_call_output", "id": "fco_A", "call_id": "call_A",
              "output": [{"type": "input_text", "text": "ok"}], "status": "completed"}
    second = await _published(
        pipe_instance_async, monkeypatch, continued=True, open_webui_runs_tools=True,
        steps=_answer_steps("Part two."),
        body_input=history + [
            {"type": "function_call", "call_id": "call_A", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_A", "output": "ok"},
        ],
    )

    stored = _stored_after_a_tool_round(stored_answer, first, [result], second)
    ids = [item.get("id") for item in stored]

    assert sorted(i for i in ids if i in stored_ids) == sorted(stored_ids), ids
    assert _texts(stored).count("Part one.") == 1, _texts(stored)
    assert "Part two." in "".join(_texts(stored))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_id", ["call_B", "old-1"], ids=["a-fresh-call-id", "a-call-id-the-stored-round-already-used"]
)
async def test_a_tool_round_that_reuses_a_stored_call_id_still_stores_the_continued_answer_once(
    monkeypatch, pipe_instance_async, call_id
):
    """A re-call is recognised by how many results it brings, not by whether their ids are new.

    Tool call ids can repeat: until the pipe made its ids unique, both chat-completions adapters minted
    `toolcall-{model}-{index}` whenever the provider sent a tool-call delta without one, so the first call of
    every request over that transport carried the same id, and chats saved then still hold those ids.
    Deciding "Open WebUI is already holding this round" by asking whether the turn's ids appear in the
    stored output therefore answers yes for a round Open WebUI has only just run, and the pipe republishes
    the stored answer that Open WebUI then puts back in front itself.

    The two arms differ only in the id, so a gate that counts results passes both and a gate that tests
    membership passes only the first.
    """
    _select_open_webui(monkeypatch, stored=STORED)
    history = [
        _user("hi"),
        {"type": "function_call", "call_id": "old-1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "old-1", "output": "x"},
        _assistant("Part one."),
    ]
    call = {"type": "function_call", "id": "fc_B", "call_id": call_id, "name": "lookup", "arguments": "{}",
            "status": "completed"}
    first = await _published(
        pipe_instance_async, monkeypatch, continued=True, open_webui_runs_tools=True, handing_back=True,
        steps=[
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-B"}}),
            (1.0, {"type": "response.reasoning_text.delta", "item_id": "rs-B", "delta": "Thinking. "}),
            (0.2, {"type": "response.output_item.done", "item": call}),
            (0.0, {"type": "response.completed", "response": {"output": [call], "usage": {}}}),
        ],
        body_input=history,
    )
    result = {"type": "function_call_output", "id": "fco_B", "call_id": call_id,
              "output": [{"type": "input_text", "text": "ok"}], "status": "completed"}
    second = await _published(
        pipe_instance_async, monkeypatch, continued=True, open_webui_runs_tools=True,
        steps=_answer_steps("Part two."),
        body_input=history + [
            {"type": "function_call", "call_id": call_id, "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": call_id, "output": "ok"},
        ],
    )

    stored = _stored_after_a_tool_round(STORED, first, [result], second)

    assert _texts(stored).count("Part one.") == 1, _texts(stored)
    assert "Part two." in "".join(_texts(stored))


def _open_webui_convert_output_to_messages():
    """Open WebUI's own `convert_output_to_messages`, compiled from the installed source with the helpers it calls.

    Importing `open_webui.utils.misc` pulls in Open WebUI's configuration, so only these functions are compiled, in a
    namespace of their own; nothing is added to `sys.modules`.
    """
    source = Path(sysconfig.get_paths()["purelib"]) / "open_webui" / "utils" / "misc.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    wanted = {"convert_output_to_messages", "reconcile_tool_pairs", "get_content_from_message", "get_output_text"}
    nodes: list[ast.stmt] = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert sorted(node.name for node in nodes if isinstance(node, ast.FunctionDef)) == sorted(wanted)
    namespace: dict[str, Any] = {"json": json}
    code = compile(
        ast.Module(body=nodes, type_ignores=[]),
        str(source),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, namespace)
    return namespace["convert_output_to_messages"]


_ONE_PIXEL_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.mark.asyncio
async def test_a_pipeline_turn_replaying_its_own_tool_results_still_republishes_the_stored_answer(
    monkeypatch, pipe_instance_async
):
    # Not a re-call: the results came back from the pipe's own rows, so Open WebUI does not hold them and replaces
    # the stored output with whatever this call publishes. The stored answer has to go back out with it.
    _select_open_webui(monkeypatch, stored=[STORED[2]])

    published = await _published(
        pipe_instance_async, monkeypatch, continued=False,
        steps=_answer_steps("Part two."),
        body_input=[
            _user("hi"),
            {"type": "function_call", "call_id": "call_A", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_A", "output": "ok"},
        ],
    )

    assert _texts(published).count("Part one.") == 1, _texts(published)
    assert "Part two." in "".join(_texts(published))


async def _request_input(pipe, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return await transform_messages_to_input(
        pipe, messages, chat_id="c-1", openwebui_model_id="test", model_id="test/model", valves=pipe.valves
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["text", "image"])
async def test_a_tool_round_that_returns_an_image_stores_the_continued_answer_once(
    monkeypatch, pipe_instance_async, result
):
    """For its re-call Open WebUI rebuilds the round with its own converter, which moves a tool result's images into
    a user message after the tool messages. That message belongs to the round; it does not start a new turn."""
    pipe = pipe_instance_async
    convert_output_to_messages = _open_webui_convert_output_to_messages()
    stored_answer = [STORED[2]]
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Part one."}]
    _select_open_webui(monkeypatch, stored=stored_answer)
    call = {"type": "function_call", "id": "fc_A", "call_id": "call_A", "name": "lookup", "arguments": "{}",
            "status": "completed"}
    first = await _published(
        pipe, monkeypatch, continued=True, open_webui_runs_tools=True, handing_back=True,
        steps=[
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-1"}}),
            (1.0, {"type": "response.reasoning_text.delta", "item_id": "rs-1", "delta": "Thinking. "}),
            (0.2, {"type": "response.output_item.done", "item": call}),
            (0.0, {"type": "response.completed", "response": {"output": [call], "usage": {}}}),
        ],
        body_input=await _request_input(pipe, history),
    )
    parts: list[dict[str, Any]] = [{"type": "input_text", "text": "ok"}]
    if result == "image":
        parts.append({"type": "input_image", "image_url": _ONE_PIXEL_PNG})
    # Open WebUI built its own call item from the streamed tool-call chunks and settled it once the stream ended; it
    # then appends the tool's result and rebuilds the round for its re-call from what it holds.
    held = _held_after_hand_back(first)
    assert [(i.get("call_id"), i.get("status")) for i in held if i.get("type") == "function_call"] == [
        ("call_A", "completed")
    ], held
    round_items: list[dict[str, Any]] = [
        {"type": "function_call_output", "id": "fco_A", "call_id": "call_A", "output": parts, "status": "completed"},
    ]
    re_call = history + convert_output_to_messages(held + round_items, raw=True, flatten_tool_images=True)
    assert [message["role"] for message in re_call[len(history):] if message["role"] != "assistant" or message.get("tool_calls")] == (
        ["assistant", "tool", "user"] if result == "image" else ["assistant", "tool"]
    ), re_call
    second = await _published(
        pipe, monkeypatch, continued=True, open_webui_runs_tools=True, steps=_answer_steps("Part two."),
        body_input=await _request_input(pipe, re_call),
    )

    stored = _stored_after_a_tool_round(stored_answer, first, round_items, second)

    assert _texts(stored).count("Part one.") == 1, _texts(stored)
    assert "Part two." in "".join(_texts(stored))


@pytest.mark.asyncio
async def test_a_continue_after_a_turn_that_ended_on_a_tool_result_keeps_its_stored_answer(
    monkeypatch, pipe_instance_async
):
    """A user message straight after a tool result starts a new turn whenever anything follows it in the request."""
    stored_answer = [STORED[2]]
    _select_open_webui(monkeypatch, stored=stored_answer)
    published = await _published(
        pipe_instance_async, monkeypatch, continued=True, open_webui_runs_tools=True,
        steps=_answer_steps("Part two."),
        body_input=[
            _user("earlier question"),
            {"type": "function_call", "call_id": "call-earlier", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call-earlier", "output": "earlier result"},
            _user("hi"),
            _assistant("Part one."),
        ],
    )

    stored = _stored_after_one_call(stored_answer, published, continued=True)

    assert _texts(stored).count("Part one.") == 1, _texts(stored)
    assert "Part two." in "".join(_texts(stored))


async def _pipeline_generation_with_a_persisted_tool_round(pipe, monkeypatch, persisted: dict[str, dict]):
    """The message's first generation: the pipe runs one tool round with results persisted and tool cards off.

    Returns what Open WebUI stores for this fresh turn and the content, which carries the hidden markers.
    """
    rounds = iter([
        [
            {"type": "response.output_item.done", "item": {
                "id": "rs-a", "type": "reasoning", "status": "completed",
                "content": [{"type": "reasoning_text", "text": "Looking it up."}], "summary": [], "signature": "SIG-A"}},
            {"type": "response.output_item.done", "item": {
                "type": "function_call", "call_id": "call-X", "name": "lookup", "arguments": "{}",
                "status": "completed"}},
            {"type": "response.completed", "response": {"output": [
                {"type": "function_call", "call_id": "call-X", "name": "lookup", "arguments": "{}"}], "usage": {}}},
        ],
        [
            {"type": "response.output_item.done", "item": {
                "id": "rs-b", "type": "reasoning", "status": "completed",
                "content": [{"type": "reasoning_text", "text": "Answering."}], "summary": [], "signature": "SIG-B"}},
            {"type": "response.output_text.delta", "delta": "Part one."},
            {"type": "response.completed", "response": {"output": [], "usage": {}}},
        ],
    ])

    async def streaming(self, session, request_body, **_kwargs):
        for event in next(rounds):
            yield event

    async def run_tools(calls, _registry):
        return [
            {"type": "function_call_output", "call_id": call.get("call_id"), "output": "lookup result",
             "status": "completed"}
            for call in calls
        ]

    async def persist(rows):
        ulids = [generate_item_id() for _ in rows]
        persisted.update(zip(ulids, (row["payload"] for row in rows)))
        return ulids

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", streaming)
    monkeypatch.setattr(pipe._ensure_tool_executor(), "_execute_function_calls", run_tools)
    monkeypatch.setattr(pipe._artifact_store, "_make_db_row", lambda _c, _m, _model, payload: {"payload": payload})
    monkeypatch.setattr(pipe._artifact_store, "_db_persist", persist)
    emitted: list[dict[str, Any]] = []

    async def emitter(event):
        emitted.append(event)

    valves = pipe.valves.model_copy(
        update={
            "THINKING_OUTPUT_MODE": "open_webui",
            "TOOL_EXECUTION_MODE": "Pipeline",
            "PERSIST_TOOL_RESULTS": True,
            "SHOW_TOOL_CARDS": False,
            "PERSIST_REASONING_TOKENS": "conversation",
            "MAX_FUNCTION_CALL_LOOPS": 3,
        }
    )
    content = await pipe._streaming_handler._run_streaming_loop(
        ResponsesBody(model="test/model", input=[_user("hi")], stream=True),
        valves,
        emitter,
        metadata={"model": {"id": "test"}, "chat_id": "c-1", "message_id": "m-1"},
        tools={"lookup": {"callable": lambda **_kwargs: "lookup result"}},
        session=cast(Any, object()),
        user_id="user-123",
    )
    completions = _events_of(emitted, "response.completed")
    assert completions, "the first generation published no terminal output"
    return (completions[-1].get("response") or {}).get("output") or [], content


@pytest.mark.asyncio
async def test_a_continue_keeps_the_stored_answer_when_the_pipes_own_tool_results_replay_into_the_request(
    monkeypatch, pipe_instance_async
):
    """A Continue replaces the stored output with what the pipe publishes, so the pipe must still
    publish the stored answer when the request carries tool results the stored output lacks because they came back
    from the pipe's own rows rather than from an Open WebUI tool round."""
    pipe = pipe_instance_async
    persisted: dict[str, dict] = {}
    _select_open_webui(monkeypatch, stored=[])
    stored, content = await _pipeline_generation_with_a_persisted_tool_round(pipe, monkeypatch, persisted)

    async def loader(_chat_id, _message_id, ulids):
        return {ulid: persisted[ulid] for ulid in ulids if ulid in persisted}

    continue_input = await transform_messages_to_input(
        pipe,
        [{"role": "user", "content": "hi"}, {"role": "assistant", "message_id": "m-1", "content": content}],
        chat_id="c-1",
        openwebui_model_id="test",
        artifact_loader=loader,
        model_id="test/model",
        valves=pipe.valves,
    )
    # Cards are off, so the round is hidden from the user and not saved in the message; it comes back from the
    # pipe's own rows instead. That is exactly the case this test exists for.
    assert not [item for item in stored if item.get("type") in ("function_call", "function_call_output")], stored
    assert "call-X" in {
        item.get("call_id") for item in continue_input if item.get("type") == "function_call_output"
    }, continue_input

    _select_open_webui(monkeypatch, stored=stored)
    published = await _published(
        pipe, monkeypatch, continued=True, steps=_answer_steps("Part two."), body_input=continue_input
    )

    after = _stored_after_one_call(stored, published, continued=True)
    stored_ids = [item.get("id") for item in stored]
    after_ids = [item.get("id") for item in after]
    assert sorted(i for i in after_ids if i in stored_ids) == sorted(stored_ids), after_ids
    assert len(after_ids) == len(set(after_ids)), after_ids
    assert "".join(_texts(after)).count("Part one.") == 1, _texts(after)
    assert "Part two." in "".join(_texts(after))


@pytest.mark.asyncio
async def test_a_continued_turn_leaves_a_stranded_call_as_open_webui_stored_it(monkeypatch, pipe_instance_async):
    # A Stop during a tool call leaves the call in_progress. Open WebUI keeps the stored items itself on a
    # continue, so the pipe republishes none of them and the call stays exactly as stored: parity with Open WebUI,
    # chosen over a repair that events cannot make without duplicating the item.
    stranded = [dict(STORED[0], status="in_progress"), STORED[1], STORED[2]]
    _select_open_webui(monkeypatch, stored=stranded)
    published = await _published(
        pipe_instance_async, monkeypatch, continued=True, steps=_answer_steps("Part two."),
        body_input=[_user("hi"), _assistant("Part one.")],
    )

    stored = _stored_after_one_call(stranded, published, continued=True)

    assert not {item.get("id") for item in published} & set(STORED_IDS)
    assert [item.get("status") for item in stored if item.get("id") == "fc-old"] == ["in_progress"]
