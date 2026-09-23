"""A continued message replays each generation's reasoning in order, and never two blocks side by side.

Continue adds a second generation to the same assistant message, and the next request replays the whole message as one
turn. Replay rebuilds the message as each generation's text followed by that generation's persisted items, and places
each reasoning block by an ordinal counted across the turn: the tool call it came before or after, or, with no call
beside it, the text chunk it came before. The second generation is streamed by a new request, so its ordinals must
continue from what the first generation left in the turn. Anthropic rejects thinking blocks that sit next to each
other in the replay without having been generated next to each other ("thinking blocks ... cannot be modified").

Both generations run through the real streaming loop with their items persisted, and both the Continue request's input
and the next turn's replay come from the real transformer. Each shape runs with the upstream `response.completed`
listing the round's reasoning, which orders it, and without, which leaves the stream positions to order it.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any, cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from open_webui_openrouter_pipe import Pipe, ResponsesBody, generate_item_id
from open_webui_openrouter_pipe.requests.transformer import transform_messages_to_input
from tests.test_continue_stores_once import _open_webui_convert_output_to_messages
from tests.test_storage import _install_internal_db

MODEL = "anthropic/claude-opus-4.8"
LABELS = ("BEFORE-A", "AFTER-A", "BEFORE-B", "AFTER-B", "THINK-A", "THINK-B", "THINK-0")

SHAPES = {
    "tools-then-tools": (
        [("call", "BEFORE-A", "call-a"), ("answer", "AFTER-A", "Part one.")],
        [("call", "BEFORE-B", "call-b"), ("answer", "AFTER-B", "Part two.")],
        ["user:q1", "BEFORE-A", "call-a", "result:call-a", "AFTER-A", "assistant:Part one.",
         "BEFORE-B", "call-b", "result:call-b", "AFTER-B", "assistant:Part two.", "user:q2"],
    ),
    "answer-then-answer": (
        [("answer", "THINK-A", "Part one.")],
        [("answer", "THINK-B", "Part two.")],
        ["user:q1", "THINK-A", "assistant:Part one.", "THINK-B", "assistant:Part two.", "user:q2"],
    ),
    "tools-then-answer": (
        [("call", "BEFORE-A", "call-a"), ("answer", "AFTER-A", "Part one.")],
        [("answer", "THINK-B", "Part two.")],
        ["user:q1", "BEFORE-A", "call-a", "result:call-a", "AFTER-A", "assistant:Part one.",
         "THINK-B", "assistant:Part two.", "user:q2"],
    ),
    "answer-then-tools": (
        [("answer", "THINK-A", "Part one.")],
        [("call", "BEFORE-B", "call-b"), ("answer", "AFTER-B", "Part two.")],
        ["user:q1", "THINK-A", "assistant:Part one.", "BEFORE-B", "call-b", "result:call-b", "AFTER-B",
         "assistant:Part two.", "user:q2"],
    ),
    # The first generation stopped after thinking, before any text (Stop during thinking, or a length cut), so the
    # continued turn ends on reasoning: the continuation's thinking goes after its own text, never beside THINK-A.
    "thinking-then-answer": (
        [("answer", "THINK-A", "")],
        [("answer", "THINK-B", "Part two.")],
        ["user:q1", "THINK-A", "assistant:Part two.", "THINK-B", "user:q2"],
    ),
}


def _valves(pipe):
    return pipe.valves.model_copy(
        update={
            "TOOL_EXECUTION_MODE": "Pipeline",
            "PERSIST_REASONING_TOKENS": "conversation",
            "PERSIST_TOOL_RESULTS": True,
            "SHOW_TOOL_CARDS": False,
            "MAX_FUNCTION_CALL_LOOPS": 4,
        }
    )


def _reasoning(label: str) -> dict[str, Any]:
    return {
        "type": "reasoning",
        "id": f"rs-{label}",
        "status": "completed",
        "content": [{"type": "reasoning_text", "text": label}],
        "summary": [],
        "signature": f"sig-{label}",
    }


async def _generation(
    pipe, monkeypatch, valves, persisted, *, rounds, body_input, listing, continued, message_id="m1", refs=None,
    stream=True, emitter=None,
) -> str:
    """One request, streamed unless `stream` is False; each entry of `rounds` is one model response within it.

    ("call", label, call_id) reasons, then calls a tool. ("answer", label, text) reasons, then answers.
    ("answer-beside-an-unlisted-call", label, text) also streams a call that its completion never lists.
    ("search-then-call", label, call_id) reasons, has OpenRouter run a web search (item "ws-<call_id>"), then calls.
    Returns the content the loop produced: its text plus the hidden markers of what it persisted.
    With `persisted` None the rows go to the pipe's own artifact store; `refs` are the cleanup refs the transformer
    collected for this request, carried on the body the way `ResponsesBody.from_completions` carries them.
    """
    step = [0]

    async def streaming(self, session, request_body, **_kwargs):
        kind, label, value = rounds[step[0]]
        step[0] += 1
        reasoning = _reasoning(label)
        yield {"type": "response.output_item.done", "item": reasoning}
        searched: list[dict[str, Any]] = []
        if kind == "search-then-call":
            search = {"type": "openrouter:web_search", "id": f"ws-{value}", "status": "completed", "action": {"sources": []}}
            yield {"type": "response.output_item.done", "item": search}
            searched = [search]
        if kind in ("call", "search-then-call"):
            call = {"type": "function_call", "call_id": value, "name": "lookup", "arguments": "{}",
                    "status": "completed"}
            yield {"type": "response.output_item.done", "item": call}
            listed = [reasoning, *searched, call] if listing == "with-reasoning" else [*searched, call]
        else:
            if kind == "answer-beside-an-unlisted-call":
                yield {"type": "response.output_item.done", "item": {
                    "type": "function_call", "call_id": "call-unlisted", "name": "lookup", "arguments": "{}",
                    "status": "completed"}}
            yield {"type": "response.output_text.delta", "delta": value}
            message = {"type": "message", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": value}]}
            if kind == "answer-beside-an-unlisted-call":
                listed = []
            else:
                listed = [reasoning, message] if listing == "with-reasoning" else [message]
        yield {"type": "response.completed", "response": {"output": listed, "usage": {}}}

    async def run_tools(calls, _registry):
        return [
            {"type": "function_call_output", "call_id": c.get("call_id"), "output": f"result of {c.get('call_id')}",
             "status": "completed"}
            for c in calls
        ]

    def make_row(_chat_id, _message_id, _model_id, payload):
        return {"payload": payload}

    async def persist(rows):
        ulids = [generate_item_id() for _ in rows]
        persisted.update(zip(ulids, (row["payload"] for row in rows)))
        return ulids

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", streaming)
    monkeypatch.setattr(Pipe, "send_openrouter_nonstreaming_request_as_events", streaming)
    monkeypatch.setattr(pipe._ensure_tool_executor(), "_execute_function_calls", run_tools)
    if persisted is not None:
        monkeypatch.setattr(pipe._artifact_store, "_make_db_row", make_row)
        monkeypatch.setattr(pipe._artifact_store, "_db_persist", persist)

    metadata: dict[str, Any] = {"model": {"id": MODEL}, "chat_id": "c1", "message_id": message_id}
    if continued:
        metadata["assistant_message_id"] = message_id
    extra: dict[str, Any] = {} if refs is None else {"_replayed_reasoning_refs": refs}
    handler = pipe._streaming_handler
    runner = handler._run_streaming_loop if stream else handler._run_nonstreaming_loop
    content = await runner(
        ResponsesBody(model=MODEL, input=body_input, stream=stream, **extra),
        valves,
        emitter,
        metadata=metadata,
        tools={"lookup": {"callable": lambda **_k: "ok"}},
        session=cast(Any, object()),
        user_id="u1",
    )
    assert isinstance(content, str)
    return content


async def _replay(pipe, valves, persisted, messages) -> list[dict[str, Any]]:
    async def loader(_chat_id, _message_id, ulids):
        return {u: persisted[u] for u in ulids if u in persisted}

    return await transform_messages_to_input(
        pipe, messages, chat_id="c1", openwebui_model_id="owui", artifact_loader=loader, model_id=MODEL,
        valves=valves,
    )


def _label(item: dict[str, Any]) -> str:
    kind = item.get("type")
    if kind == "reasoning":
        text = json.dumps(item)
        return next((label for label in LABELS if label in text), "reasoning")
    if kind == "function_call":
        return str(item.get("call_id"))
    if kind == "function_call_output":
        return f"result:{item.get('call_id')}"
    if kind == "message":
        parts = item.get("content")
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)) if isinstance(parts, list) else str(parts)
        return f"{item.get('role')}:{text.strip()}"
    return str(kind)


def _published_output(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The output a turn published, which is what Open WebUI stores as the message's record."""
    for event in reversed(events):
        if event.get("type") == "response.completed":
            return (event.get("response") or {}).get("output") or []
    return []


def _open_webui_rebuilds(record: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What `process_messages_with_output` hands the pipe for an assistant message carrying `output`."""
    return _open_webui_convert_output_to_messages()(record, raw=True, flatten_tool_images=True)


def _has_consecutive_reasoning(items: list[dict[str, Any]]) -> bool:
    positions = [i for i, item in enumerate(items) if item.get("type") == "reasoning"]
    return any(b == a + 1 for a, b in zip(positions, positions[1:]))


async def _continued_turn_replay(pipe, monkeypatch, *, first_rounds, second_rounds, listing, earlier=()):
    valves = _valves(pipe)
    persisted: dict[str, dict[str, Any]] = {}
    history = [*earlier, {"role": "user", "content": "q1"}]
    first = await _generation(
        pipe, monkeypatch, valves, persisted, rounds=first_rounds, listing=listing, continued=False,
        body_input=await _replay(pipe, valves, persisted, history),
    )
    continue_input = await _replay(
        pipe, valves, persisted, [*history, {"role": "assistant", "message_id": "m1", "content": first}]
    )
    second = await _generation(
        pipe, monkeypatch, valves, persisted, rounds=second_rounds, listing=listing, continued=True,
        body_input=continue_input,
    )
    return await _replay(
        pipe, valves, persisted,
        [*history, {"role": "assistant", "message_id": "m1", "content": first + second},
         {"role": "user", "content": "q2"}],
    )


def _assert_replays_exactly_and_apart(replay: list[dict[str, Any]], expected: list[str]) -> None:
    order = [_label(item) for item in replay]
    assert not _has_consecutive_reasoning(replay), order
    assert order == expected, order


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "listing", ["with-reasoning", "without-reasoning"],
    ids=["completion-lists-the-reasoning", "completion-omits-the-reasoning"],
)
@pytest.mark.parametrize("shape", list(SHAPES))
async def test_a_continued_message_replays_its_reasoning_in_order_and_never_side_by_side(
    monkeypatch, pipe_instance_async, shape, listing
):
    first_rounds, second_rounds, expected = SHAPES[shape]
    replay = await _continued_turn_replay(
        pipe_instance_async, monkeypatch, first_rounds=first_rounds, second_rounds=second_rounds, listing=listing
    )

    _assert_replays_exactly_and_apart(replay, expected)


@pytest.mark.asyncio
async def test_an_earlier_turns_tool_calls_do_not_shift_a_continued_messages_reasoning(
    monkeypatch, pipe_instance_async
):
    """Replay counts calls within one turn, so only the continued message's own calls may offset the continuation."""
    first_rounds, second_rounds, expected = SHAPES["tools-then-tools"]
    earlier = (
        {"role": "user", "content": "q0"},
        {"role": "assistant", "content": "Earlier answer.",
         "tool_calls": [{"id": "call-0", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-0", "content": "earlier result"},
    )
    replay = await _continued_turn_replay(
        pipe_instance_async, monkeypatch, first_rounds=first_rounds, second_rounds=second_rounds,
        listing="with-reasoning", earlier=earlier,
    )

    _assert_replays_exactly_and_apart(replay, ["user:q0", "assistant:Earlier answer.", "call-0", "result:call-0", *expected])


@pytest.mark.asyncio
async def test_a_continuation_whose_completion_disagrees_with_its_stream_still_keeps_its_reasoning_apart(
    monkeypatch, pipe_instance_async
):
    """A call streamed but never listed at completion leaves stream positions out of step with the calls, so the
    reasoning is placed by the last fallback, which must also count only the continuation's own calls."""
    first_rounds, _, expected = SHAPES["tools-then-answer"]
    replay = await _continued_turn_replay(
        pipe_instance_async, monkeypatch, first_rounds=first_rounds,
        second_rounds=[("answer-beside-an-unlisted-call", "THINK-B", "Part two.")], listing="with-reasoning",
    )

    _assert_replays_exactly_and_apart(replay, expected)


@contextlib.contextmanager
def _real_artifact_store(pipe):
    """The pipe's own artifact store on an in-memory SQLite database its worker threads share."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with _install_internal_db(engine):
        pipe._artifact_store._init_artifact_store(pipe_identifier="pipe", table_fragment="pipe")
    try:
        yield pipe._artifact_store
    finally:
        engine.dispose()


def _rows_of(store, message_id: str, item_type: str | None = None) -> int:
    with store._session_factory() as session:
        query = session.query(store._item_model).filter(store._item_model.message_id == message_id)
        if item_type is not None:
            query = query.filter(store._item_model.item_type == item_type)
        return query.count()


async def _replay_from_store(pipe, valves, messages, refs=None) -> list[dict[str, Any]]:
    return await transform_messages_to_input(
        pipe, messages, chat_id="c1", openwebui_model_id="owui", artifact_loader=pipe._artifact_store._db_fetch,
        model_id=MODEL, valves=valves, replayed_reasoning_refs=refs,
    )


WHOLE_TURN = {
    "tools-then-tools": ["user:q0", "assistant:Earlier answer.", "user:q1", "BEFORE-A", "call-a", "result:call-a",
                         "AFTER-A", "assistant:Part one.", "BEFORE-B", "call-b", "result:call-b", "AFTER-B",
                         "assistant:Part two.", "user:q2"],
    "answer-then-answer": ["user:q0", "assistant:Earlier answer.", "user:q1", "THINK-A", "assistant:Part one.",
                           "THINK-B", "assistant:Part two.", "user:q2"],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", list(WHOLE_TURN))
async def test_next_reply_cleanup_keeps_the_rows_of_the_message_a_continue_is_still_writing(
    monkeypatch, pipe_instance_async, shape
):
    """With reasoning kept only until the next reply, a finished request deletes the rows it replayed, except the rows
    of the message it is still writing: a Continue replays that message's first generation and must leave it for the
    next reply. The messages carry no ids, as Open WebUI sends them, and the rows live in the real artifact store."""
    pipe = pipe_instance_async
    valves = _valves(pipe).model_copy(update={"PERSIST_REASONING_TOKENS": "next_reply", "PERSIST_TOOL_RESULTS": False})
    first_rounds, second_rounds, _ = SHAPES[shape]

    with _real_artifact_store(pipe) as store:

        async def reply(history, rounds, message_id, *, continued):
            refs: list[tuple[str, str]] = []
            body_input = await _replay_from_store(pipe, valves, history, refs)
            return await _generation(
                pipe, monkeypatch, valves, None, rounds=rounds, body_input=body_input, listing="with-reasoning",
                continued=continued, message_id=message_id, refs=refs,
            )

        earlier = await reply([{"role": "user", "content": "q0"}], [("answer", "THINK-0", "Earlier answer.")], "m0",
                              continued=False)
        assert _rows_of(store, "m0") == 1
        history = [{"role": "user", "content": "q0"}, {"role": "assistant", "content": earlier},
                   {"role": "user", "content": "q1"}]

        first = await reply(history, first_rounds, "m1", continued=False)
        # The reply to q1 replayed the earlier message's reasoning, so those rows are gone ...
        assert _rows_of(store, "m0") == 0
        rows_after_the_first_generation = _rows_of(store, "m1")
        assert rows_after_the_first_generation > 0

        second = await reply([*history, {"role": "assistant", "content": first}], second_rounds, "m1", continued=True)
        # ... while the Continue, still writing m1, keeps the rows it replayed from m1's first generation.
        assert _rows_of(store, "m1") > rows_after_the_first_generation

        replay = await _replay_from_store(
            pipe, valves,
            [*history, {"role": "assistant", "content": first + second}, {"role": "user", "content": "q2"}],
        )

    _assert_replays_exactly_and_apart(replay, WHOLE_TURN[shape])



# --- a Continue that is not streamed -----------------------------------------------------------------------------------

# Open WebUI keeps the first generation on a Continue whether or not the request streamed: its non-streaming
# handler rebuilds the stored output, merges the continued message with the new one, and saves
# `previous + response_output`. So a Continue that did not stream replays the whole turn, exactly as a streamed
# one does.


@pytest.mark.asyncio
@pytest.mark.parametrize("results_kept", [True, False], ids=["results-kept", "results-not-kept"])
@pytest.mark.parametrize(
    "listing", ["with-reasoning", "without-reasoning"],
    ids=["completion-lists-the-reasoning", "completion-omits-the-reasoning"],
)
@pytest.mark.parametrize("shape", list(SHAPES))
async def test_a_continue_that_is_not_streamed_replays_the_whole_turn_just_as_a_streamed_one_does(
    monkeypatch, pipe_instance_async, shape, listing, results_kept
):
    pipe = pipe_instance_async
    valves = _valves(pipe).model_copy(update={"PERSIST_TOOL_RESULTS": results_kept})
    first_rounds, second_rounds, expected = SHAPES[shape]
    persisted: dict[str, dict[str, Any]] = {}
    history = [{"role": "user", "content": "q1"}]
    first = await _generation(
        pipe, monkeypatch, valves, persisted, rounds=first_rounds, listing=listing, continued=False, stream=False,
        body_input=await _replay(pipe, valves, persisted, history),
    )
    second = await _generation(
        pipe, monkeypatch, valves, persisted, rounds=second_rounds, listing=listing, continued=True, stream=False,
        body_input=await _replay(
            pipe, valves, persisted, [*history, {"role": "assistant", "message_id": "m1", "content": first}]
        ),
    )

    replay = await _replay(
        pipe, valves, persisted,
        [*history, {"role": "assistant", "message_id": "m1", "content": first + second},
         {"role": "user", "content": "q2"}],
    )

    _assert_replays_exactly_and_apart(replay, expected)


# --- a Continue retried without its replayed thinking ------------------------------------------------------------------

# Chronological, read straight off SHAPES: each generation reasons, calls, reasons again and THEN answers, so its
# text comes after its rounds. It is the same with tool cards on or off: the card switch decides what the user sees,
# never what the model is handed, and the pipe commits each round where it happened.
AFTER_A_THINKING_RETRY = {
    "tools-then-tools": ["user:q1", "BEFORE-A", "call-a", "result:call-a", "AFTER-A", "assistant:Part one.",
                         "BEFORE-B", "call-b", "result:call-b", "AFTER-B", "assistant:Part two.", "user:q2"],
    "tools-then-answer": ["user:q1", "BEFORE-A", "call-a", "result:call-a", "AFTER-A", "assistant:Part one.",
                          "THINK-B", "assistant:Part two.", "user:q2"],
    "thinking-then-answer": ["user:q1", "THINK-A", "assistant:Part two.", "THINK-B", "user:q2"],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("cards", [True, False], ids=["cards-on", "cards-off"])
@pytest.mark.parametrize("shape", list(AFTER_A_THINKING_RETRY))
async def test_a_continue_retried_without_its_replayed_thinking_still_numbers_its_reasoning_after_the_first_generation(
    monkeypatch, pipe_instance_async, shape, cards
):
    """Results are not kept. On a saved chat each generation records its tool rounds and Open WebUI replays them. The
    Continue's first attempt is rejected for a thinking-block signature; the retry drops the replayed thinking before
    streaming again. The continuation must still be numbered after the rounds the message's rows keep."""
    import open_webui_openrouter_pipe.pipe as pipe_mod
    from open_webui_openrouter_pipe import EncryptedStr
    from open_webui_openrouter_pipe.core.errors import _build_openrouter_api_error

    pipe = pipe_instance_async
    valves = _valves(pipe).model_copy(update={"PERSIST_TOOL_RESULTS": False, "SHOW_TOOL_CARDS": cards})
    first_rounds, second_rounds, _ = SHAPES[shape]
    persisted: dict[str, dict[str, Any]] = {}
    history = [{"role": "user", "content": "q1"}]
    first_events: list[dict[str, Any]] = []

    async def capture_first(event):
        first_events.append(event)

    # Both generations of a real Continue run with Open WebUI's emitter on a saved chat, so each RECORDS its
    # round and Open WebUI stores it; neither writes a skeleton. The replays below rebuild the turn from those
    # records with Open WebUI's own converter, as `process_messages_with_output` does before calling the pipe.
    first = await _generation(
        pipe, monkeypatch, valves, persisted, rounds=first_rounds, listing="with-reasoning", continued=False,
        body_input=await _replay(pipe, valves, persisted, history), emitter=capture_first,
    )
    first_record = _published_output(first_events)
    assert any(i.get("type") == "function_call" for i in first_record) is (
        cards and any(step[0] == "call" for step in first_rounds)
    ), (
        f"cards={cards}: the saved message {'lacks' if cards else 'shows'} the tool round: {first_record}"
    )

    attempts: list[int] = []

    async def rejected_once_then_streams(self, session, request_body, **_kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise _build_openrouter_api_error(
                400,
                "Bad Request",
                json.dumps({"error": {"message": "messages.1.content.0: Invalid `signature` in `thinking` block"}}),
                requested_model=MODEL,
            )
        kind, label, value = second_rounds[len(attempts) - 2]
        reasoning = _reasoning(label)
        yield {"type": "response.output_item.done", "item": reasoning}
        if kind == "call":
            call = {"type": "function_call", "call_id": value, "name": "lookup", "arguments": "{}", "status": "completed"}
            yield {"type": "response.output_item.done", "item": call}
            listed = [reasoning, call]
        else:
            yield {"type": "response.output_text.delta", "delta": value}
            listed = [reasoning, {"type": "message", "role": "assistant", "status": "completed",
                                  "content": [{"type": "output_text", "text": value}]}]
        yield {"type": "response.completed", "response": {"output": listed, "usage": {}}}

    async def loaded(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def stored_rows(_chat_id, _message_id, ulids):
        return {ulid: persisted[ulid] for ulid in ulids if ulid in persisted}

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", rejected_once_then_streams)
    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: ("sk-test-key", None))
    monkeypatch.setattr(pipe._artifact_store, "_ensure_artifact_store", lambda *_a, **_k: None)
    monkeypatch.setattr(pipe._artifact_store, "_db_fetch", stored_rows)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", loaded)
    monkeypatch.setattr(
        pipe_mod.OpenRouterModelRegistry,
        "list_models",
        lambda: [{"id": MODEL, "name": "Claude Opus 4.8", "norm_id": "anthropic.claude-opus-4.8"}],
    )
    saved = valves.model_copy()
    saved.API_KEY = EncryptedStr("sk-test-key")
    pipe.valves = saved

    result = await pipe.pipe(
        body={"model": "anthropic.claude-opus-4.8", "stream": True,
              "messages": [*history, *_open_webui_rebuilds(first_record)]},
        __user__={"id": "u1", "role": "user"},
        __request__=None,
        __event_emitter__=None,
        __event_call__=None,
        __metadata__={"model": {"id": MODEL}, "chat_id": "c1", "message_id": "m1", "assistant_message_id": "m1"},
        __tools__={"lookup": {"callable": lambda **_k: "ok"}},
    )
    second = ""
    second_record: list[dict[str, Any]] = []
    if hasattr(result, "__aiter__"):
        async for chunk in cast(Any, result):
            if isinstance(chunk, dict) and chunk.get("type") == "response.completed":
                second_record = (chunk.get("response") or {}).get("output") or second_record
            choices = chunk.get("choices") if isinstance(chunk, dict) else None
            if choices:
                second += choices[0].get("delta", {}).get("content") or ""
    assert len(attempts) >= 2, second
    assert second_record, "the Continue published no record through the stream"

    # Open WebUI keeps the continued message's stored output and appends what the Continue published.
    replay = await _replay(
        pipe, valves, persisted,
        [*history, *_open_webui_rebuilds(first_record + second_record), {"role": "user", "content": "q2"}],
    )

    order = [_label(item) for item in replay]
    assert order == AFTER_A_THINKING_RETRY[shape], order
