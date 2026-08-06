"""A tool call that failed or never ran must not be reported as one that succeeded.

Open WebUI persists a function_call_output into the stored assistant message and
replays it to the model on the next turn, so a false "completed" is durable: the user
sees a success, and the model is told the call worked.

Each test here runs the code that DECIDES the status and reads what it produced. That
distinction is the whole point and it is easy to lose: an earlier version of this file
built the output dict itself, with the status already in it, and asserted that
``normalize_persisted_item`` handed the same value back. It never called the producer,
so flipping the real one to "completed" left it green -- a tautology sitting under a
docstring claiming it could not be defeated.

Source scans are not the answer either. Four of them once covered these sites by
looking for a hardcoded "completed", and each was defeated in turn by spelling the same
value differently -- ``str(...)``, ``"comp" + "leted"``, and finally a local name
assigned the literal, which slipped past even an allow-list of approved shapes because
a name is an indirection, not a shape. Driving the producer and reading its output does
not care how the value was spelled.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import Pipe, ResponsesBody

_TERMINAL_FAILURE = "incomplete"


def _outputs_replayed_to_the_model(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every function_call_output the pipe sent back on the follow-up request."""
    if len(requests) < 2:
        return []
    return [
        item
        for item in (requests[1].get("input") or [])
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]


async def _drive_tool_loop(
    pipe, monkeypatch, *, first_round_output: list[dict[str, Any]], max_loops: int = 2
) -> list[dict[str, Any]]:
    """Run one tool round and return the requests the pipe actually sent.

    Mocks only the transport. The tool-call bookkeeping under test -- validating each
    call, deciding what to synthesise for the ones that cannot run, and assembling the
    continuation input -- is the real code.
    """
    body = ResponsesBody(model="test/model", input=[], stream=True)
    valves = pipe.valves.model_copy(
        update={
            "TOOL_EXECUTION_MODE": "Pipeline",
            "PERSIST_TOOL_RESULTS": True,
            "SHOW_TOOL_CARDS": False,
            "MAX_FUNCTION_CALL_LOOPS": max_loops,
        }
    )

    rounds = [
        [{"type": "response.completed", "response": {"output": first_round_output, "usage": {}}}],
        [
            {"type": "response.output_text.delta", "delta": "Done."},
            {"type": "response.completed", "response": {"output": [], "usage": {}}},
        ],
    ]
    captured: list[dict[str, Any]] = []
    call_index = 0

    async def streaming(self, session, request_body, **_kwargs):
        nonlocal call_index
        idx = min(call_index, len(rounds) - 1)
        call_index += 1
        captured.append(json.loads(json.dumps(request_body)))
        for event in rounds[idx]:
            yield event

    async def _no_persist(rows):
        return [f"ulid-{i}" for i in range(len(rows))]

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", streaming)
    monkeypatch.setattr(pipe._artifact_store, "_db_persist", _no_persist)

    async def emitter(_event):
        return None

    await pipe._streaming_handler._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "test"}, "chat_id": "chat-1", "message_id": "msg-1"},
        tools={"lookup": {"callable": lambda **_kwargs: "ok"}},
        session=cast(Any, object()),
        user_id="user-123",
    )
    return captured


def test_an_orphaned_tool_call_gets_a_failure_stub(pipe_instance):
    """A function_call with no output is the paradigmatic "produced nothing" case.

    The sanitizer synthesises the missing output so the model is not left with a call
    that never resolves. That stub must carry a failure status: with none, or with
    "completed", the model is told a call that never ran succeeded.

    Only INTERIOR orphans are stubbed -- ones sitting before the last user message, so
    the conversation has already moved past them. A trailing orphan is the current
    turn's in-flight call and is left alone.
    """
    from open_webui_openrouter_pipe.api.transforms import ResponsesBody as _Body
    from open_webui_openrouter_pipe.requests.sanitizer import _sanitize_request_input

    body = _Body.model_validate(
        {
            "model": "openrouter/test",
            "input": [
                {"type": "function_call", "call_id": "orphan-1", "name": "search", "arguments": "{}"},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "next"}]},
            ],
            "stream": True,
        }
    )
    _sanitize_request_input(pipe_instance, body)

    stubs = [
        item
        for item in body.input
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and item.get("call_id") == "orphan-1"
    ]
    assert stubs, (
        "the orphaned function_call got no synthesised output at all, so the model "
        "sees a tool call that never resolves"
    )
    assert stubs[0].get("status") == _TERMINAL_FAILURE, (
        f"the orphan stub reports {stubs[0].get('status')!r}. A call that produced "
        "nothing is being replayed to the model as one that completed."
    )


@pytest.mark.asyncio
async def test_a_call_missing_its_arguments_is_reported_to_the_model_as_a_failure(
    pipe_instance_async, monkeypatch
):
    """A malformed call cannot be executed, so its synthesised output must say so.

    The stream yields a function_call the pipe cannot run. Nothing is mocked between
    that event and the decision under test, so the status asserted below is the one the
    pipe genuinely chose.

    The output only reaches the model paired with a call bearing the same call_id --
    an unpaired function_call_output is an orphan and the sanitizer drops it before the
    request goes out. Here the model named a real tool, so the call can be repaired
    into a valid one and the pair survives.
    """
    captured = await _drive_tool_loop(
        pipe_instance_async,
        monkeypatch,
        first_round_output=[{"type": "function_call", "call_id": "c2", "name": "lookup"}],
    )

    outputs = _outputs_replayed_to_the_model(captured)
    assert outputs, (
        "a tool call with no arguments produced no output item at all, so the model is "
        "told nothing about a call it emitted and can repeat it indefinitely"
    )
    assert [o.get("status") for o in outputs] == [_TERMINAL_FAILURE], (
        f"a tool call with no arguments is replayed to the model as "
        f"{[o.get('status') for o in outputs]!r}; it was never executed, so reporting "
        "success is a lie to both the user and the model"
    )
    assert "missing arguments" in str(outputs[0].get("output", "")), (
        "the output does not say what was wrong with the call"
    )


@pytest.mark.asyncio
async def test_a_call_missing_its_name_still_gets_the_model_a_recovery_turn(
    pipe_instance_async, monkeypatch
):
    """The one malformed shape that cannot be reported, and why that is deliberate.

    Pairing the output with a repaired call is what gets it past the orphan check, and
    a call with no name cannot be repaired: any placeholder names a function the model
    never requested and need not exist in ``tools``, which risks the provider rejecting
    the recovery turn outright. That is a worse outcome than silence, so the output is
    left to be dropped.

    What must NOT be lost is the recovery turn itself. Without a second request the
    user gets an empty response, so this pins the follow-up rather than the payload.
    """
    captured = await _drive_tool_loop(
        pipe_instance_async,
        monkeypatch,
        first_round_output=[{"type": "function_call", "call_id": "c1", "arguments": "{}"}],
    )

    assert len(captured) >= 2, (
        "a nameless tool call ended the turn instead of giving the model a chance to "
        "answer, so the user sees nothing at all"
    )
    replayed = [
        item
        for item in (captured[1].get("input") or [])
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]
    assert not replayed, (
        f"a call the pipe could not repair was replayed anyway as {replayed!r}; its "
        "name is not one the model requested and may not exist in the tools list"
    )


@pytest.mark.asyncio
async def test_calls_skipped_by_the_loop_limit_are_carded_as_failures(
    pipe_instance_async, monkeypatch
):
    """Hitting MAX_FUNCTION_CALL_LOOPS abandons pending calls; the stubs must admit it.

    With the limit reached the pipe injects an output for every call it will not run.
    Reporting those as completed tells the model it has results it never received, and
    the text it then writes is grounded in nothing.
    """
    captured = await _drive_tool_loop(
        pipe_instance_async,
        monkeypatch,
        first_round_output=[
            {"type": "function_call", "call_id": "limit-1", "name": "lookup", "arguments": "{}"}
        ],
        max_loops=1,
    )

    outputs = _outputs_replayed_to_the_model(captured)
    assert outputs, (
        "the loop limit was reached and no stub was injected for the pending call, so "
        "the model sees a tool call that never resolves"
    )
    assert all(o.get("status") == _TERMINAL_FAILURE for o in outputs), (
        f"calls abandoned at the loop limit are replayed as "
        f"{[o.get('status') for o in outputs]!r}; they were never executed"
    )
    assert any("TOOL_CALL_SKIPPED" in str(o.get("output", "")) for o in outputs), (
        "the stub does not tell the model why it has no result, so the model cannot "
        "tell an empty result from a skipped call"
    )


def test_a_missing_status_is_persisted_as_completed():
    """The reason the stubs above must be explicit.

    This pins the default that makes an omission dangerous: normalize_persisted_item
    fills in "completed", so a producer that simply forgets the field reports success.
    If this ever stops being true, the urgency of the tests above changes and their
    docstrings need revisiting.
    """
    from open_webui_openrouter_pipe.storage.persistence import normalize_persisted_item

    emitted = normalize_persisted_item(
        {"type": "function_call_output", "call_id": "c9", "output": "whatever"}
    )
    assert emitted is not None
    assert emitted.get("status") == "completed", (
        "a statusless function_call_output is no longer defaulted to 'completed' -- "
        "the failure stubs' explicit status may no longer be load-bearing"
    )
