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

import asyncio
import json
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import Pipe, ResponsesBody

_TERMINAL_FAILURE = "incomplete"


def _outputs_in_request(request: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in (request.get("input") or [])
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]


async def _drive_tool_loop(
    pipe, monkeypatch, *, first_round_output: list[dict[str, Any]] | None = None, max_loops: int = 2,
    real_executor: bool = False, script: list[list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Run one tool round and return the requests the pipe actually sent.

    Mocks only the transport. The tool-call bookkeeping under test -- validating each
    call, deciding what to synthesise for the ones that cannot run, and assembling the
    continuation input -- is the real code.

    ``script`` is the per-round list of `response.completed` output lists, the last entry
    repeating; it is the only way to say "the model calls again on the next round", which a
    ceiling test needs, because a single call round never reaches a ceiling above one. With
    ``script`` None the default is one call round then an answer.
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

    if script is None:
        if first_round_output is None:
            raise TypeError("_drive_tool_loop needs either first_round_output or script")
        script = [
            first_round_output,
            [{"type": "message", "role": "assistant", "status": "completed",
              "content": [{"type": "output_text", "text": "Done."}]}],
        ]
    rounds = [[{"type": "response.completed", "response": {"output": listed, "usage": {}}}]
              for listed in script]
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

    context = token = None
    if real_executor:
        from open_webui_openrouter_pipe import _ToolExecutionContext

        context = _ToolExecutionContext(
            queue=asyncio.Queue(maxsize=50), per_request_semaphore=asyncio.Semaphore(1),
            global_semaphore=None, timeout=5.0, batch_timeout=5.0, idle_timeout=None,
            user_id="user-123", event_emitter=None, batch_cap=1,
        )
        context.workers.append(asyncio.create_task(pipe._ensure_tool_executor()._tool_worker_loop(context)))
        token = pipe._TOOL_CONTEXT.set(context)
    try:
        await pipe._streaming_handler._run_streaming_loop(
            body,
            valves,
            emitter,
            metadata={"model": {"id": "test"}, "chat_id": "chat-1", "message_id": "msg-1"},
            tools={"lookup": {"callable": lambda **_kwargs: "ok"}},
            session=cast(Any, object()),
            user_id="user-123",
        )
    finally:
        if context is not None:
            pipe._TOOL_CONTEXT.reset(token)
            for worker in context.workers:
                worker.cancel()
            await asyncio.gather(*context.workers, return_exceptions=True)
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

    The cap is 2 and the model calls on all three rounds, because a cap of 1 is now spent by the
    first round's own call and a cap of 2 is spent by the third: reaching a ceiling takes a call on
    every round up to it, so a script of two calls stops the loop before the ceiling ever fires.
    """
    call = {"type": "function_call", "call_id": "limit-1", "name": "lookup", "arguments": "{}"}
    captured = await _drive_tool_loop(
        pipe_instance_async,
        monkeypatch,
        script=[
            [call],
            [{**call, "call_id": "limit-2"}],
            [{**call, "call_id": "limit-3"}],
            [{"type": "message", "role": "assistant", "status": "completed",
              "content": [{"type": "output_text", "text": "Done."}]}],
        ],
        max_loops=2,
    )

    outputs = _outputs_in_request(captured[-1])
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
