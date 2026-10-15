"""A call whose id an earlier call of the reply already used is a new call: it runs and gets its card.

Some providers restart their tool-call ids from zero on every round, so a later call — even a
verbatim repeat of an earlier one — can carry a taken id. The pipe keys reply-wide state by that
id — the emitted cards and the already-run guard — so the repeat lost its card and was refused
with TOOL_CALL_ALREADY_RUN instead of running, and its pair doubled an id in the next request.
Open WebUI re-mints the id of any call that collides with an earlier call of the same reply; the
pipe now does the same where the round's calls are built, so the call, its card, its result and
the request sent back to the provider all carry the new id.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import Pipe, ResponsesBody

MODEL = "test/restarting-provider"
CALL_ID = "call-zero"
THE_ARGS = '{"q": "status"}'


def _call_item() -> dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": CALL_ID,
        "name": "lookup",
        "arguments": THE_ARGS,
        "status": "completed",
    }


@pytest.mark.asyncio
async def test_an_identical_repeat_on_a_reused_id_runs_and_get_its_card(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    body = ResponsesBody(model=MODEL, input=[], stream=True)
    valves = pipe.valves.model_copy(update={"SHOW_TOOL_CARDS": True, "TOOL_EXECUTION_MODE": "Pipeline"})

    rounds: list[list[dict[str, Any]]] = [[_call_item()], [_call_item()]]
    requests: list[dict[str, Any]] = []
    step = [0]

    async def streaming(self, _session, request_body, **_kwargs):
        requests.append(json.loads(json.dumps(request_body)))
        index = step[0]
        step[0] += 1
        if index < len(rounds):
            yield {"type": "response.output_item.done", "item": rounds[index][0]}
            yield {
                "type": "response.completed",
                "response": {"output": list(rounds[index]), "usage": {}},
            }
        else:
            yield {"type": "response.output_text.delta", "delta": "done"}
            yield {"type": "response.completed", "response": {"output": [], "usage": {}}}

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", streaming)

    executed: list[list[dict[str, Any]]] = []

    async def run_tools(calls, _tools):
        executed.append([dict(call) for call in calls])
        return [
            {
                "type": "function_call_output",
                "call_id": call.get("call_id"),
                "output": f"result for {call.get('call_id')}",
                "status": "completed",
            }
            for call in calls
        ]

    monkeypatch.setattr(pipe._ensure_tool_executor(), "_execute_function_calls", run_tools)

    emitted: list[dict[str, Any]] = []

    async def emitter(event):
        emitted.append(event)

    await pipe._streaming_handler._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": MODEL}, "chat_id": "c1", "message_id": "m1"},
        tools={"lookup": {"type": "function", "callable": lambda **_: {}, "spec": {"name": "lookup"}}},
        session=cast(Any, object()),
        user_id="u1",
    )

    assert len(executed) == 2, f"expected both rounds to reach the executor, got {len(executed)}"
    second_call = executed[1][0]
    assert second_call.get("arguments") == THE_ARGS
    assert second_call.get("call_id") != CALL_ID, "the second call kept the id of the first"

    start_cards = [
        event["item"]
        for event in emitted
        if event.get("type") == "response.output_item.added"
        and event.get("item", {}).get("type") == "function_call"
    ]
    assert len(start_cards) == 2, f"expected a card per call, got {len(start_cards)}"
    assert start_cards[1].get("call_id") == second_call.get("call_id")

    final_inputs = requests[-1].get("input") or []
    final_calls = [
        str(item.get("call_id"))
        for item in final_inputs
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]
    final_outputs = [
        (str(item.get("call_id")), str(item.get("output")))
        for item in final_inputs
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    expected_call_ids = {CALL_ID, str(second_call.get("call_id"))}
    assert sorted(final_calls) == sorted(expected_call_ids), (
        f"the provider was handed ambiguous call pairs: {final_calls}"
    )
    assert all("TOOL_CALL_ALREADY_RUN" not in text for _, text in final_outputs), (
        f"the repeat was refused instead of run: {final_outputs}"
    )
