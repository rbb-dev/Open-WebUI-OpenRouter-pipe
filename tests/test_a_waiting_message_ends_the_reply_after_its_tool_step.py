"""A message the user sends while a Pipeline-mode reply runs ends that reply after its current tool step.

Open WebUI 0.13.4 keeps a message typed while a reply is still running and, inside its own tool
loop, checks after each tool step whether one is waiting (`ChatMessages.has_queued_message`). If
so it stops there, without asking the model again, and the waiting message is answered next with
what was done. The pipe's own tool loop never checked, so in Pipeline mode the waiting message sat
until every tool step of the reply had finished.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import Pipe, ResponsesBody
from open_webui_openrouter_pipe.core.config import NO_CONTENT_AFTER_TOOLS_FALLBACK
from open_webui_openrouter_pipe.storage import owui_chats

MODEL = "test/tool-model"


@pytest.mark.asyncio
@pytest.mark.parametrize("message_waiting", [True, False])
async def test_a_waiting_message_ends_the_reply_after_its_tool_step(
    monkeypatch, pipe_instance_async, message_waiting
):
    pipe = pipe_instance_async
    body = ResponsesBody(model=MODEL, input=[], stream=True)
    valves = pipe.valves.model_copy(update={"TOOL_EXECUTION_MODE": "Pipeline"})

    requests: list[dict[str, Any]] = []

    async def streaming(self, _session, request_body, **_kwargs):
        requests.append(json.loads(json.dumps(request_body)))
        if len(requests) == 1:
            call = {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": '{"q": "status"}',
                "status": "completed",
            }
            yield {"type": "response.output_item.done", "item": call}
            yield {"type": "response.completed", "response": {"output": [call], "usage": {}}}
        else:
            yield {"type": "response.output_text.delta", "delta": "All looked up."}
            yield {"type": "response.completed", "response": {"output": [], "usage": {}}}

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", streaming)

    executed: list[str] = []

    async def run_tools(calls, _tools):
        executed.extend(str(call.get("call_id")) for call in calls)
        return [
            {
                "type": "function_call_output",
                "call_id": call.get("call_id"),
                "output": "status: green",
                "status": "completed",
            }
            for call in calls
        ]

    monkeypatch.setattr(pipe._ensure_tool_executor(), "_execute_function_calls", run_tools)

    asked: list[tuple[str, str]] = []

    async def has_queued_message(chat_id, message_id):
        asked.append((chat_id, message_id))
        return message_waiting

    monkeypatch.setattr(owui_chats, "_owui_queued_check", lambda: has_queued_message)

    async def emitter(_event):
        return None

    answer = await pipe._streaming_handler._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": MODEL}, "chat_id": "chat-1", "message_id": "reply-1"},
        tools={"lookup": {"type": "function", "callable": lambda **_: {}, "spec": {"name": "lookup"}}},
        session=cast(Any, object()),
        user_id="u1",
    )

    assert executed == ["call-1"], f"the tool step did not run exactly once: {executed}"
    assert asked == [("chat-1", "reply-1")], f"Open WebUI was not asked about this reply: {asked}"
    if message_waiting:
        assert len(requests) == 1, (
            f"the model was asked {len(requests)} times; with a message waiting the reply "
            "must stop after its tool step, as Open WebUI's own loop does"
        )
        assert NO_CONTENT_AFTER_TOOLS_FALLBACK not in answer, (
            "a reply stopped for a waiting message was given the no-content filler"
        )
    else:
        assert len(requests) == 2, f"with nothing waiting the model must answer with the result: {len(requests)}"
        assert "All looked up." in answer
