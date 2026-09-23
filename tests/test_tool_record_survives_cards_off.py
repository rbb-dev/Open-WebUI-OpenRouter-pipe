"""The card switch decides what the USER sees. It never decides what the MODEL knows.

In Open WebUI a tool call saved in a message IS its card: `function_call` is one of the `GROUPABLE_OUTPUT_TYPES`
in `structuredOutput.ts`, every stored one renders as a collapsible card, and no setting hides it. So a user who
switches tool cards off can only be spared them if the round is kept OUT of the saved message. That used to cost
the model its memory of the round: with nothing saved, the next turn arrived with no trace a tool had run, and a
model asked what it had done concluded it had made the answer up (measured on claude-opus-4.8, t227a CONTROL).

Decided 2026-09-22: switch ON (the default, matching Open WebUI) -- the round is saved in the message, shown as a
card, and Open WebUI replays it. Switch OFF -- nothing is saved in the message, so no card appears, live or after
the answer or on reload; the pipe hands the round to the model on the next turn through its own storage, which
Open WebUI never displays. The switch is a per-user valve as well as a site default.

The result TEXT is a separate axis, owned by `PERSIST_TOOL_RESULTS`: it governs what the model is handed again
("Let the AI reuse outputs from tools ... later in the conversation"), not what the user may read.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe.core.utils import PIPE_ONLY_TOOL_ROUND_KEY
from open_webui_openrouter_pipe.requests.transformer import transform_messages_to_input
from tests.test_continue_stores_once import _open_webui_convert_output_to_messages
from tests.test_reasoning_skeleton_replay import (
    MODEL,
    RESULT_CANARY,
    SEQUENTIAL_TWO_ROUNDS,
    _has_consecutive_reasoning,
    _recorded_output,
    _shape,
    _stage_a,
    _stage_b,
    _valves,
)

async def _run_server_tool(pipe, monkeypatch, item_type, fields, *, cards, stream=True):
    from open_webui_openrouter_pipe import Pipe, ResponsesBody
    from tests.test_streaming_handler import _make_fake_stream

    events = [
        {"type": "response.output_item.added", "item": {"type": item_type, "id": "st-1", "status": "in_progress"}},
        {"type": "response.output_item.done",
         "item": {"type": item_type, "id": "st-1", "status": "completed", **fields}},
        {"type": "response.completed", "response": {"output": [], "usage": {}}},
    ]
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _make_fake_stream(events))
    monkeypatch.setattr(Pipe, "send_openrouter_nonstreaming_request_as_events", _make_fake_stream(events))
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    loop = pipe._streaming_handler._run_streaming_loop if stream else pipe._streaming_handler._run_nonstreaming_loop
    returned = await loop(
        ResponsesBody(model="test/model", input=[], stream=stream),
        pipe.valves.model_copy(update={"SHOW_TOOL_CARDS": cards}),
        emitter,
        metadata={"model": {"id": "test"}, "chat_id": "c1", "message_id": "m1"},
        tools={}, session=cast(Any, object()), user_id="u1",
    )
    if not stream:
        emitted.append({"type": "chat:message:delta", "data": {"content": returned if isinstance(returned, str) else ""}})
    return emitted
