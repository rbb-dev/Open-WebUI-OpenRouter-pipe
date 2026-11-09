from __future__ import annotations

import json
from typing import Any

import pytest

from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import (
    answer_round,
    call,
    calls_round,
    install,
    resolved_tool,
    run,
)
from tests.test_fusion_engine import _prepare_pipe

SAVED = {"chat_id": "chat-1", "message_id": "message-1", "session_id": "session-1"}
LABELS = {"running": "Looking up the stock", "completed": "Looked up the stock"}


def _labelled(entry: dict[str, Any]) -> dict[str, Any]:
    parameters = entry["spec"]["parameters"]
    label_schema = {
        "type": "object",
        "properties": {"running": {"type": "string"}, "completed": {"type": "string"}},
        "required": ["running", "completed"],
        "additionalProperties": False,
    }
    return {
        **entry,
        "display_labels": True,
        "spec": {
            **entry["spec"],
            "parameters": {
                **parameters,
                "properties": {"__labels__": label_schema, **parameters["properties"]},
                "required": ["__labels__", *parameters.get("required", [])],
            },
        },
    }


@pytest.mark.asyncio
async def test_a_labelled_call_runs_its_tool_with_its_own_arguments_and_shows_the_labels_on_its_card(
    pipe_instance_async, monkeypatch
):
    ran: list[str] = []

    async def lookup(q: str) -> str:
        ran.append(q)
        return "12 in stock"

    def script(n: int, _request: dict[str, Any]) -> list[dict[str, Any]]:
        if n == 1:
            return calls_round(call("lookup", {"q": "alpha", "__labels__": LABELS}))
        return answer_round("done")

    _prepare_pipe(pipe_instance_async)
    install(pipe_instance_async, monkeypatch, script)
    chunks, _events = await run(
        pipe_instance_async,
        mode="Pipeline",
        stream=True,
        metadata=SAVED,
        owui_tools={"lookup": _labelled(resolved_tool("lookup", lookup))},
    )

    assert ran == ["alpha"]
    card = next(
        chunk["item"]
        for chunk in chunks
        if isinstance(chunk, dict)
        and chunk.get("type") == "response.output_item.added"
        and chunk["item"].get("type") == "function_call"
    )
    assert card["display_labels"] == LABELS
    assert json.loads(card["arguments"]) == {"q": "alpha"}
    assert json.loads(card["original_arguments"]) == {"q": "alpha", "__labels__": LABELS}
