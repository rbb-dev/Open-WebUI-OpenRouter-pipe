from __future__ import annotations

from typing import Any

import pytest

from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import (
    answer_round,
    call,
    calls_round,
    install,
    offered,
    resolved_tool,
    run,
)
from tests.test_fusion_engine import _prepare_pipe

SAVED = {"chat_id": "chat-1", "message_id": "message-1", "session_id": "session-1"}
TEMPORARY = {"chat_id": "temporary:t1", "message_id": "message-1", "session_id": "session-1"}


def _script(n: int, _request: dict[str, Any]) -> list[dict[str, Any]]:
    return calls_round(call("lookup", {"q": "x"}, f"call-{n}")) if n == 1 else answer_round("done")


def _always_calls(n: int, _request: dict[str, Any]) -> list[dict[str, Any]]:
    return calls_round(call("lookup", {"q": "x"}, f"call-{n}"))


def _hosted_call_item(chunks: list[Any]) -> bool:
    return any(
        isinstance(chunk, dict)
        and chunk.get("type") == "response.output_item.added"
        and isinstance(chunk.get("item"), dict)
        and chunk["item"].get("type") == "function_call"
        for chunk in chunks
    )


async def _ask_turn(
    pipe: Any,
    monkeypatch: Any,
    metadata: dict[str, Any],
    *,
    script: Any = None,
    times: int = 1,
    loops: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Any]]:
    made: list[dict[str, Any]] = []

    async def spy(**kwargs: Any) -> str:
        made.append(dict(kwargs))
        return "ok"

    _prepare_pipe(pipe)
    requests = install(pipe, monkeypatch, script or _script)
    chunks: list[Any] = []
    for _ in range(times):
        round_chunks, _events = await run(
            pipe,
            mode="Pipeline",
            stream=True,
            metadata=metadata,
            owui_tools={"lookup": resolved_tool("lookup", spy)},
            params={"tool_approval_mode": "ask"},
            loops=loops,
        )
        chunks.extend(round_chunks)
    return made, requests, chunks


@pytest.mark.asyncio
async def test_ask_in_a_saved_streamed_chat_offers_the_tools_and_hands_the_call_back(
    pipe_instance_async, monkeypatch
):
    made, requests, chunks = await _ask_turn(pipe_instance_async, monkeypatch, SAVED)
    assert offered(requests[0]) == ["function:lookup"]
    assert made == []
    assert len(requests) == 1
    assert _hosted_call_item(chunks)


@pytest.mark.asyncio
async def test_ask_in_a_temporary_streamed_chat_hands_the_call_back_instead_of_running_it(
    pipe_instance_async, monkeypatch
):
    made, requests, chunks = await _ask_turn(pipe_instance_async, monkeypatch, TEMPORARY)
    assert made == []
    assert len(requests) == 1
    assert _hosted_call_item(chunks)


@pytest.mark.asyncio
async def test_ask_keeps_handing_back_past_the_reply_budget(pipe_instance_async, monkeypatch):
    made, requests, chunks = await _ask_turn(
        pipe_instance_async,
        monkeypatch,
        SAVED,
        script=_always_calls,
        times=2,
        loops=1,
    )
    assert made == []
    assert len(requests) == 2
    assert _hosted_call_item(chunks)
