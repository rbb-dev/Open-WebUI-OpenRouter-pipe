"""A turn's hidden markers must reach what Open WebUI stores, and the pipe must warn only when they do not.

Every artifact row the pipe commits during a turn (persisted reasoning, for one) is found again on the next turn
through a hidden marker line in the assistant's content. While streaming, the markers go out as a message delta. With
streaming off, Open WebUI stores the content the pipe returns, so the markers arrive by being in that content; a
warning there was a false alarm. The one non-streaming exit that returns before the markers are added (the
Open-WebUI-mode tool hand-back) still loses them, and that must keep warning.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator
from typing import Any, cast

import pytest

import open_webui_openrouter_pipe.pipe as pipe_mod
from open_webui_openrouter_pipe import Pipe, _serialize_marker, generate_item_id
from open_webui_openrouter_pipe.core.utils import is_hidden_marker_line

_UNADDRESSED = "left unaddressed"


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _catalog_entry(model_id: str) -> dict[str, str]:
    return {"id": model_id, "name": f"Model {model_id}", "norm_id": model_id}


async def _turn(pipe: Pipe, monkeypatch, *, stream: bool, answer: str, hand_back: bool = False):
    """One turn through `_handle_pipe_call`, with the reasoning it produced committed as an artifact row.

    Returns what the pipe handed back, the markers of the rows it committed, the events it emitted, and the warnings
    about unaddressed rows.
    """
    output: list[dict[str, Any]] = []
    if hand_back:
        output.append({"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}"})
    events = [
        {"type": "response.output_item.done", "item": {
            "id": "rs-1", "type": "reasoning", "status": "completed",
            "content": [{"type": "reasoning_text", "text": "THOUGHT"}], "summary": [], "signature": "SIG"}},
        {"type": "response.output_text.delta", "delta": answer},
        {"type": "response.completed", "response": {"output": output, "usage": {}}},
    ]

    async def transport(self, *_args, **_kwargs):
        for event in events:
            yield event

    committed: list[str] = []

    async def persist(rows):
        ulids = [generate_item_id() for _ in rows]
        committed.extend(ulids)
        return ulids

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", transport)
    monkeypatch.setattr(Pipe, "send_openrouter_nonstreaming_request_as_events", transport)
    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: ("sk-test", None))
    monkeypatch.setattr(pipe._artifact_store, "_ensure_artifact_store", lambda *_a, **_k: None)
    monkeypatch.setattr(pipe._artifact_store, "_make_db_row", lambda _c, _m, _model, payload: {"payload": payload})
    monkeypatch.setattr(pipe._artifact_store, "_db_persist", persist)

    async def loaded(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", loaded)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "list_models", lambda: [_catalog_entry("m1")])

    valves = pipe.valves.model_copy(
        update={
            "PERSIST_REASONING_TOKENS": "conversation",
            "TOOL_EXECUTION_MODE": "Open-WebUI" if hand_back else "Pipeline",
        }
    )
    emitted: list[dict[str, Any]] = []

    async def emitter(event: dict[str, Any]) -> None:
        emitted.append(event)

    records = _Records()
    logger = pipe._streaming_handler.logger
    previous_level = logger.level
    logger.addHandler(records)
    logger.setLevel(logging.DEBUG)
    try:
        result = await pipe._handle_pipe_call(
            {"stream": stream, "model": "m1", "messages": [{"role": "user", "content": "hi"}],
             **({"tools": [{"type": "function", "function": {"name": "lookup"}}]} if hand_back else {})},
            {"id": "user-1"},
            None,
            emitter,
            None,
            {"chat_id": "chat-1", "message_id": "message-1", "model": {"id": "m1"}},
            None,
            None,
            None,
            session=cast(Any, object()),
        valves=valves,
        )
    finally:
        logger.removeHandler(records)
        logger.setLevel(previous_level)
    markers = [_serialize_marker(ulid) for ulid in committed]
    warnings = [r for r in records.records if _UNADDRESSED in r.getMessage() and r.levelno >= logging.WARNING]
    return result, markers, emitted, warnings


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["The answer is four.", "A different answer entirely."])
async def test_a_non_streaming_turn_returns_its_markers_and_does_not_warn(monkeypatch, pipe_instance_async, answer):
    result, markers, _, warnings = await _turn(pipe_instance_async, monkeypatch, stream=False, answer=answer)

    assert markers, "the turn committed no artifact row, so it proves nothing about markers"
    assert isinstance(result, str) and answer in result
    assert all(marker in result for marker in markers), "Open WebUI stores the returned content; the markers ride in it"
    assert warnings == []


@pytest.mark.asyncio
async def test_a_streaming_turn_publishes_its_markers_and_does_not_warn(monkeypatch, pipe_instance_async):
    _, markers, emitted, warnings = await _turn(pipe_instance_async, monkeypatch, stream=True, answer="Streamed.")

    published = "".join(
        event["data"]["content"] for event in emitted if event.get("type") == "chat:message:delta"
    )
    assert markers
    assert all(marker in published for marker in markers)
    assert warnings == []


@pytest.mark.asyncio
async def test_a_non_streaming_tool_hand_back_that_returns_before_its_markers_still_warns(
    monkeypatch, pipe_instance_async
):
    # A hand-back the request declared a tool for returns its tool_calls response before the markers are added,
    # so those rows really are unaddressed.
    result, markers, _, warnings = await _turn(
        pipe_instance_async, monkeypatch, stream=False, answer="Let me look.", hand_back=True
    )

    assert markers
    content = str(result["choices"][0]["message"].get("content") or "") if isinstance(result, dict) else str(result)
    assert not any(marker in content for marker in markers)
    assert len(warnings) == 1
    assert all(marker in warnings[0].getMessage() for marker in markers)


def _every_string(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _every_string(item)
    elif isinstance(value, list):
        for item in value:
            yield from _every_string(item)


def _marker_lines_in(chunks: list[Any]) -> list[str]:
    return [
        line.strip()
        for text in _every_string(chunks)
        for line in text.splitlines()
        if is_hidden_marker_line(line)
    ]


def _reasoned_answer_with_a_phase(text: str) -> list[dict[str, Any]]:
    message = {
        "type": "message", "role": "assistant", "status": "completed", "phase": "final_answer",
        "content": [{"type": "output_text", "text": text}],
    }
    return [
        {"type": "response.output_item.done", "item": {
            "id": "rs-1", "type": "reasoning", "status": "completed",
            "content": [{"type": "reasoning_text", "text": "THOUGHT"}], "summary": [], "signature": "SIG"}},
        {"type": "response.output_text.delta", "delta": text},
        {"type": "response.output_item.done", "item": message},
        {"type": "response.completed", "response": {"output": [message], "usage": {}}},
    ]


@pytest.mark.asyncio
async def test_a_realtime_voice_call_answer_carries_no_marker_lines(monkeypatch, pipe_instance_async):
    from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import CHAT, install, run

    pipe = pipe_instance_async
    install(pipe, monkeypatch, lambda _n, _request: _reasoned_answer_with_a_phase("Two plus two is four."))
    realtime_user_message = {
        "id": "user-message-1", "role": "user", "content": "What is two plus two?",
        "meta": {"voice": {"call_id": "call-1", "input_item_id": "item-1", "model": "realtime-model"}},
    }
    standard_user_message = {"id": "user-message-2", "role": "user", "content": "What is two plus two?"}

    realtime_chunks, _ = await run(
        pipe, mode="Pipeline", stream=True,
        metadata={**CHAT, "features": {"voice": True}, "user_message": realtime_user_message},
    )
    standard_chunks, _ = await run(
        pipe, mode="Pipeline", stream=True,
        metadata={**CHAT, "features": {"voice": True}, "user_message": standard_user_message},
    )

    standard_lines = _marker_lines_in(standard_chunks)
    assert any(line.startswith("[P:") for line in standard_lines), standard_lines
    assert any(not line.startswith("[P:") for line in standard_lines), standard_lines
    assert "Two plus two is four." in "".join(_every_string(realtime_chunks))
    assert _marker_lines_in(realtime_chunks) == []
