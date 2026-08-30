"""Native reasoning output-item emission tests (Part A migration).

Drives the real _run_streaming_loop with fake upstream streams and asserts the
append-only native protocol: one completed reasoning output_item.added per
reasoning block, real receive-window durations, no legacy reasoning:delta /
reasoning:completed / reasoning_content emissions, no timing-marker spaces.

A stream-synchronized fake clock advances streaming_core._monotonic between
yielded events, so duration assertions are exact.
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportCallIssue=false

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest

import open_webui_openrouter_pipe.streaming.streaming_core as streaming_core_mod

from open_webui_openrouter_pipe import Pipe, ResponsesBody
from open_webui_openrouter_pipe.streaming import streaming_core


def _make_timed_stream(steps: list[tuple[float, dict[str, Any]]], clock: dict[str, float]):
    async def fake_stream(self, session, request_body, **_kwargs):
        for advance, event in steps:
            clock["now"] += advance
            yield event
    return fake_stream


def _install_clock(monkeypatch) -> dict[str, float]:
    clock = {"now": 1000.0}
    monkeypatch.setattr(streaming_core, "_monotonic", lambda: clock["now"])
    return clock


def _native_reasoning_items(emitted: list[dict]) -> list[dict]:
    items = []
    for e in emitted:
        if e.get("type") == "response.output_item.added":
            item = e.get("item") or {}
            if item.get("type") == "reasoning":
                items.append(item)
    return items


def _events_of(emitted: list[dict], event_type: str) -> list[dict]:
    return [e for e in emitted if e.get("type") == event_type]


async def _run(pipe, valves, steps, clock, monkeypatch) -> list[dict]:
    body = ResponsesBody(model="test/model", input=[], stream=True)
    monkeypatch.setattr(
        Pipe, "send_openrouter_streaming_request", _make_timed_stream(steps, clock)
    )
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe._streaming_handler._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "test"}},
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )
    return emitted


def _assert_completed_shape(item: dict) -> None:
    assert item.get("status") == "completed"
    assert "attributes" not in item
    assert isinstance(item.get("started_at"), float)
    assert isinstance(item.get("ended_at"), float)
    assert isinstance(item.get("duration"), float)
    assert item["duration"] >= 0.1
    summary = item.get("summary")
    assert isinstance(summary, list) and summary
    assert summary[0].get("type") == "summary_text"
    assert summary[0].get("text")


class TestNativeReasoningDelta:
    """T1: delta-shaped streams emit one completed box per item."""

    @pytest.mark.asyncio
    async def test_delta_shape_single_completed_item_before_answer(
        self, monkeypatch, pipe_instance_async
    ):
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        steps = [
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-1"}}),
            (3.0, {"type": "response.reasoning_text.delta", "item_id": "rs-1", "delta": "Thinking hard about it. "}),
            (4.0, {"type": "response.output_text.delta", "delta": "The answer is 42."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)

        items = _native_reasoning_items(emitted)
        assert len(items) == 1
        _assert_completed_shape(items[0])
        assert items[0]["id"] == "rs-1"
        assert "Thinking hard" in items[0]["summary"][0]["text"]
        assert items[0]["duration"] == pytest.approx(7.0)

        native_idx = next(
            i for i, e in enumerate(emitted)
            if e.get("type") == "response.output_item.added"
            and (e.get("item") or {}).get("type") == "reasoning"
        )
        answer_idx = next(
            i for i, e in enumerate(emitted)
            if e.get("type") == "chat:message:delta"
            and "42" in (e.get("data") or {}).get("content", "")
        )
        assert native_idx < answer_idx, "reasoning box must be emitted before the answer delta"

        assert _events_of(emitted, "reasoning:delta") == []
        assert _events_of(emitted, "reasoning:completed") == []

    @pytest.mark.asyncio
    async def test_late_done_after_answer_is_deduped(self, monkeypatch, pipe_instance_async):
        """Anthropic shape: terminal output_item.done arrives after the answer."""
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        steps = [
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-a"}}),
            (2.0, {"type": "response.reasoning_text.delta", "item_id": "rs-a", "delta": "Deliberating. "}),
            (1.0, {"type": "response.output_text.delta", "delta": "Final answer."}),
            (5.0, {
                "type": "response.output_item.done",
                "item": {"type": "reasoning", "id": "rs-a",
                         "content": [{"type": "reasoning_text", "text": "Deliberating. "}]},
            }),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)

        items = _native_reasoning_items(emitted)
        assert len(items) == 1, "late duplicate done must not create a second box"
        assert items[0]["duration"] == pytest.approx(3.0)


class TestNativeReasoningSnapshot:
    """T2: snapshot-shaped streams (text only in output_item.done)."""

    @pytest.mark.asyncio
    async def test_snapshot_duration_excludes_tool_latency(
        self, monkeypatch, pipe_instance_async
    ):
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        steps = [
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-1", "summary": []}}),
            (5.0, {"type": "response.output_item.added", "item": {"type": "openrouter:web_search", "id": "ws-1"}}),
            (60.0, {"type": "response.output_item.done", "item": {"type": "openrouter:web_search", "id": "ws-1", "action": {}}}),
            (2.0, {
                "type": "response.output_item.done",
                "item": {"type": "reasoning", "id": "rs-1", "status": "completed",
                         "summary": [{"type": "summary_text", "text": "Considered the search plan."}],
                         "encrypted_content": "opaque"},
            }),
            (1.0, {"type": "response.output_text.delta", "delta": "Here is what I found."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)

        items = _native_reasoning_items(emitted)
        assert len(items) == 1
        _assert_completed_shape(items[0])
        assert "search plan" in items[0]["summary"][0]["text"]
        assert items[0]["duration"] == pytest.approx(5.0), (
            "snapshot duration must close at the next item boundary, excluding tool latency"
        )

    @pytest.mark.asyncio
    async def test_multiple_snapshot_items_distinct_texts(self, monkeypatch, pipe_instance_async):
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        steps = [
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-1", "summary": []}}),
            (2.0, {"type": "response.output_item.done", "item": {"type": "reasoning", "id": "rs-1", "summary": [{"type": "summary_text", "text": "First thought."}]}}),
            (1.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-2", "summary": []}}),
            (3.0, {"type": "response.output_item.done", "item": {"type": "reasoning", "id": "rs-2", "summary": [{"type": "summary_text", "text": "Second thought."}]}}),
            (1.0, {"type": "response.output_text.delta", "delta": "Answer."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)

        items = _native_reasoning_items(emitted)
        assert [i["id"] for i in items] == ["rs-1", "rs-2"]
        assert items[0]["summary"][0]["text"] == "First thought."
        assert items[1]["summary"][0]["text"] == "Second thought."
        for item in items:
            _assert_completed_shape(item)


class TestEmptyAndModes:
    @pytest.mark.asyncio
    async def test_encrypted_only_items_are_dropped(self, monkeypatch, pipe_instance_async):
        """T3: empty/encrypted-only reasoning emits nothing."""
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        steps = [
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-e", "summary": []}}),
            (1.0, {"type": "response.output_item.done", "item": {"type": "reasoning", "id": "rs-e", "summary": [], "encrypted_content": "opaque"}}),
            (1.0, {"type": "response.output_text.delta", "delta": "Answer."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)
        assert _native_reasoning_items(emitted) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "mode,expect_native,expect_status",
        [("open_webui", 1, False), ("status", 0, True), ("both", 1, True)],
    )
    async def test_mode_matrix(self, monkeypatch, pipe_instance_async, mode, expect_native, expect_status):
        """T4: box/status/both gating."""
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": mode})
        steps = [
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-1"}}),
            (2.0, {"type": "response.reasoning_text.delta", "item_id": "rs-1", "delta": "Weighing the options carefully."}),
            (1.0, {"type": "response.output_text.delta", "delta": "Answer."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)

        assert len(_native_reasoning_items(emitted)) == expect_native
        status_texts = [
            (e.get("data") or {}).get("description", "")
            for e in _events_of(emitted, "status")
        ]
        assert any("Weighing" in t for t in status_texts) == expect_status
        assert _events_of(emitted, "reasoning:delta") == []
        assert _events_of(emitted, "reasoning:completed") == []


class TestIdsAndRobustness:
    @pytest.mark.asyncio
    async def test_idless_items_get_minted_ids_and_stay_distinct(
        self, monkeypatch, pipe_instance_async
    ):
        """T5: two id-less reasoning items render as two distinct boxes."""
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        steps = [
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning"}}),
            (1.0, {"type": "response.reasoning_text.delta", "delta": "Alpha line."}),
            (1.0, {"type": "response.output_item.done", "item": {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Alpha line."}]}}),
            (1.0, {"type": "response.output_item.added", "item": {"type": "reasoning"}}),
            (1.0, {"type": "response.reasoning_text.delta", "delta": "Beta line."}),
            (1.0, {"type": "response.output_item.done", "item": {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Beta line."}]}}),
            (1.0, {"type": "response.output_text.delta", "delta": "Answer."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)

        items = _native_reasoning_items(emitted)
        assert len(items) == 2
        texts = [i["summary"][0]["text"] for i in items]
        assert "Alpha" in texts[0] and "Beta" in texts[1]
        ids = [i.get("id") for i in items]
        assert all(isinstance(x, str) and x for x in ids)
        assert ids[0] != ids[1]

    @pytest.mark.asyncio
    async def test_truncated_stream_emits_terminating_box(self, monkeypatch, pipe_instance_async):
        """T6: stream dies mid-reasoning -> finally emits a completed box."""
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        steps = [
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-t"}}),
            (2.0, {"type": "response.reasoning_text.delta", "item_id": "rs-t", "delta": "Partial thought"}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)

        items = _native_reasoning_items(emitted)
        assert len(items) == 1
        _assert_completed_shape(items[0])
        assert "Partial thought" in items[0]["summary"][0]["text"]

    @pytest.mark.asyncio
    async def test_done_without_added_lazy_init(self, monkeypatch, pipe_instance_async):
        """T8: reasoning done for an unseen id must not crash and must emit."""
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        steps = [
            (0.0, {"type": "response.output_item.done", "item": {"type": "reasoning", "id": "rs-orphan", "summary": [{"type": "summary_text", "text": "Orphan thought."}]}}),
            (1.0, {"type": "response.output_text.delta", "delta": "Answer."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)

        items = _native_reasoning_items(emitted)
        assert len(items) == 1
        assert items[0]["duration"] == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_delta_resume_clears_close_candidate(self, monkeypatch, pipe_instance_async):
        """T8: reasoning text resuming after a tool boundary re-opens the window."""
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        steps = [
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-1"}}),
            (1.0, {"type": "response.reasoning_text.delta", "item_id": "rs-1", "delta": "Part one. "}),
            (1.0, {"type": "response.output_item.added", "item": {"type": "openrouter:web_search", "id": "ws-1"}}),
            (10.0, {"type": "response.reasoning_text.delta", "item_id": "rs-1", "delta": "Part two."}),
            (2.0, {"type": "response.output_text.delta", "delta": "Answer."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)

        items = _native_reasoning_items(emitted)
        assert len(items) == 1
        assert items[0]["duration"] == pytest.approx(14.0), (
            "resumed reasoning must extend the window to the final boundary"
        )
        assert "Part one" in items[0]["summary"][0]["text"]
        assert "Part two" in items[0]["summary"][0]["text"]


class TestAnswerAndWireIntegrity:
    @pytest.mark.asyncio
    async def test_no_marker_space_and_content_none_at_completion(
        self, monkeypatch, pipe_instance_async
    ):
        """T10: no stray marker-space deltas; completion carries content=None."""
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        steps = [
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-1"}}),
            (2.0, {"type": "response.reasoning_text.delta", "item_id": "rs-1", "delta": "Reasoned."}),
            (1.0, {"type": "response.output_text.delta", "delta": "Hello world."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)

        for e in _events_of(emitted, "chat:message:delta"):
            assert (e.get("data") or {}).get("content") != " ", "timing-marker space must be retired"
        completions = _events_of(emitted, "chat:completion")
        assert completions
        assert all((e.get("data") or {}).get("content") is None for e in completions)
        deltas = "".join(
            (e.get("data") or {}).get("content", "") for e in _events_of(emitted, "chat:message:delta")
        )
        assert "Hello world." in deltas

    @pytest.mark.asyncio
    async def test_upstream_response_completed_is_never_forwarded(
        self, monkeypatch, pipe_instance_async
    ):
        """N5: the upstream array is a stub; forwarding it would wipe OWUI's output."""
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        upstream_stub = [{"type": "message"}]
        steps = [
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-1"}}),
            (1.0, {"type": "response.reasoning_text.delta", "item_id": "rs-1", "delta": "Thought."}),
            (1.0, {"type": "response.output_text.delta", "delta": "Answer."}),
            (0.0, {"type": "response.completed",
                   "response": {"output": upstream_stub, "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)
        for event in _events_of(emitted, "response.completed"):
            published = (event.get("response") or {}).get("output") or []
            assert published != upstream_stub
            text = "".join(
                part.get("text", "")
                for item in published
                if item.get("type") == "message"
                for part in (item.get("content") or [])
            )
            assert "Answer." in text

    @pytest.mark.asyncio
    async def test_published_output_array_carries_every_emitted_item(
        self, monkeypatch, pipe_instance_async
    ):
        """The terminal array must be a superset of what OWUI could assemble alone."""
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(
            update={"THINKING_OUTPUT_MODE": "open_webui", "SHOW_TOOL_CARDS": True}
        )
        steps = [
            (0.0, {"type": "response.output_text.delta", "delta": "Checking. "}),
            (0.2, {"type": "response.output_item.added",
                   "item": {"type": "openrouter:web_search", "id": "ws-1"}}),
            (1.0, {"type": "response.output_item.done",
                   "item": {"type": "openrouter:web_search", "id": "ws-1", "action": {}}}),
            (0.2, {"type": "response.output_text.delta", "delta": "It is 22 degrees."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)
        completions = _events_of(emitted, "response.completed")
        assert completions
        published = (completions[-1].get("response") or {}).get("output") or []
        for item in _events_of(emitted, "response.output_item.added"):
            item_id = (item.get("item") or {}).get("id")
            assert any(entry.get("id") == item_id for entry in published)
        text = "".join(
            part.get("text", "")
            for entry in published
            if entry.get("type") == "message"
            for part in (entry.get("content") or [])
        )
        assert text == "Checking. It is 22 degrees."
        calls = [e for e in published if e.get("type") == "function_call"]
        assert calls and all(
            e.get("status") in {"completed", "failed", "rejected"} for e in calls
        )

        kinds = [e.get("type") for e in published]
        first_call = kinds.index("function_call")
        lead = "".join(
            part.get("text", "")
            for entry in published[:first_call]
            if entry.get("type") == "message"
            for part in (entry.get("content") or [])
        )
        tail = "".join(
            part.get("text", "")
            for entry in published[first_call:]
            if entry.get("type") == "message"
            for part in (entry.get("content") or [])
        )
        assert lead == "Checking. "
        assert tail == "It is 22 degrees."


class TestFailedToolCardsStillReachHistory:
    """A tool that failed must still be replayable, not silently dropped."""

    @pytest.mark.asyncio
    async def test_an_incomplete_tool_card_is_published_at_a_terminal_status(
        self, monkeypatch, pipe_instance_async
    ):
        """`_server_tool_status` returns "incomplete" for a failed call, and Open WebUI's
        converter only pairs a call with its result when the call sits at completed,
        failed or rejected. Published verbatim, a failed tool round vanishes from the
        replayed history entirely -- the model never learns the tool was tried.
        """
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(
            update={"THINKING_OUTPUT_MODE": "open_webui", "SHOW_TOOL_CARDS": True}
        )
        steps = [
            (0.0, {"type": "response.output_item.added",
                   "item": {"type": "openrouter:web_search", "id": "ws-1"}}),
            (1.0, {"type": "response.output_item.done",
                   "item": {"type": "openrouter:web_search", "id": "ws-1",
                            "action": {}, "httpStatus": 503}}),
            (0.2, {"type": "response.output_text.delta", "delta": "That failed."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)

        live = [
            (event.get("item") or {}).get("status")
            for event in _events_of(emitted, "response.output_item.added")
            if (event.get("item") or {}).get("type") == "function_call"
        ]
        assert live == ["incomplete"]

        completions = _events_of(emitted, "response.completed")
        assert completions
        published = (completions[-1].get("response") or {}).get("output") or []
        calls = [e for e in published if e.get("type") == "function_call"]
        assert calls, "the failed call must still be in the published array"
        assert all(e.get("status") in {"completed", "failed", "rejected"} for e in calls), (
            f"published statuses {[e.get('status') for e in calls]} include one Open WebUI "
            f"will not pair with its result"
        )


    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("http_status", "expected"),
        [(200, "completed"), (503, "failed")],
    )
    async def test_published_call_status_reflects_whether_the_tool_succeeded(
        self, monkeypatch, pipe_instance_async, http_status, expected
    ):
        """The published status must come from the result's outcome, not from the mere
        existence of a result. Recording a failed tool round as "completed" tells the
        model on every later turn that a tool it never got an answer from had worked.
        """
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(
            update={"THINKING_OUTPUT_MODE": "open_webui", "SHOW_TOOL_CARDS": True}
        )
        steps = [
            (0.0, {"type": "response.output_item.added",
                   "item": {"type": "openrouter:web_search", "id": "ws-1"}}),
            (1.0, {"type": "response.output_item.done",
                   "item": {"type": "openrouter:web_search", "id": "ws-1",
                            "action": {}, "httpStatus": http_status}}),
            (0.2, {"type": "response.output_text.delta", "delta": "Done."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)

        completions = _events_of(emitted, "response.completed")
        assert completions
        published = (completions[-1].get("response") or {}).get("output") or []
        calls = [e for e in published if e.get("type") == "function_call"]
        assert calls, "the call must be in the published array"
        assert [e.get("status") for e in calls] == [expected]


class TestSeededHistoryIsHealed:
    """A tool call stranded by an earlier turn must be repaired, not carried forward."""

    @staticmethod
    def _seed(monkeypatch, output):
        class _Chats:
            @staticmethod
            async def get_message_by_id_and_message_id(_chat_id, _message_id):
                return {"output": output}

        monkeypatch.setattr(streaming_core_mod, "Chats", _Chats)

    @staticmethod
    async def _run_with_chat(pipe, valves, steps, clock, monkeypatch) -> list[dict]:
        body = ResponsesBody(model="test/model", input=[], stream=True)
        monkeypatch.setattr(
            Pipe, "send_openrouter_streaming_request", _make_timed_stream(steps, clock)
        )
        emitted: list[dict] = []

        async def emitter(event):
            emitted.append(event)

        await pipe._streaming_handler._run_streaming_loop(
            body,
            valves,
            emitter,
            metadata={"model": {"id": "test"}, "chat_id": "c-1", "message_id": "m-1"},
            tools={},
            session=cast(Any, object()),
            user_id="user-123",
        )
        return emitted

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("result_status", "expected"),
        [("completed", "completed"), ("incomplete", "failed")],
    )
    async def test_a_paired_seeded_call_is_resolved_from_its_result(
        self, monkeypatch, pipe_instance_async, result_status, expected
    ):
        """The pipe rewrites the whole stored array every turn, so it must heal what it
        reads back -- otherwise a call stranded once stays stranded for the chat's life.
        """
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        self._seed(monkeypatch, [
            {"type": "function_call", "id": "old-1", "call_id": "old-1",
             "name": "t", "arguments": "{}", "status": "in_progress"},
            {"type": "function_call_output", "id": "fco-old", "call_id": "old-1",
             "output": [{"type": "input_text", "text": "x"}], "status": result_status},
        ])
        steps = [
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-1"}}),
            (1.0, {"type": "response.reasoning_text.delta", "item_id": "rs-1", "delta": "Thinking. "}),
            (0.2, {"type": "response.output_text.delta", "delta": "Hi."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await self._run_with_chat(pipe, valves, steps, clock, monkeypatch)

        completions = _events_of(emitted, "response.completed")
        assert completions
        published = (completions[-1].get("response") or {}).get("output") or []
        calls = [e for e in published if e.get("type") == "function_call"]
        assert [e.get("status") for e in calls] == [expected]

    @pytest.mark.asyncio
    async def test_an_unpaired_seeded_call_is_left_alone(
        self, monkeypatch, pipe_instance_async
    ):
        """Without a result there is no evidence of an outcome, so inventing one would
        tell the model a tool succeeded when nothing ever came back.
        """
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        self._seed(monkeypatch, [
            {"type": "function_call", "id": "old-2", "call_id": "old-2",
             "name": "t", "arguments": "{}", "status": "in_progress"},
        ])
        steps = [
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-1"}}),
            (1.0, {"type": "response.reasoning_text.delta", "item_id": "rs-1", "delta": "Thinking. "}),
            (0.2, {"type": "response.output_text.delta", "delta": "Hi."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await self._run_with_chat(pipe, valves, steps, clock, monkeypatch)

        completions = _events_of(emitted, "response.completed")
        assert completions
        published = (completions[-1].get("response") or {}).get("output") or []
        calls = [e for e in published if e.get("type") == "function_call"]
        assert [e.get("status") for e in calls] == ["in_progress"]


    @pytest.mark.asyncio
    @pytest.mark.parametrize("held_status", ["queued", "pending", "requires_approval"])
    async def test_a_call_awaiting_approval_is_never_resolved(
        self, monkeypatch, pipe_instance_async, held_status
    ):
        """Open WebUI runs its own tool-approval queue keyed on these statuses. Settling
        one here would steal the call out of that queue and the approval would never fire.
        """
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        self._seed(monkeypatch, [
            {"type": "function_call", "id": "held-1", "call_id": "held-1",
             "name": "t", "arguments": "{}", "status": held_status},
            {"type": "function_call_output", "id": "fco-held", "call_id": "held-1",
             "output": [{"type": "input_text", "text": "x"}], "status": "completed"},
        ])
        steps = [
            (0.0, {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs-1"}}),
            (1.0, {"type": "response.reasoning_text.delta", "item_id": "rs-1", "delta": "Thinking. "}),
            (0.2, {"type": "response.output_text.delta", "delta": "Hi."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await self._run_with_chat(pipe, valves, steps, clock, monkeypatch)

        completions = _events_of(emitted, "response.completed")
        assert completions
        published = (completions[-1].get("response") or {}).get("output") or []
        calls = [e for e in published if e.get("type") == "function_call"]
        assert [e.get("status") for e in calls] == [held_status]


class TestRetireDriftGuard:
    """T7: the legacy reasoning wire vocabulary must not return."""

    def test_production_sources_free_of_legacy_vocabulary(self):
        root = Path(__file__).resolve().parent.parent / "open_webui_openrouter_pipe" / "streaming"
        targets = [root / "streaming_core.py", root / "event_emitter.py"]
        banned = [
            r"reasoning:delta",
            r"reasoning:completed",
            r"reasoning_content",
            r"OWUI_REASONING_TIMING_BOUNDARY_MARKER",
            r"\breasoning_buffer\b",
        ]
        for path in targets:
            source = path.read_text(encoding="utf-8")
            for pattern in banned:
                assert not re.search(pattern, source), f"{pattern} found in {path.name}"

    def test_kept_state_still_present(self):
        source = (
            Path(__file__).resolve().parent.parent
            / "open_webui_openrouter_pipe"
            / "streaming"
            / "streaming_core.py"
        ).read_text(encoding="utf-8")
        assert "reasoning_stream_active" in source, "status-suppression gate state must be kept"
        assert "_monotonic" in source, "injectable monotonic alias must exist"
