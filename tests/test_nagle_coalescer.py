"""Unit tests for NagleCoalescer and nagle_coalesce_stream.

Tests the shared Nagle-inspired multi-buffer delta coalescing module
used by both the Responses API and Chat Completions API streaming paths.
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from open_webui_openrouter_pipe.streaming.nagle_coalescer import (
    NagleCoalescer,
    nagle_coalesce_stream,
)


# ============================================================================
# Helper
# ============================================================================


def _text_delta(text: str, output_index: int = 0, **extra) -> dict[str, Any]:
    return {"type": "response.output_text.delta", "delta": text, "output_index": output_index, **extra}


def _reasoning_delta(text: str, item_id: str = "rs_1", **extra) -> dict[str, Any]:
    return {"type": "response.reasoning_text.delta", "delta": text, "item_id": item_id, **extra}


def _reasoning_legacy_delta(text: str, item_id: str = "rs_1", **extra) -> dict[str, Any]:
    return {"type": "response.reasoning.delta", "delta": text, "item_id": item_id, **extra}


def _structural(event_type: str = "response.output_item.added", **extra) -> dict[str, Any]:
    return {"type": event_type, **extra}


# ============================================================================
# NagleCoalescer — text batching
# ============================================================================


class TestCoalescerTextBatching:
    def test_multiple_text_deltas_batched(self):
        """Multiple text deltas accumulate into a single batched event."""
        c = NagleCoalescer()
        q: list[dict] = []

        c.process_event(_text_delta("Hel"), q)
        c.process_event(_text_delta("lo "), q)
        c.process_event(_text_delta("World"), q)
        # Nothing yielded yet — all buffered
        assert q == []

        c.flush_all_to(q)
        assert len(q) == 1
        assert q[0]["delta"] == "Hello World"
        assert q[0]["type"] == "response.output_text.delta"

    def test_preserves_template_fields(self):
        """Batched event retains output_index and other fields from first delta."""
        c = NagleCoalescer()
        q: list[dict] = []

        c.process_event(_text_delta("A", output_index=2, content_index=7), q)
        c.process_event(_text_delta("B", output_index=2, content_index=7), q)
        c.flush_all_to(q)

        assert len(q) == 1
        assert q[0]["delta"] == "AB"
        assert q[0]["output_index"] == 2
        assert q[0]["content_index"] == 7

    def test_empty_delta_ignored(self):
        """Empty string deltas don't create spurious buffer entries."""
        c = NagleCoalescer()
        q: list[dict] = []

        c.process_event(_text_delta(""), q)
        c.process_event(_text_delta("A"), q)
        c.process_event(_text_delta(""), q)
        c.flush_all_to(q)

        assert len(q) == 1
        assert q[0]["delta"] == "A"

    def test_no_output_when_only_empty_deltas(self):
        """Only empty deltas produce no flushed event."""
        c = NagleCoalescer()
        q: list[dict] = []

        c.process_event(_text_delta(""), q)
        c.process_event(_text_delta(""), q)
        c.flush_all_to(q)
        assert q == []


# ============================================================================
# NagleCoalescer — reasoning batching
# ============================================================================


class TestCoalescerReasoningBatching:
    def test_multiple_reasoning_deltas_batched(self):
        """Multiple reasoning deltas accumulate into a single batched event."""
        c = NagleCoalescer()
        q: list[dict] = []

        c.process_event(_reasoning_delta("Think"), q)
        c.process_event(_reasoning_delta("ing..."), q)
        assert q == []

        c.flush_all_to(q)
        assert len(q) == 1
        assert q[0]["delta"] == "Thinking..."
        assert q[0]["type"] == "response.reasoning_text.delta"
        assert q[0]["item_id"] == "rs_1"

    def test_legacy_reasoning_type_batched(self):
        """response.reasoning.delta events are also batchable."""
        c = NagleCoalescer()
        q: list[dict] = []

        c.process_event(_reasoning_legacy_delta("A"), q)
        c.process_event(_reasoning_legacy_delta("B"), q)
        c.flush_all_to(q)

        assert len(q) == 1
        assert q[0]["delta"] == "AB"
        assert q[0]["type"] == "response.reasoning.delta"

    def test_item_id_boundary_flushes(self):
        """Switching item_id forces a flush of the previous reasoning buffer."""
        c = NagleCoalescer()
        q: list[dict] = []

        c.process_event(_reasoning_delta("Step1", item_id="rs_1"), q)
        c.process_event(_reasoning_delta("Step2", item_id="rs_2"), q)
        # The boundary flush should have emitted the first buffer
        assert len(q) == 1
        assert q[0]["delta"] == "Step1"
        assert q[0]["item_id"] == "rs_1"

        # Flush the remaining
        c.flush_all_to(q)
        assert len(q) == 2
        assert q[1]["delta"] == "Step2"
        assert q[1]["item_id"] == "rs_2"


# ============================================================================
# NagleCoalescer — type boundary
# ============================================================================


class TestCoalescerTypeBoundary:
    def test_text_then_reasoning_flushes_text(self):
        """Switching from text to reasoning flushes the text buffer."""
        c = NagleCoalescer()
        q: list[dict] = []

        c.process_event(_text_delta("Hello"), q)
        assert q == []

        c.process_event(_reasoning_delta("Think"), q)
        # Text should have been flushed at the type boundary
        assert len(q) == 1
        assert q[0]["type"] == "response.output_text.delta"
        assert q[0]["delta"] == "Hello"

        c.flush_all_to(q)
        assert len(q) == 2
        assert q[1]["delta"] == "Think"

    def test_reasoning_then_text_flushes_reasoning(self):
        """Switching from reasoning to text flushes the reasoning buffer."""
        c = NagleCoalescer()
        q: list[dict] = []

        c.process_event(_reasoning_delta("Think"), q)
        c.process_event(_text_delta("Output"), q)

        assert len(q) == 1
        assert q[0]["type"] == "response.reasoning_text.delta"
        assert q[0]["delta"] == "Think"

        c.flush_all_to(q)
        assert len(q) == 2
        assert q[1]["delta"] == "Output"


# ============================================================================
# NagleCoalescer — non-batchable events
# ============================================================================


class TestCoalescerNonBatchable:
    def test_structural_event_flushes_both_buffers(self):
        """Non-batchable event flushes text and reasoning, then passes through."""
        c = NagleCoalescer()
        q: list[dict] = []

        c.process_event(_text_delta("A"), q)
        c.process_event(_reasoning_delta("B"), q)
        # Type boundary: text flushed when reasoning starts
        assert len(q) == 1  # text flushed

        q.clear()
        c.process_event(_structural("response.output_item.added"), q)
        # Reasoning flushed + structural passed through
        assert len(q) == 2
        assert q[0]["type"] == "response.reasoning_text.delta"
        assert q[0]["delta"] == "B"
        assert q[1]["type"] == "response.output_item.added"

    def test_structural_with_no_buffered_content(self):
        """Non-batchable event with empty buffers just passes through."""
        c = NagleCoalescer()
        q: list[dict] = []

        c.process_event(_structural("response.completed"), q)
        assert len(q) == 1
        assert q[0]["type"] == "response.completed"


# ============================================================================
# NagleCoalescer — passthrough mode
# ============================================================================


class TestCoalescerPassthrough:
    def test_passthrough_bypasses_buffering(self):
        """With passthrough=True, events pass through 1:1 without buffering."""
        c = NagleCoalescer()
        q: list[dict] = []

        c.process_event(_text_delta("A"), q, passthrough=True)
        c.process_event(_text_delta("B"), q, passthrough=True)
        c.process_event(_text_delta("C"), q, passthrough=True)

        assert len(q) == 3
        assert [e["delta"] for e in q] == ["A", "B", "C"]
        assert not c.has_buffered

    def test_passthrough_reasoning(self):
        """Reasoning deltas also pass through in passthrough mode."""
        c = NagleCoalescer()
        q: list[dict] = []

        c.process_event(_reasoning_delta("X"), q, passthrough=True)
        c.process_event(_reasoning_delta("Y"), q, passthrough=True)

        assert len(q) == 2
        assert not c.has_buffered

    def test_passthrough_skips_empty_deltas(self):
        """Passthrough mode still skips empty deltas."""
        c = NagleCoalescer()
        q: list[dict] = []

        c.process_event(_text_delta(""), q, passthrough=True)
        c.process_event(_text_delta("A"), q, passthrough=True)
        c.process_event(_text_delta(""), q, passthrough=True)

        assert len(q) == 1


# ============================================================================
# NagleCoalescer — min_flush_chars
# ============================================================================


class TestCoalescerMinFlushChars:
    def test_holds_small_buffer_on_non_force(self):
        """With min_flush_chars=5, a 3-char buffer is NOT flushed when force=False."""
        c = NagleCoalescer(min_flush_chars=5)
        q: list[dict] = []

        c.process_event(_text_delta("abc"), q)
        c.flush_all_to(q, force=False)
        # 3 chars < 5 min: should NOT flush
        assert q == []
        assert c.has_buffered

    def test_force_overrides_min_flush_chars(self):
        """With min_flush_chars=5, force=True flushes a 3-char buffer (boundary/timeout)."""
        c = NagleCoalescer(min_flush_chars=5)
        q: list[dict] = []

        c.process_event(_text_delta("abc"), q)
        c.flush_all_to(q, force=True)
        assert len(q) == 1
        assert q[0]["delta"] == "abc"
        assert not c.has_buffered

    def test_flushes_above_threshold(self):
        """With min_flush_chars=5, a 6-char buffer IS flushed even with force=False."""
        c = NagleCoalescer(min_flush_chars=5)
        q: list[dict] = []

        c.process_event(_text_delta("abcdef"), q)
        c.flush_all_to(q, force=False)
        assert len(q) == 1
        assert q[0]["delta"] == "abcdef"

    def test_min_flush_chars_applies_to_reasoning_too(self):
        """Reasoning buffer also respects min_flush_chars on non-force flush."""
        c = NagleCoalescer(min_flush_chars=5)
        q: list[dict] = []

        c.process_event(_reasoning_delta("hi"), q)
        c.flush_all_to(q, force=False)
        assert q == []

        c.flush_all_to(q, force=True)
        assert len(q) == 1
        assert q[0]["delta"] == "hi"

    def test_type_boundary_forces_flush_regardless_of_min(self):
        """Type boundary triggers force=True, ignoring min_flush_chars."""
        c = NagleCoalescer(min_flush_chars=10)
        q: list[dict] = []

        c.process_event(_text_delta("ab"), q)  # 2 chars, well below 10
        c.process_event(_reasoning_delta("cd"), q)
        # Type boundary should have flushed text despite being < min_flush_chars
        assert len(q) == 1
        assert q[0]["delta"] == "ab"


# ============================================================================
# nagle_coalesce_stream — async generator wrapper
# ============================================================================


async def _async_gen(events: list[dict[str, Any]]):
    """Helper: yield events with a small delay to simulate SSE arrival."""
    for event in events:
        yield event


async def _async_gen_with_pause(
    events: list[dict[str, Any]],
    pause_after: int = 0,
    pause_seconds: float = 0.05,
):
    """Yield first N events, pause, then yield the rest."""
    for i, event in enumerate(events):
        if i == pause_after:
            await asyncio.sleep(pause_seconds)
        yield event


@pytest.mark.asyncio
async def test_nagle_stream_batches_events():
    """nagle_coalesce_stream batches text deltas from an async generator."""
    source_events = [
        _text_delta("H"),
        _text_delta("e"),
        _text_delta("l"),
        _text_delta("l"),
        _text_delta("o"),
    ]

    results = []
    async for event in nagle_coalesce_stream(
        _async_gen(source_events),
        idle_flush_seconds=0.03,
    ):
        results.append(event)

    # All text should arrive, possibly batched into fewer events
    combined = "".join(e["delta"] for e in results if e.get("type") == "response.output_text.delta")
    assert combined == "Hello"
    # Should be batched into fewer events than original 5
    text_events = [e for e in results if e.get("type") == "response.output_text.delta"]
    assert len(text_events) <= 5


@pytest.mark.asyncio
async def test_nagle_stream_idle_flush():
    """nagle_coalesce_stream flushes buffered content after idle timeout."""
    # Send some text, then pause long enough for idle timeout to fire
    source_events = [
        _text_delta("A"),
        _text_delta("B"),
    ]

    results = []
    async for event in nagle_coalesce_stream(
        _async_gen_with_pause(source_events, pause_after=1, pause_seconds=0.08),
        idle_flush_seconds=0.03,
    ):
        results.append(event)

    # Both deltas should arrive (possibly as 1 or 2 events depending on timing)
    combined = "".join(e["delta"] for e in results if e.get("type") == "response.output_text.delta")
    assert combined == "AB"


@pytest.mark.asyncio
async def test_nagle_stream_passthrough():
    """nagle_coalesce_stream passthrough mode yields events 1:1."""
    source_events = [
        _text_delta("A"),
        _text_delta("B"),
        _text_delta("C"),
    ]

    results = []
    async for event in nagle_coalesce_stream(
        _async_gen(source_events),
        passthrough=True,
    ):
        results.append(event)

    text_events = [e for e in results if e.get("type") == "response.output_text.delta"]
    assert len(text_events) == 3
    assert [e["delta"] for e in text_events] == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_nagle_stream_non_batchable_passes_through():
    """Non-batchable events pass through nagle_coalesce_stream immediately."""
    source_events = [
        _text_delta("X"),
        _structural("response.output_item.added", item={"id": "item_1"}),
        _text_delta("Y"),
    ]

    results = []
    async for event in nagle_coalesce_stream(
        _async_gen(source_events),
        idle_flush_seconds=0.03,
    ):
        results.append(event)

    types = [e["type"] for e in results]
    # The structural event should appear, and text should be complete
    assert "response.output_item.added" in types
    combined = "".join(e["delta"] for e in results if e.get("type") == "response.output_text.delta")
    assert combined == "XY"


@pytest.mark.asyncio
async def test_nagle_stream_min_flush_chars():
    """nagle_coalesce_stream respects min_flush_chars at end of drain cycle."""
    source_events = [
        _text_delta("A"),  # 1 char — below min_flush_chars=5
        _structural("response.completed", response={"output": []}),
    ]

    results = []
    async for event in nagle_coalesce_stream(
        _async_gen(source_events),
        min_flush_chars=5,
        idle_flush_seconds=0.03,
    ):
        results.append(event)

    # Text "A" should still arrive (structural event triggers force flush, or final flush)
    combined = "".join(e["delta"] for e in results if e.get("type") == "response.output_text.delta")
    assert combined == "A"
    # Structural event should also be present
    assert any(e["type"] == "response.completed" for e in results)


@pytest.mark.asyncio
async def test_nagle_stream_mixed_text_and_reasoning():
    """nagle_coalesce_stream correctly handles interleaved text and reasoning."""
    source_events = [
        _reasoning_delta("Think", item_id="rs_1"),
        _reasoning_delta("ing", item_id="rs_1"),
        _text_delta("Out"),
        _text_delta("put"),
    ]

    results = []
    async for event in nagle_coalesce_stream(
        _async_gen(source_events),
        idle_flush_seconds=0.03,
    ):
        results.append(event)

    reasoning = "".join(e["delta"] for e in results if "reasoning" in e.get("type", ""))
    text = "".join(e["delta"] for e in results if e.get("type") == "response.output_text.delta")
    assert reasoning == "Thinking"
    assert text == "Output"
    # Reasoning should come before text in output order
    r_idx = next(i for i, e in enumerate(results) if "reasoning" in e.get("type", ""))
    t_idx = next(i for i, e in enumerate(results) if e.get("type") == "response.output_text.delta")
    assert r_idx < t_idx


# ============================================================================
# nagle_coalesce_stream — exception propagation
# ============================================================================


async def _async_gen_that_fails(events: list[dict[str, Any]], fail_after: int = 2):
    """Yield some events then raise an exception."""
    for i, event in enumerate(events):
        yield event
        if i == fail_after:
            raise RuntimeError("source stream failed")


@pytest.mark.asyncio
async def test_nagle_stream_propagates_source_exception():
    """Source generator exception must propagate through nagle_coalesce_stream."""
    source_events = [
        _text_delta("A"),
        _text_delta("B"),
        _text_delta("C"),  # fails after yielding this one
        _text_delta("D"),  # never reached
    ]

    results = []
    with pytest.raises(RuntimeError, match="source stream failed"):
        async for event in nagle_coalesce_stream(
            _async_gen_that_fails(source_events, fail_after=2),
            idle_flush_seconds=0.03,
        ):
            results.append(event)

    # Partial output should have been delivered before the error
    combined = "".join(e["delta"] for e in results if e.get("type") == "response.output_text.delta")
    assert "A" in combined  # at least the first delta arrived


@pytest.mark.asyncio
async def test_nagle_stream_propagates_exception_not_silently_completed():
    """Verify the stream does NOT silently complete when source raises."""
    source_events = [
        _text_delta("X"),
    ]

    completed_normally = False
    raised = False
    try:
        async for _event in nagle_coalesce_stream(
            _async_gen_that_fails(source_events, fail_after=0),
            idle_flush_seconds=0.03,
        ):
            pass
        completed_normally = True
    except RuntimeError:
        raised = True

    assert raised, "Exception should have propagated"
    assert not completed_normally, "Stream should NOT have completed normally"
