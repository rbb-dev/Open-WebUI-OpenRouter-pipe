
import asyncio
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import Pipe, ResponsesBody


def _fake_stream(events):
    async def fake_stream(self, session, request_body, **_kwargs):
        for event in events:
            yield event
    return fake_stream


FUSION_EVENTS = [
    {"type": "response.created", "response": {"model": "anthropic/claude-opus"}},
    {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}},
    {"type": "response.output_text.done", "output_index": 0, "text": "I'll research this."},
    {"type": "response.output_item.added", "output_index": 1,
     "item": {"type": "openrouter:fusion", "status": "in_progress"}},
    {"type": "response.fusion_call.in_progress", "output_index": 1},
    {"type": "response.fusion_call.panel.added", "output_index": 1, "model": "~google/gemini-flash"},
    {"type": "response.fusion_call.panel.completed", "output_index": 1,
     "model": "~google/gemini-flash", "content": "# Answer A"},
    {"type": "response.fusion_call.analysis.in_progress", "output_index": 1, "judge_model": "anthropic/claude"},
    {"type": "response.fusion_call.analysis.completed", "output_index": 1,
     "analysis": {"consensus": ["agree"], "contradictions": [], "partial_coverage": [],
                  "unique_insights": [], "blind_spots": []}},
    {"type": "response.fusion_call.completed", "output_index": 1},
    {"type": "response.output_item.added", "output_index": 2, "item": {"type": "message"}},
    {"type": "response.output_text.delta", "output_index": 2, "delta": "The answer is 42."},
    {"type": "response.output_text.done", "output_index": 2, "text": "The answer is 42."},
    {"type": "response.completed",
     "response": {"output": [], "usage": {"cost": 0.4, "total_tokens": 100}, "model": "anthropic/claude-opus"}},
]


async def _run(pipe, monkeypatch, *, fusion_live_enabled, events=None, plugins=None, endpoint_override=None):
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _fake_stream(events or FUSION_EVENTS))
    body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)
    if plugins is not None:
        body.plugins = plugins
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    result = await pipe._streaming_handler._run_streaming_loop(
        body, pipe.valves, emitter, metadata={}, tools={},
        session=cast(Any, object()), user_id="u", fusion_live_enabled=fusion_live_enabled,
        endpoint_override=endpoint_override,
    )
    return result, emitted


REASONING_DELTA_EVENTS = [
    {"type": "response.created", "response": {"model": "anthropic/claude-opus"}},
    {"type": "response.output_item.added", "output_index": 0,
     "item": {"type": "openrouter:fusion", "status": "in_progress"}},
    {"type": "response.fusion_call.in_progress", "output_index": 0},
    {"type": "response.fusion_call.panel.added", "output_index": 0, "model": "~google/gemini-flash"},
    {"type": "response.fusion_call.panel.reasoning.delta", "output_index": 0,
     "model": "~google/gemini-flash", "delta": "r" * 300},
    {"type": "response.fusion_call.panel.reasoning.delta", "output_index": 0,
     "model": "~google/gemini-flash", "delta": "s" * 300},
    {"type": "response.fusion_call.panel.delta", "output_index": 0,
     "model": "~google/gemini-flash", "delta": "pending-answer-tail"},
    {"type": "response.fusion_call.panel.completed", "output_index": 0,
     "model": "~google/gemini-flash", "content": "# Answer A"},
    {"type": "response.fusion_call.panel.added", "output_index": 0, "model": "~x/failed-model"},
    {"type": "response.fusion_call.panel.reasoning.delta", "output_index": 0,
     "model": "~x/failed-model", "delta": "orphan-think"},
    {"type": "response.fusion_call.analysis.in_progress", "output_index": 0, "judge_model": "anthropic/claude"},
    {"type": "response.fusion_call.analysis.completed", "output_index": 0,
     "analysis": {"consensus": ["agree"], "contradictions": [], "partial_coverage": [],
                  "unique_insights": [], "blind_spots": []}},
    {"type": "response.fusion_call.completed", "output_index": 0},
    {"type": "response.output_item.added", "output_index": 1, "item": {"type": "message"}},
    {"type": "response.output_text.done", "output_index": 1, "text": "The answer is 42."},
    {"type": "response.completed",
     "response": {"output": [], "usage": {"cost": 0.4, "total_tokens": 100}, "model": "anthropic/claude-opus"}},
]


class TestFusionPanelDeltaStreaming:

    @staticmethod
    def _fusion_events(emitted):
        return [e["data"]["event"] for e in emitted if e.get("type") == "fusion:event"]

    @pytest.mark.asyncio
    async def test_reasoning_deltas_batched_not_relayed_one_by_one(
        self, monkeypatch, pipe_instance_async
    ):
        _result, emitted = await _run(pipe_instance_async, monkeypatch,
                                      fusion_live_enabled=True, events=REASONING_DELTA_EVENTS)
        reasoning = [e for e in self._fusion_events(emitted)
                     if e.get("type") == "response.fusion_call.panel.reasoning.delta"
                     and e.get("model") == "~google/gemini-flash"]
        assert len(reasoning) == 1
        assert reasoning[0]["delta"] == "r" * 300 + "s" * 300

    @pytest.mark.asyncio
    async def test_panel_completed_carries_accumulated_reasoning(
        self, monkeypatch, pipe_instance_async
    ):
        _result, emitted = await _run(pipe_instance_async, monkeypatch,
                                      fusion_live_enabled=True, events=REASONING_DELTA_EVENTS)
        completed = [e for e in self._fusion_events(emitted)
                     if e.get("type") == "response.fusion_call.panel.completed"]
        assert completed
        assert completed[0].get("reasoning") == "r" * 300 + "s" * 300
        assert completed[0].get("content") == "# Answer A"

    @pytest.mark.asyncio
    async def test_pending_answer_deltas_discarded_at_completed(
        self, monkeypatch, pipe_instance_async
    ):
        _result, emitted = await _run(pipe_instance_async, monkeypatch,
                                      fusion_live_enabled=True, events=REASONING_DELTA_EVENTS)
        answer_deltas = [e for e in self._fusion_events(emitted)
                        if e.get("type") == "response.fusion_call.panel.delta"]
        assert answer_deltas == []

    @pytest.mark.asyncio
    async def test_orphan_reasoning_flushed_at_analysis_start(
        self, monkeypatch, pipe_instance_async
    ):
        _result, emitted = await _run(pipe_instance_async, monkeypatch,
                                      fusion_live_enabled=True, events=REASONING_DELTA_EVENTS)
        orphan = [e for e in self._fusion_events(emitted)
                  if e.get("type") == "response.fusion_call.panel.reasoning.delta"
                  and e.get("model") == "~x/failed-model"]
        assert len(orphan) == 1
        assert orphan[0]["delta"] == "orphan-think"

    @pytest.mark.asyncio
    async def test_source_fixture_events_never_mutated(
        self, monkeypatch, pipe_instance_async
    ):
        src_completed = next(e for e in REASONING_DELTA_EVENTS
                             if e.get("type") == "response.fusion_call.panel.completed")
        await _run(pipe_instance_async, monkeypatch,
                   fusion_live_enabled=True, events=REASONING_DELTA_EVENTS)
        assert "reasoning" not in src_completed


PLAIN_EVENTS = [
    {"type": "response.created", "response": {"model": "anthropic/claude-opus"}},
    {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}},
    {"type": "response.output_text.delta", "output_index": 0, "delta": "Plain answer."},
    {"type": "response.output_text.done", "output_index": 0, "text": "Plain answer."},
    {"type": "response.completed",
     "response": {"output": [], "usage": {"cost": 0.01, "total_tokens": 10}, "model": "anthropic/claude-opus"}},
]


def _tripwire_records(caplog):
    return [r for r in caplog.records if "structured fusion events" in r.getMessage()]


class TestFusionActivationTripwire:

    @pytest.mark.asyncio
    async def test_warns_when_fusion_never_opens_despite_enabled_plugin(
        self, monkeypatch, pipe_instance_async, caplog
    ):
        import logging
        with caplog.at_level(logging.WARNING):
            await _run(pipe_instance_async, monkeypatch, fusion_live_enabled=True,
                       events=PLAIN_EVENTS, plugins=[{"id": "fusion"}])
        assert len(_tripwire_records(caplog)) == 1

    @pytest.mark.asyncio
    async def test_silent_when_no_fusion_plugin_outbound(
        self, monkeypatch, pipe_instance_async, caplog
    ):
        import logging
        with caplog.at_level(logging.WARNING):
            await _run(pipe_instance_async, monkeypatch, fusion_live_enabled=True,
                       events=PLAIN_EVENTS, plugins=None)
        assert not _tripwire_records(caplog)

    @pytest.mark.asyncio
    async def test_silent_when_fusion_plugin_disabled(
        self, monkeypatch, pipe_instance_async, caplog
    ):
        import logging
        with caplog.at_level(logging.WARNING):
            await _run(pipe_instance_async, monkeypatch, fusion_live_enabled=True,
                       events=PLAIN_EVENTS, plugins=[{"id": "fusion", "enabled": False}])
        assert not _tripwire_records(caplog)

    @pytest.mark.asyncio
    async def test_silent_on_chat_completions_fallback(
        self, monkeypatch, pipe_instance_async, caplog
    ):
        import logging
        with caplog.at_level(logging.WARNING):
            await _run(pipe_instance_async, monkeypatch, fusion_live_enabled=True,
                       events=PLAIN_EVENTS, plugins=[{"id": "fusion"}],
                       endpoint_override="chat_completions")
        assert not _tripwire_records(caplog)

    @pytest.mark.asyncio
    async def test_silent_when_fusion_actually_ran(
        self, monkeypatch, pipe_instance_async, caplog
    ):
        import logging
        with caplog.at_level(logging.WARNING):
            await _run(pipe_instance_async, monkeypatch, fusion_live_enabled=True,
                       plugins=[{"id": "fusion"}])
        assert not _tripwire_records(caplog)

    @pytest.mark.asyncio
    async def test_silent_on_internal_chat_fallback(
        self, monkeypatch, pipe_instance_async, caplog
    ):
        import logging
        events = [{"type": "openrouter_pipe.chat_fallback"}] + PLAIN_EVENTS
        with caplog.at_level(logging.WARNING):
            await _run(pipe_instance_async, monkeypatch, fusion_live_enabled=True,
                       events=events, plugins=[{"id": "fusion"}])
        assert not _tripwire_records(caplog)


@pytest.mark.asyncio
async def test_fusion_armed_emits_embed_once_and_streams_fusion_events(monkeypatch, pipe_instance_async):
    _result, emitted = await _run(pipe_instance_async, monkeypatch, fusion_live_enabled=True)
    embeds = [e for e in emitted if e.get("type") == "embeds"]
    assert len(embeds) == 1
    assert not embeds[0]["data"].get("replace")
    assert "OpenRouter Fusion" in embeds[0]["data"]["embeds"][0]
    fusion_events = [e for e in emitted if e.get("type") == "fusion:event"]
    types = [e["data"]["event"].get("type") for e in fusion_events]
    assert "response.fusion_call.panel.completed" in types
    assert "response.output_text.done" in types


@pytest.mark.asyncio
async def test_fusion_off_emits_no_embeds(monkeypatch, pipe_instance_async):
    result, emitted = await _run(pipe_instance_async, monkeypatch, fusion_live_enabled=False)
    assert not [e for e in emitted if e.get("type") == "embeds"]
    delta_text = "".join(
        e.get("data", {}).get("content", "")
        for e in emitted if e.get("type") == "chat:message:delta"
    )
    assert "The answer is 42." in delta_text
    assert "The answer is 42." in (result or "")


@pytest.mark.asyncio
async def test_fusion_tool_card_status_suppressed(monkeypatch, pipe_instance_async):
    _result, emitted = await _run(pipe_instance_async, monkeypatch, fusion_live_enabled=True)
    descriptions = " ".join(
        e.get("data", {}).get("description", "")
        for e in emitted if e.get("type") == "status"
    )
    assert "Running openrouter:fusion" not in descriptions


@pytest.mark.asyncio
async def test_fusion_final_answer_streams_over_socket_and_persists_collapsed(monkeypatch, pipe_instance_async):
    result, emitted = await _run(pipe_instance_async, monkeypatch, fusion_live_enabled=True)
    fusion_answer = "".join(
        (e["data"]["event"].get("delta") or e["data"]["event"].get("text") or "")
        for e in emitted if e.get("type") == "fusion:event"
        and str(e["data"]["event"].get("type", "")).startswith("response.output_text")
    )
    assert "The answer is 42." in fusion_answer
    assert (result or "").startswith("<details")
    assert "<summary>Final answer</summary>" in (result or "")
    assert "The answer is 42." in (result or "")
    assert not [e for e in emitted if e.get("type") == "chat:message:delta"], (
        "fusion-armed turn must emit zero chat:message:delta frames"
    )


def _native_message_items(emitted):
    added = [
        e for e in emitted
        if e.get("type") == "response.output_item.added"
        and (e.get("item") or {}).get("type") == "message"
    ]
    done = [
        e for e in emitted
        if e.get("type") == "response.output_item.done"
        and (e.get("item") or {}).get("type") == "message"
    ]
    return added, done


@pytest.mark.asyncio
async def test_fusion_emits_native_message_item_with_clean_answer(monkeypatch, pipe_instance_async):
    result, emitted = await _run(pipe_instance_async, monkeypatch, fusion_live_enabled=True)

    added, done = _native_message_items(emitted)
    assert len(added) == 1
    assert len(done) == 1
    item = added[0]["item"]
    assert item["role"] == "assistant"
    assert item["status"] == "completed"
    assert item["id"].startswith("msg-")
    assert item["content"] == [{"type": "output_text", "text": "The answer is 42."}]
    assert "<details" not in item["content"][0]["text"]
    assert done[0]["item"] == item

    assert (result or "").startswith("<details")
    assert "The answer is 42." in (result or "")

    final_frames = [
        e for e in emitted
        if e.get("type") == "chat:completion" and (e.get("data") or {}).get("done") is True
    ]
    assert final_frames
    assert final_frames[-1]["data"].get("content") is None


@pytest.mark.asyncio
async def test_fusion_disabled_emits_no_native_message_item(monkeypatch, pipe_instance_async):
    events = [
        {"type": "response.created", "response": {"model": "anthropic/claude"}},
        {"type": "response.output_text.delta", "output_index": 0, "delta": "plain "},
        {"type": "response.output_text.done", "output_index": 0, "text": "plain answer"},
        {"type": "response.completed",
         "response": {"output": [], "usage": {"total_tokens": 5}, "model": "anthropic/claude"}},
    ]
    result, emitted = await _run(
        pipe_instance_async, monkeypatch, fusion_live_enabled=False, events=events,
    )
    added, done = _native_message_items(emitted)
    assert added == []
    assert done == []
    assert "plain" in (result or "")


@pytest.mark.asyncio
async def test_fusion_incomplete_run_emits_no_native_message_item(monkeypatch, pipe_instance_async):
    events = [e for e in FUSION_EVENTS if e.get("output_index") != 1]
    result, emitted = await _run(
        pipe_instance_async, monkeypatch, fusion_live_enabled=True, events=events,
    )
    added, done = _native_message_items(emitted)
    assert added == []
    assert done == []
    assert "<details" not in (result or "")


@pytest.mark.asyncio
async def test_fusion_errored_run_emits_no_native_message_item(monkeypatch, pipe_instance_async):
    events = [e for e in FUSION_EVENTS if e.get("type") != "response.completed"]

    def _raising_stream(evts):
        async def raising_stream(self, session, request_body, **_kwargs):
            for event in evts:
                yield event
            raise RuntimeError("stream exploded")
        return raising_stream

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _raising_stream(events))
    body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    result = await pipe_instance_async._streaming_handler._run_streaming_loop(
        body, pipe_instance_async.valves, emitter, metadata={}, tools={},
        session=cast(Any, object()), user_id="u", fusion_live_enabled=True,
    )
    added, done = _native_message_items(emitted)
    assert added == []
    assert done == []
    assert "<details" not in (result or "")


@pytest.mark.asyncio
async def test_fusion_missing_analysis_synthesizes_completed_in_persisted_snapshot(monkeypatch, pipe_instance_async):
    import json

    import open_webui_openrouter_pipe.streaming.streaming_core as sc

    events = [e for e in FUSION_EVENTS if e.get("type") != "response.fusion_call.analysis.completed"]
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _fake_stream(events))

    captured: dict = {}

    class _CapturingChats:
        @staticmethod
        async def get_message_by_id_and_message_id(*_a, **_k):
            return None

        @staticmethod
        async def upsert_message_to_chat_by_id_and_message_id(_chat_id, _message_id, payload, *_a, **_k):
            captured["payload"] = payload
            return None

    monkeypatch.setattr(sc, "Chats", _CapturingChats)

    body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)

    async def emitter(_event):
        return None

    await pipe_instance_async._streaming_handler._run_streaming_loop(
        body, pipe_instance_async.valves, emitter,
        metadata={"chat_id": "c1", "message_id": "m1"}, tools={},
        session=cast(Any, object()), user_id="u", fusion_live_enabled=True,
    )

    embeds = (captured.get("payload") or {}).get("embeds") or []
    html = next((e for e in embeds if isinstance(e, str) and "OpenRouter Fusion" in e), "")
    assert html, "no persisted fusion embed captured"

    marker = "var EVENTS = "
    decoded, _ = json.JSONDecoder().raw_decode(html, html.index(marker) + len(marker))
    analysis_events = [
        e for e in decoded
        if isinstance(e, dict) and e.get("type") == "response.fusion_call.analysis.completed"
    ]
    assert len(analysis_events) == 1
    assert analysis_events[0]["analysis"] == {
        "consensus": [], "contradictions": [], "partial_coverage": [],
        "unique_insights": [], "blind_spots": [],
    }
    types = [e.get("type") for e in decoded if isinstance(e, dict)]
    assert types.index("response.fusion_call.analysis.completed") < types.index("response.completed")
    assert "No consensus was produced." in html


@pytest.mark.asyncio
async def test_fusion_delta_preamble_excluded_from_answer(monkeypatch, pipe_instance_async):
    events = [
        {"type": "response.created", "response": {"model": "anthropic/claude"}},
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}},
        {"type": "response.output_text.delta", "output_index": 0, "delta": "Let me "},
        {"type": "response.output_text.delta", "output_index": 0, "delta": "research this."},
        {"type": "response.output_text.done", "output_index": 0, "text": "Let me research this."},
        {"type": "response.output_item.added", "output_index": 1,
         "item": {"type": "openrouter:fusion", "status": "in_progress"}},
        {"type": "response.fusion_call.in_progress", "output_index": 1},
        {"type": "response.fusion_call.completed", "output_index": 1},
        {"type": "response.output_item.added", "output_index": 2, "item": {"type": "message"}},
        {"type": "response.output_text.delta", "output_index": 2, "delta": "The answer is 42."},
        {"type": "response.output_text.done", "output_index": 2, "text": "The answer is 42."},
        {"type": "response.completed",
         "response": {"output": [], "usage": {}, "model": "anthropic/claude"}},
    ]
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _fake_stream(events))
    body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    result = await pipe_instance_async._streaming_handler._run_streaming_loop(
        body, pipe_instance_async.valves, emitter, metadata={}, tools={},
        session=cast(Any, object()), user_id="u", fusion_live_enabled=True,
    )
    assert "The answer is 42." in (result or "")
    assert "Let me research this." not in (result or "")
    assert "<summary>Final answer</summary>" in (result or "")
    assert not [e for e in emitted if e.get("type") == "chat:message:delta"], (
        "fusion-armed turn must emit zero chat:message:delta frames"
    )
    embeds = [e for e in emitted if e.get("type") == "embeds"]
    seed_html = embeds[0]["data"]["embeds"][0] if embeds else ""
    assert "research this." in seed_html, "preamble must reach the panel via the embed seed"


@pytest.mark.asyncio
async def test_fusion_second_fusion_item_does_not_wipe_answer(monkeypatch, pipe_instance_async):
    events = [
        {"type": "response.created", "response": {"model": "anthropic/claude"}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"type": "openrouter:fusion", "status": "in_progress"}},
        {"type": "response.fusion_call.in_progress", "output_index": 0},
        {"type": "response.fusion_call.completed", "output_index": 0},
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "message"}},
        {"type": "response.output_text.delta", "output_index": 1, "delta": "ANSWER-ONE. "},
        {"type": "response.output_text.done", "output_index": 1, "text": "ANSWER-ONE. "},
        {"type": "response.output_item.added", "output_index": 2,
         "item": {"type": "openrouter:fusion", "status": "in_progress"}},
        {"type": "response.fusion_call.completed", "output_index": 2},
        {"type": "response.output_item.added", "output_index": 3, "item": {"type": "message"}},
        {"type": "response.output_text.delta", "output_index": 3, "delta": "ANSWER-TWO."},
        {"type": "response.output_text.done", "output_index": 3, "text": "ANSWER-TWO."},
        {"type": "response.completed",
         "response": {"output": [], "usage": {}, "model": "anthropic/claude"}},
    ]
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _fake_stream(events))
    body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    result = await pipe_instance_async._streaming_handler._run_streaming_loop(
        body, pipe_instance_async.valves, emitter, metadata={}, tools={},
        session=cast(Any, object()), user_id="u", fusion_live_enabled=True,
    )
    assert "ANSWER-ONE." in (result or "")
    assert "ANSWER-TWO." in (result or "")


@pytest.mark.asyncio
async def test_fusion_deferred_timer_seeds_roster_before_completion(monkeypatch, pipe_instance_async):
    roster = [
        {"type": "response.created", "response": {"model": "anthropic/claude-opus"}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"type": "openrouter:fusion", "status": "in_progress"}},
        {"type": "response.fusion_call.in_progress", "output_index": 0},
        {"type": "response.fusion_call.panel.added", "output_index": 0, "model": "~google/gemini-flash"},
        {"type": "response.fusion_call.panel.added", "output_index": 0, "model": "~openai/gpt-5"},
    ]
    tail = [
        {"type": "response.fusion_call.panel.completed", "output_index": 0,
         "model": "~google/gemini-flash", "content": "# Answer A"},
        {"type": "response.fusion_call.completed", "output_index": 0},
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "message"}},
        {"type": "response.output_text.done", "output_index": 1, "text": "The answer is 42."},
        {"type": "response.completed",
         "response": {"output": [], "usage": {}, "model": "anthropic/claude-opus"}},
    ]

    emitted: list[dict] = []
    seen_at_gap: dict[str, list[dict]] = {}

    async def fake_stream(self, session, request_body, **_kwargs):
        for event in roster:
            yield event
        await asyncio.sleep(0.15)
        seen_at_gap["emitted"] = list(emitted)
        for event in tail:
            yield event

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", fake_stream)
    body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)

    async def emitter(event):
        emitted.append(event)

    await pipe_instance_async._streaming_handler._run_streaming_loop(
        body, pipe_instance_async.valves, emitter, metadata={}, tools={},
        session=cast(Any, object()), user_id="u", fusion_live_enabled=True,
    )

    embeds = [e for e in emitted if e.get("type") == "embeds"]
    assert len(embeds) == 1
    mid_embeds = [e for e in seen_at_gap.get("emitted", []) if e.get("type") == "embeds"]
    assert len(mid_embeds) == 1, "deferred timer must emit the seed during the roster gap"
    seed = mid_embeds[0]["data"]["embeds"][0]
    assert "gemini-flash" in seed, "seed must capture the roster panel.added burst"
    assert "gpt-5" in seed
    assert "Answer A" not in seed, "seed must predate the post-gap panel.completed"
    fusion_types = [
        e["data"]["event"].get("type") for e in emitted if e.get("type") == "fusion:event"
    ]
    assert "response.fusion_call.panel.completed" in fusion_types


@pytest.mark.asyncio
async def test_completion_without_output_still_emits_final_status(monkeypatch, pipe_instance_async):
    events = [
        {"type": "response.created", "response": {"model": "anthropic/claude-opus"}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"type": "openrouter:fusion", "status": "in_progress"}},
        {"type": "response.fusion_call.in_progress", "output_index": 0},
        {"type": "response.fusion_call.completed", "output_index": 0},
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "message"}},
        {"type": "response.output_text.done", "output_index": 1, "text": "The answer."},
        {"type": "response.completed",
         "response": {"usage": {"cost": 0.5, "total_tokens": 200}, "model": "anthropic/claude-opus"}},
    ]
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _fake_stream(events))
    body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe_instance_async._streaming_handler._run_streaming_loop(
        body, pipe_instance_async.valves, emitter, metadata={}, tools={},
        session=cast(Any, object()), user_id="u", fusion_live_enabled=True,
    )
    final_statuses = [
        e for e in emitted
        if e.get("type") == "status"
        and e.get("data", {}).get("done")
        and "Cost" in (e.get("data", {}).get("description") or "")
    ]
    assert final_statuses, "final cost/stats status must emit even when the completion lacks 'output'"


def _fusion_events_of_type(emitted, etype):
    return [
        e["data"]["event"]
        for e in emitted
        if e.get("type") == "fusion:event" and e["data"]["event"].get("type") == etype
    ]


def _source_events(emitted):
    return [e for e in emitted if e.get("type") == "source"]


_EMPTY_ANALYSIS = {
    "consensus": [], "contradictions": [], "partial_coverage": [],
    "unique_insights": [], "blind_spots": [],
}


@pytest.mark.asyncio
async def test_fusion_missing_analysis_emits_synthetic_analysis_over_socket(monkeypatch, pipe_instance_async):
    events = [e for e in FUSION_EVENTS if e.get("type") != "response.fusion_call.analysis.completed"]
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _fake_stream(events))
    body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe_instance_async._streaming_handler._run_streaming_loop(
        body, pipe_instance_async.valves, emitter, metadata={}, tools={},
        session=cast(Any, object()), user_id="u", fusion_live_enabled=True,
    )
    synthetic = _fusion_events_of_type(emitted, "response.fusion_call.analysis.completed")
    assert len(synthetic) == 1, "live spinner must receive exactly one synthetic analysis.completed"
    assert synthetic[0]["analysis"] == _EMPTY_ANALYSIS


@pytest.mark.asyncio
async def test_fusion_output_item_done_synthesizes_analysis_and_emits_sources(monkeypatch, pipe_instance_async):
    sources = [
        {"url": "https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe",
         "title": "rbb-dev/Open-WebUI-OpenRouter-pipe - GitHub"},
        {"url": "https://docs.openwebui.com/", "title": "Connect a Provider"},
    ]
    events = [
        {"type": "response.created", "response": {"model": "anthropic/claude-opus"}},
        {"type": "response.output_item.added", "output_index": 1,
         "item": {"type": "openrouter:fusion", "status": "in_progress"}},
        {"type": "response.fusion_call.in_progress", "output_index": 1},
        {"type": "response.fusion_call.panel.completed", "output_index": 1,
         "model": "~google/gemini-flash", "content": "# Answer A"},
        {"type": "response.fusion_call.analysis.in_progress", "output_index": 1,
         "judge_model": "anthropic/claude"},
        {"type": "response.fusion_call.completed", "output_index": 1},
        {"type": "response.output_item.done", "output_index": 1,
         "item": {"type": "openrouter:fusion", "status": "completed",
                  "responses": [{"model": "~google/gemini-flash", "content": "# Answer A"}],
                  "sources": sources}},
        {"type": "response.output_item.added", "output_index": 2, "item": {"type": "message"}},
        {"type": "response.output_text.delta", "output_index": 2, "delta": "The answer is 42."},
        {"type": "response.output_text.done", "output_index": 2, "text": "The answer is 42."},
        {"type": "response.completed",
         "response": {"output": [], "usage": {}, "model": "anthropic/claude-opus"}},
    ]
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _fake_stream(events))
    body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe_instance_async._streaming_handler._run_streaming_loop(
        body, pipe_instance_async.valves, emitter, metadata={}, tools={},
        session=cast(Any, object()), user_id="u", fusion_live_enabled=True,
    )

    synthetic = _fusion_events_of_type(emitted, "response.fusion_call.analysis.completed")
    assert len(synthetic) == 1
    assert synthetic[0]["analysis"] == _EMPTY_ANALYSIS

    source_events = _source_events(emitted)
    urls = {e["data"]["source"]["url"] for e in source_events}
    assert urls == {s["url"] for s in sources}
    docs = " ".join(d for e in source_events for d in e["data"]["document"])
    assert "rbb-dev/Open-WebUI-OpenRouter-pipe - GitHub" in docs


@pytest.mark.asyncio
async def test_fusion_sources_dedup_and_persist_to_sources_field(monkeypatch, pipe_instance_async):
    import open_webui_openrouter_pipe.streaming.streaming_core as sc

    dup_url = "https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe"
    events = [
        {"type": "response.created", "response": {"model": "anthropic/claude-opus"}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"type": "openrouter:fusion", "status": "in_progress"}},
        {"type": "response.fusion_call.in_progress", "output_index": 0},
        {"type": "response.fusion_call.completed", "output_index": 0},
        {"type": "response.output_item.done", "output_index": 0,
         "item": {"type": "openrouter:fusion", "status": "completed",
                  "sources": [
                      {"url": dup_url, "title": "GitHub repo"},
                      {"url": "https://docs.openwebui.com/", "title": "Docs"},
                      {"url": dup_url, "title": "GitHub repo (again)"},
                  ]}},
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "message"}},
        {"type": "response.output_text.done", "output_index": 1, "text": "Answer."},
        {"type": "response.completed",
         "response": {"output": [], "usage": {}, "model": "anthropic/claude-opus"}},
    ]
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _fake_stream(events))

    persisted: list[dict] = []

    class _CollectingChats:
        @staticmethod
        async def get_message_by_id_and_message_id(*_a, **_k):
            return None

        @staticmethod
        async def upsert_message_to_chat_by_id_and_message_id(_chat_id, _message_id, payload, *_a, **_k):
            persisted.append(payload)
            return None

    monkeypatch.setattr(sc, "Chats", _CollectingChats)

    body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe_instance_async._streaming_handler._run_streaming_loop(
        body, pipe_instance_async.valves, emitter,
        metadata={"chat_id": "c1", "message_id": "m1"}, tools={},
        session=cast(Any, object()), user_id="u", fusion_live_enabled=True,
    )

    source_urls = [e["data"]["source"]["url"] for e in _source_events(emitted)]
    assert source_urls.count(dup_url) == 1, "duplicate source URL must be emitted once"
    assert set(source_urls) == {dup_url, "https://docs.openwebui.com/"}

    sources_payload = next(
        (p["sources"] for p in persisted if isinstance(p.get("sources"), list) and p["sources"]),
        None,
    )
    assert sources_payload is not None, "fusion sources must persist to the message 'sources' field"
    persisted_urls = {s["source"]["url"] for s in sources_payload}
    assert persisted_urls == {dup_url, "https://docs.openwebui.com/"}


@pytest.mark.asyncio
async def test_fusion_good_path_emits_single_analysis_no_duplicate(monkeypatch, pipe_instance_async):
    _result, emitted = await _run(pipe_instance_async, monkeypatch, fusion_live_enabled=True)
    analysis = _fusion_events_of_type(emitted, "response.fusion_call.analysis.completed")
    assert len(analysis) == 1
    assert analysis[0]["analysis"]["consensus"] == ["agree"], "must be the real analysis, not synthetic"


def _fusion_done_events(sources):
    return [
        {"type": "response.created", "response": {"model": "anthropic/claude-opus"}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"type": "openrouter:fusion", "status": "in_progress"}},
        {"type": "response.fusion_call.in_progress", "output_index": 0},
        {"type": "response.fusion_call.completed", "output_index": 0},
        {"type": "response.output_item.done", "output_index": 0,
         "item": {"type": "openrouter:fusion", "status": "completed", "sources": sources}},
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "message"}},
        {"type": "response.output_text.done", "output_index": 1, "text": "Answer."},
        {"type": "response.completed",
         "response": {"output": [], "usage": {}, "model": "anthropic/claude-opus"}},
    ]


@pytest.mark.asyncio
async def test_fusion_sources_skips_malformed_entries(monkeypatch, pipe_instance_async):
    events = _fusion_done_events([
        "not-a-dict",
        {"title": "no url key"},
        {"url": ""},
        {"url": "   "},
        {"url": "https://ok.example/", "title": "OK"},
    ])
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _fake_stream(events))
    body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe_instance_async._streaming_handler._run_streaming_loop(
        body, pipe_instance_async.valves, emitter, metadata={}, tools={},
        session=cast(Any, object()), user_id="u", fusion_live_enabled=True,
    )
    source_events = _source_events(emitted)
    assert len(source_events) == 1
    assert source_events[0]["data"]["source"]["url"] == "https://ok.example/"


@pytest.mark.asyncio
async def test_fusion_source_document_falls_back_to_url_when_title_missing(monkeypatch, pipe_instance_async):
    events = _fusion_done_events([
        {"url": "https://x.example/", "title": ""},
        {"url": "https://y.example/"},
    ])
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _fake_stream(events))
    body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe_instance_async._streaming_handler._run_streaming_loop(
        body, pipe_instance_async.valves, emitter, metadata={}, tools={},
        session=cast(Any, object()), user_id="u", fusion_live_enabled=True,
    )
    docs_by_url = {
        e["data"]["source"]["url"]: e["data"]["document"] for e in _source_events(emitted)
    }
    assert docs_by_url["https://x.example/"] == ["https://x.example/"]
    assert docs_by_url["https://y.example/"] == ["https://y.example/"]


@pytest.mark.asyncio
async def test_fusion_sources_emitted_when_analysis_present(monkeypatch, pipe_instance_async):
    events = [
        {"type": "response.created", "response": {"model": "anthropic/claude-opus"}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"type": "openrouter:fusion", "status": "in_progress"}},
        {"type": "response.fusion_call.in_progress", "output_index": 0},
        {"type": "response.fusion_call.analysis.in_progress", "output_index": 0,
         "judge_model": "anthropic/claude"},
        {"type": "response.fusion_call.analysis.completed", "output_index": 0,
         "analysis": {"consensus": ["c"], "contradictions": [], "partial_coverage": [],
                      "unique_insights": [], "blind_spots": []}},
        {"type": "response.fusion_call.completed", "output_index": 0},
        {"type": "response.output_item.done", "output_index": 0,
         "item": {"type": "openrouter:fusion", "status": "completed",
                  "sources": [{"url": "https://ok.example/", "title": "OK"}]}},
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "message"}},
        {"type": "response.output_text.done", "output_index": 1, "text": "Answer."},
        {"type": "response.completed",
         "response": {"output": [], "usage": {}, "model": "anthropic/claude-opus"}},
    ]
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _fake_stream(events))
    body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe_instance_async._streaming_handler._run_streaming_loop(
        body, pipe_instance_async.valves, emitter, metadata={}, tools={},
        session=cast(Any, object()), user_id="u", fusion_live_enabled=True,
    )
    analysis = _fusion_events_of_type(emitted, "response.fusion_call.analysis.completed")
    assert len(analysis) == 1
    assert analysis[0]["analysis"]["consensus"] == ["c"], "real analysis, not synthetic"
    source_urls = {e["data"]["source"]["url"] for e in _source_events(emitted)}
    assert source_urls == {"https://ok.example/"}


class TestEventSourceSeam:

    @staticmethod
    def _forbidden_stream():
        async def fake_stream(self, session, request_body, **_kwargs):
            raise AssertionError("upstream send must not be called with event_source")
            yield
        return fake_stream

    async def _run_with_source(self, pipe, monkeypatch, events, **kwargs):
        monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", self._forbidden_stream())
        monkeypatch.setattr(Pipe, "send_openrouter_nonstreaming_request_as_events",
                            self._forbidden_stream(), raising=False)

        async def source():
            for event in events:
                yield event

        body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)
        emitted: list[dict] = []

        async def emitter(event):
            emitted.append(event)

        result = await pipe._streaming_handler._run_streaming_loop(
            body, pipe.valves, emitter, metadata={}, tools={},
            session=cast(Any, object()), user_id="u",
            event_source=source(), **kwargs,
        )
        return result, emitted

    @pytest.mark.asyncio
    async def test_event_source_replaces_upstream_send(self, monkeypatch, pipe_instance_async):
        result, emitted = await self._run_with_source(
            pipe_instance_async, monkeypatch, FUSION_EVENTS, fusion_live_enabled=True,
        )
        assert "The answer is 42." in result
        assert [e for e in emitted if e.get("type") == "fusion:event"]

    @pytest.mark.asyncio
    async def test_event_source_completion_usage_flows(self, monkeypatch, pipe_instance_async):
        _result, emitted = await self._run_with_source(
            pipe_instance_async, monkeypatch, FUSION_EVENTS, fusion_live_enabled=False,
        )
        completions = [e for e in emitted if e.get("type") == "chat:completion"
                       and isinstance(e.get("data"), dict) and e["data"].get("usage")]
        assert completions

    @pytest.mark.asyncio
    async def test_event_source_without_fusion_events_plain_stream(self, monkeypatch, pipe_instance_async):
        plain = [
            {"type": "response.created", "response": {"model": "openai/gpt-5"}},
            {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "hello"},
            {"type": "response.output_text.done", "output_index": 0, "text": "hello"},
            {"type": "response.completed", "response": {"output": [], "usage": {}, "model": "openai/gpt-5"}},
        ]
        result, _emitted = await self._run_with_source(
            pipe_instance_async, monkeypatch, plain, fusion_live_enabled=False,
        )
        assert result == "hello"


class TestOutcomeSink:

    async def _run_sink(self, pipe, monkeypatch, events, *, explode=False):
        async def source():
            for event in events:
                yield event
            if explode:
                raise RuntimeError("mid-stream death")

        body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)
        sink: dict[str, Any] = {}

        async def emitter(event):
            pass

        result = await pipe._streaming_handler._run_streaming_loop(
            body, pipe.valves, emitter, metadata={}, tools={},
            session=cast(Any, object()), user_id="u", fusion_live_enabled=False,
            event_source=source(), outcome_sink=sink,
        )
        return result, sink

    @pytest.mark.asyncio
    async def test_clean_run_reports_no_error(self, monkeypatch, pipe_instance_async):
        result, sink = await self._run_sink(pipe_instance_async, monkeypatch, FUSION_EVENTS)
        assert sink["error_occurred"] is False
        assert sink["was_cancelled"] is False
        assert "The answer is 42." in result

    @pytest.mark.asyncio
    async def test_mid_stream_exception_reported(self, monkeypatch, pipe_instance_async):
        partial = FUSION_EVENTS[:3]
        result, sink = await self._run_sink(pipe_instance_async, monkeypatch, partial, explode=True)
        assert sink["error_occurred"] is True
        assert sink["was_cancelled"] is False
        assert "mid-stream death" in (sink.get("reason") or "")


class TestInnerBreakerSuppression:

    async def _capture_send_kwargs(self, pipe, monkeypatch, metadata):
        captured: dict[str, Any] = {}

        async def fake_stream(self, session, request_body, **kwargs):
            captured.update(kwargs)
            yield {"type": "response.completed",
                   "response": {"output": [], "usage": {}, "model": "openai/gpt-5"}}

        monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", fake_stream)
        body = ResponsesBody(model="openai/gpt-5", input=[], stream=True)
        await pipe._streaming_handler._run_streaming_loop(
            body, pipe.valves, None, metadata=metadata, tools={},
            session=cast(Any, object()), user_id="u",
        )
        return captured

    @pytest.mark.asyncio
    async def test_normal_request_keeps_breaker_key(self, monkeypatch, pipe_instance_async):
        captured = await self._capture_send_kwargs(pipe_instance_async, monkeypatch, {})
        assert captured.get("breaker_key") == "u"

    @pytest.mark.asyncio
    async def test_inner_marker_suppresses_breaker_key(self, monkeypatch, pipe_instance_async):
        from open_webui_openrouter_pipe.core.config import _PIPE_METADATA_KEY

        metadata = {_PIPE_METADATA_KEY: {"fusion_inner": True}}
        captured = await self._capture_send_kwargs(pipe_instance_async, monkeypatch, metadata)
        assert captured.get("breaker_key") is None


class TestInnerReasoningForward:

    REASONING_STREAM = [
        {"type": "response.created", "response": {"model": "openai/gpt-5"}},
        {"type": "response.reasoning_text.delta", "delta": "thinking hard"},
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}},
        {"type": "response.output_text.done", "output_index": 0, "text": "answer"},
        {"type": "response.completed", "response": {"output": [], "usage": {}, "model": "openai/gpt-5"}},
    ]

    async def _run_marked(self, pipe, monkeypatch, *, marked):
        from open_webui_openrouter_pipe.core.config import _PIPE_METADATA_KEY

        monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _fake_stream(self.REASONING_STREAM))
        metadata = {_PIPE_METADATA_KEY: {"fusion_inner": True}} if marked else {}
        emitted: list[dict] = []

        async def emitter(event):
            emitted.append(event)

        body = ResponsesBody(model="openai/gpt-5", input=[], stream=True)
        await pipe._streaming_handler._run_streaming_loop(
            body, pipe.valves, emitter, metadata=metadata, tools={},
            session=cast(Any, object()), user_id="u",
        )
        return [e for e in emitted if e.get("type") == "fusion_inner:reasoning.delta"]

    @pytest.mark.asyncio
    async def test_inner_call_forwards_reasoning_deltas(self, monkeypatch, pipe_instance_async):
        fwd = await self._run_marked(pipe_instance_async, monkeypatch, marked=True)
        assert fwd
        assert fwd[0]["data"]["delta"] == "thinking hard"

    @pytest.mark.asyncio
    async def test_normal_call_never_forwards(self, monkeypatch, pipe_instance_async):
        fwd = await self._run_marked(pipe_instance_async, monkeypatch, marked=False)
        assert fwd == []


class TestInternalBackendTripwire:

    @pytest.mark.asyncio
    async def test_event_source_run_never_trips(self, monkeypatch, pipe_instance_async, caplog):
        import logging

        async def source():
            for event in FUSION_EVENTS:
                yield event

        body = ResponsesBody(model="openrouter/fusion", input=[], stream=True,
                             plugins=[{"id": "fusion"}])
        with caplog.at_level(logging.WARNING):
            await pipe_instance_async._streaming_handler._run_streaming_loop(
                body, pipe_instance_async.valves, None, metadata={}, tools={},
                session=cast(Any, object()), user_id="u", fusion_live_enabled=True,
                event_source=source(),
            )
        assert not _tripwire_records(caplog)


STAGE_FEEDBACK_EVENTS = [
    {"type": "response.created", "response": {"model": "openrouter/fusion"}},
    {"type": "response.output_item.added", "output_index": 0,
     "item": {"type": "openrouter:fusion", "status": "in_progress"}},
    {"type": "response.fusion_call.in_progress", "output_index": 0},
    {"type": "response.fusion_call.panel.added", "output_index": 0, "model": "m/a"},
    {"type": "response.fusion_call.panel.completed", "output_index": 0, "model": "m/a", "content": "A"},
    {"type": "response.fusion_call.analysis.in_progress", "output_index": 0, "judge_model": "j/x"},
    {"type": "response.fusion_call.analysis.reasoning.delta", "output_index": 0,
     "model": "j/x", "delta": "w" * 600},
    {"type": "response.fusion_call.analysis.completed", "output_index": 0,
     "analysis": {"consensus": [], "contradictions": [], "partial_coverage": [],
                  "unique_insights": [], "blind_spots": []}},
    {"type": "response.fusion_call.completed", "output_index": 0},
    {"type": "response.output_item.done", "output_index": 0,
     "item": {"type": "openrouter:fusion", "status": "completed"}},
    {"type": "response.fusion_call.synthesis.in_progress", "output_index": 0, "model": "j/x"},
    {"type": "response.fusion_call.synthesis.reasoning.delta", "output_index": 0,
     "model": "j/x", "delta": "s" * 600},
    {"type": "response.output_item.added", "output_index": 1, "item": {"type": "message"}},
    {"type": "response.output_text.delta", "output_index": 1, "delta": "final"},
    {"type": "response.output_text.done", "output_index": 1, "text": "final"},
    {"type": "response.completed",
     "response": {"output": [], "usage": {"cost": 0.1}, "model": "openrouter/fusion"}},
]


class TestStageFeedbackStreaming:
    @staticmethod
    def _fusion_events(emitted):
        return [e["data"]["event"] for e in emitted if e.get("type") == "fusion:event"]

    async def _run_stage(self, pipe, monkeypatch):
        async def source():
            for ev in STAGE_FEEDBACK_EVENTS:
                yield ev

        body = ResponsesBody(model="openrouter/fusion", input=[], stream=True,
                             plugins=[{"id": "fusion"}])
        emitted: list[dict] = []

        async def emitter(event):
            emitted.append(event)

        result = await pipe._streaming_handler._run_streaming_loop(
            body, pipe.valves, emitter, metadata={}, tools={},
            session=cast(Any, object()), user_id="u", fusion_live_enabled=True,
            event_source=source(),
        )
        return result, emitted

    @pytest.mark.asyncio
    async def test_stage_reasoning_live_emitted_and_not_baked(self, monkeypatch, pipe_instance_async):
        _result, emitted = await self._run_stage(pipe_instance_async, monkeypatch)
        fev = self._fusion_events(emitted)
        jr = [e for e in fev if e.get("type") == "response.fusion_call.analysis.reasoning.delta"]
        sr = [e for e in fev if e.get("type") == "response.fusion_call.synthesis.reasoning.delta"]
        assert jr and "w" in jr[0]["delta"]
        assert sr and "s" in sr[0]["delta"]
        embeds = [e for e in emitted if e.get("type") == "embeds"]
        assert embeds
        baked = embeds[-1]["data"]["embeds"][0]
        assert "reasoning.delta" not in baked or baked.count("analysis.reasoning.delta") <= 1

    @pytest.mark.asyncio
    async def test_analysis_completed_carries_judge_reasoning(self, monkeypatch, pipe_instance_async):
        _result, emitted = await self._run_stage(pipe_instance_async, monkeypatch)
        fev = self._fusion_events(emitted)
        done = [e for e in fev if e.get("type") == "response.fusion_call.analysis.completed"]
        assert done and done[0].get("reasoning", "").startswith("w")

    @pytest.mark.asyncio
    async def test_synthesis_in_progress_milestone_emitted(self, monkeypatch, pipe_instance_async):
        _result, emitted = await self._run_stage(pipe_instance_async, monkeypatch)
        fev = self._fusion_events(emitted)
        assert any(e.get("type") == "response.fusion_call.synthesis.in_progress" for e in fev)

    @pytest.mark.asyncio
    async def test_final_answer_carries_synthesis_reasoning(self, monkeypatch, pipe_instance_async):
        _result, emitted = await self._run_stage(pipe_instance_async, monkeypatch)
        fev = self._fusion_events(emitted)
        done = [e for e in fev if e.get("type") == "response.output_text.done"]
        assert done and done[-1].get("reasoning", "").startswith("s")
