"""Phase C tests: the openrouter/fusion live-UI streaming wiring."""

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


async def _run(pipe, monkeypatch, *, fusion_live_enabled):
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _fake_stream(FUSION_EVENTS))
    body = ResponsesBody(model="openrouter/fusion", input=[], stream=True)
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    result = await pipe._streaming_handler._run_streaming_loop(
        body, pipe.valves, emitter, metadata={}, tools={},
        session=cast(Any, object()), user_id="u", fusion_live_enabled=fusion_live_enabled,
    )
    return result, emitted


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
    """Judge skips analysis and no fusion output_item.done arrives: the terminal
    backstop must still heal the live spinner by emitting a synthetic
    analysis.completed over the socket (not only into the persisted snapshot)."""
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
    """A fusion output_item.done with missing analysis + item.sources must both
    heal the live spinner (synthetic analysis over socket) and surface the
    item-level sources as OWUI 'source' citation events."""
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
    """Duplicate source URLs are emitted once, and surfaced citations are
    persisted to the message 'sources' field so they survive a page refresh."""
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
    """When the judge DOES emit analysis.completed, the fix must be a no-op:
    exactly one (real) analysis.completed over the socket, no synthetic duplicate."""
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
    """Non-dict, missing-url, and blank-url source entries are skipped without error."""
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
    """A source with empty/absent title uses the URL as the document body (never the
    generic 'Citation' placeholder) — pins the intentional title-or-url fallback."""
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
    """Sources surface on the good path too: the real analysis is emitted once and
    no synthetic duplicate is added when analysis.completed was provided."""
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
