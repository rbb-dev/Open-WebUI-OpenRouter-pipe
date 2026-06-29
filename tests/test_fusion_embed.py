"""Tests for the Fusion live-UI embed module."""

import json

from open_webui_openrouter_pipe.streaming.fusion_embed import (
    FusionDeliberationState,
    build_fusion_embed_html,
)


def _events_literal(html: str):
    marker = "var EVENTS = "
    idx = html.index(marker) + len(marker)
    decoded, _ = json.JSONDecoder().raw_decode(html, idx)
    return decoded


def _build(events):
    state = FusionDeliberationState()
    for event in events:
        state.record(event)
    return state, build_fusion_embed_html(state)


_OMIT = [
    'id="demoBg"', 'class="demo-bar"', 'id="panelSel"', 'id="replayBtn"',
    'id="themeBtn"', "function runFeed", "var DATA = JSON.parse", 'id="fusion-feed"',
    "addEventListener('message'",
    'id="wideBtn"', "function applyWide", "max-width:72ch", "document.documentElement.scrollHeight",
]

_KEEP = [
    "window.FusionUI", "function renderAnalysis", "function syncTheme",
    "function reportHeight", 'id="copyAllBtn"',
    "function writeClipboard", "function analysisToHtml", "function buildAll",
    'id="sec-final"', "function startFinal", "function appendFinal", "function endFinal",
    "fusionLiveSocket", "user-join",
    "document.body.scrollHeight", "requestAnimationFrame", "cancelAnimationFrame",
]


def test_template_omits_demo_surface():
    html = build_fusion_embed_html(FusionDeliberationState())
    for token in _OMIT:
        assert token not in html, f"production template must not contain {token!r}"


def test_template_keeps_render_engine():
    html = build_fusion_embed_html(FusionDeliberationState())
    for token in _KEEP:
        assert token in html, f"production template must retain {token!r}"


def test_esc_and_safehref_are_hardened():
    html = build_fusion_embed_html(FusionDeliberationState())
    assert "&quot;" in html and "&#39;" in html
    assert "[\"'<>" in html


def test_empty_state_builds_waiting_ui():
    html = build_fusion_embed_html(FusionDeliberationState())
    assert "/*__FUSION_EVENTS_JSON__*/[]" not in html
    assert _events_literal(html) == []


def test_events_injected_and_round_trip():
    events = [
        {"type": "response.fusion_call.panel.added", "output_index": 1, "model": "a/b"},
        {"type": "response.fusion_call.panel.completed", "output_index": 1,
         "model": "a/b", "content": "# Heading\n\n- bullet"},
    ]
    _, html = _build(events)
    assert "/*__FUSION_EVENTS_JSON__*/[]" not in html
    assert _events_literal(html) == events


def test_script_breakout_is_neutralized():
    base = build_fusion_embed_html(FusionDeliberationState())
    nasty = "</script><script>alert(1)</script> <!-- ]]>    \U0001f600 \" '"
    events = [{"type": "response.fusion_call.panel.completed", "output_index": 1,
               "model": "x", "content": nasty}]
    _, html = _build(events)
    assert html.count("</script>") == base.count("</script>")
    assert html.count("<script") == base.count("<script")
    assert _events_literal(html)[0]["content"] == nasty


def test_record_milestones_match_verified_shapes():
    s = FusionDeliberationState()

    assert s.record({"type": "response.output_item.added", "output_index": 0,
                     "item": {"type": "message"}}) is None
    assert s.record({"type": "response.output_text.delta", "output_index": 0,
                     "delta": "I'll "}) is None
    assert s.record({"type": "response.output_text.done", "output_index": 0,
                     "text": "I'll research."}) is None

    assert s.record({"type": "response.output_item.added", "output_index": 1,
                     "item": {"type": "openrouter:fusion", "status": "in_progress"}}) == "fusion_open"
    assert s.fusion_index == 1

    assert s.record({"type": "response.fusion_call.in_progress", "output_index": 1}) == "roster"
    assert s.seen_roster is True

    assert s.record({"type": "response.fusion_call.panel.added", "output_index": 1,
                     "model": "g/f"}) == "panels"
    assert s.record({"type": "response.fusion_call.panel.completed", "output_index": 1,
                     "model": "g/f", "content": "# answer"}) == "panels"

    assert s.record({"type": "response.fusion_call.analysis.in_progress", "output_index": 1,
                     "judge_model": "anthropic/claude"}) == "analysis_start"
    assert s.seen_analysis_started is True
    assert s.record({"type": "response.fusion_call.analysis.completed", "output_index": 1,
                     "analysis": {"consensus": [], "contradictions": []}}) == "analysis_done"
    assert s.seen_analysis_done is True

    assert s.record({"type": "response.fusion_call.completed", "output_index": 1}) is None
    _before = len(s.events)
    assert s.record({"type": "response.output_item.done", "output_index": 1,
                     "item": {"type": "openrouter:fusion"}}) is None
    assert len(s.events) == _before

    assert s.record({"type": "response.output_item.added", "output_index": 2,
                     "item": {"type": "message"}}) is None
    assert s.record({"type": "response.output_text.delta", "output_index": 2,
                     "delta": "Here"}) is None
    assert s.record({"type": "response.output_text.done", "output_index": 2,
                     "text": "Here is the answer."}) == "answer"
    assert s.completed is False

    assert s.record({"type": "response.completed",
                     "response": {"usage": {"cost": 0.4}}}) == "completed"
    assert s.completed is True


def test_first_panel_added_is_roster_when_in_progress_skipped():
    s = FusionDeliberationState()
    s.record({"type": "response.output_item.added", "output_index": 1,
              "item": {"type": "openrouter:fusion"}})
    assert s.record({"type": "response.fusion_call.panel.added", "output_index": 1,
                     "model": "x"}) == "roster"
    assert s.record({"type": "response.fusion_call.panel.added", "output_index": 1,
                     "model": "y"}) == "panels"


def test_record_guards_malformed_events():
    s = FusionDeliberationState()
    assert s.record(None) is None  # pyright: ignore[reportArgumentType]
    assert s.record({}) is None
    assert s.record({"type": 123}) is None
    assert s.record({"type": "response.output_item.added", "item": "nope"}) is None
    assert len(s.events) == 1


def test_no_fusion_item_leaves_arming_signals_clear():
    s = FusionDeliberationState()
    for event in [
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}},
        {"type": "response.output_text.delta", "output_index": 0, "delta": "Answer"},
        {"type": "response.output_text.done", "output_index": 0, "text": "Answer."},
        {"type": "response.completed", "response": {}},
    ]:
        s.record(event)
    assert s.fusion_index is None
    assert s.seen_roster is False
    assert s.seen_analysis_done is False


def test_synthesize_missing_analysis_inserts_empty_completed_before_response_completed():
    s = FusionDeliberationState()
    for event in [
        {"type": "response.output_item.added", "output_index": 1,
         "item": {"type": "openrouter:fusion"}},
        {"type": "response.fusion_call.analysis.in_progress", "output_index": 1,
         "judge_model": "anthropic/claude"},
        {"type": "response.output_item.added", "output_index": 2, "item": {"type": "message"}},
        {"type": "response.output_text.done", "output_index": 2, "text": "The answer."},
        {"type": "response.completed", "response": {}},
    ]:
        s.record(event)
    assert s.seen_analysis_started is True
    assert s.seen_analysis_done is False

    synthetic = s.synthesize_missing_analysis()
    assert isinstance(synthetic, dict)
    assert synthetic["type"] == "response.fusion_call.analysis.completed"
    assert s.seen_analysis_done is True

    types = [e.get("type") for e in s.events]
    analysis_idx = types.index("response.fusion_call.analysis.completed")
    completed_idx = types.index("response.completed")
    assert analysis_idx < completed_idx
    synthetic = s.events[analysis_idx]
    assert synthetic["output_index"] == 1
    assert synthetic["analysis"] == {
        "consensus": [], "contradictions": [], "partial_coverage": [],
        "unique_insights": [], "blind_spots": [],
    }


def test_synthesize_missing_analysis_noop_when_analysis_already_done():
    s = FusionDeliberationState()
    for event in [
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "openrouter:fusion"}},
        {"type": "response.fusion_call.analysis.in_progress", "output_index": 1, "judge_model": "j"},
        {"type": "response.fusion_call.analysis.completed", "output_index": 1,
         "analysis": {"consensus": ["x"], "contradictions": []}},
        {"type": "response.completed", "response": {}},
    ]:
        s.record(event)
    before = len(s.events)
    assert s.synthesize_missing_analysis() is None
    assert len(s.events) == before


def test_synthesize_missing_analysis_noop_when_not_started():
    s = FusionDeliberationState()
    s.record({"type": "response.completed", "response": {}})
    before = len(s.events)
    assert s.synthesize_missing_analysis() is None
    assert len(s.events) == before


def test_empty_analysis_sections_use_reworded_neutral_strings():
    html = build_fusion_embed_html(FusionDeliberationState())
    for msg in [
        "No consensus was produced.",
        "No contradictions were produced.",
        "No partial coverage was produced.",
        "No unique insights were produced.",
        "No blind spots were produced.",
    ]:
        assert msg in html
    for stale in [
        "All three answers diverged",
        "answers diverged — no shared ground",
        "No direct conflicts between the answers",
        "Every point was covered by all models",
        "No model raised anything the others missed",
        "No shared blind spots flagged",
    ]:
        assert stale not in html
