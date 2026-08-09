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
    "function panelReasoningDelta", "function panelAnswerDelta",
    "'response.fusion_call.panel.delta'", "'response.fusion_call.panel.reasoning.delta'",
    'class="ticker"', "tick-words", "think-lbl", "think-toggle", "thinking-open",
    "_lastH",
    'id="synthArea"', "think-view", "tv-scroll", "tv-expand", "think-block",
    "function synthesisStarting", "function stageReasoningDelta", "function buildThinkBlock",
    "'response.fusion_call.synthesis.in_progress'",
    "'response.fusion_call.analysis.reasoning.delta'",
    "'response.fusion_call.synthesis.reasoning.delta'",
    "_sawSynthesis", "panel + judge + synthesis", "Fusion made <b>", "Fusion ran <b>",
    'think-lbl">Thinking</span><span class="show-lbl">Show',
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




def test_record_panel_deltas_not_appended_and_no_milestone():
    state = FusionDeliberationState()
    assert state.record({"type": "response.fusion_call.panel.delta",
                         "model": "a/b", "delta": "chunk"}) is None
    assert state.record({"type": "response.fusion_call.panel.reasoning.delta",
                         "model": "a/b", "delta": "think"}) is None
    assert state.events == []


def test_record_accumulates_reasoning_per_model():
    state = FusionDeliberationState()
    state.record({"type": "response.fusion_call.panel.reasoning.delta", "model": "a/b", "delta": "one "})
    state.record({"type": "response.fusion_call.panel.reasoning.delta", "model": "a/b", "delta": "two"})
    state.record({"type": "response.fusion_call.panel.reasoning.delta", "model": "c/d", "delta": "other"})
    assert state.panel_reasoning_buf == {"a/b": "one two", "c/d": "other"}


def test_record_reasoning_buffer_capped():
    state = FusionDeliberationState()
    state.record({"type": "response.fusion_call.panel.reasoning.delta",
                  "model": "a/b", "delta": "x" * 600_000})
    state.record({"type": "response.fusion_call.panel.reasoning.delta",
                  "model": "a/b", "delta": "more"})
    assert len(state.panel_reasoning_buf["a/b"]) <= 512 * 1024


def test_augment_panel_completed_attaches_reasoning_without_mutating_original():
    state = FusionDeliberationState()
    state.record({"type": "response.fusion_call.panel.reasoning.delta", "model": "a/b", "delta": "thought"})
    original = {"type": "response.fusion_call.panel.completed",
                "output_index": 1, "model": "a/b", "content": "answer"}
    augmented = state.augment_panel_completed(original)
    assert augmented is not original
    assert augmented["reasoning"] == "thought"
    assert augmented["content"] == "answer"
    assert "reasoning" not in original


def test_augment_panel_completed_passthrough_without_reasoning():
    state = FusionDeliberationState()
    original = {"type": "response.fusion_call.panel.completed", "model": "a/b", "content": "answer"}
    assert state.augment_panel_completed(original) is original


def test_augmented_completed_round_trips_reasoning_into_snapshot():
    state = FusionDeliberationState()
    state.record({"type": "response.output_item.added", "output_index": 1,
                  "item": {"type": "openrouter:fusion"}})
    state.record({"type": "response.fusion_call.panel.reasoning.delta", "model": "a/b", "delta": "thought"})
    augmented = state.augment_panel_completed(
        {"type": "response.fusion_call.panel.completed",
         "output_index": 1, "model": "a/b", "content": "ans"}
    )
    state.record(augmented)
    events = _events_literal(build_fusion_embed_html(state))
    completed = [e for e in events if e.get("type") == "response.fusion_call.panel.completed"]
    assert completed and completed[0]["reasoning"] == "thought"
    assert not [e for e in events if str(e.get("type", "")).endswith(".delta")
                and "panel" in str(e.get("type", ""))]




def _delta_ev(kind, model, delta, **extra):
    ev = {"type": f"response.fusion_call.panel{kind}", "output_index": 1,
          "item_id": "st_1", "model": model, "delta": delta}
    ev.update(extra)
    return ev


def test_batcher_flushes_on_size():
    from open_webui_openrouter_pipe.streaming.fusion_embed import FusionDeltaBatcher
    b = FusionDeltaBatcher(max_chars=10, max_age=999.0)
    assert b.add(_delta_ev(".delta", "a/b", "12345"), now=0.0) is None
    out = b.add(_delta_ev(".delta", "a/b", "67890X"), now=0.0)
    assert out is not None
    assert out["delta"] == "1234567890X"
    assert out["type"] == "response.fusion_call.panel.delta"
    assert out["model"] == "a/b"
    assert out["output_index"] == 1


def test_batcher_flushes_on_age():
    from open_webui_openrouter_pipe.streaming.fusion_embed import FusionDeltaBatcher
    b = FusionDeltaBatcher(max_chars=10_000, max_age=0.2)
    assert b.add(_delta_ev(".delta", "a/b", "a"), now=0.0) is None
    out = b.add(_delta_ev(".delta", "a/b", "b"), now=0.5)
    assert out is not None
    assert out["delta"] == "ab"


def test_batcher_keys_by_model_and_kind():
    from open_webui_openrouter_pipe.streaming.fusion_embed import FusionDeltaBatcher
    b = FusionDeltaBatcher(max_chars=10_000, max_age=999.0)
    b.add(_delta_ev(".delta", "a/b", "answer-a"), now=0.0)
    b.add(_delta_ev(".reasoning.delta", "a/b", "think-a"), now=0.0)
    b.add(_delta_ev(".delta", "c/d", "answer-c"), now=0.0)
    flushed = b.flush_all()
    deltas = sorted((e["type"], e["model"], e["delta"]) for e in flushed)
    assert deltas == [
        ("response.fusion_call.panel.delta", "a/b", "answer-a"),
        ("response.fusion_call.panel.delta", "c/d", "answer-c"),
        ("response.fusion_call.panel.reasoning.delta", "a/b", "think-a"),
    ]
    assert b.flush_all() == []


def test_batcher_discard_model_drops_both_kinds():
    from open_webui_openrouter_pipe.streaming.fusion_embed import FusionDeltaBatcher
    b = FusionDeltaBatcher(max_chars=10_000, max_age=999.0)
    b.add(_delta_ev(".delta", "a/b", "answer"), now=0.0)
    b.add(_delta_ev(".reasoning.delta", "a/b", "think"), now=0.0)
    b.add(_delta_ev(".delta", "c/d", "keep"), now=0.0)
    b.discard_model("a/b")
    flushed = b.flush_all()
    assert [(e["model"], e["delta"]) for e in flushed] == [("c/d", "keep")]


class TestCopyAllModelCount:
    def test_copy_all_usage_line_is_backend_conditional(self):
        from open_webui_openrouter_pipe.streaming.fusion_embed import build_fusion_embed_html, FusionDeliberationState

        state = FusionDeliberationState()
        html = build_fusion_embed_html(state)
        assert "_sawSynthesis ? (panelCount+2)+' model calls' : (panelCount+1)+' models'" in html


class TestStageReasoningState:
    def _state(self):
        return FusionDeliberationState()

    def test_stage_reasoning_accumulates_not_baked(self):
        s = self._state()
        for i in range(3):
            ms = s.record({"type": "response.fusion_call.analysis.reasoning.delta",
                           "output_index": 0, "model": "j/x", "delta": f"j{i} "})
            assert ms is None
        ms = s.record({"type": "response.fusion_call.synthesis.reasoning.delta",
                       "output_index": 0, "model": "j/x", "delta": "s0 "})
        assert ms is None
        assert s.judge_reasoning_buf == "j0 j1 j2 "
        assert s.synthesis_reasoning_buf == "s0 "
        assert s.events == []

    def test_stage_buffers_do_not_mix_with_panel_buffer(self):
        s = self._state()
        s.record({"type": "response.fusion_call.panel.reasoning.delta",
                  "output_index": 0, "model": "j/x", "delta": "panel-think"})
        s.record({"type": "response.fusion_call.analysis.reasoning.delta",
                  "output_index": 0, "model": "j/x", "delta": "judge-think"})
        assert s.panel_reasoning_buf["j/x"] == "panel-think"
        assert s.judge_reasoning_buf == "judge-think"

    def test_synthesis_in_progress_baked_with_milestone(self):
        s = self._state()
        ev = {"type": "response.fusion_call.synthesis.in_progress",
              "output_index": 0, "model": "j/x"}
        assert s.record(ev) == "synthesis_start"
        assert ev in s.events

    def test_augment_analysis_completed(self):
        s = self._state()
        s.record({"type": "response.fusion_call.analysis.reasoning.delta",
                  "output_index": 0, "model": "j/x", "delta": "why"})
        ev = {"type": "response.fusion_call.analysis.completed", "output_index": 0,
              "analysis": {"consensus": [], "contradictions": [], "partial_coverage": [],
                           "unique_insights": [], "blind_spots": []}}
        out = s.augment_analysis_completed(ev)
        assert out["reasoning"] == "why"
        assert "reasoning" not in ev

    def test_augment_final_answer(self):
        s = self._state()
        s.record({"type": "response.fusion_call.synthesis.reasoning.delta",
                  "output_index": 0, "model": "j/x", "delta": "how"})
        ev = {"type": "response.output_text.done", "output_index": 1, "text": "answer"}
        out = s.augment_final_answer(ev)
        assert out["reasoning"] == "how"

    def test_augment_noop_when_empty(self):
        s = self._state()
        ev = {"type": "response.fusion_call.analysis.completed", "output_index": 0}
        assert s.augment_analysis_completed(ev) is ev
        ev2 = {"type": "response.output_text.done", "output_index": 1, "text": "a"}
        assert s.augment_final_answer(ev2) is ev2


def test_template_footer_has_both_backend_variants():
    html = build_fusion_embed_html(FusionDeliberationState())
    assert "panel + judge + synthesis" in html
    assert "Fusion made <b>" in html
    assert "Fusion ran <b>" in html
    assert "'panel + judge')" in html


def test_panel_think_row_carries_show_hide_control():
    html = build_fusion_embed_html(FusionDeliberationState())
    assert 'think-lbl">Thinking</span><span class="show-lbl">Show' in html


def test_synthesis_lifecycle_bakes_milestone_and_augments_reasoning():
    s = FusionDeliberationState()
    s.record({"type": "response.output_item.added", "output_index": 1,
              "item": {"type": "openrouter:fusion"}})
    s.record({"type": "response.fusion_call.panel.added", "output_index": 1, "model": "a/b"})
    s.record({"type": "response.fusion_call.panel.completed", "output_index": 1,
              "model": "a/b", "content": "ans"})
    s.record({"type": "response.fusion_call.analysis.in_progress", "output_index": 1,
              "judge_model": "j/x"})
    s.record({"type": "response.fusion_call.analysis.reasoning.delta",
              "output_index": 0, "model": "j/x", "delta": "judge-think"})
    analysis_completed = s.augment_analysis_completed({
        "type": "response.fusion_call.analysis.completed", "output_index": 1,
        "analysis": {"consensus": [], "contradictions": [], "partial_coverage": [],
                     "unique_insights": [], "blind_spots": []}})
    s.record(analysis_completed)
    assert s.record({"type": "response.fusion_call.synthesis.in_progress",
                     "output_index": 0, "model": "s/y"}) == "synthesis_start"
    s.record({"type": "response.fusion_call.synthesis.reasoning.delta",
              "output_index": 0, "model": "s/y", "delta": "synth-think"})
    s.record({"type": "response.output_item.added", "output_index": 2,
              "item": {"type": "message"}})
    s.record({"type": "response.output_text.delta", "output_index": 2, "delta": "Hello"})
    final_done = s.augment_final_answer({"type": "response.output_text.done",
                                         "output_index": 2, "text": "Hello world"})
    s.record(final_done)
    s.record({"type": "response.completed", "response": {"usage": {"cost": 0.1}}})

    baked = _events_literal(build_fusion_embed_html(s))
    types = [e["type"] for e in baked]
    assert "response.fusion_call.synthesis.in_progress" in types
    assert "response.fusion_call.analysis.reasoning.delta" not in types
    assert "response.fusion_call.synthesis.reasoning.delta" not in types
    ac = next(e for e in baked if e["type"] == "response.fusion_call.analysis.completed")
    assert ac["reasoning"] == "judge-think"
    fd = next(e for e in baked if e["type"] == "response.output_text.done"
              and e.get("output_index") == 2)
    assert fd["reasoning"] == "synth-think"
    assert s.judge_reasoning_buf == "judge-think"
    assert s.synthesis_reasoning_buf == "synth-think"


# ── Socket lifetime: a finished deliberation must not hold a connection ──


def _js_fn_body(js: str, name: str) -> str:
    """The body of a JS function declaration, by brace matching.

    An identifier appearing says nothing about what it does: a ``freshToken`` that
    memoises its first read satisfies every name-based check while replaying the
    render-time token on every reconnect forever.
    """
    start = js.index("function " + name)
    open_brace = js.index("{", start)
    depth = 0
    for i in range(open_brace, len(js)):
        if js[i] == "{":
            depth += 1
        elif js[i] == "}":
            depth -= 1
            if depth == 0:
                return js[open_brace + 1:i]
    raise AssertionError(f"unbalanced braces in {name}")


def _socket_block(html: str) -> str:
    """The bootstrap's socket IIFE, up to the point it actually dials out."""
    return html[html.index("fusionLiveSocket"):html.index("io(origin")]


def test_finished_deliberation_does_not_open_a_socket():
    """A replayed answer can never receive a fusion event, so it must not connect.

    Without the gate, opening a chat with N finished fusion answers opens N websockets
    and registers N OWUI sessions, each also joining ``user:{id}`` and so multiplying
    fan-out of every streaming event for that user.
    """
    import re

    html = build_fusion_embed_html(FusionDeliberationState(), final=True)
    assert "var _FUSION_FINAL = true;" in html
    assert re.search(r"if\s*\(\s*_FUSION_FINAL\s*\)\s*\{?\s*return", _socket_block(html))


def test_live_deliberation_still_opens_a_socket():
    """The mirror of the above: the gate must not disarm the live path."""
    html = build_fusion_embed_html(FusionDeliberationState(), final=False)
    assert "var _FUSION_FINAL = false;" in html
    assert "io(origin" in html


def test_live_embed_releases_its_socket_after_the_answer_completes():
    """A finished run must not keep receiving every later event for the user.

    The socket sits in ``user:{id}``, and Open WebUI fans every event to that room, so a
    still-armed iframe from a finished deliberation gets a full copy of every subsequent
    generation's stream and throws it away client-side.

    The terminal flag must be STICKY rather than a branch: the analysis synthesised after
    the stream loop arrives as a fusion event *after* ``response.output_text.done``, so a
    plain "re-arm unless this is the terminal event" would restart the long backstop.
    """
    import re

    html = build_fusion_embed_html(FusionDeliberationState(), final=False)
    assert "function armIdle()" in html
    assert 'd.data.event.type === "response.output_text.done"' in html
    # Defined is not called: deleting both call sites leaves the socket never released.
    connect_body = html[html.index('sock.on("connect", function(){'):html.index('sock.on("disconnect"')]
    assert "armIdle();" in connect_body, "the timer is never armed on connect"
    events_body = html[html.index('sock.on("events"'):]
    assert "armIdle();" in events_body, "the timer is never re-armed by activity"
    # Sticky: a latch, never a branch. An else-arm resetting it to false would let the
    # post-stream analysis event restart the long backstop.
    assert re.findall(r"finished\s*=\s*(\w+)", html) == ["false", "true"]
    # The latch must sit ON the terminal branch, and every other event must still re-arm.
    latch = re.search(r'if \(d\.data\.event\.type === "response\.output_text\.done"\) \{([^}]*)\}', events_body)
    assert latch is not None and "finished = true" in latch.group(1), \
        "the latch is not on the terminal-event branch"
    after_latch = events_body[events_body.index(latch.group(0)) + len(latch.group(0)):]
    assert "armIdle();" in after_latch, "non-terminal events no longer re-arm the idle timer"

    # Both bounds are parsed, never substring-matched: "600000" is a substring of
    # "6000000", so a presence check cannot tell ten minutes from a hundred.
    # The effect must be bound to the timer, not merely present somewhere in the embed:
    # a callback that touches the socket without disconnecting never releases it.
    m = re.search(
        r"idleTimer = setTimeout\(function\s*\(\)\s*\{(.*?)\}\s*,\s*finished \? (\d+) : (\d+)\)",
        html, re.S)
    assert m is not None, "idle release is not a sticky two-bound timer"
    assert "sock.disconnect()" in m.group(1), "the idle timer fires but never releases the socket"
    after_answer, backstop = int(m.group(2)), int(m.group(3))
    assert after_answer <= 30_000, "a finished run holds its socket too long"
    assert backstop <= 600_000, "the quiet-run backstop is longer than agreed"
    assert after_answer < backstop

    disconnect = html[html.index('sock.on("disconnect"'):][:250]
    assert "clearTimeout(idleTimer)" in disconnect


def test_fusion_socket_does_not_heartbeat():
    """Fusion events route by room membership, never through OWUI's SESSION_POOL.

    OWUI's heartbeat handler writes SESSION_POOL and issues an unthrottled
    Users.update_last_active_by_id UPDATE per beat. Nothing on the fusion path reads
    either, so a beat here is a database write per embed per interval buying nothing.
    """
    html = build_fusion_embed_html(FusionDeliberationState(), final=False)
    assert "heartbeat" not in html.replace('emitReserved("heartbeat")', "")
    assert "hbTimer" not in html


def test_fusion_credentials_are_read_at_handshake():
    import re

    html = build_fusion_embed_html(FusionDeliberationState(), final=False)
    assert "auth: function(cb)" in html
    assert "auth: { token: token }" not in html

    # The helper must read storage on every call and hold nothing between them.
    body = _js_fn_body(html, "freshToken")
    assert 'localStorage.getItem("token")' in body, "freshToken does not read storage"
    assert re.search(r"[^=!<>]=(?!=)", body) is None, \
        "freshToken keeps state across calls, so a rotated token never reaches the server"

    # And the fresh read must win the fallback, not lose to the captured value.
    handshake = re.search(r"auth:\s*function\(cb\)\{\s*cb\(\{\s*token:\s*([^}]+?)\s*\}\)", html)
    assert handshake is not None, "handshake auth is not a function form"
    assert handshake.group(1).strip().startswith("freshToken()")
    join = re.search(r'user-join",\s*\{\s*auth:\s*\{\s*token:\s*([^}]+?)\s*\}', html)
    assert join is not None, "user-join payload not found"
    assert join.group(1).strip().startswith("freshToken()"), "the captured token wins the fallback"
