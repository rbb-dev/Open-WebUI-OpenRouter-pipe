
import pytest

from open_webui_openrouter_pipe.core.fusion_defaults import (
    DEFAULT_FUSION_JUDGE_SYSTEM_PROMPT,
    DEFAULT_FUSION_MAX_TOOL_CALLS,
    DEFAULT_FUSION_PANEL_SYSTEM_PROMPT,
    DEFAULT_FUSION_PRESET,
    DEFAULT_FUSION_SYNTHESIS_SYSTEM_PROMPT,
    FUSION_PRESET_JUDGES,
    FUSION_PRESET_PANELS,
    FusionRunPlan,
    MAX_FUSION_MAX_TOOL_CALLS,
    MAX_FUSION_PANEL_MODELS,
    find_fusion_entry,
    resolve_fusion_run,
)


class TestPromptConstants:
    def test_panel_prompt_substantive(self):
        assert len(DEFAULT_FUSION_PANEL_SYSTEM_PROMPT) > 5000
        assert "independent" in DEFAULT_FUSION_PANEL_SYSTEM_PROMPT
        assert "Never mention that you are one of several models" in DEFAULT_FUSION_PANEL_SYSTEM_PROMPT

    def test_judge_prompt_substantive(self):
        assert len(DEFAULT_FUSION_JUDGE_SYSTEM_PROMPT) > 8000
        for key in ("consensus", "contradictions", "partial_coverage", "unique_insights", "blind_spots"):
            assert f'"{key}"' in DEFAULT_FUSION_JUDGE_SYSTEM_PROMPT
        assert '"evidence"' in DEFAULT_FUSION_JUDGE_SYSTEM_PROMPT

    def test_synthesis_prompt_substantive(self):
        assert len(DEFAULT_FUSION_SYNTHESIS_SYSTEM_PROMPT) > 5000
        assert "never blend" in DEFAULT_FUSION_SYNTHESIS_SYSTEM_PROMPT
        assert "NEVER mention or imply the drafts" in DEFAULT_FUSION_SYNTHESIS_SYSTEM_PROMPT

    def test_prompts_distinct(self):
        prompts = {
            DEFAULT_FUSION_PANEL_SYSTEM_PROMPT,
            DEFAULT_FUSION_JUDGE_SYSTEM_PROMPT,
            DEFAULT_FUSION_SYNTHESIS_SYSTEM_PROMPT,
        }
        assert len(prompts) == 3


class TestPresetConstants:
    def test_quality_panel(self):
        assert FUSION_PRESET_PANELS["general-high"] == (
            "~anthropic/claude-opus-latest",
            "~openai/gpt-latest",
            "~google/gemini-pro-latest",
        )

    def test_budget_and_fast_panels_identical(self):
        expected = (
            "~google/gemini-flash-latest",
            "deepseek/deepseek-v4-flash",
            "~moonshotai/kimi-latest",
        )
        assert FUSION_PRESET_PANELS["general-budget"] == expected
        assert FUSION_PRESET_PANELS["general-fast"] == expected

    def test_judges(self):
        assert FUSION_PRESET_JUDGES["general-high"] == "~anthropic/claude-opus-latest"
        assert FUSION_PRESET_JUDGES["general-budget"] == "~anthropic/claude-opus-latest"
        assert FUSION_PRESET_JUDGES["general-fast"] == "~anthropic/claude-sonnet-latest"

    def test_bounds(self):
        assert DEFAULT_FUSION_PRESET == "general-high"
        assert MAX_FUSION_PANEL_MODELS == 8
        assert DEFAULT_FUSION_MAX_TOOL_CALLS == 8
        assert MAX_FUSION_MAX_TOOL_CALLS == 16


class TestFindFusionEntry:
    def test_absent(self):
        assert find_fusion_entry(None) is None
        assert find_fusion_entry([]) is None
        assert find_fusion_entry([{"id": "web"}]) is None
        assert find_fusion_entry("junk") is None

    def test_present(self):
        entry = {"id": "fusion", "preset": "general-budget"}
        assert find_fusion_entry([{"id": "web"}, entry]) is entry

    def test_disabled_entry_still_returned(self):
        entry = {"id": "fusion", "enabled": False}
        assert find_fusion_entry([entry]) is entry

    def test_non_dict_entries_ignored(self):
        entry = {"id": "fusion"}
        assert find_fusion_entry(["fusion", 3, entry]) is entry


class TestResolveFusionRun:
    def test_defaults_when_absent(self):
        plan = resolve_fusion_run(None)
        assert isinstance(plan, FusionRunPlan)
        assert plan.panel_models == FUSION_PRESET_PANELS["general-high"]
        assert plan.judge_model == "~anthropic/claude-opus-latest"
        assert plan.synthesis_model == plan.judge_model
        assert plan.max_tool_calls == 8

    def test_preset_selects_roster_and_judge(self):
        plan = resolve_fusion_run({"id": "fusion", "preset": "general-fast"})
        assert plan.panel_models == FUSION_PRESET_PANELS["general-fast"]
        assert plan.judge_model == "~anthropic/claude-sonnet-latest"
        assert plan.synthesis_model == "~anthropic/claude-sonnet-latest"

    def test_unknown_preset_falls_back_to_default(self):
        plan = resolve_fusion_run({"id": "fusion", "preset": "nope"})
        assert plan.panel_models == FUSION_PRESET_PANELS["general-high"]

    def test_explicit_models_beat_preset(self):
        plan = resolve_fusion_run({
            "id": "fusion",
            "preset": "general-budget",
            "analysis_models": ["a/x", "b/y"],
            "model": "c/z",
        })
        assert plan.panel_models == ("a/x", "b/y")
        assert plan.judge_model == "c/z"
        assert plan.synthesis_model == "c/z"

    def test_explicit_judge_without_models(self):
        plan = resolve_fusion_run({"id": "fusion", "model": "c/z"})
        assert plan.panel_models == FUSION_PRESET_PANELS["general-high"]
        assert plan.judge_model == "c/z"

    def test_panel_capped_at_eight(self):
        models = [f"v/m{i}" for i in range(12)]
        plan = resolve_fusion_run({"id": "fusion", "analysis_models": models})
        assert plan.panel_models == tuple(models[:8])

    def test_panel_entries_sanitized(self):
        plan = resolve_fusion_run({"id": "fusion", "analysis_models": [" a/x ", "", 7, "b/y"]})
        assert plan.panel_models == ("a/x", "b/y")

    def test_empty_analysis_models_falls_back(self):
        plan = resolve_fusion_run({"id": "fusion", "analysis_models": [], "preset": "general-budget"})
        assert plan.panel_models == FUSION_PRESET_PANELS["general-budget"]

    @pytest.mark.parametrize("raw,expected", [(None, 8), (0, 8), (-3, 8), (5, 5), (16, 16), (99, 16), ("7", 8)])
    def test_max_tool_calls_clamped(self, raw, expected):
        entry = {"id": "fusion"}
        if raw is not None:
            entry["max_tool_calls"] = raw
        assert resolve_fusion_run(entry).max_tool_calls == expected


class TestFusionValves:
    def test_backend_valve_defaults_to_internal(self):
        from typing import Literal, get_args, get_origin
        from open_webui_openrouter_pipe.core.config import Valves

        field = Valves.model_fields["FUSION_BACKEND"]
        assert field.default == "internal"
        assert get_origin(field.annotation) is Literal
        assert set(get_args(field.annotation)) == {"openrouter", "internal"}

    def test_prompt_valves_default_to_constants(self):
        from open_webui_openrouter_pipe.core.config import Valves

        valves = Valves()
        assert valves.FUSION_PANEL_SYSTEM_PROMPT == DEFAULT_FUSION_PANEL_SYSTEM_PROMPT
        assert valves.FUSION_JUDGE_SYSTEM_PROMPT == DEFAULT_FUSION_JUDGE_SYSTEM_PROMPT
        assert valves.FUSION_SYNTHESIS_SYSTEM_PROMPT == DEFAULT_FUSION_SYNTHESIS_SYSTEM_PROMPT

    def test_judge_valve_description_warns_json_contract(self):
        from open_webui_openrouter_pipe.core.config import Valves

        desc = Valves.model_fields["FUSION_JUDGE_SYSTEM_PROMPT"].description or ""
        assert "JSON" in desc


class TestPanelDedup:
    def test_duplicate_models_deduped_order_preserved(self):
        plan = resolve_fusion_run({"id": "fusion", "analysis_models": ["a/x", "b/y", "a/x", "a/x", "c/z"]})
        assert plan.panel_models == ("a/x", "b/y", "c/z")

    def test_dedup_applies_before_cap(self):
        models = ["dup/m", "dup/m"] + [f"v/m{i}" for i in range(9)]
        plan = resolve_fusion_run({"id": "fusion", "analysis_models": models})
        assert plan.panel_models == ("dup/m",) + tuple(f"v/m{i}" for i in range(7))
