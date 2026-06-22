"""Tests for the rendered OpenRouter Fusion filter."""
# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportIndexIssue=false, reportMissingImports=false
from __future__ import annotations

import ast
import sys
from types import ModuleType

import pytest

from open_webui_openrouter_pipe.filters.fusion_filter_renderer import (
    FUSION_FILTER_FUNCTION_ID,
    canonical_model_slug,
    is_fusion_model,
    render_openrouter_fusion_filter_source,
)

MARKER = "openrouter_pipe:fusion_filter:v1"
FUSION = "openrouter/fusion"                               # raw OpenRouter slug
PREFIXED = "openrouter.openrouter/fusion"                  # pipe-prefix + raw slug
SANITIZED = "openrouter.fusion"                            # sanitize_model_id: '/'->'.'
OWUI_FULL = "open_webui_openrouter_pipe.openrouter.fusion"  # real runtime body["model"]
OWUI_SHORT = "openrouter.openrouter.fusion"                # pipe-prefix == "openrouter"


def _load(module_name: str = "fusion_filter_rendered") -> ModuleType:
    """Load the rendered filter as a real module — the way OWUI installs it — so
    pydantic can resolve the Literal forward refs (mirrors
    tests/test_image_generation.py:_load_filter_from_source)."""
    if "open_webui.env" not in sys.modules:
        env_mock = ModuleType("open_webui.env")
        env_mock.SRC_LOG_LEVELS = {}  # type: ignore[attr-defined]
        sys.modules["open_webui.env"] = env_mock
    src = render_openrouter_fusion_filter_source(marker=MARKER)
    ast.parse(src)  # rendered source must be valid Python
    module = ModuleType(module_name)
    module.__file__ = f"<{module_name}_rendered>"
    sys.modules[module_name] = module
    exec(compile(src, f"<{module_name}>", "exec"), module.__dict__)  # noqa: S102 - testing generated source
    module.Filter.UserValves.model_rebuild()
    module.Filter.Valves.model_rebuild()
    return module


def _inlet(body, *, valves=None, metadata=None, allow_non_fusion=False):
    mod = _load()
    f = mod.Filter()
    if allow_non_fusion:
        f.valves.ALLOW_ON_NON_FUSION_MODELS = True
    return f.inlet(body, metadata, {"valves": valves or {}})


def _fusion_plugin(body):
    plugins = body.get("plugins") or []
    return next((p for p in plugins if isinstance(p, dict) and p.get("id") == "fusion"), None)


# --- rendering / identity ----------------------------------------------------

def test_rendered_source_is_valid_python_and_self_contained():
    src = render_openrouter_fusion_filter_source(marker=MARKER)
    ast.parse(src)
    assert MARKER in src
    assert f"id: {FUSION_FILTER_FUNCTION_ID}" in src
    assert "class Filter" in src
    assert "open_webui_openrouter_pipe" not in src  # must not import the pipe at runtime


# --- model gating ------------------------------------------------------------

@pytest.mark.parametrize("mid", [FUSION, PREFIXED, SANITIZED, OWUI_FULL, OWUI_SHORT])
def test_is_fusion_model_matches(mid):
    assert is_fusion_model(mid) is True


@pytest.mark.parametrize("mid", [
    "openai/gpt-4o", "openrouter/fusion-pro", "x/openrouter/fusion", "fusion", "", 123,
])
def test_is_fusion_model_rejects(mid):
    assert is_fusion_model(mid) is False


def test_canonical_slug_strips_pipe_prefix():
    assert canonical_model_slug(PREFIXED) == FUSION
    assert canonical_model_slug(FUSION) == FUSION


# --- hard lock ---------------------------------------------------------------

def test_noop_on_non_fusion_model_by_default():
    body = {"model": "openai/gpt-4o", "input": "hi"}
    out = _inlet(body, valves={"FUSION_PRESET": "general-high"})
    assert "plugins" not in out  # untouched


def test_acts_on_non_fusion_model_when_allowed():
    body = {"model": "openai/gpt-4o", "input": "hi"}
    out = _inlet(body, valves={"FUSION_PRESET": "general-high"}, allow_non_fusion=True)
    assert _fusion_plugin(out) == {"id": "fusion", "preset": "general-high"}


def test_inlet_acts_on_real_owui_model_id():
    # Regression: the real runtime body["model"] is the sanitized, function-prefixed
    # all-dots id (sanitize_model_id rewrites '/'->'.'), NOT the raw slug. The inlet
    # must still recognize it as the fusion model and inject config — otherwise the
    # filter is a no-op on the very model it is auto-attached to.
    out = _inlet({"model": OWUI_FULL}, valves={"FUSION_PRESET": "general-high"})
    assert _fusion_plugin(out) == {"id": "fusion", "preset": "general-high"}


# --- config mapping ----------------------------------------------------------

def test_alias_with_no_overrides_adds_no_plugin():
    body = {"model": PREFIXED, "input": "hi"}
    out = _inlet(body)
    assert "plugins" not in out  # don't append an empty {"id":"fusion"} on the alias


def test_preset_maps():
    out = _inlet({"model": FUSION}, valves={"FUSION_PRESET": "general-budget"})
    assert _fusion_plugin(out) == {"id": "fusion", "preset": "general-budget"}


def test_panel_and_judge_and_max_tool_calls_map():
    out = _inlet({"model": FUSION}, valves={
        "FUSION_ANALYSIS_MODELS": "openai/gpt-4o-mini, anthropic/claude-3.5-haiku",
        "FUSION_JUDGE_MODEL": "openai/gpt-4o-mini",
        "FUSION_MAX_TOOL_CALLS": 5,
    })
    p = _fusion_plugin(out)
    assert p["analysis_models"] == ["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku"]
    assert p["model"] == "openai/gpt-4o-mini"
    assert p["max_tool_calls"] == 5


@pytest.mark.parametrize("n,raises", [(0, False), (1, False), (8, False), (9, True)])
def test_panel_cardinality_fails_visibly(n, raises):
    panel = ",".join(f"v/m{i}" for i in range(n))
    mod = _load()
    f = mod.Filter()

    def call():
        return f.inlet({"model": FUSION}, None, {"valves": {"FUSION_ANALYSIS_MODELS": panel}})

    if raises:
        with pytest.raises(mod.FusionConfigError):
            call()
    else:
        out = call()
        p = _fusion_plugin(out)
        if n == 0:
            assert p is None or "analysis_models" not in p
        else:
            assert len(p["analysis_models"]) == n


# --- merge-not-clobber -------------------------------------------------------

def test_existing_fusion_plugin_keys_preserved_and_position_kept():
    body = {
        "model": FUSION,
        "plugins": [
            {"id": "web"},
            {"id": "fusion", "enabled": False, "model": "x/keep-judge"},
            {"id": "file-parser"},
        ],
    }
    out = _inlet(body, valves={"FUSION_PRESET": "general-high"})
    plugins = out["plugins"]
    assert [p["id"] for p in plugins] == ["web", "fusion", "file-parser"]  # position kept
    fp = plugins[1]
    assert fp["enabled"] is False          # preserved
    assert fp["model"] == "x/keep-judge"   # preserved
    assert fp["preset"] == "general-high"  # added


# --- forcing -----------------------------------------------------------------
# Forcing is a single client-side flag: the inlet sets body["tool_choice"]="required"
# directly, exactly as OpenRouter documents (fusion-router.md "Forcing fusion on
# every request"). On the openrouter/fusion alias Fusion is the only injected tool,
# so "required" forces it; with other tools present the model may pick one of those
# (documented OpenRouter behaviour — the filter does not try to defend against it).

def test_force_sets_tool_choice_required():
    out = _inlet({"model": FUSION}, valves={"FUSION_FORCE_TOOL_CALL": True, "FUSION_PRESET": "general-high"})
    assert out["tool_choice"] == "required"


def test_force_default_off_leaves_tool_choice_unset():
    out = _inlet({"model": FUSION}, valves={"FUSION_PRESET": "general-high"})
    assert "tool_choice" not in out


def test_force_sets_required_even_when_other_tools_present():
    # Documented OpenRouter behaviour: with other tools the model MAY pick one, but
    # the client still sets tool_choice="required" — we don't second-guess the docs.
    out = _inlet({"model": FUSION, "tools": [{"type": "function", "function": {"name": "x"}}]},
                 valves={"FUSION_FORCE_TOOL_CALL": True})
    assert out["tool_choice"] == "required"


@pytest.mark.parametrize("body_extra", [
    {"function_call": {"name": "x"}},   # legacy caller tool selection
    {"tool_choice": "auto"},            # caller already chose
    {"tool_choice": ""},                # explicit (falsy) caller value must still block
])
def test_force_not_applied_when_caller_set_tool_choice_or_function_call(body_extra):
    out = _inlet({"model": FUSION, **body_extra}, valves={"FUSION_FORCE_TOOL_CALL": True})
    assert out.get("tool_choice") != "required"   # caller's choice preserved


def test_force_not_applied_when_any_fusion_plugin_disabled():
    # A LATER duplicate {"id":"fusion","enabled":false} must still block forcing —
    # "required" with no active Fusion tool would force some OTHER tool.
    body = {"model": FUSION, "plugins": [{"id": "fusion"}, {"id": "fusion", "enabled": False}]}
    out = _inlet(body, valves={"FUSION_FORCE_TOOL_CALL": True})
    assert "tool_choice" not in out
