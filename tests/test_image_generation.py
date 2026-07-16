"""Tests for OpenRouter native image generation feature.

Coverage:
- Registry: register_image_models dedupe, stale-norm cleanup, atomic rebuild
- Filter renderer: 3 variants produce valid Python with unique markers
- Filter manager: per-model installer with regex prefix matching
- Pydantic image_config field: typed dict round-trip
- Auto-attach: pipe_capabilities image_output, defaultFilterIds writes
- image_help.py: per-model entries, KNOB_GATE consistency

The existing chat-completions response handler (`_materialize_image_entry`,
`_collect_image_output_urls`, `_render_image_markdown`) at
`streaming/streaming_core.py:595-672` is REUSED unchanged for image rendering;
no new adapter or response handler is built.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from open_webui_openrouter_pipe.api.transforms import CompletionsBody, ResponsesBody
from open_webui_openrouter_pipe.filters.image_filter_renderer import (
    build_generic_image_filter_spec,
    build_gemini_image_filter_spec,
    build_sourceful_image_filter_spec,
    build_sourceful_v25_image_filter_spec,
    build_recraft_common_image_filter_spec,
    build_recraft_v3_image_filter_spec,
    build_grok_image_filter_spec,
    render_generic_image_filter_source,
    render_gemini_image_filter_source,
    render_sourceful_image_filter_source,
    render_sourceful_v25_image_filter_source,
    render_recraft_common_image_filter_source,
    render_recraft_v3_image_filter_source,
    render_grok_image_filter_source,
    sanitize_image_filter_id,
)
from open_webui_openrouter_pipe.integrations.image_help import (
    IMAGE_HELP_BY_MODEL,
    render_image_help,
)
from open_webui_openrouter_pipe.models.registry import (
    ModelFamily,
    OpenRouterModelRegistry,
    sanitize_model_id,
)


def _load_filter_from_source(source: str, module_name: str) -> ModuleType:
    """Exec a rendered filter source as a module and return it.

    Mirror of `tests/test_video_generation.py:_load_filter_from_source`. Mocks
    `open_webui.env` so the filter's `from open_webui.env import SRC_LOG_LEVELS`
    succeeds in the test environment.
    """
    if "open_webui.env" not in sys.modules:
        env_mock = ModuleType("open_webui.env")
        env_mock.SRC_LOG_LEVELS = {}  # type: ignore[attr-defined]
        sys.modules["open_webui.env"] = env_mock

    module = ModuleType(module_name)
    module.__file__ = f"<{module_name}_rendered_source>"
    sys.modules[module_name] = module
    exec(compile(source, f"<{module_name}>", "exec"), module.__dict__)
    module.Filter.UserValves.model_rebuild()
    module.Filter.Valves.model_rebuild()
    return module


_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "image_models_catalog.json").read_text()
)
IMAGE_MODELS: list[dict[str, Any]] = _FIXTURE["data"]
IMAGE_BY_ID: dict[str, dict[str, Any]] = {m["id"]: m for m in IMAGE_MODELS}


# =============================================================================
# Registry: register_image_models
# =============================================================================


def test_register_image_models_dedupe_skips_multimodal_in_chat_catalog():
    """Multimodal text+image models already registered via chat catalog must NOT be
    clobbered by `register_image_models`. Only pure-image-only entries are added."""
    OpenRouterModelRegistry._specs = {
        "openai.gpt-5-image": {
            "features": {"image_gen_tool"},
            "capabilities": {"image_generation": True, "vision": True, "file_upload": True},
            "max_completion_tokens": None,
            "supported_parameters": frozenset(),
            "full_model": {
                "id": "openai/gpt-5-image",
                "architecture": {"output_modalities": ["image", "text"]},
            },
            "architecture": {"output_modalities": ["image", "text"]},
        }
    }
    OpenRouterModelRegistry._id_map = {"openai.gpt-5-image": "openai/gpt-5-image"}
    OpenRouterModelRegistry._models = [
        {"id": "openai.gpt-5-image", "norm_id": "openai.gpt-5-image",
         "original_id": "openai/gpt-5-image", "name": "GPT-5 Image"}
    ]

    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)

    # Multimodal entry preserved (chat-side capabilities still present)
    assert "openai.gpt-5-image" in OpenRouterModelRegistry._specs
    spec = OpenRouterModelRegistry._specs["openai.gpt-5-image"]
    assert spec["capabilities"].get("image_generation") is True
    # Pure-image-only models added
    assert "sourceful.riverflow-v2-pro" in OpenRouterModelRegistry._specs
    assert "black-forest-labs.flux.2-pro" in OpenRouterModelRegistry._specs


def test_register_image_models_pure_image_features():
    """Pure-image-only models get features={'image_output', 'image_gen_tool'} —
    NOT 'image_generation' (which is a capability key, not a feature)."""
    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []

    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)

    spec = OpenRouterModelRegistry._specs["sourceful.riverflow-v2-pro"]
    features = set(spec.get("features") or set())
    assert "image_output" in features
    assert "image_gen_tool" in features
    # 'image_generation' is a capabilities key; NOT a feature
    assert "image_generation" not in features


def test_register_image_models_atomic_rebuild_id_changes():
    """Atomic publish: `_specs`/`_id_map`/`_models` are replaced (new objects)."""
    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    pre_specs_id = id(OpenRouterModelRegistry._specs)
    pre_id_map_id = id(OpenRouterModelRegistry._id_map)
    pre_models_id = id(OpenRouterModelRegistry._models)

    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)

    assert id(OpenRouterModelRegistry._specs) != pre_specs_id
    assert id(OpenRouterModelRegistry._id_map) != pre_id_map_id
    assert id(OpenRouterModelRegistry._models) != pre_models_id


def test_register_image_models_stale_norm_cleanup():
    """Image-only norms in `_specs` that aren't in the new fetch are dropped."""
    # Pre-seed with a stale image-only model not in the new fetch.
    OpenRouterModelRegistry._specs = {
        "stale.dropped-model": {
            "features": {"image_output", "image_gen_tool"},
            "architecture": {"output_modalities": ["image"]},
            "capabilities": {},
            "max_completion_tokens": None,
            "supported_parameters": frozenset(),
            "full_model": {},
        }
    }
    OpenRouterModelRegistry._id_map = {"stale.dropped-model": "stale/dropped-model"}
    OpenRouterModelRegistry._models = [
        {"id": "stale.dropped-model", "norm_id": "stale.dropped-model",
         "original_id": "stale/dropped-model", "name": "Stale"}
    ]

    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)

    # Stale norm dropped
    assert "stale.dropped-model" not in OpenRouterModelRegistry._specs
    # New ones added
    assert "sourceful.riverflow-v2-pro" in OpenRouterModelRegistry._specs


def test_register_image_models_last_image_fetch_bumps_only_on_non_empty():
    """`_last_image_fetch` updates only when image_models is non-empty."""
    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry._last_image_fetch = 0.0

    OpenRouterModelRegistry.register_image_models([])
    assert OpenRouterModelRegistry._last_image_fetch == 0.0  # no bump on empty

    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)
    assert OpenRouterModelRegistry._last_image_fetch > 0.0


def test_record_image_attempt_updates_clock():
    OpenRouterModelRegistry._last_image_attempt = 0.0
    OpenRouterModelRegistry.record_image_attempt()
    assert OpenRouterModelRegistry._last_image_attempt > 0.0


@pytest.mark.asyncio
async def test_chat_refresh_preserves_image_only_models():
    """Regression: chat catalog _refresh() must preserve pure-image-only models.

    Bug: pre-fix, /api/v1/models refresh rebuilt cls._models from chat catalog
    only, wiping image-only models registered separately via register_image_models.
    Symptom: Sourceful/FLUX/Seedream models would 'pop in/out' as TTL cycled,
    rejected with not_in_catalog mid-session.
    """
    import aiohttp
    from aioresponses import aioresponses

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry._last_fetch = 0.0
    OpenRouterModelRegistry._next_refresh_after = 0.0

    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)
    pre_refresh_image_norms = {
        n for n, s in OpenRouterModelRegistry._specs.items()
        if "image_output" in (s.get("features") or set())
        and "video_generation" not in (s.get("features") or set())
        and "text" not in ((s.get("architecture") or {}).get("output_modalities") or [])
    }
    assert pre_refresh_image_norms, "Pre-condition: image-only models must be registered"

    chat_catalog_payload = {
        "data": [
            {
                "id": "openai/gpt-5",
                "name": "OpenAI: GPT-5",
                "context_length": 128000,
                "supported_parameters": ["tools"],
                "architecture": {"output_modalities": ["text"], "input_modalities": ["text"]},
                "pricing": {"prompt": "0.000005", "completion": "0.000015"},
            }
        ]
    }
    with aioresponses() as mocked:
        mocked.get(
            "https://openrouter.ai/api/v1/models",
            payload=chat_catalog_payload,
        )
        mocked.get(
            "https://openrouter.ai/api/v1/endpoints/zdr",
            payload={"data": []},
        )
        async with aiohttp.ClientSession() as session:
            await OpenRouterModelRegistry._refresh(
                session,
                base_url="https://openrouter.ai/api/v1",
                api_key="test",
                logger=MagicMock(),
            )

    post_refresh_norms = {m["norm_id"] for m in OpenRouterModelRegistry._models}
    missing = pre_refresh_image_norms - post_refresh_norms
    assert not missing, f"Image-only models wiped by chat refresh: {missing}"
    assert "openai.gpt-5" in post_refresh_norms, "Chat refresh should also register chat models"


# =============================================================================
# Filter renderer: all 7 variants produce valid Python with unique markers
# =============================================================================


@pytest.mark.parametrize("variant_fn,expected_id_suffix", [
    (render_generic_image_filter_source, "openrouter_image_filter_generic"),
    (render_gemini_image_filter_source, "openrouter_image_filter_gemini"),
    (render_sourceful_image_filter_source, "openrouter_image_filter_sourceful"),
    (render_sourceful_v25_image_filter_source, "openrouter_image_filter_sourceful_v25"),
    (render_recraft_common_image_filter_source, "openrouter_image_filter_recraft"),
    (render_recraft_v3_image_filter_source, "openrouter_image_filter_recraft_v3"),
    (render_grok_image_filter_source, "openrouter_image_filter_grok"),
])
def test_filter_source_parses_as_valid_python(variant_fn, expected_id_suffix):
    import ast
    source = variant_fn()
    ast.parse(source)
    assert "class Filter:" in source


def test_filter_specs_have_unique_markers_and_ids():
    specs = [
        build_generic_image_filter_spec(),
        build_gemini_image_filter_spec(),
        build_sourceful_image_filter_spec(),
        build_sourceful_v25_image_filter_spec(),
        build_recraft_common_image_filter_spec(),
        build_recraft_v3_image_filter_spec(),
        build_grok_image_filter_spec(),
    ]
    markers = [s.marker for s in specs]
    function_ids = [s.function_id for s in specs]
    assert len(set(markers)) == 7
    assert len(set(function_ids)) == 7
    for spec in specs:
        assert spec.marker.startswith("openrouter_pipe:image_filter:v1:")


def test_sanitize_image_filter_id_produces_function_ids():
    assert sanitize_image_filter_id("generic") == "openrouter_image_filter_generic"
    assert sanitize_image_filter_id("Gemini") == "openrouter_image_filter_gemini"
    # Empty falls back to generic
    assert sanitize_image_filter_id("") == "openrouter_image_filter_generic"


def test_generic_filter_source_includes_standard_aspect_ratios():
    source = render_generic_image_filter_source()
    for aspect in ["1:1", "16:9", "9:16", "21:9"]:
        assert f'"{aspect}"' in source
    # Standard 1K/2K/4K size
    for size in ["1K", "2K", "4K"]:
        assert f'"{size}"' in source


def test_gemini_filter_source_includes_extended_ratios_and_0_5K():
    source = render_gemini_image_filter_source()
    for aspect in ["1:4", "4:1", "1:8", "8:1"]:
        assert f'"{aspect}"' in source
    assert '"0.5K"' in source


def test_sourceful_filter_source_includes_font_inputs_and_super_res():
    source = render_sourceful_image_filter_source()
    assert "IMAGE_FONT_INPUTS_JSON" in source
    assert "IMAGE_SUPER_RESOLUTION_REFERENCES_JSON" in source
    assert "_MAX_FONT_INPUTS = 2" in source
    assert "_MAX_SUPER_RESOLUTION_REFERENCES = 4" in source
    # The Sourceful Options filter is V2 Pro/Fast ONLY — Riverflow 2.5 gets its
    # own dedicated filter instead (one Sourceful filter per version, never two)
    assert r"^~?sourceful/riverflow-v2-(pro|fast)$" in source


def test_sourceful_v25_filter_source_is_the_single_25_filter():
    """The dedicated 2.5 filter carries fonts (from V2) + all 2.5 additions,
    and intentionally has NO super-resolution knob (dropped in 2.5)."""
    source = render_sourceful_v25_image_filter_source()
    assert "IMAGE_FONT_INPUTS_JSON" in source
    assert "_MAX_FONT_INPUTS = 2" in source
    assert "IMAGE_SCORING_PROMPT" in source
    assert "IMAGE_SCORING_RUBRIC" in source
    assert "IMAGE_BACKGROUND_MODE" in source
    assert "IMAGE_BACKGROUND_HEX_COLOR" in source
    assert "IMAGE_SUPER_RESOLUTION_REFERENCES_JSON" not in source
    # Model gate is v2.5-exact; hex validation pattern survives f-string escaping
    assert r"^~?sourceful/riverflow-v2\.5-(pro|fast)$" in source
    assert "[0-9a-fA-F]{3}" in source and "[0-9a-fA-F]{6}" in source
    for mode in ["original", "transparent", "solid"]:
        assert f'"{mode}"' in source


def test_grok_filter_source_includes_grok_ratios_and_count():
    source = render_grok_image_filter_source()
    assert "IMAGE_GROK_ASPECT_RATIO" in source
    assert "IMAGE_GROK_N" in source
    # Grok-only tall/auto ratios beyond the generic 10
    for aspect in ["9:19.5", "19.5:9", "9:20", "20:9", "1:2", "2:1", "auto"]:
        assert f'"{aspect}"' in source
    assert r"^~?x-ai/grok-imagine-image-" in source


# =============================================================================
# Pydantic image_config: typed dict accepts nested object on both bodies
# =============================================================================


def test_responses_body_image_config_accepts_full_dict():
    body = ResponsesBody.model_validate({
        "model": "sourceful/riverflow-v2-pro",
        "input": [],
        "image_config": {
            "aspect_ratio": "16:9",
            "image_size": "2K",
            "font_inputs": [{"font_url": "https://x/f.ttf", "text": "Hi"}],
            "super_resolution_references": ["https://x/r.jpg"],
        },
    })
    assert body.image_config is not None
    assert body.image_config["aspect_ratio"] == "16:9"
    assert body.image_config["font_inputs"][0]["text"] == "Hi"


def test_responses_body_image_config_extras_preserved():
    """`Dict[str, Any]` allows unknown keys (quality, background, etc. per OpenRouter docs)."""
    body = ResponsesBody.model_validate({
        "model": "openai/gpt-5-image",
        "input": [],
        "image_config": {"aspect_ratio": "16:9", "quality": "high", "background": "transparent"},
    })
    assert body.image_config == {
        "aspect_ratio": "16:9",
        "quality": "high",
        "background": "transparent",
    }


def test_completions_body_image_config_via_extra_allow():
    """`CompletionsBody` doesn't declare image_config typed but accepts via extra='allow'."""
    body = CompletionsBody.model_validate({
        "model": "openai/gpt-5-image",
        "messages": [],
        "image_config": {"aspect_ratio": "16:9"},
    })
    dumped = body.model_dump(exclude_none=True)
    assert dumped["image_config"] == {"aspect_ratio": "16:9"}


# =============================================================================
# Filter manager: regex prefix matching for Gemini and Sourceful
# =============================================================================


def test_gemini_pattern_matches_flash_3x_image_only():
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    pat = FilterManager._GEMINI_IMAGE_PATTERN
    assert pat.match("google/gemini-3.1-flash-image")
    assert pat.match("google/gemini-3.1-flash-image-preview")
    assert pat.match("google/gemini-3.5-flash-image-experimental-preview")
    assert not pat.match("google/gemini-3-pro-image")
    assert not pat.match("google/gemini-3-pro-image-preview")
    assert not pat.match("google/gemini-2.5-flash-image")
    assert not pat.match("google/gemini-4.0-flash-image-preview")
    assert not pat.match("google/gemini-3-pro")
    assert not pat.match("google/gemini-2.5-flash")
    assert not pat.match("openai/gpt-image-2")


def test_image_filter_attach_and_runtime_patterns_are_in_sync():
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    cases = [
        (render_gemini_image_filter_source, "gemini", "_GEMINI_MODEL_PATTERN", FilterManager._GEMINI_IMAGE_PATTERN),
        (render_sourceful_image_filter_source, "sourceful", "_SOURCEFUL_MODEL_PATTERN", FilterManager._SOURCEFUL_IMAGE_PATTERN),
        (render_sourceful_v25_image_filter_source, "sourceful_v25", "_SOURCEFUL_V25_MODEL_PATTERN", FilterManager._SOURCEFUL_V25_IMAGE_PATTERN),
        (render_recraft_common_image_filter_source, "recraft", "_RECRAFT_MODEL_PATTERN", FilterManager._RECRAFT_COMMON_IMAGE_PATTERN),
        (render_recraft_v3_image_filter_source, "recraft_v3", "_RECRAFT_V3_MODEL_PATTERN", FilterManager._RECRAFT_V3_IMAGE_PATTERN),
        (render_grok_image_filter_source, "grok", "_GROK_IMAGINE_IMAGE_PATTERN", FilterManager._GROK_IMAGINE_IMAGE_PATTERN),
    ]
    for render_fn, variant, var_name, attach_pat in cases:
        module = _load_filter_from_source(render_fn(), f"test_drift_guard_{variant}")
        runtime_pat = getattr(module, var_name)
        assert runtime_pat.pattern == attach_pat.pattern, (
            f"{variant}: runtime gate {runtime_pat.pattern!r} != attach pattern {attach_pat.pattern!r}"
        )


def test_sourceful_pattern_matches_v2_pro_and_fast_only():
    """One Sourceful filter per Riverflow version: the V2 'Sourceful Options'
    filter must NOT attach to 2.5 (which has its own dedicated filter) nor to
    hypothetical future versions (they get their own filter when added)."""
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    pat = FilterManager._SOURCEFUL_IMAGE_PATTERN
    assert pat.match("sourceful/riverflow-v2-pro")
    assert pat.match("sourceful/riverflow-v2-fast")
    # Riverflow 2.5 gets the dedicated sourceful_v25 filter instead
    assert not pat.match("sourceful/riverflow-v2.5-pro")
    assert not pat.match("sourceful/riverflow-v2.5-fast")
    # Future versions deliberately excluded — per-version dedicated filters
    assert not pat.match("sourceful/riverflow-v3-pro")
    # Should NOT match: preview variants, max
    assert not pat.match("sourceful/riverflow-v2-max-preview")
    assert not pat.match("sourceful/riverflow-v2-standard-preview")
    assert not pat.match("sourceful/riverflow-v2-fast-preview")


def test_sourceful_v25_pattern_matches_25_pro_and_fast_only():
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    pat = FilterManager._SOURCEFUL_V25_IMAGE_PATTERN
    assert pat.match("sourceful/riverflow-v2.5-pro")
    assert pat.match("sourceful/riverflow-v2.5-fast")
    # V2 must NOT get the 2.5 extras filter
    assert not pat.match("sourceful/riverflow-v2-pro")
    assert not pat.match("sourceful/riverflow-v2-fast")
    # Other versions / preview variants excluded
    assert not pat.match("sourceful/riverflow-v3-pro")
    assert not pat.match("sourceful/riverflow-v2.5-fast-preview")


def test_grok_imagine_pattern_matches_image_models_only():
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    pat = FilterManager._GROK_IMAGINE_IMAGE_PATTERN
    assert pat.match("x-ai/grok-imagine-image-quality")
    # Video sibling and chat models must NOT match
    assert not pat.match("x-ai/grok-imagine-video")
    assert not pat.match("x-ai/grok-4")


def test_mai_image_matches_no_family_pattern():
    """microsoft/mai-image-2.5 gets the generic filter ONLY — no documented
    model-specific params, so no family pattern may claim it."""
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    model_id = "microsoft/mai-image-2.5"
    for pat in (
        FilterManager._GEMINI_IMAGE_PATTERN,
        FilterManager._SOURCEFUL_IMAGE_PATTERN,
        FilterManager._SOURCEFUL_V25_IMAGE_PATTERN,
        FilterManager._RECRAFT_COMMON_IMAGE_PATTERN,
        FilterManager._RECRAFT_V3_IMAGE_PATTERN,
        FilterManager._GROK_IMAGINE_IMAGE_PATTERN,
    ):
        assert not pat.match(model_id), f"{pat.pattern} unexpectedly claims {model_id}"


def test_gpt_image_matches_no_family_pattern():
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    for model_id in ("openai/gpt-image-1", "openai/gpt-image-1-mini", "openai/gpt-image-2"):
        for pat in (
            FilterManager._GEMINI_IMAGE_PATTERN,
            FilterManager._SOURCEFUL_IMAGE_PATTERN,
            FilterManager._SOURCEFUL_V25_IMAGE_PATTERN,
            FilterManager._RECRAFT_COMMON_IMAGE_PATTERN,
            FilterManager._RECRAFT_V3_IMAGE_PATTERN,
            FilterManager._GROK_IMAGINE_IMAGE_PATTERN,
        ):
            assert not pat.match(model_id), f"{pat.pattern} unexpectedly claims {model_id}"


# =============================================================================
# image_help.py: per-model entries
# =============================================================================


def test_image_help_covers_pure_image_models():
    """Pure-image-only models that we'll register have curated help entries."""
    pure_image_models = [
        "sourceful/riverflow-v2-pro",
        "sourceful/riverflow-v2-fast",
        "sourceful/riverflow-v2.5-pro",
        "sourceful/riverflow-v2.5-fast",
        "microsoft/mai-image-2.5",
        "x-ai/grok-imagine-image-quality",
        "black-forest-labs/flux.2-pro",
        "bytedance-seed/seedream-4.5",
    ]
    for model_id in pure_image_models:
        assert model_id in IMAGE_HELP_BY_MODEL, f"{model_id} missing curated help entry"


def test_image_help_covers_multimodal_models():
    """Multimodal text+image models (already in chat catalog) also have curated help."""
    multimodal_models = [
        "openai/gpt-5-image",
        "google/gemini-3.1-flash-image-preview",
        "openrouter/auto",
    ]
    for model_id in multimodal_models:
        assert model_id in IMAGE_HELP_BY_MODEL, f"{model_id} missing curated help entry"


def test_render_image_help_for_known_model_includes_display_name():
    rendered = render_image_help("sourceful/riverflow-v2-pro", IMAGE_BY_ID["sourceful/riverflow-v2-pro"])
    assert "Sourceful: Riverflow V2 Pro" in rendered
    assert "tips" in rendered.lower() or "Tips" in rendered


def test_render_image_help_falls_back_to_catalog_for_unknown_model():
    rendered = render_image_help("unknown/model", {"name": "Unknown", "description": "desc", "architecture": {}})
    assert "Unknown" in rendered
    assert "No curated help" in rendered


def test_render_image_help_gemini_extended_knob_only_for_flash_3x():
    gemini_flash = render_image_help(
        "google/gemini-3.1-flash-image-preview", IMAGE_BY_ID["google/gemini-3.1-flash-image-preview"]
    )
    assert "Image aspect ratio (Gemini extended)" in gemini_flash

    gemini_pro = render_image_help(
        "google/gemini-3-pro-image-preview", IMAGE_BY_ID["google/gemini-3-pro-image-preview"]
    )
    assert "Image aspect ratio (Gemini extended)" not in gemini_pro, (
        "Pro does not advertise extended ratios — the extended knob must not appear in its help"
    )

    gpt_image = render_image_help(
        "openai/gpt-5-image", IMAGE_BY_ID["openai/gpt-5-image"]
    )
    assert "Image aspect ratio (Gemini extended)" not in gpt_image


def test_image_help_gemini_extended_gate_excludes_pro_and_2_5():
    from open_webui_openrouter_pipe.integrations.image_help import _is_gemini_extended_ratio_model

    assert _is_gemini_extended_ratio_model("google/gemini-3.1-flash-image")
    assert _is_gemini_extended_ratio_model("google/gemini-3.1-flash-image-preview")
    assert _is_gemini_extended_ratio_model("~google/gemini-3.1-flash-image")
    assert not _is_gemini_extended_ratio_model("google/gemini-3-pro-image")
    assert not _is_gemini_extended_ratio_model("google/gemini-3-pro-image-preview")
    assert not _is_gemini_extended_ratio_model("google/gemini-2.5-flash-image")
    assert not _is_gemini_extended_ratio_model("openai/gpt-image-2")


def test_image_gate_helpers_and_attach_patterns_accept_tilde_aliases():
    """All image model gates tolerate a leading ~ (router aliases) without
    loosening anything else; the equality-turned-regex helper keeps non-str safety."""
    from open_webui_openrouter_pipe.integrations.image_help import (
        _is_gemini_extended_ratio_model,
        _is_grok_imagine_image,
        _is_recraft,
        _is_recraft_v3,
        _is_sourceful_pro_or_fast,
        _is_sourceful_v2_superres,
        _is_sourceful_v25,
    )
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    assert _is_gemini_extended_ratio_model("~google/gemini-3.1-flash-image")
    assert _is_sourceful_pro_or_fast("~sourceful/riverflow-v2-pro")
    assert _is_sourceful_v2_superres("~sourceful/riverflow-v2-fast")
    assert _is_sourceful_v25("~sourceful/riverflow-v2.5-pro")
    assert _is_recraft("~recraft/recraft-v4")
    assert _is_recraft_v3("~recraft/recraft-v3")
    assert not _is_recraft_v3("~recraft/recraft-v4")
    assert _is_grok_imagine_image("~x-ai/grok-imagine-image-1")

    assert FilterManager._GEMINI_IMAGE_PATTERN.match("~google/gemini-3.1-flash-image")
    assert FilterManager._SOURCEFUL_IMAGE_PATTERN.match("~sourceful/riverflow-v2-pro")
    assert FilterManager._SOURCEFUL_V25_IMAGE_PATTERN.match("~sourceful/riverflow-v2.5-fast")
    assert FilterManager._RECRAFT_COMMON_IMAGE_PATTERN.match("~recraft/recraft-v4")
    assert FilterManager._RECRAFT_V3_IMAGE_PATTERN.match("~recraft/recraft-v3")
    assert not FilterManager._RECRAFT_V3_IMAGE_PATTERN.match("~recraft/recraft-v4")
    assert FilterManager._GROK_IMAGINE_IMAGE_PATTERN.match("~x-ai/grok-imagine-image-1")

    from typing import cast

    assert _is_recraft_v3(cast(str, None)) is False
    assert _is_recraft_v3(cast(str, 42)) is False


def test_render_image_help_sourceful_knobs_only_for_pro_and_fast():
    """Behaviour test: 'Font inputs' knob description appears only in help for Sourceful
    Pro/Fast (non-preview), not for Sourceful preview variants or other models."""
    pro_help = render_image_help(
        "sourceful/riverflow-v2-pro", IMAGE_BY_ID["sourceful/riverflow-v2-pro"]
    )
    assert "Font inputs (JSON array)" in pro_help

    fast_help = render_image_help(
        "sourceful/riverflow-v2-fast", IMAGE_BY_ID["sourceful/riverflow-v2-fast"]
    )
    assert "Font inputs (JSON array)" in fast_help

    max_preview_help = render_image_help(
        "sourceful/riverflow-v2-max-preview", IMAGE_BY_ID["sourceful/riverflow-v2-max-preview"]
    )
    assert "Font inputs (JSON array)" not in max_preview_help


def test_render_image_help_sourceful_v25_shows_extras_but_not_super_res():
    """Riverflow 2.5 help lists fonts + 2.5 extras; super-res (dropped in 2.5)
    must not appear. V2 help keeps the super-res knob."""
    for model_id in ["sourceful/riverflow-v2.5-pro", "sourceful/riverflow-v2.5-fast"]:
        rendered = render_image_help(model_id, IMAGE_BY_ID[model_id])
        assert "Font inputs (JSON array)" in rendered
        assert "Scoring prompt" in rendered
        assert "Scoring rubric" in rendered
        assert "Background mode" in rendered
        assert "Background hex color" in rendered
        assert "Super-resolution references (JSON array)" not in rendered, (
            f"{model_id} help must not advertise the dropped super-res param"
        )

    v2_help = render_image_help(
        "sourceful/riverflow-v2-pro", IMAGE_BY_ID["sourceful/riverflow-v2-pro"]
    )
    assert "Super-resolution references (JSON array)" in v2_help


def test_render_image_help_mai_image_generic_knobs_only():
    rendered = render_image_help(
        "microsoft/mai-image-2.5", IMAGE_BY_ID["microsoft/mai-image-2.5"]
    )
    assert "Microsoft: MAI-Image-2.5" in rendered
    assert "Image aspect ratio" in rendered
    assert "Image size" in rendered
    # No family-specific knobs may leak into MAI help
    for foreign_knob in [
        "Font inputs (JSON array)",
        "Scoring prompt",
        "Image aspect ratio (Gemini extended)",
        "Strength (image-to-image)",
        "Image aspect ratio (Grok Imagine)",
    ]:
        assert foreign_knob not in rendered


def test_render_image_help_grok_shows_grok_knobs():
    rendered = render_image_help(
        "x-ai/grok-imagine-image-quality", IMAGE_BY_ID["x-ai/grok-imagine-image-quality"]
    )
    assert "xAI: Grok Imagine Image Quality" in rendered
    assert "Image aspect ratio (Grok Imagine)" in rendered
    assert "Number of images (1-10)" in rendered


def test_render_image_help_generic_knobs_present_for_all_models():
    """Behaviour test: 'Image aspect ratio' and 'Image size' (the always-active knobs)
    appear in help for any image-output model."""
    for model_id in ["sourceful/riverflow-v2-pro", "openai/gpt-5-image"]:
        rendered = render_image_help(model_id, IMAGE_BY_ID[model_id])
        assert "Image aspect ratio" in rendered
        assert "Image size" in rendered


# =============================================================================
# image_filter_renderer: filter inlet writes correct image_config (sourcecode check)
# =============================================================================


def test_generic_filter_inlet_writes_aspect_ratio():
    source = render_generic_image_filter_source()
    # Source includes the inlet logic that writes overrides["aspect_ratio"]
    assert 'overrides["aspect_ratio"]' in source
    assert 'overrides["image_size"]' in source
    assert 'body["image_config"]' in source


def test_sourceful_filter_inlet_validates_cardinality():
    source = render_sourceful_image_filter_source()
    # Pre-validation rejects > 2 font_inputs and > 4 super_resolution_references
    assert "len(font_inputs) > _MAX_FONT_INPUTS" in source
    assert "len(super_refs) > _MAX_SUPER_RESOLUTION_REFERENCES" in source


def test_gemini_filter_inlet_writes_extended_overrides():
    source = render_gemini_image_filter_source()
    assert 'overrides["aspect_ratio"]' in source
    assert 'overrides["image_size"]' in source
    assert "IMAGE_ASPECT_RATIO_EXTENDED" in source
    assert "IMAGE_SIZE_GEMINI" in source


# =============================================================================
# Integration: full fixture round-trip through register_image_models
# =============================================================================


def test_register_image_models_full_fixture_handles_all_entries():
    """Full image-api dump registers without errors and dedupe works.
    Asserts every pure-image-only model in the catalog appears in _specs."""
    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []

    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)

    # All pure-image-only models registered
    pure_image_norm_ids = [
        ModelFamily.base_model(sanitize_model_id(mid))
        for mid in [
            "sourceful/riverflow-v2-pro",
            "sourceful/riverflow-v2-fast",
            "sourceful/riverflow-v2-max-preview",
            "sourceful/riverflow-v2-standard-preview",
            "sourceful/riverflow-v2-fast-preview",
            "black-forest-labs/flux.2-pro",
            "black-forest-labs/flux.2-max",
            "black-forest-labs/flux.2-flex",
            "black-forest-labs/flux.2-klein-4b",
            "bytedance-seed/seedream-4.5",
            "recraft/recraft-v3",
            "recraft/recraft-v4",
            "recraft/recraft-v4-pro",
            "openai/gpt-image-2",
        ]
    ]
    for norm_id in pure_image_norm_ids:
        assert norm_id in OpenRouterModelRegistry._specs, f"{norm_id} not registered"

    # All registered pure-image models have features
    for norm_id in pure_image_norm_ids:
        features = set(OpenRouterModelRegistry._specs[norm_id].get("features") or set())
        assert "image_output" in features
        assert "image_gen_tool" in features


def test_image_output_in_pipe_capabilities_keys():
    """Runtime check: when catalog_manager builds pipe_capabilities for an
    image-output model, the resulting dict must contain `image_output` as a key
    so the per-model attach decision can gate on it.

    Replaces a prior inspect.getsource grep that would pass even if the literal
    `"image_output"` was only present in a comment or unreachable code.
    """
    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)

    sourceful_norm = ModelFamily.base_model(sanitize_model_id("sourceful/riverflow-v2-pro"))
    assert ModelFamily.supports("image_output", sourceful_norm), (
        "Sourceful Riverflow Pro must report image_output via ModelFamily.supports — "
        "this is the predicate pipe_capabilities['image_output'] derives from."
    )

    chat_only_norm = "synthetic.chat-only"
    OpenRouterModelRegistry._specs[chat_only_norm] = {
        "features": frozenset(),
        "capabilities": {},
        "max_completion_tokens": None,
        "supported_parameters": frozenset(),
        "full_model": {},
        "architecture": {"output_modalities": ["text"]},
    }
    ModelFamily.set_dynamic_specs(OpenRouterModelRegistry._specs)
    assert not ModelFamily.supports("image_output", chat_only_norm), (
        "Text-only model must NOT report image_output."
    )


def test_sourceful_pattern_uses_fullmatch_anchors():
    """Sourceful regex must anchor with `^...$` so suffix variants don't sneak through.
    Originally the test used `pat.match()` which only anchors at start; this
    explicit fullmatch test ensures `riverflow-v2-pro-extra` is rejected.
    """
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    pat = FilterManager._SOURCEFUL_IMAGE_PATTERN
    assert pat.fullmatch("sourceful/riverflow-v2-pro")
    assert pat.fullmatch("sourceful/riverflow-v2-fast")
    # Suffix attack — must NOT match
    assert not pat.fullmatch("sourceful/riverflow-v2-pro-extra")
    assert not pat.fullmatch("sourceful/riverflow-v2-fast-experimental")
    # Prefix attack
    assert not pat.fullmatch("evil-sourceful/riverflow-v2-pro")


def test_register_image_models_full_fixture_exact_pure_image_count():
    """Exact-count assertion: register_image_models must add EXACTLY the 18
    pure-image-only models from the fixture (7 Sourceful + 4 Flux + 1 Seedream
    + 3 Recraft + 1 Microsoft MAI + 1 Grok Imagine + 1 GPT Image), NOT any
    multimodal entries (gpt-5-image variants, gemini-image variants,
    openrouter/auto — those land in chat catalog separately).
    """
    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []

    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)

    image_only_specs = [
        norm_id for norm_id, spec in OpenRouterModelRegistry._specs.items()
        if "image_output" in (spec.get("features") or set())
    ]
    assert len(image_only_specs) == 18, (
        f"Expected 18 pure-image-only registrations, got {len(image_only_specs)}: {image_only_specs}"
    )


# =============================================================================
# Filter inlet RUNTIME behaviour — exec the rendered filter and run inlet()
# =============================================================================


def test_generic_filter_inlet_writes_aspect_and_size_into_body():
    source = render_generic_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_generic_runtime")

    user_valves = module.Filter.UserValves(IMAGE_ASPECT_RATIO="16:9", IMAGE_SIZE="2K")
    body: dict[str, Any] = {"model": "openai/gpt-5-image", "messages": []}
    module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})

    assert body["image_config"] == {"aspect_ratio": "16:9", "image_size": "2K"}


def test_generic_filter_inlet_preserves_existing_image_config_keys():
    """If body already has image_config with unrelated keys, the filter merges
    its overrides while preserving the existing keys (per-key overwrite semantics)."""
    source = render_generic_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_generic_preserve")

    user_valves = module.Filter.UserValves(IMAGE_ASPECT_RATIO="16:9")
    body: dict[str, Any] = {
        "model": "openai/gpt-5-image",
        "image_config": {"quality": "high", "background": "transparent"},
        "messages": [],
    }
    module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})

    assert body["image_config"]["aspect_ratio"] == "16:9"  # added
    assert body["image_config"]["quality"] == "high"  # preserved
    assert body["image_config"]["background"] == "transparent"  # preserved


def test_generic_filter_inlet_no_op_when_user_valves_empty():
    source = render_generic_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_generic_noop")

    body: dict[str, Any] = {"model": "openai/gpt-5-image", "messages": []}
    module.Filter().inlet(body, __metadata__={}, __user__={"valves": module.Filter.UserValves()})

    assert "image_config" not in body


def test_gemini_filter_inlet_writes_extended_for_flash_3x_not_pro():
    source = render_gemini_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_gemini_runtime")

    def applied(model_id: str) -> Any:
        body: dict[str, Any] = {"model": model_id, "messages": []}
        user_valves = module.Filter.UserValves(IMAGE_ASPECT_RATIO_EXTENDED="4:1", IMAGE_SIZE_GEMINI="0.5K")
        module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})
        return body.get("image_config")

    for flash_id in ("google/gemini-3.1-flash-image", "google/gemini-3.1-flash-image-preview"):
        assert applied(flash_id) == {"aspect_ratio": "4:1", "image_size": "0.5K"}, (
            f"Flash 3.x {flash_id} must apply the extended ratio/0.5K"
        )

    for excluded_id in (
        "google/gemini-3-pro-image",
        "google/gemini-3-pro-image-preview",
        "google/gemini-2.5-flash-image",
        "openai/gpt-5-image",
    ):
        assert applied(excluded_id) is None, (
            f"{excluded_id} must not get extended knobs"
        )


def test_sourceful_filter_inlet_writes_only_for_sourceful_pro_or_fast():
    source = render_sourceful_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_sourceful_runtime")
    font_input_json = '[{"font_url": "https://example.com/f.ttf", "text": "Hi"}]'
    user_valves = module.Filter.UserValves(IMAGE_FONT_INPUTS_JSON=font_input_json)

    # Sourceful Pro — written
    pro_body: dict[str, Any] = {"model": "sourceful/riverflow-v2-pro", "messages": []}
    module.Filter().inlet(pro_body, __metadata__={}, __user__={"valves": user_valves})
    assert pro_body["image_config"]["font_inputs"] == [
        {"font_url": "https://example.com/f.ttf", "text": "Hi"}
    ]

    # Sourceful Max Preview — must be ignored (model gate excludes preview variants)
    preview_body: dict[str, Any] = {"model": "sourceful/riverflow-v2-max-preview", "messages": []}
    module.Filter().inlet(preview_body, __metadata__={}, __user__={"valves": user_valves})
    assert "image_config" not in preview_body, (
        "Sourceful filter must not modify body when model is preview variant"
    )

    # Non-Sourceful model — must be ignored
    other_body: dict[str, Any] = {"model": "openai/gpt-5-image", "messages": []}
    module.Filter().inlet(other_body, __metadata__={}, __user__={"valves": user_valves})
    assert "image_config" not in other_body


def test_sourceful_filter_inlet_rejects_excess_font_inputs():
    """Cardinality cap enforcement: 3+ font_inputs raises ImageGenerationError."""
    source = render_sourceful_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_sourceful_cap")

    too_many = json.dumps([
        {"font_url": "https://example.com/1.ttf", "text": "a"},
        {"font_url": "https://example.com/2.ttf", "text": "b"},
        {"font_url": "https://example.com/3.ttf", "text": "c"},
    ])
    user_valves = module.Filter.UserValves(IMAGE_FONT_INPUTS_JSON=too_many)
    body: dict[str, Any] = {"model": "sourceful/riverflow-v2-pro", "messages": []}

    with pytest.raises(module.ImageGenerationError, match="font_inputs"):
        module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})


def test_sourceful_filter_inlet_rejects_excess_super_resolution_references():
    source = render_sourceful_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_sourceful_super")

    too_many = json.dumps(["https://x/1.jpg", "https://x/2.jpg", "https://x/3.jpg", "https://x/4.jpg", "https://x/5.jpg"])
    user_valves = module.Filter.UserValves(IMAGE_SUPER_RESOLUTION_REFERENCES_JSON=too_many)
    body: dict[str, Any] = {"model": "sourceful/riverflow-v2-fast", "messages": []}

    with pytest.raises(module.ImageGenerationError, match="super_resolution_references"):
        module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})


def test_sourceful_filter_inlet_fully_no_ops_on_v25_models():
    """One Sourceful filter per Riverflow version: the V2 'Sourceful Options'
    filter must leave 2.5 bodies completely untouched (model gate), even with
    malformed valve values — 2.5 models use the dedicated sourceful_v25 filter."""
    source = render_sourceful_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_sourceful_v25_gate")
    user_valves = module.Filter.UserValves(
        IMAGE_FONT_INPUTS_JSON='[{"font_url": "https://x/f.ttf", "text": "Hi"}]',
        IMAGE_SUPER_RESOLUTION_REFERENCES_JSON='["https://x/ref.png"]',
    )

    # V2 — both knobs emitted
    v2_body: dict[str, Any] = {"model": "sourceful/riverflow-v2-pro", "messages": []}
    module.Filter().inlet(v2_body, __metadata__={}, __user__={"valves": user_valves})
    assert "font_inputs" in v2_body["image_config"]
    assert "super_resolution_references" in v2_body["image_config"]

    # 2.5 — the V2 filter does not touch the body at all
    v25_body: dict[str, Any] = {"model": "sourceful/riverflow-v2.5-pro", "messages": []}
    module.Filter().inlet(v25_body, __metadata__={}, __user__={"valves": user_valves})
    assert "image_config" not in v25_body, (
        "the V2 Sourceful filter must not attach knobs to Riverflow 2.5 models"
    )

    # Even malformed JSON is inert on 2.5 (model gate precedes parsing)...
    bad_valves = module.Filter.UserValves(
        IMAGE_SUPER_RESOLUTION_REFERENCES_JSON="{not valid json",
    )
    inert_body: dict[str, Any] = {"model": "sourceful/riverflow-v2.5-fast", "messages": []}
    module.Filter().inlet(inert_body, __metadata__={}, __user__={"valves": bad_valves})
    assert "image_config" not in inert_body

    # ...while the same malformed value still raises loudly on V2
    v2_bad: dict[str, Any] = {"model": "sourceful/riverflow-v2-fast", "messages": []}
    with pytest.raises(module.ImageGenerationError, match="not valid JSON"):
        module.Filter().inlet(v2_bad, __metadata__={}, __user__={"valves": bad_valves})


def test_sourceful_v25_filter_inlet_writes_fonts_and_extras_only_for_25_models():
    source = render_sourceful_v25_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_sourceful_v25_runtime")
    user_valves = module.Filter.UserValves(
        IMAGE_FONT_INPUTS_JSON='[{"font_url": "https://x/f.ttf", "text": "Hi"}]',
        IMAGE_SCORING_PROMPT="crisp vector logo",
        IMAGE_SCORING_RUBRIC="sharp edges, flat colors",
        IMAGE_BACKGROUND_MODE="solid",
        IMAGE_BACKGROUND_HEX_COLOR="#0AF",
    )

    v25_body: dict[str, Any] = {"model": "sourceful/riverflow-v2.5-fast", "messages": []}
    module.Filter().inlet(v25_body, __metadata__={}, __user__={"valves": user_valves})
    assert v25_body["image_config"] == {
        "font_inputs": [{"font_url": "https://x/f.ttf", "text": "Hi"}],
        "scoring_prompt": "crisp vector logo",
        "scoring_rubric": "sharp edges, flat colors",
        "background_mode": "solid",
        "background_hex_color": "#0AF",
    }

    # V2 model with the SAME valves — must be ignored (model gate)
    v2_body: dict[str, Any] = {"model": "sourceful/riverflow-v2-pro", "messages": []}
    module.Filter().inlet(v2_body, __metadata__={}, __user__={"valves": user_valves})
    assert "image_config" not in v2_body


def test_sourceful_v25_filter_inlet_rejects_excess_font_inputs():
    """The dedicated 2.5 filter enforces the same font cardinality cap as V2."""
    source = render_sourceful_v25_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_sourceful_v25_fontcap")

    too_many = json.dumps([
        {"font_url": "https://example.com/1.ttf", "text": "a"},
        {"font_url": "https://example.com/2.ttf", "text": "b"},
        {"font_url": "https://example.com/3.ttf", "text": "c"},
    ])
    user_valves = module.Filter.UserValves(IMAGE_FONT_INPUTS_JSON=too_many)
    body: dict[str, Any] = {"model": "sourceful/riverflow-v2.5-pro", "messages": []}

    with pytest.raises(module.ImageGenerationError, match="font_inputs"):
        module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})


def test_sourceful_v25_filter_inlet_validates_background_hex():
    source = render_sourceful_v25_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_sourceful_v25_hex")
    body: dict[str, Any] = {"model": "sourceful/riverflow-v2.5-pro", "messages": []}

    # Hex without solid mode → clear inlet error
    no_mode = module.Filter.UserValves(IMAGE_BACKGROUND_HEX_COLOR="#FFFFFF")
    with pytest.raises(module.ImageGenerationError, match="solid"):
        module.Filter().inlet(dict(body), __metadata__={}, __user__={"valves": no_mode})

    # Malformed hex → clear inlet error
    bad_hex = module.Filter.UserValves(
        IMAGE_BACKGROUND_MODE="solid", IMAGE_BACKGROUND_HEX_COLOR="red"
    )
    with pytest.raises(module.ImageGenerationError, match="#RGB or #RRGGBB"):
        module.Filter().inlet(dict(body), __metadata__={}, __user__={"valves": bad_hex})

    # transparent mode alone is valid (no hex required)
    transparent = module.Filter.UserValves(IMAGE_BACKGROUND_MODE="transparent")
    out_body = dict(body)
    module.Filter().inlet(out_body, __metadata__={}, __user__={"valves": transparent})
    assert out_body["image_config"] == {"background_mode": "transparent"}


@pytest.mark.parametrize("render_fn,slug,valves,expected_key", [
    (render_gemini_image_filter_source, "google/gemini-3.1-flash-image-preview",
     {"IMAGE_ASPECT_RATIO_EXTENDED": "4:1"}, "aspect_ratio"),
    (render_sourceful_image_filter_source, "sourceful/riverflow-v2-pro",
     {"IMAGE_FONT_INPUTS_JSON": '[{"font_url": "https://x/f.ttf", "text": "Hi"}]'}, "font_inputs"),
    (render_sourceful_v25_image_filter_source, "sourceful/riverflow-v2.5-pro",
     {"IMAGE_SCORING_PROMPT": "crisp logo"}, "scoring_prompt"),
    (render_recraft_common_image_filter_source, "recraft/recraft-v4",
     {"IMAGE_STRENGTH": 0.5}, "strength"),
    (render_recraft_v3_image_filter_source, "recraft/recraft-v3",
     {"IMAGE_RECRAFT_STYLE": "Photorealism"}, "style"),
    (render_grok_image_filter_source, "x-ai/grok-imagine-image-quality",
     {"IMAGE_GROK_N": 3}, "n"),
])
def test_image_filter_inlet_matches_pipe_prefixed_model_id(render_fn, slug, valves, expected_key):
    """OWUI manifold passes pipe-namespaced model ids ("<pipe_id>.<vendor>/<model>")
    to inlet filters. The model gates must normalize that prefix or they silently
    no-op in production. Prefixed ids must write knobs, bare ids must still work,
    and a foreign model must still be ignored."""
    module = _load_filter_from_source(render_fn(), f"test_prefix_{expected_key}")
    prefix = "open_webui_openrouter_pipe."

    prefixed: dict[str, Any] = {"model": prefix + slug, "messages": []}
    module.Filter().inlet(prefixed, __metadata__={}, __user__={"valves": dict(valves)})
    assert expected_key in (prefixed.get("image_config") or {}), (
        f"{slug}: gate failed to match the pipe-prefixed model id"
    )

    bare: dict[str, Any] = {"model": slug, "messages": []}
    module.Filter().inlet(bare, __metadata__={}, __user__={"valves": dict(valves)})
    assert expected_key in (bare.get("image_config") or {}), f"{slug}: bare id regressed"

    foreign: dict[str, Any] = {"model": prefix + "openai/gpt-5-image", "messages": []}
    module.Filter().inlet(foreign, __metadata__={}, __user__={"valves": dict(valves)})
    assert "image_config" not in foreign, f"{slug}: foreign model must stay ignored"


def test_grok_filter_inlet_writes_grok_knobs_only_for_grok_models():
    source = render_grok_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_grok_runtime")
    user_valves = module.Filter.UserValves(IMAGE_GROK_ASPECT_RATIO="9:19.5", IMAGE_GROK_N=4)

    grok_body: dict[str, Any] = {"model": "x-ai/grok-imagine-image-quality", "messages": []}
    module.Filter().inlet(grok_body, __metadata__={}, __user__={"valves": user_valves})
    assert grok_body["image_config"] == {"aspect_ratio": "9:19.5", "n": 4}

    # Non-Grok model with the SAME valves — must be ignored (model gate)
    other_body: dict[str, Any] = {"model": "openai/gpt-5-image", "messages": []}
    module.Filter().inlet(other_body, __metadata__={}, __user__={"valves": user_valves})
    assert "image_config" not in other_body

    # n=0 is the skip sentinel; empty ratio skipped too
    defaults_body: dict[str, Any] = {"model": "x-ai/grok-imagine-image-quality", "messages": []}
    module.Filter().inlet(defaults_body, __metadata__={}, __user__={"valves": module.Filter.UserValves()})
    assert "image_config" not in defaults_body


def test_sourceful_filter_inlet_rejects_invalid_json():
    source = render_sourceful_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_sourceful_badjson")

    user_valves = module.Filter.UserValves(IMAGE_FONT_INPUTS_JSON="not-json")
    body: dict[str, Any] = {"model": "sourceful/riverflow-v2-pro", "messages": []}

    with pytest.raises(module.ImageGenerationError, match="not valid JSON"):
        module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})


def test_sourceful_filter_inlet_rejects_missing_font_url_or_text():
    source = render_sourceful_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_sourceful_missing")

    user_valves = module.Filter.UserValves(IMAGE_FONT_INPUTS_JSON='[{"font_url": "https://x/f.ttf"}]')
    body: dict[str, Any] = {"model": "sourceful/riverflow-v2-pro", "messages": []}

    with pytest.raises(module.ImageGenerationError, match="non-empty 'font_url' and 'text'"):
        module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})


def test_two_filters_concurrent_merge_into_image_config():
    """Generic + Gemini filters running on the same body produce a merged
    image_config (Gemini overrides aspect/size when set; otherwise generic
    values stand)."""
    generic_module = _load_filter_from_source(
        render_generic_image_filter_source(), "test_image_filter_two_generic"
    )
    gemini_module = _load_filter_from_source(
        render_gemini_image_filter_source(), "test_image_filter_two_gemini"
    )

    body: dict[str, Any] = {"model": "google/gemini-3.1-flash-image-preview", "messages": []}

    generic_uv = generic_module.Filter.UserValves(IMAGE_ASPECT_RATIO="1:1", IMAGE_SIZE="2K")
    generic_module.Filter().inlet(body, __metadata__={}, __user__={"valves": generic_uv})

    # Now Gemini filter runs — extended ratio overrides generic
    gemini_uv = gemini_module.Filter.UserValves(IMAGE_ASPECT_RATIO_EXTENDED="4:1", IMAGE_SIZE_GEMINI="0.5K")
    gemini_module.Filter().inlet(body, __metadata__={}, __user__={"valves": gemini_uv})

    assert body["image_config"]["aspect_ratio"] == "4:1"  # gemini override won
    assert body["image_config"]["image_size"] == "0.5K"  # gemini override won


# =============================================================================
# ensure_openrouter_image_filter_function_ids — installer behaviour
# =============================================================================


@pytest.mark.asyncio
async def test_installer_returns_generic_only_for_image_output_model():
    """Installer must return [generic_id] for plain image-output models (e.g. Flux)."""
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry.register_image_models([
        m for m in IMAGE_MODELS if m["id"] == "black-forest-labs/flux.2-pro"
    ])

    pipe = MagicMock()
    pipe.valves.AUTO_INSTALL_IMAGE_FILTERS = True
    pipe.valves.ENABLE_OPENROUTER_IMAGE_GENERATION = True
    fm = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    fm._ensure_filter_installed = AsyncMock(side_effect=lambda **kwargs: kwargs["preferred_id"])

    flux = OpenRouterModelRegistry.list_models()[0]
    result = await fm.ensure_openrouter_image_filter_function_ids([flux])

    assert flux["id"] in result
    assert "openrouter_image_filter_generic" in result[flux["id"]]
    assert "openrouter_image_filter_gemini" not in result[flux["id"]]
    assert "openrouter_image_filter_sourceful" not in result[flux["id"]]


@pytest.mark.asyncio
async def test_installer_isolates_per_filter_install_failures():
    """If gemini install raises, generic still gets attached to all matching models."""
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)

    pipe = MagicMock()
    fm = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())

    async def _fake_install(**kwargs):
        if "Gemini" in kwargs["desired_name"]:
            raise RuntimeError("simulated gemini install failure")
        return kwargs["preferred_id"]

    fm._ensure_filter_installed = AsyncMock(side_effect=_fake_install)

    image_only_models = [
        m for m in OpenRouterModelRegistry.list_models()
        if "image_output" in (OpenRouterModelRegistry._specs.get(m["norm_id"], {}).get("features") or set())
    ]
    result = await fm.ensure_openrouter_image_filter_function_ids(image_only_models)

    # All registered pure-image models should at least have generic attached
    assert all(
        "openrouter_image_filter_generic" in ids
        for ids in result.values()
    )


@pytest.mark.asyncio
async def test_installer_dual_keys_no_aliasing():
    """`installed[model_id]` and `installed[original_id]` must be SEPARATE list
    objects (no shared reference) so future mutation of one doesn't corrupt the other."""
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry.register_image_models([
        m for m in IMAGE_MODELS if m["id"] == "sourceful/riverflow-v2-pro"
    ])

    pipe = MagicMock()
    fm = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    fm._ensure_filter_installed = AsyncMock(side_effect=lambda **kwargs: kwargs["preferred_id"])

    sourceful = OpenRouterModelRegistry.list_models()[0]
    result = await fm.ensure_openrouter_image_filter_function_ids([sourceful])

    # Both sanitized and original ids should map to the SAME list contents
    sanitized_ids = result.get(sourceful["id"])
    original_ids = result.get(sourceful["original_id"])
    assert sanitized_ids is not None and original_ids is not None
    assert sanitized_ids == original_ids
    # But they must be DIFFERENT list objects (no aliasing)
    assert sanitized_ids is not original_ids
    sanitized_ids.append("test-injection")
    assert "test-injection" not in original_ids, (
        "Aliasing detected: mutating sanitized_ids modified original_ids"
    )


@pytest.mark.asyncio
async def test_installer_attaches_gemini_filter_to_flash_3x_not_pro():
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []

    architecture = {
        "input_modalities": ["text", "image"],
        "output_modalities": ["image", "text"],
    }
    derived_features = OpenRouterModelRegistry._derive_features(  # type: ignore[attr-defined]
        supported_parameters={"temperature", "tools"},
        architecture=architecture,
        pricing={},
    )
    assert "image_output" in derived_features

    flash_ids = {"google/gemini-3.1-flash-image", "google/gemini-3.1-flash-image-preview"}
    pro_ids = {"google/gemini-3-pro-image", "google/gemini-3-pro-image-preview"}
    models = []
    for original_id in sorted(flash_ids | pro_ids):
        sanitized = sanitize_model_id(original_id)
        norm_id = ModelFamily.base_model(sanitized)
        OpenRouterModelRegistry._specs[norm_id] = {
            "features": set(derived_features),
            "capabilities": {"image_generation": True},
            "max_completion_tokens": None,
            "supported_parameters": frozenset({"temperature", "tools"}),
            "full_model": {"id": original_id, "architecture": architecture},
            "architecture": architecture,
        }
        OpenRouterModelRegistry._id_map[norm_id] = original_id
        models.append(
            {"id": sanitized, "norm_id": norm_id, "original_id": original_id, "name": original_id}
        )
    ModelFamily.set_dynamic_specs(OpenRouterModelRegistry._specs)

    pipe = MagicMock()
    fm = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    fm._ensure_filter_installed = AsyncMock(side_effect=lambda **kwargs: kwargs["preferred_id"])

    result = await fm.ensure_openrouter_image_filter_function_ids(models)

    for model in models:
        ids = result.get(model["id"]) or result.get(model["original_id"])
        assert ids is not None, f"{model['original_id']} received no filter ids"
        assert "openrouter_image_filter_generic" in ids
        if model["original_id"] in flash_ids:
            assert "openrouter_image_filter_gemini" in ids, (
                f"Flash 3.x {model['original_id']} must auto-attach the gemini extended-ratio filter"
            )
        else:
            assert "openrouter_image_filter_gemini" not in ids, (
                f"Pro {model['original_id']} must NOT get the gemini filter — it lacks extended ratios"
            )


@pytest.mark.asyncio
async def test_installer_attaches_generic_only_to_gpt_image():
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry.register_image_models(
        [m for m in IMAGE_MODELS if m["id"] == "openai/gpt-image-2"]
    )

    pipe = MagicMock()
    fm = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    fm._ensure_filter_installed = AsyncMock(side_effect=lambda **kwargs: kwargs["preferred_id"])

    gpt_image = OpenRouterModelRegistry.list_models()[0]
    result = await fm.ensure_openrouter_image_filter_function_ids([gpt_image])

    ids = result.get(gpt_image["id"])
    assert ids == ["openrouter_image_filter_generic"], (
        f"gpt-image must auto-attach generic-only, got {ids}"
    )


# =============================================================================
# image_catalog and image_client — TTL + error handling smoke tests
# =============================================================================


@pytest.mark.asyncio
async def test_image_catalog_skip_when_disabled():
    """Master valve disabled → no fetch attempt."""
    from open_webui_openrouter_pipe.integrations.image_catalog import ensure_image_catalog_loaded

    OpenRouterModelRegistry._last_image_attempt = 0.0
    OpenRouterModelRegistry._last_image_fetch = 0.0
    valves = MagicMock()
    valves.ENABLE_OPENROUTER_IMAGE_GENERATION = False
    valves.BASE_URL = "https://openrouter.ai/api/v1"

    session = MagicMock()
    await ensure_image_catalog_loaded(
        session, valves=valves, api_key="test", logger=MagicMock(), cache_seconds=3600
    )
    # Skipped: no attempt timestamp recorded
    assert OpenRouterModelRegistry._last_image_attempt == 0.0


@pytest.mark.asyncio
async def test_image_catalog_master_disable_clears_existing_image_models():
    """When ENABLE_OPENROUTER_IMAGE_GENERATION flips off after models were
    previously registered, the catalog loader must drop them via register_image_models([])
    so they vanish from OWUI's dropdown (don't persist until restart)."""
    from open_webui_openrouter_pipe.integrations.image_catalog import ensure_image_catalog_loaded

    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)
    pre_count = len([
        norm_id for norm_id, spec in OpenRouterModelRegistry._specs.items()
        if "image_output" in (spec.get("features") or set())
    ])
    assert pre_count > 0

    valves = MagicMock()
    valves.ENABLE_OPENROUTER_IMAGE_GENERATION = False
    valves.BASE_URL = "https://openrouter.ai/api/v1"
    session = MagicMock()

    await ensure_image_catalog_loaded(
        session, valves=valves, api_key="test", logger=MagicMock(), cache_seconds=3600
    )

    post_count = len([
        norm_id for norm_id, spec in OpenRouterModelRegistry._specs.items()
        if "image_output" in (spec.get("features") or set())
    ])
    assert post_count == 0, "Master-disable must clean up previously-registered image models"


@pytest.mark.asyncio
async def test_image_catalog_ttl_gate_skips_within_window():
    """Within `cache_seconds` of last attempt, no new fetch is initiated."""
    import time

    from open_webui_openrouter_pipe.integrations.image_catalog import ensure_image_catalog_loaded

    OpenRouterModelRegistry._last_image_attempt = time.time()  # just now
    OpenRouterModelRegistry._last_image_fetch = 0.0

    valves = MagicMock()
    valves.ENABLE_OPENROUTER_IMAGE_GENERATION = True
    valves.BASE_URL = "https://openrouter.ai/api/v1"
    valves.HTTP_REFERER_OVERRIDE = ""

    session = MagicMock()
    # Should return without instantiating client / attempting fetch
    await ensure_image_catalog_loaded(
        session, valves=valves, api_key="test", logger=MagicMock(), cache_seconds=3600
    )

    # session.get was never called — TTL gate worked
    assert not session.get.called


@pytest.mark.asyncio
async def test_image_client_list_models_returns_filtered_list():
    """OpenRouterImageClient.list_models() returns only dict entries from the response."""
    from open_webui_openrouter_pipe.integrations.image_client import OpenRouterImageClient

    payload = {
        "data": [
            {"id": "sourceful/riverflow-v2-pro", "name": "Pro"},
            "not_a_dict",  # Should be filtered out
            {"id": "black-forest-labs/flux.2-pro", "name": "Flux Pro"},
            42,  # Should be filtered out
        ]
    }

    class _MockResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def json(self):
            return payload

        def raise_for_status(self):
            pass

    class _MockSession:
        def get(self, url, headers=None):
            return _MockResponse()

    client = OpenRouterImageClient(
        _MockSession(),  # type: ignore[arg-type]
        base_url="https://openrouter.ai/api/v1",
        api_key="test",
        logger=MagicMock(),
    )
    models = await client.list_models()
    assert len(models) == 2
    assert all(isinstance(m, dict) for m in models)
    assert {m["id"] for m in models} == {"sourceful/riverflow-v2-pro", "black-forest-labs/flux.2-pro"}


@pytest.mark.asyncio
async def test_image_client_list_models_handles_missing_data_field():
    """If response is missing 'data' field, return empty list (don't crash)."""
    from open_webui_openrouter_pipe.integrations.image_client import OpenRouterImageClient

    class _MockResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def json(self):
            return {}

        def raise_for_status(self):
            pass

    class _MockSession:
        def get(self, url, headers=None):
            return _MockResponse()

    client = OpenRouterImageClient(
        _MockSession(),  # type: ignore[arg-type]
        base_url="https://openrouter.ai/api/v1",
        api_key="test",
        logger=MagicMock(),
    )
    models = await client.list_models()
    assert models == []


@pytest.mark.asyncio
async def test_image_catalog_happy_path_registers_models():
    """End-to-end: enabled + TTL expired + successful fetch → register_image_models
    is called with the response payload, last_image_attempt is bumped."""
    import aiohttp

    from open_webui_openrouter_pipe.integrations import image_catalog

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry._last_image_attempt = 0.0
    OpenRouterModelRegistry._last_image_fetch = 0.0

    valves = MagicMock()
    valves.ENABLE_OPENROUTER_IMAGE_GENERATION = True
    valves.BASE_URL = "https://openrouter.ai/api/v1"
    valves.HTTP_REFERER_OVERRIDE = ""

    class _MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def list_models(self):
            return [
                m for m in IMAGE_MODELS
                if m["id"] in ("sourceful/riverflow-v2-pro", "black-forest-labs/flux.2-pro")
            ]

    original_client = image_catalog.OpenRouterImageClient
    image_catalog.OpenRouterImageClient = _MockClient  # type: ignore[misc]
    try:
        await image_catalog.ensure_image_catalog_loaded(
            session=MagicMock(),
            valves=valves,
            api_key="test-key",
            logger=MagicMock(),
            cache_seconds=3600,
        )
    finally:
        image_catalog.OpenRouterImageClient = original_client  # type: ignore[misc]

    # Both registrations succeeded
    sourceful_norm = ModelFamily.base_model(sanitize_model_id("sourceful/riverflow-v2-pro"))
    flux_norm = ModelFamily.base_model(sanitize_model_id("black-forest-labs/flux.2-pro"))
    assert sourceful_norm in OpenRouterModelRegistry._specs
    assert flux_norm in OpenRouterModelRegistry._specs
    # Attempt clock bumped on success
    assert OpenRouterModelRegistry._last_image_attempt > 0.0
    # Fetch clock bumped because non-empty result
    assert OpenRouterModelRegistry._last_image_fetch > 0.0


@pytest.mark.asyncio
async def test_image_catalog_network_failure_records_attempt_no_models():
    """If list_models raises aiohttp.ClientError, record_image_attempt is still
    bumped (so TTL gate kicks in for the next call), no models registered."""
    import aiohttp

    from open_webui_openrouter_pipe.integrations import image_catalog

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry._last_image_attempt = 0.0
    OpenRouterModelRegistry._last_image_fetch = 0.0

    valves = MagicMock()
    valves.ENABLE_OPENROUTER_IMAGE_GENERATION = True
    valves.BASE_URL = "https://openrouter.ai/api/v1"
    valves.HTTP_REFERER_OVERRIDE = ""

    class _FailingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def list_models(self):
            raise aiohttp.ClientError("simulated network failure")

    original_client = image_catalog.OpenRouterImageClient
    image_catalog.OpenRouterImageClient = _FailingClient  # type: ignore[misc]
    logger = MagicMock()
    try:
        await image_catalog.ensure_image_catalog_loaded(
            session=MagicMock(),
            valves=valves,
            api_key="test-key",
            logger=logger,
            cache_seconds=3600,
        )
    finally:
        image_catalog.OpenRouterImageClient = original_client  # type: ignore[misc]

    # Attempt clock bumped (so TTL gate engages next call)
    assert OpenRouterModelRegistry._last_image_attempt > 0.0
    # Fetch clock NOT bumped (no successful registration)
    assert OpenRouterModelRegistry._last_image_fetch == 0.0
    # Warning logged with the specific failure message (not just any warning)
    assert logger.warning.called
    warn_message = logger.warning.call_args[0][0]
    assert "Image catalog fetch failed" in warn_message


@pytest.mark.asyncio
async def test_image_catalog_empty_response_records_attempt_warns():
    """Empty fetch response: attempt bumped, fetch clock NOT bumped, warning logged."""
    from open_webui_openrouter_pipe.integrations import image_catalog

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry._last_image_attempt = 0.0
    OpenRouterModelRegistry._last_image_fetch = 0.0

    valves = MagicMock()
    valves.ENABLE_OPENROUTER_IMAGE_GENERATION = True
    valves.BASE_URL = "https://openrouter.ai/api/v1"
    valves.HTTP_REFERER_OVERRIDE = ""

    class _EmptyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def list_models(self):
            return []

    original_client = image_catalog.OpenRouterImageClient
    image_catalog.OpenRouterImageClient = _EmptyClient  # type: ignore[misc]
    logger = MagicMock()
    try:
        await image_catalog.ensure_image_catalog_loaded(
            session=MagicMock(),
            valves=valves,
            api_key="test-key",
            logger=logger,
            cache_seconds=3600,
        )
    finally:
        image_catalog.OpenRouterImageClient = original_client  # type: ignore[misc]

    assert OpenRouterModelRegistry._last_image_attempt > 0.0
    assert OpenRouterModelRegistry._last_image_fetch == 0.0
    assert logger.warning.called
    warn_message = logger.warning.call_args[0][0]
    assert "returned 0 models" in warn_message


# =============================================================================
# Auto-attach: end-to-end via the catalog_manager closure paths
# =============================================================================


@pytest.mark.parametrize("auto_attach,auto_default,supported,expect_filter_ids,expect_default_ids", [
    # Full 8-row truth table over (auto_attach × auto_default × supported)
    (True, True, True, ["openrouter_image_filter_generic"], ["openrouter_image_filter_generic"]),
    (True, False, True, ["openrouter_image_filter_generic"], []),
    (False, True, True, [], []),
    (False, False, True, [], []),
    (True, True, False, [], []),
    (True, False, False, [], []),
    (False, True, False, [], []),
    (False, False, False, [], []),
])
def test_apply_image_filter_ids_truth_table(
    auto_attach, auto_default, supported, expect_filter_ids, expect_default_ids
):
    """End-to-end truth table for the apply functions across all 5 critical
    combinations of (auto_attach, auto_default, supported)."""
    from open_webui_openrouter_pipe.models.catalog_manager import (
        _apply_list_filter_ids,
        _apply_list_default_filter_ids,
    )

    meta_dict: dict[str, Any] = {}
    _apply_list_filter_ids(
        meta_dict,
        filter_function_ids=["openrouter_image_filter_generic"],
        filter_supported=supported,
        auto_attach=auto_attach,
        prune_key="image_filter_ids",
    )
    _apply_list_default_filter_ids(
        meta_dict,
        filter_function_ids=["openrouter_image_filter_generic"],
        filter_supported=supported,
        auto_default=auto_default,
    )

    assert meta_dict.get("filterIds", []) == expect_filter_ids
    assert meta_dict.get("defaultFilterIds", []) == expect_default_ids


def test_apply_image_filter_ids_cleans_up_stale_previously_attached():
    """When `image_filter_ids` from pipe_meta contains an id NOT in the new
    `image_filter_function_ids`, the stale id must be removed from filterIds.
    Also verify pipe_meta["image_filter_ids"] is updated to track the new
    set so the NEXT cleanup cycle sees the right "previous" baseline."""
    from open_webui_openrouter_pipe.models.catalog_manager import _apply_list_filter_ids

    meta_dict: dict[str, Any] = {
        "filterIds": ["openrouter_image_filter_generic", "openrouter_image_filter_old"],
        "openrouter_pipe": {
            "image_filter_ids": ["openrouter_image_filter_old"],  # stale: old previously-attached
        },
    }
    _apply_list_filter_ids(
        meta_dict,
        filter_function_ids=["openrouter_image_filter_generic"],
        filter_supported=True,
        auto_attach=True,
        prune_key="image_filter_ids",
    )
    # Stale id removed, generic kept
    assert "openrouter_image_filter_old" not in meta_dict["filterIds"]
    assert "openrouter_image_filter_generic" in meta_dict["filterIds"]
    # pipe_meta updated to reflect new attached set (drives next cleanup cycle)
    assert meta_dict["openrouter_pipe"]["image_filter_ids"] == ["openrouter_image_filter_generic"]


def test_apply_image_filter_ids_idempotent_when_unchanged():
    """Running apply twice with same inputs returns False the second time
    (no change → no metadata mutation)."""
    from open_webui_openrouter_pipe.models.catalog_manager import _apply_list_filter_ids

    meta_dict: dict[str, Any] = {}
    first = _apply_list_filter_ids(
        meta_dict,
        filter_function_ids=["openrouter_image_filter_generic"],
        filter_supported=True,
        auto_attach=True,
        prune_key="image_filter_ids",
    )
    assert first is True

    second = _apply_list_filter_ids(
        meta_dict,
        filter_function_ids=["openrouter_image_filter_generic"],
        filter_supported=True,
        auto_attach=True,
        prune_key="image_filter_ids",
    )
    assert second is False  # No changes on second call


def test_apply_image_filter_ids_preserves_unrelated_filter_ids():
    """Apply must NOT touch unrelated filter ids in the model's filterIds list
    (e.g. user-attached community filters, OWUI built-in filters)."""
    from open_webui_openrouter_pipe.models.catalog_manager import _apply_list_filter_ids

    meta_dict: dict[str, Any] = {
        "filterIds": ["my_custom_filter", "owui_translate_filter"],
    }
    _apply_list_filter_ids(
        meta_dict,
        filter_function_ids=["openrouter_image_filter_generic"],
        filter_supported=True,
        auto_attach=True,
        prune_key="image_filter_ids",
    )
    assert "my_custom_filter" in meta_dict["filterIds"]
    assert "owui_translate_filter" in meta_dict["filterIds"]
    assert "openrouter_image_filter_generic" in meta_dict["filterIds"]


def test_apply_image_filter_ids_two_filter_dict_order_preserved():
    """When applying [generic, sourceful], both ids land in filterIds in order."""
    from open_webui_openrouter_pipe.models.catalog_manager import _apply_list_filter_ids

    meta_dict: dict[str, Any] = {}
    _apply_list_filter_ids(
        meta_dict,
        filter_function_ids=["openrouter_image_filter_generic", "openrouter_image_filter_sourceful"],
        filter_supported=True,
        auto_attach=True,
        prune_key="image_filter_ids",
    )
    assert meta_dict["filterIds"] == [
        "openrouter_image_filter_generic",
        "openrouter_image_filter_sourceful",
    ]


def test_apply_image_filter_ids_dedupes_input():
    """If `image_filter_function_ids` contains duplicates, the resulting filterIds
    must NOT contain duplicates (dedupe via _dedupe_preserve_order)."""
    from open_webui_openrouter_pipe.models.catalog_manager import _apply_list_filter_ids

    meta_dict: dict[str, Any] = {}
    _apply_list_filter_ids(
        meta_dict,
        filter_function_ids=[
            "openrouter_image_filter_generic",
            "openrouter_image_filter_sourceful",
            "openrouter_image_filter_generic",  # duplicate
        ],
        filter_supported=True,
        auto_attach=True,
        prune_key="image_filter_ids",
    )
    assert meta_dict["filterIds"] == [
        "openrouter_image_filter_generic",
        "openrouter_image_filter_sourceful",
    ]


def test_apply_image_default_filter_ids_skips_when_filter_id_not_in_filterIds():
    """default-on must require the id to actually be in filterIds first
    (don't leave a model in 'default on' state for a filter that isn't attached)."""
    from open_webui_openrouter_pipe.models.catalog_manager import _apply_list_default_filter_ids

    meta_dict: dict[str, Any] = {"filterIds": []}  # empty — generic NOT attached
    result = _apply_list_default_filter_ids(
        meta_dict,
        filter_function_ids=["openrouter_image_filter_generic"],
        filter_supported=True,
        auto_default=True,
    )
    assert result is False
    assert "defaultFilterIds" not in meta_dict


def test_capability_gated_web_search_overlay_skips_image_models():
    """Image-output models must NOT have web_search overlaid to True (regression
    defense — pre-fix, all non-video models defaulted to web_search=True)."""
    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)

    sourceful_norm = ModelFamily.base_model(sanitize_model_id("sourceful/riverflow-v2-pro"))
    spec = OpenRouterModelRegistry._specs[sourceful_norm]
    capabilities = spec.get("capabilities") or {}
    assert capabilities.get("web_search") is False, (
        "Image-output models must default web_search=False"
    )


def test_inject_image_modalities_pure_image_model():
    """`_inject_image_modalities` writes ['image'] for pure-image-only models."""
    from open_webui_openrouter_pipe.requests.orchestrator import _inject_image_modalities

    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)

    body: dict[str, Any] = {"model": "sourceful/riverflow-v2-pro", "messages": []}
    _inject_image_modalities(body)
    assert body["modalities"] == ["image"]


def test_inject_image_modalities_multimodal_model():
    """`_inject_image_modalities` writes ['image', 'text'] for multimodal output models."""
    from open_webui_openrouter_pipe.requests.orchestrator import _inject_image_modalities

    OpenRouterModelRegistry._specs = {
        "openai.gpt-5-image": {
            "features": frozenset({"image_gen_tool", "image_output"}),
            "capabilities": {},
            "max_completion_tokens": None,
            "supported_parameters": frozenset(),
            "full_model": {},
            "architecture": {"output_modalities": ["image", "text"]},
        }
    }
    OpenRouterModelRegistry._id_map = {"openai.gpt-5-image": "openai/gpt-5-image"}
    ModelFamily.set_dynamic_specs(OpenRouterModelRegistry._specs)

    body: dict[str, Any] = {"model": "openai/gpt-5-image", "messages": []}
    _inject_image_modalities(body)
    assert body["modalities"] == ["image", "text"]


def test_inject_image_modalities_text_only_model_untouched():
    """Text-only chat models don't get modalities injected."""
    from open_webui_openrouter_pipe.requests.orchestrator import _inject_image_modalities

    OpenRouterModelRegistry._specs = {
        "anthropic.claude-3-opus": {
            "features": frozenset(),
            "capabilities": {},
            "max_completion_tokens": None,
            "supported_parameters": frozenset(),
            "full_model": {},
            "architecture": {"output_modalities": ["text"]},
        }
    }
    OpenRouterModelRegistry._id_map = {"anthropic.claude-3-opus": "anthropic/claude-3-opus"}
    ModelFamily.set_dynamic_specs(OpenRouterModelRegistry._specs)

    body: dict[str, Any] = {"model": "anthropic/claude-3-opus", "messages": []}
    _inject_image_modalities(body)
    assert "modalities" not in body


def test_inject_image_modalities_explicit_user_modalities_preserved():
    """If user/filter already set modalities, helper does not overwrite."""
    from open_webui_openrouter_pipe.requests.orchestrator import _inject_image_modalities

    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)

    body: dict[str, Any] = {
        "model": "sourceful/riverflow-v2-pro",
        "messages": [],
        "modalities": ["image", "text"],  # explicit override
    }
    _inject_image_modalities(body)
    assert body["modalities"] == ["image", "text"]  # not overwritten


def test_inject_image_modalities_unknown_model_untouched():
    """Unregistered model id is a no-op (defensive — no spec lookup result)."""
    from open_webui_openrouter_pipe.requests.orchestrator import _inject_image_modalities

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    body: dict[str, Any] = {"model": "made-up/model", "messages": []}
    _inject_image_modalities(body)
    assert "modalities" not in body


def test_inject_image_modalities_non_dict_body_no_crash():
    """Helper returns gracefully when body isn't a dict (defensive)."""
    from open_webui_openrouter_pipe.requests.orchestrator import _inject_image_modalities

    # str body: must NOT mutate (assert by reference equality)
    body_str = "not-a-dict"
    body_str_before = body_str
    _inject_image_modalities(body_str)  # type: ignore[arg-type]
    assert body_str == body_str_before
    # None body: no crash, no return value
    assert _inject_image_modalities(None) is None  # type: ignore[arg-type]


def test_inject_image_modalities_multimodal_via_chat_catalog_path():
    """End-to-end: the chat-catalog `_derive_features` path adds `image_output`
    to multimodal models with `image` in output_modalities. Verify a model
    registered through that path (not via register_image_models) still gets
    modalities injected correctly when sent through the orchestrator helper."""
    from open_webui_openrouter_pipe.requests.orchestrator import _inject_image_modalities

    # Simulate what `_register_models` (chat catalog) produces for a multimodal model
    derived_features = OpenRouterModelRegistry._derive_features(  # type: ignore[attr-defined]
        supported_parameters={"temperature", "tools"},
        architecture={
            "input_modalities": ["text", "image"],
            "output_modalities": ["image", "text"],
        },
        pricing={},
    )
    # Confirm the chat path emits image_output
    assert "image_output" in derived_features
    assert "image_gen_tool" in derived_features

    # Plant a chat-catalog-style spec
    OpenRouterModelRegistry._specs["openai.gpt-5-image"] = {
        "features": derived_features,
        "capabilities": {"image_generation": True},
        "max_completion_tokens": None,
        "supported_parameters": frozenset({"temperature", "tools"}),
        "full_model": {},
        "architecture": {
            "input_modalities": ["text", "image"],
            "output_modalities": ["image", "text"],
        },
    }
    OpenRouterModelRegistry._id_map["openai.gpt-5-image"] = "openai/gpt-5-image"
    ModelFamily.set_dynamic_specs(OpenRouterModelRegistry._specs)

    body: dict[str, Any] = {"model": "openai/gpt-5-image", "messages": []}
    _inject_image_modalities(body)
    assert body["modalities"] == ["image", "text"]


# =============================================================================
# Recraft filters — common (V3/V4/V4 Pro) and V3-only extras
# =============================================================================


def test_recraft_common_filter_inlet_writes_strength_into_body():
    source = render_recraft_common_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_strength")
    user_valves = module.Filter.UserValves(IMAGE_STRENGTH=0.7)
    body: dict[str, Any] = {"model": "recraft/recraft-v4-pro", "messages": []}
    module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})
    assert body["image_config"] == {"strength": 0.7}


def test_recraft_common_filter_skips_strength_when_zero():
    """0.0 is the skip sentinel — should NOT write strength to body."""
    source = render_recraft_common_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_strength_zero")
    user_valves = module.Filter.UserValves(IMAGE_STRENGTH=0.0)
    body: dict[str, Any] = {"model": "recraft/recraft-v4", "messages": []}
    module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})
    assert "image_config" not in body  # nothing written


def test_recraft_common_filter_writes_rgb_colors():
    source = render_recraft_common_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_rgb")
    user_valves = module.Filter.UserValves(
        IMAGE_RGB_COLORS_JSON="[[255, 0, 0], [0, 128, 0]]"
    )
    body: dict[str, Any] = {"model": "recraft/recraft-v3", "messages": []}
    module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})
    assert body["image_config"]["rgb_colors"] == [[255, 0, 0], [0, 128, 0]]


def test_recraft_common_filter_writes_background_rgb():
    source = render_recraft_common_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_bg")
    user_valves = module.Filter.UserValves(
        IMAGE_BACKGROUND_RGB_JSON="[0, 0, 255]"
    )
    body: dict[str, Any] = {"model": "recraft/recraft-v4", "messages": []}
    module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})
    assert body["image_config"]["background_rgb_color"] == [0, 0, 255]


def test_recraft_common_filter_rejects_oversaturated_rgb():
    source = render_recraft_common_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_rgb_oversat")
    user_valves = module.Filter.UserValves(
        IMAGE_RGB_COLORS_JSON="[[300, 0, 0]]"
    )
    body: dict[str, Any] = {"model": "recraft/recraft-v3", "messages": []}
    with pytest.raises(module.ImageGenerationError, match="0-255"):
        module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})


def test_recraft_common_filter_rejects_wrong_rgb_arity():
    source = render_recraft_common_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_rgb_arity")
    user_valves = module.Filter.UserValves(
        IMAGE_RGB_COLORS_JSON="[[255, 0]]"  # only 2 components
    )
    body: dict[str, Any] = {"model": "recraft/recraft-v3", "messages": []}
    with pytest.raises(module.ImageGenerationError, match="3-element"):
        module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})


def test_recraft_common_filter_rejects_malformed_json():
    source = render_recraft_common_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_bad_json")
    user_valves = module.Filter.UserValves(
        IMAGE_RGB_COLORS_JSON="not valid json"
    )
    body: dict[str, Any] = {"model": "recraft/recraft-v3", "messages": []}
    with pytest.raises(module.ImageGenerationError, match="not valid JSON"):
        module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})


def test_recraft_common_filter_skips_non_recraft_model():
    """Defensive gate: filter must no-op on non-Recraft models even if attached."""
    source = render_recraft_common_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_gate")
    user_valves = module.Filter.UserValves(IMAGE_STRENGTH=0.7)
    body: dict[str, Any] = {"model": "openai/gpt-5-image", "messages": []}
    module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})
    assert "image_config" not in body  # filter no-op on non-Recraft


def test_recraft_common_filter_combines_all_three_params():
    """All three Recraft Common params combined into a single image_config."""
    source = render_recraft_common_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_combined")
    user_valves = module.Filter.UserValves(
        IMAGE_STRENGTH=0.6,
        IMAGE_RGB_COLORS_JSON="[[255, 0, 0]]",
        IMAGE_BACKGROUND_RGB_JSON="[255, 255, 255]",
    )
    body: dict[str, Any] = {"model": "recraft/recraft-v4-pro", "messages": []}
    module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})
    assert body["image_config"] == {
        "strength": 0.6,
        "rgb_colors": [[255, 0, 0]],
        "background_rgb_color": [255, 255, 255],
    }


# Recraft V3 Extras filter


def test_recraft_v3_filter_inlet_writes_style():
    source = render_recraft_v3_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_v3_style")
    user_valves = module.Filter.UserValves(IMAGE_RECRAFT_STYLE="Photorealism")
    body: dict[str, Any] = {"model": "recraft/recraft-v3", "messages": []}
    module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})
    assert body["image_config"] == {"style": "Photorealism"}


def test_recraft_v3_filter_inlet_writes_text_layout():
    source = render_recraft_v3_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_v3_layout")
    layout = (
        '[{"text":"Hello","bbox":[[0.3,0.45],[0.6,0.45],[0.6,0.55],[0.3,0.55]]}]'
    )
    user_valves = module.Filter.UserValves(IMAGE_TEXT_LAYOUT_JSON=layout)
    body: dict[str, Any] = {"model": "recraft/recraft-v3", "messages": []}
    module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})
    assert body["image_config"]["text_layout"][0]["text"] == "Hello"
    assert len(body["image_config"]["text_layout"][0]["bbox"]) == 4


def test_recraft_v3_filter_skips_v4_silently():
    """Per OpenRouter docs: V4 and V4 Pro do NOT support style or text_layout.
    The filter must no-op on V4/V4 Pro even if manually attached."""
    source = render_recraft_v3_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_v3_skip_v4")
    user_valves = module.Filter.UserValves(IMAGE_RECRAFT_STYLE="Photorealism")
    body: dict[str, Any] = {"model": "recraft/recraft-v4", "messages": []}
    module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})
    assert "image_config" not in body  # V4 skipped


def test_recraft_v3_filter_skips_v4_pro_silently():
    source = render_recraft_v3_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_v3_skip_v4pro")
    user_valves = module.Filter.UserValves(IMAGE_RECRAFT_STYLE="Photorealism")
    body: dict[str, Any] = {"model": "recraft/recraft-v4-pro", "messages": []}
    module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})
    assert "image_config" not in body  # V4 Pro skipped too


def test_recraft_v3_filter_rejects_bbox_out_of_range():
    """bbox coords must be 0.0-1.0; 1.5 must raise."""
    source = render_recraft_v3_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_v3_bbox_range")
    layout = (
        '[{"text":"Hello","bbox":[[1.5,0.45],[0.6,0.45],[0.6,0.55],[0.3,0.55]]}]'
    )
    user_valves = module.Filter.UserValves(IMAGE_TEXT_LAYOUT_JSON=layout)
    body: dict[str, Any] = {"model": "recraft/recraft-v3", "messages": []}
    with pytest.raises(module.ImageGenerationError, match="0.0-1.0"):
        module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})


def test_recraft_v3_filter_rejects_wrong_bbox_arity():
    """bbox must have exactly 4 corner points."""
    source = render_recraft_v3_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_v3_bbox_arity")
    layout = (
        '[{"text":"Hello","bbox":[[0.3,0.45],[0.6,0.45],[0.6,0.55]]}]'  # only 3 corners
    )
    user_valves = module.Filter.UserValves(IMAGE_TEXT_LAYOUT_JSON=layout)
    body: dict[str, Any] = {"model": "recraft/recraft-v3", "messages": []}
    with pytest.raises(module.ImageGenerationError, match="4-element"):
        module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})


def test_recraft_v3_filter_rejects_missing_text():
    """text_layout entry without 'text' key must raise."""
    source = render_recraft_v3_image_filter_source()
    module = _load_filter_from_source(source, "test_image_filter_recraft_v3_no_text")
    layout = '[{"bbox":[[0.3,0.45],[0.6,0.45],[0.6,0.55],[0.3,0.55]]}]'
    user_valves = module.Filter.UserValves(IMAGE_TEXT_LAYOUT_JSON=layout)
    body: dict[str, Any] = {"model": "recraft/recraft-v3", "messages": []}
    with pytest.raises(module.ImageGenerationError, match="non-empty string"):
        module.Filter().inlet(body, __metadata__={}, __user__={"valves": user_valves})


# Pattern matching


def test_recraft_common_pattern_matches_prefix():
    """Recraft common pattern must match all recraft/recraft-* but not other prefixes."""
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager
    pat = FilterManager._RECRAFT_COMMON_IMAGE_PATTERN
    assert pat.match("recraft/recraft-v3")
    assert pat.match("recraft/recraft-v4")
    assert pat.match("recraft/recraft-v4-pro")
    assert pat.match("recraft/recraft-v5-future")
    # Wrong prefix
    assert not pat.match("evil/recraft-v3")
    assert not pat.match("recraft/other-model")


def test_recraft_v3_pattern_uses_fullmatch_anchors():
    """Recraft V3 pattern must match V3 EXACTLY (not V3-something or v30)."""
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager
    pat = FilterManager._RECRAFT_V3_IMAGE_PATTERN
    assert pat.fullmatch("recraft/recraft-v3")
    # Suffix attacks
    assert not pat.fullmatch("recraft/recraft-v30")
    assert not pat.fullmatch("recraft/recraft-v3-extra")
    assert not pat.fullmatch("recraft/recraft-v4")
    assert not pat.fullmatch("recraft/recraft-v4-pro")


# Installer auto-attach truth table for Recraft variants


@pytest.mark.asyncio
async def test_installer_attaches_recraft_v3_with_v3_extras():
    """V3 model should get: generic + recraft + recraft_v3."""
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry.register_image_models([
        m for m in IMAGE_MODELS if m["id"] == "recraft/recraft-v3"
    ])

    pipe = MagicMock()
    fm = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    fm._ensure_filter_installed = AsyncMock(side_effect=lambda **kwargs: kwargs["preferred_id"])

    v3 = OpenRouterModelRegistry.list_models()[0]
    result = await fm.ensure_openrouter_image_filter_function_ids([v3])

    ids = result[v3["id"]]
    assert "openrouter_image_filter_generic" in ids
    assert "openrouter_image_filter_recraft" in ids
    assert "openrouter_image_filter_recraft_v3" in ids
    assert "openrouter_image_filter_gemini" not in ids
    assert "openrouter_image_filter_sourceful" not in ids


@pytest.mark.asyncio
async def test_installer_attaches_recraft_v4_without_v3_extras():
    """V4 model should get: generic + recraft (NO recraft_v3)."""
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry.register_image_models([
        m for m in IMAGE_MODELS if m["id"] == "recraft/recraft-v4"
    ])

    pipe = MagicMock()
    fm = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    fm._ensure_filter_installed = AsyncMock(side_effect=lambda **kwargs: kwargs["preferred_id"])

    v4 = OpenRouterModelRegistry.list_models()[0]
    result = await fm.ensure_openrouter_image_filter_function_ids([v4])

    ids = result[v4["id"]]
    assert "openrouter_image_filter_generic" in ids
    assert "openrouter_image_filter_recraft" in ids
    assert "openrouter_image_filter_recraft_v3" not in ids


@pytest.mark.asyncio
async def test_installer_attaches_recraft_v4_pro_without_v3_extras():
    """V4 Pro model should get: generic + recraft (NO recraft_v3) — same as V4."""
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry.register_image_models([
        m for m in IMAGE_MODELS if m["id"] == "recraft/recraft-v4-pro"
    ])

    pipe = MagicMock()
    fm = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    fm._ensure_filter_installed = AsyncMock(side_effect=lambda **kwargs: kwargs["preferred_id"])

    v4pro = OpenRouterModelRegistry.list_models()[0]
    result = await fm.ensure_openrouter_image_filter_function_ids([v4pro])

    ids = result[v4pro["id"]]
    assert "openrouter_image_filter_generic" in ids
    assert "openrouter_image_filter_recraft" in ids
    assert "openrouter_image_filter_recraft_v3" not in ids


# Help coverage


def test_help_covers_all_three_recraft_models():
    """All 3 Recraft entries must have curated help with knob descriptions."""
    for model_id in ("recraft/recraft-v3", "recraft/recraft-v4", "recraft/recraft-v4-pro"):
        assert model_id in IMAGE_HELP_BY_MODEL, f"Missing help entry for {model_id}"
        entry = IMAGE_HELP_BY_MODEL[model_id]
        assert entry["display_name"].startswith("Recraft:")
        assert entry["best_known_for"]
        assert entry["tips_and_pitfalls"]
        assert entry["knob_descriptions"]


def test_help_renders_v3_only_knobs_only_for_v3():
    """The `Recraft style` and `Text layout` knob descriptions must appear ONLY
    in the rendered V3 help, not V4 or V4 Pro."""
    v3_help = render_image_help("recraft/recraft-v3")
    v4_help = render_image_help("recraft/recraft-v4")
    v4pro_help = render_image_help("recraft/recraft-v4-pro")

    assert "Recraft style" in v3_help
    assert "Text layout" in v3_help
    assert "Recraft style" not in v4_help
    assert "Text layout" not in v4_help
    assert "Recraft style" not in v4pro_help
    assert "Text layout" not in v4pro_help


def test_help_renders_recraft_common_knobs_for_all_three():
    """`Strength`, `RGB color palette`, and `Background RGB color` must appear
    in all three Recraft model help blurbs."""
    for model_id in ("recraft/recraft-v3", "recraft/recraft-v4", "recraft/recraft-v4-pro"):
        rendered = render_image_help(model_id)
        assert "Strength (image-to-image)" in rendered, f"missing in {model_id}"
        assert "RGB color palette" in rendered, f"missing in {model_id}"
        assert "Background RGB color" in rendered, f"missing in {model_id}"
