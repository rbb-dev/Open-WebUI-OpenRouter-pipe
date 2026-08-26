"""Tests for OpenRouter native image generation feature.

Coverage:
- Registry: register_image_models dedupe, stale-norm cleanup, atomic rebuild
- Filter renderer: one filter per model, offering only that model's published knobs
- Filter manager: per-model install, driven by the endpoint contract on the spec
- Pydantic image_config field: typed dict round-trip
- Auto-attach: pipe_capabilities image_output, defaultFilterIds writes
- image_help.py: per-model entries, KNOB_GATE consistency
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, get_args
from unittest.mock import AsyncMock, MagicMock

import logging

import pytest

from open_webui_openrouter_pipe.api.transforms import CompletionsBody, ResponsesBody
from open_webui_openrouter_pipe.filters.image_filter_renderer import (
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
    (Path(__file__).parent / "fixtures" / "openrouter_image_models.json").read_text()
)

_SERVER_TOOL_RECORDING = json.loads(
    (Path(__file__).parent / "fixtures" / "openrouter_request_schema_fields.json").read_text()
)["image_server_tool"]

_RECORDED_SERVER_TOOL_PARAMS = tuple(_SERVER_TOOL_RECORDING["parameters"])
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
    rendered = render_image_help("sourceful/riverflow-v2-pro", IMAGE_BY_ID["sourceful/riverflow-v2-pro"], dedicated_image_api=True)
    assert "Sourceful: Riverflow V2 Pro" in rendered
    assert "tips" in rendered.lower() or "Tips" in rendered


def test_render_image_help_falls_back_to_catalog_for_unknown_model():
    rendered = render_image_help("unknown/model", {"name": "Unknown", "description": "desc", "architecture": {}}, dedicated_image_api=True)
    assert "Unknown" in rendered
    assert "No curated help" in rendered


















# =============================================================================
# image_filter_renderer: filter inlet writes correct image_config (sourcecode check)
# =============================================================================








# =============================================================================
# Integration: full fixture round-trip through register_image_models
# =============================================================================




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




def test_register_image_models_full_fixture_exact_pure_image_count():
    """Registers exactly the image-only models the catalogue holds, and no others.

    The expected set is derived from the fixture rather than counted by hand, because a
    hand-written number goes stale the moment OpenRouter adds a model and says nothing
    about *which* models were registered.
    """
    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []

    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)

    image_only_specs = [
        norm_id for norm_id, spec in OpenRouterModelRegistry._specs.items()
        if "image_output" in (spec.get("features") or set())
    ]
    expected = {
        sanitize_model_id(m["id"])
        for m in IMAGE_MODELS
        if "text" not in ((m.get("architecture") or {}).get("output_modalities") or [])
    }
    assert set(image_only_specs) == expected, (
        f"registered but not image-only: {sorted(set(image_only_specs) - expected)}\n"
        f"image-only but not registered: {sorted(expected - set(image_only_specs))}"
    )


# =============================================================================
# Filter inlet RUNTIME behaviour — exec the rendered filter and run inlet()
# =============================================================================


































# =============================================================================
# ensure_openrouter_image_filter_function_ids — installer behaviour
# =============================================================================












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


def _image_only_catalog_entry(model_id: str, display_name: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "name": display_name,
        "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
        "pricing": {},
    }


@pytest.mark.parametrize(
    "catalog",
    [
        pytest.param([_image_only_catalog_entry("acme/paintbox-1", "Paintbox 1")], id="paintbox"),
        pytest.param([_image_only_catalog_entry("zeta/inkwell-9", "Inkwell 9")], id="inkwell"),
    ],
)
@pytest.mark.asyncio
async def test_master_disable_takes_image_models_out_of_the_model_list(
    monkeypatch, pipe_instance_async, catalog
):
    """Turning ENABLE_OPENROUTER_IMAGE_GENERATION off empties the picker on the next pipes().

    Driven through `pipes()`, which is what Open WebUI calls, not through the loader.
    Both call sites used to guard the loader with the same valve, so the branch that
    clears the models could never be reached and an administrator who switched image
    generation off kept seeing image-only models until an unrelated TTL expired. A test
    that calls `ensure_image_catalog_loaded` directly exercises the branch and passes
    whatever the wiring does.
    """
    from open_webui_openrouter_pipe.integrations.image_client import OpenRouterImageClient

    pipe = pipe_instance_async
    monkeypatch.setattr(pipe, "_maybe_start_startup_checks", lambda: None)
    monkeypatch.setattr(pipe, "_maybe_start_redis", lambda: None)
    monkeypatch.setattr(pipe, "_maybe_start_cleanup", lambda: None)
    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: ("sk-test", None))

    async def _no_chat_catalog(*_args, **_kwargs):
        return None

    monkeypatch.setattr(OpenRouterModelRegistry, "ensure_loaded", _no_chat_catalog)

    async def _list_models(_self):
        return [dict(entry) for entry in catalog]

    monkeypatch.setattr(OpenRouterImageClient, "list_models", _list_models)

    pipe.valves.ENABLE_OPENROUTER_IMAGE_GENERATION = True
    pipe.valves.ENABLE_VIDEO_GENERATION = False
    pipe.valves.AUTO_INSTALL_IMAGE_FILTERS = False
    pipe.valves.AUTO_ATTACH_IMAGE_FILTERS = False

    offered = {entry["name"] for entry in await pipe.pipes()}
    wanted = {entry["name"] for entry in catalog}
    assert wanted <= offered, (
        f"{sorted(wanted)} never reached the model list, so the disable arm below would "
        "pass for the wrong reason"
    )

    pipe.valves.ENABLE_OPENROUTER_IMAGE_GENERATION = False
    still_offered = {entry["name"] for entry in await pipe.pipes()}

    assert not (wanted & still_offered), (
        f"{sorted(wanted & still_offered)} stayed selectable after "
        "ENABLE_OPENROUTER_IMAGE_GENERATION was switched off"
    )


@pytest.mark.asyncio
async def test_image_catalog_ttl_gate_skips_within_window():
    """Within `cache_seconds` of last attempt, no new fetch is initiated."""
    import time

    from open_webui_openrouter_pipe.integrations.image_catalog import ensure_image_catalog_loaded

    now = time.time()
    OpenRouterModelRegistry._last_image_attempt = now  # just now
    OpenRouterModelRegistry._last_image_contract_attempt = now
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

    assert not session.get.called


@pytest.mark.asyncio
async def test_a_request_that_skipped_contracts_does_not_suppress_the_next_sweep():
    """The two phases have separate clocks, because they do separate work.

    A chat message refreshes the model list without reading contracts. Sharing one clock
    meant that message suppressed the next catalogue refresh's sweep for a whole window,
    and every image model lost its controls until the window expired.
    """
    from open_webui_openrouter_pipe.integrations import image_catalog

    record = {
        "provider_slug": "p",
        "supported_parameters": {"aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]}},
    }
    catalog = json.loads(
        (Path(__file__).parent / "fixtures" / "openrouter_image_models.json").read_text()
    )["data"][:2]
    awaited: list[str] = []

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def list_models(self):
            return catalog

        async def endpoints(self, model_id):
            awaited.append(model_id)
            return [record]

    valves = MagicMock()
    valves.ENABLE_OPENROUTER_IMAGE_GENERATION = True
    valves.AUTO_INSTALL_IMAGE_FILTERS = True
    valves.AUTO_ATTACH_IMAGE_FILTERS = True
    valves.BASE_URL = "https://openrouter.ai/api/v1"
    valves.HTTP_REFERER_OVERRIDE = ""

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry._image_endpoints = {}
    OpenRouterModelRegistry._last_image_attempt = 0.0
    OpenRouterModelRegistry._last_image_fetch = 0.0
    OpenRouterModelRegistry._last_image_contract_attempt = 0.0

    original = image_catalog.OpenRouterImageClient
    image_catalog.OpenRouterImageClient = _Client  # type: ignore[misc]
    try:
        await image_catalog.ensure_image_catalog_loaded(
            session=MagicMock(), valves=valves, api_key="k",
            logger=logging.getLogger("t"), cache_seconds=3600, with_contracts=False,
        )
        assert awaited == [], "the request path must not read contracts"

        await image_catalog.ensure_image_catalog_loaded(
            session=MagicMock(), valves=valves, api_key="k",
            logger=logging.getLogger("t"), cache_seconds=3600,
        )
    finally:
        image_catalog.OpenRouterImageClient = original  # type: ignore[misc]

    assert len(awaited) == len(catalog), (
        f"the catalogue refresh must still read every contract; got {awaited}"
    )
    assert OpenRouterModelRegistry.image_endpoint(catalog[0]["id"]) == [record]


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
        def get(self, url, headers=None, timeout=None):
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
        def get(self, url, headers=None, timeout=None):
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
    from open_webui_openrouter_pipe.integrations import image_catalog

    assert logger.log.called, (
        "the catalog failure was not reported at all. It is now emitted through "
        "warn_level so a persistent outage does not warn once per /api/models request, "
        "but the FIRST occurrence must still be a warning."
    )
    assert image_catalog._warned_image_catalog, (
        "the latch was never armed, so warn_level never ran and the failure was not "
        "reported through the shared decision"
    )
    level, warn_message = logger.log.call_args[0][0], logger.log.call_args[0][1]
    assert level == logging.WARNING, (
        f"the FIRST catalog failure was reported at level {level}, not WARNING; the "
        "latch is meant to quiet repeats, not the first occurrence"
    )
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




















# Recraft V3 Extras filter
















# Pattern matching






# Installer auto-attach truth table for Recraft variants








# Help coverage








def _recorded_endpoint(name: str) -> list[dict]:
    """Every record the model published, as the live probe recorded them.

    This returned `records[0]`. A model served by more than one provider publishes one
    record each and they disagree, so taking the first silently replaced the set the
    providers agree on with whatever the sweep happened to write down first -- the exact
    narrowing the checked-in contracts exist to exercise.
    """
    import json

    raw = json.loads(
        (Path(__file__).parent / "fixtures" / f"openrouter_image_endpoints_{name}.json").read_text()
    )
    records = raw.get("endpoints") or [raw]
    assert records, name
    return [record for record in records if isinstance(record, dict)]


def _recorded_contract_slugs() -> dict[str, str]:
    """Every checked-in contract, mapped fixture slug -> the model id it recorded."""
    import json

    found: dict[str, str] = {}
    for path in sorted((Path(__file__).parent / "fixtures").glob("openrouter_image_endpoints_*.json")):
        slug = path.stem[len("openrouter_image_endpoints_") :]
        found[slug] = json.loads(path.read_text())["id"]
    return found


def _multi_provider_contracts() -> dict[str, str]:
    return {
        slug: model_id
        for slug, model_id in _recorded_contract_slugs().items()
        if len(_recorded_endpoint(slug)) > 1
    }


EVERY_CONTRACT = sorted(_recorded_contract_slugs().items())
"""Every checked-in contract as a (slug, model id) pair, for the fleet-wide sweeps.

Two models used to stand for forty. A claim about what every model is offered cannot be
measured on a pair, and the pair that was chosen published a shape the rest do not: both
led with a ratio list, so nothing noticed that three models lead with `quality` instead.
`test_every_catalogued_image_model_has_a_checked_in_contract` asserts this set is exactly
the catalogue, in both directions, so a sweep parametrised over it cannot silently shrink.
"""



@pytest.mark.parametrize(
    ("fixture", "model_id", "expected_ratios", "expected_passthrough"),
    [
        (
            "recraft_recraft-v3",
            "recraft/recraft-v3",
            ("1:1", "4:3", "3:4", "16:9", "9:16", "auto"),
            ("style", "controls", "text_layout"),
        ),
        (
            "qwen_qwen-image-3",
            "qwen/qwen-image-3",
            (
                "1:1", "1:2", "1:4", "2:1", "2:3", "3:2", "3:4",
                "4:1", "4:3", "4:5", "5:4", "9:16", "16:9",
            ),
            (),
        ),
    ],
)
def test_a_model_is_offered_the_ratios_its_own_contract_publishes(
    fixture, model_id, expected_ratios, expected_passthrough
):
    """The fixed variants handed every model the same ten ratios.

    Thirty-three of forty rejected at least one of them, and twenty-eight could not reach
    a ratio they do support -- `auto` among them, which twenty-six publish and none were
    offered. Reading the model's own record is what closes both gaps at once.

    Two models, deliberately, where the sweeps beside this one run the whole set. These
    expectations were read off the recorded contracts by eye and typed out, and that
    second, independent statement of the values is the entire assertion. Computing them
    from the same record the spec is built from would compare the builder with itself and
    pass on any misreading it makes consistently, which is the failure being guarded.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )

    spec = build_image_model_filter_spec(
        model_id, {"id": model_id, "name": model_id}, _recorded_endpoint(fixture),
        dedicated_image_api=True,
    )

    assert dict(spec.enums).get("aspect_ratio") == expected_ratios, (
        f"got {dict(spec.enums).get('aspect_ratio')!r}"
    )
    assert spec.passthrough == expected_passthrough, (
        "provider knobs come from allowed_passthrough_parameters, not from a regex on the "
        f"model id. got {spec.passthrough!r}"
    )


def test_an_unreadable_contract_offers_no_knobs_rather_than_inventing_them():
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )

    spec = build_image_model_filter_spec("vendor/unknown", {"id": "vendor/unknown"}, None, dedicated_image_api=True)

    assert not spec.has_knobs, (
        "with no published contract there is nothing to offer; guessing a knob set is how "
        f"a third of models ended up rejecting values the filter presented. got {spec!r}"
    )


@pytest.mark.parametrize("failing", [0, 1, 3])
@pytest.mark.asyncio
async def test_a_model_whose_contract_cannot_be_read_is_simply_absent(failing, caplog):
    """One unreachable contract must not cost the whole refresh.

    Every model that did answer keeps its knobs; the ones that did not are absent, so
    their filters render nothing this cycle and the next refresh retries.
    """
    import logging as _logging

    from open_webui_openrouter_pipe.integrations.image_catalog import _fetch_endpoint_records

    models = [{"id": f"vendor/model-{index}"} for index in range(3)]
    broken = {f"vendor/model-{index}" for index in range(failing)}

    class _Client:
        async def endpoints(self, model_id: str):
            if model_id in broken:
                raise TimeoutError("upstream did not answer")
            return [{"provider_slug": "p", "supported_parameters": {"n": {"type": "range", "min": 1, "max": 4}}}]

    with caplog.at_level(_logging.DEBUG):
        records = await _fetch_endpoint_records(
            _Client(), models, _logging.getLogger("test.image.catalog")
        )

    assert set(records) == {m["id"] for m in models} - broken, (
        f"{failing} unreachable contract(s) should leave {3 - failing} usable. got {sorted(records)}"
    )
    if failing:
        assert "could not read the published knob contract" in caplog.text.lower(), (
            "an operator whose filters lost their knobs needs to know why"
        )


@pytest.mark.parametrize(
    ("fixture", "model_id", "expected_valves"),
    [
        (
            "recraft_recraft-v3",
            "recraft/recraft-v3",
            ["IMAGE_ASPECT_RATIO", "IMAGE_CONTROLS", "IMAGE_N", "IMAGE_STYLE", "IMAGE_TEXT_LAYOUT"],
        ),
        (
            "qwen_qwen-image-3",
            "qwen/qwen-image-3",
            ["IMAGE_ASPECT_RATIO", "IMAGE_N", "IMAGE_RESOLUTION", "IMAGE_SEED"],
        ),
    ],
)
def test_a_generated_filter_loads_and_offers_only_the_published_knobs(
    fixture, model_id, expected_valves
):
    """Loaded the way Open WebUI loads one, then driven.

    Asserting on the rendered source would pass on a comment; this executes the filter and
    reads the fields it really exposes, then puts a value through inlet.

    Two models, deliberately. The field list is typed out here as a separate reading of
    the contracts; deriving it from the spec would leave the rendered module compared only
    against the object that produced it, so a knob wrongly built and wrongly rendered would
    agree with itself. The sweeps that need no literal run the whole set instead.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )

    spec = build_image_model_filter_spec(
        model_id, {"id": model_id, "name": model_id}, _recorded_endpoint(fixture),
        dedicated_image_api=True,
    )
    source = render_image_model_filter_source(spec)

    module_name = f"generated_image_filter_{fixture}"
    try:
        module = _load_filter_from_source(source, module_name)

        from open_webui_openrouter_pipe.filters.image_filter_renderer import (
            ALWAYS_ON_VALVE_NAMES,
        )

        published = sorted(
            set(module.Filter.UserValves.model_fields)
            - ALWAYS_ON_VALVE_NAMES
            - {f"IMAGE_{name.upper()}" for name in spec.schema_only}
        )
        assert published == expected_valves, (
            "each knob comes from this model's own contract; a knob it does not publish "
            f"must have no field at all. got {published}"
        )
        assert ALWAYS_ON_VALVE_NAMES <= set(module.Filter.UserValves.model_fields), (
            "the controls that do not come from a contract are offered on every model, "
            "or a model that publishes nothing about them cannot reach them at all"
        )

        chosen = spec.enums[0][1][0]
        valves = module.Filter.UserValves(IMAGE_ASPECT_RATIO=chosen)
        body = module.Filter().inlet({"model": model_id}, None, {"valves": valves})
        assert body["image_config"]["aspect_ratio"] == chosen, (
            f"the filter must write the chosen value through. got {body.get('image_config')!r}"
        )
    finally:
        sys.modules.pop(module_name, None)


async def _install_ids(model_ids, endpoint_records):
    """Run the real install loop over a registry holding just these models.

    Sourced from the recorded sweep rather than the older curated catalog, because the
    endpoint contracts these tests feed in were recorded against the same models.
    """
    import json
    from unittest.mock import AsyncMock, MagicMock

    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    catalog = json.loads(
        (Path(__file__).parent / "fixtures" / "openrouter_image_models.json").read_text()
    )["data"]
    wanted = set(model_ids)
    picked = [m for m in catalog if m["id"] in wanted]
    assert {m["id"] for m in picked} == wanted, (
        f"missing from the recorded catalog: {sorted(wanted - {m['id'] for m in picked})}"
    )

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry.set_image_endpoints(endpoint_records)
    OpenRouterModelRegistry.register_image_models(picked)

    pipe = MagicMock()
    pipe.valves.AUTO_INSTALL_IMAGE_FILTERS = True
    pipe.valves.ENABLE_OPENROUTER_IMAGE_GENERATION = True
    fm = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    fm._ensure_filter_installed = AsyncMock(side_effect=lambda **kwargs: kwargs["preferred_id"])

    models = OpenRouterModelRegistry.list_models()
    return models, await fm.ensure_openrouter_image_filter_function_ids(models)


@pytest.mark.asyncio
async def test_each_model_gets_its_own_filter_not_a_shared_variant():
    """Two models, two distinct filters.

    The variants this replaced installed one shared `..._recraft` filter across all
    eleven Recraft models, so a knob published by one reached the others.
    """
    models, result = await _install_ids(
        ["recraft/recraft-v3", "qwen/qwen-image-3"],
        {
            "recraft/recraft-v3": _recorded_endpoint("recraft_recraft-v3"),
            "qwen/qwen-image-3": _recorded_endpoint("qwen_qwen-image-3"),
        },
    )

    ids = [result.get(m["id"]) or [] for m in models]
    assert all(len(v) == 1 for v in ids), f"one filter per model, got {ids}"
    assert len({v[0] for v in ids}) == 2, (
        f"two models must not share one filter id, got {ids}"
    )


@pytest.mark.asyncio
async def test_a_model_with_no_published_contract_gets_no_filter():
    """No contract means no knobs, and a filter offering nothing is not installed.

    The alternative — installing it anyway — puts an empty toggle in the admin list and
    an empty section in the chat controls for a model nothing is known about.
    """
    models, result = await _install_ids(["openai/gpt-image-2"], {})

    assert models, "the model must still register and be selectable"
    assert result == {}, f"a knobless model must attach no filter, got {result}"


@pytest.mark.asyncio
async def test_both_id_forms_map_to_separate_lists():
    """`installed[model_id]` and `installed[original_id]` hold equal but distinct lists.

    Sharing one list object means a later mutation through either key corrupts both.
    """
    models, result = await _install_ids(
        ["recraft/recraft-v3"], {"recraft/recraft-v3": _recorded_endpoint("recraft_recraft-v3")}
    )
    model = models[0]

    sanitized = result.get(model["id"])
    original = result.get(model["original_id"])
    assert sanitized is not None and original is not None
    assert sanitized == original
    assert sanitized is not original

    sanitized.append("test-injection")
    assert "test-injection" not in original, (
        "mutating one key's list changed the other; they alias the same object"
    )

@pytest.mark.parametrize(("fixture", "model_id"), EVERY_CONTRACT, ids=[s for s, _ in EVERY_CONTRACT])
def test_the_filter_writes_for_the_id_open_webui_actually_sends(fixture, model_id):
    """The id is built the way production builds it, never typed as a literal.

    Open WebUI's model id is its function id joined to what `pipes()` returned, and
    `pipes()` returns `sanitize_model_id`'s output, which has no slash. A filter that
    only recognises the slash form is inert for every request, and a test that types the
    slash form by hand cannot tell.

    The id that must be ignored is drawn from the contract set rather than typed, because
    the one that was typed here -- `openai.gpt-image-2` -- is itself a catalogued model.
    Once this runs over the whole set it becomes the model under test, and a filter doing
    exactly the right thing would have failed on it.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )

    spec = build_image_model_filter_spec(
        model_id, {"id": model_id, "name": model_id}, _recorded_endpoint(fixture),
        dedicated_image_api=True,
    )
    module = _load_filter_from_source(
        render_image_model_filter_source(spec), f"owui_id_form_{fixture}"
    )
    knob, published = spec.enums[0]
    chosen = published[0]
    field = f"IMAGE_{knob.upper()}"
    dotted = sanitize_model_id(model_id)

    other = next(
        sanitize_model_id(other_id) for other_slug, other_id in EVERY_CONTRACT if other_slug != fixture
    )
    assert other != dotted, "the stand-in for a different model must not be this model"

    writes = {
        "open_webui_openrouter_pipe." + dotted,   # what Open WebUI sends
        "some_other_function_id." + dotted,       # any function id, not just ours
        dotted,                                   # sanitized, unprefixed
        model_id,                                 # the raw catalog slug
        "~" + model_id,                           # catalog alias
    }
    ignores = {
        other,                                    # a different catalogued model
        "not" + dotted,                           # shares a suffix, different model
        "",
    }

    for sent in sorted(writes):
        body = module.Filter().inlet(
            {"model": sent}, None, {"valves": module.Filter.UserValves(**{field: chosen})}
        )
        assert body.get("image_config") == {knob: chosen}, (
            f"{sent!r} is a form Open WebUI can send; the filter must write for it"
        )

    for sent in sorted(ignores):
        body = module.Filter().inlet(
            {"model": sent}, None, {"valves": module.Filter.UserValves(**{field: chosen})}
        )
        assert "image_config" not in body, (
            f"{sent!r} is not this model; the filter must leave the body alone"
        )


@pytest.mark.parametrize(("fixture", "model_id"), EVERY_CONTRACT, ids=[s for s, _ in EVERY_CONTRACT])
def test_the_chosen_value_is_the_value_that_travels(fixture, model_id):
    """Two different choices must produce two different requests.

    One choice cannot establish this: a filter that ignores the user and always writes
    the model's first published ratio satisfies a single-value assertion.

    The knob is whichever one the model lists first, not `aspect_ratio` by name -- three
    catalogued models lead with `quality`, and reading a hard-coded ratio back out of
    their requests would have looked like a filter that drops the user's choice.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )

    spec = build_image_model_filter_spec(
        model_id, {"id": model_id, "name": model_id}, _recorded_endpoint(fixture),
        dedicated_image_api=True,
    )
    module = _load_filter_from_source(
        render_image_model_filter_source(spec), f"value_travels_{fixture}"
    )
    knob, published = spec.enums[0]
    field = f"IMAGE_{knob.upper()}"
    first, last = published[0], published[-1]
    assert first != last, (
        f"{model_id} must publish at least two {knob} values for this to mean anything"
    )

    got = [
        module.Filter()
        .inlet({"model": model_id}, None, {"valves": module.Filter.UserValves(**{field: value})})
        .get("image_config", {})
        .get(knob)
        for value in (first, last)
    ]
    assert got == [first, last], f"each choice must arrive as itself; got {got}"


@pytest.mark.parametrize(("fixture", "model_id"), EVERY_CONTRACT, ids=[s for s, _ in EVERY_CONTRACT])
def test_every_control_the_filter_shows_is_a_control_that_writes(fixture, model_id):
    """A field the chat UI renders and `inlet` drops is worse than no field.

    The expectation is derived from what was rendered, not hand-listed, so a knob kind
    added later is covered the day it appears.

    A passthrough the renderer knows published values for is drawn as a choice list, not
    a text box, so the value put through it has to come from that list. Free text was
    typed here for every passthrough alike, which the six OpenAI models drop on the floor
    as the invalid choice it is -- reading like a filter that ignores its own control.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        PASSTHROUGH_ENUMS,
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )

    spec = build_image_model_filter_spec(
        model_id, {"id": model_id, "name": model_id}, _recorded_endpoint(fixture),
        dedicated_image_api=True,
    )
    module = _load_filter_from_source(
        render_image_model_filter_source(spec), f"every_control_{fixture}"
    )

    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        ALWAYS_ON_VALVE_NAMES,
    )

    chosen: dict[str, object] = {}
    expected: dict[str, object] = {}
    for name in spec.schema_only:
        chosen[f"IMAGE_{name.upper()}"] = "1024x1024"
        expected[name] = "1024x1024"
    for name, values in spec.enums:
        chosen[f"IMAGE_{name.upper()}"] = values[0]
        expected[name] = values[0]
    for name, low, high in spec.ranges:
        value = max(1, low)
        chosen[f"IMAGE_{name.upper()}"] = value
        expected[name] = value
    for name in spec.supported:
        chosen[f"IMAGE_{name.upper()}"] = 12345
        expected[name] = 12345
    for name in spec.passthrough:
        offered = PASSTHROUGH_ENUMS.get(name)
        value = offered[0][0] if offered else "a_value"
        chosen[f"IMAGE_{name.upper()}"] = value
        expected[name] = value

    assert set(chosen) == set(module.Filter.UserValves.model_fields) - ALWAYS_ON_VALVE_NAMES, (
        "the spec and the rendered fields must describe the same knob set"
    )
    body = module.Filter().inlet(
        {"model": model_id}, None, {"valves": module.Filter.UserValves(**chosen)}
    )
    assert body.get("image_config") == expected, (
        f"every rendered control must reach the request; got {body.get('image_config')}"
    )


def test_the_filter_adds_to_image_config_rather_than_replacing_it():
    """A caller may send `image_config` itself; the filter contributes to it.

    Replacing the dict destroys a key the user set directly, and writing an empty dict
    when nothing was chosen puts `image_config` on every request from the model.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )

    spec = build_image_model_filter_spec(
        "recraft/recraft-v3",
        {"id": "recraft/recraft-v3", "name": "Recraft V3"},
        _recorded_endpoint("recraft_recraft-v3"),
        dedicated_image_api=True,
    )
    module = _load_filter_from_source(render_image_model_filter_source(spec), "merge_semantics")
    ratio = spec.enums[0][1][0]

    merged = module.Filter().inlet(
        {"model": "recraft/recraft-v3", "image_config": {"seed": 99}},
        None,
        {"valves": module.Filter.UserValves(IMAGE_ASPECT_RATIO=ratio)},
    )
    assert merged["image_config"] == {"seed": 99, "aspect_ratio": ratio}, (
        f"a key the caller set must survive; got {merged['image_config']}"
    )

    untouched = module.Filter().inlet(
        {"model": "recraft/recraft-v3"}, None, {"valves": module.Filter.UserValves()}
    )
    assert "image_config" not in untouched, (
        f"nothing chosen means nothing written; got {untouched}"
    )


def _spec(**record):
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )

    return build_image_model_filter_spec("v/m", {"id": "v/m", "name": "M"}, record, dedicated_image_api=True)


@pytest.mark.parametrize(
    ("descriptor", "expected_ranges"),
    [
        ({"type": "range", "min": 1, "max": 6}, (("n", 1, 6),)),
        ({"type": "range", "min": "1", "max": "6"}, ()),
        ({"type": "range", "min": 0, "max": None}, ()),
        ({"type": "range", "min": 0.0, "max": 0.9}, ()),
        ({"type": "range", "min": 2}, ()),
        ({"type": "range", "min": 3, "max": 3}, ()),
    ],
)
def test_a_range_the_pipe_cannot_read_becomes_no_knob(descriptor, expected_ranges):
    """A bound that is not a whole number is not a bound.

    Substituting a default produced a control advertising "accepts 0 to 0" whose only
    value was the one the write path refuses to send.
    """
    spec = _spec(provider_slug="p", supported_parameters={"n": descriptor})
    assert spec.ranges == expected_ranges, f"{descriptor} produced {spec.ranges}"
    assert spec.knob_count == len(expected_ranges)


def test_a_knob_the_renderer_skips_is_not_counted_as_a_knob():
    """The install gate and the field renderer must agree on what a knob is.

    `input_references` counts attached images; it is never a control. Counting it while
    refusing to render it installs a filter whose whole body is `pass`.
    """
    only = _spec(
        provider_slug="p",
        supported_parameters={"input_references": {"type": "range", "min": 0, "max": 4}},
    )
    assert only.knob_count == 0, "a contract with no real control must count zero"

    alongside = _spec(
        provider_slug="p",
        supported_parameters={
            "input_references": {"type": "range", "min": 0, "max": 4},
            "aspect_ratio": {"type": "enum", "values": ["1:1"]},
        },
    )
    assert alongside.knob_count == 1, "the real control still counts"
    assert [n for n, _ in alongside.enums] == ["aspect_ratio"]


def test_passthrough_needs_a_provider_to_carry_it():
    """Without a provider slug the adapter drops the whole provider block.

    Rendering the control anyway shows the user a knob whose value is discarded on every
    request.
    """
    named = {"allowed_passthrough_parameters": ["style"], "supported_parameters": {}}
    assert _spec(**named).passthrough == (), "no slug means the value cannot be addressed"
    assert _spec(provider_slug="   ", **named).passthrough == ()
    assert _spec(provider_slug=["recraft"], **named).passthrough == ()
    assert _spec(provider_slug="recraft", **named).passthrough == ("style",)


def test_a_passthrough_name_cannot_shadow_a_published_control():
    """When both name the same parameter, the one carrying allowed values wins.

    Rendering both defines the field twice; pydantic keeps the last, so the validated
    dropdown silently became a free-text box.
    """
    spec = _spec(
        provider_slug="p",
        supported_parameters={"quality": {"type": "enum", "values": ["low", "high"]}},
        allowed_passthrough_parameters=["quality", "style"],
    )
    assert spec.passthrough == ("style",), f"quality must not be duplicated; got {spec.passthrough}"

    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        render_image_model_filter_source,
    )

    module = _load_filter_from_source(render_image_model_filter_source(spec), "no_shadow")
    annotation = module.Filter.UserValves.model_fields["IMAGE_QUALITY"].annotation
    assert annotation is not str, "the published values must survive as a choice, not free text"
    body = module.Filter().inlet(
        {"model": "v/m"}, None, {"valves": module.Filter.UserValves(IMAGE_QUALITY="low")}
    )
    assert body["image_config"] == {"quality": "low"}


@pytest.mark.parametrize(
    "hostile",
    [
        'v/m"\nBREAKOUT = 1\nX = "v/m',
        'v/m\u2028BREAKOUT = 1',
        'v/m""" + __import__("os").getcwd() + """',
    ],
)
def test_a_catalog_string_cannot_become_a_statement(hostile):
    """Model ids and names come from OpenRouter and end up in code Open WebUI executes.

    The check is on the compiled module's own constants, not on the text of the source:
    the source containing the string is exactly what a successful injection looks like.
    """
    import ast

    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )

    for spec in (
        build_image_model_filter_spec(hostile, {"id": hostile, "name": "M"}, {}, dedicated_image_api=True),
        build_image_model_filter_spec("v/m", {"id": "v/m", "name": hostile}, {}, dedicated_image_api=True),
    ):
        source = render_image_model_filter_source(spec)
        tree = ast.parse(source)
        assigned = {
            t.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Name)
        }
        assert "BREAKOUT" not in assigned, "a catalog string became a module-level assignment"
        compile(source, "<hostile>", "exec")


def test_a_name_that_is_not_an_identifier_costs_only_itself():
    """`cfg-scale` is a real provider parameter and is not a legal Python name.

    Rendering it stopped the whole module parsing, so the model lost every other knob it
    published rather than just the one that could not be expressed.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        render_image_model_filter_source,
    )

    spec = _spec(
        provider_slug="p",
        supported_parameters={"aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]}},
        allowed_passthrough_parameters=["cfg-scale", "2fast", "", "style"],
    )
    assert spec.passthrough == ("style",), f"only the usable name survives; got {spec.passthrough}"

    module = _load_filter_from_source(render_image_model_filter_source(spec), "odd_names")
    assert "IMAGE_ASPECT_RATIO" in module.Filter.UserValves.model_fields, (
        "the knobs that can be expressed must still be offered"
    )


@pytest.mark.asyncio
async def test_one_models_install_failure_costs_only_that_model():
    """A failure installing for one model must not skip the models after it.

    The install path reaches Open WebUI's database, whose driver errors belong to no
    tuple this package can enumerate, so the failure injected here is a RuntimeError --
    a class no narrow catch would have listed.
    """
    from unittest.mock import AsyncMock, MagicMock

    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    wanted = ["recraft/recraft-v3", "recraft/recraft-v4", "qwen/qwen-image-3"]
    record = {
        "provider_slug": "p",
        "supported_parameters": {"aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]}},
    }
    catalog = json.loads(
        (Path(__file__).parent / "fixtures" / "openrouter_image_models.json").read_text()
    )["data"]
    picked = [m for m in catalog if m["id"] in set(wanted)]
    assert len(picked) == len(wanted), "the recorded catalog must carry all three"

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    records = {m["id"]: record for m in picked}
    OpenRouterModelRegistry.set_image_endpoints(records)
    OpenRouterModelRegistry.register_image_models(picked)

    doomed = "openrouter_image_filter_recraft_recraft_v3"

    async def _install(**kwargs):
        if kwargs["preferred_id"] == doomed:
            raise RuntimeError("the database is locked")
        return kwargs["preferred_id"]

    pipe = MagicMock()
    fm = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    fm._ensure_filter_installed = AsyncMock(side_effect=_install)

    models = OpenRouterModelRegistry.list_models()
    result = await fm.ensure_openrouter_image_filter_function_ids(models)

    assert fm._ensure_filter_installed.await_count == len(wanted), (
        f"every model must be attempted; only {fm._ensure_filter_installed.await_count} were"
    )
    survivors = {fid for ids in result.values() for fid in ids}
    assert doomed not in survivors, "the model whose install raised must not be reported installed"
    assert len(survivors) == len(wanted) - 1, (
        f"the other models keep their filters; got {sorted(survivors)}"
    )


@pytest.mark.asyncio
async def test_two_models_of_the_same_family_do_not_share_one_filter():
    """Same vendor, same family prefix, same provider -- still two filters.

    A cross-vendor pair cannot show this: an id derived from the vendor, the family or
    the provider slug would still produce two distinct values for models from different
    vendors.
    """
    record = {
        "provider_slug": "recraft",
        "supported_parameters": {"aspect_ratio": {"type": "enum", "values": ["1:1"]}},
    }
    models, result = await _install_ids(
        ["recraft/recraft-v3", "recraft/recraft-v4"],
        {"recraft/recraft-v3": record, "recraft/recraft-v4": record},
    )

    ids = [result.get(m["id"]) or [] for m in models]
    assert all(len(v) == 1 for v in ids), f"one filter per model, got {ids}"
    assert len({v[0] for v in ids}) == 2, (
        f"two models of one family must not share a filter id, got {ids}"
    )


def test_a_knob_is_offered_only_if_every_provider_of_the_model_accepts_it():
    """Which provider serves a request is decided per request, after the controls exist.

    A model served by several providers publishes one contract each and they disagree,
    so the only set that is right whichever provider serves is the one they share.
    """
    wide = {
        "provider_slug": "a",
        "supported_parameters": {
            "aspect_ratio": {"type": "enum", "values": ["1:1", "16:9", "4:3"]},
            "n": {"type": "range", "min": 1, "max": 6},
        },
        "allowed_passthrough_parameters": ["style", "controls"],
    }
    narrow = {
        "provider_slug": "b",
        "supported_parameters": {
            "aspect_ratio": {"type": "enum", "values": ["1:1", "4:3"]},
            "n": {"type": "range", "min": 2, "max": 4},
        },
        "allowed_passthrough_parameters": ["style"],
    }

    alone = _spec_from(wide)
    assert dict(alone.enums)["aspect_ratio"] == ("1:1", "16:9", "4:3")
    assert alone.ranges == (("n", 1, 6),)
    assert alone.passthrough == ("style", "controls")

    both = _spec_from([wide, narrow])
    assert dict(both.enums)["aspect_ratio"] == ("1:1", "4:3"), (
        "a ratio only one provider accepts must not be offered"
    )
    assert both.ranges == (("n", 2, 4),), "the range narrows to what both accept"
    assert both.passthrough == ("style",), "a passthrough only one provider names is not offered"

    unaddressable = _spec_from([wide, {"supported_parameters": {}, "allowed_passthrough_parameters": ["style"]}])
    assert unaddressable.passthrough == (), (
        "a record with no provider slug cannot carry a provider option, so none is offered"
    )


@pytest.mark.parametrize(
    ("fixture", "model_id"),
    sorted(_multi_provider_contracts().items()),
)
def test_a_multi_provider_contract_narrows_to_what_its_providers_share(fixture, model_id):
    """The narrowing, driven by a contract OpenRouter really published.

    `google/gemini-3-pro-image` is served by Vertex and by AI Studio, and only AI Studio
    publishes 4K; a filter built from either record alone offers a set that is wrong for
    the other provider, and which provider serves is decided per request. This ran on
    hand-built records until the recorded sweep was checked in.

    The order the records arrive in is asserted to make no difference, because on this
    contract the intersection happens to equal the *first* record -- so an implementation
    that simply took `records[0]` satisfied the intersection assertion on its own, and did
    when it was run as a mutation.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )

    records = _recorded_endpoint(fixture)
    assert len(records) > 1, f"{fixture} is not a multi-provider contract"

    model = {"id": model_id, "name": model_id}
    agreed = dict(build_image_model_filter_spec(model_id, model, records, dedicated_image_api=True).enums)
    per_provider = [
        dict(build_image_model_filter_spec(model_id, model, record, dedicated_image_api=True).enums) for record in records
    ]

    for name, values in agreed.items():
        published = [set(alone.get(name, ())) for alone in per_provider]
        assert set(values) == set.intersection(*published), (
            f"{model_id} offers {name}={values!r}; the providers publish {published!r}"
        )

    reordered = dict(
        build_image_model_filter_spec(model_id, model, list(reversed(records)), dedicated_image_api=True).enums
    )
    assert reordered == agreed, (
        f"{model_id} offers a different set when its providers are listed in the other "
        f"order: {agreed!r} vs {reordered!r}. Which provider OpenRouter lists first is "
        "not a decision about what the model supports"
    )


def test_a_recorded_contract_really_does_withhold_something_a_provider_publishes():
    """The narrowing above is only proven if some recorded contract exercises it.

    Every multi-provider contract could publish identical records, in which case the
    intersection is the identity and the assertions pass on a model that cannot tell a
    working narrowing from none at all.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )

    withheld: dict[str, dict[str, list[str]]] = {}
    for fixture, model_id in sorted(_multi_provider_contracts().items()):
        records = _recorded_endpoint(fixture)
        model = {"id": model_id, "name": model_id}
        agreed = dict(build_image_model_filter_spec(model_id, model, records, dedicated_image_api=True).enums)
        per_provider = [
            dict(build_image_model_filter_spec(model_id, model, record, dedicated_image_api=True).enums) for record in records
        ]
        dropped = {
            name: sorted(set().union(*[set(a.get(name, ())) for a in per_provider]) - set(values))
            for name, values in agreed.items()
        }
        if any(dropped.values()):
            withheld[model_id] = {name: values for name, values in dropped.items() if values}

    assert withheld, (
        "no checked-in contract has providers that disagree, so nothing here would notice "
        "a build that offered the union, or the first record, instead of the intersection"
    )


def _spec_from(record):
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )

    return build_image_model_filter_spec("v/m", {"id": "v/m", "name": "M"}, record, dedicated_image_api=True)


def test_a_filter_id_is_derived_from_the_model_id_and_stays_readable():
    """The id is what an operator reads in Open WebUI's function list.

    It is also derived on the same terms as the video sibling, because both take a model
    id; the tighter thresholds this replaces hashed away the readable part of most ids.
    """
    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        sanitize_video_filter_id,
    )

    assert sanitize_image_filter_id("recraft/recraft-v3") == "openrouter_image_filter_recraft_recraft_v3"
    assert sanitize_image_filter_id("qwen/qwen-image-3") == "openrouter_image_filter_qwen_qwen_image_3"

    # Long enough to cross the 54-character gate. The previous id here cleaned to 33, so
    # the truncation, the digest and the surrogate guard behind that gate were all dead to
    # the suite -- and the "same terms" assertion below compared two functions that were
    # both returning the untouched string.
    long_id = "black-forest-labs/flux.2-klein-4b-preview-2026-08-experimental-build"
    cleaned = sanitize_image_filter_id(long_id).removeprefix("openrouter_image_filter_")
    assert len(cleaned) <= 54, f"the gate must actually shorten it. got {len(cleaned)}"
    assert re.fullmatch(r"[0-9a-f]{8}", cleaned[-8:]), (
        f"and end in a digest, so two long ids cannot collide. got {cleaned!r}"
    )
    assert "flux_2_klein_4b" in cleaned, (
        "the readable part must survive; it is how an operator finds the row"
    )
    assert cleaned == sanitize_video_filter_id(long_id).removeprefix("openrouter_video_"), (
        "both sanitizers take a model id and must shorten it on the same terms -- asserted "
        "inside the branch that shortens, or it compares two untouched strings"
    )


@pytest.mark.parametrize("sanitizer", ["image", "video"])
def test_a_model_id_carrying_a_surrogate_gets_a_filter_id_rather_than_a_crash(sanitizer):
    """A catalogue id can carry an unpaired surrogate, and `str.encode` refuses one.

    Reached only past the length gate, where the id is hashed. The image side raises
    without the guard; the video side raised with it, because the guard was added to one
    of the two identical call sites. A raise here costs every model of that kind its
    filter, since the failure escapes the per-model loop.
    """
    import json

    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        sanitize_image_filter_id,
    )
    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        sanitize_video_filter_id,
    )

    model_id = json.loads(r'"vendorlongname/vendorlongname-vendorlongname-vendorlongname-model-\ud800-x"')
    assert any(0xD800 <= ord(c) <= 0xDFFF for c in model_id), "the input must be hostile"
    run = sanitize_image_filter_id if sanitizer == "image" else sanitize_video_filter_id

    result = run(model_id)
    assert result.isascii() and result, f"a usable id, not a crash. got {result!r}"

    assert sanitize_image_filter_id("") == "openrouter_image_filter_model", (
        "an unusable id must not fall back onto a retired filter's id"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("filters_wanted", [True, False])
async def test_the_catalog_publishes_contracts_the_installer_can_actually_read(filters_wanted):
    """The wire between reading a contract and a model having knobs.

    Both halves are tested on their own; nothing tested that they are connected, so
    replacing the fetch with an empty dict left every model knobless with a green suite.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )
    from open_webui_openrouter_pipe.integrations import image_catalog

    record = {
        "provider_slug": "recraft",
        "supported_parameters": {"aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]}},
    }
    catalog = json.loads(
        (Path(__file__).parent / "fixtures" / "openrouter_image_models.json").read_text()
    )["data"]
    picked = [m for m in catalog if m["id"] == "recraft/recraft-v3"]
    assert picked, "the recorded catalog must carry the model"
    awaited = []

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def list_models(self):
            return picked

        async def endpoints(self, model_id):
            awaited.append(model_id)
            return [record]

    valves = MagicMock()
    valves.ENABLE_OPENROUTER_IMAGE_GENERATION = True
    valves.AUTO_INSTALL_IMAGE_FILTERS = filters_wanted
    valves.AUTO_ATTACH_IMAGE_FILTERS = filters_wanted
    valves.BASE_URL = "https://openrouter.ai/api/v1"
    valves.HTTP_REFERER_OVERRIDE = ""

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry._image_endpoints = {}
    OpenRouterModelRegistry._last_image_attempt = 0.0
    OpenRouterModelRegistry._last_image_fetch = 0.0

    original = image_catalog.OpenRouterImageClient
    image_catalog.OpenRouterImageClient = _Client  # type: ignore[misc]
    try:
        await image_catalog.ensure_image_catalog_loaded(
            session=MagicMock(),
            valves=valves,
            api_key="k",
            logger=logging.getLogger("t"),
            cache_seconds=0,
        )
    finally:
        image_catalog.OpenRouterImageClient = original  # type: ignore[misc]

    published = OpenRouterModelRegistry.image_endpoint("recraft/recraft-v3")
    if filters_wanted:
        assert awaited == ["recraft/recraft-v3"], "the contract must be read once for the model"
        assert published == [record], f"what the client published must be what is stored; got {published}"
        spec = build_image_model_filter_spec("recraft/recraft-v3", picked[0], published, dedicated_image_api=True)
        assert spec.knob_count > 0, "a model with a published contract must end up with knobs"
    else:
        assert awaited == [], "no filter consumes contracts, so none should be read"
        assert published is None


@pytest.mark.asyncio
async def test_a_contract_already_read_survives_a_later_failed_read():
    """A read that timed out is not a contract that changed.

    Dropping it strips the model of its controls for a whole cycle, and disagrees with
    the adapter, which keeps its own cached record on exactly the same failure.
    """
    from open_webui_openrouter_pipe.integrations.image_catalog import _fetch_endpoint_records

    record = {"provider_slug": "p", "supported_parameters": {}}
    OpenRouterModelRegistry._image_endpoints = {}
    OpenRouterModelRegistry.set_image_endpoints({"a/b": [record]}, known_ids={"a/b", "c/d"})
    assert OpenRouterModelRegistry.image_endpoint("a/b") == [record]
    # The installer looks a model up by whichever id form it holds, so both must resolve.
    assert OpenRouterModelRegistry.image_endpoint(sanitize_model_id("a/b")) == [record], (
        "the sanitized form of a stored id must find the same record"
    )
    assert OpenRouterModelRegistry.image_endpoint("a.c") is None, (
        "and an id belonging to no stored model must not match one by accident"
    )

    class _Failing:
        async def endpoints(self, model_id):
            raise TimeoutError("upstream is slow")

    logs = []

    class _Log:
        def log(self, level, msg, *args):
            logs.append(msg % args if args else msg)

        def debug(self, *a, **k):
            pass

    failed = await _fetch_endpoint_records(_Failing(), [{"id": "a/b"}], _Log())
    assert failed == {}, "a failed read publishes nothing"
    OpenRouterModelRegistry.set_image_endpoints(failed, known_ids={"a/b", "c/d"})
    assert OpenRouterModelRegistry.image_endpoint("a/b") == [record], (
        "the contract the pipe already had must survive a failed refresh"
    )

    OpenRouterModelRegistry.set_image_endpoints({}, known_ids={"c/d"})
    assert OpenRouterModelRegistry.image_endpoint("a/b") is None, (
        "a model the catalog no longer lists must not keep its contract forever"
    )


@pytest.mark.asyncio
async def test_a_contract_that_cannot_be_read_is_always_reported():
    """A model absent from the result is always named, whatever went wrong.

    A tuple of enumerated exception classes let anything unlisted be dropped by the
    gather with no diagnostic at all -- and a mock client missing the method was
    silently exercising that hole inside a test named "happy path".
    """
    from open_webui_openrouter_pipe.integrations.image_catalog import _fetch_endpoint_records

    class _NoMethod:
        pass

    logs = []

    class _Log:
        def log(self, level, msg, *args):
            logs.append(msg % args if args else msg)

        def debug(self, *a, **k):
            pass

    records = await _fetch_endpoint_records(_NoMethod(), [{"id": "a/b"}, {"id": "c/d"}], _Log())
    assert records == {}
    assert logs, "a model that lost its contract must produce a diagnostic"
    assert "a/b" in logs[0] and "c/d" in logs[0], f"both models must be named; got {logs}"


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ('{"artistic_level": 2}', {"artistic_level": 2}),
        ('["a", "b"]', ["a", "b"]),
        ("realistic_image", "realistic_image"),
        ("null", "null"),
        ("true", "true"),
        ("2K", "2K"),
        ("1_000", "1_000"),
        ("123", 123),
        ("-7", -7),
        ("1.5", 1.5),
        ("NaN", "NaN"),
        ("Infinity", "Infinity"),
        ("-Infinity", "-Infinity"),
        ("1e400", "1e400"),
    ],
)
def test_a_passthrough_value_arrives_as_what_the_user_meant(typed, expected):
    """A container or a bare number is parsed; nothing else is.

    `style` really does take a bare word, so parsing everything turned `null` into None --
    a value the user never asked for, and one the adapter drops. `123` is the other way
    round: 8 of the 17 published passthrough names are numeric, and a quoted number is a
    different request. `NaN`, `Infinity` and `1e400` stay strings because the float they
    would produce is one no JSON encoder can put on the wire.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )

    spec = build_image_model_filter_spec(
        "recraft/recraft-v3",
        {"id": "recraft/recraft-v3", "name": "Recraft V3"},
        _recorded_endpoint("recraft_recraft-v3"),
        dedicated_image_api=True,
    )
    module = _load_filter_from_source(render_image_model_filter_source(spec), "decode_ok")
    body = module.Filter().inlet(
        {"model": "recraft/recraft-v3"},
        None,
        {"valves": module.Filter.UserValves(IMAGE_CONTROLS=typed)},
    )
    assert body["image_config"]["controls"] == expected
    assert type(body["image_config"]["controls"]) is type(expected)


@pytest.mark.parametrize(
    "typed", ["[NaN]", '{"scale": Infinity}', "[1e400]", "[-Infinity]"]
)
def test_a_number_no_encoder_can_serialise_is_refused_inside_a_container(typed):
    """`json.dumps` emits a bare `NaN`, and `allow_nan=False` raises.

    Parsed into `image_config` it reaches the request builder as a float nothing can
    encode, and the failure surfaces somewhere that names neither the field nor the value.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )

    spec = build_image_model_filter_spec(
        "recraft/recraft-v3",
        {"id": "recraft/recraft-v3", "name": "Recraft V3"},
        _recorded_endpoint("recraft_recraft-v3"),
        dedicated_image_api=True,
    )
    module = _load_filter_from_source(render_image_model_filter_source(spec), "decode_nan")
    with pytest.raises(Exception) as caught:
        module.Filter().inlet(
            {"model": "recraft/recraft-v3"},
            None,
            {"valves": module.Filter.UserValves(IMAGE_CONTROLS=typed)},
        )
    assert "controls" in str(caught.value), (
        f"the message must name the control the user typed into; got {caught.value}"
    )


def test_a_container_that_does_not_parse_names_the_field_it_came_from():
    """The user typed it, so the message has to say which control it was.

    Sending it as a string instead produces a provider error about something else.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )

    spec = build_image_model_filter_spec(
        "recraft/recraft-v3",
        {"id": "recraft/recraft-v3", "name": "Recraft V3"},
        _recorded_endpoint("recraft_recraft-v3"),
        dedicated_image_api=True,
    )
    module = _load_filter_from_source(render_image_model_filter_source(spec), "decode_bad")
    with pytest.raises(Exception) as caught:
        module.Filter().inlet(
            {"model": "recraft/recraft-v3"},
            None,
            {"valves": module.Filter.UserValves(IMAGE_TEXT_LAYOUT='[{"text":')},
        )
    assert "text_layout" in str(caught.value), (
        f"the message must name the control the user typed into; got {caught.value}"
    )


def test_the_source_check_asks_the_question_open_webui_asks():
    """Open WebUI compiles the filter; the check must too.

    `ast.parse` accepts a module whose `from __future__` import is no longer first, so a
    filter could pass validation, be stored, and then fail to load forever.
    """
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    ok, err = FilterManager.validate_filter_source("x = 1\n")
    assert (ok, err) == (True, None)

    ok, err = FilterManager.validate_filter_source("x = 1\nfrom __future__ import annotations\n")
    assert ok is False and err, (
        "a module Open WebUI cannot compile must not pass validation"
    )


@pytest.mark.asyncio
async def test_both_catalog_reads_are_bounded():
    """Unbounded, these run on the path that builds Open WebUI's model list.

    The bound is read from the module constant, so raising it is a one-line change and
    removing it is a test failure.
    """
    import aiohttp

    from open_webui_openrouter_pipe.integrations import image_client as image_client_module
    from open_webui_openrouter_pipe.integrations.image_client import OpenRouterImageClient

    seen = []

    class _Resp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return {"data": []}

        def raise_for_status(self):
            return None

    class _Session:
        def get(self, url, headers=None, timeout=None):
            seen.append(timeout)
            return _Resp()

    client = OpenRouterImageClient(
        _Session(),  # type: ignore[arg-type]  - a stub standing in for the aiohttp session
        base_url="https://x/api/v1",
        api_key="k",
        logger=logging.getLogger("t"),
    )
    await client.list_models()
    await client.endpoints("a/b")

    assert len(seen) == 2, "both reads must go through the session"
    for timeout in seen:
        assert isinstance(timeout, aiohttp.ClientTimeout)
        assert timeout.total == image_client_module._CATALOG_TIMEOUT_SECONDS


def test_a_numeric_published_value_stays_numeric():
    """The adapter checks the chosen value against the contract's own list.

    Stringifying rendered a control whose every option was then rejected, because "512"
    is not 512 -- a knob built from the contract that the contract refuses.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )
    from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter

    record = {
        "provider_slug": "p",
        "supported_parameters": {"resolution": {"type": "enum", "values": [512, 1024]}},
    }
    spec = build_image_model_filter_spec("v/m", {"id": "v/m", "name": "M"}, record, dedicated_image_api=True)
    assert dict(spec.enums)["resolution"] == (512, 1024), (
        f"the values must survive as published; got {dict(spec.enums).get('resolution')}"
    )

    module = _load_filter_from_source(render_image_model_filter_source(spec), "numeric_enum")
    body = module.Filter().inlet(
        {"model": "v/m"}, None, {"valves": module.Filter.UserValves(IMAGE_RESOLUTION=512)}
    )
    assert body["image_config"] == {"resolution": 512}

    top_level, _provider, notes = ImageGenerationAdapter._split_image_config(
        body, allowed_passthrough=(), record=record
    )
    assert top_level == {"resolution": 512}, f"the adapter must accept it; notes were {notes}"
    assert notes == []


@pytest.mark.parametrize("published", ["1:1,16:9", {"a": 1}, None, 42])
def test_a_values_field_that_is_not_a_list_yields_no_control(published):
    """A malformed `values` was iterated, so a string became one option per character."""
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )

    spec = build_image_model_filter_spec(
        "v/m",
        {"id": "v/m", "name": "M"},
        {"provider_slug": "p", "supported_parameters": {"aspect_ratio": {"type": "enum", "values": published}}},
        dedicated_image_api=True,
    )
    assert spec.enums == (), f"{published!r} is not a published option list; got {spec.enums}"
    assert spec.knob_count == 0


@pytest.mark.parametrize(("fixture", "model_id"), EVERY_CONTRACT, ids=[s for s, _ in EVERY_CONTRACT])
def test_help_names_exactly_the_controls_the_filter_draws(fixture, model_id):
    """One authority for what a model offers, read by both surfaces.

    Help used to carry its own hand-written knob list, which is how it came to advertise
    controls from filters that no longer exist. The expectation is read off the filter
    source the chat UI is built from -- the panel the user is actually looking at -- so
    a control drawn there and absent here fails, which is what the earlier expectation
    (rebuilt from the same spec fields help listed) could not see: it agreed with help
    about the always-on controls and the size field by leaving all four out of both.
    """
    import re

    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )

    record = _recorded_endpoint(fixture)
    model = {"id": model_id, "name": model_id}
    spec = build_image_model_filter_spec(model_id, model, record, dedicated_image_api=True)
    rendered = render_image_help(model_id, model, endpoint_record=record, dedicated_image_api=True)

    expected = [
        double or single
        for double, single in re.findall(
            r"""^\s+title=(?:"([^"]+)"|'([^']+)'),$""",
            render_image_model_filter_source(spec),
            re.M,
        )
    ]
    assert expected, "the fixture must publish something for this to mean anything"

    assert "## Controls" in rendered, "the section must exist for a model with a contract"
    controls = rendered.split("## Controls", 1)[1]
    named = [
        line.split("**")[1]
        for line in controls.splitlines()
        if line.startswith("- **") and "**" in line[4:]
    ]
    assert named == expected, (
        f"help must name exactly the controls the filter draws.\n  help: {named}\n  filter: {expected}"
    )


@pytest.mark.parametrize(
    "fixture",
    [
        "openai_gpt-image-2",
        "openai_gpt-image-1",
        "openai_gpt-image-1-mini",
        "openai_gpt-5-image",
        "openai_gpt-5-image-mini",
        "openai_gpt-5.4-image-2",
    ],
)
def test_help_lists_the_choices_a_provider_setting_is_drawn_with(fixture):
    """A control drawn as a dropdown must not be described as free text.

    The filter draws `moderation` from the published choices, so the panel offers
    `auto` and `low` and nothing else. Help listed every provider setting with one flat
    sentence, so the six models that publish this one told the reader to guess at a
    value the control would not have accepted.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )
    from open_webui_openrouter_pipe.integrations.image_types import PASSTHROUGH_ENUMS

    record = _recorded_endpoint(fixture)
    model_id = _recorded_contract_slugs()[fixture]
    model = {"id": model_id, "name": model_id}
    drawn = render_image_model_filter_source(build_image_model_filter_spec(model_id, model, record, dedicated_image_api=True))
    values, meaning = PASSTHROUGH_ENUMS["moderation"]
    literal = f"Literal[{', '.join(repr(choice) for choice in ('', *values))}]"
    assert literal in drawn, "the panel must draw these choices for this to matter"

    rendered = render_image_help(model_id, model, endpoint_record=record, dedicated_image_api=True)
    line = next(
        line for line in rendered.splitlines() if line.startswith("- **moderation**")
    )
    assert meaning in line, line
    assert f"Choices: {', '.join(values)}." in line, line


def test_help_still_describes_a_provider_setting_with_no_published_choices():
    """Only settings OpenRouter documents get a list. The rest keep the plain sentence."""
    from open_webui_openrouter_pipe.integrations.image_types import PASSTHROUGH_ENUMS

    assert "font_inputs" not in PASSTHROUGH_ENUMS, "this case exists to cover the other branch"
    rendered = render_image_help(
        "sourceful/riverflow-v2.5-fast",
        {"id": "sourceful/riverflow-v2.5-fast", "name": "Riverflow V2.5 Fast"},
        endpoint_record=_recorded_endpoint("sourceful_riverflow-v2.5-fast"),
        dedicated_image_api=True,
    )
    assert "- **font_inputs** — a setting this model's provider accepts." in rendered
    assert "Choices:" not in rendered.split("- **font_inputs**", 1)[1]


def test_help_says_so_when_a_model_publishes_no_controls():
    """Silence would read as "the help is broken", which is a different thing."""
    rendered = render_image_help(
        "vendor/unknown", {"id": "vendor/unknown", "name": "Unknown"}, endpoint_record={},
        dedicated_image_api=True,
    )
    assert "publishes no adjustable settings" in rendered, rendered


def test_help_without_a_contract_still_describes_the_model():
    """A contract that could not be read must not cost the user the prose as well."""
    rendered = render_image_help("recraft/recraft-v3", {"id": "recraft/recraft-v3", "name": "R"}, dedicated_image_api=True)
    assert rendered.strip(), "the model description must survive"
    assert "## Controls" not in rendered, (
        "with no contract there is nothing truthful to list, so nothing is listed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("readable", [True, False])
async def test_a_contract_the_pipe_could_not_read_is_not_reported_as_a_model_with_nothing(
    readable, monkeypatch
):
    """A failed lookup must not be printed to the user as the model's own answer.

    With both auto-install settings off nothing pre-fetches contracts, so every `help`
    does a live read -- and the read answers with an EMPTY LIST on a network error, a
    404, a rate limit or a missing key, not with None. That list is not None, so it
    walked straight past the guard and the reply stated as fact that the model publishes
    no adjustable settings. Recraft V3 publishes four controls.

    The value fed to the renderer is the one the real reader produces, taken from a
    client stubbed one seam below the subject -- comparing the renderer against a
    hand-written `[]` proves nothing about what the reader returns. Both outcomes are
    driven, so a reply that never makes the claim fails the readable row.
    """
    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.core.config import EncryptedStr

    published = _recorded_endpoint("recraft_recraft-v3")

    class _Contract:
        async def endpoints(self, _model_id):
            if not readable:
                raise RuntimeError("the contract read failed")
            return published

    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    adapter = pipe._ensure_image_generation_adapter()
    monkeypatch.setattr(adapter, "_client", lambda *_a, **_k: _Contract())
    try:
        records = await adapter._published_records(object(), pipe.valves, "recraft/recraft-v3")
    finally:
        await pipe.close()

    assert bool(records) is readable, (
        f"precondition: the reader answers {records!r} for readable={readable}"
    )

    rendered = render_image_help(
        "recraft/recraft-v3",
        {"id": "recraft/recraft-v3", "name": "Recraft V3"},
        endpoint_record=records,
        dedicated_image_api=True,
    )

    assert rendered.strip(), "the model description must survive either way"
    assert ("publishes no adjustable settings" in rendered) is False, (
        f"Recraft V3 publishes four controls; the reply denies them: {rendered}"
    )
    assert ("## Controls" in rendered) is readable, (
        f"controls may only be listed when the contract they came from was read: {rendered}"
    )


@pytest.mark.asyncio
async def test_help_reads_the_contract_itself_when_nothing_cached_it():
    """Filters and help answer different questions.

    Contracts are cached only because the installer needs them. An operator who installs
    no filters still asks "what can this model do", and the answer must not become
    nothing. The assertion is on the help a user would receive, driven through the
    orchestrator -- comparing the renderer to itself proves only that it is a function.
    """
    from open_webui_openrouter_pipe.integrations.image_help import render_image_help

    record = _recorded_endpoint("recraft_recraft-v3")
    model = {"id": "recraft/recraft-v3", "name": "Recraft V3"}

    OpenRouterModelRegistry._image_endpoints = {}
    assert OpenRouterModelRegistry.image_endpoint("recraft/recraft-v3") is None, (
        "nothing cached, so the orchestrator must read the contract itself"
    )

    without = render_image_help("recraft/recraft-v3", model, dedicated_image_api=True)
    assert "## Controls" not in without, (
        "with no contract at all there is nothing truthful to list"
    )

    with_record = render_image_help("recraft/recraft-v3", model, endpoint_record=record, dedicated_image_api=True)
    assert "## Controls" in with_record
    assert "Aspect ratio" in with_record.split("## Controls", 1)[1], (
        "the fetched contract must reach the rendered controls"
    )
    assert len(with_record) > len(without), (
        "the fetched contract must add to what the user is told, not replace it"
    )


def _declares_valve(source: str, name: str) -> bool:
    """Whether a rendered panel declares *name* as a control, not merely mentions it.

    Every always-on control is named twice in a rendered panel: once where the field is
    declared and once where the filter reads it back. Searching the whole source for the
    name is answered by the read-back alone, so a panel that stopped declaring the field
    still satisfied it.
    """
    return re.search(rf"\n\s+{re.escape(name)}: [^\n]*= Field\(", source) is not None


@pytest.mark.asyncio
async def test_a_contract_that_shrinks_to_nothing_replaces_the_old_controls():
    """A model can lose every knob: a provider joins and the intersection empties.

    Returning early left the previous filter installed, active and attached, so the user
    kept a panel of controls the model no longer accepts and every message carried values
    the request path then rejected.

    What replaces it still has to carry the controls the panel supplies itself rather than
    the model, which no contract can empty -- help closes every such card by listing them,
    so a panel rewritten without them promises controls that exist nowhere. Which starting
    state the model was in is varied where production can still see it, one test along, by
    driving the same shrunk contract at a model with a row and at a model without one.
    """
    from unittest.mock import AsyncMock, MagicMock

    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager
    from open_webui_openrouter_pipe.filters.image_filter_renderer import ALWAYS_ON_VALVE_NAMES

    wide = {
        "provider_slug": "a",
        "supported_parameters": {"aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]}},
    }
    disjoint = {
        "provider_slug": "b",
        "supported_parameters": {"aspect_ratio": {"type": "enum", "values": ["4:3"]}},
    }

    pipe = MagicMock()
    fm = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    written: list[str] = []
    fm._ensure_filter_installed = AsyncMock(
        side_effect=lambda **kw: (written.append(kw["desired_source"]), kw["preferred_id"])[1]
    )

    result = await fm._ensure_single_image_filter_function_id(
        model_id="v/m", image_model={"id": "v/m", "name": "M"}, endpoint_record=[wide, disjoint],
        dedicated_image_api=True,
    )

    assert result, "an installed filter must be reachable so it can be overwritten"
    assert written, "the stored filter must be rewritten, not left as it was"
    assert "IMAGE_ASPECT_RATIO" not in written[0], (
        "the control the model no longer accepts must be gone from the stored filter"
    )
    for name in ALWAYS_ON_VALVE_NAMES:
        assert _declares_valve(written[0], name), (
            f"{name} is drawn on every image-API panel and help lists it, so the panel "
            "that replaces the old one must declare it"
        )




@pytest.mark.asyncio
async def test_retirement_touches_only_the_rows_the_previous_design_left():
    """This is the one place the changeset deactivates rows in shared Open WebUI state.

    Deactivating too much costs an operator filters they installed themselves; too little
    leaves an ungated filter writing values into every request.
    """
    import sys
    from types import ModuleType, SimpleNamespace
    from unittest.mock import MagicMock

    from open_webui_openrouter_pipe.core.config import _OPENROUTER_IMAGE_FILTER_MARKER
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )

    current = render_image_model_filter_source(
        build_image_model_filter_spec(
            "recraft/recraft-v3",
            {"id": "recraft/recraft-v3", "name": "R"},
            _recorded_endpoint("recraft_recraft-v3"),
            dedicated_image_api=True,
        )
    )
    rows = [
        SimpleNamespace(id="old_variant", content=f'MARKER = "{_OPENROUTER_IMAGE_FILTER_MARKER}"'),
        SimpleNamespace(id="current_per_model", content=current),
        SimpleNamespace(id="another_pipe_filter", content='MARKER = "openrouter_pipe:image_gen_filter:v1"'),
        SimpleNamespace(id="someone_elses", content="class Filter:\n    pass\n"),
    ]
    deactivated: list[str] = []

    class _Functions:
        @staticmethod
        async def get_functions_by_type(kind, active_only=True):
            return rows

        @staticmethod
        async def update_function_by_id(row_id, payload):
            if payload.get("is_active") is False:
                deactivated.append(row_id)
            return True

    module = ModuleType("open_webui.models.functions")
    module.Functions = _Functions  # type: ignore[attr-defined]
    saved = sys.modules.get("open_webui.models.functions")
    sys.modules["open_webui.models.functions"] = module
    try:
        pipe = MagicMock()
        fm = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
        await fm._retire_variant_image_filters()
    finally:
        if saved is not None:
            sys.modules["open_webui.models.functions"] = saved
        else:
            sys.modules.pop("open_webui.models.functions", None)

    assert deactivated == ["old_variant"], (
        f"only the superseded row may be deactivated; got {deactivated}"
    )


@pytest.mark.parametrize(("fixture", "model_id"), EVERY_CONTRACT, ids=[s for s, _ in EVERY_CONTRACT])
def test_help_prose_never_names_a_setting_the_model_does_not_publish(fixture, model_id):
    """The prose and the Controls list sit in one reply and must not disagree.

    Hand-written prose describing which knobs exist is a second copy of the contract;
    it drifted before and told users to use filters that no longer existed.
    """
    records = _recorded_endpoint(fixture)
    model = {"id": model_id, "name": model_id}
    rendered = render_image_help(model_id, model, endpoint_record=records, dedicated_image_api=True)
    prose = rendered.split("## Controls", 1)[0]

    published: set[str] = set()
    for record in records:
        published |= set(record.get("supported_parameters") or {})
        published |= set(record.get("allowed_passthrough_parameters") or [])

    import re

    from open_webui_openrouter_pipe.integrations.image_types import RENDERABLE_FIELD_NAME_RE

    # Every backticked token that looks like a parameter name, rather than a fixed list --
    # a list cannot see a passthrough name the model does not publish, which is most of
    # what the curated prose talks about.
    NOT_PARAMETERS = {
        "help", "auto", "png", "jpeg", "webp", "svg", "true", "false", "null",
    }
    claimed = {
        token
        for token in re.findall(r"`([^`]+)`", prose)
        if RENDERABLE_FIELD_NAME_RE.fullmatch(token) and token.lower() not in NOT_PARAMETERS
    }
    unpublished = sorted(claimed - published)
    assert not unpublished, (
        f"the prose names {unpublished}, which this model does not publish -- the Controls "
        f"section below it correctly omits those controls. Published: {sorted(published)}"
    )


@pytest.mark.asyncio
async def test_the_contract_sweep_is_bounded_as_a_whole_not_only_per_read():
    """The number of models must not appear in the wait a user experiences.

    This runs inside the call that builds Open WebUI's model list. Capping each read
    still lets the catalogue's size multiply the stall.
    """
    import asyncio
    import time

    from open_webui_openrouter_pipe.integrations import image_catalog

    class _Slow:
        async def endpoints(self, model_id):
            await asyncio.sleep(30)
            return [{"provider_slug": "p", "supported_parameters": {}}]

    logs: list[str] = []

    class _Log:
        def log(self, level, msg, *args):
            logs.append(msg % args if args else msg)

        def warning(self, msg, *args, **kwargs):
            logs.append(msg % args if args else msg)

        def debug(self, *a, **k):
            pass

    models = [{"id": f"v/m{i}"} for i in range(24)]
    original = image_catalog._SWEEP_BUDGET_SECONDS
    image_catalog._SWEEP_BUDGET_SECONDS = 1
    started = time.monotonic()
    try:
        records = await image_catalog._fetch_endpoint_records(_Slow(), models, _Log())
    finally:
        image_catalog._SWEEP_BUDGET_SECONDS = original
    elapsed = time.monotonic() - started

    assert elapsed < 10, (
        f"the sweep must give up on its own budget, not run the catalogue through; "
        f"took {elapsed:.1f}s"
    )
    assert records == {}, "nothing was read, so nothing is published"
    assert any("still unread" in line for line in logs), (
        f"an operator must be told the sweep ran out of time; got {logs}"
    )


def test_the_documented_help_example_is_what_the_code_produces():
    """A sample of product output in the docs must be reproducible.

    The block this replaces showed one model's identity above another model's controls,
    and carried tips that had been deleted from the code -- it had been hand-edited in
    place, so nothing tied it to anything the product does.
    """
    doc = (Path(__file__).parent.parent / "docs" / "openrouter_image_generation.md").read_text()
    marker = "reproducible from the contract recorded in"
    assert marker in doc, "the example must say what it is reproducible from"

    block = doc.split(marker, 1)[1].split("```", 2)[1].strip()
    rendered = render_image_help(
        "recraft/recraft-v3",
        {"id": "recraft/recraft-v3", "name": "Recraft V3"},
        endpoint_record=_recorded_endpoint("recraft_recraft-v3"),
        dedicated_image_api=True,
    ).strip()
    assert block == rendered, (
        "the documented example no longer matches what the code renders.\n"
        f"--- doc ---\n{block[:400]}\n--- code ---\n{rendered[:400]}"
    )


def test_every_catalogued_image_model_has_curated_help():
    """`help` is the only in-product documentation a model has.

    Without this census a model added to the catalogue answers "No curated help available"
    while the README names it by name. The video side has had this check all along.
    """
    catalogue = json.loads(
        (Path(__file__).parent / "fixtures" / "openrouter_image_models.json").read_text()
    )["data"]
    ids = {m["id"] for m in catalogue if isinstance(m, dict) and m.get("id")}
    missing = sorted(ids - set(IMAGE_HELP_BY_MODEL))
    assert not missing, (
        f"{len(missing)} catalogued image model(s) have no curated help entry: {missing}"
    )


def test_every_catalogued_image_model_has_a_checked_in_contract():
    """The contracts the fleet-wide claims are measured on must live inside the tree.

    Forty were recorded into `.external/`, which is gitignored, so every sweep that read
    them measured nothing on CI and read a neighbouring checkout when run from a scratch
    copy. Both directions are asserted: a contract for a model the catalogue dropped is
    as wrong as a catalogued model with no contract, and the sweeps are parametrised over
    this set, so an empty one would quietly collect no nodes at all.
    """
    catalogue = json.loads(
        (Path(__file__).parent / "fixtures" / "openrouter_image_models.json").read_text()
    )["data"]
    ids = {m["id"] for m in catalogue if isinstance(m, dict) and m.get("id")}
    recorded = _recorded_contract_slugs()

    assert set(recorded.values()) == ids, (
        f"catalogued with no contract: {sorted(ids - set(recorded.values()))}; "
        f"contract for an uncatalogued model: {sorted(set(recorded.values()) - ids)}"
    )
    for slug, model_id in recorded.items():
        assert slug == model_id.replace("/", "_"), (
            f"{slug!r} does not name {model_id!r}; the sweeps derive one from the other"
        )


def test_the_documented_model_table_lists_exactly_the_catalogued_models():
    """A model table is an inventory, and an inventory that drifts misleads.

    It listed three Riverflow previews OpenRouter had removed and omitted twelve models
    the README named, while the prose above it claimed a count matching neither.
    """
    import re

    doc = (Path(__file__).parent.parent / "docs" / "openrouter_image_generation.md").read_text()
    listed = {m for m in re.findall(r"^\| `([^`]+)` \|", doc, re.M) if "/" in m}
    catalogued = {
        m["id"]
        for m in json.loads(
            (Path(__file__).parent / "fixtures" / "openrouter_image_models.json").read_text()
        )["data"]
        if isinstance(m, dict) and m.get("id")
    }

    assert listed == catalogued, (
        f"documented but not catalogued: {sorted(listed - catalogued)}\n"
        f"catalogued but not documented: {sorted(catalogued - listed)}"
    )
    assert f"{len(catalogued)}" in doc.split("\n\n", 2)[1] or "forty" in doc.split("\n\n", 2)[1], (
        "the count in the opening paragraph must match the table"
    )


def test_every_image_only_model_in_the_catalogue_is_registered():
    """The catalogue decides which models register, not a list typed beside it.

    The hand-written list this replaces named three Riverflow previews OpenRouter had
    withdrawn, so it asserted the presence of models that no longer exist while saying
    nothing about the twelve that had appeared.
    """
    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []

    OpenRouterModelRegistry.register_image_models(IMAGE_MODELS)

    expected = {
        ModelFamily.base_model(sanitize_model_id(m["id"]))
        for m in IMAGE_MODELS
        if "text" not in ((m.get("architecture") or {}).get("output_modalities") or [])
    }
    assert expected, "the catalogue must contain image-only models for this to mean anything"

    registered = set(OpenRouterModelRegistry._specs)
    assert expected <= registered, (
        f"image-only models missing from the registry: {sorted(expected - registered)}"
    )

    multimodal = {
        sanitize_model_id(m["id"])
        for m in IMAGE_MODELS
        if "text" in ((m.get("architecture") or {}).get("output_modalities") or [])
    }
    assert multimodal, "the catalogue must contain multimodal models too"
    assert not (multimodal & registered), (
        "models that also emit text belong to the chat catalogue and must not be "
        f"re-registered here: {sorted(multimodal & registered)}"
    )


def test_a_detached_filter_stops_being_default_on_but_other_owners_survive():
    """Driven in the caller's order, with the caller's own derived flags.

    The test this replaces called the default helper directly with flags the caller
    cannot produce, so it passed while the prune was unreachable in production: the
    attach pass rewrites the ownership record before the default pass reads it, and the
    caller had already switched the default pass off when a model lost its filter.
    """
    from open_webui_openrouter_pipe.core.config import _PIPE_METADATA_KEY
    from open_webui_openrouter_pipe.models.catalog_manager import (
        _apply_list_default_filter_ids,
        _apply_list_filter_ids,
        _detached_by_this_pass,
    )

    def run(meta, current):
        """The order and arguments `_update_or_insert_model_with_metadata` uses."""
        detached = _detached_by_this_pass(
            meta, prune_key="image_filter_ids", filter_function_ids=current
        )
        _apply_list_filter_ids(
            meta,
            filter_function_ids=current,
            filter_supported=True,
            auto_attach=True,
            prune_key="image_filter_ids",
        )
        _apply_list_default_filter_ids(
            meta,
            detached=detached,
            filter_function_ids=current,
            filter_supported=True,
            auto_default=True,
        )

    losing = {
        "filterIds": ["openrouter_image_filter_old"],
        "defaultFilterIds": ["openrouter_image_filter_old", "a_global_filter_i_do_not_own"],
        _PIPE_METADATA_KEY: {"image_filter_ids": ["openrouter_image_filter_old"]},
    }
    run(losing, [])
    assert losing["filterIds"] == [], "the model no longer has a filter of ours"
    assert losing["defaultFilterIds"] == ["a_global_filter_i_do_not_own"], (
        "our detached filter must stop being default-on, and nobody else's may be touched; "
        f"got {losing['defaultFilterIds']}"
    )

    keeping = {
        "filterIds": [],
        "defaultFilterIds": ["a_global_filter_i_do_not_own"],
        _PIPE_METADATA_KEY: {},
    }
    run(keeping, ["openrouter_image_filter_new"])
    assert keeping["filterIds"] == ["openrouter_image_filter_new"]
    assert "openrouter_image_filter_new" in keeping["defaultFilterIds"], (
        "a filter attached this pass becomes default-on"
    )
    assert "a_global_filter_i_do_not_own" in keeping["defaultFilterIds"], (
        "and another owner's default still survives"
    )


@pytest.mark.asyncio
async def test_the_refresh_retires_superseded_filters_whether_or_not_it_installs_any():
    """The sweep must run in both configurations, and its call sites are what deletes.

    Its own test drove the method directly, so both call sites could be removed with the
    suite green -- and the valves-off path is the upgrade where nothing supersedes the
    old rows, which is the whole reason the sweep exists.
    """
    import sys
    from types import ModuleType, SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from open_webui_openrouter_pipe.core.config import _OPENROUTER_IMAGE_FILTER_MARKER
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    deactivated: list[str] = []
    rows = [SimpleNamespace(id="old_variant", content=f'M = "{_OPENROUTER_IMAGE_FILTER_MARKER}"')]

    class _Functions:
        @staticmethod
        async def get_functions_by_type(kind, active_only=True):
            return rows

        @staticmethod
        async def update_function_by_id(row_id, payload):
            if payload.get("is_active") is False:
                deactivated.append(row_id)
            return True

    module = ModuleType("open_webui.models.functions")
    module.Functions = _Functions  # type: ignore[attr-defined]
    saved = sys.modules.get("open_webui.models.functions")
    sys.modules["open_webui.models.functions"] = module
    try:
        models, _ = await _install_ids(
            ["recraft/recraft-v3"],
            {"recraft/recraft-v3": _recorded_endpoint("recraft_recraft-v3")},
        )
        assert models, "the model must register"
    finally:
        if saved is not None:
            sys.modules["open_webui.models.functions"] = saved
        else:
            sys.modules.pop("open_webui.models.functions", None)

    assert deactivated == ["old_variant"], (
        "installing filters must also retire the ones a previous design left behind; "
        f"got {deactivated}"
    )


def test_a_stored_value_the_contract_no_longer_accepts_costs_only_itself():
    """Open WebUI builds UserValves from the stored dict and passes NO valves if it raises.

    These fields track a live contract, so a provider joining a model narrows a range
    while an older choice is still stored -- an ordinary event. Without the validator one
    stale entry throws away every other choice the user made.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )

    spec = build_image_model_filter_spec(
        "v/m",
        {"id": "v/m", "name": "M"},
        {
            "provider_slug": "p",
            "supported_parameters": {
                "aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]},
                "n": {"type": "range", "min": 1, "max": 4},
            },
            "allowed_passthrough_parameters": ["style"],
        },
        dedicated_image_api=True,
    )
    module = _load_filter_from_source(render_image_model_filter_source(spec), "stale_valves")

    # Exactly how Open WebUI constructs them -- outside inlet, where a raise costs everything.
    valves = module.Filter.UserValves(
        **{
            "IMAGE_ASPECT_RATIO": "9:21",   # no longer published
            "IMAGE_N": 99,                  # outside the published range
            "IMAGE_STYLE": "realistic",     # still fine
            "GONE": "x",                    # not a field any more
        }
    )
    assert valves.IMAGE_STYLE == "realistic", "a value that still fits must survive"
    assert valves.IMAGE_ASPECT_RATIO == "", "a value the contract dropped falls back"
    assert valves.IMAGE_N is None, "a value outside the published range falls back"

    kept = module.Filter.UserValves(**{"IMAGE_ASPECT_RATIO": "16:9", "IMAGE_N": 2})
    assert (kept.IMAGE_ASPECT_RATIO, kept.IMAGE_N) == ("16:9", 2), (
        "values that fit must round-trip untouched"
    )


def test_a_case_variant_of_a_published_knob_cannot_shadow_it():
    """Two published names differing only in case render one field.

    Without the guard the typed control loses to the free-text one, and the override
    block still runs `int(...)` on it -- so `int("")` raises on EVERY request through
    that filter, not only when the user sets something. OpenRouter ships mixed-case
    parameter names, so this is a live shape.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        render_image_model_filter_source,
    )

    spec = _spec(
        provider_slug="p",
        supported_parameters={"seed": {"type": "boolean"}},
        allowed_passthrough_parameters=["Seed", "style"],
    )
    assert spec.passthrough == ("style",), (
        f"the case-variant of a published knob must be dropped; got {spec.passthrough}"
    )

    source = render_image_model_filter_source(spec)
    assert source.count("IMAGE_SEED:") == 1, "one definition, or pydantic keeps the wrong one"

    module = _load_filter_from_source(source, "case_variant_shadow")
    assert module.Filter.UserValves.model_fields["IMAGE_SEED"].annotation is not str, (
        "the typed control must survive, not be replaced by free text"
    )

    # The failure this really guards: a default-valued request must not raise.
    body = module.Filter().inlet(
        {"model": "v/m"}, None, {"valves": module.Filter.UserValves()}
    )
    assert "image_config" not in body, "nothing chosen means nothing written, and no crash"


def test_a_non_finite_published_value_never_reaches_a_field_annotation():
    """JSON admits NaN and Infinity, so a catalogue response can carry them.

    `repr(nan)` is the bare name `nan`, so the field annotation would refer to an
    undefined name -- and the source still parses, so the install-time check passes it.
    The failure lands on the first request to that model instead.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        render_image_model_filter_source,
    )

    spec = _spec(
        provider_slug="p",
        supported_parameters={
            "n": {"type": "enum", "values": [float("nan"), float("inf"), 1, 2]}
        },
    )
    assert dict(spec.enums)["n"] == (1, 2), (
        f"non-finite values must not survive into the spec; got {dict(spec.enums).get('n')}"
    )

    module = _load_filter_from_source(render_image_model_filter_source(spec), "non_finite")
    body = module.Filter().inlet(
        {"model": "v/m"}, None, {"valves": module.Filter.UserValves(IMAGE_N=1)}
    )
    assert body["image_config"] == {"n": 1}, "the finite values must still work"


@pytest.mark.asyncio
async def test_the_catalogue_ttl_still_applies_when_no_filter_consumes_contracts():
    """The contract clock gates a sweep that only some configurations perform.

    Reading it unconditionally meant a deployment with both filter valves off never
    stamped it, so the freshness check could never be satisfied and every model-list
    build refetched the catalogue.
    """
    from open_webui_openrouter_pipe.integrations import image_catalog

    fetches = []

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def list_models(self):
            fetches.append(1)
            return []

        async def endpoints(self, model_id):
            return []

    valves = MagicMock()
    valves.ENABLE_OPENROUTER_IMAGE_GENERATION = True
    valves.AUTO_INSTALL_IMAGE_FILTERS = False
    valves.AUTO_ATTACH_IMAGE_FILTERS = False
    valves.BASE_URL = "https://openrouter.ai/api/v1"
    valves.HTTP_REFERER_OVERRIDE = ""

    OpenRouterModelRegistry._last_image_attempt = 0.0
    OpenRouterModelRegistry._last_image_contract_attempt = 0.0
    OpenRouterModelRegistry._image_endpoints = {}

    original = image_catalog.OpenRouterImageClient
    image_catalog.OpenRouterImageClient = _Client  # type: ignore[misc]
    try:
        for _ in range(2):
            await image_catalog.ensure_image_catalog_loaded(
                session=MagicMock(), valves=valves, api_key="k",
                logger=logging.getLogger("t"), cache_seconds=3600,
            )
    finally:
        image_catalog.OpenRouterImageClient = original  # type: ignore[misc]

    assert len(fetches) == 1, (
        f"the second call is inside the TTL window and must not refetch; got {len(fetches)}"
    )


def test_a_published_provider_option_keeps_its_own_name():
    """A compatibility alias must not rename a key the model itself claims.

    `image_size` is the pipe's old spelling for `resolution` and is also a real provider
    option elsewhere. Renaming first meant a value typed into the provider box landed on
    the resolution the user had chosen from a dropdown, silently replacing it.
    """
    from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter

    record = {
        "provider_slug": "p",
        "supported_parameters": {"resolution": {"type": "enum", "values": ["1K", "2K"]}},
        "allowed_passthrough_parameters": ["image_size"],
    }
    top_level, provider, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"resolution": "2K", "image_size": "1K"}},
        allowed_passthrough=("image_size",),
        record=record,
    )
    assert top_level == {"resolution": "2K"}, (
        f"the user's dropdown choice must survive; got {top_level}"
    )
    assert provider == {"image_size": "1K"}, (
        f"the provider option must travel under its published name; got {provider}"
    )
    assert notes == [], f"nothing was wrong, so nothing should be reported; got {notes}"

    # And where the record does NOT claim it, the alias still applies.
    top_level, _provider, _notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"image_size": "2K"}},
        allowed_passthrough=(),
        record={"supported_parameters": {"resolution": {"type": "enum", "values": ["1K", "2K"]}}},
    )
    assert top_level == {"resolution": "2K"}, "the compatibility spelling still works"


def test_help_says_which_kind_of_nothing_a_model_offers():
    """Two very different states produced one sentence.

    A model that offers nothing and a model whose providers publish different things are
    not the same, and only one of them means the user should stop looking.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )

    a = {"provider_slug": "a", "supported_parameters": {"aspect_ratio": {"type": "enum", "values": ["1:1"]}}}
    b = {"provider_slug": "b", "supported_parameters": {"aspect_ratio": {"type": "enum", "values": ["16:9"]}}}
    model = {"id": "v/m", "name": "M"}

    disagreeing = build_image_model_filter_spec("v/m", model, [a, b], dedicated_image_api=True)
    assert disagreeing.knob_count == 0, "the intersection is empty"
    assert disagreeing.published_any_parameter, "but the records did publish something"
    text = render_image_help("v/m", model, endpoint_record=[a, b], dedicated_image_api=True)
    assert "publish different settings" in text, text
    assert "publishes no adjustable settings" not in text

    empty = build_image_model_filter_spec("v/m", model, [{"provider_slug": "a"}], dedicated_image_api=True)
    assert not empty.published_any_parameter
    text = render_image_help("v/m", model, endpoint_record=[{"provider_slug": "a"}], dedicated_image_api=True)
    assert "publishes no adjustable settings" in text, text


def _priced_record(pricing: Any, **extra: Any) -> dict[str, Any]:
    """A published contract carrying a price, shaped as the live endpoint listing is."""
    record: dict[str, Any] = {
        "provider_name": "Recraft",
        "provider_slug": "recraft",
        "supported_parameters": {"aspect_ratio": {"type": "enum", "values": ["1:1"]}},
        "pricing": pricing,
    }
    record.update(extra)
    return record


def test_no_price_reaches_a_user_at_all():
    """A rate a reader is shown goes stale in silence, whoever supplied it.

    Prose figures were the first shape of that: eight hardcoded into the descriptions,
    one quoting a model's reference-image charge as its generation charge and another its
    cheapest of three rates. Reading them out of the contract instead only moved the
    problem -- a rate on screen is still a rate somebody has to keep true, and the answer
    is that help quotes none.

    Every curated model is rendered twice, and the second rendering is handed a contract
    that publishes two different figures in two different units, so a panel that reads
    published pricing back onto the page fails here rather than passing on an input that
    never carried a figure at all.
    """
    published = _priced_record(
        [
            {"billable": "output_image", "unit": "image", "cost_usd": 0.25},
            {"billable": "input_reference", "unit": "megapixel", "cost_usd": 0.014},
        ]
    )
    census = 0
    for model_id in IMAGE_HELP_BY_MODEL:
        model = {"id": model_id, "name": model_id}
        for record in (None, published):
            rendered = render_image_help(
                model_id, model, endpoint_record=record, dedicated_image_api=True
            )
            assert IMAGE_HELP_BY_MODEL[model_id]["display_name"] in rendered, (
                f"{model_id} did not render its own card, so this proves nothing"
            )
            assert not re.search(r"\$\s*\d", rendered), f"{model_id} quotes a price"
            census += 1
    assert census == 2 * len(IMAGE_HELP_BY_MODEL), "the sweep skipped a curated model"


@pytest.mark.parametrize(
    "config",
    [
        {"image_size": "1K", "resolution": "2K"},
        {"resolution": "2K", "image_size": "1K"},
    ],
)
def test_both_spellings_of_one_setting_do_not_race(config):
    """Two filters can each write their own spelling into one `image_config`.

    The older filter writes `image_size`; a model's own filter writes the `resolution`
    its contract publishes. Aliasing one onto the other made them the same destination,
    so whichever landed later in the dict won and the other vanished -- with the outcome
    depending on insertion order and nothing said to the user.
    """
    from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter

    record = {"supported_parameters": {"resolution": {"type": "enum", "values": ["1K", "2K"]}}}
    top_level, _provider, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": config}, allowed_passthrough=(), record=record
    )

    assert top_level == {"resolution": "2K"}, (
        f"the spelling the model publishes must win, whatever the order; got {top_level}"
    )
    assert [n.kind for n in notes] == ["superseded"], (
        f"and the user must be told the other was ignored; got {notes}"
    )


def test_the_compatibility_spelling_still_works_on_its_own():
    """The alias is load-bearing for the older filter; only the collision changed."""
    from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter

    top_level, _provider, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"image_size": "1K"}},
        allowed_passthrough=(),
        record={"supported_parameters": {"resolution": {"type": "enum", "values": ["1K", "2K"]}}},
    )
    assert top_level == {"resolution": "1K"}
    assert notes == []



# ============================================================================
# REGFIX: infrastructure fix 5 -- a failed contract read is not a shrunk contract
# ============================================================================


@pytest.mark.parametrize(
    ("endpoint_record", "overwrites"),
    [(None, False), ([], True)],
    ids=["read-failed", "read-empty"],
)
@pytest.mark.asyncio
async def test_only_a_contract_that_was_read_may_blank_an_installed_filter(
    endpoint_record, overwrites
):
    """A 5xx on a cold worker is not a contract that shrank.

    Both inputs produce a knobless spec, so the two answers cannot come from the knob
    count; only the read outcome separates them. Overwriting on a failed read rewrites
    the model's filter to `pass`, and `_keep_what_still_fits` then discards every stored
    user choice as an unknown field.
    """
    from unittest.mock import AsyncMock, MagicMock

    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    pipe = MagicMock()
    manager = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    manager._ensure_filter_installed = AsyncMock(side_effect=lambda **kw: kw["preferred_id"])

    result = await manager._ensure_single_image_filter_function_id(
        model_id="v/m",
        image_model={"id": "v/m", "name": "M"},
        endpoint_record=endpoint_record,
        dedicated_image_api=True,
    )

    assert bool(manager._ensure_filter_installed.await_count) is overwrites, (
        "a failed contract read must leave the installed filter alone; a contract that "
        "was read and is genuinely empty must replace it"
    )
    assert (result is not None) is overwrites


@pytest.mark.asyncio
async def test_a_genuinely_empty_contract_is_not_collapsed_into_a_failed_read():
    """The registry lookup must not turn an empty list into None on its way down."""
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )

    assert build_image_model_filter_spec("v/m", {"id": "v/m"}, [], dedicated_image_api=True).contract_read is True
    assert build_image_model_filter_spec("v/m", {"id": "v/m"}, None, dedicated_image_api=True).contract_read is False


# ============================================================================
# The image-generation server-tool filter, built from the selected model's contract
# ============================================================================


def _image_gen_module(spec, *, name, catalog_match=True, selected_model=""):
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        render_image_gen_filter_source,
    )

    return _load_filter_from_source(
        render_image_gen_filter_source(
            spec, catalog_match=catalog_match, selected_model=selected_model
        ),
        name,
    )


def _published_union(records: list[dict], parameter: str) -> list:
    values: list = []
    for record in records:
        descriptor = (record.get("supported_parameters") or {}).get(parameter)
        if not isinstance(descriptor, dict):
            continue
        for value in descriptor.get("values") or []:
            if value not in values:
                values.append(value)
    return values


@pytest.mark.parametrize(
    ("slug", "model_id"), sorted(_recorded_contract_slugs().items())
)
def test_the_server_tool_filter_offers_exactly_what_the_model_publishes(slug, model_id):
    """Every option offered is published, and every published option is offered.

    One direction alone is satisfiable by a defect: offering nothing passes "nothing
    unpublished is offered", and offering every string ever seen passes "everything
    published is reachable". The fixed list this replaced failed both at once -- 37 of
    the 40 recorded models rejected at least one ratio it offered, and 28 published a
    ratio it could not reach.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )

    records = _recorded_endpoint(slug)
    spec = build_image_model_filter_spec(model_id, {"id": model_id, "name": model_id}, records, dedicated_image_api=True)
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        render_image_gen_filter_source,
    )

    source = render_image_gen_filter_source(
        spec, catalog_match=True, selected_model=model_id
    )
    module = _load_filter_from_source(
        source, f"image_gen_offers_{slug.replace('.', '_').replace('-', '_')}"
    )

    offered = dict(spec.enums)
    published = {
        name
        for record in records
        for name, descriptor in (record.get("supported_parameters") or {}).items()
        if isinstance(descriptor, dict)
        and descriptor.get("type") == "enum"
        and (descriptor.get("values") or [])
    }
    assert published, (
        f"{model_id} publishes no choice list at all, so this node asserts nothing"
    )

    for parameter in sorted(published | set(offered)):
        values = offered.get(parameter, ())
        union = _published_union(records, parameter)
        assert set(values) <= set(union), (
            f"{model_id} is offered {sorted(set(values) - set(union))!r} for {parameter}, "
            "which no provider of it publishes"
        )
        agreed = [
            value
            for value in union
            if all(
                value in (((r.get("supported_parameters") or {}).get(parameter) or {}).get("values") or [])
                for r in records
            )
        ]
        assert set(agreed) <= set(values) | set(dict(spec.narrowed).get(parameter, ())), (
            f"{model_id} publishes {sorted(set(agreed) - set(values))!r} for {parameter} "
            "and the filter offers no way to ask for it"
        )

    assert "0.5K" not in source, (
        f"{model_id} is offered the 0.5K tier, which no recorded contract publishes -- "
        "the published spelling is 512"
    )
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        IMAGE_GEN_TOOL_PARAMS,
    )

    drawn = set(module.Filter.UserValves.model_fields)
    seeded = re.search(r"params: dict\[str, Any\] = \{(.*?)\}", source)
    written = re.findall(r"[\"']([a-z_]+)[\"']\s*:", seeded.group(1) if seeded else "")
    written += re.findall(r"params\[[\"']([a-z_]+)[\"']\]\s*=", source)
    for parameter in _RECORDED_SERVER_TOOL_PARAMS:
        assert written.count(parameter) == 1, (
            f"{model_id}: OpenRouter's server-tool parameter table documents {parameter}, so "
            f"exactly one control must set it; {written.count(parameter)} do. The per-model "
            "images contract describes a different endpoint and does not decide which "
            "parameters the server tool carries."
        )
    assert ("IMAGE_RESOLUTION" in drawn) == bool(offered.get("resolution")), (
        f"{model_id} publishes resolution={bool(offered.get('resolution'))!r} and the panel "
        f"drew {('IMAGE_RESOLUTION' in drawn)!r}; the tier control is the one that does "
        "track this model's own published values"
    )
    assert ("IMAGE_SIZE" in drawn) != bool(offered.get("resolution")), (
        f"{model_id}: the tier dropdown and the free-text size box write the same request "
        "key, so drawing both loses whichever runs first without saying so"
    )


@pytest.mark.parametrize("candidate", ["resolution", "size", "image_size"])
def test_the_server_tool_tier_key_is_decided_in_one_place(monkeypatch, candidate):
    """The tier spelling is a constant, not a literal buried at a call site.

    Three spellings are in play and no source settles it: OpenRouter's own server-tool
    parameter table names `size`, its image request schema names `resolution`, and the
    filter shipped `image_size`, which appears in neither. Parametrised over all three so
    a hardcoded answer cannot pass -- one literal cannot satisfy three expectations.
    """
    from open_webui_openrouter_pipe.filters import image_filter_renderer as renderer
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )

    assert candidate in renderer.IMAGE_GEN_TOOL_TIER_KEY_CANDIDATES, (
        "a spelling a probe would test is not recorded, so settling the probe needs a "
        "new literal rather than a new default"
    )
    monkeypatch.setattr(renderer, "IMAGE_GEN_TOOL_TIER_KEY", candidate)

    records = _recorded_endpoint("google_gemini-3.1-flash-image")
    spec = build_image_model_filter_spec(
        "google/gemini-3.1-flash-image",
        {"id": "google/gemini-3.1-flash-image", "name": "Gemini"},
        records,
        dedicated_image_api=True,
    )
    tier = dict(spec.enums)["resolution"][0]
    module = _image_gen_module(spec, name=f"image_gen_tier_{candidate}")

    metadata: dict = {}
    module.Filter().inlet(
        {},
        metadata,
        {"valves": module.Filter.UserValves(IMAGE_RESOLUTION=tier)},
    )
    params = metadata["openrouter_pipe"]["server_tools"]["image_generation"]

    assert params.get(candidate) == tier, (
        f"the constant says {candidate!r} and the tool was sent {sorted(params)!r}"
    )
    for other in renderer.IMAGE_GEN_TOOL_TIER_KEY_CANDIDATES:
        if other != candidate:
            assert other not in params, (
                f"{other!r} went out alongside {candidate!r}; two spellings of one tier "
                "is a request that names the same thing twice"
            )


def test_the_shipped_tier_key_is_the_one_openrouters_own_parameter_table_publishes():
    """The candidate list is hand-written and the shipped key is its first entry, so
    asking whether one is in the other asks nothing. The recorded table is OpenRouter's.

    `image_size` appears in no OpenRouter document; `resolution` is the /images request
    field, not a chat server-tool parameter. Sending either names a knob the tool has
    no entry for, and unrecognised keys are dropped without a word.
    """
    from open_webui_openrouter_pipe.filters import image_filter_renderer as renderer

    published = set(_RECORDED_SERVER_TOOL_PARAMS)

    assert renderer.IMAGE_GEN_TOOL_TIER_KEY in published
    rejected = set(renderer.IMAGE_GEN_TOOL_TIER_KEY_CANDIDATES) - published
    assert rejected, "every candidate is published, so the table discriminates nothing"
    assert renderer.IMAGE_GEN_TOOL_TIER_KEY not in rejected


@pytest.mark.parametrize(("ratio", "ratio_survives"), [("16:9", False), ("1:1", True)])
def test_the_emitted_tool_call_never_carries_a_ratio_the_pixel_size_contradicts(
    ratio, ratio_survives
):
    """One size box, two ratios, opposite outcomes.

    "An explicit pixel size is authoritative: a mismatched `resolution` or `aspect_ratio`
    alongside it is rejected with a 400", and the server tool passes "all parameters except
    `model` ... directly to the underlying image generation API" -- so a contradicting pair
    does not cost the user the ratio, it costs them the whole tool call. ``1024x1024`` *is*
    1:1, so dropping the ratio there would take away a setting the API accepts: both
    directions have to hold, and one input pair proving both is what rules out an
    unconditional drop and an unconditional keep at the same time.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )
    from open_webui_openrouter_pipe.requests.orchestrator import _build_server_tool_entries

    model_id = "google/gemini-2.5-flash-image"
    spec = build_image_model_filter_spec(
        model_id,
        {"id": model_id, "name": "Gemini"},
        _recorded_endpoint("google_gemini-2.5-flash-image"),
        dedicated_image_api=True,
    )
    assert not dict(spec.enums).get("resolution"), (
        f"{model_id} publishes tiers, so its panel draws the tier dropdown rather than the "
        "free-text size box this case is about"
    )
    assert ratio in dict(spec.enums)["aspect_ratio"], (
        f"{model_id} does not publish {ratio}, so the panel never drew it and this case "
        "would pass on a value the user could not have chosen"
    )
    module = _image_gen_module(spec, name=f"image_gen_size_ratio_{ratio.replace(':', '_')}")

    metadata: dict = {}
    module.Filter().inlet(
        {},
        metadata,
        {"valves": module.Filter.UserValves(IMAGE_SIZE="1024x1024", IMAGE_ASPECT_RATIO=ratio)},
    )
    entries, _ = _build_server_tool_entries(metadata["openrouter_pipe"]["server_tools"])
    emitted = [e for e in entries if e["type"] == "openrouter:image_generation"]
    assert len(emitted) == 1, f"one panel must produce one tool entry; got {entries!r}"
    parameters = emitted[0]["parameters"]

    assert parameters["size"] == "1024x1024", (
        f"the box the user filled in is the authoritative one; got {parameters!r}"
    )
    assert parameters.get("aspect_ratio") == (ratio if ratio_survives else None), (
        f"{ratio} alongside 1024x1024 must be "
        f"{'sent' if ratio_survives else 'dropped'}; got {parameters!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("typed", "sent"), [("2K", True), ("8K", False)])
async def test_the_tool_panel_measures_a_size_tier_against_the_four_names(typed, sent):
    """The size box on the chat tool says a tier is checked, so a tier has to be checked.

    The model driven here publishes no tier list, which is the case the panel describes:
    its own sentence promises the value is measured against OpenRouter's four names
    before it goes out. `2K` is one of them and `8K` is not, so one value must survive
    and the other must be withheld -- neither an unconditional send nor an unconditional
    drop passes, and no constant satisfies both rows.

    The reporting half is driven end to end rather than assumed: the note the request
    path produces is handed to the reporter the request path uses, and the assertion is
    on the notification that reaches the socket.
    """
    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )
    from open_webui_openrouter_pipe.requests.orchestrator import _build_server_tool_entries

    model_id = "google/gemini-2.5-flash-image"
    spec = build_image_model_filter_spec(
        model_id,
        {"id": model_id, "name": "Gemini"},
        _recorded_endpoint("google_gemini-2.5-flash-image"),
        dedicated_image_api=True,
    )
    assert not dict(spec.enums).get("resolution"), (
        f"{model_id} publishes tiers, so its panel makes the other promise and this case "
        "would be measuring the wrong sentence"
    )
    module = _image_gen_module(spec, name=f"image_gen_size_tier_{typed}")
    described = module.Filter.UserValves.model_fields["IMAGE_SIZE"].description or ""
    assert "checked only against those four names" in described, (
        "the panel no longer makes the promise this test holds the request path to; "
        f"it says {described!r}"
    )

    metadata: dict = {}
    module.Filter().inlet(
        {}, metadata, {"valves": module.Filter.UserValves(IMAGE_SIZE=typed)}
    )
    entries, superseded = _build_server_tool_entries(
        metadata["openrouter_pipe"]["server_tools"]
    )
    emitted = [e for e in entries if e["type"] == "openrouter:image_generation"]
    assert len(emitted) == 1, f"one panel must produce one tool entry; got {entries!r}"
    parameters = emitted[0].get("parameters") or {}

    assert (parameters.get("size") == typed) is sent, (
        f"{typed} had to be {'sent' if sent else 'withheld'}; the tool call carries "
        f"{parameters!r}"
    )
    assert bool(superseded) is not sent, (
        f"{typed} was {'sent' if sent else 'withheld'} and the notes say {superseded!r}"
    )
    if sent:
        return

    assert [drawn_by for drawn_by, _note in superseded] == [model_id], (
        "the note names the wrong model, so the toast blames a model the user did not pick"
    )
    pipe = Pipe()
    events: list[dict] = []

    async def emitter(event):
        events.append(event)

    try:
        await pipe._ensure_image_generation_adapter()._report_notes(
            [note for _drawn_by, note in superseded],
            api_model_id=model_id,
            event_emitter=emitter,
        )
    finally:
        await pipe.close()

    told = [
        event["data"]["content"]
        for event in events
        if event.get("type") == "notification"
        and isinstance(event.get("data"), dict)
        and isinstance(event["data"].get("content"), str)
    ]
    assert told, f"nothing reached the user; the events were {events!r}"
    assert typed in told[-1], (
        f"the toast says {told[-1]!r} and never names the value that was withheld"
    )


def _note_cases():
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )

    model = {"id": "v/m", "name": "M"}
    return {
        "unmatched": (build_image_model_filter_spec("v/m", None, None, dedicated_image_api=True), False),
        "unreadable": (build_image_model_filter_spec("v/m", model, None, dedicated_image_api=True), True),
        "empty": (build_image_model_filter_spec("v/m", model, [{"provider_slug": "a"}], dedicated_image_api=True), True),
        "disagreeing": (
            build_image_model_filter_spec(
                "v/m",
                model,
                [
                    {
                        "provider_slug": "a",
                        "supported_parameters": {
                            "aspect_ratio": {"type": "enum", "values": ["1:1"]}
                        },
                    },
                    {
                        "provider_slug": "b",
                        "supported_parameters": {
                            "aspect_ratio": {"type": "enum", "values": ["16:9"]}
                        },
                    },
                ],
                dedicated_image_api=True,
            ),
            True,
        ),
    }


def test_a_model_that_publishes_nothing_offers_nothing_and_says_which_nothing():
    """Four ways to end up with no knobs, four different things to do about them.

    A wrong id, a read that has not landed, a model with no settings and providers that
    disagree all render the same empty panel. Collapsing them into one sentence sends the
    reader to check the id when the remedy was to wait, and the previous filter did
    neither -- it showed a fixed list of ratios the model had never published.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        image_gen_model_note,
        render_image_gen_filter_source,
    )

    cases = _note_cases()
    notes = {
        label: image_gen_model_note(spec, catalog_match=match)
        for label, (spec, match) in cases.items()
    }

    assert len(set(notes.values())) == len(notes), (
        f"two of these outcomes read identically: {notes}"
    )
    assert "could not be read" in notes["unreadable"]
    assert "could not be read" not in notes["empty"], (
        "a read that failed and a model with nothing to publish need different remedies"
    )
    assert "different settings" in notes["disagreeing"]

    for label, (spec, match) in cases.items():
        assert spec.knob_count == 0, f"{label} must offer no knobs, got {spec.knob_count}"
        source = render_image_gen_filter_source(spec, catalog_match=match, selected_model="v/m")
        compile(source, f"<{label}>", "exec")
        assert notes[label] in source, (
            f"the {label} filter renders no visible statement of why it is empty"
        )
        module = _load_filter_from_source(source, f"image_gen_empty_{label}")
        offered = {
            value
            for field in module.Filter.UserValves.model_fields.values()
            for value in (get_args(field.annotation) or ())
            if isinstance(value, str)
        }
        for retired in ("1024x1024", "1536x672", "0.5K", "21:9"):
            assert retired not in offered, (
                f"the {label} filter still offers {retired!r} as a choice, which came from "
                "the fixed list rather than from any contract"
            )


@pytest.mark.parametrize(
    ("stored", "slug"),
    [
        ("openai/gpt-5-image-mini", "openai_gpt-5-image-mini"),
        ("google/gemini-3.1-flash-image", "google_gemini-3.1-flash-image"),
    ],
)
@pytest.mark.asyncio
async def test_the_installed_filter_is_built_for_the_model_its_own_valve_names(
    monkeypatch, stored, slug
):
    """The admin's choice lives in the installed row, and has to reach the renderer.

    Parametrised over two models whose contracts differ, so a renderer that ignores the
    selection and emits one fixed panel fails: gpt-5-image-mini publishes no aspect ratio
    at all, Gemini publishes fourteen.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import open_webui.models.functions as functions_module

    from open_webui_openrouter_pipe.core.config import _OPENROUTER_IMAGE_GEN_FILTER_MARKER
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

    records = _recorded_endpoint(slug)
    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry.set_image_endpoints({stored: records})
    OpenRouterModelRegistry.register_image_models(
        [{"id": stored, "name": stored, "architecture": {"output_modalities": ["image"]}}]
    )

    class _Table:
        @staticmethod
        async def get_functions_by_type(kind, active_only=False):
            return [SimpleNamespace(id="or_image_gen", content=_OPENROUTER_IMAGE_GEN_FILTER_MARKER)]

        @staticmethod
        async def get_function_valves_by_id(function_id):
            return {"IMAGE_GENERATION_MODEL": stored}

    monkeypatch.setattr(functions_module, "Functions", _Table)
    pipe = MagicMock()
    manager = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    captured: dict = {}

    async def _record(**kwargs):
        captured.update(kwargs)
        return "or_image_gen"

    manager._ensure_filter_installed = _record

    await manager.ensure_openrouter_image_gen_filter_function_id()
    source = captured["desired_source"]

    assert f"default={stored!r}" in source, (
        f"the installed filter was not built for {stored}; the admin's selection never "
        "reached the renderer"
    )
    module = _load_filter_from_source(source, f"image_gen_selected_{slug.replace('.', '_')}")
    fields = set(module.Filter.UserValves.model_fields)
    published = {
        name
        for record in records
        for name in (record.get("supported_parameters") or {})
    }
    assert "IMAGE_ASPECT_RATIO" in fields, (
        f"{stored}: aspect_ratio is in OpenRouter's server-tool parameter table, so this "
        "panel carries it whatever the per-model images contract says -- that contract "
        "describes a different endpoint"
    )
    assert ("IMAGE_RESOLUTION" in fields) == ("resolution" in published), (
        f"{stored} publishes resolution={('resolution' in published)!r} and the panel drew "
        f"{'IMAGE_RESOLUTION' in fields!r}; the tier control tracks the model's own values"
    )
    assert stored in captured["desired_meta"]["description"], (
        "the admin list shows one line per filter; it must name the model this one draws with"
    )


@pytest.mark.asyncio
async def test_a_selection_the_catalog_does_not_carry_reads_differently_from_an_unread_one(
    monkeypatch,
):
    """`image_gen_filter_inputs` separates a wrong id from a sweep that has not run."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import open_webui.models.functions as functions_module

    from open_webui_openrouter_pipe.core.config import _OPENROUTER_IMAGE_GEN_FILTER_MARKER
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

    known = "openai/gpt-5-image-mini"
    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry.set_image_endpoints({})
    OpenRouterModelRegistry.register_image_models(
        [{"id": known, "name": known, "architecture": {"output_modalities": ["image"]}}]
    )

    selected = known

    class _Table:
        @staticmethod
        async def get_functions_by_type(kind, active_only=False):
            return [SimpleNamespace(id="or_image_gen", content=_OPENROUTER_IMAGE_GEN_FILTER_MARKER)]

        @staticmethod
        async def get_function_valves_by_id(function_id):
            return {"IMAGE_GENERATION_MODEL": selected}

    monkeypatch.setattr(functions_module, "Functions", _Table)
    pipe = MagicMock()
    manager = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())

    model_id, image_model, record, _transport = await manager.image_gen_filter_inputs()
    assert (model_id, record) == (known, None)
    assert isinstance(image_model, dict), (
        "the model is in the catalog and its contract is not; that is a read that has "
        "not landed, not a wrong id"
    )

    selected = "vendor/not-a-model"
    model_id, image_model, record, _transport = await manager.image_gen_filter_inputs()
    assert (model_id, image_model, record) == ("vendor/not-a-model", None, None)


@pytest.mark.parametrize(
    ("model_id", "spec_name", "expected_name"),
    [
        ("openai/gpt-5-image", None, "openai/gpt-5-image"),
        ("google/gemini-3-pro-image", "Gemini 3 Pro Image", "Gemini 3 Pro Image"),
    ],
)
@pytest.mark.asyncio
async def test_a_chat_catalog_selection_is_given_a_stand_in_model_record(
    monkeypatch, model_id, spec_name, expected_name
):
    """A model in the chat catalog carries no image record, so one is synthesised.

    Only `register_image_models` writes `image_model`. A multimodal text+image model
    lives in the chat catalog instead, so the selection resolves to a spec that has no
    such record, and the caller needs a name and an id to render the filter's heading
    with. Two rows, one whose spec names the model and one whose spec does not, so a
    constant cannot satisfy both sides of the fallback.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import open_webui.models.functions as functions_module

    from open_webui_openrouter_pipe.core.config import _OPENROUTER_IMAGE_GEN_FILTER_MARKER
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    norm_id = ModelFamily.base_model(sanitize_model_id(model_id))
    spec: dict[str, Any] = {
        "features": {"image_gen_tool"},
        "capabilities": {},
        "supported_parameters": frozenset(),
        "full_model": {"id": model_id},
        "architecture": {"output_modalities": ["image", "text"]},
    }
    if spec_name is not None:
        spec["name"] = spec_name
    OpenRouterModelRegistry._specs = {norm_id: spec}
    OpenRouterModelRegistry._id_map = {norm_id: model_id}
    OpenRouterModelRegistry._models = [
        {"id": sanitize_model_id(model_id), "norm_id": norm_id, "original_id": model_id}
    ]
    assert "image_model" not in OpenRouterModelRegistry.spec(model_id)

    class _Table:
        @staticmethod
        async def get_functions_by_type(kind, active_only=False):
            return [SimpleNamespace(id="or_image_gen", content=_OPENROUTER_IMAGE_GEN_FILTER_MARKER)]

        @staticmethod
        async def get_function_valves_by_id(function_id):
            return {"IMAGE_GENERATION_MODEL": model_id}

    monkeypatch.setattr(functions_module, "Functions", _Table)
    pipe = MagicMock()
    manager = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())

    resolved_id, image_model, _record, _dedicated = await manager.image_gen_filter_inputs()

    assert resolved_id == model_id
    assert image_model == {"id": model_id, "name": expected_name}


@pytest.mark.parametrize(
    "broken_source",
    ["def (:\n", "class Filter\n    pass\n"],
)
@pytest.mark.asyncio
async def test_an_unparseable_image_gen_filter_is_refused_rather_than_installed(
    monkeypatch, broken_source
):
    """A source that will not parse is raised on, not written into Open WebUI.

    Open WebUI executes a filter's content on load, so a row whose source cannot be
    parsed breaks the filter list for everything else installed. Two different broken
    sources, so the check cannot be a match against one string.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import open_webui.models.functions as functions_module

    from open_webui_openrouter_pipe.core.config import _OPENROUTER_IMAGE_GEN_FILTER_MARKER
    from open_webui_openrouter_pipe.filters import image_filter_renderer
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    model_id = "openai/gpt-5-image"
    OpenRouterModelRegistry.register_image_models(
        [{"id": model_id, "name": model_id, "architecture": {"output_modalities": ["image"]}}]
    )

    installed: list[Any] = []

    class _Table:
        @staticmethod
        async def get_functions_by_type(kind, active_only=False):
            return [SimpleNamespace(id="or_image_gen", content=_OPENROUTER_IMAGE_GEN_FILTER_MARKER)]

        @staticmethod
        async def get_function_valves_by_id(function_id):
            return {"IMAGE_GENERATION_MODEL": model_id}

        @staticmethod
        async def get_function_by_id(function_id):
            return None

        @staticmethod
        async def insert_new_function(user_id, function_type, form):
            installed.append(form)
            return form

        @staticmethod
        async def update_function_by_id(function_id, updates):
            installed.append(updates)
            return None

    monkeypatch.setattr(functions_module, "Functions", _Table)
    monkeypatch.setattr(
        image_filter_renderer,
        "render_image_gen_filter_source",
        lambda *_args, **_kwargs: broken_source,
    )

    pipe = MagicMock()
    manager = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())

    with pytest.raises(ValueError, match="Image Generation filter is invalid"):
        await manager.ensure_openrouter_image_gen_filter_function_id()

    assert installed == [], "a filter Open WebUI cannot parse was written anyway"


@pytest.mark.parametrize(
    ("model_id", "already_installed"),
    [
        ("acme/paintbox-1", True),
        ("acme/paintbox-1", False),
        ("zeta/inkwell-9", True),
        ("zeta/inkwell-9", False),
    ],
)
@pytest.mark.asyncio
async def test_a_contract_that_shrank_to_nothing_overwrites_the_filter_it_left_behind(
    monkeypatch, model_id, already_installed
):
    """A model that now publishes no knobs must not leave yesterday's controls on screen.

    A shrunk contract that was read has to overwrite the row it left behind -- otherwise
    the panel keeps writing values the model no longer accepts into every request -- and
    has to install one where none exists, because the panel carries controls of its own
    that the model's contract has no say over. Both starting states are driven here with
    two model ids, so the row the model ends with has to be built from that model's own
    spec rather than returned from a constant.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import open_webui.models.functions as functions_module

    from open_webui_openrouter_pipe.core.config import _OPENROUTER_IMAGE_FILTER_MARKER
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        ALWAYS_ON_VALVE_NAMES,
        build_image_model_filter_spec,
    )

    image_model = {"id": model_id, "name": model_id}
    spec = build_image_model_filter_spec(model_id, image_model, [], dedicated_image_api=True)
    assert (spec.knob_count, spec.contract_read) == (0, True)

    rows: dict[str, Any] = {}
    if already_installed:
        rows[spec.function_id] = SimpleNamespace(
            id=spec.function_id,
            name=spec.display_name,
            content=f"{_OPENROUTER_IMAGE_FILTER_MARKER}\nIMAGE_FILTER_MODEL_ID = {spec.model_id!r}\nclass Filter:\n    pass\n",
            meta=None,
            is_active=True,
            updated_at=1,
        )

    class _Table:
        @staticmethod
        async def get_functions_by_type(kind, active_only=False):
            return list(rows.values())

        @staticmethod
        async def get_function_by_id(function_id):
            return rows.get(function_id)

        @staticmethod
        async def insert_new_function(user_id, function_type, form):
            rows[form.id] = SimpleNamespace(
                id=form.id, name=form.name, content=form.content, meta=form.meta,
                is_active=False, is_global=False, updated_at=2,
            )
            return rows[form.id]

        @staticmethod
        async def update_function_by_id(function_id, updates):
            row = rows.get(function_id)
            if row is None:
                return None
            for key, value in dict(updates).items():
                setattr(row, key, value)
            return row

    monkeypatch.setattr(functions_module, "Functions", _Table)
    pipe = MagicMock()
    pipe.valves.AUTO_INSTALL_IMAGE_FILTERS = True
    manager = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())

    result = await manager._ensure_single_image_filter_function_id(
        model_id=model_id,
        image_model=image_model,
        endpoint_record=[],
        dedicated_image_api=True,
    )

    assert result == spec.function_id, (
        "the model ends without a panel of its own, so the controls the panel supplies "
        "are unreachable and any stale row stays on screen"
    )
    assert list(rows) == [spec.function_id], (
        f"the model must end with exactly one panel under its own id: {sorted(rows)}"
    )
    content = rows[spec.function_id].content
    assert f"IMAGE_FILTER_MODEL_ID = {spec.model_id!r}" in content
    for name in ALWAYS_ON_VALVE_NAMES:
        assert _declares_valve(content, name), (
            f"{name} is drawn on every image-API panel, and the stored row omits it"
        )


@pytest.mark.parametrize(
    "billed", [{"cost": 0.04, "total_tokens": 9}, {"cost": 0.11, "total_tokens": 21}]
)
@pytest.mark.asyncio
async def test_one_billed_generation_is_costed_once_even_when_the_client_disconnects(billed):
    """`asyncio.shield` finishes the write and still raises at the await.

    So the line after it never ran, the settle handler saw an uncosted outcome, and one
    billed image was recorded twice -- exactly when the database is slow enough for the
    cancel to land mid-write, which is when a duplicate costs the most. The cancel is
    delivered from inside the cost write, one seam below the ordering under test.
    """
    import asyncio as _asyncio
    from unittest.mock import MagicMock

    from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter

    snapshots: list[dict] = []

    class _Adapter(ImageGenerationAdapter):
        def __init__(self) -> None:
            self.logger = MagicMock()
            self._pipe = MagicMock()

        async def _record_cost(self, valves, usage, **_kw):
            snapshots.append(dict(usage))
            raise _asyncio.CancelledError

        async def _report_generation(self, *_a, **_kw):
            return None

    adapter = _Adapter()
    outcome: dict = {"usage": None, "costed": False, "reported": False}

    outcome["usage"] = billed
    outcome["costed"] = True
    try:
        await _asyncio.shield(
            adapter._record_cost(None, billed, user=None, metadata=None, user_obj=None,
                                 api_model_id="m/x")
        )
    except _asyncio.CancelledError:
        await adapter._settle(outcome, None, None, None, None, "m/x")

    assert len(snapshots) == 1, (
        f"one billed image produced {len(snapshots)} cost snapshots: {snapshots}"
    )
    assert snapshots[0] == billed


def test_the_cost_flag_is_claimed_before_the_write_it_describes():
    """The ordering is the property; a shielded write already claimed cannot double.

    Read from the source because the interleaving only shows up when a cancel lands
    between the two statements, which no test can schedule deterministically.
    """
    import ast
    import inspect
    import textwrap

    from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter

    body = textwrap.dedent(inspect.getsource(ImageGenerationAdapter._generate))
    generate = ast.parse(body).body[0]
    assert isinstance(generate, ast.AsyncFunctionDef), (
        f"_generate parsed as {type(generate).__name__}; the scan below reads the "
        "statement list of an async def"
    )
    statements = [ast.unparse(node) for node in generate.body]
    claim = next(i for i, line in enumerate(statements) if line == "outcome['costed'] = True")
    write = next(i for i, line in enumerate(statements) if "_record_cost" in line)
    assert claim < write, (
        "the flag that suppresses a second cost snapshot is set after the snapshot it "
        "describes, so a cancel at the await leaves it False and the settle path repeats it"
    )


# REGFIX: supersede_size_conflicts -- the tier half, and what OpenRouter says is legal


@pytest.mark.parametrize(
    ("size", "resolution", "dropped"),
    [("2K", "1K", True), ("2K", "2K", False), ("4K", "4k", False), ("1K", "4K", True)],
)
def test_a_tier_size_supersedes_only_a_resolution_that_disagrees_with_it(
    size, resolution, dropped
):
    """OpenRouter: a tier size "is equivalent to setting `resolution`".

    Equivalent, so naming both is legal when they agree -- dropping the agreeing one
    prints a note claiming a conflict the API does not have. Case differs in one row
    because `4k` and `4K` are one value.
    """
    from open_webui_openrouter_pipe.integrations.image_types import supersede_size_conflicts

    params = {"size": size, "resolution": resolution}
    notes = supersede_size_conflicts(params)

    assert bool(notes) is dropped
    assert ("resolution" in params) is not dropped
    if dropped:
        assert notes[0][0] == "resolution"
        assert notes[0][1] == resolution


@pytest.mark.parametrize(
    ("size", "resolution"), [("1024x1024", "2K"), ("1536x1024", "1K")]
)
def test_an_exact_size_supersedes_the_tier_beside_it_and_says_which_one_went(
    size, resolution
):
    """A pixel size and a tier both set the output dimensions, and the API takes one.

    Only the tier half of this rule was ever driven: with an exact size in the slot the
    branch that removes `resolution` never ran, so the request went out naming both and
    was rejected. The note has to name the value that was dropped or the user re-sends
    the same pair.

    Two sizes, one square and one not, so a rule keyed to a single shape fails.
    """
    from open_webui_openrouter_pipe.integrations.image_types import supersede_size_conflicts

    params = {"size": size, "resolution": resolution}
    dropped = supersede_size_conflicts(params)

    assert params == {"size": size}, f"the request still carries {sorted(params)}"
    assert [name for name, _value, _why in dropped] == ["resolution"]
    assert dropped[0][1] == resolution, (
        f"the note names {dropped[0][1]!r} rather than the value that was removed"
    )
    assert "dimensions" in dropped[0][2]


@pytest.mark.parametrize(
    ("size", "ratio", "kept"),
    [("1024x1024", "1:1", True), ("1024x1024", "16:9", False), ("1536x1024", "3:2", True)],
)
def test_an_exact_size_only_supersedes_an_aspect_ratio_it_contradicts(size, ratio, kept):
    """An agreeing ratio is not a conflict, and dropping it prints a note that is untrue."""
    from open_webui_openrouter_pipe.integrations.image_types import supersede_size_conflicts

    params = {"size": size, "aspect_ratio": ratio}
    dropped = supersede_size_conflicts(params)

    assert ("aspect_ratio" in params) is kept
    assert (dropped == []) is kept
    if not kept:
        assert dropped[0][0] == "aspect_ratio" and dropped[0][1] == ratio


@pytest.mark.parametrize("ratio", ["16:9", "1:4"])
def test_a_tier_size_never_conflicts_with_an_aspect_ratio(ratio):
    """OpenRouter: a tier "combines with `aspect_ratio`".

    Suppressing a working combination and blaming the model is the defect class this
    whole redesign exists to remove.
    """
    from open_webui_openrouter_pipe.integrations.image_types import supersede_size_conflicts

    params = {"size": "2K", "aspect_ratio": ratio}

    assert supersede_size_conflicts(params) == []
    assert params["aspect_ratio"] == ratio


def test_a_request_naming_no_size_is_left_exactly_as_the_user_built_it():
    from open_webui_openrouter_pipe.integrations.image_types import supersede_size_conflicts

    params = {"resolution": "2K", "aspect_ratio": "16:9"}

    assert supersede_size_conflicts(params) == []
    assert params == {"resolution": "2K", "aspect_ratio": "16:9"}


@pytest.mark.parametrize(
    ("slug", "model_id"), EVERY_CONTRACT, ids=[slug for slug, _ in EVERY_CONTRACT]
)
def test_the_drawing_model_note_describes_the_panel_that_model_actually_gets(slug, model_id):
    """The valve that picks the drawing model explains the panel; it has to match it.

    The sentence replaced said the settings offered were the ones that model publishes
    and that choosing another changes them. The set is fixed at six for every recorded
    contract, and not one of the forty publishes all six -- what a different model
    changes is whether each control lists that model's own values or the API's general
    ones. An admin reading the old sentence and finding a control the model never
    published goes looking for a bug that is not there.

    Both halves are measured against the RENDERED panel rather than against the
    sentence: which size control this model actually draws, and whether any control fell
    back to the API's own values. Sixteen contracts draw Resolution and twenty-four draw
    Output size, so a note naming one of them regardless fails on the other family.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_gen_tool_spec,
        build_image_model_filter_spec,
        image_gen_model_note,
        render_image_gen_filter_source,
    )

    records = _recorded_endpoint(slug)
    spec = build_image_model_filter_spec(
        model_id, {"id": model_id}, records, dedicated_image_api=True
    )
    assert spec.has_knobs, (
        f"{model_id} publishes nothing, so this drives one of the empty branches rather "
        "than the one under test"
    )

    body = (
        render_image_gen_filter_source(spec, catalog_match=True, selected_model=model_id)
        .split("class UserValves(BaseModel):", 1)[1]
        .split("    def __init__", 1)[0]
    )
    titles = [
        found.strip("'\"") for found in re.findall(r"^                    title=(.+),$", body, re.M)
    ]
    drawn, alternative = (
        ("Resolution", "Output size")
        if "Resolution" in titles
        else ("Output size", "Resolution")
    )
    assert drawn in titles and alternative not in titles, (
        f"{model_id} draws {titles}: exactly one of the two size controls must appear"
    )

    note = image_gen_model_note(spec, catalog_match=True)
    assert drawn in note and alternative in note, (
        f"the note names neither the size control {model_id} gets nor the one it does "
        f"not, so an admin cannot tell why the panel moved: {note!r}"
    )
    assert note.index(drawn) < note.index(alternative), (
        f"the note leads with {alternative}, which this model does not draw: {note!r}"
    )

    unnarrowed = set(build_image_gen_tool_spec(spec).schema_only)
    assert unnarrowed, (
        f"{model_id} narrows all six controls to its own values, which no recorded "
        "contract did when this was written -- the note's fallback clause is now unproven"
    )
    assert "in general" in note, (
        f"{model_id} draws {sorted(unnarrowed)} with the image API's own values rather "
        f"than its own, and the note does not say so: {note!r}"
    )


@pytest.mark.parametrize(
    ("name", "published"),
    [("aspect_ratio", True), ("not_a_knob", False)],
    ids=["a-knob-the-panel-draws", "a-key-only-the-client-sent"],
)
def test_a_setting_no_panel_draws_is_named_by_itself_and_not_by_an_empty_title(
    name, published
):
    """``image_config`` arrives from the client, so a key need not be one of the nine.

    Every note about a dropped or clamped setting is written through this one labeller,
    and an unknown key has no title. Handing the titled shape back regardless renders the
    note as " (whatever_they_sent) is not offered by this model" -- opening on a space and
    an empty pair of parentheses, with nothing between them to say what the parentheses
    are qualifying.

    One name the panel draws and one it does not, so the two expected shapes differ: a
    labeller that always returns the bare name fails the first row, and one that always
    returns the titled form fails the second. The titled expectation is read out of
    ``IMAGE_KNOB_TITLES`` rather than written here, so a retitled control cannot leave
    this asserting wording that no longer ships. The function is pure; nothing is stubbed.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import IMAGE_KNOB_TITLES
    from open_webui_openrouter_pipe.integrations.image import _labelled

    assert (name in IMAGE_KNOB_TITLES) is published, (
        f"precondition: {name!r} has to be {'a published knob' if published else 'absent'}"
    )
    expected = f"{IMAGE_KNOB_TITLES[name][0]} ({name})" if published else name

    assert _labelled(name) == expected, (
        f"a note about {name!r} would read {_labelled(name)!r}"
    )
