from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import aiohttp
import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe.filters.filter_manager import FilterManager
from open_webui_openrouter_pipe.filters.router_filter_renderer import (
    build_router_filter_spec,
    render_router_filter_source,
)
from open_webui_openrouter_pipe.integrations import router_catalog
from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

AUTO_MODEL = "open_webui_openrouter_pipe.openrouter.auto"
SWITCHYARD_MODEL = "open_webui_openrouter_pipe.nvidia.switchyard"
SCHEMA_URL = "https://openrouter.ai/openapi.json"


def invented_schema(cost_tiers: tuple[str, ...] = ("thrifty", "lavish")) -> dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "components": {
            "schemas": {
                "InventedAutoRouter": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": {"type": "string", "enum": ["auto-router"]},
                        "enabled": {"type": "boolean", "description": "Switches the router off."},
                        "pin_model": {"type": "boolean", "description": "Reuses the last model."},
                        "old_knob": {"type": "integer", "deprecated": True, "minimum": 0, "maximum": 10},
                        "allowed_models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Patterns the router may pick from. Copy them from config before use.",
                        },
                        "cost_tier": {
                            "type": "string",
                            "enum": list(cost_tiers),
                            "description": "How much the router may spend.",
                        },
                        "ceiling": {
                            "anyOf": [{"type": "number", "minimum": 0, "maximum": 1}, {"type": "null"}],
                            "description": "A share between zero and one.",
                        },
                        "extras": {"type": "object", "description": "Free-form extras."},
                    },
                },
                "InventedSwitchyard": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": {"type": "string", "enum": ["switchyard-router"]},
                        "algorithm": {"type": "string", "enum": ["stage", "passthrough"]},
                    },
                },
            }
        },
    }


def _render(model_id: str, plugin_id: str, name: str) -> tuple[str, ModuleType]:
    schemas = invented_schema()["components"]["schemas"]
    schema = next(item for item in schemas.values() if item["properties"]["id"]["enum"] == [plugin_id])
    options, _skipped = router_catalog.derive_router_options(plugin_id, schema, schemas)
    source = render_router_filter_source(build_router_filter_spec(model_id, plugin_id, name, options))
    module = ModuleType(f"rendered_{plugin_id.replace('-', '_')}")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, f"<{module.__name__}>", "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    return source, module


@pytest.fixture(scope="module")
def rendered() -> SimpleNamespace:
    auto_source, auto = _render("openrouter/auto", "auto-router", "Auto Router")
    switchyard_source, switchyard = _render("nvidia/switchyard", "switchyard-router", "NVIDIA: Switchyard")
    return SimpleNamespace(
        auto_source=auto_source, auto=auto, switchyard_source=switchyard_source, switchyard=switchyard
    )


def _run(module: ModuleType, model: str, admin: dict[str, Any], user: dict[str, Any] | None = None) -> dict[str, Any]:
    router = module.Filter()
    router.valves = module.Filter.Valves(**admin)
    person = {"valves": module.Filter.UserValves(**(user or {}))}
    return router.inlet({"model": model, "messages": []}, __user__=person)


def test_the_rendered_settings_offer_every_current_option_and_survive_open_webuis_import_rewrite(rendered):
    offered = {"ALLOWED_MODELS", "COST_TIER", "CEILING"}
    assert set(rendered.auto.Filter.Valves.model_fields) == {
        "priority",
        *offered,
        *(f"ALLOW_USER_{name}" for name in offered),
    }
    assert set(rendered.auto.Filter.UserValves.model_fields) == offered
    assert "from config" not in rendered.auto_source
    accepted, error = FilterManager.validate_filter_source(rendered.auto_source)
    assert accepted, error


def test_admin_defaults_reach_the_plugin_entry_and_a_user_value_counts_only_where_allowed(rendered):
    assert _run(rendered.auto, AUTO_MODEL, {}) == {"model": AUTO_MODEL, "messages": []}

    admin = {"ALLOWED_MODELS": "acme/*, acme/*, other/x", "COST_TIER": "thrifty", "CEILING": "0.5"}
    entry = {"id": "auto-router", "allowed_models": ["acme/*", "other/x"], "cost_tier": "thrifty", "ceiling": 0.5}
    assert _run(rendered.auto, AUTO_MODEL, admin)["plugins"] == [entry]

    assert _run(rendered.auto, AUTO_MODEL, admin, {"COST_TIER": "lavish"})["plugins"] == [entry]
    allowed = {**admin, "ALLOW_USER_COST_TIER": True}
    assert _run(rendered.auto, AUTO_MODEL, allowed, {"COST_TIER": "lavish", "CEILING": 0.9})["plugins"] == [
        {**entry, "cost_tier": "lavish"}
    ]

    assert "plugins" not in _run(rendered.auto, "open_webui_openrouter_pipe.openrouter.auto-beta", admin)


def test_a_stored_value_the_schema_no_longer_lists_is_dropped_and_the_rest_kept(rendered):
    stored = {"COST_TIER": "gone", "ALLOWED_MODELS": "acme/*", "CEILING": 7, "REMOVED_OPTION": "x"}
    for form in (rendered.auto.Filter.Valves, rendered.auto.Filter.UserValves):
        kept = form(**stored)
        assert (kept.COST_TIER, kept.ALLOWED_MODELS, kept.CEILING) == ("", "acme/*", None)


def test_the_switchyard_candidates_travel_as_the_fallback_list_not_inside_the_plugin_entry(rendered):
    body = _run(rendered.switchyard, SWITCHYARD_MODEL, {"ALGORITHM": "passthrough", "MODELS": "acme/one, acme/two"})
    assert body["plugins"] == [{"id": "switchyard-router", "algorithm": "passthrough"}]
    assert body["model_fallback"] == "acme/one,acme/two"

    only_candidates = _run(rendered.switchyard, SWITCHYARD_MODEL, {"MODELS": "acme/one"})
    assert only_candidates["model_fallback"] == "acme/one"
    assert "plugins" not in only_candidates


@pytest.mark.asyncio
async def test_a_failed_refresh_keeps_the_options_the_last_good_read_produced(caplog):
    valves = SimpleNamespace(
        AUTO_INSTALL_ROUTER_FILTERS=True, AUTO_ATTACH_ROUTER_FILTERS=True, HTTP_REFERER_OVERRIDE=""
    )
    logger = logging.getLogger("test_router_catalog")

    async def read(**reply: Any) -> None:
        async with aiohttp.ClientSession() as session:
            with aioresponses() as mocked:
                mocked.get(SCHEMA_URL, **reply)
                await router_catalog.ensure_router_catalog_loaded(
                    session, valves=valves, logger=logger, cache_seconds=0
                )

    await read(payload=invented_schema())
    options = router_catalog.router_options_for("auto-router")
    stamp = OpenRouterModelRegistry.content_stamp()
    assert options is not None and [option.name for option in options] == [
        "allowed_models",
        "cost_tier",
        "ceiling",
    ]

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="test_router_catalog"):
        await read(exception=aiohttp.ClientConnectionError("offline"))
    assert router_catalog.router_options_for("auto-router") is options
    assert OpenRouterModelRegistry.content_stamp() == stamp
    assert [record.levelno for record in caplog.records if record.name == "test_router_catalog"] == [
        logging.WARNING
    ]

    await read(payload=invented_schema(("thrifty", "lavish", "balanced")))
    assert OpenRouterModelRegistry.content_stamp() != stamp
