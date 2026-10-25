from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import threading
import time
import weakref
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import aiohttp

from ..core.config import _OPENROUTER_OPENAPI_URL, openrouter_attribution_headers
from ..core.warn_latch import warn_level
from ..models.registry import OpenRouterModelRegistry, _failure_backoff_window
from .image_types import RENDERABLE_FIELD_NAME_RE

ROUTER_PLUGIN_BY_MODEL: dict[str, str] = {
    "openrouter/auto": "auto-router",
    "openrouter/auto-beta": "auto-beta-router",
    "openrouter/pareto-code": "pareto-router",
    "nvidia/switchyard": "switchyard-router",
    "typesafe/jev-router": "jev-router",
}

ROUTER_OPTIONS_NOT_OFFERED: frozenset[str] = frozenset({"id", "enabled", "pin_model"})

ROUTER_KIND_ENUM = "enum"
ROUTER_KIND_TEXT = "text"
ROUTER_KIND_LIST = "list"
ROUTER_KIND_TOGGLE = "toggle"
ROUTER_KIND_INT = "int"
ROUTER_KIND_FLOAT = "float"

ROUTER_USER_FLAG_PREFIX = "ALLOW_USER_"

ROUTER_REWRITE_HAZARD_RE = re.compile(r"from (utils|apps|main|config)")

_ROUTER_CATALOG_TIMEOUT_SECONDS = 15
_ROUTER_REF_DEPTH = 4


@dataclass(frozen=True, slots=True)
class RouterOption:
    name: str
    field: str
    kind: str
    title: str
    description: str
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: float | None = None
    exclusive_maximum: float | None = None
    send_as: str = ""


ROUTER_REQUEST_FIELDS: dict[str, tuple[RouterOption, ...]] = {
    "switchyard-router": (
        RouterOption(
            name="models",
            field="MODELS",
            kind=ROUTER_KIND_LIST,
            title="Models",
            description=(
                "The models Switchyard chooses between, separated by commas. "
                "OpenRouter's Switchyard guide limits how many."
            ),
            send_as="model_fallback",
        ),
    ),
}

_warned_router_catalog: set[str] = set()

_ROUTER_CATALOG_LOCK = asyncio.Lock()
_router_catalog_lock_guard = threading.Lock()
_router_catalog_locks: weakref.WeakKeyDictionary[Any, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)

_router_options_store: dict[str, tuple[RouterOption, ...]] | None = None
_router_last_attempt = 0.0
_router_failure_count = 0


def router_plugin_for_model(model: Mapping[str, Any]) -> str | None:
    for key in ("original_id", "id"):
        candidate = model.get(key)
        if isinstance(candidate, str) and candidate in ROUTER_PLUGIN_BY_MODEL:
            return ROUTER_PLUGIN_BY_MODEL[candidate]
    return None


def router_catalog_known() -> bool:
    return _router_options_store is not None


def router_options_for(plugin_id: str) -> tuple[RouterOption, ...] | None:
    if _router_options_store is None:
        return None
    return _router_options_store.get(plugin_id)


def reset_router_catalog() -> None:
    global _router_options_store, _router_last_attempt, _router_failure_count
    _router_options_store = None
    _router_last_attempt = 0.0
    _router_failure_count = 0
    _warned_router_catalog.clear()
    OpenRouterModelRegistry.register_router_digest("")


def router_settings_wanted(valves: Any) -> bool:
    return bool(
        getattr(valves, "AUTO_INSTALL_ROUTER_FILTERS", False)
        or getattr(valves, "AUTO_ATTACH_ROUTER_FILTERS", False)
    )


def _resolve_ref(node: Any, components: Mapping[str, Any], depth: int = 0) -> Any:
    if isinstance(node, dict) and isinstance(node.get("$ref"), str) and depth < _ROUTER_REF_DEPTH:
        name = node["$ref"].rsplit("/", 1)[-1]
        return _resolve_ref(components.get(name), components, depth + 1)
    return node


def _only_non_null(
    node: Any, components: Mapping[str, Any], depth: int = 0
) -> dict[str, Any] | None:
    node = _resolve_ref(node, components)
    if not isinstance(node, dict) or depth >= _ROUTER_REF_DEPTH:
        return None
    for key in ("anyOf", "oneOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            live = [
                resolved
                for resolved in (_resolve_ref(variant, components) for variant in variants)
                if not (isinstance(resolved, dict) and resolved.get("type") == "null")
            ]
            if len(live) != 1:
                return None
            return _only_non_null(live[0], components, depth + 1)
    kind = node.get("type")
    if isinstance(kind, list):
        kinds = [item for item in kind if item != "null"]
        if len(kinds) != 1:
            return None
        node = {**node, "type": kinds[0]}
    return node


def _schema_bound(node: Mapping[str, Any], key: str) -> float | None:
    value = node.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _option_title(name: str) -> str:
    words = name.replace("_", " ").strip()
    return words[:1].upper() + words[1:]


def _option_description(raw: Any, node: Mapping[str, Any]) -> str:
    for source in (raw, node):
        if isinstance(source, dict) and isinstance(source.get("description"), str):
            return " ".join(source["description"].split())
    return ""


def _option_from_node(
    name: str, raw: Any, node: Mapping[str, Any], components: Mapping[str, Any]
) -> RouterOption | str:
    common: dict[str, Any] = {
        "name": name,
        "field": name.upper(),
        "title": _option_title(name),
        "description": _option_description(raw, node),
    }
    kind = node.get("type")
    if kind == "string":
        values = node.get("enum")
        if values is None:
            return RouterOption(kind=ROUTER_KIND_TEXT, **common)
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value and not ROUTER_REWRITE_HAZARD_RE.search(value)
            for value in values
        ):
            return "a list of choices this pipe cannot offer"
        return RouterOption(kind=ROUTER_KIND_ENUM, choices=tuple(dict.fromkeys(values)), **common)
    if kind in ("integer", "number"):
        return RouterOption(
            kind=ROUTER_KIND_INT if kind == "integer" else ROUTER_KIND_FLOAT,
            minimum=_schema_bound(node, "minimum"),
            maximum=_schema_bound(node, "maximum"),
            exclusive_minimum=_schema_bound(node, "exclusiveMinimum"),
            exclusive_maximum=_schema_bound(node, "exclusiveMaximum"),
            **common,
        )
    if kind == "boolean":
        return RouterOption(kind=ROUTER_KIND_TOGGLE, **common)
    if kind == "array":
        items = _only_non_null(node.get("items"), components)
        if isinstance(items, dict) and items.get("type") == "string":
            return RouterOption(kind=ROUTER_KIND_LIST, **common)
        return "a list of something other than text"
    return f"a value of type {kind!r}"


def derive_router_options(
    plugin_id: str,
    schema: Mapping[str, Any],
    components: Mapping[str, Any],
) -> tuple[tuple[RouterOption, ...], tuple[tuple[str, str], ...]]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return (), ()
    options: list[RouterOption] = []
    skipped: list[tuple[str, str]] = []
    taken: set[str] = set()
    for name, raw in properties.items():
        if not isinstance(name, str) or name in ROUTER_OPTIONS_NOT_OFFERED:
            continue
        flagged = (raw, _resolve_ref(raw, components))
        if any(isinstance(node, dict) and node.get("deprecated") is True for node in flagged):
            continue
        node = _only_non_null(raw, components)
        if node is None:
            skipped.append((name, "a shape this pipe cannot offer as a control"))
            continue
        if node.get("deprecated") is True:
            continue
        if not RENDERABLE_FIELD_NAME_RE.fullmatch(name):
            skipped.append((name, "a name that cannot become a control"))
            continue
        resolved = _option_from_node(name, raw, node, components)
        if isinstance(resolved, str):
            skipped.append((name, resolved))
            continue
        if resolved.field in taken or resolved.field.startswith(ROUTER_USER_FLAG_PREFIX):
            skipped.append((name, "a name another control already uses"))
            continue
        taken.add(resolved.field)
        options.append(resolved)
    for extra in ROUTER_REQUEST_FIELDS.get(plugin_id, ()):
        if extra.field not in taken:
            options.append(extra)
            taken.add(extra.field)
    return tuple(options), tuple(skipped)


def _plugin_schemas(schemas: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for schema in schemas.values():
        properties = schema.get("properties") if isinstance(schema, dict) else None
        id_property = properties.get("id") if isinstance(properties, dict) else None
        ids = id_property.get("enum") if isinstance(id_property, dict) else None
        if not isinstance(ids, list):
            continue
        for plugin_id in ids:
            if isinstance(plugin_id, str):
                found.setdefault(plugin_id, schema)
    return found


def _first_time_level(latch: set[str], cause: str) -> int:
    if cause in latch:
        return logging.DEBUG
    latch.add(cause)
    return logging.INFO


def parse_router_document(
    document: Any, logger: Any
) -> dict[str, tuple[RouterOption, ...]] | None:
    components = document.get("components") if isinstance(document, dict) else None
    schemas = components.get("schemas") if isinstance(components, dict) else None
    if not isinstance(schemas, dict):
        return None
    plugin_schemas = _plugin_schemas(schemas)
    paired = set(ROUTER_PLUGIN_BY_MODEL.values())
    if not paired & set(plugin_schemas):
        return None
    for plugin_id in sorted(plugin_schemas):
        if plugin_id.endswith("-router") and plugin_id not in paired:
            logger.log(
                _first_time_level(_warned_router_catalog, f"unpaired:{plugin_id}"),
                "OpenRouter publishes settings for a router plugin, %r, that this pipe has no "
                "model for, so it is left out.",
                plugin_id,
            )
    parsed: dict[str, tuple[RouterOption, ...]] = {}
    for plugin_id in sorted(paired & set(plugin_schemas)):
        required = plugin_schemas[plugin_id].get("required")
        unmet = sorted(
            str(name) for name in (required if isinstance(required, list) else []) if name != "id"
        )
        if unmet:
            logger.log(
                warn_level(_warned_router_catalog, f"required:{plugin_id}"),
                "OpenRouter requires %s in the settings of the %r router, which this pipe cannot "
                "offer as an optional control, so that router is left without settings.",
                ", ".join(unmet),
                plugin_id,
            )
            continue
        options, skipped = derive_router_options(plugin_id, plugin_schemas[plugin_id], schemas)
        for name, reason in skipped:
            logger.log(
                warn_level(_warned_router_catalog, f"option:{plugin_id}:{name}"),
                "OpenRouter publishes the %r setting of the %r router as %s, so it is left out "
                "of that router's settings.",
                name,
                plugin_id,
                reason,
            )
        parsed[plugin_id] = options
    return parsed


def _options_digest_of(parsed: Mapping[str, tuple[RouterOption, ...]]) -> str:
    material = json.dumps(
        {plugin_id: [asdict(option) for option in options] for plugin_id, options in parsed.items()},
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("ascii")).hexdigest()[:32]


def _current_router_catalog_lock() -> asyncio.Lock:
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return _ROUTER_CATALOG_LOCK
    existing = _router_catalog_locks.get(running)
    if existing is not None:
        return existing
    with _router_catalog_lock_guard:
        existing = _router_catalog_locks.get(running)
        if existing is None:
            existing = asyncio.Lock()
            _router_catalog_locks[running] = existing
    return existing


def _router_catalog_stale(cache_seconds: int) -> bool:
    if not _router_last_attempt:
        return True
    window = _failure_backoff_window(cache_seconds, _router_failure_count)
    return (time.time() - _router_last_attempt) >= window


def _record_router_attempt(*, succeeded: bool) -> None:
    global _router_last_attempt, _router_failure_count
    _router_last_attempt = time.time()
    _router_failure_count = 0 if succeeded else _router_failure_count + 1


def _store_router_options(parsed: Mapping[str, tuple[RouterOption, ...]]) -> None:
    global _router_options_store
    _router_options_store = dict(parsed)
    OpenRouterModelRegistry.register_router_digest(_options_digest_of(parsed))


async def ensure_router_catalog_loaded(
    session: aiohttp.ClientSession,
    *,
    valves: Any,
    logger: Any,
    cache_seconds: int,
) -> None:
    if not router_settings_wanted(valves) or not _router_catalog_stale(cache_seconds):
        return
    async with _current_router_catalog_lock():
        if not _router_catalog_stale(cache_seconds):
            return
        try:
            async with session.get(
                _OPENROUTER_OPENAPI_URL,
                headers=openrouter_attribution_headers(valves),
                timeout=aiohttp.ClientTimeout(total=_ROUTER_CATALOG_TIMEOUT_SECONDS),
            ) as resp:
                resp.raise_for_status()
                document = await resp.json(content_type=None)
        except (TimeoutError, aiohttp.ClientError, OSError, ValueError) as exc:
            _record_router_attempt(succeeded=False)
            logger.log(
                warn_level(_warned_router_catalog, f"fetch:{type(exc).__name__}"),
                "OpenRouter's published request schema could not be read (%s): router settings "
                "keep the options they last had, and the read is tried again.",
                exc,
            )
            return
        parsed = parse_router_document(document, logger)
        if parsed is None:
            _record_router_attempt(succeeded=False)
            logger.log(
                warn_level(_warned_router_catalog, "unrecognised"),
                "OpenRouter's published request schema does not list the router settings this "
                "pipe looks for: router settings keep the options they last had, and the read is "
                "tried again.",
            )
            return
        _store_router_options(parsed)
        _record_router_attempt(succeeded=True)
        logger.info(
            "Read the settings of %d OpenRouter router(s) from OpenRouter's published request "
            "schema.",
            len(parsed),
        )
