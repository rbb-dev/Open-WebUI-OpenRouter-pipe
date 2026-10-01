"""Tool registry building and collision resolution.

This module handles tool spec management:
- _dedupe_tools: Remove duplicate tool definitions
- _build_collision_safe_tool_specs_and_registry: Handle name collisions

Ensures collision-safe tool names and builds execution registry for dispatcher.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import re
from typing import TYPE_CHECKING, Any

from ..core.config import _EMPTY_TOOL_SCHEMA, _PIPE_METADATA_KEY
from ..core.timing_logger import timed
from ..storage.owui_files import is_linkable_chat

# Import tool schema functions
from .tool_schema import (
    _declared_parameter_names,
    _strictify_schema,
)

# Import runtime dependencies
if TYPE_CHECKING:
    from ..api.transforms import ResponsesBody
    from ..core.config import Valves
    from ..pipe import Pipe
else:
    # At runtime, import these to avoid circular import issues
    try:
        from ..api.transforms import ResponsesBody
    except ImportError:
        ResponsesBody = Any  # type: ignore
    try:
        from ..core.config import Valves
    except ImportError:
        Valves = Any  # type: ignore


_module_logger = logging.getLogger(__name__)



def open_webui_runs_the_calls(valves: Pipe.Valves, metadata: Any, *, stream: bool) -> bool:
    meta = metadata if isinstance(metadata, dict) else {}
    pipe_meta = meta.get(_PIPE_METADATA_KEY)
    if isinstance(pipe_meta, dict) and pipe_meta.get("fusion_inner"):
        return False
    params = meta.get("params")
    approval_mode = params.get("tool_approval_mode") if isinstance(params, dict) else None
    if (
        approval_mode == "ask"
        and is_linkable_chat(meta.get("chat_id"))
        and meta.get("message_id")
        and stream
    ):
        return True
    return valves.TOOL_EXECUTION_MODE == "Open-WebUI" and bool(stream)


def _dedupe_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """(Internal) Deduplicate a tool list with simple, stable identity keys.

    Identity:
      - Function tools -> key = ("function", <name>)
      - Non-function tools -> key = (<type>, None)

    Later entries win (last write wins).

    Args:
        tools: List of tool dicts (OpenAI Responses schema).

    Returns:
        list: Deduplicated list, preserving only the last occurrence per identity.
    """
    if not tools:
        return []
    canonical: dict[tuple, dict[str, Any]] = {}
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function":
            key = ("function", t.get("name"))
        else:
            key = (t.get("type"), None)
        if key[0]:
            canonical[key] = t
    return list(canonical.values())


def _advertise_strict(spec: dict[str, Any], source: dict[str, Any], *, strictify: bool) -> None:
    if strictify:
        spec["strict"] = True
    elif "strict" in source:
        spec["strict"] = source["strict"]


def _normalize_responses_function_tool_spec(tool: Any, *, strictify: bool) -> dict[str, Any] | None:
    """Return a normalized Responses-style function tool spec, or None when invalid."""
    if not isinstance(tool, dict):
        return None
    if tool.get("type") != "function":
        return None
    name = tool.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    spec: dict[str, Any] = {"type": "function", "name": name.strip()}
    description = tool.get("description")
    if isinstance(description, str) and description.strip():
        spec["description"] = description.strip()
    parameters = tool.get("parameters")
    if isinstance(parameters, dict):
        spec["parameters"] = _strictify_schema(parameters) if strictify else parameters
    else:
        spec["parameters"] = _EMPTY_TOOL_SCHEMA
    if isinstance(tool.get("cache_control"), dict):
        spec["cache_control"] = tool["cache_control"]
    _advertise_strict(spec, tool, strictify=strictify)
    return spec


def _responses_spec_from_owui_tool_cfg(tool_cfg: dict[str, Any], *, strictify: bool) -> dict[str, Any] | None:
    """Return a Responses-style function tool spec from an OWUI tool registry entry."""
    if not isinstance(tool_cfg, dict):
        return None
    spec = tool_cfg.get("spec")
    if not isinstance(spec, dict):
        return None
    name = spec.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    params = spec.get("parameters") or {"type": "object", "properties": {}}
    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}}
    out: dict[str, Any] = {
        "type": "function",
        "name": name.strip(),
        "description": spec.get("description") or name.strip(),
        "parameters": _strictify_schema(params) if strictify else params,
    }
    _advertise_strict(out, spec, strictify=strictify)
    return out


def _bound_description_and_parameters(bound: dict[str, Any] | None) -> dict[str, Any] | None:
    if not bound:
        return None
    return {"description": bound["description"], "parameters": bound["parameters"]}


def _tool_prefix_for_collision(source: str, tool_cfg: dict[str, Any] | None) -> str:
    """Return the prefix to apply when collision renaming is required."""
    if source == "owui_request_tools":
        return "owui__"
    if source == "direct_tool_server":
        return "direct__"
    if source == "extra_tools":
        return "extra__"
    # Registry tools (tool_ids / extensions).
    if tool_cfg and bool(tool_cfg.get("direct")):
        return "direct__"
    return "tool__"


_PROVIDER_TOOL_NAME_MAX = 64
_NOT_IN_PROVIDER_TOOL_NAME = re.compile(r"[^A-Za-z0-9_-]")


def _provider_tool_name(wanted: str, digest: str, used_names: set[str]) -> str:
    safe = _NOT_IN_PROVIDER_TOOL_NAME.sub("_", wanted)
    if safe == wanted and len(wanted) <= _PROVIDER_TOOL_NAME_MAX and wanted not in used_names:
        return wanted
    base = f"{safe[: _PROVIDER_TOOL_NAME_MAX - len(digest) - 2]}__{digest}"
    if base not in used_names:
        return base
    for ordinal in itertools.count(2):
        suffix = f"_{ordinal}"
        candidate = f"{base[: _PROVIDER_TOOL_NAME_MAX - len(suffix)]}{suffix}"
        if candidate not in used_names:
            return candidate
    raise AssertionError("unreachable")


def _advertised_wire_is_admissible(cfg_spec: Any, advertised_from: Any, advertised: Any) -> bool:
    if not isinstance(cfg_spec, dict) or not isinstance(advertised, dict):
        return False
    entry_params = cfg_spec.get("parameters")
    if not isinstance(entry_params, dict) or not isinstance(advertised_from, dict):
        return False
    advertised_params = advertised.get("parameters")
    if not isinstance(advertised_params, dict):
        return False
    return advertised_from == entry_params and isinstance(advertised_params.get("properties"), dict)


def _advertised_names_for_replayed_calls(items: Any, exposed_to_origin: dict[str, str] | None) -> None:
    if not isinstance(items, list):
        return
    advertised = exposed_to_origin or {}
    exposed_by_origin: dict[str, str] = {}
    shared_origins: set[str] = set()
    for exposed, origin in advertised.items():
        if exposed_by_origin.setdefault(origin, exposed) != exposed:
            shared_origins.add(origin)
    for item in items:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name or name in advertised:
            continue
        if name in exposed_by_origin and name not in shared_origins:
            item["name"] = exposed_by_origin[name]
        else:
            digest = hashlib.sha1(f"replayed::{name}".encode()).hexdigest()[:8]
            reserved = {name} if name in shared_origins else set()
            item["name"] = _provider_tool_name(name, digest, reserved)



OWUI_OWNS_KEY = "owui_owns"


def _owui_owned_entry(origin_source: str, origin_name: str, exposed_name: str) -> dict[str, Any]:
    return {
        "origin_source": origin_source,
        "origin_name": origin_name,
        "exposed_name": exposed_name,
        OWUI_OWNS_KEY: True,
    }


@timed
def _build_collision_safe_tool_specs_and_registry(
    *,
    request_tool_specs: list[dict[str, Any]] | None,
    owui_registry: dict[str, dict[str, Any]] | None,
    direct_registry: dict[str, dict[str, Any]] | None,
    builtin_registry: dict[str, dict[str, Any]] | None,
    extra_tools: list[dict[str, Any]] | None,
    strictify: bool,
    owui_tool_passthrough: bool,
    logger: logging.Logger | None,
    builtin_ask_user_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    """Build collision-safe tool specs and an execution registry.

    Returns:
      - tools: Responses-style tool specs with collision-safe names.
      - exec_registry: mapping exposed_name -> OWUI tool cfg dict (callable/spec/etc).
      - exposed_to_origin: mapping exposed_name -> origin tool name (for passthrough execution).
    """
    from .tool_executor import is_builtin_ask_user, is_owui_builtin

    log = logger or _module_logger
    request_tool_specs = request_tool_specs or []
    extra_tools = extra_tools or []
    owui_registry = owui_registry or {}
    direct_registry = direct_registry or {}
    builtin_registry = builtin_registry or {}

    # Normalize all registries into lists so we can preserve collisions.
    owui_entries: list[tuple[str, dict[str, Any]]] = [
        (key, entry) for key, entry in owui_registry.items() if isinstance(entry, dict)
    ]
    direct_entries: list[dict[str, Any]] = [
        entry for entry in direct_registry.values() if isinstance(entry, dict)
    ]
    builtin_entries: list[dict[str, Any]] = [
        entry for entry in builtin_registry.values() if isinstance(entry, dict)
    ]

    def _indexed_name(entry: dict[str, Any]) -> str | None:
        _spec = entry.get("spec")
        if not isinstance(_spec, dict):
            return None
        _name = _spec.get("name")
        return _name if isinstance(_name, str) else None

    _builtin_index: dict[str, dict[str, Any]] = {}
    for _e in builtin_entries:
        _name = _indexed_name(_e)
        if _name is not None:
            _builtin_index.setdefault(_name, _e)
    _owui_index: dict[str, dict[str, Any]] = {}
    _owui_name_counts: dict[str, int] = {}
    for _, _e in owui_entries:
        _name = _indexed_name(_e)
        if _name is not None:
            _owui_index.setdefault(_name, _e)
            _owui_name_counts[_name] = _owui_name_counts.get(_name, 0) + 1
    _direct_index: dict[str, dict[str, Any]] = {}
    for _e in direct_entries:
        _name = _indexed_name(_e)
        if _name is not None:
            _direct_index.setdefault(_name, _e)

    def _pick_executor(name: str, *, prefer: str | None = None) -> dict[str, Any] | None:
        if prefer == "builtin" and name in _builtin_index:
            return _builtin_index[name]
        if prefer == "owui" and name in _owui_index:
            return _owui_index[name]
        if prefer == "direct" and name in _direct_index:
            return _direct_index[name]
        if name in _builtin_index:
            return _builtin_index[name]
        if name in _owui_index:
            return _owui_index[name]
        return _direct_index.get(name)

    def _registry_tools_named(name: str) -> int:
        return _owui_name_counts.get(name, 0)

    _bound_specs: dict[int, dict[str, Any] | None] = {}

    def _bound_spec(tool_cfg: Any) -> dict[str, Any] | None:
        if not isinstance(tool_cfg, dict):
            return None
        key = id(tool_cfg)
        if key not in _bound_specs:
            built = _responses_spec_from_owui_tool_cfg(tool_cfg, strictify=False)
            _bound_specs[key] = dict(built) if built is not None else None
        return _bound_specs[key]

    candidates: list[dict[str, Any]] = []
    resolved_request_names: set[str] = set()

    # 1) Request-provided tool specs (OWUI-native `tools`).
    for raw_tool in request_tool_specs:
        raw_name = raw_tool.get("name") if isinstance(raw_tool, dict) else None
        lookup_name = raw_name.strip() if isinstance(raw_name, str) else ""
        tool_cfg = _pick_executor(lookup_name) if lookup_name else None
        runnable = isinstance(tool_cfg, dict) and tool_cfg.get("callable") is not None
        spec = _normalize_responses_function_tool_spec(raw_tool, strictify=False)
        if not spec:
            continue
        origin_name = spec["name"]
        same_name_entries = _registry_tools_named(origin_name)
        if same_name_entries > 1:
            log.debug("Skipping request tool %s: %d registry tools share the name.", origin_name, same_name_entries)
            continue
        handed_back = owui_tool_passthrough or not runnable
        if handed_back:
            spec = {**raw_tool, "name": origin_name}
        else:
            carried = _bound_description_and_parameters(_bound_spec(tool_cfg))
            if carried:
                spec = {**spec, **carried}
        resolved_request_names.add(origin_name)
        candidates.append(
            {
                "origin_source": "owui_request_tools",
                "origin_name": origin_name,
                "spec": spec,
                "tool_cfg": tool_cfg,
                "handed_back": handed_back,
                "origin_key": f"owui_request::{origin_name}",
            }
        )

    # 2) Direct tool servers (always include; collisions handled later).
    carried_by_request = {id(c["tool_cfg"]) for c in candidates if c.get("tool_cfg") is not None}
    carried_by_registry = {
        str(entry.get("origin_name") or (entry.get("spec") or {}).get("name") or "")
        for _, entry in owui_entries
        if entry.get("direct") is True and callable(entry.get("callable"))
    }
    for tool_cfg in direct_entries:
        if id(tool_cfg) in carried_by_request:
            continue
        spec = _responses_spec_from_owui_tool_cfg(tool_cfg, strictify=False)
        if not spec:
            continue
        origin_name = spec["name"]
        if origin_name in carried_by_registry:
            continue
        if (not owui_tool_passthrough) and tool_cfg.get("callable") is None:
            continue
        candidates.append(
            {
                "origin_source": "direct_tool_server",
                "origin_name": origin_name,
                "spec": spec,
                "tool_cfg": tool_cfg,
                "handed_back": False,
                "origin_key": str(tool_cfg.get("origin_key") or f"direct::{origin_name}::{id(tool_cfg)}"),
            }
        )

    for key, tool_cfg in owui_entries:
        spec = _bound_spec(tool_cfg)
        if not spec:
            continue
        if spec["name"] in resolved_request_names:
            continue
        carried_origin = tool_cfg.get("origin_name") if tool_cfg.get("origin_source") else None
        if isinstance(carried_origin, str) and carried_origin.strip():
            origin_name = carried_origin.strip()
        else:
            origin_name = key.strip() if isinstance(key, str) and key.strip() else spec["name"]
        spec["name"] = origin_name
        if (not owui_tool_passthrough) and tool_cfg.get("callable") is None:
            continue
        candidates.append(
            {
                "origin_source": "owui_registry_tools",
                "origin_name": origin_name,
                "spec": spec,
                "tool_cfg": tool_cfg,
                "handed_back": False,
                "origin_key": str(tool_cfg.get("origin_key") or f"owui_registry::{origin_name}"),
            }
        )

    # 4) Extra tools (schema-only). Include only when executable (pipeline) or passthrough is enabled.
    for raw_tool in extra_tools:
        spec = _normalize_responses_function_tool_spec(raw_tool, strictify=False)
        if not spec:
            continue
        origin_name = spec["name"]
        same_name_entries = _registry_tools_named(origin_name)
        if same_name_entries > 1:
            log.debug("Skipping extra tool %s: %d registry tools share the name.", origin_name, same_name_entries)
            continue
        tool_cfg = _pick_executor(origin_name)
        runnable = isinstance(tool_cfg, dict) and tool_cfg.get("callable") is not None
        if (not owui_tool_passthrough) and (not tool_cfg or tool_cfg.get("callable") is None):
            log.debug("Skipping unexecutable extra tool %s (no callable).", origin_name)
            continue
        if not (owui_tool_passthrough or not runnable):
            carried = _bound_description_and_parameters(_bound_spec(tool_cfg))
            if carried:
                spec = {**spec, **carried}
        candidates.append(
            {
                "origin_source": "extra_tools",
                "origin_name": origin_name,
                "spec": spec,
                "tool_cfg": tool_cfg,
                "handed_back": False,
                "origin_key": f"extra::{origin_name}",
            }
        )

    survivors: list[dict[str, Any]] = []
    seen_executors: set[tuple[str, int]] = set()
    for c in candidates:
        tool_cfg = c.get("tool_cfg")
        if tool_cfg is not None:
            key = (c["origin_name"], id(tool_cfg))
            if key in seen_executors:
                log.debug("Skipping duplicate advertisement %s; the same tool already has one.", c["origin_name"])
                continue
            seen_executors.add(key)
        survivors.append(c)
    candidates = survivors

    # Collision-safe rename: only rename when multiple origins share the same name.
    by_name: dict[str, list[dict[str, Any]]] = {}
    for c in candidates:
        by_name.setdefault(c["origin_name"], []).append(c)

    used_names: set[str] = set()
    exec_registry: dict[str, dict[str, Any]] = {}
    exposed_to_origin: dict[str, str] = {}
    tools_out: list[dict[str, Any]] = []

    for c in candidates:
        origin_name = c["origin_name"]
        group = by_name.get(origin_name) or [c]
        needs_rename = len(group) > 1

        prefix = _tool_prefix_for_collision(c["origin_source"], c.get("tool_cfg"))
        digest = hashlib.sha1(
            f"{c['origin_source']}::{c.get('origin_key')}::{origin_name}".encode()
        ).hexdigest()[:8]
        exposed_name = _provider_tool_name(
            origin_name if not needs_rename else f"{prefix}{origin_name}", digest, used_names
        )
        if exposed_name == "ask_user" and not is_builtin_ask_user(c.get("tool_cfg")):
            exposed_name = _provider_tool_name(
                origin_name, digest, used_names | {"ask_user"}
            )
        used_names.add(exposed_name)

        spec = dict(c["spec"])
        spec["name"] = exposed_name
        if strictify and not c.get("handed_back"):
            spec["parameters"] = _strictify_schema(
                spec.get("parameters") or {"type": "object", "properties": {}}
            )
            spec["strict"] = True
        tools_out.append(spec)
        exposed_to_origin[exposed_name] = origin_name
        if builtin_ask_user_names is not None and is_builtin_ask_user(c.get("tool_cfg")):
            builtin_ask_user_names.add(exposed_name)

        tool_cfg = c.get("tool_cfg")
        if not isinstance(tool_cfg, dict) or tool_cfg.get("callable") is None:
            continue
        if owui_tool_passthrough and (tool_cfg.get("direct") is True or is_owui_builtin(tool_cfg)):
            log.debug("Skipping registry entry %s (Open WebUI runs this one).", exposed_name)
            exec_registry[exposed_name] = _owui_owned_entry(c["origin_source"], origin_name, exposed_name)
            continue
        cfg = dict(tool_cfg)
        cfg["origin_source"] = c["origin_source"]
        cfg["origin_name"] = origin_name
        cfg["exposed_name"] = exposed_name
        cfg["declared_params"] = _declared_parameter_names(spec.get("parameters"))
        cfg_spec = cfg.get("spec")
        if not _advertised_wire_is_admissible(cfg_spec, c["spec"].get("parameters"), spec) and isinstance(cfg_spec, dict):
            cfg["declared_params"] = _declared_parameter_names(cfg_spec.get("parameters"))
        if isinstance(cfg_spec, dict):
            updated_spec = dict(cfg_spec)
            updated_spec["name"] = origin_name
            cfg["spec"] = updated_spec
        exec_registry[exposed_name] = cfg

    return _dedupe_tools(tools_out), exec_registry, exposed_to_origin
