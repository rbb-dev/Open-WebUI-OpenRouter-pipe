"""Tool registry building and collision resolution.

This module handles tool spec management:
- build_tools: Build OpenAI Responses-API tool spec list
- _dedupe_tools: Remove duplicate tool definitions
- _build_collision_safe_tool_specs_and_registry: Handle name collisions

Ensures collision-safe tool names and builds execution registry for dispatcher.
"""

from __future__ import annotations

import functools
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

# Import ModelFamily for function calling support check
from ..models.registry import ModelFamily

# Import tool schema functions
from .tool_schema import _strictify_schema
from ..core.timing_logger import timed

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

LOGGER = logging.getLogger(__name__)

@timed
def build_tools(
    responses_body: "ResponsesBody",
    valves: "Pipe.Valves",
    __tools__: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    *,
    features: Optional[Dict[str, Any]] = None,
    extra_tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build the OpenAI Responses-API tool spec list for this request.

    - Returns [] if the target model doesn't support function calling.
    - Includes Open WebUI registry tools (strictified if enabled).
    - Adds OpenAI web_search (if allowed + supported + not minimal effort).
    - Appends any caller-provided extra_tools (already-valid OpenAI tool specs).
    - Deduplicates by (type,name) identity; last one wins.

    NOTE: This builds the *schema* to send to OpenAI. For executing function
    calls at runtime, you can keep passing the raw `__tools__` registry into
    your streaming/non-streaming loops; those functions expect name->callable.
    """
    features = features or {}

    owui_tool_passthrough = getattr(valves, "TOOL_EXECUTION_MODE", "Pipeline") == "Open-WebUI"

    # 1) If model can't do function calling, no tools (unless Open-WebUI tool pass-through is enabled).
    if (not owui_tool_passthrough) and (not ModelFamily.supports("function_calling", responses_body.model)):
        return []

    tools: List[Dict[str, Any]] = []

    # 2) Baseline: Open WebUI registry tools -> OpenAI tool specs
    if isinstance(__tools__, dict) and __tools__:
        tools.extend(
            ResponsesBody.transform_owui_tools(
                __tools__,
                strict=bool(valves.ENABLE_STRICT_TOOL_CALLING) and (not owui_tool_passthrough),
            )
        )
    elif isinstance(__tools__, list) and __tools__:
        tools.extend([tool for tool in __tools__ if isinstance(tool, dict)])

    # 3) Optional extra tools (already OpenAI-format)
    if isinstance(extra_tools, list) and extra_tools:
        tools.extend(extra_tools)

    return _dedupe_tools(tools)


_STRICT_SCHEMA_CACHE_SIZE = 128


@timed
def _dedupe_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
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
    canonical: Dict[tuple, Dict[str, Any]] = {}
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


@timed
def _normalize_responses_function_tool_spec(tool: Any, *, strictify: bool) -> Optional[dict[str, Any]]:
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
    return spec


@timed
def _responses_spec_from_owui_tool_cfg(tool_cfg: dict[str, Any], *, strictify: bool) -> Optional[dict[str, Any]]:
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
    return out


@timed
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
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    """Build collision-safe tool specs and an execution registry.

    Returns:
      - tools: Responses-style tool specs with collision-safe names.
      - exec_registry: mapping exposed_name -> OWUI tool cfg dict (callable/spec/etc).
      - exposed_to_origin: mapping exposed_name -> origin tool name (for passthrough execution).
    """
    log = logger or LOGGER
    request_tool_specs = request_tool_specs or []
    extra_tools = extra_tools or []
    owui_registry = owui_registry or {}
    direct_registry = direct_registry or {}
    builtin_registry = builtin_registry or {}

    # Normalize all registries into lists so we can preserve collisions.
    owui_entries: list[dict[str, Any]] = [
        entry for entry in owui_registry.values() if isinstance(entry, dict)
    ]
    direct_entries: list[dict[str, Any]] = [
        entry for entry in direct_registry.values() if isinstance(entry, dict)
    ]
    builtin_entries: list[dict[str, Any]] = [
        entry for entry in builtin_registry.values() if isinstance(entry, dict)
    ]

    @timed
    def _pick_executor(name: str, *, prefer: str | None = None) -> dict[str, Any] | None:
        if prefer == "builtin":
            for e in builtin_entries:
                spec = e.get("spec")
                if isinstance(spec, dict) and spec.get("name") == name:
                    return e
        if prefer == "owui":
            for e in owui_entries:
                spec = e.get("spec")
                if isinstance(spec, dict) and spec.get("name") == name:
                    return e
        if prefer == "direct":
            for e in direct_entries:
                spec = e.get("spec")
                if isinstance(spec, dict) and spec.get("name") == name:
                    return e
        # Default preference order for request/extra tools.
        for e in builtin_entries:
            spec = e.get("spec")
            if isinstance(spec, dict) and spec.get("name") == name:
                return e
        for e in owui_entries:
            spec = e.get("spec")
            if isinstance(spec, dict) and spec.get("name") == name:
                return e
        for e in direct_entries:
            spec = e.get("spec")
            if isinstance(spec, dict) and spec.get("name") == name:
                return e
        return None

    request_names: set[str] = set()
    for tool in request_tool_specs:
        if isinstance(tool, dict) and tool.get("type") == "function":
            name = tool.get("name")
            if isinstance(name, str) and name.strip():
                request_names.add(name.strip())

    candidates: list[dict[str, Any]] = []

    # 1) Request-provided tool specs (OWUI-native `tools`).
    for raw_tool in request_tool_specs:
        spec = _normalize_responses_function_tool_spec(raw_tool, strictify=strictify)
        if not spec:
            continue
        origin_name = spec["name"]
        tool_cfg = _pick_executor(origin_name)
        if (not owui_tool_passthrough) and (not tool_cfg or tool_cfg.get("callable") is None):
            log.debug("Skipping unexecutable request tool %s (no callable).", origin_name)
            continue
        candidates.append(
            {
                "origin_source": "owui_request_tools",
                "origin_name": origin_name,
                "spec": spec,
                "tool_cfg": tool_cfg,
                "origin_key": f"owui_request::{origin_name}",
            }
        )

    # 2) Direct tool servers (always include; collisions handled later).
    for tool_cfg in direct_entries:
        spec = _responses_spec_from_owui_tool_cfg(tool_cfg, strictify=strictify)
        if not spec:
            continue
        origin_name = spec["name"]
        if (not owui_tool_passthrough) and tool_cfg.get("callable") is None:
            continue
        candidates.append(
            {
                "origin_source": "direct_tool_server",
                "origin_name": origin_name,
                "spec": spec,
                "tool_cfg": tool_cfg,
                "origin_key": str(tool_cfg.get("origin_key") or f"direct::{origin_name}::{id(tool_cfg)}"),
            }
        )

    # 3) Registry tools (__tools__), excluding those already present in request tools (to avoid duplication).
    for tool_cfg in owui_entries:
        spec = _responses_spec_from_owui_tool_cfg(tool_cfg, strictify=strictify)
        if not spec:
            continue
        origin_name = spec["name"]
        if origin_name in request_names:
            continue
        if (not owui_tool_passthrough) and tool_cfg.get("callable") is None:
            continue
        candidates.append(
            {
                "origin_source": "owui_registry_tools",
                "origin_name": origin_name,
                "spec": spec,
                "tool_cfg": tool_cfg,
                "origin_key": str(tool_cfg.get("origin_key") or f"owui_registry::{origin_name}"),
            }
        )

    # 4) Extra tools (schema-only). Include only when executable (pipeline) or passthrough is enabled.
    for raw_tool in extra_tools:
        spec = _normalize_responses_function_tool_spec(raw_tool, strictify=strictify)
        if not spec:
            continue
        origin_name = spec["name"]
        tool_cfg = _pick_executor(origin_name)
        if (not owui_tool_passthrough) and (not tool_cfg or tool_cfg.get("callable") is None):
            log.debug("Skipping unexecutable extra tool %s (no callable).", origin_name)
            continue
        candidates.append(
            {
                "origin_source": "extra_tools",
                "origin_name": origin_name,
                "spec": spec,
                "tool_cfg": tool_cfg,
                "origin_key": f"extra::{origin_name}",
            }
        )

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
        exposed_name = origin_name if not needs_rename else f"{prefix}{origin_name}"
        if exposed_name in used_names:
            digest = hashlib.sha1(
                f"{c['origin_source']}::{c.get('origin_key')}::{origin_name}".encode("utf-8")
            ).hexdigest()[:8]
            exposed_name = f"{exposed_name}__{digest}"
        used_names.add(exposed_name)

        spec = dict(c["spec"])
        spec["name"] = exposed_name
        tools_out.append(spec)
        exposed_to_origin[exposed_name] = origin_name

        tool_cfg = c.get("tool_cfg")
        if owui_tool_passthrough:
            continue
        if not isinstance(tool_cfg, dict) or tool_cfg.get("callable") is None:
            continue
        cfg = dict(tool_cfg)
        cfg["origin_source"] = c["origin_source"]
        cfg["origin_name"] = origin_name
        cfg["exposed_name"] = exposed_name
        cfg_spec = cfg.get("spec")
        if isinstance(cfg_spec, dict):
            updated_spec = dict(cfg_spec)
            updated_spec["name"] = origin_name
            if isinstance(spec.get("parameters"), dict):
                updated_spec["parameters"] = spec["parameters"]
            cfg["spec"] = updated_spec
        exec_registry[exposed_name] = cfg

    return _dedupe_tools(tools_out), exec_registry, exposed_to_origin
