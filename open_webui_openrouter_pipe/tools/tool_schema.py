"""Tool schema strictification for OpenAI Structured Outputs.

This module enforces strict schema rules:
- additionalProperties: false on all objects
- required: all property keys
- Optional fields become nullable (add "null" to type)
- Traverse properties, items, anyOf/oneOf branches

Implements aggressive schema transformation for compatibility with strict mode.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Dict, Optional
from ..core.timing_logger import timed

LOGGER = logging.getLogger(__name__)

_STRICT_SCHEMA_CACHE_SIZE = 128

@lru_cache(maxsize=_STRICT_SCHEMA_CACHE_SIZE)
@timed
def _strictify_schema_cached(serialized_schema: str) -> str:
    """Cached worker that enforces strict schema rules on serialized JSON."""
    schema_dict = json.loads(serialized_schema)
    strict_schema = _strictify_schema_impl(schema_dict)
    return json.dumps(strict_schema, ensure_ascii=False)


@timed
def _strictify_schema(schema):
    """
    Minimal, predictable transformer to make a JSON schema strict-compatible.

    Rules for every object node (root + nested):
      - additionalProperties := false
      - required := all property keys
      - fields that were optional become nullable (add "null" to their type)

    We traverse properties, items (dict or list), and anyOf/oneOf branches.
    We do NOT rewrite anyOf/oneOf; we only enforce object rules inside them.

    Returns a new dict. Non-dict inputs return {}.
    """
    if not isinstance(schema, dict):
        return {}

    canonical = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    cached = _strictify_schema_cached(canonical)
    return json.loads(cached)


@timed
def _strictify_schema_impl(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Internal implementation for `_strictify_schema` that assumes input is a fresh dict.

    Applies strict-mode transformations to JSON schemas:
    - Ensures all object nodes have `additionalProperties: false`
    - Makes all properties required
    - Makes optional properties nullable by adding "null" to their type
    - **Auto-infers missing types**: Adds default types to properties without one:
      * Empty schemas `{}` -> `{"type": "object"}`
      * Schemas with `properties` but no type -> `{"type": "object"}`
      * Schemas with `items` but no type -> `{"type": "array"}`

    This defensive type inference ensures schemas are valid for OpenAI strict mode.
    """
    root_t = schema.get("type")
    if not (
        root_t == "object"
        or (isinstance(root_t, list) and "object" in root_t)
        or "properties" in schema
    ):
        schema = {
            "type": "object",
            "properties": {"value": schema},
            "required": ["value"],
            "additionalProperties": False,
        }

    stack = [schema]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue

        t = node.get("type")
        is_object = ("properties" in node) or (t == "object") or (
            isinstance(t, list) and "object" in t
        )
        if is_object:
            props = node.get("properties")
            if not isinstance(props, dict):
                props = {}
                node["properties"] = props

            raw_required = node.get("required") or []
            raw_required_names: list[str] = [
                name for name in raw_required if isinstance(name, str)
            ]
            all_property_names = list(props.keys())

            node["additionalProperties"] = False
            node["required"] = all_property_names

            explicitly_required = {name for name in raw_required_names if name in props}
            optional_candidates = {
                name for name in all_property_names if name not in explicitly_required
            }

            for name, p in props.items():
                if not isinstance(p, dict):
                    continue

                # Ensure every property schema has a type key (strict mode requirement)
                if "type" not in p:
                    schema_structure_keys = {"properties", "items", "anyOf", "oneOf", "allOf"}
                    has_nested_structure = any(k in p for k in schema_structure_keys)

                    if has_nested_structure:
                        if "properties" in p:
                            p["type"] = "object"
                            LOGGER.debug(
                                "Added inferred type 'object' to property '%s' which has 'properties' but no explicit type. "
                                "Consider fixing the schema definition at the source.",
                                name
                            )
                        elif "items" in p:
                            p["type"] = "array"
                            LOGGER.debug(
                                "Added inferred type 'array' to property '%s' which has 'items' but no explicit type. "
                                "Consider fixing the schema definition at the source.",
                                name
                            )
                        # For anyOf/oneOf/allOf without type, don't add a default
                        # Let OpenAI validation handle these complex cases
                    else:
                        # Empty or minimal schema (e.g., {"description": "..."} or just {})
                        # Default to object as the safest, most flexible type
                        p["type"] = "object"
                        LOGGER.debug(
                            "Added default type 'object' to property '%s' with no type or schema structure. "
                            "This indicates an incomplete schema definition that should be fixed at the source.",
                            name
                        )

                # Handle optional fields by adding null to type
                if name in optional_candidates:
                    ptype = p.get("type")
                    if isinstance(ptype, str) and ptype != "null":
                        p["type"] = [ptype, "null"]
                    elif isinstance(ptype, list) and "null" not in ptype:
                        p["type"] = ptype + ["null"]
                stack.append(p)

        items = node.get("items")
        if isinstance(items, dict):
            # Ensure items schema has a type key
            if "type" not in items and "properties" not in items and "items" not in items:
                items["type"] = "object"
                LOGGER.debug("Added default type 'object' to empty items schema")
            stack.append(items)
        elif isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    stack.append(it)

        for key in ("anyOf", "oneOf"):
            branches = node.get(key)
            if isinstance(branches, list):
                for br in branches:
                    if isinstance(br, dict):
                        # Ensure branch schema has a type key
                        if "type" not in br and "properties" not in br and "items" not in br:
                            br["type"] = "object"
                            LOGGER.debug("Added default type 'object' to empty %s branch", key)
                        stack.append(br)

    return schema




@timed
def _normalize_responses_function_tool_spec(tool: Any, *, strictify: bool) -> Optional[Dict[str, Any]]:
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
def _classify_function_call_artifacts(
    artifacts: Dict[str, Dict[str, Any]]
) -> tuple[set[str], set[str], set[str]]:
    """
    Inspect persisted artifacts and return three identifier sets:

    - valid_ids: call_ids that have both a function_call and function_call_output
    - orphaned_calls: call_ids that only have a function_call entry
    - orphaned_outputs: call_ids that only have a function_call_output entry
    """
    call_ids: set[str] = set()
    output_ids: set[str] = set()

    for payload in artifacts.values():
        if not isinstance(payload, dict):
            continue
        call_id = payload.get("call_id")
        if not isinstance(call_id, str):
            continue
        call_id = call_id.strip()
        if not call_id:
            continue
        payload_type = (payload.get("type") or "").lower()
        if payload_type == "function_call":
            call_ids.add(call_id)
        elif payload_type == "function_call_output":
            output_ids.add(call_id)

    valid_ids = call_ids & output_ids
    orphaned_calls = call_ids - valid_ids
    orphaned_outputs = output_ids - valid_ids
    return valid_ids, orphaned_calls, orphaned_outputs
