"""Tool schema strictification for OpenAI Structured Outputs.

This module enforces strict schema rules:
- additionalProperties: false on all objects
- required: all property keys
- Optional fields become nullable (add "null" to type)
- Traverse properties, items, anyOf/oneOf branches

Implements aggressive schema transformation for compatibility with strict mode.
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from typing import Any, Dict
from ..core.config import LOGGER

_STRICT_SCHEMA_CACHE_SIZE = 128

_STRICT_UNSUPPORTED_KEYS = (
    "default",
    "$schema",
    "pattern",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minItems",
    "maxItems",
    "uniqueItems",
    "format",
)


def _inline_allof(
    node: Dict[str, Any],
    defs_lookup: Dict[str, Any],
    resolve_budget: list[int],
) -> None:
    for _ in range(8):
        had_allof = "allOf" in node
        _inline_allof_once(node, defs_lookup, resolve_budget)
        if not had_allof or "allOf" not in node or resolve_budget[0] <= 0:
            return


def _inline_allof_once(
    node: Dict[str, Any],
    defs_lookup: Dict[str, Any],
    resolve_budget: list[int],
) -> None:
    branches = node.get("allOf")
    if not isinstance(branches, list) or not branches:
        if isinstance(branches, list):
            node.pop("allOf", None)
        return
    dict_branches = [branch for branch in branches if isinstance(branch, dict)]
    if len(dict_branches) != len(branches):
        return
    structural_keys = ("properties", "required", "items", "anyOf", "oneOf")
    if (
        len(dict_branches) == 1
        and "$ref" in dict_branches[0]
        and not any(key in node for key in structural_keys)
    ):
        replacement = dict(dict_branches[0])
        node.clear()
        node.update(replacement)
        return
    resolved_branches: list[Dict[str, Any]] = []
    for branch in dict_branches:
        if "$ref" not in branch:
            resolved_branches.append(branch)
            continue
        ref_value = branch.get("$ref")
        target = defs_lookup.get(ref_value) if isinstance(ref_value, str) else None
        if not isinstance(target, dict) or resolve_budget[0] <= 0:
            return
        resolve_budget[0] -= 1
        inlined = copy.deepcopy(target)
        inlined.update({key: value for key, value in branch.items() if key != "$ref"})
        resolved_branches.append(inlined)
    merged_props: Dict[str, Any] = {}
    merged_required: list[str] = []
    extras: Dict[str, Any] = {}
    for branch in resolved_branches:
        for key, value in branch.items():
            if key == "properties" and isinstance(value, dict):
                merged_props.update(value)
            elif key == "required" and isinstance(value, list):
                merged_required.extend(
                    str(name) for name in value if str(name) not in merged_required
                )
            else:
                extras[key] = value
    node.pop("allOf", None)
    for key, value in extras.items():
        node.setdefault(key, value)
    if merged_props:
        existing = node.get("properties")
        if isinstance(existing, dict):
            combined_props = dict(merged_props)
            combined_props.update(existing)
            node["properties"] = combined_props
        else:
            node["properties"] = merged_props
        node.setdefault("type", "object")
    if merged_required:
        existing_required = node.get("required")
        combined = list(existing_required) if isinstance(existing_required, list) else []
        combined.extend(name for name in merged_required if name not in combined)
        node["required"] = combined

@lru_cache(maxsize=_STRICT_SCHEMA_CACHE_SIZE)
def _strictify_schema_cached(serialized_schema: str) -> str:
    """Cached worker that enforces strict schema rules on serialized JSON."""
    schema_dict = json.loads(serialized_schema)
    strict_schema = _strictify_schema_impl(schema_dict)
    return json.dumps(strict_schema, ensure_ascii=False)


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
        hoisted: Dict[str, Any] = {}
        for defs_key in ("$defs", "definitions"):
            defs = schema.get(defs_key)
            if isinstance(defs, dict):
                hoisted[defs_key] = schema.pop(defs_key)
        schema = {
            "type": "object",
            "properties": {"value": schema},
            "required": ["value"],
            "additionalProperties": False,
            **hoisted,
        }

    defs_lookup: Dict[str, Any] = {}
    for defs_key in ("$defs", "definitions"):
        defs = schema.get(defs_key)
        if isinstance(defs, dict):
            for def_name, def_body in defs.items():
                if isinstance(def_body, dict):
                    defs_lookup[f"#/{defs_key}/{def_name}"] = def_body
    resolve_budget = [64]

    stack = [schema]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue

        for unsupported_key in _STRICT_UNSUPPORTED_KEYS:
            node.pop(unsupported_key, None)

        if "$ref" in node:
            continue

        _inline_allof(node, defs_lookup, resolve_budget)
        if "$ref" in node:
            continue

        for defs_key in ("$defs", "definitions"):
            defs = node.get(defs_key)
            if isinstance(defs, dict):
                for definition in defs.values():
                    if isinstance(definition, dict):
                        stack.append(definition)

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

                for unsupported_key in _STRICT_UNSUPPORTED_KEYS:
                    p.pop(unsupported_key, None)
                if "$ref" in p:
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
            if (
                "type" not in items
                and "properties" not in items
                and "items" not in items
                and "$ref" not in items
            ):
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
                        if (
                            "type" not in br
                            and "properties" not in br
                            and "items" not in br
                            and "$ref" not in br
                        ):
                            br["type"] = "object"
                            LOGGER.debug("Added default type 'object' to empty %s branch", key)
                        stack.append(br)

    return schema




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
