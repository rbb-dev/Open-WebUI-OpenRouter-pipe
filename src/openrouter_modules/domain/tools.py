"""Tool execution orchestration for OpenRouter Responses Pipe.

This module manages tool/function calling for the OpenAI Responses API:
- Schema strictification for OpenAI tools (strict mode)
- Tool registry transformation (Open WebUI → OpenAI format)
- MCP server integration
- Tool deduplication logic
- JSON schema enforcement (required fields, additionalProperties: false)

Layer: domain (business logic - tool management)

Dependencies:
- domain.types: ResponsesBody, CompletionsBody
- domain.registry: ModelFamily for feature detection
- core.config: Valves configuration

This module handles the *schema* transformation. Actual tool execution happens
in the engine/streaming modules.
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..core.config import Valves
    from .types import ResponsesBody
    from .registry import ModelFamily

# Cache size for strictified schemas
_STRICT_SCHEMA_CACHE_SIZE = 128


def build_tools(
    responses_body: ResponsesBody,
    valves: Valves,
    __tools__: Optional[Dict[str, Any]] = None,
    *,
    features: Optional[Dict[str, Any]] = None,
    extra_tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build the OpenAI Responses-API tool spec list for this request.

    Args:
        responses_body: Pydantic model containing request parameters
        valves: Pipe configuration
        __tools__: Open WebUI tool registry (name → callable dict)
        features: Model feature flags (optional)
        extra_tools: Additional OpenAI-format tools to append

    Returns:
        List of OpenAI tool specification dicts

    Process:
        1. Returns [] if the target model doesn't support function calling
        2. Includes Open WebUI registry tools (strictified if enabled)
        3. Adds MCP tools from REMOTE_MCP_SERVERS_JSON valve
        4. Appends any caller-provided extra_tools (already-valid OpenAI tool specs)
        5. Deduplicates by (type,name) identity; last one wins

    Note:
        This builds the *schema* to send to OpenAI. For executing function
        calls at runtime, pass the raw `__tools__` registry to your
        streaming/non-streaming loops; those functions expect name→callable.

    Example:
        >>> tools = build_tools(
        ...     responses_body,
        ...     valves,
        ...     __tools__={"calculator": <callable>},
        ...     extra_tools=[{"type": "function", "function": {...}}]
        ... )
        >>> responses_body.tools = tools
    """
    from .registry import ModelFamily
    from .types import ResponsesBody as RB

    features = features or {}

    # 1) If model can't do function calling, no tools
    if not ModelFamily.supports("function_calling", responses_body.model):
        return []

    tools: List[Dict[str, Any]] = []

    # 2) Baseline: Open WebUI registry tools → OpenAI tool specs
    if isinstance(__tools__, dict) and __tools__:
        tools.extend(
            RB.transform_owui_tools(
                __tools__,
                strict=valves.ENABLE_STRICT_TOOL_CALLING,
            )
        )

    # 3) Optional MCP servers
    if valves.REMOTE_MCP_SERVERS_JSON:
        tools.extend(RB._build_mcp_tools(valves.REMOTE_MCP_SERVERS_JSON))

    # 4) Optional extra tools (already OpenAI-format)
    if isinstance(extra_tools, list) and extra_tools:
        tools.extend(extra_tools)

    return _dedupe_tools(tools)


@functools.lru_cache(maxsize=_STRICT_SCHEMA_CACHE_SIZE)
def _strictify_schema_cached(serialized_schema: str) -> str:
    """Cached worker that enforces strict schema rules on serialized JSON.

    Args:
        serialized_schema: JSON-serialized schema dict

    Returns:
        JSON-serialized strict schema

    Note:
        LRU cache avoids reprocessing identical schemas across requests.
        Cache key is the canonical JSON representation (sorted keys, no whitespace).
    """
    schema_dict = json.loads(serialized_schema)
    strict_schema = _strictify_schema_impl(schema_dict)
    return json.dumps(strict_schema, ensure_ascii=False)


def _strictify_schema(schema):
    """Minimal, predictable transformer to make a JSON schema strict-compatible.

    Enforces OpenAI's strict mode requirements for function calling:

    Rules for every object node (root + nested):
      - additionalProperties := false (reject unknown fields)
      - required := all property keys (all fields mandatory)
      - fields that were optional become nullable (add "null" to their type)

    Traversal:
      - Processes properties, items (dict or list), and anyOf/oneOf branches
      - Does NOT rewrite anyOf/oneOf structure; only enforces object rules inside them

    Args:
        schema: JSON schema dict (or non-dict, which returns empty object wrapper)

    Returns:
        New dict with strict rules applied

    Example:
        Before:
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name"]
        }

        After:
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": ["number", "null"]}  # Now nullable
            },
            "required": ["name", "age"],  # Both required
            "additionalProperties": false  # No extras allowed
        }

    Note:
        - Non-dict inputs are wrapped: {"type": "object", "properties": {"value": <input>}, ...}
        - Non-object root schemas also get wrapped
        - Uses JSON canonicalization + LRU cache for performance
    """
    if not isinstance(schema, dict):
        return {}

    # Canonicalize for cache key
    canonical = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    cached = _strictify_schema_cached(canonical)
    return json.loads(cached)


def _strictify_schema_impl(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Internal implementation for `_strictify_schema` that assumes input is a fresh dict.

    Args:
        schema: JSON schema dict to process

    Returns:
        Modified schema with strict rules applied

    Implementation Details:
        - Uses iterative stack-based traversal (not recursive)
        - Wraps non-object root schemas in object container
        - Marks optional fields nullable by adding "null" to type array
        - Sets all properties as required
        - Disables additionalProperties on all objects
    """
    root_t = schema.get("type")

    # Wrap non-object schemas
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

    # Iterative traversal with stack
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

            # Get explicitly required fields
            raw_required = node.get("required") or []
            raw_required_names: list[str] = [
                name for name in raw_required if isinstance(name, str)
            ]
            all_property_names = list(props.keys())

            # Enforce strict rules
            node["additionalProperties"] = False
            node["required"] = all_property_names

            explicitly_required = {name for name in raw_required_names if name in props}
            optional_candidates = {
                name for name in all_property_names if name not in explicitly_required
            }

            # Make optional fields nullable
            for name, p in props.items():
                if not isinstance(p, dict):
                    continue
                if name in optional_candidates:
                    ptype = p.get("type")
                    if isinstance(ptype, str) and ptype != "null":
                        p["type"] = [ptype, "null"]
                    elif isinstance(ptype, list) and "null" not in ptype:
                        p["type"] = ptype + ["null"]
                stack.append(p)

        # Traverse nested structures
        items = node.get("items")
        if isinstance(items, dict):
            stack.append(items)
        elif isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    stack.append(it)

        # Traverse anyOf/oneOf branches
        for key in ("anyOf", "oneOf"):
            branches = node.get(key)
            if isinstance(branches, list):
                for br in branches:
                    if isinstance(br, dict):
                        stack.append(br)

    return schema


def _dedupe_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Deduplicate a tool list with simple, stable identity keys.

    Identity Rules:
      - Function tools → key = ("function", <name>)
      - Non-function tools → key = (<type>, None)

    Deduplication Strategy:
      - Later entries win (last write wins)
      - Preserves order of final occurrences

    Args:
        tools: List of tool dicts (OpenAI Responses schema)

    Returns:
        Deduplicated list, preserving only the last occurrence per identity

    Example:
        >>> tools = [
        ...     {"type": "function", "function": {"name": "calc"}},
        ...     {"type": "web_search"},
        ...     {"type": "function", "function": {"name": "calc"}},  # Overwrites first
        ... ]
        >>> result = _dedupe_tools(tools)
        >>> len(result)
        2
        >>> result[0]["type"]
        'web_search'
        >>> result[1]["function"]["name"]
        'calc'

    Note:
        - Non-dict entries are silently skipped
        - Tools without a type field are skipped
        - Empty/None input returns empty list
    """
    if not tools:
        return []

    canonical: Dict[tuple, Dict[str, Any]] = {}
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function":
            key = ("function", t.get("function", {}).get("name"))
        else:
            key = (t.get("type"), None)
        if key[0]:  # Skip if type is missing
            canonical[key] = t

    return list(canonical.values())


__all__ = [
    "build_tools",
    "_strictify_schema",
    "_dedupe_tools",
]
