"""JSON-schema response_format builder for OpenRouter Structured Outputs.

Ports the pattern from `.external/seedream.py:695-750`.
"""
from __future__ import annotations

from typing import Any


def build_response_format(
    *,
    name: str,
    schema: dict[str, Any],
    strict: bool = True,
) -> dict[str, Any]:
    """Wrap a JSON schema in OpenAI/OpenRouter Structured Outputs envelope.

    Args:
        name: Schema name (used as the json_schema.name field).
        schema: The JSON schema (must satisfy strict-mode constraints if strict=True:
            additionalProperties: False at every object level, all properties listed
            in `required`, nullable fields encoded as ["type", "null"]).
        strict: Whether to enforce strict mode.

    Returns:
        Dict shaped: {"type": "json_schema", "json_schema": {...}}.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": strict,
            "schema": schema,
        },
    }


def downgrade_strict_for_provider(
    response_format: dict[str, Any],
    *,
    supported_parameters: list[str] | tuple[str, ...] | set[str] | frozenset[str] | None,
) -> dict[str, Any]:
    """If `structured_outputs` is not in supported_parameters, set strict=False.

    Some OpenRouter providers don't support strict mode. Downgrade quietly so
    the call still succeeds; the pipe-side validator is the second-line defense.

    Returns a NEW dict (does not mutate the input).
    """
    supports = supported_parameters or ()
    if "structured_outputs" in supports:
        return response_format
    inner = dict(response_format.get("json_schema") or {})
    inner["strict"] = False
    return {"type": response_format.get("type", "json_schema"), "json_schema": inner}
