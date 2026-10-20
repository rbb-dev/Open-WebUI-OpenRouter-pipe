from __future__ import annotations

from typing import Any

from ..models.registry import ModelFamily


def build_response_format(
    *,
    name: str,
    schema: dict[str, Any],
    strict: bool = True,
) -> dict[str, Any]:
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
    supports = supported_parameters or ()
    if "structured_outputs" in supports:
        return response_format
    inner = dict(response_format.get("json_schema") or {})
    inner["strict"] = False
    return {"type": response_format.get("type", "json_schema"), "json_schema": inner}


def build_response_format_for_model(
    *, name: str, schema: dict[str, Any], model_id: str
) -> dict[str, Any]:
    return downgrade_strict_for_provider(
        build_response_format(name=name, schema=schema, strict=True),
        supported_parameters=ModelFamily.supported_parameters(model_id),
    )
