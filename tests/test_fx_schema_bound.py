"""The API-wide schema may bound a control's TYPE; it may never become its value list.

Two properties, and the tension between them is the whole point.

1. A control whose description states a numeric range must enforce that range. The
   image-gen tool renders `output_compression` from OpenRouter's API-wide schema on the
   34 recorded contracts that do not publish it, and told the user "a whole number from
   0 to 100" while accepting 100000 and -1 and putting them in the outbound tool call.

2. A control must never offer a VALUE the model does not publish. This is the defect an
   earlier fix removed: the API-wide `output_format` enum carries `svg`, which is a
   vectorization-only format, and offering it on all forty models advertised a value
   every one of them rejects. A bound is not a value list -- `ge`/`le` refuse only what
   OpenRouter itself refuses and assert nothing about any particular number, while a
   dropdown asserts that each option works here.

So: bounds come from the schema, choices come from the model, and the second assertion
here sweeps every recorded contract to prove no choice was invented.
"""

from __future__ import annotations

import json
import typing
from pathlib import Path
from typing import Any

import pytest

from open_webui_openrouter_pipe.filters.image_filter_renderer import (
    build_image_gen_tool_spec,
    build_image_model_filter_spec,
    render_image_gen_filter_source,
    render_image_model_filter_source,
)
from open_webui_openrouter_pipe.integrations.image_types import SCHEMA_ENUMS, SCHEMA_RANGES
from tests.test_image_generation import _load_filter_from_source

FIX = Path(__file__).parent / "fixtures"
CONTRACTS = sorted(FIX.glob("openrouter_image_endpoints_*.json"))
assert len(CONTRACTS) >= 40, f"only {len(CONTRACTS)} recorded contracts"

_N = [0]


def _spec(path: Path):
    raw = json.loads(path.read_text())
    return raw["id"], build_image_model_filter_spec(
        raw["id"], {"id": raw["id"], "name": raw["id"]}, raw["endpoints"],
        dedicated_image_api=True,
    )


def _module(source: str):
    _N[0] += 1
    return _load_filter_from_source(source, f"fxbound{_N[0]}")


def _literal_choices(field: Any) -> tuple[Any, ...] | None:
    annotation = field.annotation
    if getattr(annotation, "__origin__", None) is typing.Literal or str(
        annotation
    ).startswith("typing.Literal"):
        return tuple(v for v in typing.get_args(annotation) if v != "")
    return None


@pytest.mark.parametrize("path", CONTRACTS, ids=lambda p: p.stem[27:])
@pytest.mark.parametrize(
    ("typed", "accepted"),
    [(0, True), (100, True), (-1, False), (100000, False)],
)
def test_a_stated_numeric_range_is_the_range_the_control_enforces(path, typed, accepted):
    model_id, spec = _spec(path)
    tool = build_image_gen_tool_spec(spec)
    if "output_compression" not in tool.schema_only:
        pytest.skip(f"{model_id} publishes its own output_compression range")
    module = _module(
        render_image_gen_filter_source(spec, catalog_match=True, selected_model=model_id)
    )
    low, high = SCHEMA_RANGES["output_compression"]
    described = module.Filter.UserValves.model_fields["IMAGE_OUTPUT_COMPRESSION"].description
    assert f"from {low} to {high}" in (described or ""), (
        f"the bound under test is not the one the control states: {described!r}"
    )

    meta: dict[str, Any] = {}
    module.Filter().inlet(
        {"model": model_id},
        meta,
        {"valves": {"IMAGE_OUTPUT_COMPRESSION": typed}},
    )
    params = meta["openrouter_pipe"]["server_tools"]["image_generation"]
    if accepted:
        assert params.get("output_compression") == typed, (
            f"{typed} is inside the stated {low}-{high} range and must travel: {params!r}"
        )
        return
    assert "output_compression" not in params, (
        f"{typed} is outside the {low}-{high} range this control states, yet it reached "
        f"the outbound tool call as {params!r}"
    )


@pytest.mark.parametrize("path", CONTRACTS, ids=lambda p: p.stem[27:])
def test_no_control_offers_a_value_its_own_model_does_not_publish(path):
    model_id, spec = _spec(path)
    published: dict[str, set[Any]] = {}
    raw = json.loads(path.read_text())
    for record in raw["endpoints"]:
        for name, descriptor in (record.get("supported_parameters") or {}).items():
            if isinstance(descriptor, dict) and descriptor.get("type") == "enum":
                published.setdefault(name, set()).update(descriptor.get("values") or [])

    for label, source in (
        ("panel", render_image_model_filter_source(spec)),
        (
            "tool",
            render_image_gen_filter_source(
                spec, catalog_match=True, selected_model=model_id
            ),
        ),
    ):
        module = _module(source)
        for field_name, field in module.Filter.UserValves.model_fields.items():
            choices = _literal_choices(field)
            if choices is None or not field_name.startswith("IMAGE_"):
                continue
            param = field_name[len("IMAGE_") :].lower()
            if param not in SCHEMA_ENUMS:
                continue
            invented = [
                value
                for value in choices
                if value not in published.get(param, set())
                and value not in published.get("resolution", set())
            ]
            assert not invented, (
                f"{label} filter for {model_id} offers {invented!r} under {field_name}, "
                f"which it publishes as {sorted(published.get(param, set()))!r} -- the "
                "API-wide schema is not this model's list"
            )
