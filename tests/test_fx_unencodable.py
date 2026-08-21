"""Nothing the adapter puts in the request body may be a float JSON cannot write.

`image_config` reaches the adapter exactly as the client sent it, and Python's own
`json.loads` accepts the bare tokens `NaN`, `Infinity` and `-Infinity`. The range check
compared with `>` and `<`, and both are False for NaN, so a NaN sailed through every
bound and the pipe then emitted `{"n": NaN}` -- which no strict JSON reader will accept
and which OpenRouter answers with a parse error naming nothing the user typed.

Parametrised over three non-finite floats and over three destinations that reach the wire
by different routes -- a bounded top-level parameter, a support-flag parameter with no
bounds at all, and a provider passthrough key the contract never describes -- so a guard
placed on only one route fails the others. The finite rows are what stops a "refuse every
number" fix from passing.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter

RECORD: dict[str, Any] = {
    "provider_slug": "openai",
    "provider_tag": "openai",
    "allowed_passthrough_parameters": ["style"],
    "supported_parameters": {
        "n": {"type": "range", "min": 1, "max": 4},
        "seed": {"type": "boolean"},
    },
}

UNWRITABLE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("value", [*UNWRITABLE, 2, 3])
@pytest.mark.parametrize("key", ["n", "seed", "style"])
def test_a_float_json_cannot_write_is_refused_and_reported(key, value):
    top_level, passthrough, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {key: value}},
        allowed_passthrough=frozenset({"style"}),
        record=RECORD,
        records=[RECORD],
    )
    sent = {**passthrough, **top_level}
    if math.isfinite(value):
        assert sent.get(key) == value, f"{key}={value!r} is writable and must travel: {notes}"
        return
    assert key not in sent, (
        f"{key}={value!r} cannot be written as JSON, yet it reached the request as "
        f"{sent!r}"
    )
    assert any(key in note.text for note in notes), (
        f"{key}={value!r} was dropped with nothing said: {[n.text for n in notes]!r}"
    )


@pytest.mark.parametrize("value", UNWRITABLE)
@pytest.mark.parametrize(
    "container",
    [
        lambda v: {"style": [1, v]},
        lambda v: {"style": {"weight": v}},
    ],
)
def test_a_float_json_cannot_write_is_refused_inside_a_container(container, value):
    top_level, passthrough, _notes = ImageGenerationAdapter._split_image_config(
        {"image_config": container(value)},
        allowed_passthrough=frozenset({"style"}),
        record=RECORD,
        records=[RECORD],
    )
    body = {"model": "m", "prompt": "p", **passthrough, **top_level}
    json.dumps(body, allow_nan=False)


@pytest.mark.parametrize("value", UNWRITABLE)
def test_the_body_built_from_a_non_finite_choice_is_writable_json(value):
    top_level, passthrough, _notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"n": value, "seed": value, "style": value}},
        allowed_passthrough=frozenset({"style"}),
        record=RECORD,
        records=[RECORD],
    )
    body = {"model": "m", "prompt": "p", **passthrough, **top_level}
    json.dumps(body, allow_nan=False)


@pytest.mark.parametrize("value", [*UNWRITABLE, 2])
def test_a_non_finite_value_never_counts_as_inside_a_published_range(value):
    fitted, _reason = ImageGenerationAdapter._fit_descriptor(
        {"type": "range", "min": 1, "max": 4}, value
    )
    assert (fitted is not None) is math.isfinite(value), (
        f"{value!r} was measured against 1-4 and came back {fitted!r}"
    )
