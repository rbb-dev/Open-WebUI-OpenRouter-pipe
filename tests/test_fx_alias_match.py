"""A catalog alias must be recognised in the id form Open WebUI actually sends.

The pipe rewrites `/` to `.` and Open WebUI prefixes its own function id, so the runtime
body carries `<function_id>.<vendor>.<model>`. For an alias like `~google/gemini-flash-
latest` the `~` therefore sits mid-string, opening the vendor segment. `lstrip("~")`
strips only a leading character, so the whole panel silently went dark for every prefixed
alias id: the spec's own dotted id is built with the `~` removed, and the match compared
against a string that still had it.

Parametrised over two different alias ids and over every id form Open WebUI can send, and
the ignore rows are what keeps a match-everything fix from passing.
"""

from __future__ import annotations

from typing import Any

import pytest

from open_webui_openrouter_pipe.filters.image_filter_renderer import (
    build_image_model_filter_spec,
    render_image_model_filter_source,
)
from open_webui_openrouter_pipe.models.registry import sanitize_model_id
from tests.test_image_generation import _load_filter_from_source

RECORD: dict[str, Any] = {
    "provider_slug": "google-vertex/global",
    "provider_tag": "google-vertex/global",
    "supported_parameters": {"aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]}},
}

ALIASES = ["~google/gemini-flash-latest", "~openai/gpt-latest"]

_N = [0]


def _filter(model_id: str):
    spec = build_image_model_filter_spec(
        model_id, {"id": model_id, "name": model_id}, [RECORD], dedicated_image_api=True
    )
    _N[0] += 1
    return _load_filter_from_source(
        render_image_model_filter_source(spec), f"fxalias{_N[0]}"
    )


@pytest.mark.parametrize("model_id", ALIASES)
def test_a_prefixed_alias_id_still_finds_its_own_panel(model_id):
    module = _filter(model_id)
    dotted = sanitize_model_id(model_id)
    writes = [
        f"open_webui_openrouter_pipe.{dotted}",
        f"some_other_function_id.{dotted}",
        dotted,
        model_id,
        sanitize_model_id(model_id.lstrip("~")),
        f"open_webui_openrouter_pipe.{sanitize_model_id(model_id.lstrip('~'))}",
    ]
    for sent in writes:
        assert "~" in dotted, "the alias marker must survive sanitisation, or nothing is proven"
        body = module.Filter().inlet(
            {"model": sent},
            None,
            {"valves": module.Filter.UserValves(IMAGE_ASPECT_RATIO="16:9")},
        )
        assert body.get("image_config") == {"aspect_ratio": "16:9"}, (
            f"{sent!r} is a form Open WebUI can send for {model_id!r}; the panel went dark"
        )


@pytest.mark.parametrize("model_id", ALIASES)
def test_a_prefixed_alias_panel_still_ignores_a_different_model(model_id):
    module = _filter(model_id)
    dotted = sanitize_model_id(model_id)
    other = sanitize_model_id(next(a for a in ALIASES if a != model_id))
    ignores = [
        other,
        f"open_webui_openrouter_pipe.{other}",
        "not" + dotted.lstrip("~"),
        f"open_webui_openrouter_pipe.not{dotted.lstrip('~')}",
        "",
    ]
    for sent in ignores:
        body = module.Filter().inlet(
            {"model": sent},
            None,
            {"valves": module.Filter.UserValves(IMAGE_ASPECT_RATIO="16:9")},
        )
        assert "image_config" not in body, (
            f"{sent!r} is not {model_id!r}; the panel must leave the body alone"
        )
