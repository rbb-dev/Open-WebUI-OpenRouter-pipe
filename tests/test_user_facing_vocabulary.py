"""No surface a user reads may name a thing only the implementation can see.

Three surfaces carry text into the chat window: the `help` panel, the warning
notifications raised while a generation runs, and the `Field(description=...)` of every
control a generated filter draws. A reader handed `image_config` or `OWUI` is handed a
term they cannot look up, in place of the thing they could act on.

Published OpenRouter parameter names are deliberately NOT banned: `style`,
`safety_tolerance`, `moderation` and `aspect_ratio` are the labels the user is looking at
on screen, so a note naming one is naming what they typed. That is why the rule is a
named list rather than "no snake_case" or "nothing in backticks" -- both of those proxies
would delete the names that make a rejection actionable, and neither would have caught
`image_config`, `output_modalities` or `b64_json`, which are written in these surfaces
without backticks.

`modalities` is listed separately from `output_modalities` because the catalog fallback
had prettified the key into a heading -- `- **Output modalities**:` -- which a check for
the key alone reads straight past.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
from typing import Any

import pytest

from open_webui_openrouter_pipe.filters.image_filter_renderer import (
    build_image_model_filter_spec,
    render_image_model_filter_source,
)
from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter
from open_webui_openrouter_pipe.integrations.image_help import (
    IMAGE_HELP_BY_MODEL,
    render_image_help,
)

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "open_webui_openrouter_pipe"

BANNED_KEYS = (
    "image_config",
    "output_modalities",
    "input_modalities",
    "supported_parameters",
    "allowed_passthrough_parameters",
    "provider_slug",
    "b64_json",
    "endpoint record",
    "endpoint_record",
    "our side",
)

BANNED_NAMES = (
    "OWUI",
    "modalities",
    "modality",
    "ImageGenerationAdapter",
    "ImageGenerationError",
    "ImageGenerationResult",
    "ImageModelFilterSpec",
    "GeneratedImage",
    "image_filter_renderer",
    "image_help",
    "image_types",
    "image_client",
    "image_catalog",
)

PUBLISHED_PARAMETER_NAMES = (
    "aspect_ratio",
    "moderation",
    "output_compression",
    "safety_tolerance",
    "style",
)
"""Names the user chose in a control and sees on screen. Banning these is the wrong fix.

Every one is snake_case and three of them are also request keys, so a rule that reasons
about the shape of a token rather than about who the token belongs to deletes these too.
"""

_NAME_RES = tuple(re.compile(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])") for n in BANNED_NAMES)


def offences(text: str) -> list[str]:
    """Every banned term in one string, matched the way that term is written."""
    found = [key for key in BANNED_KEYS if key in text]
    found.extend(
        name for name, pattern in zip(BANNED_NAMES, _NAME_RES) if pattern.search(text)
    )
    return found


def _report(hits: list[tuple[str, str, str]]) -> str:
    return "\n".join(f"  {where}: {term!r} in {text!r}" for where, term, text in hits)


def _contracts() -> list[tuple[str, str]]:
    out = []
    for path in sorted(FIXTURES.glob("openrouter_image_endpoints_*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = raw.get("endpoints") or [raw]
        model_id = next(
            (r.get("model_id") or r.get("model") for r in records if isinstance(r, dict)), None
        )
        slug = path.stem[len("openrouter_image_endpoints_") :]
        out.append((slug, model_id or slug.replace("_", "/", 1)))
    assert out, "the checked-in contracts are what make this sweep fleet-wide"
    unresolved = [(slug, model) for slug, model in out if model not in IMAGE_HELP_BY_MODEL]
    assert not unresolved, (
        "every contract must name a model that has curated prose, or this sweep quietly "
        f"becomes forty renderings of the catalog fallback instead: {unresolved}"
    )
    return out


def _records(slug: str) -> list[dict]:
    raw = json.loads(
        (FIXTURES / f"openrouter_image_endpoints_{slug}.json").read_text(encoding="utf-8")
    )
    records = raw.get("endpoints") or [raw]
    return [r for r in records if isinstance(r, dict)]


EVERY_CONTRACT = _contracts()


@pytest.mark.parametrize(("slug", "model_id"), EVERY_CONTRACT, ids=[s for s, _ in EVERY_CONTRACT])
def test_a_rendered_help_panel_names_nothing_only_the_code_can_see(slug, model_id):
    rendered = render_image_help(
        model_id, {"id": model_id, "name": model_id}, endpoint_record=_records(slug)
    )
    assert "## Controls" in rendered, "an empty panel would make this vacuous"
    hits = [(model_id, term, rendered) for term in offences(rendered)]
    assert not hits, f"help panel carries implementation vocabulary:\n{_report(hits)}"


@pytest.mark.parametrize("model_id", sorted(IMAGE_HELP_BY_MODEL))
def test_curated_prose_names_nothing_only_the_code_can_see(model_id):
    rendered = render_image_help(model_id, {"id": model_id, "name": model_id})
    assert rendered.strip(), "the prose must exist for this to mean anything"
    hits = [(model_id, term, rendered) for term in offences(rendered)]
    assert not hits, f"curated help carries implementation vocabulary:\n{_report(hits)}"


def test_the_catalog_fallback_panel_names_nothing_only_the_code_can_see():
    """The fallback labels the catalog's own fields, which is where their names leak."""
    rendered = render_image_help(
        "vendor/unlisted",
        {
            "name": "Unlisted",
            "description": "A model with no curated entry.",
            "architecture": {
                "output_modalities": ["image"],
                "input_modalities": ["text", "image"],
            },
        },
    )
    assert "image" in rendered, "the fallback must still report what the model does"
    hits = [("vendor/unlisted", term, rendered) for term in offences(rendered)]
    assert not hits, f"catalog fallback carries implementation vocabulary:\n{_report(hits)}"


@pytest.mark.parametrize(("slug", "model_id"), EVERY_CONTRACT, ids=[s for s, _ in EVERY_CONTRACT])
def test_every_control_a_filter_draws_is_labelled_in_words_a_user_can_act_on(slug, model_id):
    """Read off the rendered filter, which is the panel the chat UI actually draws."""
    spec = build_image_model_filter_spec(model_id, {"id": model_id, "name": model_id}, _records(slug))
    source = render_image_model_filter_source(spec)
    labels: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg in ("title", "description") and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, str):
                    labels.append((keyword.arg, keyword.value.value))
    assert labels, "a filter with no labelled control would make this vacuous"
    hits = [
        (f"{model_id} {arg}", term, text) for arg, text in labels for term in offences(text)
    ]
    assert not hits, f"a drawn control is labelled with implementation vocabulary:\n{_report(hits)}"


def _key_or_log_constants(tree: ast.Module) -> set[int]:
    """Literals that are a mapping key or a log record, neither of which a user reads."""
    excluded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            logging_call = isinstance(func, ast.Attribute) and func.attr in (
                "log",
                "debug",
                "info",
                "warning",
                "warn",
                "error",
                "exception",
                "critical",
            )
            if logging_call:
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Constant):
                        excluded.add(id(inner))
                continue
            lookup = (isinstance(func, ast.Attribute) and func.attr in ("get", "pop", "setdefault")) or (
                isinstance(func, ast.Name) and func.id in ("getattr", "setattr", "hasattr")
            )
            if lookup:
                for arg in node.args:
                    if isinstance(arg, ast.Constant):
                        excluded.add(id(arg))
        elif isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant):
                excluded.add(id(node.slice))
        elif isinstance(node, ast.Compare):
            if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
                for operand in (node.left, *node.comparators):
                    if isinstance(operand, ast.Constant):
                        excluded.add(id(operand))
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant):
                    excluded.add(id(key))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            excluded.add(id(node.value))
    return excluded


def test_no_text_the_image_adapter_can_show_a_user_names_the_implementation():
    """Every literal in the adapter that is not a key and not a log record.

    A note is assembled from f-string parts at a dozen sites and one of them builds its
    text in a helper that returns a plain tuple, so a check anchored on `_Note(` calls
    would read most of them and silently miss the rest. Anchoring on what a literal is
    NOT -- a mapping key, a log record, a docstring -- leaves the notes, the chat errors
    and nothing that a user cannot reach.
    """
    path = PACKAGE / "integrations" / "image.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _key_or_log_constants(tree)
    hits = [
        (f"image.py:{node.lineno}", term, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in skip
        for term in offences(node.value)
    ]
    assert not hits, f"text the adapter can put in front of a user names the code:\n{_report(hits)}"


def _notes(config: dict[str, Any], record: dict | None) -> list[Any]:
    _top, _provider, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": config},
        allowed_passthrough=("style", "safety_tolerance"),
        record=record,
    )
    return notes


RECORD = {
    "provider_slug": "recraft",
    "supported_parameters": {
        "aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]},
        "n": {"type": "range", "min": 1, "max": 2},
        "output_compression": {"type": "range", "min": 0, "max": 100},
    },
    "allowed_passthrough_parameters": ["style", "safety_tolerance"],
}


def test_a_rejection_note_reaching_the_chat_names_the_setting_not_the_request_key():
    """The must-pass half: the names the user typed survive, and nothing else appears.

    `safety_tolerance` and `style` are published passthrough parameters, `aspect_ratio`
    and `output_compression` are published top-level ones, and all four are drawn as
    controls with exactly those names. A rule that stripped them to avoid `image_config`
    would leave the reader a rejection that does not say what was rejected.
    """
    notes = _notes(
        {
            "aspect_ratio": "4:3",
            "output_compression": 5000,
            "safety_tolerance": 6,
            "style": "digital_illustration",
            "unheard_of": "x",
        },
        RECORD,
    )
    texts = [note.text for note in notes]
    assert texts, "no note means nothing was checked"
    hits = [("note", term, text) for text in texts for term in offences(text)]
    assert not hits, f"a chat notification names the implementation:\n{_report(hits)}"
    joined = " ".join(texts)
    assert "aspect_ratio" in joined, joined
    assert "output_compression" in joined, joined
    for published in PUBLISHED_PARAMETER_NAMES:
        assert not offences(published), (
            f"{published} is a control the user filled in; banning it would delete the "
            "one word that makes the rejection actionable"
        )


def test_the_overflow_note_counts_settings_rather_than_naming_the_request_block():
    """The note raised when there are more rejections than a toast can carry."""
    notes = _notes({f"junk_{index}": "x" for index in range(80)}, RECORD)
    overflow = [note.text for note in notes if note.kind == "overflow"]
    assert overflow, "the remainder must still be reported as a count"
    hits = [("overflow", term, text) for text in overflow for term in offences(text)]
    assert not hits, f"the overflow note names the implementation:\n{_report(hits)}"
    assert "setting" in overflow[0], overflow[0]
