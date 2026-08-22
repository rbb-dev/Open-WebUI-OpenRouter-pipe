"""Every sentence here is one a human reads, checked against what the code actually does.

Each test below defends one property of a user- or admin-facing string, measured by
RENDERING the thing the string describes rather than by re-reading the string. A claim
about "which controls the tool shows" is only worth what the rendering says, and the
eight sentences these tests replaced were each a claim that had drifted from its
rendering without anything noticing.

Where a value could be satisfied by a constant, the test is parametrised over two
distinct values, or asserts a SET rather than a count -- a count is satisfied by the
wrong members.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from open_webui_openrouter_pipe.filters.image_filter_renderer import (
    build_image_model_filter_spec,
    image_gen_model_note,
    render_image_gen_filter_source,
    render_image_model_filter_source,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DOCS = Path(__file__).resolve().parents[1] / "docs"

_FIELD_RE = re.compile(r"^        (IMAGE_[A-Z0-9_]+): (.*?) = Field\($", re.M)
_TITLE_RE = re.compile(r"^                    title=(.+),$", re.M)


def _contracts() -> list[tuple[str, str]]:
    found = []
    for path in sorted(FIXTURES.glob("openrouter_image_endpoints_*.json")):
        found.append((path.stem[len("openrouter_image_endpoints_") :], json.loads(path.read_text())["id"]))
    assert len(found) > 30, f"only {len(found)} contracts; the fleet-wide sweeps went hollow"
    return found


def _records(slug: str) -> list[dict]:
    raw = json.loads((FIXTURES / f"openrouter_image_endpoints_{slug}.json").read_text())
    return [r for r in (raw.get("endpoints") or [raw]) if isinstance(r, dict)]


def _tool_body(model_id: str, records: list[dict] | None) -> str:
    spec = build_image_model_filter_spec(
        model_id, {"id": model_id, "name": model_id}, records, dedicated_image_api=True
    )
    source = render_image_gen_filter_source(
        spec, catalog_match=records is not None, selected_model=model_id
    )
    return source.split("class UserValves(BaseModel):", 1)[1].split("    def __init__", 1)[0]


def _model_body(model_id: str, records: list[dict] | None) -> str:
    spec = build_image_model_filter_spec(
        model_id, {"id": model_id, "name": model_id}, records, dedicated_image_api=True
    )
    source = render_image_model_filter_source(spec)
    return source.split("class UserValves(BaseModel):", 1)[1].split("    def __init__", 1)[0]


def _tool_controls(model_id: str, records: list[dict] | None) -> tuple[list[str], list[str]]:
    body = _tool_body(model_id, records)
    fields = [name for name, _ in _FIELD_RE.findall(body)]
    titles = [t.strip("'\"") for t in _TITLE_RE.findall(body)]
    assert len(fields) == len(titles), body
    return fields, titles


EVERY_CONTRACT = _contracts()

ALWAYS = ("Quality", "Aspect ratio", "Background", "Output format", "Output compression")


@pytest.mark.parametrize(("slug", "model_id"), EVERY_CONTRACT, ids=[s for s, _ in EVERY_CONTRACT])
def test_the_image_tool_draws_five_shared_controls_plus_exactly_one_size_control(slug, model_id):
    """The set the config tab and the docs promise, measured on every recorded model.

    The claim replaced was that the tool shows "a fixed set ... the same on every drawing
    model". It is not fixed: Resolution and Output size are alternatives decided by that
    model's own published tiers, and no model gets both.
    """
    _fields, titles = _tool_controls(model_id, _records(slug))

    missing = [name for name in ALWAYS if name not in titles]
    assert not missing, f"{model_id} is missing {missing}; drawn: {titles}"
    assert ("Resolution" in titles) != ("Output size" in titles), (
        f"{model_id} draws {titles}: exactly one of Resolution / Output size must appear"
    )
    assert len(titles) == 6, f"{model_id} draws {len(titles)} controls: {titles}"


def test_which_models_get_resolution_and_which_get_output_size():
    """Asserted as two SETS, because a count is satisfied by the wrong models."""
    tiered, typed = set(), set()
    for slug, model_id in EVERY_CONTRACT:
        _fields, titles = _tool_controls(model_id, _records(slug))
        (tiered if "Resolution" in titles else typed).add(model_id)

    assert tiered & typed == set()
    assert "google/gemini-3-pro-image" in tiered and "qwen/qwen-image-3" in tiered
    assert "openai/gpt-image-2" in typed and "recraft/recraft-v3" in typed
    assert len(tiered) == 16 and len(typed) == 24, (
        f"the documented split moved: {len(tiered)} tiered / {len(typed)} typed"
    )


@pytest.mark.parametrize("catalog_match", [True, False])
def test_a_model_with_nothing_published_still_gets_all_six_controls(catalog_match):
    """Both unread branches, because a fix that only repaired one leaves the other false.

    The two sentences replaced said "no settings are offered". Six are.
    """
    fields, titles = _tool_controls("vendor/unlisted", None)

    assert len(fields) == 6, fields
    assert set(titles) == {*ALWAYS, "Output size"}, titles

    spec = build_image_model_filter_spec(
        "vendor/unlisted", {"id": "vendor/unlisted"}, None, dedicated_image_api=True
    )
    note = image_gen_model_note(spec, catalog_match=catalog_match)
    assert "no settings are offered" not in note.casefold(), note
    assert "in general" in note, (
        "the reader must be told the controls carry the API's own values, not this "
        f"model's; got {note!r}"
    )


@pytest.mark.parametrize(
    ("slug", "model_id", "control", "published"),
    [
        ("openai_gpt-image-2", "openai/gpt-image-2", "IMAGE_QUALITY", True),
        ("openai_gpt-image-2", "openai/gpt-image-2", "IMAGE_ASPECT_RATIO", True),
        ("recraft_recraft-v3", "recraft/recraft-v3", "IMAGE_QUALITY", False),
        ("google_gemini-3-pro-image", "google/gemini-3-pro-image", "IMAGE_BACKGROUND", False),
    ],
)
def test_a_control_the_model_says_nothing_about_still_names_what_the_api_takes(
    slug, model_id, control, published
):
    """Two published and two unpublished, so a constant answer cannot satisfy the row.

    The documented "Shown when" column said these controls appear only where the model
    publishes a list. They appear either way; what changes is whether the values are a
    dropdown or free text with the API's own list spelled out.
    """
    body = _tool_body(model_id, _records(slug))
    block = body.split(f"{control}: ", 1)[1].split("Field(", 1)[1].split("\n                )", 1)[0]

    assert f"{control}: " in body, f"{control} is not drawn for {model_id}"
    if published:
        assert "Literal[" in body.split(f"{control}: ", 1)[1].split(" = Field", 1)[0]
        assert "OpenRouter's image API takes" not in block
    else:
        assert "OpenRouter's image API takes" in block, block


@pytest.mark.parametrize("panel", ["model", "tool"])
@pytest.mark.parametrize(
    ("slug", "model_id", "publishes_tiers"),
    [
        ("recraft_recraft-v3", "recraft/recraft-v3", False),
        ("google_gemini-3-pro-image", "google/gemini-3-pro-image", True),
    ],
)
def test_the_output_size_control_describes_both_forms_openrouter_accepts(
    slug, model_id, publishes_tiers, panel
):
    """Both forms are named, and the check the box claims is the check it gets.

    OpenRouter's request schema for `size` accepts a tier OR explicit pixels; the
    sentence replaced said "exact pixel dimensions, where the model takes them rather
    than a tier", which tells a reader not to type the tier that works.

    A tier is measured against the model's own published list only where the model
    publishes one. On the 24 of 40 recorded contracts that publish no `resolution`
    descriptor the tier goes out unmeasured against anything of the model's, so a box
    claiming otherwise describes a check that does not happen -- and on those, the
    panel does not draw a Resolution control for the sentence to point at either.

    One row of each kind, because a fixed sentence cannot be right for both.
    """
    body = (_model_body if panel == "model" else _tool_body)(model_id, _records(slug))
    if "IMAGE_SIZE: " not in body:
        assert panel == "tool" and publishes_tiers, (
            f"the {panel} panel for {model_id} drew no size box at all: {body}"
        )
        return
    block = body.split("IMAGE_SIZE: ", 1)[1].split("\n                )", 1)[0]
    resolution_drawn = "IMAGE_RESOLUTION: " in body

    assert resolution_drawn is (publishes_tiers and panel == "model"), (
        f"the {panel} panel for {model_id} drew Resolution={resolution_drawn}; the case "
        "is not the one it was set up to be"
    )
    for tier in ("512", "1K", "2K", "4K"):
        assert tier in block, f"{tier} is a tier this field accepts and is not named: {block}"
    assert "1024x1024" in block, block
    assert "Aspect ratio" in block, (
        f"a tier still takes its shape from Aspect ratio, and exact pixels supersede it; "
        f"the control must say so: {block}"
    )
    assert ("Resolution" in block) is resolution_drawn, (
        f"the box names Resolution={('Resolution' in block)} while the panel draws it "
        f"={resolution_drawn}; a control the reader cannot see must not be named: {block}"
    )
    claims_own_tiers = "checked against the tiers this model publishes" in block
    assert claims_own_tiers is resolution_drawn, (
        f"the box claims the model's own tiers are checked={claims_own_tiers} while this "
        f"panel publishes a tier list={resolution_drawn}; where none is published the "
        f"tier is measured only against OpenRouter's four names: {block}"
    )
    assert "rather than a tier" not in block


def test_the_server_tools_table_lists_exactly_the_controls_the_tool_draws():
    """The doc table named three controls that are never drawn and one that is not a valve.

    Measured against the rendering, in both directions: a row for something never drawn
    misleads as much as a drawn control with no row.
    """
    doc = (DOCS / "openrouter_server_tools.md").read_text(encoding="utf-8")
    section = doc.split("### OpenRouter Image Generation filter user valves", 1)[1].split("\n---\n", 1)[0]
    listed = set(re.findall(r"^\| `(IMAGE_[A-Z_]+)`", section, re.M))
    assert listed, "the table stopped parsing; this guard is asserting nothing"

    drawn: set[str] = set()
    for slug, model_id in EVERY_CONTRACT:
        fields, _titles = _tool_controls(model_id, _records(slug))
        drawn.update(fields)
    drawn.update(_tool_controls("vendor/unlisted", None)[0])

    assert listed == drawn, (
        f"documented but never drawn: {sorted(listed - drawn)}; "
        f"drawn but undocumented: {sorted(drawn - listed)}"
    )
    assert "offers none of these" not in section, (
        "a model with no readable settings still gets all six controls"
    )


def test_the_config_tab_describes_the_controls_the_tool_actually_draws():
    """An admin reading the Config tab must be told which six, and that two are alternatives."""
    pytest.importorskip(
        "open_webui_openrouter_pipe.plugins.pipe_dashboard",
        reason="the --no-plugins artifacts omit pipe_dashboard by design",
    )
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META

    detail = CONFIG_META["ENABLE_IMAGE_GENERATION"]["detail"]

    for title in (*ALWAYS, "Resolution", "Output size"):
        assert title in detail, f"{title!r} is drawn on screen and the Config tab omits it"
    assert "never both" in detail, (
        "Resolution and Output size are alternatives; a reader told about both without "
        "that word will look for a control that is not there"
    )
    assert "size tier" not in detail.replace("list of size tiers", ""), (
        "no control is titled 'size tier'; the tier control is titled Resolution"
    )
    assert "fixed set" not in detail and "same on every drawing model" not in detail


def test_the_image_doc_and_the_config_tab_agree_on_the_control_set():
    """Two surfaces, one claim. They drifted apart once already."""
    pytest.importorskip(
        "open_webui_openrouter_pipe.plugins.pipe_dashboard",
        reason="the --no-plugins artifacts omit pipe_dashboard by design",
    )
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META

    doc = (DOCS / "openrouter_image_generation.md").read_text(encoding="utf-8")
    section = doc.split("## What settings a model offers", 1)[1].split("\n## ", 1)[0]

    assert "same six controls on every drawing model" not in section
    assert "quietly ignored" not in section.split("A request carries at most", 1)[0]
    for title in (*ALWAYS, "Resolution", "Output size"):
        assert f"**{title}**" in section, f"{title!r} is drawn on screen and the doc omits it"
    assert "No model gets\nboth." in section or "No model gets both." in section

    detail = CONFIG_META["ENABLE_IMAGE_GENERATION"]["detail"]
    for title in (*ALWAYS, "Resolution", "Output size"):
        assert (title in detail) == (f"**{title}**" in section), (
            f"{title!r} is claimed on one surface and not the other"
        )


def _video_catalog() -> list[dict]:
    raw = json.loads((FIXTURES / "video_models_catalog.json").read_text(encoding="utf-8"))
    models = [m for m in raw.get("data", []) if isinstance(m, dict) and m.get("id")]
    assert len(models) > 15, f"only {len(models)} video models; the sweep went hollow"
    return models


@pytest.mark.parametrize(
    ("field", "spoken"), [("seed", "seed"), ("generate_audio", "audio")]
)
def test_the_atlas_names_every_model_whose_control_is_shown_on_a_published_null(field, spoken):
    """Asserted as an id SET, never a count: a count is satisfied by the wrong models.

    The sentence replaced named one model for seed and two for audio. Recomputing the
    way the filter builder computes it gives three and four, and only one of the four
    was named.
    """
    from open_webui_openrouter_pipe.integrations.image_types import capability_declared_off
    from open_webui_openrouter_pipe.filters.video_filter_renderer import build_video_filter_spec

    shown_on_null = set()
    for model in _video_catalog():
        declared = model.get(field, "<absent>")
        if declared is None and field in model and not capability_declared_off(declared):
            shown_on_null.add(model["id"])
    assert shown_on_null, "no model publishes null here, so the sentence has nothing to say"

    declared_off = {m["id"] for m in _video_catalog() if m.get(field) is False}
    assert declared_off, "no model publishes false here, so the contrast proves nothing"

    for model_id in shown_on_null | declared_off:
        spec = build_video_filter_spec(
            model_id=model_id, video_model=next(m for m in _video_catalog() if m["id"] == model_id)
        )
        drawn = spec.supports_seed if field == "seed" else spec.supports_generate_audio_toggle
        assert drawn == (model_id in shown_on_null), (
            f"{model_id} publishes {model_id in shown_on_null and 'null' or 'false'} for "
            f"{field} and the control is {'not ' if not drawn else ''}drawn"
        )

    atlas = (DOCS / "valves_and_configuration_atlas.md").read_text(encoding="utf-8")
    sentence = next(
        line for line in atlas.splitlines() if "On the recorded catalog" in line and "seed" in line
    )
    clause = sentence.split(f"the {spoken} control", 1)[1].split(", and the ", 1)[0]

    named = set(re.findall(r"`([^`]+)`", clause))
    assert named == shown_on_null, (
        f"the atlas names {sorted(named)} for {spoken}; the catalog says {sorted(shown_on_null)}"
    )


def test_a_control_whose_limits_nobody_published_says_so_and_names_the_way_round_it():
    """The cap is the pipe's own precaution; a user told nothing reads it as the model's rule."""
    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        _PASSTHROUGH_CONTROLS,
        _UNCONFIRMED_PASSTHROUGH_DOMAINS,
    )

    assert _UNCONFIRMED_PASSTHROUGH_DOMAINS, "nothing to check; the tripwire emptied"
    for control in _PASSTHROUGH_CONTROLS:
        if control.param not in _UNCONFIRMED_PASSTHROUGH_DOMAINS:
            continue
        assert "No page anywhere publishes" in control.description, control.description
        assert "this pipe's own" in control.description, control.description
        assert "Provider options JSON" in control.description, (
            "a user who needs a value outside the cap must be told where to put it"
        )
        assert f"{control.minimum:g} to {control.maximum:g}" in control.description


@pytest.mark.parametrize("kind", ["number", "enum"])
def test_an_undisclosed_unconfirmed_domain_is_refused_at_import(kind):
    """The gate is exercised by handing it a control that hides the gap, not by observing
    that the real table does not trip it. Two kinds, so one branch cannot stand for both."""
    from dataclasses import replace

    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        _CONTROL_ENUM,
        _CONTROL_NUMBER,
        _PASSTHROUGH_CONTROLS,
        _UNCONFIRMED_DOMAIN,
        _unconfirmed_notice,
        _validate_passthrough_controls,
    )

    honest = next(c for c in _PASSTHROUGH_CONTROLS if c.param == "conditioningScale")
    _validate_passthrough_controls((honest,))

    if kind == "number":
        hidden = replace(honest, description="How hard the stills steer the result.")
    else:
        hidden = replace(
            honest,
            kind=_CONTROL_ENUM,
            choices=(("a", _UNCONFIRMED_DOMAIN), ("b", _UNCONFIRMED_DOMAIN)),
            source="",
            description="Pick a or b.",
        )
    with pytest.raises(ValueError):
        _validate_passthrough_controls((hidden,))

    disclosed = replace(
        hidden,
        description=hidden.description + " " + _unconfirmed_notice(
            hidden.kind, hidden.minimum, hidden.maximum
        ),
    )
    _validate_passthrough_controls((disclosed,))


@pytest.mark.parametrize(
    ("model_id", "pixels"),
    [("bytedance/seedance-2.0", 407696), ("bytedance/seedance-1-5-pro", 407696)],
)
def test_every_pixel_floor_records_where_its_number_came_from(model_id, pixels):
    """A number that drops a paid-for attachment must carry its provenance beside it."""
    from open_webui_openrouter_pipe.integrations.video import (
        _FLOOR_OBSERVED_PREFIX,
        _FLOOR_UNPUBLISHED,
        _INPUT_PIXEL_FLOORS,
    )

    floor = _INPUT_PIXEL_FLOORS[model_id]
    assert floor.pixels == pixels
    assert (
        floor.source == _FLOOR_UNPUBLISHED
        or floor.source.startswith(_FLOOR_OBSERVED_PREFIX)
        or floor.source.startswith("https://")
    ), floor.source


@pytest.mark.parametrize("bad_source", ["", "seedance minimum"])
def test_a_pixel_floor_with_no_recorded_provenance_is_refused_at_import(bad_source):
    """Two bad values, so a check that only rejects the empty string cannot pass."""
    from open_webui_openrouter_pipe.integrations.video import (
        _InputPixelFloor,
        _validate_input_pixel_floors,
    )

    _validate_input_pixel_floors({"vendor/ok": _InputPixelFloor(1000, "https://example.invalid/doc")})
    with pytest.raises(ValueError):
        _validate_input_pixel_floors({"vendor/bad": _InputPixelFloor(1000, bad_source)})


@pytest.mark.parametrize(
    ("width", "height"), [(640, 360), (320, 240)]
)
@pytest.mark.asyncio
async def test_the_dropped_clip_message_keeps_the_number_and_says_whose_rule_it_is(
    monkeypatch, width, height
):
    """Two frame sizes, so a hardcoded sentence cannot satisfy both.

    The message must not be softened to "needs a larger frame" while the number is still
    what decides; and it must not present that number as OpenRouter's rule.

    The sentence is read off `_clip_too_small_note`, not rebuilt here: a copy of the
    f-string in the test cannot fail when the production one changes, and every earlier
    shape of this test passed while the note said whatever it liked.
    """
    import base64
    import logging
    from unittest.mock import MagicMock

    from open_webui_openrouter_pipe.integrations import video as video_module
    from open_webui_openrouter_pipe.integrations.video import (
        _INPUT_PIXEL_FLOORS,
        VideoGenerationAdapter,
    )
    from open_webui_openrouter_pipe.media.frame_extraction import VideoMetadata

    floor = _INPUT_PIXEL_FLOORS["bytedance/seedance-2.0"]
    pixels = width * height
    assert pixels < floor.pixels, "this row is not under the floor it tests"

    async def _probe(_path):
        return VideoMetadata(
            duration_seconds=2.0, width=width, height=height, fps=24.0, has_audio=False
        )

    monkeypatch.setattr(video_module, "probe_video", _probe)
    adapter = VideoGenerationAdapter(pipe=MagicMock(), logger=logging.getLogger("clip-note"))
    note = await adapter._clip_too_small_note(
        floor, base64.b64encode(b"\x00\x00\x00\x18ftypmp42").decode(), "video/mp4"
    )

    assert note, "a clip under the floor is dropped, and the user is told nothing"
    assert f"{width}" in note and f"{height}" in note, (
        f"the user cannot tell which attachment was dropped: {note}"
    )
    assert f"{pixels:,}" in note and f"{floor.pixels:,}" in note, (
        "softening the message while the number still decides leaves the user unable to act"
    )
    assert "pipe" in note.casefold(), (
        f"nobody publishes this floor, so the message must attribute it to the pipe: {note}"
    )
    assert not re.search(
        r"\b(this model|the model|openrouter)\b[^.;]{0,40}\b(needs|requires|demands)\b",
        note,
        re.I,
    ), f"the pipe imposes this floor and the message reads as somebody else's rule: {note}"


def test_no_document_sends_an_administrator_to_the_wrong_website():
    """Functions is an Open WebUI screen. This line survived one previous rewrite."""
    offending = []
    for path in sorted(DOCS.rglob("*.md")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "OpenRouter Admin" in line:
                offending.append(f"{path.name}:{number}: {line.strip()}")
    assert not offending, "\n".join(offending)


_VIDEO_CATALOG = json.loads(
    (FIXTURES / "video_models_catalog.json").read_text(encoding="utf-8")
)["data"]

_VIDEO_MODALITIES = json.loads(
    (FIXTURES / "openrouter_video_input_modalities.json").read_text(encoding="utf-8")
)["input_modalities"]


_VIDEO_MODEL_WORDS = sorted(
    {
        word
        for item in _VIDEO_CATALOG
        for word in (item["id"].split("/", 1)[1], item["id"])
    }
    | {"Wan 2.6", "Wan 2.7", "Seedance", "Veo", "Kling", "Hailuo", "Sora", "Aleph", "Grok"},
    key=len,
    reverse=True,
)
"""Every way a document might name a model, so "available on X" cannot slip past."""


def _video_model(model_id: str) -> dict:
    base = next(item for item in _VIDEO_CATALOG if item["id"] == model_id)
    declared = _VIDEO_MODALITIES.get(model_id)
    return dict(base, input_modalities=declared) if declared is not None else dict(base)


def _video_field(model_id: str, field: str) -> "tuple[tuple[Any, ...], str] | None":
    import ast

    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        render_video_filter_source,
    )

    source = render_video_filter_source(
        model_id=model_id, video_model=_video_model(model_id)
    )
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.AnnAssign) or getattr(node.target, "id", "") != field:
            continue
        annotation = node.annotation
        offered = annotation.slice if isinstance(annotation, ast.Subscript) else None
        choices = (
            tuple(item.value for item in offered.elts if isinstance(item, ast.Constant))
            if isinstance(offered, ast.Tuple)
            else ()
        )
        call = node.value if isinstance(node.value, ast.Call) else None
        described = next(
            (
                keyword.value.value
                for keyword in (call.keywords if call else [])
                if keyword.arg == "description" and isinstance(keyword.value, ast.Constant)
            ),
            "",
        )
        return choices, str(described)
    return None


@pytest.mark.parametrize(
    ("model_id", "offers_first_last"),
    [("kwaivgi/kling-video-o1", True), ("minimax/hailuo-2.3", False)],
)
def test_the_frames_control_names_only_the_modes_it_offers(model_id, offers_first_last):
    """The sentence and the dropdown come off one list, so neither can name what the other does not.

    Seven of the sixteen models with frame support publish `first_frame` and no
    `last_frame`. The description was a fixed string naming `first_last`, so on those
    seven it told the reader to pin a closing still with a choice the dropdown does not
    contain -- and the `help` panel for the same models correctly said the opposite.

    One row of each kind, so a fixed sentence cannot satisfy both.
    """
    found = _video_field(model_id, "VIDEO_FRAME_MODE")
    assert found is not None, f"{model_id} draws no Frames control to check"
    choices, described = found

    assert ("first_last" in choices) is offers_first_last, (
        f"{model_id} was chosen as the offers_first_last={offers_first_last} row and its "
        f"choices are {choices}; the row proves nothing"
    )
    for mode in ("auto", "none", "first_only", "first_last"):
        assert (mode in described) is (mode in choices), (
            f"{model_id} offers {choices} and its description names {mode}="
            f"{mode in described}: {described}"
        )


@pytest.mark.parametrize(
    ("model_id", "publishes_the_flag"),
    [("google/veo-3.1", True), ("x-ai/grok-imagine-video", False)],
)
def test_the_seed_control_hedges_the_way_openrouter_hedges(model_id, publishes_the_flag):
    """No model is told a seed guarantees a repeat, because OpenRouter does not say so.

    Their video request schema says repeated requests with the same seed "should" return
    the same result and that "Determinism is not guaranteed for all providers", so the
    control cannot promise one. It said "make the same clip again".

    The non-guarantee is per-provider and covers all sixteen; the three that publish no
    seed flag at all owe the reader one sentence more, which is the distinction the
    `help` panel already draws from the same field. One row of each.
    """
    found = _video_field(model_id, "VIDEO_SEED")
    assert found is not None, f"{model_id} draws no Seed control to check"
    _choices, described = found

    assert not re.search(r"\bnumber makes? the same clip\b", described), (
        f"the control states the repeat as fact; OpenRouter's own schema says only that "
        f"it *should* happen: {described}"
    )
    assert re.search(r"\bnot guarantee\b", described), (
        f"every one of the sixteen must carry the vendor's own non-guarantee: {described}"
    )
    # Measured against the other row's rendering rather than against a phrase, so any
    # rewording of the extra sentence survives and only its DISAPPEARANCE fails.
    other = next(
        _video_field(other_id, "VIDEO_SEED")
        for other_id, publishes in (("google/veo-3.1", True), ("x-ai/grok-imagine-video", False))
        if publishes is not publishes_the_flag
    )
    assert other is not None
    _other_choices, other_described = other
    if publishes_the_flag:
        assert len(described) < len(other_described), (
            f"{model_id} publishes the flag, so its control owes the reader LESS than the "
            f"one whose model publishes nothing: {described!r} vs {other_described!r}"
        )
    else:
        assert len(described) > len(other_described), (
            f"{model_id} publishes no seed flag at all, so its control owes the reader a "
            f"sentence the published one does not carry: {described!r} vs "
            f"{other_described!r}"
        )


@pytest.mark.parametrize(
    ("field", "control"),
    [
        ("VIDEO_REFERENCE_VIDEO_URL", "Reference video URL"),
        ("VIDEO_REFERENCE_VIDEOS_JSON", "Reference videos JSON"),
        ("VIDEO_AUDIO_URL", "Audio reference URL"),
    ],
)
def test_no_document_offers_a_video_reference_control_no_model_draws(field, control):
    """A control drawn on nothing must not be documented as available on something.

    OpenRouter declares Wan 2.6 and Wan 2.7 as taking only text and pictures, so the
    gate on declared input modalities withholds all three of these -- they now render on
    zero models. The tables still named Wan 2.6 / Wan 2.7 in their "Exposed on" column,
    and the help tips still told users to supply the references.

    Measured by rendering every model in the catalog rather than by reading the tables,
    so the day a model does declare video or audio input this test stops demanding the
    documents deny it.
    """
    drawn = [
        item["id"] for item in _VIDEO_CATALOG if _video_field(item["id"], field) is not None
    ]
    if drawn:
        pytest.skip(f"{field} now renders on {drawn}; the documents may name them")

    rows = []
    for path in sorted(DOCS.rglob("*.md")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.startswith(f"| `{field}` |"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            rows.append((f"{path.name}:{number}", cells[-1]))
    assert rows, f"no table anywhere documents {field}; the check is hollow"

    named = re.compile(
        r"\b(?:" + "|".join(re.escape(part) for part in _VIDEO_MODEL_WORDS) + r")\b", re.I
    )
    denied = re.compile(r"\b(?:none|no|not|never|neither)\b", re.I)

    offending = []
    for where, exposed in rows:
        says_no = denied.search(exposed)
        names_one = named.search(exposed)
        if says_no is None or (names_one is not None and names_one.start() < says_no.start()):
            offending.append(f"{where}: {exposed}")
    assert not offending, (
        f"{control} is drawn on no model at all; these rows present a model as a place "
        f"it is available, or say nothing about its being unavailable:\n"
        + "\n".join(offending)
    )


def test_the_withheld_aspect_ratio_notice_does_not_blame_openrouter():
    """The pipe withholds the ratio; OpenRouter publishes no rule that it must.

    Their video schema says only that `size` is "interchangeable with resolution +
    aspect_ratio". The rejection language -- "a mismatched resolution or aspect_ratio
    alongside it is rejected with a 400" -- belongs to the IMAGE API. The threshold is
    the pipe's own 2.5% tolerance, so a notice reading "which the video API rejects"
    told the user a false thing about a third party.
    """
    from open_webui_openrouter_pipe.integrations.video import (
        _SIZE_CONTRADICTS_THE_RATIO,
        _SIZE_FIXES_THE_PIXELS,
        _SIZE_IS_A_TIER,
    )

    for notice in (_SIZE_CONTRADICTS_THE_RATIO, _SIZE_FIXES_THE_PIXELS, _SIZE_IS_A_TIER):
        assert not re.search(
            r"\b(the video api|openrouter)\b[^.;]{0,40}\b(rejects?|refuses?|will not take)\b",
            notice,
            re.I,
        ), f"the pipe made this rule and the notice reads as OpenRouter's: {notice}"
    assert "pipe" in _SIZE_CONTRADICTS_THE_RATIO.casefold(), (
        f"a withholding nobody else asked for must say who asked for it: "
        f"{_SIZE_CONTRADICTS_THE_RATIO}"
    )


_TRIMMING_WORDS = re.compile(r"\btrim\w*\b|\bcompress\w*\b|\bmiddle[-_ ]?out\b|\bcontext window\b", re.I)


def _render_rejection_template(*, include_model_limits: bool) -> str:
    from open_webui_openrouter_pipe.core.config import DEFAULT_OPENROUTER_ERROR_TEMPLATE
    from open_webui_openrouter_pipe.core.errors import (
        OpenRouterAPIError,
        _build_error_template_values,
    )
    from open_webui_openrouter_pipe.core.utils import _render_error_template

    error = OpenRouterAPIError(
        status=400,
        reason="Bad Request",
        openrouter_message="This endpoint's maximum context length is 400000 tokens.",
    )
    values = _build_error_template_values(
        error,
        heading="Anthropic: anthropic/claude-3",
        diagnostics=[],
        metrics={},
        model_identifier="anthropic/claude-3",
        normalized_model_id=None,
        api_model_id=None,
    )
    values["include_model_limits"] = include_model_limits
    return _render_error_template(DEFAULT_OPENROUTER_ERROR_TEMPLATE, values)


@pytest.mark.parametrize(
    "status, reason, message",
    [
        (403, "Forbidden", "Your key is not permitted to use this model."),
        (404, "Not Found", "No allowed provider serves this model under your data policy."),
        (400, "Bad Request", "Only HTTPS URLs are allowed for a video reference."),
    ],
)
def test_an_error_unrelated_to_context_length_is_given_no_trimming_advice(status, reason, message):
    """The rejected-request template is the fallback for every status without one of its own.

    Everything but 401, 402, 408, 413, 429 and 5xx lands on it -- and the chat orchestrator
    passes it explicitly, so those statuses land on it too -- which means its unconditional
    closing line is read by a forbidden key, an unroutable model and a malformed reference URL
    alike. It closed by telling all of them to ask an admin to enable a control that no longer
    exists, advice that is both irrelevant to the failure and unfindable in the admin UI.
    """
    from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError

    md = OpenRouterAPIError(status=status, reason=reason, openrouter_message=message).to_markdown()

    assert message in md, "the message did not render, so the rest of this proves nothing"
    found = _TRIMMING_WORDS.search(md)
    assert found is None, (
        f"a {status} carrying {message!r} was closed with context-length advice: {found.group(0)!r}"
    )


def test_context_length_advice_renders_only_inside_the_model_limits_block():
    """Advice about prompt length belongs to the one block a context overflow renders.

    The two renders below are the same error under the block's own guard,
    ``include_model_limits``, set each way, so they differ by that block alone. Whether the
    guard fires for a given upstream message is decided in ``core/errors.py`` and is not what
    this defends; what it defends is that no wording outside the block talks about length.
    """
    with_limits = _render_rejection_template(include_model_limits=True)
    without_limits = _render_rejection_template(include_model_limits=False)

    advice = _TRIMMING_WORDS.search(with_limits)
    assert advice is not None, "a context overflow is told nothing about how to make it fit"
    assert with_limits.index("**Model limits:**") < advice.start(), (
        "the trimming advice escaped the limits block and is unconditional again"
    )
    leaked = _TRIMMING_WORDS.search(without_limits)
    assert leaked is None, (
        f"an error with no known context limits still carries {leaked.group(0)!r}"
    )


def test_the_context_length_advice_names_the_control_an_admin_can_actually_find():
    """`AUTO_CONTEXT_TRIMMING` is a Python identifier; no admin can search the panel for it.

    The name is taken from the dashboard's own metadata rather than quoted, so renaming the
    valve on one surface and not the other reddens this. The retired name this replaced --
    "the middle-out option" -- was not merely differently worded, it named a control that had
    been migrated away to OpenRouter's `context-compression` plugin.
    """
    pytest.importorskip(
        "open_webui_openrouter_pipe.plugins.pipe_dashboard",
        reason="the --no-plugins artifacts omit pipe_dashboard by design",
    )
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META

    control = CONFIG_META["AUTO_CONTEXT_TRIMMING"]["title"]
    assert control in _render_rejection_template(include_model_limits=True), (
        f"the limits advice does not name {control!r}, the label the config tab shows for the "
        "control that fixes an over-long prompt"
    )
    assert control not in _render_rejection_template(include_model_limits=False)
