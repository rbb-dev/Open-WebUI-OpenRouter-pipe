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

import ast
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError
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

CONTRACTS_WITH_KNOBS = [
    (slug, model_id) for slug, model_id in EVERY_CONTRACT if model_id != "meta/muse-image"
]
"""`EVERY_CONTRACT` minus the one model whose contract publishes no settable parameter.

`meta/muse-image` is agentic and exposes no framing, size, count or provider parameter. The
nodes below assert that a panel draws MORE than the model published, and that the drawn list
ranks in the documented order; both carry their own guard -- "publishes nothing, so a superset
claim proves nothing" -- which fires correctly on it. Excluding it keeps those guards meaning
what they say. `tests/test_image_generation.py::test_the_no_knob_models_are_exactly_the_ones_that_publish_nothing`
measures this membership against the contracts, so a model that LOSES its parameters reddens
rather than silently dropping out of these sweeps.
"""


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


_TIERED_MODELS = frozenset({
    "bytedance-seed/seedream-5-0-lite",
    "bytedance-seed/seedream-5-0-pro",
    "x-ai/grok-imagine-image-2.0",
    "bytedance-seed/seedream-4.5",
    "google/gemini-3-pro-image",
    "google/gemini-3-pro-image-preview",
    "google/gemini-3.1-flash-image",
    "google/gemini-3.1-flash-image-preview",
    "google/gemini-3.1-flash-lite-image",
    "krea/krea-2-large",
    "krea/krea-2-medium",
    "krea/krea-2-medium-turbo",
    "qwen/qwen-image-3",
    "qwen/qwen-image-3-pro",
    "sourceful/riverflow-v2-fast",
    "sourceful/riverflow-v2-pro",
    "sourceful/riverflow-v2.5-fast",
    "sourceful/riverflow-v2.5-pro",
    "x-ai/grok-imagine-image-quality",
})

_TYPED_MODELS = frozenset({
    "meta/muse-image",
    "microsoft/mai-image-2.6",
    "microsoft/mai-image-2.6-flash",
    "openai/gpt-image-2.5-flare",
    "openai/gpt-image-2.5-sunburst",
    "recraft/recraft-v4-styles",
    "recraft/recraft-v4-styles-pro",
    "recraft/recraft-v4-styles-pro-vector",
    "recraft/recraft-v4-styles-vector",
    "black-forest-labs/flux.2-flex",
    "black-forest-labs/flux.2-klein-4b",
    "black-forest-labs/flux.2-max",
    "black-forest-labs/flux.2-pro",
    "google/gemini-2.5-flash-image",
    "microsoft/mai-image-2.5",
    "microsoft/mai-image-2.5-pro",
    "openai/gpt-5-image",
    "openai/gpt-5-image-mini",
    "openai/gpt-5.4-image-2",
    "openai/gpt-image-1",
    "openai/gpt-image-1-mini",
    "openai/gpt-image-2",
    "recraft/recraft-v3",
    "recraft/recraft-v4",
    "recraft/recraft-v4-pro",
    "recraft/recraft-v4-pro-vector",
    "recraft/recraft-v4-vector",
    "recraft/recraft-v4.1",
    "recraft/recraft-v4.1-pro",
    "recraft/recraft-v4.1-pro-vector",
    "recraft/recraft-v4.1-utility",
    "recraft/recraft-v4.1-utility-pro",
    "recraft/recraft-v4.1-vector",
})


def test_which_models_get_resolution_and_which_get_output_size():
    """Asserted as two SETS, because a count is satisfied by the wrong models.

    The membership is written out in full. An `assert tiered & typed == set()` stood here
    and could not fail: every recorded model id is distinct and the loop writes each into
    exactly one of the two sets, so the intersection is empty however the panels render.
    A count could not fail usefully either -- sixteen and twenty-four are satisfied by any
    sixteen and any twenty-four. Naming both sets is the only form in which a model moving
    from one control to the other reddens.
    """
    tiered, typed = set(), set()
    for slug, model_id in EVERY_CONTRACT:
        _fields, titles = _tool_controls(model_id, _records(slug))
        (tiered if "Resolution" in titles else typed).add(model_id)

    assert tiered == _TIERED_MODELS, (
        f"the tiered half moved: {sorted(tiered - _TIERED_MODELS)} joined and "
        f"{sorted(_TIERED_MODELS - tiered)} left"
    )
    assert typed == _TYPED_MODELS, (
        f"the typed half moved: {sorted(typed - _TYPED_MODELS)} joined and "
        f"{sorted(_TYPED_MODELS - typed)} left"
    )
    assert _TIERED_MODELS & _TYPED_MODELS == frozenset(), (
        "the two recorded halves overlap, so one of them is mistyped"
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


_NOTE_CONTRACTS: dict[str, list[dict]] = {
    "publishes-nothing": [{"provider_slug": "solo", "supported_parameters": {}}],
    "providers-disagree": [
        {
            "provider_slug": "one",
            "supported_parameters": {"quality": {"type": "enum", "values": ["high"]}},
        },
        {
            "provider_slug": "two",
            "supported_parameters": {"background": {"type": "enum", "values": ["opaque"]}},
        },
    ],
    "publishes-values": [
        {
            "provider_slug": "solo",
            "supported_parameters": {"quality": {"type": "enum", "values": ["high", "low"]}},
        },
    ],
}


def test_the_image_gen_note_never_promises_fewer_settings_than_the_panel_draws():
    """This note is the description of the filter, printed directly above its six controls.

    Two branches of it said "only the ones every model carries are offered" for a contract
    that WAS read and yielded nothing, while the panel underneath drew all six. The flag
    those branches were gated on counts published enums, ranges and passthroughs only --
    and when a model publishes none of those, every one of the six is drawn from the API
    schema instead. So the flag goes false in exactly the case where all six are still on
    the screen.

    Measured against the RENDERED panel, over one contract that publishes values and two
    that do not, so a note naming a fixed list of controls fails on the case whose panel
    names a different one. The two empty causes must stay apart: waiting for providers to
    agree and accepting that a model has nothing to publish are different remedies.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        image_gen_model_note,
    )

    notes: dict[str, str] = {}
    published: dict[str, bool] = {}
    for label, records in _NOTE_CONTRACTS.items():
        spec = build_image_model_filter_spec(
            "vendor/probe", {"id": "vendor/probe"}, records, dedicated_image_api=True
        )
        assert spec.has_knobs is (label == "publishes-values"), (
            f"{label} drives the wrong branch of the note: has_knobs={spec.has_knobs}"
        )
        notes[label] = image_gen_model_note(spec, catalog_match=True)
        published[label] = spec.published_any_parameter

        _fields, titles = _tool_controls("vendor/probe", records)
        assert titles, f"{label} drew no controls at all, so this proves nothing"
        unnamed = [title for title in titles if title not in notes[label]]
        assert not unnamed, (
            f"the {label} panel draws {titles} and the note above it never names "
            f"{unnamed}, so a reader is promised a smaller set than the one in front of "
            f"them: {notes[label]!r}"
        )

    assert len(set(notes.values())) == len(notes), (
        f"two of these contracts produce the same sentence, so the note is not reading "
        f"the contract at all: {notes}"
    )
    assert published["publishes-nothing"] != published["providers-disagree"], (
        "both empty causes now report the same flag, so the note has no way to tell an "
        "operator whether to wait for the providers to agree or to accept that the model "
        "publishes nothing"
    )


def _size_records(published: dict[str, tuple[str, ...] | None]) -> list[dict]:
    """One record per company, publishing the tier list it was given."""
    return [
        {
            "provider_slug": slug,
            "provider_tag": slug,
            "supported_parameters": (
                {} if values is None else {"resolution": {"type": "enum", "values": list(values)}}
            ),
        }
        for slug, values in published.items()
    ]


_GEN_TIER_CONTRACTS: dict[str, list[dict]] = {
    "companies-sharing-a-list": _size_records({"alpha": ("1K", "2K"), "beta": ("1K", "2K", "4K")}),
    "companies-with-different-lists": _size_records({"alpha": ("1K", "2K"), "beta": ("4K",)}),
    "one-company-with-no-list": _size_records({"alpha": None}),
}


def _rendered_description(body: str, field: str) -> str:
    """The description Open WebUI will show for one field of the rendered filter."""
    block = body.split(f"        {field}: ", 1)
    assert len(block) == 2, f"{field} is not drawn at all:\n{body}"
    match = re.search(r"^ {20}description=(.+),$", block[1], re.M)
    assert match is not None, f"{field} carries no description:\n{block[1]}"
    return cast(str, ast.literal_eval(match.group(1)))


@pytest.mark.parametrize("case", sorted(_GEN_TIER_CONTRACTS))
def test_the_gen_note_and_the_size_field_read_one_classification_of_the_tiers(case):
    """The note above the panel and the box inside it, on the same rendered filter.

    They classified the same contract twice by different tests: the box asked whether the
    companies publish tiers between them, the note asked only whether one list survived
    the intersection. Where the companies publish different lists those answers differ,
    and the reader was told the model publishes no tiers directly above a box saying a
    tier is checked against every tier they publish. Both values here are read back off
    the RENDERED filter -- the title the panel draws and the sentence in the box -- and
    the three contracts land on three different classifications, so no fixed clause
    satisfies them.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        _SIZE_MEANING,
        _valve_name,
        IMAGE_KNOB_TITLES,
        gen_tier_clause,
    )
    from open_webui_openrouter_pipe.integrations.image_types import TIER_EQUIVALENT

    model_id = "vendor/probe"
    records = _GEN_TIER_CONTRACTS[case]
    spec = build_image_model_filter_spec(
        model_id, {"id": model_id}, records, dedicated_image_api=True
    )
    body = _tool_body(model_id, records)
    _fields, titles = _tool_controls(model_id, records)
    note = image_gen_model_note(spec, catalog_match=True)

    tiered = IMAGE_KNOB_TITLES[TIER_EQUIVALENT["size"]][0]
    plain = IMAGE_KNOB_TITLES["size"][0]
    drawn = [title for title in titles if title in (tiered, plain)]
    assert drawn == sorted(set(drawn)) and len(drawn) == 1, (
        f"{case}: the panel draws {drawn}, so there is no one size control to describe"
    )

    if drawn[0] == tiered:
        state = "own"
    else:
        shown = _rendered_description(body, _valve_name("size"))
        states = {
            key for (key, _ratio), (clause, _pixels) in _SIZE_MEANING.items() if clause in shown
        }
        assert len(states) == 1, (
            f"{case}: the box reads as {states or 'none'} of the known classifications: {shown!r}"
        )
        state = states.pop()

    clause = gen_tier_clause(state, model_id)
    assert clause in note, (
        f"{case}: the panel draws {drawn[0]} and its box reads as {state!r}, and the note "
        f"above it accounts for the size control differently: {note!r}"
    )
    assert clause.startswith(drawn[0]), (
        f"{case}: the note names {clause.split(',')[0]!r} where the panel draws "
        f"{drawn[0]!r}"
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
    ("slug", "model_id", "publishes_tiers", "publishes_ratio"),
    [
        ("recraft_recraft-v3", "recraft/recraft-v3", False, True),
        ("google_gemini-3-pro-image", "google/gemini-3-pro-image", True, True),
        ("openai_gpt-5-image", "openai/gpt-5-image", False, False),
    ],
)
def test_the_output_size_control_describes_both_forms_openrouter_accepts(
    slug, model_id, publishes_tiers, publishes_ratio, panel
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

    The same holds for the shape. Three recorded contracts -- the OpenAI `gpt-5-image`
    family -- publish no `aspect_ratio` descriptor, so their per-model panel draws no
    Aspect ratio control and a box telling the reader to set the shape there points at
    nothing. The tool panel draws one for every model, so the row must be measured
    against the rendering rather than against the contract alone.

    One row of each kind, because a fixed sentence cannot be right for all three.
    """
    body = (_model_body if panel == "model" else _tool_body)(model_id, _records(slug))
    if "IMAGE_SIZE: " not in body:
        assert panel == "tool" and publishes_tiers, (
            f"the {panel} panel for {model_id} drew no size box at all: {body}"
        )
        return
    block = body.split("IMAGE_SIZE: ", 1)[1].split("\n                )", 1)[0]
    resolution_drawn = "IMAGE_RESOLUTION: " in body
    ratio_drawn = "IMAGE_ASPECT_RATIO: " in body

    assert resolution_drawn is (publishes_tiers and panel == "model"), (
        f"the {panel} panel for {model_id} drew Resolution={resolution_drawn}; the case "
        "is not the one it was set up to be"
    )
    assert ratio_drawn is (publishes_ratio or panel == "tool"), (
        f"the {panel} panel for {model_id} drew Aspect ratio={ratio_drawn}; the case is "
        "not the one it was set up to be"
    )
    for tier in ("512", "1K", "2K", "4K"):
        assert tier in block, f"{tier} is a tier this field accepts and is not named: {block}"
    assert "1024x1024" in block, block
    assert ("Aspect ratio" in block) is ratio_drawn, (
        f"the box names Aspect ratio={('Aspect ratio' in block)} while the panel draws it "
        f"={ratio_drawn}; a tier takes its shape from that control and exact pixels "
        f"supersede it, but only where the reader can see it: {block}"
    )
    assert ("Resolution" in block) is resolution_drawn, (
        f"the box names Resolution={('Resolution' in block)} while the panel draws it "
        f"={resolution_drawn}; a control the reader cannot see must not be named: {block}"
    )
    from open_webui_openrouter_pipe.filters.image_filter_renderer import _SIZE_MEANING

    own_clauses = {
        clause for (state, _ratio), (clause, _pixels) in _SIZE_MEANING.items() if state == "own"
    }
    claims_own_tiers = any(clause in block for clause in own_clauses)
    assert claims_own_tiers is resolution_drawn, (
        f"the box claims the model's own limit is checked={claims_own_tiers} while this "
        f"panel publishes a tier list={resolution_drawn}; where nothing is published for "
        f"size the tier is measured only against OpenRouter's four names: {block}"
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


def _image_doc() -> str:
    return (DOCS / "openrouter_image_generation.md").read_text(encoding="utf-8")


def _atlas() -> str:
    return (DOCS / "valves_and_configuration_atlas.md").read_text(encoding="utf-8")


def _doc_section(doc: str, heading: str) -> str:
    return doc.split(heading, 1)[1].split("\n### ", 1)[0].split("\n## ", 1)[0]


def _claims(text: str) -> list[str]:
    """One claim per paragraph or per top-level bullet, flattened, carrying its heading.

    Splitting prose on "." cuts `9:19.5` in half and cuts a semicolon-joined clause away
    from the subject it qualifies, so a claim spread over two source lines was invisible to
    a sentence-level scan. Splitting only on blank lines is the opposite error: a list of
    twenty recommendation bullets is one block, and a bullet naming one model is then
    "corrected" by a neighbouring bullet naming another. So a bulleted block is split per
    bullet, continuation lines included. The heading rides along because a per-model
    section names its model only in the heading.
    """
    claims, heading = [], ""
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        if block.lstrip().startswith("#"):
            heading = " ".join(block.split())
            continue
        items, current = [], []
        for line in block.splitlines():
            if re.match(r"^\s{0,3}[-*] ", line) and current:
                items.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            items.append(current)
        for item in items:
            flat = " ".join(" ".join(item).split())
            if flat:
                claims.append(f"{heading} {flat}".strip())
    return claims


def _enum_values(records: list[dict], name: str) -> set[str]:
    values: set[str] = set()
    for record in records:
        descriptor = (record.get("supported_parameters") or {}).get(name) or {}
        if descriptor.get("type") == "enum":
            values.update(str(v) for v in (descriptor.get("values") or []))
    return values


def _models_publishing_ratios(*wanted: str) -> set[str]:
    found = set()
    for slug, model_id in EVERY_CONTRACT:
        if set(wanted) <= _enum_values(_records(slug), "aspect_ratio"):
            found.add(model_id)
    return found


@pytest.mark.parametrize("ratio", ["9:19.5", "9:20"])
def test_the_doc_does_not_call_a_tall_ratio_exclusive_to_one_model(ratio):
    """"Only" steers an admin away from capability they already have.

    Measured against every recorded contract rather than against the sentence: two models
    publish these, and the pipe's own in-chat help names the second one explicitly, so the
    doc calling them Grok-only contradicted a surface the same admin reads. Parametrised
    over both ratios, so a fix that named the second model for one of them still fails.
    """
    publishers = _models_publishing_ratios(ratio)
    assert len(publishers) > 1, (
        f"only {sorted(publishers)} publishes {ratio}; an exclusivity claim would be true "
        "and this guard has nothing to defend"
    )

    families = {model_id: model_id.split("/", 1)[1].split("-")[0] for model_id in publishers}
    mentioning = [claim for claim in _claims(_image_doc()) if ratio in claim]
    assert mentioning, f"the doc never mentions {ratio}; the recommendation cannot be checked"

    for claim in mentioning:
        missing = [
            model_id
            for model_id, family in families.items()
            if not re.search(family, claim, re.I)
        ]
        assert not missing, (
            f"a passage offers {ratio} without naming {sorted(missing)}, which publish it "
            f"too: {claim!r}"
        )


def test_the_doc_never_tells_a_reader_to_raise_a_number_of_images_a_model_fixes_at_one():
    """Advice to "bump to 3-5" on a model whose contract fixes n at 1 sends them hunting.

    Measured by RENDERING that model's controls: the range control is only drawn when the
    published maximum exceeds the minimum, so on this contract there is no control at all.
    Swept over every recorded model, so the guard follows the fleet rather than one id.
    """
    fixed_at_one = set()
    for slug, model_id in EVERY_CONTRACT:
        descriptor = None
        for record in _records(slug):
            descriptor = (record.get("supported_parameters") or {}).get("n") or descriptor
        if isinstance(descriptor, dict) and descriptor.get("min") == descriptor.get("max") == 1:
            _fields, titles = _tool_controls(model_id, _records(slug))
            assert "Number of images" not in titles, (
                f"{model_id} fixes n at 1 and a control was drawn anyway: {titles}"
            )
            fixed_at_one.add(model_id)
    assert len(fixed_at_one) > 5, f"only {sorted(fixed_at_one)} fix n at 1; the sweep went hollow"

    doc = _image_doc()
    grok = _doc_section(doc, "### SpaceXAI: Grok Imagine Image Quality")
    assert "x-ai/grok-imagine-image-quality" in fixed_at_one, (
        "the recorded contract no longer fixes n at 1; re-derive this section"
    )
    assert not re.search(r"bump to \d", grok), grok
    assert not re.search(r"`n` fans out", grok), grok
    assert "one image per request" in grok.lower(), grok


def test_no_surface_attributes_the_text_streaming_status_to_a_model_that_cannot_stream():
    """The handler is real and documented by OpenRouter; the attribution was the defect.

    Measured from the contracts: the endpoints publishing streaming are named, and whether
    any vector model is among them is computed rather than asserted. The day a vendor turns
    streaming on for one, this guard stops demanding the disclaimer.
    """
    streaming, vector = set(), set()
    for slug, model_id in EVERY_CONTRACT:
        if any(record.get("supports_streaming") is True for record in _records(slug)):
            streaming.add(model_id)
        if "vector" in model_id:
            vector.add(model_id)
    assert streaming and vector, f"streaming={sorted(streaming)} vector={sorted(vector)}"

    reachable = streaming & vector
    checked = 0
    for text in (_image_doc(), _atlas()):
        for claim in _claims(text):
            if "Drawing the image" not in claim:
                continue
            checked += 1
            names_vector = re.search(r"\bvector\b", claim, re.I)
            disclaimed = re.search(
                r"nothing reaches|no recorded contract|does both|cannot reach|"
                r"supports_streaming`?: ?false",
                claim,
                re.I,
            )
            if reachable:
                assert not disclaimed, (
                    f"{sorted(reachable)} publish streaming, so the disclaimer is now false: "
                    f"{claim!r}"
                )
            else:
                assert not names_vector or disclaimed, (
                    f"the status is attributed to vector models, none of which publish "
                    f"streaming: {claim!r}"
                )
    assert checked >= 3, f"only {checked} passages mention the status; the sweep went hollow"


_REPORT_PROMISE_RE = re.compile(r"you are told|tells you|told afterwards|told which", re.I)

_NARROWED_LEAD_IN_RE = re.compile(r"only some", re.I)


def test_every_surface_agrees_that_a_narrowed_value_is_sent_rather_than_reported():
    """Four surfaces promised a report of a value that is in fact sent and routed.

    The behaviour is measured, not read: for the one recorded model whose providers differ,
    the adapter pins the request to the provider that accepts the value and returns nothing
    to report. A surface promising to name it afterwards describes a message the pipe has no
    way to produce -- nothing checks which provider served the request.

    The two documents were swept when the promise came out, and the Config tab has a test of
    its own; the CONTROL was not, and reverting its sentence left the whole suite green. It
    is the only one of the four a user reads at the moment of choosing, so it is read here
    too -- and by RENDERING it, in both places it reaches a reader: the filter Open WebUI
    draws the box from, and the help panel's own prose. Asserting on the helper that builds
    the sentence would pin the helper and let the promise reappear in the help panel around
    it. The scan is of the whole rendered surface, so a reflow cannot move the promise onto
    a line the filter skipped; the lead-in is only used to prove the sentence is still there.
    """
    from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter
    from open_webui_openrouter_pipe.integrations.image_help import render_image_help

    narrowed = []
    for slug, model_id in EVERY_CONTRACT:
        records = _records(slug)
        spec = build_image_model_filter_spec(
            model_id, {"id": model_id, "name": model_id}, records, dedicated_image_api=True
        )
        for name, values in spec.narrowed:
            narrowed.append((model_id, records, name, values))
    assert narrowed, "no recorded model narrows a value; this guard has nothing to measure"

    for model_id, records, name, values in narrowed:
        provider: dict[str, Any] = {}
        notes = ImageGenerationAdapter._pin_accepting_providers(
            provider, records, {name: values[0]}
        )
        assert notes == [], f"{model_id} reported {notes} for {name}={values[0]!r}"
        assert provider.get("only"), (
            f"{model_id} neither pinned nor reported {name}={values[0]!r}"
        )

    checked = 0
    for text in (_image_doc(), _atlas()):
        for claim in _claims(text):
            if not _NARROWED_LEAD_IN_RE.search(claim):
                continue
            checked += 1
            assert not _REPORT_PROMISE_RE.search(claim), (
                f"a surface still promises a report that never happens: {claim!r}"
            )
    assert checked >= 3, f"only {checked} passages describe it; the sweep went hollow"

    rendered = 0
    for model_id, records, _name, _values in narrowed:
        model = {"id": model_id, "name": model_id}
        for surface, text in (
            ("the model's own image filter", _model_body(model_id, records)),
            (
                "the in-chat help panel",
                render_image_help(
                    model_id, model, endpoint_record=records, dedicated_image_api=True
                ),
            ),
        ):
            assert _NARROWED_LEAD_IN_RE.search(text), (
                f"{surface} for {model_id} no longer carries the sentence about a value "
                f"only some of them take, so this guard is filtering on wording the "
                f"surface has stopped using and is checking nothing:\n{text}"
            )
            rendered += 1
            found = _REPORT_PROMISE_RE.search(text)
            assert found is None, (
                f"{surface} for {model_id} still promises a report that never happens: "
                f"{found.group(0)!r} in:\n{text}"
            )
    assert rendered >= 2, (
        f"only {rendered} rendered surfaces were read; the control a user actually sees is "
        "the one this was written for"
    )


def test_the_config_tab_does_not_promise_the_report_either():
    """The admin-facing copy carried the same promise the docs did."""
    pytest.importorskip(
        "open_webui_openrouter_pipe.plugins.pipe_dashboard",
        reason="the --no-plugins artifacts omit pipe_dashboard by design",
    )
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META

    detail = CONFIG_META["AUTO_ATTACH_IMAGE_FILTERS"]["detail"]
    assert "only some of them take" in detail, detail
    assert not re.search(r"told afterwards|you are told|tells the user", detail, re.I), detail


_TIER_CHECK_RE = re.compile(
    r"tiers?\b.{0,90}?(?:checked|measured)\s+(?:only\s+)?against"
    r"|(?:checked|measured)\s+(?:only\s+)?against.{0,90}?tiers?\b",
    re.I,
)

_TIER_CONDITIONAL_RE = re.compile(
    r"publish\w*\s+no\b|where\s+(?:it|one|the model)\s+publish"
    r"|if\s+(?:it|the model)\s+publish|depends on the model|only\s+(?:against|to)\b.{0,40}four"
    r"|four tier names",
    re.I,
)


def _table_rows(text: str) -> list[str]:
    """Table rows read one at a time; ``_claims`` flattens a whole table into one block."""
    return [line.strip() for line in text.splitlines() if line.lstrip().startswith("|")]


def test_no_document_says_a_size_tier_is_checked_against_a_list_most_models_never_publish():
    """The check is conditional, and both documents asserted it unconditionally.

    Measured on the fixtures and through the adapter, not read: a tier outside a published
    list is withheld and named, while on a model that publishes no list the same tier goes
    out for the provider to interpret. Both arms are driven here with the SAME tier, so a
    fitting function that always withholds, or always sends, fails one of them -- and the
    two populations are counted from the recorded contracts rather than quoted, so the day
    the fleet moves this reports the new split instead of the documented one.

    The in-product control text already says both; the documents denied it, and one of them
    denied it in a table row and again in prose while quoting the correct control text in
    between. So every passage on either document that says a tier is checked against
    something has to record which of the two it means.
    """
    from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter

    publishing: dict[str, list[dict]] = {}
    silent: dict[str, list[dict]] = {}
    for slug, model_id in EVERY_CONTRACT:
        records = _records(slug)
        declares = any(
            isinstance((record.get("supported_parameters") or {}).get("resolution"), dict)
            for record in records
        )
        (publishing if declares else silent)[model_id] = records
    assert publishing and silent, (
        f"the recorded fleet no longer has both kinds: {len(publishing)} publish a tier "
        f"list, {len(silent)} do not; the documented split has to be re-derived"
    )

    withholds, sends = [], []
    for population, landed in ((publishing, withholds), (silent, sends)):
        for model_id, records in population.items():
            tiers = {
                str(value)
                for record in records
                for value in (
                    (record.get("supported_parameters") or {}).get("resolution") or {}
                ).get("values")
                or ()
            }
            outside = next((t for t in ("512", "1K", "2K", "4K") if t not in tiers), None)
            if outside is None:
                continue
            top, _passthrough, notes = ImageGenerationAdapter._split_image_config(
                {"image_config": {"size": outside}},
                allowed_passthrough=frozenset(),
                record=records[0],
                records=records,
            )
            landed.append((model_id, outside, top.get("size"), [n.text for n in notes]))

    assert withholds and sends, f"withholds={withholds} sends={sends}"
    for model_id, tier, sent, notes in withholds:
        assert sent is None and notes, (
            f"{model_id} publishes a tier list without {tier} and sent it anyway "
            f"({sent!r}, notes={notes}); the documents may claim the check happens"
        )
    for model_id, tier, sent, notes in sends:
        assert sent == tier and not notes, (
            f"{model_id} publishes no tier list, so {tier} has nothing of the model's to be "
            f"measured against, yet it was not sent as typed ({sent!r}, notes={notes})"
        )

    checked = 0
    for text in (_image_doc(), _atlas()):
        for claim in (*_table_rows(text), *_claims(text)):
            if not _TIER_CHECK_RE.search(claim):
                continue
            checked += 1
            assert _TIER_CONDITIONAL_RE.search(claim), (
                f"{len(silent)} of the {len(publishing) + len(silent)} recorded models "
                "publish no tier list, so on those a tier is measured against OpenRouter's "
                "four names and then sent for the provider to interpret. This passage "
                f"states the check without saying which case it means: {claim!r}"
            )
    assert checked >= 3, (
        f"only {checked} passages describe what a tier is checked against; the sweep went "
        "hollow and the claim can come back unconditional"
    )


_SUPERSEDED_DROP_RE = re.compile(r"(?:dropped|left out|omitted) (?:that|this|in that|in this) way", re.I)

_CONTROL_WITHHELD_RE = re.compile(r"\b(?:not sent|not passed|left out|dropped|nor is)\b", re.I)


def _size_variants_that_withhold_a_control() -> list[str]:
    """Every "exact pixels win" sentence that admits a control is held back.

    Read off the selection table itself, not off a list of constant names: the table
    grew a third withholding cell and the two tests that named a hardcoded PAIR went on
    passing with the retired promise planted in it. One of the table's cells reaches no
    recorded contract at all, so driving real specs through the renderer cannot cover it
    either -- the values of the table are the only enumeration that tracks the table.

    Only the withholding sentences are returned. The cell for a model with neither tiers
    nor an Aspect ratio control withholds nothing, so a notice can never fire there and
    promising one would be the false claim in the other direction.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import _SIZE_MEANING

    variants = [pixels for _tier, pixels in _SIZE_MEANING.values()]
    withholding = [text for text in variants if _CONTROL_WITHHELD_RE.search(text)]
    assert len(withholding) >= 3, (
        f"only {len(withholding)} of the {len(variants)} size variants match "
        f"{_CONTROL_WITHHELD_RE.pattern!r}, so this sweep went hollow and the promise "
        "that the notice survives a reload can be restored in whichever variant stopped "
        "matching. Widen the pattern to whatever the control text calls it now"
    )
    assert len(withholding) < len(variants), (
        f"{_CONTROL_WITHHELD_RE.pattern!r} matched every one of the {len(variants)} size "
        "variants, so it classifies nothing. The variant for a model that publishes no "
        "tiers and renders no Aspect ratio holds nothing back, and must not be made to "
        "promise a notice that cannot fire"
    )
    return withholding


def test_the_doc_and_the_control_agree_that_a_dropped_setting_is_a_toast():
    """Two surfaces, one durability claim, and Open WebUI keeps neither.

    The event carrying it is not among the ones Open WebUI writes back into the stored
    message, so a reader who reloads finds nothing. The control text says so; the doc said
    the opposite. Both are read here so they cannot drift apart again.

    The document half filters sentences and then asserts only on the ones that match, so
    it is counted: with no floor, a one-word rephrase silences it and the claim this test
    exists to keep out can be restored with the suite green.
    """
    for control_text in _size_variants_that_withhold_a_control():
        assert "toast" in control_text.lower(), control_text
        assert "reload" in control_text.lower(), control_text

    doc = _image_doc()
    matched = 0
    for sentence in re.split(r"(?<=[.;])\s", doc):
        if not _SUPERSEDED_DROP_RE.search(sentence):
            continue
        matched += 1
        assert "toast" in sentence.lower(), sentence.strip()
        assert "named in the chat" not in sentence.lower(), sentence.strip()
    assert matched, (
        f"no sentence matched {_SUPERSEDED_DROP_RE.pattern!r}, so this guard is filtering "
        "on wording the document no longer uses and asserted nothing at all -- the claim "
        "that the notice survives a reload can be restored underneath it. Widen the "
        "pattern to whatever the document calls it now"
    )


def test_every_surface_agrees_when_the_master_switch_empties_the_picker():
    """Three passages gave three settling times, and one denied the real one.

    The clear runs inside the loader, ahead of its own freshness check, so it lands on the
    next model-list build. Measured by calling the loader with a freshness window that has
    not expired: a clear gated on that window could not fire, and this would fail.
    """
    from open_webui_openrouter_pipe.integrations.image_catalog import ensure_image_catalog_loaded
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry
    from unittest.mock import MagicMock

    def _registered() -> bool:
        return any(
            m.get("original_id") == "acme/pigment-7"
            for m in OpenRouterModelRegistry.list_models()
        )

    try:
        OpenRouterModelRegistry.register_image_models(
            [
                {
                    "id": "acme/pigment-7",
                    "name": "Pigment 7",
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
                    "pricing": {},
                }
            ]
        )
        OpenRouterModelRegistry._last_image_fetch = time.time()
        OpenRouterModelRegistry._last_image_attempt = time.time()
        assert _registered(), "precondition: the model has to be in the picker to leave it"

        valves = MagicMock()
        valves.ENABLE_OPENROUTER_IMAGE_GENERATION = False
        asyncio.run(
            ensure_image_catalog_loaded(
                MagicMock(), valves=valves, api_key="k", logger=MagicMock(), cache_seconds=86400
            )
        )

        assert not _registered(), (
            "the clear waited on the freshness window instead of running ahead of it"
        )
    finally:
        OpenRouterModelRegistry.register_image_models([])
        OpenRouterModelRegistry.reset_image_fetch_timestamp()
        OpenRouterModelRegistry._last_image_attempt = 0.0

    checked = 0
    for text in (_image_doc(), _atlas()):
        for para in text.split("\n\n"):
            if "MODEL_CATALOG_REFRESH_SECONDS" not in para:
                continue
            if not re.search(r"kill switch|Master-disable|set\s*\n?\s*to `False`|this to `False`|=False", para):
                continue
            checked += 1
            assert re.search(r"ahead of the catalogue refresh window", para), (
                f"a surface still makes the disable wait on the refresh window: {para.strip()!r}"
            )
            assert not re.search(r"leave (the picker )?within|vanish[a-z]* .*immediately|next page load", para), (
                f"a surface still gives a different settling time: {para.strip()!r}"
            )
    assert checked >= 3, f"only {checked} passages describe the disable; the sweep went hollow"


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

    Everything but 401, 402, 408, 413, 429 and 5xx lands on it, which means its unconditional
    closing line is read by a forbidden key, an unroutable model and a malformed reference URL
    alike. The chat orchestrator used to pass it explicitly, which dragged those six statuses
    onto it as well; that override is gone and each of them now renders its own card. It closed by telling all of them to ask an admin to enable a control that no longer
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
# ---------------------------------------------------------------------------
# Error templates: audited by rendering, not by reading the constant.
# ---------------------------------------------------------------------------


def _error_templates() -> dict[str, str]:
    from open_webui_openrouter_pipe.core import config as _config
    from open_webui_openrouter_pipe.core.error_formatter import _FALLBACK_ERROR_TEMPLATE

    found = {
        name: getattr(_config, name)
        for name in dir(_config)
        if name.startswith("DEFAULT_") and name.endswith("_TEMPLATE")
    }
    found["_FALLBACK_ERROR_TEMPLATE"] = _FALLBACK_ERROR_TEMPLATE
    return found


_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{(\w+)\}")

_EMIT_HELPERS = ("_emit_templated_error", "_emit_templated_error_event")
_RENDER_HELPER = "_render_error_template"


def _scopes(tree: ast.AST):
    """The module, then every function and class in it, each treated as its own scope."""
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node


def _own_nodes(scope: ast.AST):
    """Everything lexically inside *scope* but not inside a nested scope of its own."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _bindings(scope: ast.AST) -> dict[str, list[ast.expr]]:
    """`name -> every expression assigned to it` in this scope, for one hop of resolution."""
    found: dict[str, list[ast.expr]] = {}
    for node in _own_nodes(scope):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            found.setdefault(node.targets[0].id, []).append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            found.setdefault(node.target.id, []).append(node.value)
    return found


def _callee(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def _template_names(node: ast.expr | None, bindings: dict[str, list[ast.expr]], hops: int = 1) -> set[str]:
    """Which template constants an expression can be, or an empty set when it cannot be told.

    `valves.RATE_LIMIT_TEMPLATE` is the same constant as `DEFAULT_RATE_LIMIT_TEMPLATE`,
    since the valve's default IS that constant. `a or b` can be either. A name bound in
    the same scope is followed once; a function parameter is followed nowhere, so a
    generic passthrough contributes nothing to anybody's allowed set.
    """
    if node is None:
        return set()
    if isinstance(node, ast.Attribute) and node.attr.endswith("_TEMPLATE"):
        return {"DEFAULT_" + node.attr}
    if isinstance(node, ast.Name):
        if node.id.endswith("_TEMPLATE"):
            return {node.id}
        if hops and node.id in bindings:
            found: set[str] = set()
            for bound in bindings[node.id]:
                found |= _template_names(bound, bindings, hops - 1)
            return found
        return set()
    if isinstance(node, ast.BoolOp):
        found = set()
        for value in node.values:
            found |= _template_names(value, bindings, hops)
        return found
    return set()


def _dict_keys(node: ast.expr | None, bindings: dict[str, list[ast.expr]], hops: int = 1) -> set[str] | None:
    """The literal keys of a dict expression, or None when they cannot all be read.

    None is not the empty set: `{**a, **b}` and a name bound to a call have keys this
    cannot see, so the site is dropped entirely rather than counted as supplying nothing.
    """
    if node is None:
        return None
    if isinstance(node, ast.Dict):
        keys: set[str] = set()
        for key in node.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                return None
            keys.add(key.value)
        return keys
    if isinstance(node, ast.Name) and hops and node.id in bindings:
        merged: set[str] = set()
        for bound in bindings[node.id]:
            bound_keys = _dict_keys(bound, bindings, hops - 1)
            if bound_keys is None:
                return None
            merged |= bound_keys
        return merged
    return None


def _emit_sites() -> tuple[dict[str, set[str]], set[str]]:
    """Read off the package: which keys each template is handed, and which get the context block.

    Returns `(per-template variables, templates emitted through the context-enriching
    helper)`. Every emit call in the package is visited; the ones whose template or
    variables cannot be resolved contribute nothing, which can only narrow an allowed set
    and so can only produce a false alarm, never a false pass.
    """
    from tests.package_sources import parsed_sources

    variables: dict[str, set[str]] = {}
    enriched: set[str] = set()
    for _path, _source, tree in parsed_sources("open_webui_openrouter_pipe"):
        for scope in _scopes(tree):
            bindings = _bindings(scope)
            for node in _own_nodes(scope):
                if not isinstance(node, ast.Call):
                    continue
                helper = _callee(node)
                if helper in _EMIT_HELPERS:
                    template = next((k.value for k in node.keywords if k.arg == "template"), None)
                    supplied = next((k.value for k in node.keywords if k.arg == "variables"), None)
                elif helper == _RENDER_HELPER:
                    template = node.args[0] if node.args else None
                    supplied = node.args[1] if len(node.args) > 1 else None
                else:
                    continue
                targets = _template_names(template, bindings)
                if not targets:
                    continue
                if helper in _EMIT_HELPERS:
                    enriched |= targets
                keys = _dict_keys(supplied, bindings)
                for target in targets:
                    variables.setdefault(target, set()).update(keys or set())
    return variables, enriched


def _builder_rendered_templates() -> set[str]:
    """The templates that reach `_build_error_template_values`, read off the code that routes them.

    Those are the status selector's own returns plus the two constants standing behind
    `or` on the way into `to_markdown`. Every other template is rendered from a dict
    written at its own call site and must not borrow this set.
    """
    from tests.package_sources import parsed_sources

    names: set[str] = set()
    for _path, _source, tree in parsed_sources("open_webui_openrouter_pipe"):
        for scope in _scopes(tree):
            bindings = _bindings(scope)
            named = getattr(scope, "name", "")
            for node in _own_nodes(scope):
                if named == "_select_openrouter_template" and isinstance(node, ast.Return):
                    names |= _template_names(node.value, bindings)
                if not isinstance(node, ast.Call):
                    continue
                if _callee(node) == "to_markdown":
                    for keyword in node.keywords:
                        if keyword.arg == "template":
                            names |= _template_names(keyword.value, bindings)
                elif named == "to_markdown" and _callee(node) == _RENDER_HELPER and node.args:
                    names |= _template_names(node.args[0], bindings)
    return names


def test_every_error_template_placeholder_is_one_the_code_supplies():
    """An unknown `{name}` is not dropped, it is printed verbatim to the user.

    The renderer only drops a line when the placeholder is a key it was handed and that
    key is empty; a name nobody supplies survives into the message as literal braces. The
    built-in fallback card shipped `{model_slug}` for exactly that reason -- the value
    builder produces `model_identifier` -- and rendered "**Model**: {model_slug}" whenever
    both an operator's status template and their generic template were blank.

    Every allowed name is read from code. The value builder and the context builder are
    called; the per-template extras are scanned out of the emit call sites themselves, so
    dropping a key from a `variables={...}` dict reddens the template that dict feeds. The
    hand-written whitelist this replaced was read from nothing: removing `requested_model`
    from all three of its call sites left the whole suite green while the restricted-model
    card showed the reader a literal `{requested_model}`.

    The value builder's set is confined to the templates that actually reach it, so a card
    rendered from a four-key dict is checked against those four keys and not against the
    forty names some unrelated card is entitled to.
    """
    from open_webui_openrouter_pipe.core.errors import (
        OpenRouterAPIError,
        _build_error_template_values,
    )

    context_keys = {"error_id", "timestamp", "session_id", "user_id", "support_email", "support_url"}
    supplied = set(
        _build_error_template_values(
            OpenRouterAPIError(status=500, reason="x", openrouter_message="m"),
            heading="h",
            diagnostics=[],
            metrics={},
            model_identifier="m",
            normalized_model_id=None,
            api_model_id=None,
            context=dict.fromkeys(context_keys, ""),
        )
    )
    assert "model_identifier" in supplied and "model_slug" not in supplied, (
        "the value builder no longer matches what this test assumes it produces"
    )

    templates = _error_templates()
    assert len(templates) > 10, f"only {len(templates)} templates found; the scan missed some"

    emitted, enriched = _emit_sites()
    builder_rendered = _builder_rendered_templates()

    unrouted = sorted(set(templates) - set(emitted) - enriched - builder_rendered)
    assert not unrouted, (
        "the scan found no call site that renders these, so their allowed names came from "
        f"nowhere and this test is not checking them: {unrouted}"
    )

    unfilled = {}
    for name, template in templates.items():
        allowed = set(emitted.get(name, set()))
        if name in enriched or name in builder_rendered:
            allowed |= context_keys
        if name in builder_rendered:
            allowed |= supplied
        missing = sorted(set(_PLACEHOLDER_RE.findall(template)) - allowed)
        if missing:
            unfilled[name] = missing
    assert not unfilled, f"these placeholders would render as literal braces: {unfilled}"


def _rendering_paths() -> dict[str, list[set[str]]]:
    """For every default template, the variables each separate rendering path really supplies.

    A list per template, not a union: the union is what some caller can fill, and a card is
    rendered by one caller at a time. `AUTHENTICATION_ERROR_TEMPLATE` has two, and everything
    the OpenRouter one fills is missing from the local one, which sends no request at all.

    A site whose variables the scanner cannot read comes back empty and is left out, exactly as
    the scanner treats it elsewhere: counting it as a path that supplies nothing would flag every
    placeholder in the template. That can hide a caller, never invent one.
    """
    from open_webui_openrouter_pipe.core.errors import (
        OpenRouterAPIError,
        _build_error_template_values,
    )

    context_keys = {"error_id", "timestamp", "session_id", "user_id", "support_email", "support_url"}
    builder_keys = set(
        _build_error_template_values(
            OpenRouterAPIError(status=500, reason="x", openrouter_message="m"),
            heading="h",
            diagnostics=[],
            metrics={},
            model_identifier="m",
            normalized_model_id=None,
            api_model_id=None,
            context=dict.fromkeys(context_keys, ""),
        )
    )

    emitted, enriched = _emit_sites()
    builder_rendered = _builder_rendered_templates()

    paths: dict[str, list[set[str]]] = {}
    for name in set(emitted) | builder_rendered:
        found: list[set[str]] = []
        if name in emitted:
            found.append(set(emitted[name]) | (context_keys if name in enriched else set()))
        if name in builder_rendered:
            found.append(builder_keys | context_keys)
        readable = [keys for keys in found if keys]
        if readable:
            paths[name] = readable
    return paths


_CONDITIONAL_WARNING_RE = re.compile(r"verbatim", re.I)


def test_a_template_description_promises_no_placeholder_the_card_cannot_fill():
    """The box's own help is the only help an admin gets on the artifacts with no config tab.

    `AUTHENTICATION_ERROR_TEMPLATE` is rendered by two callers: a 401 OpenRouter returned, and
    the pipe failing to read its own key. The second sends nothing, so it has none of the
    error-context fields -- and the description offered them as if it did, while also claiming
    "Only the names listed here are substituted". An admin who took it at its word shipped a
    literal `{request_id}` to every user hit by a blank or undecryptable key.

    Read off the call sites: a name no path fills is advice nobody can follow, and a name only
    some path fills has to be flagged as such rather than listed alongside the rest.
    """
    from open_webui_openrouter_pipe import Pipe

    paths = _rendering_paths()
    dead: dict[str, list[str]] = {}
    unflagged: dict[str, list[str]] = {}
    checked = 0

    for valve, field in Pipe.Valves.model_fields.items():
        if not valve.endswith("_TEMPLATE"):
            continue
        description = field.description or ""
        named = set(_PLACEHOLDER_RE.findall(description))
        if not named:
            continue
        found = paths.get("DEFAULT_" + valve)
        if not found:
            continue
        checked += 1
        union = set().union(*found)
        always = set.intersection(*found)
        missing = sorted(named - union)
        if missing:
            dead[valve] = missing
        conditional = sorted(named & (union - always))
        if conditional and not _CONDITIONAL_WARNING_RE.search(description):
            unflagged[valve] = conditional

    assert checked >= 10, f"only {checked} template descriptions were read; the scan has gone blind"
    assert not dead, (
        "these descriptions name placeholders no caller supplies, so an admin who uses one ships "
        f"the braces to a reader: {dead}"
    )
    assert not unflagged, (
        "these descriptions list placeholders that only some of the callers rendering that box "
        f"can fill, without saying what the others print instead: {unflagged}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["the pipe cannot read its own key", "OpenRouter rejected the key"])
async def test_an_authentication_card_prints_no_placeholder_verbatim(monkeypatch, pipe_instance_async, path):
    """Both callers render the same box, and one of them has none of the error-context fields.

    The renderer only drops a line when the placeholder is a key it was handed and that key is
    empty; a name nobody handed it survives to the reader as literal braces. Both arms render
    the shipped default, so a row added for one caller without a conditional reddens the other.
    """
    pipe = pipe_instance_async
    detail = "the configured key could not be decrypted"

    if path == "the pipe cannot read its own key":
        monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: (None, detail))
        monkeypatch.setattr(pipe._artifact_store, "_ensure_artifact_store", lambda *_a, **_k: None)
        result = await pipe._handle_pipe_call(
            {"stream": False, "model": "m1"},
            {},
            None,
            None,
            None,
            {},
            None,
            None,
            None,
            valves=pipe.valves,
            session=cast(Any, object()),
        )
        card = result["choices"][0]["message"]["content"]
    else:
        from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError

        card = await pipe._ensure_error_formatter()._report_openrouter_error(
            OpenRouterAPIError(status=401, reason="unauthorized", openrouter_message=detail),
            event_emitter=None,
            normalized_model_id="vendor/model",
            api_model_id="vendor/model",
        )

    assert detail in card, f"the {path} card does not carry what went wrong:\n{card}"
    leftover = sorted(set(_PLACEHOLDER_RE.findall(card)))
    assert not leftover, (
        f"the {path} card prints {leftover} to the reader as literal braces:\n{card}"
    )


def _unconditional_fixed_line(template: str) -> str:
    """A line of *template* that always renders and holds no placeholder, to identify a card by.

    Not simply the first heading: the generic template's heading is itself conditional, and a
    line inside an `{{#if}}` is missing from most cards, so an anchor taken from one proves
    nothing about which template rendered.
    """
    depth = 0
    for line in template.splitlines():
        opened = line.count("{{#if")
        closed = line.count("{{/if}}")
        if depth == 0 and not opened and not closed and line.strip() and "{" not in line:
            return line.strip()
        depth += opened - closed
    raise AssertionError("the template has no fixed unconditional line to identify it by")


_STATUS_TEMPLATES = {
    400: "OPENROUTER_ERROR_TEMPLATE",
    401: "AUTHENTICATION_ERROR_TEMPLATE",
    402: "INSUFFICIENT_CREDITS_TEMPLATE",
    408: "SERVER_TIMEOUT_TEMPLATE",
    413: "PAYLOAD_TOO_LARGE_TEMPLATE",
    429: "RATE_LIMIT_TEMPLATE",
    500: "SERVICE_ERROR_TEMPLATE",
    503: "SERVICE_ERROR_TEMPLATE",
}
"""Which box each status renders, stated here rather than read back from the selector.

Asking the selector which template a status picks and then checking the card against its own
answer passes whatever the selector does, including returning one constant for every status.
"""

_EMPTY_VALUE_RE = re.compile(r"^\*\*[^*]+\*\*:?\s*``\s*$")


async def _rejection_card(status: int, request_id: str | None) -> str:
    """The card a rejection really produces, rendered by the path a chat request takes."""
    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError

    pipe = Pipe()
    try:
        return await pipe._ensure_error_formatter()._report_openrouter_error(
            OpenRouterAPIError(
                status=status,
                reason="rejected",
                openrouter_message="the provider refused",
                request_id=request_id,
            ),
            event_emitter=None,
            normalized_model_id="vendor/model",
            api_model_id="vendor/model",
        )
    finally:
        await pipe.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "valve"), sorted(_STATUS_TEMPLATES.items()))
async def test_a_rejection_hands_the_reader_the_id_openrouter_can_look_up(status, valve):
    """OpenRouter's own reference is the only id its support can search; ours correlates our logs.

    The chat path used to force the generic template for every status, and that template is
    the one place the reference was written. Selecting by status instead handed a reader hit
    by a 401, 402, 408, 413, 429 or 5xx a card with our error id and nothing OpenRouter could
    trace, so "contact support" led nowhere.

    Driven through the real reporting path for every status the selector can choose, with a
    different reference on each row, so neither a fixed card nor a fixed id passes.
    """
    from open_webui_openrouter_pipe import Pipe

    pipe = Pipe()
    try:
        anchor = _unconditional_fixed_line(getattr(pipe.valves, valve))
    finally:
        await pipe.close()

    reference = f"gen-{status}-abc"
    card = await _rejection_card(status, reference)

    assert card.strip(), f"a {status} rendered no card at all"
    assert anchor in card, (
        f"a {status} rendered a card other than the one its status selects, so this row is not "
        f"measuring the {status} template:\n{card}"
    )
    assert reference in card, (
        f"a {status} card carries no OpenRouter reference, so nothing on it can be looked up "
        f"by the people who rejected the request:\n{card}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", sorted(_STATUS_TEMPLATES))
async def test_a_rejection_without_a_reference_leaves_no_empty_row(status):
    """Not every rejection carries one, and a labelled row with nothing after it is worse than none."""
    card = await _rejection_card(status, None)

    assert card.strip(), f"a {status} rendered no card at all"
    assert "Error ID" in card, f"a {status} rendered something other than its own card:\n{card}"
    assert "request ID" not in card, (
        f"a {status} card with no reference still prints the row:\n{card}"
    )
    assert "``" not in card, (
        f"a {status} card carries a labelled row with nothing in it:\n{card}"
    )
    empty = [line for line in card.splitlines() if _EMPTY_VALUE_RE.match(line.strip())]
    assert not empty, f"a {status} card carries labelled rows with no value: {empty}"


_CARD_URL_RE = re.compile(r"https://[A-Za-z0-9.-]*openrouter\.ai/\S*?(?=[)\s`]|$)")

_OPENROUTER_DESTINATIONS = {
    "https://openrouter.ai/keys": (
        "OpenRouter's API authentication reference links it as where a key is created."
    ),
    "https://openrouter.ai/credits": (
        "OpenRouter's own 402 body says 'Insufficient credits. Add more using "
        "https://openrouter.ai/credits'."
    ),
    "https://openrouter.ai/activity": (
        "OpenRouter's FAQ calls this the Activity tab and links it for reviewing usage history."
    ),
    "https://status.openrouter.ai/": (
        "The status host these cards have always named. It appears in none of the vendored "
        "OpenRouter documentation, and the maintainer confirmed on 2026-08-22 that the page "
        "exists; that confirmation is this entry's source."
    ),
}


def test_every_link_an_error_card_offers_has_somebody_recorded_a_source_for():
    """A card that links a page OpenRouter does not have wastes the one action it offers.

    The credits card sent the reader to `https://openrouter.ai/usage`, a path that appears
    in none of OpenRouter's documentation; the page that shows an account what it has been
    spending is the Activity tab. Nothing caught it, because nothing here looked at the
    links at all -- reverting the correction left the whole suite green.

    The check is membership, not wording: a sentence around a link can be rewritten freely,
    and a NEW destination cannot ship until a line recording where it came from ships with
    it. Fetching the URLs is what this deliberately does not do -- that needs network in CI,
    and the vendored documentation corpus the notes are drawn from is not committed.
    """
    templates = _error_templates()
    linked: dict[str, set[str]] = {}
    for name, template in templates.items():
        for url in _CARD_URL_RE.findall(template):
            linked.setdefault(url, set()).add(name)

    assert len(linked) >= 3, (
        f"only {sorted(linked)} came out of {len(templates)} templates; the cards send the "
        "reader somewhere and this scan is not finding it"
    )

    unrecorded = sorted(set(linked) - set(_OPENROUTER_DESTINATIONS))
    assert not unrecorded, (
        "these destinations are offered to a reader with nothing recorded about where they "
        f"came from: {[(url, sorted(linked[url])) for url in unrecorded]}"
    )

    unused = sorted(set(_OPENROUTER_DESTINATIONS) - set(linked))
    assert not unused, (
        f"no card links {unused} any more; drop the entry rather than leaving the list "
        "pre-authorising a destination nothing uses"
    )


def _render_status_card(status: int, message: str) -> str:
    from open_webui_openrouter_pipe.core.config import Valves
    from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError
    from open_webui_openrouter_pipe.core.utils import _render_error_template

    valves = Valves()
    template = {
        401: valves.AUTHENTICATION_ERROR_TEMPLATE,
        402: valves.INSUFFICIENT_CREDITS_TEMPLATE,
        408: valves.SERVER_TIMEOUT_TEMPLATE,
        413: valves.PAYLOAD_TOO_LARGE_TEMPLATE,
        429: valves.RATE_LIMIT_TEMPLATE,
    }.get(status, valves.SERVICE_ERROR_TEMPLATE)
    error = OpenRouterAPIError(status=status, reason="rejected", openrouter_message=message)
    return _render_error_template(
        template,
        {
            **{k: "" for k in ("session_id", "user_id", "support_email", "support_url")},
            "error_id": "0123456789abcdef",
            "timestamp": "2026-01-01T00:00:00Z",
            "status_code": status,
            "reason": message,
            "openrouter_code": status,
            "openrouter_message": message,
            "rate_limit_type": message,
            "model_identifier": "openai/gpt-4o",
            "detail": message,
            "sanitized_detail": message,
        },
    )


@pytest.mark.parametrize(
    "status, message",
    [
        (502, "Provider returned an invalid response"),
        (503, "No allowed provider meets your data policy"),
    ],
)
def test_the_service_error_card_does_not_swear_the_failure_cannot_be_yours(status, message):
    """One card covers every 5xx, and two of them are caused by the caller's own choices.

    OpenRouter documents 502 as the chosen model being down and 503 as no provider matching
    the routing requirements sent with the request -- and this pipe can produce that 503 on
    its own, through `Enforce ZDR routing` or a provider-routing restriction. The card used
    to state "This is **not** a problem with your request. The issue is on OpenRouter's
    side" unconditionally, and then advise waiting, which never clears a routing rejection.
    """
    card = _render_status_card(status, message)
    assert message in card, "the message did not render, so the rest of this proves nothing"
    disowned = re.search(
        r"not\b[^.]{0,30}\bproblem with your request|issue is on OpenRouter's side",
        card,
        re.I,
    )
    assert disowned is None, (
        f"a {status}, which OpenRouter attributes to the model or to the request's own "
        f"routing constraints, is told the fault is elsewhere: {disowned and disowned.group(0)!r}"
    )
    for surface, control in _surface_labels("ZDR_ENFORCE").items():
        assert control in card, (
            f"removing the disclaimer only stopped the card being wrong; a reader whose 503 "
            f"is a routing rejection still needs the control that causes it. The card does "
            f"not name {control!r}, the label {surface} shows for it:\n{card}"
        )


@pytest.mark.parametrize(
    ("status", "message", "required", "forbidden", "claim"),
    [
        (
            413,
            "Request body is too large",
            (r"not\b[^.]*context window",),
            (r"larger context window",),
            "that the cap is on the size of the request itself and not on the model's "
            "context window -- the old card sent the reader to a bigger-context model, "
            "which does nothing about a payload cap",
        ),
        (
            429,
            "Rate limit exceeded: free-models-per-day",
            (r"reach(?:ed|es)?\b[^.]*\blimits?\b", r":free\b", r"per-day|per day|daily"),
            (),
            "that the refusal is a limit the account reached rather than a burst of speed, "
            "and that the free variants carry their own daily cap -- a different limit from "
            "the per-minute one, and one that backing off never clears",
        ),
        (
            408,
            "Request timed out upstream",
            (r"(?:before any|without any|no)\s+(?:output|content|reply|tokens)",),
            (),
            "that nothing at all came back, so the reader is not left hunting for a partial "
            "reply that was never produced",
        ),
        (
            500,
            "Internal server error",
            (
                r"502[^\n]*model[^\n]*(?:down|unavailable|offline|unreachable|not respond)",
                r"503[^\n]*provider[^\n]*rout",
                r"5xx[^\n]*(?:OpenRouter|inside)",
            ),
            (),
            "which 5xx means what -- the model being down, no provider satisfying the "
            "routing requirements sent with the request, or a fault inside OpenRouter "
            "itself. One card serves every 5xx, so without that split the reader cannot "
            "tell an outage they wait out from a routing rejection they have to change "
            "something to clear. Rendered at 500 so the substituted status cannot supply "
            "the 502 and 503 the bullets are asserted on",
        ),
        (
            402,
            "Insufficient credits",
            (r"negative[^\n]*:free", r":free[^\n]*(?:restor|again|back)"),
            (),
            "that a negative balance blocks the free variants too and that clearing it "
            "brings them back -- an account that reads 'out of credits' and switches to a "
            "`:free` model otherwise finds it refused with no explanation on the card",
        ),
    ],
    ids=[
        "request-too-large",
        "rate-limit",
        "server-timeout",
        "service-error",
        "insufficient-credits",
    ],
)
def test_a_rewritten_card_still_makes_the_claim_it_was_rewritten_for(
    status, message, required, forbidden, claim
):
    """Each of these three cards was rewritten to say one thing it had never said.

    Reverting any of them to the text it replaced left the whole suite green, because
    nothing here read them: the only assertions on this harness were the ABSENCE of a
    sentence in the 5xx card. Each row pins the substantive claim against the RENDERED
    card rather than against the constant in ``core/config.py``, so a card that stops
    rendering fails here too, and a wording change that keeps the claim does not.
    """
    card = _render_status_card(status, message)
    assert message in card, "the message did not render, so the rest of this proves nothing"
    for pattern in required:
        assert re.search(pattern, card, re.I), (
            f"the {status} card no longer tells the reader {claim}; nothing matched "
            f"{pattern!r} in:\n{card}"
        )
    for pattern in forbidden:
        found = re.search(pattern, card, re.I)
        assert found is None, (
            f"the {status} card is back to advice that does not apply to it: "
            f"{found and found.group(0)!r} in:\n{card}"
        )


def _surface_labels(valve: str) -> dict[str, str]:
    """The label each settings surface this artifact ships shows for one valve.

    Open WebUI's own valve panel renders the pydantic ``title``, which every artifact has;
    the Config tab renders its own ``config_meta`` title and exists in two of the four. A
    card that cites a setting has to be findable on whichever ones are present, so the
    expectation is read from them rather than quoted here, and a rename on either surface
    reddens every card that names it.
    """
    from open_webui_openrouter_pipe import Pipe

    properties = Pipe.Valves.model_json_schema().get("properties") or {}
    spec = properties.get(valve)
    assert isinstance(spec, dict) and spec.get("title"), (
        f"the valve panel shows no title for {valve}; its schema entry is {spec!r}"
    )
    labels = {"Open WebUI's valve panel": str(spec["title"])}
    try:
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META
    except ImportError:
        return labels
    labels["the Config tab"] = str(CONFIG_META[valve]["title"])
    return labels


@pytest.mark.parametrize(
    "message", ["No auth credentials found", "User not found"], ids=["no-credentials", "no-user"]
)
def test_the_authentication_card_names_the_valve_an_admin_must_check(message):
    """The 401 card also renders when the pipe cannot read its own key, never having called
    OpenRouter -- so it has to point at the valve, under the label the reader can look up.

    The name is taken from the settings surfaces rather than quoted here, so renaming the
    valve without updating the card reddens this. It is checked against EVERY surface the
    artifact ships: the ``--no-plugins`` builds have no Config tab, and there the valve
    panel is the only place the label can be found -- which is exactly where this used to
    skip itself. The advice it replaced told the reader to "re-authenticate your session"
    if using OAuth; this pipe has no OAuth session to re-authenticate, its only credential
    being that valve.
    """
    card = _render_status_card(401, message)
    assert message in card, (
        "the message did not render, so nothing below is measuring a rendered card. Two "
        "OpenRouter 401 bodies are driven through it for that reason: a card that stopped "
        "rendering, or a constant standing in for one, cannot carry both"
    )
    for surface, control in _surface_labels("API_KEY").items():
        assert control in card, (
            f"the authentication card does not name {control!r}, the label {surface} shows "
            f"for the only credential this pipe has:\n{card}"
        )
    assert not re.search(r"\bOAuth\b", card), (
        "the card offers an OAuth action; the pipe authenticates with that valve alone"
    )
    assert re.search(
        r"WEBUI_SECRET_KEY[^\n]*(?:chang|rotat|differ)|(?:chang|rotat|differ)\w*[^\n]*"
        r"WEBUI_SECRET_KEY",
        card,
    ), (
        "a stored key encrypted under a server secret that has since changed cannot be "
        "decrypted, and it is one of the two local causes that render this card. An admin "
        "who is not told to look at WEBUI_SECRET_KEY re-types a key that was never wrong. "
        "The other test covering this cause reads the value through {openrouter_message}, "
        f"which the pipe supplies, so it passes with the remedy deleted:\n{card}"
    )


_NAMED_CONTROL_RE = re.compile(r"`([A-Z][A-Za-z-]*(?: [A-Za-z-]+)+)`")


def _settings_surfaces() -> dict[str, set[str]]:
    """Every settings interface this artifact ships, and the labels each one displays.

    Open WebUI's own valve panel renders ``valvesSpec.properties[...].title``, which it reads
    off ``function_module.Valves`` -- and ``load_function_module_by_id`` returns ``module.Pipe()``,
    so that attribute is ``Pipe.Valves``, the plugin-extended class. Where a field declares no
    ``title=``, pydantic generates one from the field name, which is why ``API_KEY`` showed as
    "Api Key". The Config tab renders ``CONFIG_META[...]["title"]`` instead, an entirely separate
    string, and the ``--no-plugins`` artifacts ship no Config tab at all -- so there the valve
    panel is the only place a name can be looked up.
    """
    from open_webui_openrouter_pipe import Pipe

    properties = Pipe.Valves.model_json_schema().get("properties") or {}
    surfaces = {
        "Open WebUI's valve panel": {
            str(spec["title"])
            for spec in properties.values()
            if isinstance(spec, dict) and spec.get("title")
        }
    }
    try:
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META
    except ImportError:
        return surfaces
    surfaces["the Config tab"] = {
        str(meta["title"])
        for meta in CONFIG_META.values()
        if isinstance(meta, dict) and meta.get("title")
    }
    return surfaces


def test_every_settings_surface_shows_the_same_name_for_the_same_setting():
    """One setting, one name -- otherwise the label a card cites exists on only one screen.

    Every core valve carried two labels: the pydantic title Open WebUI's own panel renders and
    the config_meta title the Config tab renders. An admin told to check `OpenRouter API key`
    opened the valve panel, found "Api Key", and had no match. This holds the two sources equal
    for the settings the cards actually cite; it deliberately does not demand it of all 204,
    which would be a rename sweep and not this property.
    """
    surfaces = _settings_surfaces()
    if len(surfaces) < 2:
        pytest.skip("this artifact ships one settings surface, so the two cannot disagree")

    cited = {
        label
        for template in _error_templates().values()
        for label in _NAMED_CONTROL_RE.findall(template)
        if any(label in labels for labels in surfaces.values())
    }
    assert cited, "no error card names a setting at all; this guard is checking nothing"

    split = {
        label: sorted(name for name, labels in surfaces.items() if label not in labels)
        for label in sorted(cited)
    }
    missing = {label: absent for label, absent in split.items() if absent}
    assert not missing, (
        "these settings are named by an error card under a label that only some of this "
        f"artifact's settings screens show: {missing}"
    )


def test_no_error_card_names_a_control_that_does_not_exist():
    """A card that names a setting by a label no settings screen shows sends the reader
    hunting for something they will never find.

    Both the valve labels and the card text are read at runtime, so renaming a valve without
    updating the card that cites it reddens this instead of shipping. This is the shape of
    the defect that survived for seven months in the rejected-request card, which closed by
    telling admins to enable a control that had been removed.

    Checked against EVERY surface the artifact ships rather than one of them, so a label
    present in only the Config tab -- which two of the four artifacts do not have -- is a
    failure here rather than a name those two artifacts show nowhere.
    """
    surfaces = _settings_surfaces()
    assert surfaces, "no settings surface was found at all; the schema shape changed"
    for name, labels in surfaces.items():
        assert len(labels) > 150, f"only {len(labels)} labels read from {name}; its shape changed"

    scanned: dict[str, set[str]] = {
        name: set(_NAMED_CONTROL_RE.findall(template))
        for name, template in _error_templates().items()
    }
    assert any(scanned.values()), (
        "no card names a control at all, so this scan is checking nothing; either the "
        f"cards that point an admin at a setting lost the reference or {_NAMED_CONTROL_RE.pattern!r} "
        f"stopped matching the way they write it. scanned {sorted(scanned)}"
    )

    unfindable: dict[str, dict[str, list[str]]] = {}
    for template_name, named in scanned.items():
        absent = {
            surface: sorted(named - labels)
            for surface, labels in surfaces.items()
            if named - labels
        }
        if absent:
            unfindable[template_name] = absent
    assert not unfindable, f"these cards name controls an admin cannot find: {unfindable}"


# ---------------------------------------------------------------------------
# Claims about where a setting takes effect, and what a card's links lead to.
# ---------------------------------------------------------------------------


def _chat_completions_payload(*, auto_context_trimming: bool) -> dict:
    """Send one request all the way to the body the chat-completions leg posts."""
    from open_webui_openrouter_pipe.api.transforms import (
        ResponsesBody,
        _filter_openrouter_chat_request,
        _responses_payload_to_chat_completions_payload,
        apply_context_transforms,
    )

    body = ResponsesBody(model="anthropic/claude-3", input=[], stream=True)
    apply_context_transforms(body, auto_context_trimming=auto_context_trimming)
    return _filter_openrouter_chat_request(
        _responses_payload_to_chat_completions_payload(body.model_dump(exclude_none=True))
    )


@pytest.mark.parametrize("auto_context_trimming", [True, False])
def test_the_trimming_instruction_survives_the_trip_to_chat_completions(auto_context_trimming):
    """The setting is not endpoint-specific, and the Config tab used to say it was.

    The instruction is a plugin entry on the request body, and the chat-completions
    conversion copies `plugins` across verbatim before an allow-list that names it. So a
    model forced onto that endpoint is trimmed exactly as one on `/responses` is. The
    Config tab told admins the opposite -- "a model routed to `/chat/completions` is
    unaffected either way" -- while the new context-limit card tells them to switch the
    setting on, so an admin who followed the card into the tab read that it could not
    help them and turned it back off.

    Parametrised both ways: a payload that always carries the plugin, and one that never
    does, each fail one arm.
    """
    plugins = _chat_completions_payload(auto_context_trimming=auto_context_trimming).get("plugins") or []
    carried = any(
        isinstance(entry, dict) and entry.get("id") == "context-compression" for entry in plugins
    )
    assert carried is auto_context_trimming, (
        f"with the setting {'on' if auto_context_trimming else 'off'} the chat-completions "
        f"body carries plugins={plugins!r}"
    )


def test_the_trimming_setting_is_not_described_as_one_endpoint_only():
    """The help text has to match the payload the test above measures."""
    pytest.importorskip(
        "open_webui_openrouter_pipe.plugins.pipe_dashboard",
        reason="the --no-plugins artifacts omit pipe_dashboard by design",
    )
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META

    detail = CONFIG_META["AUTO_CONTEXT_TRIMMING"]["detail"]
    excluded = re.search(
        r"/chat/completions`?[^.]{0,40}\b(is unaffected|is not affected|has no effect|does nothing)\b",
        detail,
        re.I,
    )
    assert excluded is None, (
        "the tab still tells an admin the setting cannot help a model on that endpoint: "
        f"{excluded and excluded.group(0)!r}"
    )
    assert "/chat/completions" in detail, (
        "an admin with a model forced onto that endpoint needs the question answered, not "
        "left out; say that it applies there too"
    )


@pytest.mark.parametrize("auto_context_trimming", [True, False])
def test_the_trimming_off_state_admits_what_openrouter_still_compresses(auto_context_trimming):
    """Off does not mean "no compression"; it means the pipe says nothing either way.

    OpenRouter compresses on every endpoint of 8k (8,192 tokens) or less by default, and
    turning that off takes an explicit `enabled: false` the pipe never sends -- it sets a
    different field and returns. The tab told an admin the request "fails outright", so
    someone who switched this off to guarantee full-fidelity prompts was silently compressed
    on every small-context model in the catalogue with nothing on screen to notice.

    The payload is measured both ways here, so the wording is checked against what the pipe
    actually sends rather than against itself: an off arm that started sending the explicit
    disable would redden this, and the sentence would then be the one that needs rewriting.
    """
    pytest.importorskip(
        "open_webui_openrouter_pipe.plugins.pipe_dashboard",
        reason="the --no-plugins artifacts omit pipe_dashboard by design",
    )
    from open_webui_openrouter_pipe.api.transforms import ResponsesBody, apply_context_transforms
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META

    body = ResponsesBody(model="anthropic/claude-3", input=[], stream=True)
    apply_context_transforms(body, auto_context_trimming=auto_context_trimming)
    entries = [
        entry for entry in (body.plugins or [])
        if isinstance(entry, dict) and entry.get("id") == "context-compression"
    ]
    if auto_context_trimming:
        assert entries, "the on arm sends no compression instruction, so this test is measuring nothing"
        return

    assert not entries, (
        f"the off arm now sends {entries!r}; if that is the explicit disable then OpenRouter no "
        "longer compresses small-context endpoints and the help text below has to be rewritten"
    )

    detail = CONFIG_META["AUTO_CONTEXT_TRIMMING"]["detail"]
    claim = next(
        (
            sentence for sentence in re.split(r"(?<=[.!?])\s+", detail)
            if "fails outright" in sentence
        ),
        None,
    )
    assert claim is not None, (
        "the tab no longer says what happens with the setting off; an admin needs that answered"
    )
    assert "8,192" in claim, (
        "the tab still promises an unqualified failure with the setting off, while OpenRouter "
        f"compresses every endpoint of 8,192 tokens or less by default: {claim!r}"
    )


_PAGE_NAME_RE = re.compile(r"\b([A-Za-z][A-Za-z]+)\s+(?:and\s+([A-Za-z]+)\s+)?pages?\b")


def test_a_page_a_help_text_promises_is_a_page_that_cards_default_actually_links():
    """The tab describes what is in the box; an edit to the box has to move the description.

    The out-of-credits default linked `https://openrouter.ai/usage`, a path in none of the
    vendored documentation, and the tab described it as "the OpenRouter credits and usage
    pages". Replacing the link with the Activity page left the description behind, so an
    admin reading the tab believed a link was there that was not.

    Read from the settings metadata and the template constants at runtime, so this holds
    for every template setting rather than the one that was wrong.
    """
    pytest.importorskip(
        "open_webui_openrouter_pipe.plugins.pipe_dashboard",
        reason="the --no-plugins artifacts omit pipe_dashboard by design",
    )
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META

    templates = _error_templates()
    checked: dict[str, set[str]] = {}
    wrong: dict[str, list[str]] = {}
    for valve, meta in CONFIG_META.items():
        template = templates.get(f"DEFAULT_{valve}")
        if template is None:
            continue
        linked = {
            url.rstrip("/").rsplit("/", 1)[-1].casefold()
            for url in _CARD_URL_RE.findall(template)
        }
        named = {
            word.casefold()
            for match in _PAGE_NAME_RE.finditer(str(meta.get("detail") or ""))
            for word in match.groups()
            if word and word.casefold() != "openrouter"
        }
        if named:
            checked[valve] = named
        absent = sorted(named - linked)
        if absent:
            wrong[valve] = absent

    assert checked, (
        "no template setting's help text names a page at all, so this scan is checking "
        f"nothing; either the descriptions stopped naming pages or {_PAGE_NAME_RE.pattern!r} "
        "stopped matching the way they are written"
    )
    assert not wrong, (
        "these help texts promise a page the setting's own default does not link: "
        f"{wrong} (linked destinations came from the template constants)"
    )


# ---------------------------------------------------------------------------
# The rate-limit card: the remedy has to fit the limit that was hit.
# ---------------------------------------------------------------------------


_CHANGE_MODEL_RE = re.compile(r"\b(?:switch\w*|mov\w+|chang\w+|try)\b[^.]{0,60}\bmodel\b", re.I)
"""Any way the card might tell a reader to take the work to a different model."""

_NEGATED_RE = re.compile(r"\b(?:does not|do not|doesn't|never|cannot|can't|not)\b", re.I)
"""Marks the sentence as a statement that changing model fails, not a recommendation."""


@pytest.mark.parametrize(
    "message",
    [
        "Rate limit exceeded: free-models-per-day",
        "Rate limit exceeded: requests per minute",
    ],
    ids=["free-daily-cap", "per-minute"],
)
def test_the_rate_limit_card_does_not_send_a_free_user_to_another_free_model(message):
    """One card, two limits, and switching model only clears one of them.

    OpenRouter governs capacity globally and states the `:free` per-day allowance as a
    count of `:free` requests, not a per-model one, so moving between free models cannot
    lift it. The card raised the free caps in one bullet and then closed by recommending a
    model switch in the next, which is the one action that cannot work for the failure it
    was most likely rendered for. Switching IS the fastest remedy on the paid path, which
    is why the bullet is qualified rather than dropped.
    """
    card = _render_status_card(429, message)
    assert message in card, "the message did not render, so the rest of this proves nothing"

    bullets = [line for line in card.splitlines() if line.startswith("- ")]
    assert bullets, f"the card rendered no advice at all:\n{card}"

    recommended = [
        b for b in bullets if _CHANGE_MODEL_RE.search(b) and not _NEGATED_RE.search(b)
    ]
    assert recommended, (
        f"the card no longer recommends changing model at all, which is the fastest "
        f"remedy on the paid path:\n{card}"
    )
    for bullet in recommended:
        assert re.search(r"\bpaid\b", bullet, re.I), (
            "changing model is recommended without saying it is the paid path; a reader "
            f"who hit the free daily cap follows it and hits the same cap: {bullet!r}"
        )

    free = [b for b in bullets if ":free" in b]
    assert free, f"the card no longer mentions the free caps at all:\n{card}"
    assert any(
        re.search(r"\b(?:does not|do not|doesn't|never)\b[^.]{0,80}\b(?:lift|lifts|raise|raises|help|helps|increase|increases)\b", b, re.I)
        and re.search(r"\bfree model\b", b, re.I)
        for b in free
    ), (
        "the free-cap bullet does not say that moving between free models fails to lift "
        f"the cap, so the switching advice below it reads as applying to it: {free!r}"
    )


# ---------------------------------------------------------------------------
# No card shows a reader the name a setting has in the source.
# ---------------------------------------------------------------------------


_RAW_IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")


def test_no_error_card_shows_a_reader_a_settings_raw_identifier():
    """`**FREE_MODEL_FILTER**` is a name no settings screen displays anywhere.

    The blocked-model card printed three of them in bold and then told the reader to ask
    an admin to update the filters, so the reader relayed strings the admin could not
    search for. The sibling guard over backticked labels could not see these: they were
    bold, not backticked, and the pattern it matches needs two words.

    The comparison is against the valve names read at runtime, so a setting renamed in
    the source and left in a card reddens here. An environment variable an admin really
    does set by that name -- `WEBUI_SECRET_KEY` -- is not a setting of this pipe and is
    deliberately still allowed.
    """
    from open_webui_openrouter_pipe import Pipe

    settings = set(Pipe.Valves.model_json_schema().get("properties") or {})
    assert len(settings) > 150, f"only {len(settings)} settings read; the schema shape changed"

    leaked = {
        name: sorted(found)
        for name, template in _error_templates().items()
        if (found := set(_RAW_IDENTIFIER_RE.findall(template)) & settings)
    }
    assert not leaked, (
        "these cards print a setting under the name it has in the source, which appears on "
        f"no settings screen the reader can search: {leaked}"
    )


@pytest.mark.parametrize(
    "valve",
    ["MODEL_ID", "FREE_MODEL_FILTER", "TOOL_CALLING_FILTER"],
)
def test_the_blocked_model_card_names_each_filter_it_reports(valve):
    """Dropping the identifier is only half of it -- the row still has to be identifiable.

    Each of the three settings the card reports is named under the label its settings
    screens show, read from those screens rather than quoted here, and checked against
    EVERY surface the artifact ships: the `--no-plugins` builds have no Config tab, so
    there the valve panel is the only place a reader can look the name up.
    """
    from open_webui_openrouter_pipe.core.config import DEFAULT_MODEL_RESTRICTED_TEMPLATE

    for surface, label in _surface_labels(valve).items():
        assert label in DEFAULT_MODEL_RESTRICTED_TEMPLATE, (
            f"the blocked-model card reports {valve} but never names it as {label!r}, the "
            f"label {surface} shows for it:\n{DEFAULT_MODEL_RESTRICTED_TEMPLATE}"
        )


async def _blocked_model_card(
    *,
    valve_setup,
    catalog: set[str],
    allowlist: set[str],
    model: str,
    specs: dict | None = None,
    dynamic_specs: dict | None = None,
    zdr_model_ids: set[str] | None = None,
    user_valve_rejected: list[str] | None = None,
    user_requests_zdr: bool = False,
) -> str:
    """Render the card a blocked request really produces, through the real orchestrator.

    The reasons are substituted at runtime from two files, so the constant proves nothing
    about what the reader is handed; this returns the finished text.
    """
    import logging

    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.models.registry import ModelFamily, OpenRouterModelRegistry
    from open_webui_openrouter_pipe.requests.orchestrator import RequestOrchestrator

    pipe = Pipe()
    valve_setup(pipe.valves)
    orchestrator = RequestOrchestrator(pipe, logging.getLogger("blocked-model-card"))

    if user_requests_zdr:
        carried_user_valves = Pipe.UserValves(REQUEST_ZDR=True)
    elif user_valve_rejected:
        carried_user_valves = Pipe.UserValves()
    else:
        carried_user_valves = None

    kept_specs = OpenRouterModelRegistry._specs.copy()
    kept_zdr = OpenRouterModelRegistry._zdr_model_ids
    if specs is not None:
        OpenRouterModelRegistry._specs = specs
    if dynamic_specs is not None:
        ModelFamily.set_dynamic_specs(dynamic_specs)
    OpenRouterModelRegistry._zdr_model_ids = zdr_model_ids
    try:
        shown = await orchestrator.process_request(
            body={"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": False},
            __user__={"id": "user-1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__={"model": {"id": model}},
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=pipe.valves,
            session=cast(Any, object()),
            openwebui_model_id=model,
            pipe_identifier="test-pipe",
            allowlist_norm_ids=allowlist,
            enforced_norm_ids=set(),
            catalog_norm_ids=catalog,
            features={},
            user_valves=carried_user_valves,
            rejected_user_valves=user_valve_rejected,
        )
    finally:
        OpenRouterModelRegistry._specs = kept_specs
        OpenRouterModelRegistry._zdr_model_ids = kept_zdr
        ModelFamily.set_dynamic_specs({})
        await pipe.close()
    return str(shown or "")


def _blocked_arm(arm: str) -> tuple[dict, list[str], list[str]]:
    """One way to get blocked: the orchestrator call that produces it, and what it reports."""
    paid = {"pricing": {"prompt": "0.001", "completion": "0.002"}}
    free = {"pricing": {"prompt": "0", "completion": "0"}}

    if arm == "outside the catalog and the allowlist":
        def setup(valves):
            valves.MODEL_ID = "allowed.model"

        return (
            {
                "valve_setup": setup,
                "catalog": {"allowed.model"},
                "allowlist": {"allowed.model"},
                "model": "other.model",
            },
            ["MODEL_ID"],
            [],
        )

    if arm == "caught by every filter":
        def setup(valves):
            valves.MODEL_ID = "allowed.model"
            valves.FREE_MODEL_FILTER = "only"
            valves.TOOL_CALLING_FILTER = "only"
            valves.ZDR_MODELS_ONLY = True

        return (
            {
                "valve_setup": setup,
                "catalog": {"paid.model", "allowed.model"},
                "allowlist": {"allowed.model"},
                "model": "paid.model",
                "specs": {"paid.model": paid},
                "dynamic_specs": {"paid.model": {"supported_parameters": ["temperature"]}},
                "zdr_model_ids": set(),
            },
            ["MODEL_ID", "FREE_MODEL_FILTER", "TOOL_CALLING_FILTER", "ZDR_MODELS_ONLY"],
            [],
        )

    if arm == "excluded by the same filters inverted":
        def setup(valves):
            valves.MODEL_ID = "auto"
            valves.FREE_MODEL_FILTER = "exclude"
            valves.TOOL_CALLING_FILTER = "exclude"

        return (
            {
                "valve_setup": setup,
                "catalog": {"free.model"},
                "allowlist": set(),
                "model": "free.model",
                "specs": {"free.model": free},
                "dynamic_specs": {"free.model": {"supported_parameters": ["tools"]}},
            },
            ["FREE_MODEL_FILTER", "TOOL_CALLING_FILTER"],
            [],
        )

    if arm == "refused by ZDR enforcement":
        def setup(valves):
            valves.ZDR_ENFORCE = True

        return (
            {
                "valve_setup": setup,
                "catalog": set(),
                "allowlist": set(),
                "model": "some.model",
                "zdr_model_ids": set(),
            },
            ["ZDR_ENFORCE"],
            [],
        )

    if arm == "refused with the ZDR list unreadable":
        def setup(valves):
            valves.ZDR_ENFORCE = True

        return (
            {
                "valve_setup": setup,
                "catalog": set(),
                "allowlist": set(),
                "model": "some.model",
                "zdr_model_ids": None,
            },
            ["ZDR_ENFORCE"],
            [],
        )

    if arm == "refused on an unreadable user preference":
        def setup(valves):
            valves.ZDR_ENFORCE = False
            valves.ALLOW_USER_ZDR_OVERRIDE = True

        return (
            {
                "valve_setup": setup,
                "catalog": set(),
                "allowlist": set(),
                "model": "some.model",
                "zdr_model_ids": set(),
                "user_valve_rejected": ["REQUEST_ZDR"],
            },
            [],
            ["REQUEST_ZDR"],
        )

    if arm == "refused on the user's own preference":
        def setup(valves):
            valves.ZDR_ENFORCE = False
            valves.ALLOW_USER_ZDR_OVERRIDE = True

        return (
            {
                "valve_setup": setup,
                "catalog": set(),
                "allowlist": set(),
                "model": "some.model",
                "zdr_model_ids": set(),
                "user_requests_zdr": True,
            },
            [],
            ["REQUEST_ZDR"],
        )

    raise AssertionError(f"no orchestrator call for {arm!r}")


_BLOCKED_ARMS = (
    "outside the catalog and the allowlist",
    "caught by every filter",
    "excluded by the same filters inverted",
    "refused by ZDR enforcement",
    "refused with the ZDR list unreadable",
    "refused on an unreadable user preference",
    "refused on the user's own preference",
)


def _all_setting_names() -> set[str]:
    from open_webui_openrouter_pipe import Pipe

    names = set(Pipe.Valves.model_json_schema().get("properties") or {})
    names |= set(Pipe.UserValves.model_json_schema().get("properties") or {})
    return names


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", _BLOCKED_ARMS)
async def test_the_blocked_model_card_a_reader_gets_names_no_source_identifier(arm):
    """The reasons are substituted at runtime, so only the finished card settles this.

    `restriction_reasons` is filled in by the pipe and by the orchestrator, not by the
    template, and both used to hand it the name the setting has in the source --
    `MODEL_ID`, `FREE_MODEL_FILTER=only`, `ZDR_ENFORCE`. On the ZDR arms that was the only
    name on the card, because the labelled rows below it are suppressed when their
    variables are empty. Reading the constant said none of this.

    Every way of getting blocked is driven through the orchestrator and the card it hands
    back is what is scanned, so a reason that stops being translated reddens here whether
    it is written in the pipe, in the orchestrator, or in a third place later.
    """
    call, named_valves, named_user_valves = _blocked_arm(arm)
    card = await _blocked_model_card(**call)

    assert "Restricted by" in card, (
        f"the {arm} card reports no reason at all, so a scan of it proves nothing:\n{card}"
    )
    reported = next(
        line for line in card.splitlines() if "Restricted by" in line
    ).split(":", 1)[1]
    assert reported.strip(), f"the reason row on the {arm} card is empty:\n{card}"

    leaked = sorted(name for name in _all_setting_names() if name in card)
    assert not leaked, (
        f"the {arm} card prints {leaked} -- the name those settings have in the source, "
        f"which appears on no screen the reader can search:\n{card}"
    )

    for valve in named_valves:
        for surface, label in _surface_labels(valve).items():
            assert label in reported, (
                f"the {arm} card reports {valve} as {reported!r}, which never names it as "
                f"{label!r} -- the label {surface} shows for it. A reader looking the reason "
                f"up finds nothing:\n{card}"
            )

    if named_user_valves:
        from open_webui_openrouter_pipe import Pipe

        for valve in named_user_valves:
            label = Pipe.UserValves.model_fields[valve].title
            assert label and label in reported, (
                f"the {arm} card reports the user's {valve} preference as {reported!r}, which "
                f"never names it as {label!r}, the label their own settings panel shows:\n{card}"
            )


def _reason_strings_the_code_can_report() -> set[str]:
    """Every reason the package can put on a blocked card, read off the code that writes them.

    Followed from the labeller's own call sites: a list written inline there, a name bound to
    a string in the same scope, and the list the reason builder returns. A hand-kept list here
    would only have covered the reasons that were already translated, which is the failure.
    """
    from tests.package_sources import parsed_sources

    produced: set[str] = set()
    reported: set[str] = set()
    from_builder = False

    def _constants(node: ast.expr, bindings: dict[str, list[ast.expr]]) -> set[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            return {
                bound.value
                for bound in bindings.get(node.id, [])
                if isinstance(bound, ast.Constant) and isinstance(bound.value, str)
            }
        return set()

    for _path, _source, tree in parsed_sources("open_webui_openrouter_pipe"):
        for scope in _scopes(tree):
            named = getattr(scope, "name", "")
            bindings = _bindings(scope)
            for node in _own_nodes(scope):
                if not isinstance(node, ast.Call):
                    continue
                callee = _callee(node)
                if (
                    named == "_model_restriction_reasons"
                    and callee == "append"
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "reasons"
                ):
                    for arg in node.args:
                        produced |= _constants(arg, bindings)
                if callee != "_model_restriction_labels" or not node.args:
                    continue
                first = node.args[0]
                if isinstance(first, ast.List):
                    for element in first.elts:
                        reported |= _constants(element, bindings)
                elif isinstance(first, ast.Name):
                    for bound in bindings.get(first.id, []):
                        if isinstance(bound, ast.Call) and _callee(bound) == "_model_restriction_reasons":
                            from_builder = True

    assert from_builder, (
        "no call site was seen handing the labeller what the reason builder returns; the scan "
        "has lost the path it exists to follow"
    )
    return produced | reported


def test_every_reason_the_code_can_report_resolves_to_a_label_and_not_the_fallback(pipe_instance):
    """The card names a setting or it names nothing; the fallback names nothing.

    `REQUEST_ZDR` is the arm the orchestrator takes whenever a user's own per-chat toggle
    parses cleanly -- the common one -- and it lives on the user's settings rather than the
    admin's, so the admin-settings lookup returned nothing and the card fell through to "a
    restriction configured for this pipe". A reader who ticked their own toggle was told to
    go and find a setting that does not exist, while the one remedy went unnamed.

    Every reason is scanned out of the package, so a new one added without a label reddens
    this. The expected wording is read from the settings schema rather than written out, so
    one hardcoded label cannot satisfy two rows.
    """
    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.pipe import _RESTRICTION_REASON_FALLBACK

    reasons = _reason_strings_the_code_can_report()
    assert len(reasons) >= 8, (
        f"the scan sees only {sorted(reasons)}; it has stopped finding the reason strings and "
        "would pass whatever the labeller did"
    )

    pipe = pipe_instance
    labels = {
        reason: pipe._restriction_reason_label(reason, valves=pipe.valves)
        for reason in sorted(reasons)
    }

    unnamed = sorted(r for r, label in labels.items() if not label or label == _RESTRICTION_REASON_FALLBACK)
    assert not unnamed, (
        f"these reasons put {_RESTRICTION_REASON_FALLBACK!r} on the card, which names no setting "
        f"the reader can act on: {unnamed}"
    )

    leaked = sorted(r for r, label in labels.items() if r.split("=", 1)[0] in label)
    assert not leaked, f"these reasons print their own source name on the card: {leaked}"

    wrong: dict[str, tuple[str, str]] = {}
    for reason, label in labels.items():
        field_name = reason.split("=", 1)[0]
        field = Pipe.Valves.model_fields.get(field_name) or Pipe.UserValves.model_fields.get(field_name)
        title = getattr(field, "title", None) if field is not None else None
        if isinstance(title, str) and title and label != title:
            wrong[reason] = (label, title)
    assert not wrong, (
        "these reasons name a setting, so the card has to print the label that setting's own "
        f"panel shows: {wrong}"
    )

    assert len(set(labels.values())) >= 5, (
        f"the reasons collapse onto {sorted(set(labels.values()))}; a card built from them "
        "cannot tell the reader which setting stopped the request"
    )


# ---------------------------------------------------------------------------
# A notice promised "in the chat" has to arrive somewhere Open WebUI keeps it.
# ---------------------------------------------------------------------------


_OWUI_PERSISTS = {"status", "message", "replace", "embeds", "files", "source", "citation"}
"""The event types Open WebUI's socket emitter writes to the message row.

Taken from the deployment reference: everything else -- `notification` among them -- is
forwarded to the browser and never stored, so a promise that a reader will find something
"in the chat" is only kept by a type in this set, or by the value the pipe returns.
"""


async def _video_withheld_run(
    refusal: Exception, preference: str
) -> tuple[list[dict], str]:
    """Submit a video request carrying a preference the video schema has no field for.

    ``refusal`` is what submit raises, because the two rejections leave `generate` down
    different arms: an `OpenRouterAPIError` is handed to the shared error reporter, and
    anything else builds the adapter's own failure card. Both have to carry the record.

    ``preference`` is the provider key the request asks for, so the two arms ask for
    different ones: a card hardcoded in production names one of them and fails the other.
    """
    from types import SimpleNamespace

    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.core.config import EncryptedStr

    class _NoPersistence:
        async def load_message_content(self, *, chat_id: str, message_id: str) -> str:
            return ""

    class _RefusingClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def submit(self, _payload):
            raise refusal

    Pipe._video_global_semaphore = None
    Pipe._video_global_limit = 0
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.MAX_CONCURRENT_VIDEO_GENS = 1
    adapter = pipe._ensure_video_generation_adapter()
    cast(Any, adapter)._persistence = _NoPersistence()

    import open_webui_openrouter_pipe.integrations.video as video_module

    original = video_module.OpenRouterVideoClient
    video_module.OpenRouterVideoClient = _RefusingClient
    events: list[dict] = []

    async def emitter(event):
        events.append(event)

    try:
        content = await adapter.generate(
            body={"messages": [{"role": "user", "content": "make a video"}]},
            responses_body=SimpleNamespace(provider={preference: True}),
            valves=pipe.valves,
            session=object(),
            event_emitter=emitter,
            metadata={"chat_id": "chat-1", "message_id": "msg-1", "user_id": "user-1"},
            user={"id": "user-1"},
            request=None,
            user_obj={"id": "user-1"},
            normalized_model_id="openai.sora-2-pro",
            api_model_id="openai/sora-2-pro",
        )
    finally:
        video_module.OpenRouterVideoClient = original
        await pipe.close()
    return events, content


async def _image_withheld_run() -> tuple[list[dict], str]:
    """Report one image parameter the model's own contract superseded.

    The note comes from the producer the request path uses, so its text is the product's
    rather than this file's -- exact pixels supersede a ratio that is not their shape.
    """
    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.integrations.image import size_consistency_notes

    notes = size_consistency_notes({"size": "1024x1024", "aspect_ratio": "16:9"})
    assert notes, "the pixels-supersede-the-ratio rule produced no note to report"

    pipe = Pipe()
    adapter = pipe._ensure_image_generation_adapter()
    events: list[dict] = []

    async def emitter(event):
        events.append(event)

    try:
        await adapter._report_notes(
            notes, api_model_id="google/gemini-3-pro-image", event_emitter=emitter
        )
    finally:
        await pipe.close()
    return events, ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "kept_with_the_message", "withheld_name"),
    [
        ("video", True, "zdr"),
        ("video", True, "data_collection"),
        ("video-rejected", True, "zdr"),
        ("video-rejected", True, "data_collection"),
        ("image", False, "aspect_ratio"),
    ],
)
async def test_a_withheld_notice_is_described_by_the_channel_it_really_uses(
    kind, kept_with_the_message, withheld_name
):
    """Both paths drop something the user asked for; only one of them keeps the notice.

    The video path folds the list into the assistant message, next to the file-host
    record that was already carried there, so a reader who comes back to the conversation
    and finds a clip generated without their reference still finds the reason. The image
    path raises its notes before the request is even sent, on a turn whose answer then
    streams, and there is no assistant message to fold into at that point -- so it stays a
    toast and the help text says so.

    Four video rows, because the two ways a submit is refused leave `generate` down
    different arms and only one of them was carrying the record: OpenRouter rejecting the
    request -- a 400, a bad key, a rate limit, the likeliest refusal of the two -- is
    reported by the shared error formatter, and that is the arm that returned the card
    with nothing folded in. Each arm is driven twice, asking for a DIFFERENT provider
    preference, because one row per arm is satisfied by a card with that preference
    hardcoded into the arm -- measured: a `return "data_collection was not sent"` planted
    in the rejection arm passed a three-row version of this test.

    Parametrised over all five, and asserted on the CHANNEL rather than the wording, so
    one implementation cannot stand for the other: making the image path persist reddens
    the image arm, and reverting either video path to a toast reddens that video arm.
    """
    if kind == "image":
        events, content = await _image_withheld_run()
    else:
        events, content = await _video_withheld_run(
            OpenRouterAPIError(
                status=400,
                reason="Bad Request",
                openrouter_message="this model does not take that request",
            )
            if kind == "video-rejected"
            else RuntimeError("submit refused"),
            withheld_name,
        )

    named = re.compile(rf"\b{re.escape(withheld_name)}\b[^\n]{{0,40}}was not sent")
    in_returned = bool(named.search(content))
    in_toast = any(
        event.get("type") == "notification"
        and named.search(str((event.get("data") or {}).get("content") or ""))
        for event in events
    )
    in_kept_event = any(
        event.get("type") in _OWUI_PERSISTS
        and named.search(json.dumps(event.get("data") or {}, default=str))
        for event in events
    )

    assert in_toast is not kept_with_the_message, (
        f"the {kind} path delivers the withheld list as a toast: {in_toast}; expected "
        f"{not kept_with_the_message}. Toasts are not written to the message row, so a "
        "reader who reloads finds nothing"
    )
    assert (in_returned or in_kept_event) is kept_with_the_message, (
        f"the {kind} path puts the withheld list on a channel Open WebUI keeps: "
        f"returned={in_returned} kept_event={in_kept_event}; expected "
        f"{kept_with_the_message}. events={[event.get('type') for event in events]}"
    )


# ---------------------------------------------------------------------------
# A record wrapped in hidden markers has to render with the markers still hidden.
# ---------------------------------------------------------------------------


_MARKER_RECORDS = (
    "the withheld-settings notice",
    "the file-host relay notice",
    "the pending-job message",
    "the failed-job message",
    "the intent disclosure",
    "the clarification question",
)


def _build_marker_record(name: str) -> str:
    """Build one of the marker-wrapped records the pipe folds into an assistant message.

    Each is produced by the real builder rather than quoted here, so a change to the
    record's shape is what this measures.
    """
    import logging

    from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter
    from open_webui_openrouter_pipe.integrations.video_intent import (
        ClarificationPayload,
        FramePlanEntry,
        VideoIntentResult,
        render_clarification_message,
        render_intent_disclosure_block,
    )

    if name == "the withheld-settings notice":
        return VideoGenerationAdapter._withheld_record(
            [("resolution", "was not sent because this model publishes no sizes")]
        )

    if name == "the file-host relay notice":
        class _Relaying:
            SEND_MEDIA_VIA_FILE_HOST = True
            SEND_VIDEO_VIA_FILE_HOST = True

        return VideoGenerationAdapter._file_host_record(_Relaying(), {("video", "0x0.st")})

    if name in ("the pending-job message", "the failed-job message"):
        adapter = VideoGenerationAdapter(
            pipe=cast(Any, None), logger=logging.getLogger("marker-records")
        )
        if name == "the pending-job message":
            return adapter._build_pending_content(job_id="job-1", model_id="openai/sora-2")
        return adapter._build_failure_content(
            job_id="job-1", model_id="openai/sora-2", reason="the provider stopped"
        )

    if name == "the intent disclosure":
        return render_intent_disclosure_block(
            VideoIntentResult(
                intent="image_to_video",
                frame_plan=[
                    FramePlanEntry(
                        source="uploaded_attachment",
                        source_index=0,
                        timestamp_seconds=None,
                        target="first_frame",
                    )
                ],
                prompt="a kite over the harbour",
                use_user_prompt=True,
                language="en",
                confidence="high",
                clarification=None,
                reason="one image was attached",
                downgrades=["resolution was lowered"],
            ),
            thumb_urls=["https://example.invalid/frame.png"],
        )

    if name == "the clarification question":
        return render_clarification_message(
            VideoIntentResult(
                intent="ambiguous",
                frame_plan=[],
                prompt="",
                use_user_prompt=False,
                language="en",
                confidence="low",
                clarification=ClarificationPayload(
                    needs=True,
                    question="Which clip should this continue?",
                    options=["the first", "the second"],
                    reason="two clips are in scope",
                ),
                reason="two clips are in scope",
            )
        )

    raise AssertionError(f"no builder for {name!r}")


@pytest.mark.parametrize("record_name", _MARKER_RECORDS)
def test_a_marker_wrapped_record_keeps_its_markers_out_of_what_is_rendered(record_name):
    """The hidden markers are link reference definitions, and one wrong line shows them.

    A link reference definition cannot interrupt a paragraph: put one directly beneath a
    blockquote, a list item or a table row and CommonMark absorbs it as a lazy
    continuation of that paragraph, so the reader is shown
    `[openrouter:v1:withheld_block_end:1]: #` as literal text in their chat. Preceded by
    a blank line -- or by another marker, since a run of definitions is allowed -- it
    opens a block of its own and renders to nothing.

    Both halves are asserted: the rendered output, which is what the reader sees, and the
    line-position rule that produces it, which also covers the list and table cases no
    record uses yet. The namespace is read from the source constant, so renaming the
    markers keeps this green while moving one of them beneath a paragraph does not.
    """
    from markdown_it import MarkdownIt

    from open_webui_openrouter_pipe.core.utils import _KIND_MARKER_NAMESPACE, _KIND_MARKER_RE

    record = _build_marker_record(record_name)
    assert _KIND_MARKER_NAMESPACE in record, (
        f"{record_name} carries no marker at all, so this proves nothing: {record!r}"
    )

    lines = record.split("\n")
    stranded = [
        (index, line)
        for index, line in enumerate(lines)
        if index > 0
        and _KIND_MARKER_RE.match(line)
        and lines[index - 1].strip()
        and not _KIND_MARKER_RE.match(lines[index - 1])
    ]
    assert not stranded, (
        f"{record_name} places a marker directly under a non-blank, non-marker line, which "
        f"CommonMark reads as a continuation of that paragraph: {stranded!r}"
    )

    rendered = MarkdownIt("commonmark").render(record)
    assert _KIND_MARKER_NAMESPACE not in rendered, (
        f"{record_name} prints an internal marker into the reader's chat:\n{rendered}"
    )


def test_the_image_size_help_says_its_notice_is_not_kept():
    """The control whose value gets superseded is where the reader is told what happens.

    It promised "You are told in the chat whenever it is dropped that way", which reads as
    something that stays with the message; it is a toast, so a reader who returns to the
    conversation and finds the wrong shape finds no reason for it. Every variant that
    holds a control back carries the same promise, enumerated from the selection table so
    that a variant added to it cannot arrive unguarded.
    """
    for text in _size_variants_that_withhold_a_control():
        assert re.search(r"\btoast\b", text, re.I), (
            f"the reader is not told what kind of notice this is: {text!r}"
        )
        assert re.search(r"\bnot\b[^.]{0,60}\bkeep\w*\b|\bgone\b", text, re.I), (
            f"the reader is not told the notice does not survive a reload: {text!r}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored_key", "expected_detail"),
    [
        ("", "not configured"),
        ("encrypted:not-decryptable-under-this-secret", "WEBUI_SECRET_KEY changed"),
    ],
    ids=["blank-box", "secret-rotated"],
)
async def test_the_document_records_the_one_template_reached_without_a_status(
    stored_key, expected_detail
):
    """An unreadable key renders the 401 card with no request sent and no status returned.

    The document told an operator that status selection is the only thing that chooses one
    of these templates and that no call site can override it. An operator who believed it
    wrote 401 wording that blames OpenRouter for refusing the credentials -- and shipped it
    to readers whose actual fault was local: a blank key box, or a stored value encrypted
    under a server secret that has since changed.

    Driven rather than read, and over BOTH local causes, so the card has to carry the
    detail of the one that happened -- a constant answer satisfies neither arm.
    """
    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.core.config import EncryptedStr

    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr(stored_key)
    if pipe._resolve_openrouter_api_key(pipe.valves)[1] is None:
        await pipe.close()
        pytest.skip("this build stores the key in a form that decrypts, so there is no fault to render")
    try:
        await pipe._ensure_async_subsystems_initialized()
        answered = await pipe._handle_pipe_call(
            body={"stream": False, "messages": [{"role": "user", "content": "hi"}]},
            __user__={},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__={},
            __tools__=None,
            valves=pipe.valves,
            session=pipe._http_session,
        )
    finally:
        await pipe.close()

    shown = json.dumps(answered, default=str)
    assert "Authentication Failed" in shown, (
        f"a pipe with no readable key did not answer with the authentication card: {shown[:400]}"
    )
    assert expected_detail in shown, (
        f"the card does not carry the local cause it was rendered for: {shown[:600]}"
    )

    doc = (DOCS / "error_handling_and_user_experience.md").read_text(encoding="utf-8")
    claim = doc.index("Status selection is the only thing that chooses one of these templates")
    assert re.search(
        r"AUTHENTICATION_ERROR_TEMPLATE[^\n]{0,600}\b(directly|without|no status)\b",
        doc[claim:],
        re.I,
    ), (
        "the document still states the property with no exception recorded, and the "
        "exception is exactly what decides how the 401 box may be worded"
    )


# ---------------------------------------------------------------------------
# An error card joined to a partial answer has to open its own block beneath it.
# ---------------------------------------------------------------------------


_JOINED_CARDS = {
    "heading": "### Out of credits\n\nAdd credits to this account and try again.",
    "blockquote": "> **Out of credits**\n>\n> Add credits to this account and try again.",
    "prose": "That file could not be read, so the turn stopped where it is.",
}
"""Three card openings, because two of them cannot see the defect this guards.

A heading and a blockquote may both interrupt a paragraph after a single newline, so a
card that opens with either renders correctly even when the separator is wrong. Plain
prose -- a file-access refusal, a permission denial, an upstream sentence quoted as-is --
is absorbed as a soft line break instead, and the reader gets one run-on paragraph whose
second half looks like more of the model's answer.
"""


@pytest.mark.parametrize("card_shape", sorted(_JOINED_CARDS))
@pytest.mark.parametrize(
    "answer",
    ["The capital of France is", "Photosynthesis begins when the leaf"],
    ids=["capital", "photosynthesis"],
)
def test_a_card_joined_to_a_partial_answer_renders_as_its_own_block(answer, card_shape):
    """What the reader sees, measured by rendering it, not by looking for a separator.

    ``join_answer_and_card`` exists for one reason: the separator between a truncated
    answer and the card explaining why it stopped. Both ``answer + card`` and
    ``answer + "\\n" + card`` satisfy every containment-and-ordering guard on this path,
    and both destroy the card -- the first always, the second whenever the card opens
    with prose.

    The expectation is the two halves rendered SEPARATELY and concatenated, so it is
    never the joiner's own output: a constant returned from the joiner cannot equal it,
    and neither can a join that lets the card continue the answer's paragraph. Asserting
    a blank line appears in the joined string was rejected as the alternative -- it pins
    the separator rather than the effect, and any card containing a blank line of its own
    satisfies it whatever the joiner did.
    """
    from markdown_it import MarkdownIt

    from open_webui_openrouter_pipe.core.utils import join_answer_and_card

    card = _JOINED_CARDS[card_shape]
    md = MarkdownIt("commonmark")

    joined = join_answer_and_card(answer, card)
    assert answer in joined and card in joined, (
        f"the joiner dropped one of its two halves: {joined!r}"
    )

    assert md.render(joined) == md.render(answer) + md.render(card), (
        "the card does not open a block of its own beneath the answer, so the reader is "
        f"shown one run-on paragraph.\njoined:\n{md.render(joined)}\n"
        f"answer then card:\n{md.render(answer) + md.render(card)}"
    )


# ------------------------------------------- A CARD AGREES WITH ITS OWN LIST ---
def _help_records_by_id() -> dict[str, list[dict]]:
    """Every recorded contract, keyed by the model id its help card is written for."""
    found: dict[str, list[dict]] = {}
    for path in sorted(FIXTURES.glob("openrouter_image_endpoints_*.json")):
        raw = json.loads(path.read_text())
        found[raw["id"]] = [r for r in (raw.get("endpoints") or [raw]) if isinstance(r, dict)]
    assert len(found) > 30, f"only {len(found)} contracts; the card sweep went hollow"
    return found


_HELP_RECORDS = _help_records_by_id()


def _help_data() -> dict[str, dict]:
    from open_webui_openrouter_pipe.integrations.image_help import IMAGE_HELP_BY_MODEL

    return IMAGE_HELP_BY_MODEL


_CARDED_MODELS = sorted(set(_help_data()) & set(_HELP_RECORDS))

assert len(_CARDED_MODELS) > 30, (
    f"only {len(_CARDED_MODELS)} models have both a card and a contract; the sweep is hollow"
)

_BACKTICKED = re.compile(r"`([^`]+)`")


def _card_prose(model_id: str) -> str:
    entry = _help_data()[model_id]
    return " ".join([entry.get("best_known_for", ""), *(entry.get("tips_and_pitfalls") or [])])


def _card_spec(model_id: str):
    return build_image_model_filter_spec(
        model_id, {"id": model_id, "name": model_id}, _HELP_RECORDS[model_id],
        dedicated_image_api=True,
    )


def _cards_naming_their_settings() -> list[str]:
    naming = []
    for model_id in _CARDED_MODELS:
        published = set(_card_spec(model_id).passthrough)
        if published and set(_BACKTICKED.findall(_card_prose(model_id))) & published:
            naming.append(model_id)
    return naming


assert _cards_naming_their_settings(), (
    "no card names any of its model's published settings, so the subset guard below "
    "cannot fail on anything and proves nothing"
)


@pytest.mark.parametrize("model_id", _CARDED_MODELS, ids=_CARDED_MODELS)
def test_no_image_card_names_a_smaller_setting_set_than_the_one_it_prints(model_id):
    """The prose and the Controls list under it come from the same card; they disagreed.

    One card said its model "takes `style` and `text_layout`" while the list a few lines
    below it -- built from that model's own published contract -- printed three, and a
    second sentence on the same card said three. A reader shown two numbers on one card
    has to guess which is the model.

    Both halves are measured, never asserted as a constant: the published set is read
    from the recorded contract, and the named set from the rendered card. Naming one
    published setting means naming all of them, because a partial list reads as a
    complete one. Parametrised over every recorded model -- three-setting Recraft and
    the models that publish none among them -- so a hardcoded three fails on the
    contracts whose answer is zero.
    """
    published = set(_card_spec(model_id).passthrough)
    named = set(_BACKTICKED.findall(_card_prose(model_id))) & published

    assert not named or named == published, (
        f"{model_id} names {sorted(named)} of the {len(published)} settings its own "
        f"Controls list prints ({sorted(published)}), so the card contradicts itself"
    )


# ---------------------------- A CARD SELLS ONLY WHAT ITS CONTRACT STILL PUBLISHES ---
# What is enforceable here is narrower than "the prose is true", and pretending otherwise
# would be the fake. There is no derivable mapping from an arbitrary English capability
# noun to the parameter that would implement it. ONE property is derivable, and it is the
# only one asserted: a setting named in code style is one some recorded contract
# publishes, and the model whose card names it is the model that publishes it.
#
# A guard that read a hyphenated capability ("super-resolution") as qualifying a noun the
# contract publishes ("resolution") stood here and was removed: its expectation was a
# human theory about English rather than anything production computes, and it flags
# "High-resolution counterpart" -- which is true, and is on two Recraft cards today -- on
# all sixteen recorded models that publish a `resolution` parameter. What went with it: a
# capability claim that leaves a contract without the prose being corrected is no longer
# detected by anything, unless the prose names the parameter in backticks.
_PASSTHROUGH_VOCABULARY = {
    str(name).lower()
    for model_id in _CARDED_MODELS
    for name in _card_spec(model_id).passthrough
}

assert len(_PASSTHROUGH_VOCABULARY) > 10, (
    f"only {sorted(_PASSTHROUGH_VOCABULARY)} published across every recorded contract; "
    "the borrowed-setting guard has almost no vocabulary to catch anything with"
)


@pytest.mark.parametrize("model_id", _CARDED_MODELS, ids=_CARDED_MODELS)
def test_no_card_writes_a_setting_name_its_own_model_does_not_publish(model_id):
    """A setting named in backticks on one card is a setting some contract publishes.

    The card that named `font_inputs` was right; the guard beside this one only compares
    a card against the names its own model publishes, so a card borrowing a neighbour's
    setting -- the same family, one model that publishes it and one that does not -- is
    invisible to it. The vocabulary here is every name published by any recorded
    contract, so borrowing is what it reads.

    Every recorded model is an arm. Of the forty, fifteen publish three settings, three
    publish eight, sixteen publish one, and six publish none -- and on those six the
    answer is that naming any of the vocabulary at all is wrong.
    """
    published = {str(name).lower() for name in _card_spec(model_id).passthrough}
    named = {token.strip().lower() for token in _BACKTICKED.findall(_card_prose(model_id))}

    borrowed = sorted((named & _PASSTHROUGH_VOCABULARY) - published)
    assert not borrowed, (
        f"the {model_id} card writes {borrowed}, which some other model publishes and "
        f"this one does not; it publishes {sorted(published) or 'nothing'}"
    )


# ------------------------------------ TWO SURFACES, ONE COMPARISON PER MODEL ---
_COMPARISON = re.compile(
    r"\b(same|lower|higher)\b((?:[\s\-]+[A-Za-z`_][\w`.\-]*){1,3})", re.I
)
_UNGRADED = frozenset(
    """the a an and or at in on of for to its it this that with than as but so is are
    was be by from up one two three four five six same lower higher no not all both
    each""".split()
)


def _doc_model_sections() -> dict[str, str]:
    text = (DOCS / "openrouter_image_generation.md").read_text()
    sections: dict[str, str] = {}
    for part in re.split(r"^### ", text, flags=re.M)[1:]:
        marked = re.search(r"^> \*\*id\*\*: `([^`]+)`", part, flags=re.M)
        if marked:
            sections[marked.group(1)] = part
    assert len(sections) > 20, f"only {len(sections)} model sections found in the doc"
    return sections


_DOC_SECTIONS = _doc_model_sections()


def _comparisons(text: str) -> dict[str, set[str]]:
    """Every "same/lower/higher X" verdict, indexed by each thing X could be about.

    The subject is not always the word straight after the comparator: "same Sourceful
    quality" grades quality, and reading only the adjacent word grades Sourceful and
    misses the claim entirely -- which is how the regression this guards survived a
    first attempt at a guard. So the following few words are each indexed, minus the
    ones that grade nothing.
    """
    flattened = re.sub(r"\s+", " ", text.replace("`", "").replace("\n", " "))
    found: dict[str, set[str]] = {}
    for comparator, tail in _COMPARISON.findall(flattened):
        for word in re.findall(r"[A-Za-z][\w.\-]*", tail):
            subject = word.lower().rstrip(".,;:")
            if subject in _UNGRADED:
                continue
            found.setdefault(subject, set()).add(comparator.lower())
    return found


_DOCUMENTED_CARDS = sorted(set(_help_data()) & set(_DOC_SECTIONS))


def _models_comparing_the_same_thing_twice() -> list[str]:
    return [
        model_id
        for model_id in _DOCUMENTED_CARDS
        if set(_comparisons(_card_prose(model_id))) & set(_comparisons(_DOC_SECTIONS[model_id]))
    ]


assert len(_models_comparing_the_same_thing_twice()) > 5, (
    "no model is compared on the same subject by both surfaces, so the agreement guard "
    "below has nothing to disagree about"
)


@pytest.mark.parametrize("model_id", _DOCUMENTED_CARDS, ids=_DOCUMENTED_CARDS)
def test_the_card_and_the_doc_never_grade_the_same_model_differently(model_id):
    """"Same quality" on one surface and "lower quality" on the other, for one model.

    A three-part sentence lost its middle line in a refactor and spliced "same Sourceful"
    onto "quality", turning a cheaper, lower-quality variant into an equal one at less
    money. The document kept the true wording, so the two surfaces graded the same model
    opposite ways and nothing noticed.

    Both sides are extracted, never written down here: every "same/lower/higher X" clause
    on each surface, compared on the subjects both surfaces actually grade. A subject one
    surface never mentions is out of scope -- that is an omission, not a contradiction --
    but a subject both name must carry the same verdict on both.
    """
    from_card = _comparisons(_card_prose(model_id))
    from_doc = _comparisons(_DOC_SECTIONS[model_id])

    for subject in sorted(set(from_card) & set(from_doc)):
        assert from_card[subject] == from_doc[subject], (
            f"the {model_id} card grades {subject!r} as {sorted(from_card[subject])} and "
            f"the document grades it {sorted(from_doc[subject])}"
        )


# ------------------------ THE PANEL PROMISE COVERS WHAT THE PANEL ACTUALLY DRAWS ---
def _install_panels_description() -> str:
    from open_webui_openrouter_pipe.core.config import Valves

    return Valves.model_fields["AUTO_INSTALL_IMAGE_FILTERS"].description or ""


def _panel_titles(model_id: str, records: list[dict] | None, dedicated: bool) -> list[str]:
    spec = build_image_model_filter_spec(
        model_id, {"id": model_id, "name": model_id}, records, dedicated_image_api=dedicated
    )
    body = render_image_model_filter_source(spec).split("class UserValves(BaseModel):", 1)[1]
    body = body.split("    def __init__", 1)[0]
    return [double or single for double, single in _TITLE_PAIR_RE.findall(body)]


_TITLE_PAIR_RE = re.compile(r"""^\s+title=(?:"([^"]+)"|'([^']+)'),$""", re.M)


def _unpublished_titles(dedicated: bool) -> list[str]:
    """The controls a panel draws that the model never asked for, read from the renderer."""
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        IMAGE_KNOB_TITLES,
        always_on_controls,
    )
    from open_webui_openrouter_pipe.integrations.image_types import SCHEMA_ONLY_PARAMS

    return [title for _n, _a, _d, title, _desc in always_on_controls(dedicated)] + [
        IMAGE_KNOB_TITLES[name][0] for name in SCHEMA_ONLY_PARAMS
    ]


def _picture_only_titles() -> list[str]:
    """The always-on controls production reserves for the picture-only transport.

    Read from the frozenset the renderer filters by, not from the difference between the
    two calls being compared: that difference is empty exactly when the filtering breaks,
    which is the case the branch below has to detect.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        _IMAGE_API_ONLY_CONTROLS,
        ALWAYS_ON_CONTROLS,
    )

    return [
        title
        for name, _a, _d, title, _desc in ALWAYS_ON_CONTROLS
        if name in _IMAGE_API_ONLY_CONTROLS
    ]


@pytest.mark.parametrize("dedicated", [True, False], ids=["image-api", "chat-route"])
def test_the_install_panels_valve_names_the_controls_no_model_publishes(dedicated):
    """The admin was told the panel offers "exactly" what the model publishes. It does not.

    Every panel draws Output size, which no recorded contract publishes, and a model that
    answers with a picture and nothing else draws three more the panel supplies itself.
    The description named none of them and claimed exclusivity over the published set, so
    an admin reading it could not account for a single control on the screen.

    The chat route's unbidden set is a SUBSET of the picture-only one, so "every drawn
    title is named in the description" is the weaker of the two rows and the chat row
    added nothing: a helper that ignored its transport argument altogether left both
    green. What separates them is asserted directly instead -- the picture-only controls
    are drawn on that transport and are NOT drawn on the chat route -- with the set of
    them read from production rather than counted here, because the property is WHICH
    controls each transport draws and not how many.

    A ban on the words "exactly", "only" and "nothing but" stood here and was removed. It
    was a substring test, so "commonly" failed it for containing "only"; and the words
    themselves are true of plenty of correct sentences. What survives is the derived half:
    every control the panel draws unbidden has to be NAMED, which is what an admin reading
    the description needs in order to account for the screen.
    """
    description = _install_panels_description()
    drawn_without_a_contract = _unpublished_titles(dedicated)
    assert drawn_without_a_contract, "the renderer draws nothing unbidden; this proves nothing"

    missing = [title for title in drawn_without_a_contract if title not in description]
    assert not missing, (
        f"the panel draws {drawn_without_a_contract} whatever the model publishes, and the "
        f"description never names {missing}: {description!r}"
    )

    picture_only = _picture_only_titles()
    assert picture_only, (
        "production reserves no always-on control for the picture-only transport, so "
        "neither row here can tell the two transports apart"
    )
    drawn = set(drawn_without_a_contract)
    if dedicated:
        withheld = sorted(set(picture_only) - drawn)
        assert not withheld, (
            f"the picture-only transport is missing {withheld}, which production reserves "
            f"for it; it draws {drawn_without_a_contract}"
        )
    else:
        leaked = sorted(set(picture_only) & drawn)
        assert not leaked, (
            f"the chat route draws {leaked}, which production reserves for the "
            f"picture-only transport; it draws {drawn_without_a_contract}"
        )


@pytest.mark.parametrize(
    ("publishes", "typed"),
    [("1K", "2K"), ("2K", "1K"), ("1K", "1K"), ("1K", "1024x1024")],
    ids=["tier-not-published", "the-other-way-round", "tier-published", "exact-pixels"],
)
def test_the_install_panels_valve_says_what_becomes_of_a_size_a_model_never_published(
    publishes, typed
):
    """The valve said a request can send Output size "whatever the model publishes".

    It cannot. A tier is measured against the model's own published tier list, and one
    that is not on it never leaves the pipe -- so an admin reading that sentence expects a
    2K request to reach a model publishing only 1K, and it does not. The Config tab has
    said so correctly all along, which left two admin surfaces giving opposite answers.

    Whether the value survives is taken from the adapter that decides it, not asserted
    here. Two of the rows type the same 1K and come out opposite ways -- dropped at a model
    publishing 2K, kept at a model publishing 1K -- so neither an always-drop nor an
    always-keep reading of the adapter satisfies both, and neither does a reading that
    answers on the typed string: what the model published is what decides. What the
    description must then carry is derived from that outcome.
    """
    from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter
    from open_webui_openrouter_pipe.integrations.image_types import SCHEMA_ENUMS

    tiers = SCHEMA_ENUMS["resolution"]
    assert publishes in tiers, f"{publishes} is not one of the tiers production knows"

    record = {"supported_parameters": {"resolution": {"type": "enum", "values": [publishes]}}}
    top_level, _provider, _notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"size": typed}}, allowed_passthrough=(), record=record
    )
    survived = typed in set(top_level.values())
    assert survived is (typed == publishes or typed not in tiers), (
        f"a model publishing {publishes} answered {top_level!r} to {typed!r}, so the "
        "adapter is dropping or keeping everything and the description cannot be checked "
        "against it"
    )

    description = _install_panels_description()
    if survived:
        assert typed in description or typed in tiers, (
            f"{typed!r} reaches the model and the description never shows an admin that "
            f"spelling: {description!r}"
        )
        return

    for tier in tiers:
        assert tier in description, (
            f"a tier this model never published was dropped, and the description names "
            f"neither {tier} nor the list an admin has to check against: {description!r}"
        )


@pytest.mark.parametrize("dedicated", [True, False], ids=["image-api", "chat-route"])
@pytest.mark.parametrize(("slug", "model_id"), CONTRACTS_WITH_KNOBS, ids=[s for s, _ in CONTRACTS_WITH_KNOBS])
def test_every_recorded_panel_draws_more_than_its_model_publishes(slug, model_id, dedicated):
    """Measured per contract, so the claim "panel equals contract" can never come back.

    The published half is titled the way the panel titles it -- a typed setting by its
    own name in words, a provider setting by the raw parameter -- and then asserted to be
    a STRICT subset of what the panel draws. A panel that ever drew exactly its contract
    fails on every recorded model at once, on either transport.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import image_knob_text

    records = _records(slug)
    spec = build_image_model_filter_spec(
        model_id, {"id": model_id, "name": model_id}, records, dedicated_image_api=dedicated
    )
    published = {
        *(image_knob_text(name, spec)[0] for name, _values in spec.enums),
        *(image_knob_text(name, spec)[0] for name, _low, _high in spec.ranges),
        *(image_knob_text(name, spec)[0] for name in spec.supported),
        *spec.passthrough,
    }
    drawn = set(_panel_titles(model_id, records, dedicated))
    unbidden = set(_unpublished_titles(dedicated))

    assert published, f"{model_id} publishes nothing, so a superset claim proves nothing"
    assert published < drawn, (
        f"{model_id} publishes {sorted(published)} and its panel draws {sorted(drawn)}; "
        "the panel must carry strictly more than the contract for the valve description "
        "to be readable at all"
    )
    assert drawn - published == unbidden, (
        f"{model_id} draws {sorted(drawn - published)} on top of its contract, and the "
        f"renderer's always-present set is {sorted(unbidden)}"
    )


# -------------------------------- A COST PROMISE THE READER CAN TURN OFF ---
# The first version of this guard searched for one phrase, "status line when it
# finishes". A document said "status footer ... includes the cost of the generation"
# instead and carried the same unconditional promise straight past it. What follows
# reads the claim rather than a phrasing: money beside the name of the surface beside
# any verb of showing. The three vocabularies below are anchored to production by the
# tests -- the surface word is the tail of the valve that governs it, the money word is
# the one the builder actually prints, and the setting is read from the setting.
_GATE_VALVE = "SHOW_FINAL_USAGE_STATUS"

_MONEY_SYMBOL = r"\$\d"
_MONEY_WORDS = (
    r"\bcosts?\b|\bcosting\b|\bpric\w*\b|\bcharges?d?\b|\bbill(?:s|ed|ing)?\b"
    r"|\bfees?\b"
)
_MONEY_SYMBOL_RE = re.compile(_MONEY_SYMBOL, re.I)
_MONEY_WORD_RE = re.compile(_MONEY_WORDS, re.I)
_MONEY_RE = re.compile(f"{_MONEY_SYMBOL}|{_MONEY_WORDS}", re.I)
_DISPLAY_RE = re.compile(
    r"\b(?:show\w*|display\w*|includ\w*|render\w*|appear\w*|print\w*|says?|said|tells?"
    r"|told|carr(?:y|ies|ied)|lists?|listed|lands?|landed|arriv\w*|report\w*|return\w*"
    r"|gives?|puts?|adds?|writes?|written|reads?|sits?|holds?|comes?|ends?)\b",
    re.I,
)
_CONDITIONAL_RE = re.compile(
    r"\b(?:where|wherever|whenever|if|unless|only|as long as|provided)\b", re.I
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_FENCE_RE = re.compile(r"^\s*```")


def _gate_names() -> tuple[str, str]:
    """The two names a sentence may use for the setting that suppresses the figure."""
    from open_webui_openrouter_pipe.core.config import UserValves, Valves

    assert _GATE_VALVE in Valves.model_fields and _GATE_VALVE in UserValves.model_fields, (
        f"{_GATE_VALVE} is not a valve on either model any more; the sweep below is "
        "pointing readers at a setting that does not exist"
    )
    title = UserValves.model_fields[_GATE_VALVE].title
    assert title, "the setting the cards point a reader at has no user-facing name"
    return title, _GATE_VALVE


def _status_surface_word() -> str:
    """What the pipe calls the surface, taken from the valve that governs it."""
    return _GATE_VALVE.rsplit("_", 1)[-1].lower()


def _states_a_condition(sentence: str) -> bool:
    """Whether a sentence qualifies its claim at all.

    Bare "when" is not in the vocabulary. Every shipped cost sentence ends "when it
    finishes", which dates the figure rather than conditioning it, so counting it made
    "Show usage details always puts the price on the status line when it finishes" --
    a flat promise stating the opposite of the truth -- read as conditional.
    """
    return bool(_CONDITIONAL_RE.search(sentence))


def _names_the_status_surface(sentence: str) -> bool:
    """Whether a sentence is talking about the surface the figure lands on.

    Two ways, both taken from production: the word the pipe uses for the surface, and the
    name of the setting that governs it. A valve description is read under its own field
    name and a table row carries that name in its first column, so neither has to repeat
    the word for the sweep to know what it is about. Reading the word alone is how
    "Display tokens, time, and cost at the end of each reply" -- the sentence a reader is
    shown in their own settings -- stayed invisible to every version of this sweep.
    """
    title, identifier = _gate_names()
    surface = re.compile(rf"\b{_status_surface_word()}\w*\b", re.I)
    gate = re.compile(rf"{re.escape(title)}|{re.escape(identifier)}", re.I)
    return bool(surface.search(sentence) or gate.search(sentence))


def _promises_a_cost_on_the_status_line(sentence: str) -> bool:
    return bool(
        _MONEY_RE.search(sentence)
        and _names_the_status_surface(sentence)
        and _DISPLAY_RE.search(sentence)
    )


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def _doc_sentences(path: Path) -> list[tuple[str, bool]]:
    """Every sentence in a document, each flagged for whether it is a table row.

    Wrapped lines are rejoined first: the sentence that escaped the old guard runs
    across four lines, and reading a document a line at a time splits claims in half.
    """
    found: list[tuple[str, bool]] = []
    block: list[str] = []
    fenced = False

    def flush() -> None:
        if block:
            joined = " ".join(part.strip() for part in block).strip()
            block.clear()
            found.extend((sentence, False) for sentence in _sentences(joined))

    for raw in path.read_text(encoding="utf-8").splitlines():
        if _FENCE_RE.match(raw):
            fenced = not fenced
            flush()
            continue
        if fenced:
            continue
        stripped = raw.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("|"):
            flush()
            found.append((stripped, True))
            continue
        if stripped.startswith(("#", "- ", "* ", "> ")):
            flush()
            found.append((stripped.lstrip("#-*> ").strip(), False))
            continue
        block.append(raw)
    flush()
    return found


def _rendered_cards() -> dict[str, str]:
    """Every help card a reader can be shown, rendered the way they are shown it."""
    from open_webui_openrouter_pipe.integrations.image_help import render_image_help
    from open_webui_openrouter_pipe.integrations.video_help import render_video_help

    cards: dict[str, str] = {}
    for slug, model_id in EVERY_CONTRACT:
        cards[f"image card {model_id}"] = render_image_help(
            model_id, {"id": model_id, "name": model_id},
            endpoint_record=_records(slug), dedicated_image_api=True,
        )
    catalog = json.loads((FIXTURES / "video_models_catalog.json").read_text())["data"]
    for model in catalog:
        cards[f"video card {model['id']}"] = render_video_help(model["id"], model)
    return cards


def _valve_descriptions() -> list[tuple[str, str]]:
    """(surface, one sentence as a reader meets it) for every description on either model.

    Derived by walking the models, not by naming files. A sweep that reads rendered cards
    and documents cannot see a valve description at all, and that is where the sentence an
    administrator reads while deciding whether to switch the figure on was sitting. Each
    sentence carries its field's name or title, because that is what a reader reads it
    under, and it is what says which setting the sentence is describing.
    """
    from open_webui_openrouter_pipe.core.config import UserValves, Valves

    found: list[tuple[str, str]] = []
    for model_name, model in (("Valves", Valves), ("UserValves", UserValves)):
        for field_name, field in model.model_fields.items():
            label = field.title or field_name
            for sentence in _sentences((field.description or "").strip()):
                found.append((f"{model_name}.{field_name}", f"{label}: {sentence}"))
    return found


def _cost_promises() -> list[tuple[str, str, bool]]:
    """(surface, sentence, whether the surface names the setting itself) per cost claim.

    The third field is true of a table row and of a valve description, both of which carry
    the setting's name beside the sentence rather than inside it. That spares them the
    checks about pointing a reader at the switch. It spares them nothing about whether the
    figure appears at all.
    """
    found: list[tuple[str, str, bool]] = []
    for name, card in _rendered_cards().items():
        for line in card.splitlines():
            for sentence in _sentences(line.strip().lstrip("#-*> ").strip()):
                if _promises_a_cost_on_the_status_line(sentence):
                    found.append((name, sentence, False))
    for path in sorted(DOCS.glob("*.md")):
        for sentence, is_row in _doc_sentences(path):
            if _promises_a_cost_on_the_status_line(sentence):
                found.append((path.name, sentence, is_row))
    for name, sentence in _valve_descriptions():
        if _promises_a_cost_on_the_status_line(sentence):
            found.append((name, sentence, True))
    return found


@pytest.mark.parametrize(
    ("show_usage", "usage"),
    [
        (True, {"cost": 0.0412, "input_tokens": 800, "output_tokens": 400}),
        (False, {"cost": 0.0412, "input_tokens": 800, "output_tokens": 400}),
        (True, {"input_tokens": 800, "output_tokens": 400}),
    ],
    ids=["shown", "switched-off", "no-cost-reported"],
)
def test_the_status_line_shows_a_cost_only_when_the_setting_and_the_payload_allow_it(
    pipe_instance, show_usage, usage
):
    """The two ways the promised figure never appears, driven through the real builder.

    Every image card closed by telling the reader the cost lands on the status line when
    the generation finishes. An administrator, or the reader in their own settings, can
    switch that line back to a bare elapsed time; and a provider that reports no cost
    leaves the figure out while the setting is on.

    Three rows, two outcomes, one payload shared by the first two -- a builder that always
    printed a cost, or never did, fails a row.
    """
    valves = pipe_instance.Valves(SHOW_FINAL_USAGE_STATUS=show_usage)

    description = pipe_instance._ensure_error_formatter()._format_final_status_description(
        elapsed=7.3, total_usage=dict(usage), valves=valves, stream_duration=None
    )

    expected = show_usage and "cost" in usage
    assert ("$" in description) is expected, (
        f"show_usage={show_usage} cost={usage.get('cost')} produced {description!r}"
    )


@pytest.mark.parametrize(
    ("sentence", "promises"),
    [
        (
            "The status footer rendered on the assistant message includes the cost of "
            "the generation in dollars.",
            True,
        ),
        (
            "Where the company running the model reports a cost, it is shown on the "
            "status line when it finishes.",
            True,
        ),
        ("The price of a render lands on the final status line.", True),
        ("Each font you supply: $0.03 per image", False),
        ("Duration is any integer 1-15 seconds and cost scales linearly per second.", False),
        ("The status line names the model the request is waiting on.", False),
    ],
    ids=[
        "the-footer-wording-that-escaped",
        "the-line-wording-the-old-guard-keyed-on",
        "a-third-wording-neither-guard-was-written-for",
        "a-published-rate-naming-no-surface",
        "money-that-never-reaches-the-surface",
        "the-surface-carrying-no-money",
    ],
)
def test_the_cost_promise_detector_reads_the_claim_and_not_one_phrasing(sentence, promises):
    """Three ways to make the same promise, three ways to mention money and not make it.

    The guard this replaced searched for the literal "status line when it finishes", so
    the first row here -- the wording that was live in a document at the time -- was
    invisible to it while saying exactly the same thing, minus the two conditions.

    Six rows split three-three, so a detector that answers the same way every time fails
    half of them whichever way it answers.
    """
    assert _promises_a_cost_on_the_status_line(sentence) is promises, (
        f"detector said {(not promises)!r} for {sentence!r}"
    )


def test_the_detector_hunts_the_words_production_actually_uses(pipe_instance):
    """The vocabularies are anchored to the pipe, not to what this file guesses.

    The money vocabulary has two halves and only one of them scans prose. The symbol
    half matches the figure itself, which the builder always prints, so asserting the
    union against the builder was satisfied by the dollar sign alone: relabelling the
    segment "Spend $" left this green while the WORD half -- the only half a sentence in
    a document can ever match, since prose never contains "$0.5" -- went stale unnoticed.
    Each half is therefore asserted separately.

    The text style is asked for by name rather than taken from the default, because the
    label is what the word half must track whichever style ships as the default.
    """
    valves = pipe_instance.Valves(
        SHOW_FINAL_USAGE_STATUS=True, FINAL_USAGE_STATUS_STYLE="text"
    )
    printed = pipe_instance._ensure_error_formatter()._format_final_status_description(
        elapsed=1.0, total_usage={"cost": 0.5}, valves=valves, stream_duration=None
    )

    assert _MONEY_WORD_RE.search(printed), (
        f"the status builder labels the figure in {printed!r}, and the sweep's money "
        "WORDS match none of that label; prose carries the label and never the figure, "
        "so every promise in a document would read as talking about nothing"
    )
    assert _MONEY_SYMBOL_RE.search(printed), (
        f"the status builder prints {printed!r} with no figure the symbol half can find"
    )

    surface = _status_surface_word()
    speaking = [name for name, sentence in _valve_descriptions() if surface in sentence.lower()]
    assert speaking, (
        f"no valve description calls the surface {surface!r}, which is the word taken "
        f"from {_GATE_VALVE}; the sweep is looking for a name nothing uses"
    )


@pytest.mark.parametrize(
    ("sentence", "conditional"),
    [
        (
            "Show usage details always puts the price on the status line when it "
            "finishes.",
            False,
        ),
        (
            "The cost of each generation is reported on the status line when it "
            "finishes.",
            False,
        ),
        (
            "Where the company running the model reports a charge above zero, it is "
            "shown on the status line when it finishes, as long as usage details are "
            "on.",
            True,
        ),
        (
            "The status line shows a charge only where the provider reported one.",
            True,
        ),
    ],
    ids=[
        "always-plus-a-temporal-when",
        "the-flat-promise-that-shipped-once",
        "the-wording-that-ships-now",
        "a-different-conditional-vocabulary",
    ],
)
def test_a_temporal_when_does_not_count_as_a_condition(sentence, conditional):
    """"when it finishes" says WHEN the figure lands, never WHETHER it lands.

    Both false rows end in that clause and neither qualifies the promise: the first
    states the opposite of the truth, and the second is the flat wording the sweep
    below was written to catch. Four rows split two-two, so a predicate answering the
    same way every time fails half of them whichever way it answers.
    """
    assert _states_a_condition(sentence) is conditional, (
        f"predicate said {(not conditional)!r} for {sentence!r}"
    )


def test_no_surface_promises_a_cost_on_the_status_line_without_naming_the_setting():
    """Every claim that a cost is shown, wherever worded, carries what suppresses it.

    Two conditions govern the figure and both are invisible to a reader who is only told
    it appears: the setting can be off, and the provider may report no cost at all. The
    sweep runs over rendered help cards, every document, and every valve description on
    both models, sentence by sentence with wrapped lines rejoined, because the claim that
    escaped the previous guard was a four-line sentence in a document and the one that
    escaped the guard after it was a valve description no corpus of files could reach.

    A table row and a valve description both carry the setting's name beside the sentence
    rather than inside it, so only prose is asked for the conditional as well.
    """
    title, identifier = _gate_names()
    promises = _cost_promises()

    surfaces = {name for name, _sentence, _named in promises}
    assert len({name for name in surfaces if name.endswith(".md")}) > 1, (
        f"only {sorted(surfaces)} carry the promise; the document half of the sweep is hollow"
    )
    assert any(name.startswith(("Valves.", "UserValves.")) for name in surfaces), (
        f"no valve description promises a cost ({sorted(surfaces)}); the half of the sweep "
        "that reads what an administrator is shown while deciding is hollow"
    )

    for name, sentence, names_itself in promises:
        assert title in sentence or identifier in sentence, (
            f"{name} promises a cost on the status line without naming {title!r} (or "
            f"{identifier}), which a reader can switch off: {sentence!r}"
        )
        if not names_itself:
            assert _states_a_condition(sentence), (
                f"{name} states the promise flatly, so a reader is never told the figure "
                f"is conditional at all: {sentence!r}"
            )


# ------------------------ ONE NAME FOR THE SURFACE THE FIGURE LANDS ON ---
# The sweep above recognises a promise by the NAME of the surface it is made about, so a
# second name for that surface is a way round it rather than a matter of taste. It is how
# the last untrue promise got out: a document said "status footer" while the guard of the
# day searched for "status line". Widening that guard to the surface word alone closed the
# one wording and left the shape of the hole open -- a name with no surface word in it is
# still invisible, and the readiness report was carrying one. That last shape is what the
# sweep below still refuses; the wider bans that once stood beside it are accounted for in
# its docstring.


def _sanctioned_surface_noun() -> str:
    """The noun production gives this surface, taken from the setting that styles it.

    That setting exists for the appearance of this one surface and names it while saying
    so, which makes its description the place the product states what the thing is called.
    The answer then has to turn up under a DIFFERENT setting as well, so the name rests on
    two independent production strings and one rewording cannot move it alone. It used to
    rest on a rendered help card for its second surface; help quotes no money now, so no
    card mentions this surface at all.
    """
    from open_webui_openrouter_pipe.core.config import Valves

    styled_by = "FINAL_USAGE_STATUS_STYLE"
    surface = _status_surface_word()
    styled = Valves.model_fields[styled_by].description or ""
    found = re.search(rf"\b{surface}\s+([a-z]+)\b", styled, re.I)
    assert found, (
        "the setting that styles this surface no longer names it, so there is no "
        f"production answer to hold a reader's documents to: {styled!r}"
    )
    noun = found.group(1).lower()
    spoken = [
        name
        for name, sentence in _valve_descriptions()
        if not name.endswith(f".{styled_by}")
        and re.search(rf"\b{surface}\s+{noun}s?\b", sentence, re.I)
    ]
    assert spoken, (
        f"only {styled_by} calls it a {surface} {noun!r}, so the name rests on one string "
        "and rewording that one string moves what every document is held to"
    )
    return noun


def test_nothing_a_reader_meets_names_this_surface_without_the_word_the_sweep_keys_on():
    """A name for this surface with no surface word in it is invisible to the sweep above.

    One shape is refused, and both halves of it are read off production rather than typed
    here: "final usage" followed by anything that is neither the surface word nor the noun
    production gives it. That is what "final usage banners" was -- a name for the closing
    figure's home with no "status" anywhere in it, so no wording of the sentence around it
    could have brought it to the cost sweep's attention.

    Two wider bans stood here and were removed: the surface word followed by another word
    for a strip of text ("status footer", "status banner"), and the surface word followed
    by a word for a mid-reply notice under a "final"/"usage" qualifier ("final status
    message"). Both were lists of nouns somebody chose -- twenty of them between the two --
    and BOTH shapes contain the surface word, so the cost sweep above already reads every
    sentence carrying them. They bought house style, not coverage, and they refused true
    sentences: "a status bar in the browser chrome" was a failure.

    Text is flattened before it is read, because a hard-wrapped document splits the phrase
    over two lines and a line-at-a-time scan sees neither half. The corpus is counted
    before it is judged -- a guard filtering on wording nobody uses any more passes by
    finding nothing, and the two floors are what stop that.
    """
    surface = _status_surface_word()
    noun = _sanctioned_surface_noun()

    no_surface_word = re.compile(
        rf"\bfinal\s+usage\s+(?!(?:{surface}|{noun}s?)\b)[a-z]+\b", re.I
    )
    sanctioned = re.compile(rf"\b{surface}\s+{noun}s?\b", re.I)

    corpus: list[tuple[str, str]] = [
        (path.name, path.read_text(encoding="utf-8"))
        for path in sorted(DOCS.rglob("*.md"))
    ]
    corpus += sorted(_rendered_cards().items())
    corpus += _valve_descriptions()

    speaking: set[str] = set()
    uses = 0
    offending: list[str] = []
    for name, text in corpus:
        flat = " ".join(text.split())
        if name.endswith(".md"):
            counted = len(sanctioned.findall(flat))
            uses += counted
            if counted:
                speaking.add(name)
        for hit in no_surface_word.finditer(flat):
            start = max(0, hit.start() - 70)
            offending.append(f"{name}: ...{flat[start : hit.end() + 70]}...")

    assert uses >= 10, (
        f"only {uses} mentions of a {surface} {noun!r} across every document; they have "
        "stopped using the product's name for this surface, so this guard is holding them "
        "to a phrase they no longer contain and is asserting almost nothing"
    )
    assert len(speaking) >= 4, (
        f"only {sorted(speaking)} name the surface at all; the sweep has narrowed to a "
        "corner of the documentation and a second name elsewhere would go unread"
    )
    assert not offending, (
        f"production calls this surface a {surface} {noun!r} -- in the setting that styles "
        "it, and in the cards a reader is shown. These name it without the word "
        f"{surface!r} in it at all, so the sweep that checks what is promised about this "
        "surface cannot see the sentence they are in:\n  " + "\n  ".join(offending)
    )


# ------------------- NO TIP POINTS AT A SECTION THAT MAY NOT BE THERE --------
_HEADING_RE = re.compile(r"^#{2,}\s+(.*\S)\s*$", re.M)


def _image_cards_both_ways() -> dict[tuple[str, bool], str]:
    """Every image card, rendered with a contract and without one.

    A card built with nothing published stops after the description and the tips, so the
    sections below them are not there to be pointed at.
    """
    from open_webui_openrouter_pipe.integrations.image_help import (
        _IMAGE_PER_MODEL_HELP_DATA,
        render_image_help,
    )

    recorded = dict((model_id, slug) for slug, model_id in EVERY_CONTRACT)
    cards: dict[tuple[str, bool], str] = {}
    for model_id in _IMAGE_PER_MODEL_HELP_DATA:
        model = {"id": model_id, "name": model_id}
        cards[(model_id, False)] = render_image_help(
            model_id, model, endpoint_record=None, dedicated_image_api=True
        )
        slug = recorded.get(model_id)
        if slug:
            cards[(model_id, True)] = render_image_help(
                model_id, model, endpoint_record=_records(slug), dedicated_image_api=True
            )
    return cards


def test_no_tip_sends_a_reader_to_a_section_the_card_did_not_render():
    """A tip closed by naming a section that is only there when a contract was read.

    With nothing published the card stops after the tips, so "under Controls below" ran
    off the end of the page -- and the model most likely to be short of a contract is the
    one whose tip is trying to explain what it accepts.

    The sections are read out of each rendering and the tips out of the data, so nothing
    here is typed: rename a heading and the vocabulary follows it. A heading name that is
    the first half of a hyphenated word is a compound adjective and not a pointer, which
    is what keeps "Cost-efficient variant" out of it.
    """
    from open_webui_openrouter_pipe.integrations.image_help import _IMAGE_PER_MODEL_HELP_DATA

    cards = _image_cards_both_ways()
    assert cards, "no card rendered at all"

    per_card = {key: set(_HEADING_RE.findall(card)) for key, card in cards.items()}
    vocabulary = set().union(*per_card.values())
    assert len(vocabulary) > 1, f"only {vocabulary} was ever rendered; nothing to point at"
    thin = [key for key, headings in per_card.items() if vocabulary - headings]
    assert thin, (
        "every card carries every section, so a tip could not point at a missing one and "
        "this guard proves nothing"
    )

    for (model_id, _with_contract), card in cards.items():
        entry = _IMAGE_PER_MODEL_HELP_DATA[model_id]
        written = [entry.get("best_known_for", ""), *(entry.get("tips_and_pitfalls") or [])]
        absent = vocabulary - per_card[(model_id, _with_contract)]
        for text in written:
            if not text or text not in card:
                continue
            pointed = [
                heading
                for heading in absent
                if re.search(rf"\b{re.escape(heading)}\b(?!-)", text)
            ]
            assert not pointed, (
                f"{model_id} tells the reader about {pointed}, and this rendering of its "
                f"card has only {sorted(per_card[(model_id, _with_contract)])}: {text!r}"
            )


# ------------------- ONE COMPANY CANNOT DISAGREE WITH ITSELF -----------------
def _unrenderable_descriptor() -> dict:
    """A published range no control can be drawn from, built from production's own rule.

    The renderer keeps a range only where its high bound is above its low one, so a bound
    read back off `SCHEMA_RANGES` and used for both ends is published and undrawable --
    without this file deciding for itself what "undrawable" means.
    """
    from open_webui_openrouter_pipe.integrations.image_types import SCHEMA_RANGES

    low, _high = SCHEMA_RANGES["output_compression"]
    return {"type": "range", "min": low, "max": low}


def _disagreement_contracts() -> dict[str, list[dict]]:
    from open_webui_openrouter_pipe.integrations.image_types import SCHEMA_ENUMS

    quality, background = SCHEMA_ENUMS["quality"], SCHEMA_ENUMS["background"]
    stuck = {"output_compression": _unrenderable_descriptor()}
    return {
        "one-company": [{"provider_slug": "solo", "supported_parameters": dict(stuck)}],
        "two-companies-agreeing": [
            {"provider_slug": "one", "supported_parameters": dict(stuck)},
            {"provider_slug": "two", "supported_parameters": dict(stuck)},
        ],
        "two-companies-disagreeing": [
            {
                "provider_slug": "one",
                "supported_parameters": {
                    "quality": {"type": "enum", "values": list(quality)}
                },
            },
            {
                "provider_slug": "two",
                "supported_parameters": {
                    "background": {"type": "enum", "values": list(background)}
                },
            },
        ],
    }


def test_the_card_blames_the_companies_only_where_they_actually_differ():
    """"Did anyone publish" is not "did they differ", and the card said the second.

    A lone company publishing one undrawable descriptor, and two publishing that same one,
    both produced the sentence that blames the companies for disagreeing -- naming a
    disagreement that cannot exist with one of them and does not exist with two that
    match. Only the third contract here is a real disagreement.

    All three end with no controls read from the contract, so the count cannot be what
    separates them, and the sentences are read out of the rendered card rather than off
    the flag. The sentence a real disagreement must carry is imported from production
    rather than reduced to the word "different", so rewording it moves the expectation
    instead of reddening the suite.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import DISAGREED_SETTINGS

    blame = DISAGREED_SETTINGS.format(named="this model")
    contracts = _disagreement_contracts()
    said: dict[str, str] = {}
    for label, records in contracts.items():
        spec = build_image_model_filter_spec(
            "vendor/probe", {"id": "vendor/probe"}, records, dedicated_image_api=True
        )
        assert spec.knob_count == 0, (
            f"{label} draws {spec.knob_count} controls from its contract, so it never "
            "reaches the sentence under test"
        )
        assert spec.published_any_parameter, (
            f"{label} published nothing at all, which is a different sentence again"
        )
        card = _help_text("vendor/probe", records, True)
        said[label] = card.split("## Controls", 1)[1].splitlines()[1]

    assert said["one-company"] == said["two-companies-agreeing"], (
        "one company and two that publish the same thing are both told the companies "
        f"differ, or are told two different things: {said}"
    )
    assert said["two-companies-disagreeing"] != said["one-company"], (
        "companies that genuinely publish different things get the same sentence as a "
        f"single company, so the reader is never told to expect it to settle: {said}"
    )
    for label in ("one-company", "two-companies-agreeing"):
        assert blame not in said[label], (
            f"{label} is told the companies serving it differ: {said[label]!r}"
        )
    assert blame in said["two-companies-disagreeing"], (
        f"a real disagreement is not named as one: {said['two-companies-disagreeing']!r}"
    )


# ------------------- BOTH SWITCHES, AND A CHARGE OF NOTHING ------------------
_REPORTED_COUNTS: dict[str, int] = {"input_tokens": 800, "output_tokens": 400}


def _status_with(
    pipe_instance,
    *,
    admin: bool,
    user: bool | None,
    cost: float | None,
    counters: dict[str, int] | None = None,
) -> str:
    """The status line the pipe builds, driven through the real user/admin valve merge.

    ``user=None`` is a reader who has never opened their own settings, which is the case
    the single-gate sentence was written for and the one it gets wrong: nothing of theirs
    is set, so the administrator's default is what decides.

    ``counters`` is what the reply reported. It defaults to a reply that reported some,
    which is what every caller below this one drives; ``{}`` is a reply that reported
    none, which is the shape a generation billed by the picture comes back in.
    """
    from open_webui_openrouter_pipe.core.config import UserValves

    user_valves = UserValves() if user is None else UserValves(SHOW_FINAL_USAGE_STATUS=user)
    merged = pipe_instance._merge_valves(
        pipe_instance.Valves(SHOW_FINAL_USAGE_STATUS=admin), user_valves
    )
    usage: dict[str, Any] = dict(_REPORTED_COUNTS if counters is None else counters)
    if cost is not None:
        usage["cost"] = cost
    return pipe_instance._ensure_error_formatter()._format_final_status_description(
        elapsed=7.3, total_usage=usage, valves=merged, stream_duration=None
    )


# Two regexes stood here and were removed: one demanded that a cost sentence spell the
# administrator's default as "administrator" or "site default", the other that it spell
# the zero case as "above zero", "over zero", "more than zero", "greater than zero" or
# "non-zero". Neither expectation was computed by anything; each mandated a handful of
# spellings for one idea, so "your organisation's default" and "shown only when it is not
# zero" -- both true, both clear -- were failures. Neither had a test of its own. What
# survives in both tests below is the behaviour they were pretending to police, driven
# through the production merge and the production builder, plus the derived obligation
# that a promise names the setting production actually calls it by.


@pytest.mark.parametrize("chosen", [True, False], ids=["user-says-on", "user-says-off"])
def test_every_cost_promise_names_the_switch_that_decides_when_the_reader_has_not_chosen(
    pipe_instance, chosen
):
    """Two valves carry this name, and a reader who has never chosen is governed by neither of the ones we named.

    Only fields the reader actually set override the administrator's, so someone who has
    never opened their own settings sees their own toggle reading on while the site
    default decides. They check the switch every card names, find it on, and have nothing
    left to look at. The administrator's copy has no title of its own, so it can only be
    described.

    What is asserted is the merge itself: the administrator's copy off and the reader's
    unset must show no figure, while the reader's explicit choice must be obeyed either
    way. Both rows of the parameter drive that choice, so a merge answering the same way
    every time fails one of them. The prose obligation that survives is the derived one --
    every promise names the setting by the title production gives it.

    The obligation that a sentence ALSO spell out the administrator's default was removed:
    it required one of two literal wordings, and there is no way to compute from production
    that a given sentence discloses that fact.
    """
    title, identifier = _gate_names()
    from open_webui_openrouter_pipe.core.config import Valves

    assert Valves.model_fields[identifier].title is None, (
        "the administrator's copy now has a name of its own, so the cards should point at "
        "it by that name rather than describing it"
    )

    unset = _status_with(pipe_instance, admin=False, user=None, cost=0.0412)
    explicit = _status_with(pipe_instance, admin=False, user=chosen, cost=0.0412)
    assert ("$" in explicit) is chosen, (
        f"a reader who set the switch to {chosen} is not obeyed: {explicit!r}"
    )
    assert "$" not in unset, (
        "with the administrator's copy off and nothing of the reader's set, a figure "
        f"appeared anyway, so there is only one gate after all: {unset!r}"
    )

    promises = _cost_promises()
    assert promises, "the sweep found no promise at all; it is hollow"
    for name, sentence, names_itself in promises:
        if names_itself:
            continue
        assert title in sentence or identifier in sentence, (
            f"{name} promises a cost without naming {title!r}: {sentence!r}"
        )


@pytest.mark.parametrize("cost", [0, 0.0412], ids=["reported-zero", "reported-a-charge"])
def test_every_cost_promise_says_a_charge_of_nothing_is_not_shown(pipe_instance, cost):
    """A generation reported at zero reports a cost, and the line stays empty.

    Free listings, zero-rated routes and credited generations all come back with a cost of
    zero, and the builder prints a figure only above zero. "Where a cost is reported" is
    therefore false for exactly those, and the reader is left checking a switch that was
    never the reason.

    The two rows share every other input, so a builder that answered the same way to both
    fails one of them: this is where the pipe's own zero-suppression is pinned, and it is
    pinned nowhere else -- the sibling test above drives a cost present against a cost
    absent, never a cost of zero.

    The obligation that every cost sentence ALSO spell the zero case as "above zero" (or
    one of four near-synonyms) was removed. It was a list of spellings for one idea, so
    "a charge of nothing is not shown" and "never shown for a free generation" failed it
    while saying exactly the right thing. Nothing computes, from production, whether a
    given English sentence discloses the zero case; the floor below is what keeps the
    corpus honest.
    """
    printed = _status_with(pipe_instance, admin=True, user=True, cost=cost)
    shows_a_figure = "$" in printed
    assert shows_a_figure is (cost > 0), (
        f"a reported cost of {cost} produced {printed!r}, which is not what the sentences "
        "under test are being measured against"
    )
    if shows_a_figure:
        return

    promises = _cost_promises()
    assert promises, "the sweep found no promise at all; it is hollow"


# ------------------- A CARD WITH NOTHING PUBLISHED STILL GETS A PANEL --------
@pytest.mark.parametrize(
    "contract",
    ["empty", "recorded"],
    ids=["publishes-nothing", "publishes-knobs"],
)
@pytest.mark.asyncio
async def test_a_contract_that_was_read_gets_a_panel_whether_or_not_it_names_a_knob(
    contract,
):
    """Help closes the knobless card with four controls; something must draw them.

    The installer used to weigh the knob count, which leaves out the size control and the
    three the panel supplies itself. A model whose contract reads clean but names no knob
    scored zero, got no panel, and scored zero again on every refresh after -- so the four
    controls its card listed existed nowhere, permanently.

    Two arms: a contract that was read and publishes nothing, and one recorded from a live
    model that publishes plenty. Both were read, so both draw a panel, and an installer
    that answers on the knob count fails the first while passing the second.
    """
    from unittest.mock import AsyncMock, MagicMock

    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    slug, model_id = EVERY_CONTRACT[0]
    records = [{"provider_slug": "solo", "supported_parameters": {}}]
    if contract == "recorded":
        records = _records(slug)

    spec = build_image_model_filter_spec(
        "vendor/probe", {"id": "vendor/probe"}, records, dedicated_image_api=True
    )
    assert spec.contract_read, "precondition: both arms are contracts that WERE read"
    assert (spec.knob_count == 0) is (contract == "empty"), (
        f"the {contract} arm no longer drives the branch it was written for: "
        f"{spec.knob_count}"
    )

    pipe = MagicMock()
    manager = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    written: list[str] = []
    manager._ensure_filter_installed = AsyncMock(
        side_effect=lambda **kw: (written.append(kw["desired_source"]), kw["preferred_id"])[1]
    )

    function_id = await manager._ensure_single_image_filter_function_id(
        model_id="vendor/probe",
        image_model={"id": "vendor/probe"},
        endpoint_record=records,
        dedicated_image_api=True,
    )

    assert function_id, (
        f"a {contract} contract that was read leaves the model with no panel at all, "
        "while its card lists the controls one would carry"
    )
    listed = _help_controls("vendor/probe", records, True)
    body = written[0].split("class UserValves(BaseModel):", 1)[1].split("    def __init__", 1)[0]
    drawn = [d or q for d, q in _TITLE_PAIR_RE.findall(body)]
    assert drawn, "the installed panel draws no control, so this proves nothing"
    missing = [title for title in listed if title not in drawn]
    assert not missing, (
        f"help lists {listed} for this model and the panel it gets draws {drawn}, so "
        f"{missing} is promised and unreachable"
    )


# ------------------- A CARD WITH NOTHING PUBLISHED STILL HAS CONTROLS ON IT ---
def _help_controls(model_id: str, records: list[dict], dedicated: bool) -> list[str]:
    from open_webui_openrouter_pipe.integrations.image_help import render_image_help

    rendered = render_image_help(
        model_id, {"id": model_id, "name": model_id},
        endpoint_record=records, dedicated_image_api=dedicated,
    )
    assert "## Controls" in rendered, rendered
    listed = rendered.split("## Controls", 1)[1]
    return [
        line.split("**")[1]
        for line in listed.splitlines()
        if line.startswith("- **") and "**" in line[4:]
    ]


def _help_text(model_id: str, records: list[dict], dedicated: bool) -> str:
    from open_webui_openrouter_pipe.integrations.image_help import render_image_help

    return render_image_help(
        model_id, {"id": model_id, "name": model_id},
        endpoint_record=records, dedicated_image_api=dedicated,
    )


@pytest.mark.parametrize("dedicated", [True, False], ids=["image-api", "chat-route"])
@pytest.mark.parametrize("label", sorted(_NOTE_CONTRACTS), ids=sorted(_NOTE_CONTRACTS))
def test_help_names_the_controls_the_panel_draws_even_with_nothing_published(label, dedicated):
    """The empty branches told the reader the model had no settings and stopped there.

    The count they were gated on leaves out the size control and the always-on ones, all
    of which the panel still draws -- so on a contract that shrank to nothing, help said
    "no adjustable settings" above a panel carrying four. The expectation is read off the
    rendered panel, so it moves with the transport: four controls on a model answering
    with pictures alone, one on a model answering in chat. No fixed list satisfies both.
    """
    records = _NOTE_CONTRACTS[label]
    spec = build_image_model_filter_spec(
        "vendor/probe", {"id": "vendor/probe"}, records, dedicated_image_api=dedicated
    )
    assert (spec.knob_count == 0) is (label != "publishes-values"), (
        f"{label} no longer drives the branch it was written for: {spec.knob_count}"
    )

    body = render_image_model_filter_source(spec).split("class UserValves(BaseModel):", 1)[1]
    drawn = [d or s for d, s in _TITLE_PAIR_RE.findall(body.split("    def __init__", 1)[0])]
    assert drawn, f"{label}/{dedicated} draws no control, so this proves nothing"

    named = _help_controls("vendor/probe", records, dedicated)
    unnamed = [title for title in drawn if title not in named]
    assert not unnamed, (
        f"the {label} panel on {'the image API' if dedicated else 'the chat route'} draws "
        f"{drawn} and help never names {unnamed}; help listed {named}"
    )


def test_the_six_empty_and_full_cards_each_say_something_different():
    """Three contracts, two transports, six texts -- a constant answer collapses them.

    Waiting for providers to agree and accepting that a model publishes nothing call for
    different remedies from the reader, and the two transports draw different panels, so
    no two of the six may read alike.
    """
    texts = {
        (label, dedicated): _help_text("vendor/probe", records, dedicated)
        for label, records in _NOTE_CONTRACTS.items()
        for dedicated in (True, False)
    }

    assert len(set(texts.values())) == len(texts), (
        "two of these six cards read identically, so the card is not reading its own "
        f"contract or its own transport: { {k: v[-200:] for k, v in texts.items()} }"
    )


# ------------------- A WORKED EXAMPLE IS THE STRING THAT IS BUILT ------------
def _worked_example(document: str, anchor: str, filled: dict[str, str]) -> list[str]:
    """The fenced block a document introduces with *anchor*, its placeholders filled in.

    Read out of the page rather than pasted here, so the comparison below is against what
    a reader is actually shown.
    """
    lines = (DOCS / document).read_text(encoding="utf-8").splitlines()
    anchored = [index for index, line in enumerate(lines) if anchor in line]
    assert len(anchored) == 1, (
        f"{anchor!r} introduces {len(anchored)} passages in {document}, so the example "
        "this test measures cannot be identified"
    )
    rest = lines[anchored[0] + 1 :]
    opened = next(index for index, line in enumerate(rest) if _FENCE_RE.match(line))
    closed = next(
        index for index, line in enumerate(rest[opened + 1 :]) if _FENCE_RE.match(line)
    )
    block = rest[opened + 1 : opened + 1 + closed]
    assert block, f"the example under {anchor!r} in {document} is empty"
    for token, value in filled.items():
        assert any(token in line for line in block), (
            f"the example under {anchor!r} in {document} never shows {token}, so what the "
            "reader is shown cannot be lined up against a built message"
        )
        block = [line.replace(token, value) for line in block]
    return block


@pytest.mark.parametrize(
    ("job_id", "model_id", "file_id", "elapsed", "cost"),
    [
        ("job-abc", "vendor/model-1", "file-xyz", 12.5, 0.42),
        ("job-42", "other/model-9", "file-77", 3.0, 0.0),
    ],
    ids=["one-job", "another-job-billed-nothing"],
)
def test_the_video_page_prints_the_message_the_pipe_actually_builds(
    job_id, model_id, file_id, elapsed, cost
):
    """The page printed a closing line of elapsed time and money that is built nowhere.

    A reader was shown a finished message ending "Generated in ... $...", so someone
    looking for what a clip cost was sent to a line that has never existed in any release
    and under any setting -- while the figure that does exist lives on the status line and
    goes away when usage details are off. Nothing noticed, because the page was checked
    against the shape of the message and never against the message.

    The two rows carry different job, model and file ids, so a block with values typed
    into it rather than placeholders matches at most one of them; and they carry a
    different elapsed time and a different charge, one of them nothing, so a block naming
    either figure cannot match both.
    """
    from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter

    adapter = VideoGenerationAdapter.__new__(VideoGenerationAdapter)
    built = adapter._build_success_content(
        job_id=job_id,
        model_id=model_id,
        file_ids=[file_id],
        elapsed=elapsed,
        usage={"cost": cost, "total_tokens": 0},
    )

    shown = _worked_example(
        "openrouter_video_generation.md",
        "the assistant message contains",
        {"<job_id>": job_id, "<model_id>": model_id, "<owui_file_id>": file_id},
    )

    assert shown == built.splitlines(), (
        "the video page shows a finished message the pipe does not build; the page has\n"
        f"  {shown}\nand the message is\n  {built.splitlines()}"
    )


# ------------------- A GENERATION BILLED AT NOTHING, EACH WAY IT LANDS -------
_FIGURE_RES = {
    "tokens": re.compile(r"\btokens?\b", re.I),
    "timing": re.compile(r"\btiming\b|\btimes?\b|\belapsed\b|\bseconds?\b", re.I),
    "money": re.compile(r"\$|\bcosts?\b|\bcharges?\b|\bmoney\b", re.I),
}
_RESOLUTION_RES = {True: re.compile(r"\bTrue\b"), False: re.compile(r"\bFalse\b")}


def _figures_in(text: str) -> frozenset[str]:
    """Which of the three kinds of figure a piece of text carries, or claims to."""
    return frozenset(kind for kind, pattern in _FIGURE_RES.items() if pattern.search(text))


def _merged_gate(pipe_instance, *, admin: bool, user: bool | None) -> bool:
    """What the two copies of the switch resolve to, decided by the production merge."""
    from open_webui_openrouter_pipe.core.config import UserValves

    user_valves = UserValves() if user is None else UserValves(SHOW_FINAL_USAGE_STATUS=user)
    merged = pipe_instance._merge_valves(
        pipe_instance.Valves(SHOW_FINAL_USAGE_STATUS=admin), user_valves
    )
    return bool(merged.SHOW_FINAL_USAGE_STATUS)


def _zero_cost_sentence() -> str:
    """The telemetry page's account of what a generation billed at nothing prints."""
    document = "openrouter_integrations_and_telemetry.md"
    found = [
        sentence
        for sentence, _row in _doc_sentences(DOCS / document)
        if "exactly zero" in sentence
    ]
    assert len(found) == 1, (
        f"{document} carries {len(found)} sentences about a generation billed at exactly "
        "zero, and this test reads one"
    )
    return found[0]


def _figures_claimed_per_side(sentence: str) -> dict[bool, frozenset[str]]:
    """What a sentence says is printed on each side of the resolved switch.

    Each clause is read under the resolution it names, so a sentence has to say what
    happens both ways to be read at all. The sentence this replaced named neither, and
    attached one outcome to all four ways the two copies can be set.
    """
    claimed: dict[bool, frozenset[str]] = {}
    for clause in re.split(r",\s+and\s+|;\s+", sentence):
        for side, pattern in _RESOLUTION_RES.items():
            if pattern.search(clause):
                claimed[side] = _figures_in(clause)
    return claimed


@pytest.mark.parametrize("cost", [0, 0.0412], ids=["reported-zero", "reported-a-charge"])
def test_the_telemetry_page_says_what_a_generation_billed_at_nothing_prints_each_way(
    pipe_instance, cost
):
    """The page quantified over four ways the switches can be set and was wrong in two.

    It said a generation reported at exactly zero prints tokens and timing "whichever way
    both copies are set". Where the merge resolves False the builder returns a bare
    elapsed time before it ever reads a token counter, so half of those readers are
    promised counts they will not get -- and the merged value is the whole of the
    difference, which is what makes the sentence worth having at all.

    Nothing here is typed: the four combinations are put through the production merge and
    the production builder, each resolution's outcome is read off what was printed, and
    the page is held to that. The charged row makes the measurement discriminate -- money
    joins the line there and not in the zero row -- so the same claim cannot satisfy both.
    """
    printed: dict[bool, set[frozenset[str]]] = {True: set(), False: set()}
    for admin in (True, False):
        for user in (True, False):
            side = _merged_gate(pipe_instance, admin=admin, user=user)
            printed[side].add(
                _figures_in(_status_with(pipe_instance, admin=admin, user=user, cost=cost))
            )

    assert all(printed.values()), (
        f"the four ways the two copies can be set reach only {sorted(k for k, v in printed.items() if v)} "
        "of the two resolutions, so one side of the page's sentence is never measured"
    )
    assert all(len(shapes) == 1 for shapes in printed.values()), (
        f"one resolution printed more than one shape of line, so there is no single "
        f"outcome for the page to state: {printed}"
    )
    measured = {side: next(iter(shapes)) for side, shapes in printed.items()}

    assert ("money" in measured[True]) is (cost > 0), (
        f"a reported cost of {cost} produced {sorted(measured[True])}, which is not what "
        "the sentence under test is being measured against"
    )

    sentence = _zero_cost_sentence()
    claimed = _figures_claimed_per_side(sentence)
    assert set(claimed) == {True, False}, (
        f"the page describes {sorted(claimed)} of the two resolutions, so a reader whose "
        f"switch resolves the other way is told nothing: {sentence!r}"
    )
    assert (claimed[True] == measured[True]) is (cost == 0), (
        f"the page says a generation billed at nothing prints {sorted(claimed[True])} where "
        f"the merge resolves True; a cost of {cost} prints {sorted(measured[True])}: {sentence!r}"
    )
    if cost == 0:
        assert claimed[False] == measured[False], (
            f"the page says {sorted(claimed[False])} where the merge resolves False, and "
            f"the line prints {sorted(measured[False])}: {sentence!r}"
        )


# ------------------- WHAT A PANEL DRAWS THAT ITS MODEL NEVER PUBLISHED ------
def _always_drawn_pairs(dedicated: bool) -> list[tuple[str, str]]:
    """(field, label) for every control a panel carries whatever its model publishes.

    Read off the renderer's own tables rather than listed here, so a control joining or
    leaving that set moves this expectation with it instead of leaving a list behind.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        IMAGE_KNOB_TITLES,
        _valve_name,
        always_on_controls,
    )
    from open_webui_openrouter_pipe.integrations.image_types import SCHEMA_ONLY_PARAMS

    pairs = [(name, title) for name, _a, _d, title, _desc in always_on_controls(dedicated)]
    pairs += [(_valve_name(name), IMAGE_KNOB_TITLES[name][0]) for name in SCHEMA_ONLY_PARAMS]
    return pairs


def _panel_pairs(model_id: str, records: list[dict], dedicated: bool) -> list[tuple[str, str]]:
    """(field, label) for every control the installed panel really draws, in order."""
    body = _panel_body(model_id, records, dedicated)
    fields = [name for name, _annotation in _FIELD_RE.findall(body)]
    labels = [double or single for double, single in _TITLE_PAIR_RE.findall(body)]
    assert len(fields) == len(labels), (
        f"{model_id} renders {len(fields)} fields and {len(labels)} labels, so no field "
        f"can be paired with the name it is drawn under:\n{body}"
    )
    return list(zip(fields, labels))


def _panel_body(model_id: str, records: list[dict], dedicated: bool) -> str:
    spec = build_image_model_filter_spec(
        model_id, {"id": model_id, "name": model_id}, records, dedicated_image_api=dedicated
    )
    source = render_image_model_filter_source(spec)
    return source.split("class UserValves(BaseModel):", 1)[1].split("    def __init__", 1)[0]


@pytest.mark.parametrize("dedicated", [True, False], ids=["image-api", "chat-route"])
def test_every_recorded_panel_carries_the_controls_no_contract_asks_for(dedicated):
    """The set an administrator is promised on every panel, measured on every panel.

    The valve row describing this promised four controls beyond whatever the model
    publishes: an output size on all of them, and three more on a model that answers with
    a picture and no text. Nothing had ever rendered a panel to check that the four are
    drawn, under those names, on the models the promise quantifies over.

    Both halves are read from production -- the expected pairs off the renderer's tables,
    the drawn pairs out of the source that is installed -- so the two arms differ by which
    controls the transport is entitled to. The picture-only three are asserted ABSENT on
    the chat route rather than merely not-required, because a helper that ignored its
    transport argument would otherwise leave both arms green.
    """
    expected = _always_drawn_pairs(dedicated)
    assert expected, "the renderer draws nothing unbidden, so this measures nothing"
    reserved = set(_always_drawn_pairs(True)) - set(_always_drawn_pairs(False))
    assert reserved, (
        "production reserves no control for the picture-only transport, so neither arm "
        "here can tell the two transports apart"
    )

    absent: dict[str, list[tuple[str, str]]] = {}
    leaked: dict[str, list[str]] = {}
    for slug, model_id in EVERY_CONTRACT:
        drawn = _panel_pairs(model_id, _records(slug), dedicated)
        missing = [pair for pair in expected if pair not in drawn]
        if missing:
            absent[model_id] = missing
        if not dedicated:
            names = {name for name, _label in drawn} | {label for _name, label in drawn}
            found = sorted(
                token
                for pair in reserved
                for token in pair
                if token in names
            )
            if found:
                leaked[model_id] = found

    assert not absent, (
        "these recorded models get a panel that never draws a control the valve promises "
        f"every panel carries: {absent}"
    )
    assert not leaked, (
        "these models answer with text as well, and their panel draws controls production "
        f"reserves for the models that answer only with a picture: {leaked}"
    )


# ------------------- WHERE EACH GROUP OF CONTROLS LANDS IN THE HELP LIST -----
def _listed_controls(model_id: str, records: list[dict], dedicated: bool) -> list[str]:
    """The control entries of a rendered card, in the order a reader meets them."""
    listed = _help_controls(model_id, records, dedicated)
    assert listed, f"the {model_id} card lists no control at all, so order proves nothing"
    return listed


def _published_groups(model_id: str, records: list[dict], dedicated: bool) -> dict[str, list[str]]:
    """The labels of what the model published, split into the groups help renders in turn."""
    from open_webui_openrouter_pipe.filters.image_filter_renderer import image_knob_text

    spec = build_image_model_filter_spec(
        model_id, {"id": model_id, "name": model_id}, records, dedicated_image_api=dedicated
    )
    return {
        "choices": [image_knob_text(name, spec)[0] for name, _values in spec.enums],
        "rest": (
            [image_knob_text(name, spec)[0] for name, _low, _high in spec.ranges]
            + [image_knob_text(name, spec)[0] for name in spec.supported]
            + list(spec.passthrough)
        ),
    }


@pytest.mark.parametrize("dedicated", [True, False], ids=["image-api", "chat-route"])
def test_the_help_list_orders_its_groups_the_way_the_page_describes(dedicated):
    """The page put the always-present controls "above" the published ones. One of them is not.

    Provider options, Reference images and Reference image links do head the list. Output
    size does not: it is rendered after the published lists of choices and before
    everything else the model publishes, which the page's own worked example shows a few
    lines above the sentence that contradicted it.

    Positions are asserted by INDEX, not by membership: every label checked here is in the
    list under either ordering, so a membership test cannot see the two groups swap. Each
    group's labels are read off the same spec the card is rendered from, so a model that
    publishes different things moves the expectation rather than the assertion.
    """
    unbidden = [label for _name, label in _always_drawn_pairs(dedicated)]
    heads = [label for _name, label in _always_drawn_pairs(dedicated)
             if (_name, label) in set(_always_drawn_pairs(True)) - set(_always_drawn_pairs(False))]
    sized = [label for label in unbidden if label not in heads]
    assert len(sized) == 1, f"the panel carries {sized} on every transport, and this reads one"

    wrong: dict[str, str] = {}
    exercised = {"heads": 0, "choices": 0, "rest": 0}
    for slug, model_id in CONTRACTS_WITH_KNOBS:
        records = _records(slug)
        listed = _listed_controls(model_id, records, dedicated)
        groups = _published_groups(model_id, records, dedicated)
        assert groups["choices"], f"{model_id} publishes no list of choices, so it cannot rank"
        at = {label: listed.index(label) for label in listed}
        missing = [
            label
            for label in (*heads, *sized, *groups["choices"], *groups["rest"])
            if label not in at
        ]
        assert not missing, f"the {model_id} card never lists {missing}; it lists {listed}"

        size_at = at[sized[0]]
        first_choice = min(at[label] for label in groups["choices"])
        exercised["choices"] += 1
        if heads:
            exercised["heads"] += 1
            if max(at[label] for label in heads) > first_choice:
                wrong[model_id] = (
                    f"the controls the panel supplies sit at "
                    f"{sorted(at[label] for label in heads)} and the published choices "
                    f"start at {first_choice}"
                )
                continue
        if size_at < first_choice:
            wrong[model_id] = (
                f"{sized[0]} sits at {size_at}, above the published choices at {first_choice}"
            )
            continue
        if groups["rest"]:
            exercised["rest"] += 1
            follows = min(at[label] for label in groups["rest"])
            if follows < size_at:
                wrong[model_id] = (
                    f"{sized[0]} sits at {size_at}, below the rest of what the model "
                    f"publishes at {follows}"
                )

    assert not wrong, f"these cards list their groups in another order: {wrong}"
    assert exercised["choices"] > 30 and exercised["rest"] > 30, (
        f"only {exercised} contracts reached the comparisons, so the sweep went hollow"
    )
    assert (exercised["heads"] > 30) is dedicated, (
        f"the transport no longer decides whether the panel supplies controls: {exercised}"
    )


# ------------------- A REPLY THAT REPORTED NO COUNTS AT ALL ------------------
def _status_parts(line: str) -> set[str]:
    """The figures a status line is built from, split on the separator that joins them."""
    return set(line.split(" | "))


def test_a_reply_that_reported_no_counts_gets_no_counts_on_the_line(pipe_instance):
    """The page said the switch decides whether counts are printed. The payload decides too.

    A generation billed at nothing was described as printing tokens and timing wherever
    the merge resolves True. The builder appends a token figure only for counters the
    reply actually carried, and a reply carrying a charge and no counters -- which is what
    the image path produces, since the converter copies only the keys OpenRouter sent --
    leaves the True side with the timing alone.

    Nothing is matched against a word. The line the builder printed for a reply carrying
    counts is compared against the line it printed for a reply carrying none, and the part
    that separates them is identified by moving the reported counts and seeing which part
    moves. A counter parser that answered zero for a count nobody reported would put a
    token figure on the no-count line, and the proper-subset assertion is what refuses it;
    asserting only that a token figure appears where counts were reported would not.
    """
    counted = _status_with(pipe_instance, admin=True, user=True, cost=0)
    other = _status_with(
        pipe_instance, admin=True, user=True, cost=0,
        counters={"input_tokens": 801, "output_tokens": 400},
    )
    uncounted = _status_with(pipe_instance, admin=True, user=True, cost=0, counters={})
    off_counted = _status_with(pipe_instance, admin=False, user=False, cost=0)
    off_uncounted = _status_with(pipe_instance, admin=False, user=False, cost=0, counters={})

    assert len(_status_parts(counted)) > 1, (
        f"the builder joins its figures some other way now, so {counted!r} cannot be "
        "taken apart and every comparison below is between whole lines"
    )
    assert off_counted == off_uncounted, (
        "with the merge resolving False the two replies print different lines, so the "
        f"switch is not the whole of that side after all: {off_counted!r} / {off_uncounted!r}"
    )
    assert _status_parts(uncounted) < _status_parts(counted), (
        "a reply that reported no counts prints a figure a reply that reported some does "
        f"not, so the line is carrying something nobody reported: {uncounted!r} against "
        f"{counted!r}"
    )

    dropped = _status_parts(counted) - _status_parts(uncounted)
    moved = _status_parts(counted) - _status_parts(other)
    assert moved, (
        f"changing the reported counts changed nothing on the line: {counted!r} and "
        f"{other!r} are built from the same parts, so no part of it is the counts"
    )
    assert moved <= dropped, (
        f"the parts that move with the reported counts are {sorted(moved)} and the parts "
        f"a reply reporting none loses are {sorted(dropped)}; the counts are surviving a "
        "reply that never carried them"
    )
