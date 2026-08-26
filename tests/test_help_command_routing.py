"""Typing ``help`` asks for the panel, on both generation routes, and spends nothing.

Open WebUI prepends a Workspace model's system prompt to the request, and both generation
APIs take one free-text ``prompt``, so the pipe composes the two. A system prompt is a
modifier of a request; ``help`` is not a request, so composing must not change whether it
is recognised -- missing it bills a generation of the word "help".
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import EncryptedStr, Pipe
from open_webui_openrouter_pipe.filters.video_filter_renderer import build_video_filter_spec
from open_webui_openrouter_pipe.integrations.image_help import OPENROUTER_PRICING
from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter
from open_webui_openrouter_pipe.integrations.video_help import (
    VIDEO_HELP_BY_MODEL,
    render_video_help,
)
from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

BASE = "https://openrouter.ai/api/v1"
GEMINI = "google/gemini-3-pro-image"

_VIDEO_CATALOG_FIXTURE = Path(__file__).parent / "fixtures" / "video_models_catalog.json"
VIDEO_BY_ID = {item["id"]: item for item in json.loads(_VIDEO_CATALOG_FIXTURE.read_text())["data"]}


def _image_records(model_id: str) -> list[dict[str, Any]]:
    name = model_id.replace("/", "_")
    path = Path(__file__).resolve().parent / "fixtures" / f"openrouter_image_endpoints_{name}.json"
    return json.loads(path.read_text())["endpoints"]


SYSTEM_TEXTS = [
    pytest.param(None, id="no-system-prompt"),
    pytest.param("HOUSE STYLE: always cel-shaded, teal background", id="system-house-style"),
    pytest.param("STUDIO RULE: hand-held camera, 35mm grain", id="system-studio-rule"),
]

ASKS = [
    pytest.param("help", False, id="help"),
    pytest.param("a red mug on a windowsill", True, id="a-real-request"),
]


def _messages(system_text: str | None, user_text: str) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    if system_text is not None:
        turns.append({"role": "system", "content": system_text})
    turns.append({"role": "user", "content": user_text})
    return turns


class _MemoryPersistence:
    def __init__(self) -> None:
        self.content = ""

    async def load_message_content(self, *, chat_id: str, message_id: str) -> str:
        return self.content


@pytest.mark.asyncio
@pytest.mark.parametrize(("user_text", "submits"), ASKS)
@pytest.mark.parametrize("system_text", SYSTEM_TEXTS)
@pytest.mark.parametrize("model_id", ["openai/sora-2-pro", "google/veo-3.1"])
async def test_video_help_renders_the_panel_and_starts_no_job(
    monkeypatch, model_id, system_text, user_text, submits
):
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.VIDEO_INTENT_ENABLED = False
    adapter = pipe._ensure_video_generation_adapter()
    cast(Any, adapter)._persistence = _MemoryPersistence()
    submitted: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def submit(self, payload):
            submitted.append(payload)
            raise RuntimeError("stop after the request was made")

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", FakeClient
    )

    try:
        result = await adapter.generate(
            body={"messages": _messages(system_text, user_text)},
            responses_body=SimpleNamespace(provider={}),
            valves=pipe.valves,
            session=object(),
            event_emitter=None,
            metadata={"chat_id": "chat-1", "message_id": "msg-1", "user_id": "user-1"},
            user={"id": "user-1"},
            request=None,
            user_obj={"id": "user-1"},
            normalized_model_id=model_id.replace("/", "."),
            api_model_id=model_id,
        )
    finally:
        await pipe.close()

    assert bool(submitted) is submits, (
        f"submitted={submitted!r} for {user_text!r} with system={system_text!r}"
    )
    panel = f"### {VIDEO_HELP_BY_MODEL[model_id]['display_name']}"
    if submits:
        assert panel not in result
    else:
        assert result.startswith(panel)
        assert "**Output capabilities**" in result


@pytest.mark.parametrize(("user_text", "should_run"), ASKS)
@pytest.mark.parametrize("system_text", SYSTEM_TEXTS)
def test_video_help_never_pays_the_intent_classifier(system_text, user_text, should_run):
    pipe = Pipe()
    adapter = VideoGenerationAdapter(pipe=pipe, logger=logging.getLogger("tests.help_routing"))
    body = {"messages": _messages(system_text, user_text)}

    assert (
        adapter._intent_classifier_should_run(
            valves=pipe.valves,
            persisted_content="",
            prompt=adapter._extract_prompt(body),
            body=body,
            video_meta={"frame_images": [{"file_id": "frame-1"}]},
            metadata=None,
            chat_id="chat-1",
            user_id="user-1",
        )
        is should_run
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("user_text", "submits"), ASKS)
@pytest.mark.parametrize("system_text", SYSTEM_TEXTS)
async def test_image_help_renders_the_panel_and_sends_no_request(
    system_text, user_text, submits
):
    OpenRouterModelRegistry.set_image_endpoints({GEMINI: _image_records(GEMINI)})
    pipe = Pipe()
    sent: list[dict[str, Any]] = []
    try:
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        pipe.valves.BASE_URL = BASE
        pipe.valves.ENABLE_OPENROUTER_IMAGE_GENERATION = True

        def _callback(url, **kwargs):
            from aioresponses import CallbackResult

            sent.append(kwargs["json"])
            return CallbackResult(
                status=200,
                body=(
                    'data: {"type":"response.completed","response":{"output":[],'
                    '"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
                ),
                headers={"Content-Type": "text/event-stream"},
            )

        with aioresponses() as mocked:
            mocked.post(f"{BASE}/responses", callback=_callback, repeat=True)
            mocked.post(f"{BASE}/chat/completions", callback=_callback, repeat=True)
            mocked.post(f"{BASE}/images", callback=_callback, repeat=True)
            mocked.get(
                f"{BASE}/models",
                payload={
                    "data": [
                        {
                            "id": GEMINI,
                            "name": "Gemini 3 Pro Image",
                            "architecture": {"output_modalities": ["image", "text"]},
                        }
                    ]
                },
                repeat=True,
            )
            answer = await pipe.pipe(
                body={
                    "model": GEMINI,
                    "messages": _messages(system_text, user_text),
                    "stream": False,
                },
                __user__={"id": "user_1"},
                __request__=None,
                __event_emitter__=None,
                __event_call__=None,
                __metadata__={"model": {"id": GEMINI}},
                __tools__=None,
                __task__=None,
                __task_body__=None,
            )
            stream = cast(Any, answer)
            if hasattr(stream, "__aiter__"):
                chunks = [chunk async for chunk in stream]
                answer = "".join(str(chunk) for chunk in chunks)
    finally:
        await pipe.close()
        OpenRouterModelRegistry.set_image_endpoints({})

    assert bool(sent) is submits, (
        f"sent={sent!r} for {user_text!r} with system={system_text!r}"
    )
    if not submits:
        assert "Gemini 3 Pro Image" in str(answer)
        assert "## Controls" in str(answer)


_ABSENT = object()

_CAPABILITY_LINE = {"seed": "- Deterministic seed:", "generate_audio": "- Generated audio:"}
_CAPABILITY_KNOB = {"seed": "- `Seed`", "generate_audio": "- `Audio`"}


@pytest.mark.parametrize("field", ["seed", "generate_audio"])
@pytest.mark.parametrize(
    ("published", "drawn", "reads"),
    [
        pytest.param(True, True, "yes", id="declared-on"),
        pytest.param(False, False, "no", id="declared-off"),
        pytest.param(None, True, "not published", id="declared-nothing"),
        pytest.param(_ABSENT, False, "no", id="not-in-the-contract"),
    ],
)
def test_video_help_reports_the_capability_its_own_panel_draws(field, published, drawn, reads):
    """Three published states, not two, and help must read them the way the filter does.

    `null` declares nothing: the filter offers the control and the model applies its own
    default. Help judged it with `is True`, so a model publishing `seed: null` was told
    "Deterministic seed: no" beside a panel that draws a Seed control.
    """
    model = dict(VIDEO_BY_ID["google/veo-3.1"])
    if published is _ABSENT:
        model.pop(field, None)
    else:
        model[field] = published

    spec = build_video_filter_spec(model["id"], model)
    offered = spec.supports_seed if field == "seed" else spec.supports_generate_audio_toggle
    assert offered is drawn, "the fixture no longer exercises the state this case is about"

    text = render_video_help(model["id"], model)
    line = next(row for row in text.splitlines() if row.startswith(_CAPABILITY_LINE[field]))
    assert reads in line, line
    assert (_CAPABILITY_KNOB[field] in text) is drawn, text


@pytest.mark.parametrize(
    ("fixture", "model_id", "dedicated"),
    [
        ("google_gemini-3-pro-image", "google/gemini-3-pro-image", False),
        ("openai_gpt-image-2", "openai/gpt-image-2", True),
    ],
)
def test_image_help_offers_every_value_its_own_panel_offers(fixture, model_id, dedicated):
    """The values in the control, not the values before agreement was applied.

    A model served by several companies gets the ones they all accept plus the ones only
    some do, marked. Help listed the shared set alone, so gemini's panel offered 4K while
    its help said the model does 1K and 2K.
    """
    import ast
    import re

    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        IMAGE_KNOB_TITLES,
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )
    from open_webui_openrouter_pipe.integrations.image_help import render_image_help

    record = json.loads(
        (Path(__file__).parent / "fixtures" / f"openrouter_image_endpoints_{fixture}.json")
        .read_text()
    )["endpoints"]
    model = {"id": model_id, "name": model_id}
    spec = build_image_model_filter_spec(
        model_id, model, record, dedicated_image_api=dedicated
    )
    source = render_image_model_filter_source(spec)

    drawn = {}
    for name, literals in re.findall(
        r"^\s+IMAGE_([A-Z0-9_]+): Literal\[(.+?)\] = Field\($", source, re.M
    ):
        options = [v for v in ast.literal_eval(f"[{literals}]") if v != ""]
        drawn[name] = options
    assert drawn, "the fixture must draw a choice control for this to mean anything"

    rendered = render_image_help(
        model_id, model, endpoint_record=record, dedicated_image_api=dedicated
    )
    for published, _values in spec.enums:
        title = IMAGE_KNOB_TITLES.get(published, (published, ""))[0]
        row = next(r for r in rendered.splitlines() if r.startswith(f"- **{title}** "))
        listed = row.split("Choices: ", 1)[1].split(".", 1)[0]
        assert listed == ", ".join(drawn[published.upper()]), row

    # Help lists the controls the panel drew, so a control the panel withholds on this
    # model's transport must not be described as available on it.
    assert ("**Reference images**" in rendered) is dedicated, (
        f"{model_id}: help describes the reference controls "
        f"{'**Reference images**' in rendered!r} while the panel draws them {dedicated!r}"
    )
    assert ("IMAGE_REFERENCE_MODE:" in source) is dedicated, (
        f"{model_id}: the panel draws the reference control on a transport that does "
        "not read it"
    )


_MONEY_WORDS = frozenset(
    """
    price prices pricing priced pricier priciest
    cost costs costed costing costly costlier costliest
    cheap cheaper cheapest cheaply expensive
    bill bills billed billing charge charges charged fee fees rate rates
    pay pays paid spend spends spent
    surcharge surcharges discount discounts affordable premium economics
    dollar dollars
    """.split()
)
"""Every word these tables have used to say what something is worth in money.

What a model charges is OpenRouter's to publish and to change, so the property is not
"no figure" or "no comparison" but "no money vocabulary at all". A word ban would be the
wrong shape for open documentation, where these words carry other senses; this corpus is
one table this project writes, so a word here belongs to the card that uses it.

Read off the corpus rather than invented: every entry was in these cards, plus the
inflections a rewrite reaches for next. `free` is deliberately absent -- its thirteen
uses are `free-text`, `classifier-free` and `free-running`, none about money -- and so
are `per-second`, `per-token` and `per-image`, which are units. A unit needs one of the
words above to say what is being measured, which is why banning the words catches
"billed per second" and still lets a card write "24 frames per second".
"""

_CURRENCY = re.compile(r"[$£€¥]\s*\d")


def _money_vocabulary(text: str) -> list[str]:
    """Every money word in one string, with the pointer at OpenRouter removed first.

    That one shared sentence is the only place the corpus may name money, and it names no
    figure, no direction and no comparison -- it hands the question to the company that
    owns the answer. Stripping the sentence rather than exempting a word is what keeps
    this a ban: a card carrying the pointer AND a price of its own still fails on the
    price.

    Split on letters so a hyphenated compound is read as its parts: `cost-optimised`,
    `Token-priced` and `speed-and-cost-optimised` are how the removed positioning labels
    were written, and a whole-token match would have walked past all three.
    """
    stripped = text.replace(OPENROUTER_PRICING, "")
    found = sorted(
        {word for word in re.findall(r"[A-Za-z]+", stripped.lower()) if word in _MONEY_WORDS}
    )
    if _CURRENCY.search(stripped):
        found.append("a currency figure")
    return found


def _curated_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _curated_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _curated_strings(item)


@pytest.mark.parametrize(
    ("text", "flagged"),
    [
        pytest.param("Billed at $0.12 per second of output.", True, id="a-currency-figure"),
        pytest.param(
            "Selects 720p, 1080p, or 4K, which also sets the rate you are charged "
            "(720p cheapest, 4K most expensive) and how long the render takes.",
            True,
            id="a-direction-across-one-model-s-own-tiers",
        ),
        pytest.param(
            "Google DeepMind's speed-and-cost-optimised tier of Veo 3.1.",
            True,
            id="a-positioning-label",
        ),
        pytest.param(
            "Duration is any integer 1-15 seconds and cost scales linearly per second.",
            True,
            id="the-shape-of-the-billing",
        ),
        pytest.param(
            "Pricing is dynamic: the published per-image rate is a starting point.",
            True,
            id="a-billing-rule-with-no-figure-in-it",
        ),
        pytest.param(
            "Same capability matrix as Standard, at a lower per-second rate than Pro.",
            True,
            id="a-comparison-against-another-listing",
        ),
        pytest.param(
            "Thinking Mode improves coherence with weaker fast-motion physics than "
            "Seedance 2.0.",
            False,
            id="a-trade-off-worded-without-the-money-metaphor",
        ),
        pytest.param(
            "Free-text list of things to exclude, and classifier-free guidance strength.",
            False,
            id="free-is-not-a-money-word-here",
        ),
        pytest.param(
            "Multi-shot 1080p at 24 frames per second with synchronised native audio.",
            False,
            id="a-unit-of-measure-is-not-a-price",
        ),
        pytest.param(
            "Pins exact pixel dimensions, 1920x1080 or 2048x2048.",
            False,
            id="dimensions-are-not-figures-of-money",
        ),
        pytest.param(
            "~3x slower than V4 due to the higher resolution.",
            False,
            id="a-magnitude-about-something-other-than-money",
        ),
        pytest.param(OPENROUTER_PRICING, False, id="the-pointer-at-openrouter-itself"),
    ],
)
def test_the_money_word_ban_reads_the_vocabulary_and_not_one_phrasing(text, flagged):
    """Twelve rows split six-six, so a detector answering the same way every time fails.

    The guard this replaces was a hundred lines: a lexicon built from every display name
    in both catalogues, a comparative arm, a superlative arm, a magnitude arm, and a
    scope test deciding whether the other side of a comparison was a second listing or
    one of the model's own tiers. All of that existed to allow "720p is the cheaper of
    the two" while refusing "cheaper than Pro". The decision that a card states no price
    of any kind removes the distinction the machinery was drawn to make, and with it the
    machinery.
    """
    assert bool(_money_vocabulary(text)) is flagged, _money_vocabulary(text)


def _every_curated_surface() -> list[tuple[str, str]]:
    """Both halves: the tables as written, and every card as a reader receives it.

    A knob description is only rendered where the model publishes that capability, so a
    sweep over renderings alone never reads the gated ones. A rendering carries text no
    table holds -- the shared control descriptions, the four intent knobs, the passthrough
    line -- so a sweep over the tables alone never reads those. Neither half covers the
    other.
    """
    from open_webui_openrouter_pipe.integrations.image_help import (
        IMAGE_HELP_BY_MODEL,
        render_image_help,
    )
    from open_webui_openrouter_pipe.integrations.video_help import _INTENT_KNOB_DESCRIPTIONS

    surfaces: list[tuple[str, str]] = []
    for label, table in (("image", IMAGE_HELP_BY_MODEL), ("video", VIDEO_HELP_BY_MODEL)):
        for model_id, data in table.items():
            surfaces.extend(
                (f"{label} table {model_id}", text) for text in _curated_strings(data)
            )
    surfaces.extend(
        ("intent table", text) for text in _curated_strings(_INTENT_KNOB_DESCRIPTIONS)
    )
    for model_id, model in VIDEO_BY_ID.items():
        surfaces.append((f"video card {model_id}", render_video_help(model_id, model)))
    for model_id in IMAGE_HELP_BY_MODEL:
        name = model_id.replace("/", "_")
        path = Path(__file__).resolve().parent / "fixtures" / (
            f"openrouter_image_endpoints_{name}.json"
        )
        surfaces.append((
            f"image card {model_id}",
            render_image_help(
                model_id,
                {"id": model_id, "name": model_id},
                endpoint_record=_image_records(model_id) if path.exists() else None,
                dedicated_image_api=True,
            ),
        ))
    return surfaces


def test_no_curated_help_string_names_money_at_all():
    """No card says what a thing costs, how it is billed, or how its price compares.

    Two passes before this one took out the currency figures and then the comparisons
    against other listings, and each time what was left still told a reader about money:
    "720p cheapest, 4K most expensive", "Cost-optimized Gemini 3.1", "cost scales
    linearly per second". A direction between one model's own tiers survives a reprice,
    which is why the earlier guard allowed it -- but surviving a reprice was never the
    point. Pricing is OpenRouter's, and a card that positions itself on price is doing
    OpenRouter's job with none of its information.
    """
    surfaces = _every_curated_surface()
    assert len(surfaces) > 400, f"only {len(surfaces)} surfaces; the sweep went hollow"
    assert any(
        card.startswith("### ") or card.startswith("# ")
        for where, card in surfaces
        if where.endswith("card") or " card " in where
    ), "no card rendered, so half this sweep is reading nothing"
    assert any(OPENROUTER_PRICING in text for _where, text in surfaces), (
        "nothing uses the pointer, so the one exemption above is never exercised and a "
        "card could smuggle a price through it unnoticed"
    )

    offenders = [
        f"{where}: {_money_vocabulary(text)} in {text[:140]!r}"
        for where, text in surfaces
        if _money_vocabulary(text)
    ]
    assert not offenders, "\n".join(offenders)


def test_every_knob_help_names_is_a_control_the_panel_draws_under_that_name():
    """A name only helps if it is the one printed above the control in the chat panel.

    Help called the Seedance request-key control `Req key` while the panel titled it
    `Request key`, so a user reading the panel found no entry for the control in front of
    them, on all four Seedance models. Titles are read out of the generated filter source
    rather than listed here, so renaming one without renaming the other reddens this.
    """
    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        render_video_filter_source,
    )

    mismatched: list[str] = []
    for model_id, model in VIDEO_BY_ID.items():
        source = render_video_filter_source(model_id=model_id, video_model=model)
        drawn = set(re.findall(r"""^\s+title=(?:"([^"]+)"|'([^']+)'),$""", source, re.M))
        titles = {name for pair in drawn for name in pair if name}
        assert titles, f"{model_id} draws no titled control, so this proves nothing"
        for knob in re.findall(r"^- `([^`]+)`:", render_video_help(model_id, model), re.M):
            if knob not in titles:
                mismatched.append(f"{model_id}: help names {knob!r}, the panel does not")

    assert not mismatched, "\n".join(mismatched)


@pytest.mark.asyncio
@pytest.mark.parametrize("intent_enabled", [True, False])
async def test_the_panel_the_help_command_returns_reads_the_admin_intent_valve(intent_enabled):
    """The valve has to reach the panel through the command, not just through the helper.

    The four intent controls are on the filter only while the admin keeps the classifier
    on, so the panel can only agree with the filter if the same valves reach both. The
    parity checks above call the renderer directly and would stay green if the `help`
    command stopped passing them, leaving a panel that advertises four controls the
    filter no longer draws.
    """
    model_id = "openai/sora-2-pro"
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.VIDEO_INTENT_ENABLED = intent_enabled
    adapter = pipe._ensure_video_generation_adapter()
    cast(Any, adapter)._persistence = _MemoryPersistence()

    try:
        result = await adapter.generate(
            body={"messages": _messages(None, "help")},
            responses_body=SimpleNamespace(provider={}),
            valves=pipe.valves,
            session=object(),
            event_emitter=None,
            metadata={"chat_id": "chat-1", "message_id": "msg-1", "user_id": "user-1"},
            user={"id": "user-1"},
            request=None,
            user_obj={"id": "user-1"},
            normalized_model_id=model_id.replace("/", "."),
            api_model_id=model_id,
        )
    finally:
        await pipe.close()

    from open_webui_openrouter_pipe.integrations.video_help import _INTENT_KNOB_DESCRIPTIONS

    assert _INTENT_KNOB_DESCRIPTIONS, "no intent control to look for, so this checks nothing"
    named = [knob for knob in _INTENT_KNOB_DESCRIPTIONS if f"- `{knob}`:" in result]
    expected = list(_INTENT_KNOB_DESCRIPTIONS) if intent_enabled else []
    assert named == expected, (
        f"VIDEO_INTENT_ENABLED={intent_enabled} draws "
        f"{'the intent controls' if intent_enabled else 'no intent control'} on the "
        f"filter, and the panel names {named}"
    )


def _intent_admin_valves(enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        VIDEO_INTENT_ENABLED=enabled,
        VIDEO_INTENT_MAX_CLARIFICATIONS=1,
        VIDEO_INTENT_FRAME_EXTRACTION_INDEX="last",
        VIDEO_INTENT_CONFIRM_MODE="on_reference",
    )


@pytest.mark.parametrize("intent_enabled", [True, False])
def test_help_lists_every_control_on_that_models_filter_and_no_others(intent_enabled):
    """The other direction of the same property, and it was wrong on every model.

    ``test_every_knob_help_names_is_a_control_the_panel_draws_under_that_name`` catches
    help naming a control that is not there. This catches a control that is there and is
    not named: all twenty-two filters drew `Reuse previous videos`, `Clarifying question
    limit`, `Which frame to use from previous video` and `Show what was reused`, and no
    panel mentioned any of them -- including the one that is on by default and decides
    whether "make it black" edits the clip you just got or starts an unrelated new one.

    Both sides are rendered from one admin valves object, and both admin states are run,
    because the four come off the filter entirely when an admin turns the classifier off.
    A panel that listed them unconditionally would pass the first state and fail this one.
    Equality is against that model's own rendered source, never a union: `runway/aleph-2`
    publishes no duration and no resolution, and must not be asked to list either.
    """
    from open_webui_openrouter_pipe.filters.video_filter_renderer import (
        render_video_filter_source,
    )

    curated = [model_id for model_id in VIDEO_BY_ID if model_id in VIDEO_HELP_BY_MODEL]
    assert curated, "the curated route draws the knob list; nothing to check without it"

    valves = _intent_admin_valves(intent_enabled)
    mismatched: list[str] = []
    for model_id in curated:
        model = VIDEO_BY_ID[model_id]
        source = render_video_filter_source(
            model_id=model_id, video_model=model, admin_valves=valves
        )
        drawn = set(re.findall(r"""^\s+title=(?:"([^"]+)"|'([^']+)'),$""", source, re.M))
        titles = {name for pair in drawn for name in pair if name}
        listed = set(re.findall(r"^- `([^`]+)`:", render_video_help(model_id, model, admin_valves=valves), re.M))
        if titles != listed:
            mismatched.append(
                f"{model_id}: on the filter but not in the panel {sorted(titles - listed)}; "
                f"in the panel but not on the filter {sorted(listed - titles)}"
            )

    assert not mismatched, "\n".join(mismatched)
