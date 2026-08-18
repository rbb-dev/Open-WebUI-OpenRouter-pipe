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
    ("fixture", "model_id"),
    [
        ("google_gemini-3-pro-image", "google/gemini-3-pro-image"),
        ("openai_gpt-image-2", "openai/gpt-image-2"),
    ],
)
def test_image_help_offers_every_value_its_own_panel_offers(fixture, model_id):
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
    spec = build_image_model_filter_spec(model_id, model, record)
    source = render_image_model_filter_source(spec)

    drawn = {}
    for name, literals in re.findall(
        r"^\s+IMAGE_([A-Z0-9_]+): Literal\[(.+?)\] = Field\($", source, re.M
    ):
        options = [v for v in ast.literal_eval(f"[{literals}]") if v != ""]
        drawn[name] = options
    assert drawn, "the fixture must draw a choice control for this to mean anything"

    rendered = render_image_help(model_id, model, endpoint_record=record)
    for published, _values in spec.enums:
        title = IMAGE_KNOB_TITLES.get(published, (published, ""))[0]
        row = next(r for r in rendered.splitlines() if r.startswith(f"- **{title}** "))
        listed = row.split("Choices: ", 1)[1].split(".", 1)[0]
        assert listed == ", ".join(drawn[published.upper()]), row


_MONEY_WORDS = re.compile(
    r"\b(pric\w*|cost\w*|cheap\w*|expensive|bill|bills|billed|billing|rate|rates|"
    r"charge\w*|spend\w*|paid|pay|per[- ]second|per[- ]token|per[- ]image)\b",
    re.I,
)

_CURRENCY = re.compile(r"\$\s*\d")

_MAGNITUDES = (
    ("a percentage", re.compile(r"\d\s*%")),
    ("a multiplier", re.compile(r"\d(?:\.\d+)?\s*[×x](?![\dx])")),
    ("a magnitude in words", re.compile(
        r"\b(half again|twice|double\w*|triple\w*|order of magnitude)\b", re.I
    )),
)


def _price_claims(text: str) -> list[str]:
    """Every way a curated string can state money the catalogue is free to change."""
    found: list[str] = []
    currency = _CURRENCY.search(text)
    if currency:
        found.append(f"a currency amount ({currency.group(0)!r})")
    if _MONEY_WORDS.search(text):
        for name, pattern in _MAGNITUDES:
            hit = pattern.search(text)
            if hit:
                found.append(f"{name} ({hit.group(0)!r})")
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
        pytest.param("Billed at $0.12 per second of output.", True, id="currency"),
        pytest.param(
            "Chooses 720p or 1080p; 1080p costs roughly 50% more per second.",
            True,
            id="percentage-of-a-cost",
        ),
        pytest.param(
            "Same controls as Standard but at roughly 1.33× the per-second price.",
            True,
            id="multiplier-of-a-price",
        ),
        pytest.param(
            "Pricing is per-second of output and half again as much with audio on.",
            True,
            id="magnitude-in-words",
        ),
        pytest.param(
            "Audio doubles the bill for the same clip.", True, id="doubles-the-bill"
        ),
        pytest.param(
            "Pins exact pixel dimensions, 1920×1080 or 2048x2048; the rate is unchanged.",
            False,
            id="dimensions-are-not-a-multiplier",
        ),
        pytest.param(
            "~3x slower than V4 due to the higher resolution.",
            False,
            id="a-magnitude-about-something-other-than-money",
        ),
        pytest.param(
            "1080p is billed at a higher per-second rate, listed below.",
            False,
            id="a-direction-cannot-go-stale",
        ),
        pytest.param("Frame shape: 16:9, 9:16, or 21:9.", False, id="ratios-are-framings"),
    ],
)
def test_the_price_claim_scanner_sees_every_form_it_is_meant_to(text, flagged):
    """The guard it replaces matched `$` and a digit, and nothing else.

    Every relative claim walked past it: "1.33×", "~33% more", "half again as much with
    audio on". Those are the same defect as the sixteen currency literals this changeset
    removed -- correct on the day they were typed, wrong after a reprice, and silent
    either way -- so the scanner has to see a multiplier, a percentage and a magnitude
    written out in words. It stays quiet for a magnitude that is not about money, and for
    a direction ("a higher rate"), which survives any reprice.
    """
    assert bool(_price_claims(text)) is flagged, _price_claims(text)


def test_no_curated_help_text_states_a_price_the_catalogue_can_move():
    """The panel renders money from the contract; the curated tables must not.

    Scanned over the tables rather than over one rendering of them: a knob description is
    only rendered when the model publishes that capability, so the rendered-output guards
    in the two generation test modules never see the gated ones at all.
    """
    from open_webui_openrouter_pipe.integrations.image_help import IMAGE_HELP_BY_MODEL
    from open_webui_openrouter_pipe.integrations.video_help import VIDEO_HELP_BY_MODEL

    offenders: list[str] = []
    for table in (VIDEO_HELP_BY_MODEL, IMAGE_HELP_BY_MODEL):
        for model_id, data in table.items():
            for text in _curated_strings(data):
                for claim in _price_claims(text):
                    offenders.append(f"{model_id}: {claim} in {text[:120]!r}")

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
