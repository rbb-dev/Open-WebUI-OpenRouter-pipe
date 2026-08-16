"""Typing ``help`` asks for the panel, on both generation routes, and spends nothing.

Open WebUI prepends a Workspace model's system prompt to the request, and both generation
APIs take one free-text ``prompt``, so the pipe composes the two. A system prompt is a
modifier of a request; ``help`` is not a request, so composing must not change whether it
is recognised -- missing it bills a generation of the word "help".
"""

from __future__ import annotations

import json
import logging
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
