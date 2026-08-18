"""No field of either request format may be missing from the reachability partition.

`callback_url` was unreachable for a sound reason that nobody had written down, and the
video reference list accepted audio and video assets that nobody had noticed. Both look
the same from inside the code: a name in OpenRouter's request format that appears nowhere
here. This compares the partition against the field lists recorded from those formats, so
the two cases become distinguishable and a field added later fails a check instead of
joining the second case in silence.

The fixture is the recording. Re-record it when OpenRouter's request formats change, the
same way the endpoint contracts in this directory are re-recorded.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import EncryptedStr, Pipe
from open_webui_openrouter_pipe.integrations.request_fields import (
    IMAGE_FIELD_GAPS,
    IMAGE_FIELD_ROUTES,
    IMAGE_REQUEST_FIELDS,
    VIDEO_FIELD_GAPS,
    VIDEO_FIELD_ROUTES,
    VIDEO_REQUEST_FIELDS,
)
from open_webui_openrouter_pipe.integrations.video_types import VideoGenerationError
from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

RECORDED = json.loads(
    (Path(__file__).parent / "fixtures" / "openrouter_request_schema_fields.json").read_text()
)

VIDEO_CATALOG = {
    item["id"]: item
    for item in json.loads(
        (Path(__file__).parent / "fixtures" / "video_models_catalog.json").read_text()
    )["data"]
}


@pytest.mark.parametrize(
    ("kind", "routes", "gaps", "listed"),
    [
        ("image", IMAGE_FIELD_ROUTES, IMAGE_FIELD_GAPS, IMAGE_REQUEST_FIELDS),
        ("video", VIDEO_FIELD_ROUTES, VIDEO_FIELD_GAPS, VIDEO_REQUEST_FIELDS),
    ],
)
def test_the_partition_covers_the_recorded_request_format_exactly(kind, routes, gaps, listed):
    recorded = set(RECORDED[kind]["properties"])
    assert recorded, f"the recording for {kind} is empty, so this gate asserts nothing"

    covered = set(routes) | set(gaps)
    assert covered == recorded, (
        f"{kind}: unlisted={sorted(recorded - covered)} invented={sorted(covered - recorded)}. "
        "A field OpenRouter defines and nothing here mentions is indistinguishable from "
        "one deliberately left alone, which is how two of them went unnoticed for months."
    )
    assert listed == covered, (
        f"{kind}: the exported field set is the two halves put together, not a third list "
        "beside them; a copy could pass this comparison while the halves it names do not"
    )


@pytest.mark.parametrize(
    ("kind", "routes", "gaps"),
    [
        ("image", IMAGE_FIELD_ROUTES, IMAGE_FIELD_GAPS),
        ("video", VIDEO_FIELD_ROUTES, VIDEO_FIELD_GAPS),
    ],
)
def test_no_field_is_both_reached_and_unreachable(kind, routes, gaps):
    assert not (set(routes) & set(gaps)), kind
    assert all(text.strip() for text in (*routes.values(), *gaps.values())), kind


@pytest.mark.parametrize(("kind", "gaps"), [("video", VIDEO_FIELD_GAPS)])
def test_every_gap_carries_a_reason_long_enough_to_be_one(kind, gaps):
    """A one-word reason is a label, not an explanation, and the point of the list is
    that a reader can tell a decision from an oversight without reading the code."""
    assert gaps, f"{kind} claims no gaps at all; say so by removing this row, not by emptying it"
    for field, reason in gaps.items():
        assert len(reason.split()) >= 12, f"{kind}.{field}: {reason!r}"


@pytest.mark.parametrize("kind", ["image", "video"])
def test_a_required_field_is_never_listed_as_a_gap(kind):
    """A field the format demands cannot be one this deployment declines to send.

    Every recorded required field is checked, not one named here: naming one made the
    test a constant that a constant in the recording satisfied, and the video recording
    was missing `prompt` for exactly as long as nobody compared the two.
    """
    gaps = IMAGE_FIELD_GAPS if kind == "image" else VIDEO_FIELD_GAPS
    required = RECORDED[kind]["required"]
    assert required, f"{kind} records no required fields, so this gate asserts nothing"
    assert not (set(required) & set(gaps)), sorted(set(required) & set(gaps))


def test_the_video_adapter_reads_the_partition_rather_than_a_copy_of_it():
    """The adapter carried its own list of the same twelve names with nothing comparing
    them, so one could gain a field the other never heard about."""
    from open_webui_openrouter_pipe.integrations.video import (
        _DOCUMENTED_TOP_LEVEL_VIDEO_FIELDS,
    )

    assert _DOCUMENTED_TOP_LEVEL_VIDEO_FIELDS == VIDEO_REQUEST_FIELDS


class _NoPersistence:
    """The adapter reaches for storage only once a job comes back, which it never does
    here; these two exist so the attribute is not a bare mock that answers anything."""

    async def load_message_content(self, *, chat_id: str, message_id: str) -> str:
        return ""

    async def store_video_file_from_path(self, **_kwargs) -> str:
        raise AssertionError("nothing was generated, so nothing may be stored")


@pytest.mark.parametrize("model_id", ["google/veo-3.1", "openai/sora-2-pro"])
@pytest.mark.asyncio
async def test_no_named_gap_appears_in_the_payload_the_adapter_submits(monkeypatch, model_id):
    """The reason has to describe the code. A field listed as unreachable that the
    adapter does send would be a false record, which is worse than none.

    Read the request rather than the file that builds it. Reading the file only sees a
    key spelled as a literal beside `payload[`, so a name built by concatenation, held in
    a variable, or merged in through `dict.update` went out to OpenRouter with nothing
    here noticing -- and `submit` posts what it is handed, with no key filter of its own.

    What is observed is one whole turn: a text prompt plus the per-model controls, with
    nothing attached. On such a turn `input_references` has nothing that could fill it
    and `callback_url` is never built at all, so the whole gap list must be absent, and a
    gap added to the record later is covered without anyone naming it here. Two models
    because their catalogue entries differ, so a builder returning a fixed dict fails one.
    """
    submitted: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def submit(self, payload):
            submitted.append(payload)
            raise VideoGenerationError("stop after submit")

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", FakeClient
    )

    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    OpenRouterModelRegistry.register_video_models([VIDEO_CATALOG[model_id]])
    adapter = pipe._ensure_video_generation_adapter()
    cast(Any, adapter)._persistence = _NoPersistence()

    result = await adapter.generate(
        body={"messages": [{"role": "user", "content": "a paper kite over a harbour"}]},
        responses_body=SimpleNamespace(provider={}),
        valves=pipe.valves,
        session=None,
        event_emitter=None,
        metadata={
            "chat_id": "chat-1",
            "message_id": "msg-1",
            "openrouter_pipe": {
                "video_generation": {
                    "params": {
                        "aspect_ratio": "16:9",
                        "duration": 8,
                        "resolution": "720p",
                        "generate_audio": True,
                    }
                }
            },
        },
        user={"id": "user-1"},
        request=None,
        user_obj={"id": "user-1"},
        normalized_model_id=model_id.replace("/", "."),
        api_model_id=model_id,
    )

    assert submitted, f"the request never reached submit, so this proves nothing: {result!r}"
    payload = submitted[0]
    sent = set(payload)

    assert payload.get("model") == model_id, (
        f"the payload names {payload.get('model')!r} on a turn for {model_id!r}; a builder "
        "answering with a fixed dict would satisfy every check below"
    )
    assert {"prompt", "aspect_ratio", "duration", "resolution", "generate_audio"} <= sent, (
        f"the controls never reached the request ({sorted(sent)}), so the loop that copies "
        "them -- the one place a stray key would be added -- did not run and the checks "
        "below would hold over a payload nobody built"
    )
    assert sent & set(VIDEO_FIELD_ROUTES), "an empty payload would pass vacuously"
    assert not (sent & set(VIDEO_FIELD_GAPS)), (
        f"{sorted(sent & set(VIDEO_FIELD_GAPS))} went to OpenRouter while the record calls "
        "them unreachable, so the written reason describes code that is not there"
    )
    assert sent <= set(VIDEO_REQUEST_FIELDS), (
        f"{sorted(sent - set(VIDEO_REQUEST_FIELDS))} is in neither half of the partition; a "
        "gap spelled differently escapes the check above by not being the recorded name"
    )
