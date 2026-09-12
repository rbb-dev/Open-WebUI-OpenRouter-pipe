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

import base64
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
from open_webui_openrouter_pipe.filters.image_filter_renderer import (
    IMAGE_GEN_TOOL_PARAMS,
    build_image_gen_tool_spec,
    build_image_model_filter_spec,
    render_image_gen_filter_source,
)
from open_webui_openrouter_pipe.integrations.image_types import SCHEMA_ENUMS
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


@pytest.mark.parametrize(
    "references",
    [
        pytest.param([], id="nothing-attached"),
        pytest.param(
            [{"id": "file-1", "kind": "image", "content_type": "image/png"}],
            id="one-reference-attached",
        ),
    ],
)
@pytest.mark.parametrize("model_id", ["google/veo-3.1", "openai/sora-2-pro"])
@pytest.mark.asyncio
async def test_no_named_gap_appears_in_the_payload_the_adapter_submits(
    monkeypatch, model_id, references
):
    """The reason has to describe the code. A field listed as unreachable that the
    adapter does send would be a false record, which is worse than none.

    Read the request rather than the file that builds it. Reading the file only sees a
    key spelled as a literal beside `payload[`, so a name built by concatenation, held in
    a variable, or merged in through `dict.update` went out to OpenRouter with nothing
    here noticing -- and `submit` posts what it is handed, with no key filter of its own.

    What is observed is one whole turn: a text prompt plus the per-model controls, run
    once with nothing attached and once with a reference. The empty turn alone made the
    check vacuous -- the intersection it asserts against was the empty set whatever the
    record said -- and that is what hid `input_references`, recorded as unreachable while
    the adapter sent it. Two models because their catalogue entries differ, so a builder
    returning a fixed dict fails one.
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

    async def _file(file_id, _logger):
        return SimpleNamespace(id=file_id, meta={"content_type": "image/png"})

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.get_file_by_id", _file
    )
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.infer_file_mime_type",
        lambda _obj: "image/png",
    )

    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    OpenRouterModelRegistry.register_video_models([VIDEO_CATALOG[model_id]])
    adapter = pipe._ensure_video_generation_adapter()
    cast(Any, adapter)._persistence = _NoPersistence()

    async def _b64(*_args, **_kwargs):
        return base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()

    monkeypatch.setattr(pipe._file_gateway, "read_file_record_base64", _b64)

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
                    },
                    "input_references": references,
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
    if references:
        assert "input_references" in sent, (
            "the attached reference never reached the request, so this row observes the "
            "same empty intersection the text-only row does and adds no coverage"
        )
    assert not (sent & set(VIDEO_FIELD_GAPS)), (
        f"{sorted(sent & set(VIDEO_FIELD_GAPS))} went to OpenRouter while the record calls "
        "them unreachable, so the written reason describes code that is not there"
    )
    assert sent <= set(VIDEO_REQUEST_FIELDS), (
        f"{sorted(sent - set(VIDEO_REQUEST_FIELDS))} is in neither half of the partition; a "
        "gap spelled differently escapes the check above by not being the recorded name"
    )


def test_the_closed_value_lists_match_the_recorded_request_format():
    """Every closed enum the pipe offers is the one OpenRouter's request format publishes.

    These are the values the server-tool filter falls back to when a model publishes no
    list of its own. Hand-trimming them narrowed what users could ask for below what the
    API accepts -- `svg` vanished from every vectorising model, `auto` from every
    background. The comparison runs against the recording so a literal cannot satisfy it.
    """
    recorded = RECORDED["image"]["enums"]
    assert recorded, "the enum recording is empty, so this gate asserts nothing"

    assert set(SCHEMA_ENUMS) == set(recorded), (
        f"unlisted={sorted(set(recorded) - set(SCHEMA_ENUMS))} "
        f"invented={sorted(set(SCHEMA_ENUMS) - set(recorded))}"
    )
    for name, values in sorted(recorded.items()):
        assert SCHEMA_ENUMS[name] == tuple(values), (
            f"{name}: dropped={sorted(set(values) - set(SCHEMA_ENUMS[name]))} "
            f"invented={sorted(set(SCHEMA_ENUMS[name]) - set(values))}"
        )


_RECORDED_IMAGE_CONTRACTS = sorted(
    (Path(__file__).parent / "fixtures").glob("openrouter_image_endpoints_*.json")
)

assert len(_RECORDED_IMAGE_CONTRACTS) > 30, (
    f"only {len(_RECORDED_IMAGE_CONTRACTS)} recorded contracts found; the sweeps below "
    "parametrise over this list, and an empty one collects no nodes and SKIPS rather "
    "than failing"
)


def _tool_domains(path: Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    raw = json.loads(path.read_text())
    spec = build_image_model_filter_spec(
        raw["id"], None, raw["endpoints"], dedicated_image_api=True
    )
    tool = build_image_gen_tool_spec(spec)
    published = {name: values for name, values in spec.enums}
    published.update({name: (low, high) for name, low, high in spec.ranges})
    offered = {name: values for name, values in tool.enums}
    offered.update({name: (low, high) for name, low, high in tool.ranges})
    return published, offered, tool


@pytest.mark.parametrize(
    "path", _RECORDED_IMAGE_CONTRACTS, ids=lambda p: p.stem.split("endpoints_")[-1]
)
def test_the_server_tool_offers_no_value_list_its_model_did_not_publish(path):
    """Every closed list the tool filter draws came from the model it will run on.

    The tool filter used to substitute the API-wide enum whenever a model published
    nothing -- so three models were handed a `background`, an `output_format` and an
    `output_compression` range they publish nothing for, and `svg` was a selectable
    option on all forty recorded models while no recorded contract publishes it. An
    absent key means the parameter is unsupported by that endpoint, so the control has
    to fall back to free entry, the way `quality` already did.
    """
    published, offered, tool = _tool_domains(path)

    for name, domain in offered.items():
        assert name in published, (
            f"{path.stem}: the tool draws a domain for {name} that the model publishes "
            f"nothing for: {domain!r}"
        )
        assert domain == published[name], (
            f"{path.stem}: {name} is offered as {domain!r} where the model publishes "
            f"{published[name]!r}"
        )

    for name in IMAGE_GEN_TOOL_PARAMS:
        # the tool spells the resolution tier `size` on the wire
        published_as = "resolution" if name == "size" else name
        assert (published_as in offered) == (published_as in published), (
            f"{path.stem}: {name} draws a domain {published_as in offered!r} while the "
            f"model publishes one {published_as in published!r}"
        )
        if published_as not in published:
            assert name in tool.schema_only, (
                f"{path.stem}: {name} is published by nothing and drawn by nothing, so "
                "the capability is unreachable rather than unconstrained"
            )


def test_the_sweep_sees_both_a_model_that_publishes_a_list_and_one_that_does_not():
    """Neither half of the rule above may be vacuous across the recorded fleet.

    A sweep that only ever met models publishing nothing would pass with the domain
    check never running, and one that only met models publishing everything would pass
    with the fallback check never running.
    """
    with_domain: set[str] = set()
    without_domain: set[str] = set()
    for path in _RECORDED_IMAGE_CONTRACTS:
        published, offered, _tool = _tool_domains(path)
        for name in IMAGE_GEN_TOOL_PARAMS:
            published_as = "resolution" if name == "size" else name
            (with_domain if published_as in offered else without_domain).add(name)

    assert {"background", "quality", "output_compression"} <= with_domain, (
        f"no recorded model publishes these, so the domain half asserts nothing: "
        f"{sorted({'background', 'quality', 'output_compression'} - with_domain)}"
    )
    assert {"background", "quality", "output_compression", "output_format"} <= without_domain, (
        f"every recorded model publishes these, so the free-entry half asserts nothing: "
        f"{sorted({'background', 'quality', 'output_compression', 'output_format'} - without_domain)}"
    )


@pytest.mark.parametrize(
    "path", _RECORDED_IMAGE_CONTRACTS, ids=lambda p: p.stem.split("endpoints_")[-1]
)
def test_no_rendered_tool_filter_offers_a_format_its_model_never_published(path):
    """`svg` reached the chat UI as a choice on every model, and none publishes it.

    Asserted on the rendered source rather than the spec, because the spec is what the
    renderer reads and a Literal written from anywhere else would not show up there.
    """
    import re

    raw = json.loads(path.read_text())
    spec = build_image_model_filter_spec(
        raw["id"], None, raw["endpoints"], dedicated_image_api=True
    )
    source = render_image_gen_filter_source(
        spec, catalog_match=True, selected_model=raw["id"]
    )
    _operator, _sep, user_valves = source.partition("class UserValves(BaseModel):")
    assert _sep, "the tool filter draws no user valve block at all"
    published = dict(spec.enums)
    for field, literal in re.findall(r"IMAGE_([A-Z_]+): Literal\[([^\]]*)\]", user_valves):
        name = field.lower()
        wire = "resolution" if name == "size" else name
        drawn = {
            value for value in re.findall(r"'([^']*)'", literal) if value
        }
        assert wire in published, (
            f"{raw['id']}: {name} is drawn as a closed list {sorted(drawn)} while the "
            "model publishes no list for it"
        )
        assert drawn <= set(published[wire]), (
            f"{raw['id']}: {name} offers {sorted(drawn - set(published[wire]))}, which "
            "the model does not publish"
        )


def test_the_server_tool_carries_every_parameter_its_table_documents():
    """The panel's parameter set is the published table, not a copy of itself.

    Deriving the expectation from `IMAGE_GEN_TOOL_PARAMS` made this vacuous: deleting a
    name from the constant deleted the control and the check for it in one move, and the
    suite stayed green. The recording is transcribed from OpenRouter's own server-tool
    page, so the two can disagree.
    """
    recorded = RECORDED["image_server_tool"]
    documented = set(recorded["parameters"])
    assert documented, "the server-tool recording is empty, so this gate asserts nothing"

    operator_owned = set(recorded["chosen_by_the_operator"])
    assert operator_owned <= documented, (
        f"the recording marks a parameter its own table does not list: "
        f"{sorted(operator_owned - documented)}"
    )
    assert set(IMAGE_GEN_TOOL_PARAMS) == documented - operator_owned, (
        f"unlisted={sorted(documented - operator_owned - set(IMAGE_GEN_TOOL_PARAMS))} "
        f"invented={sorted(set(IMAGE_GEN_TOOL_PARAMS) - documented)}"
    )


@pytest.mark.parametrize("transport", ["image", "video", "chat"])
def test_the_accepted_provider_keys_are_the_ones_the_format_publishes(transport):
    """Each transport accepts the keys its own provider schema defines, and no others.

    The picker's gate derived its expectation from the same constants the picker reads,
    so adding a key to one moved both sides together: five phantom price controls could
    be drawn on every image model, emitting a `max_price` the image request format has no
    field for, with the whole suite still green. The recording is transcribed from
    OpenRouter's reference pages, so production and expectation can disagree.
    """
    from open_webui_openrouter_pipe.integrations.provider_options import (
        CHAT_PROVIDER_KEYS,
        IMAGE_PROVIDER_KEYS,
        VIDEO_PROVIDER_KEYS,
    )

    accepted = {
        "image": IMAGE_PROVIDER_KEYS,
        "video": VIDEO_PROVIDER_KEYS,
        "chat": CHAT_PROVIDER_KEYS,
    }[transport]
    recorded = set(RECORDED[transport]["provider_properties"])
    assert recorded, f"the {transport} provider recording is empty, so this gate asserts nothing"
    assert set(accepted) == recorded, (
        f"{transport}: invented={sorted(set(accepted) - recorded)} "
        f"missing={sorted(recorded - set(accepted))}"
    )


def test_the_catalog_partition_covers_every_key_the_catalogue_publishes():
    """A second arm, because the two recordings go stale on different cadences.

    The request-format arm above compares against a schema recorded by hand. That recording
    was made 2026-08-19 behind the catalogue, and in that window a model arrived publishing
    two typed control domains -- `upscale_factor` and `creativity` -- that reached no control,
    no warning and no list, while the gate built to notice exactly that passed. It could not
    see them: they are catalogue keys, and it only reads request-body keys.

    The catalogue is the recording that grows a new control the day a model ships, so it gets
    its own arm. The expected set is computed from the fixture, never typed here.
    """
    from open_webui_openrouter_pipe.integrations.request_fields import (
        VIDEO_CATALOG_FIELD_GAPS,
        VIDEO_CATALOG_FIELD_ROUTES,
        VIDEO_CATALOG_FIELDS,
    )

    published: set[str] = set()
    for row in VIDEO_CATALOG.values():
        published |= set(row)
    assert len(published) > 10, f"only {len(published)} keys seen; the fixture is not loading"

    covered = set(VIDEO_CATALOG_FIELD_ROUTES) | set(VIDEO_CATALOG_FIELD_GAPS)
    assert covered == published, (
        f"unlisted={sorted(published - covered)} invented={sorted(covered - published)}. "
        "A key the catalogue publishes and nothing here mentions is indistinguishable from "
        "one deliberately left alone -- which is how a model shipped with its only two "
        "settings reaching nothing at all."
    )
    assert VIDEO_CATALOG_FIELDS == covered, (
        "the exported set is the two halves put together, not a third list beside them"
    )
    assert not (set(VIDEO_CATALOG_FIELD_ROUTES) & set(VIDEO_CATALOG_FIELD_GAPS))


def test_every_catalog_gap_carries_a_reason_long_enough_to_be_one():
    """A label is not an explanation; a reader must be able to tell a decision from an oversight."""
    from open_webui_openrouter_pipe.integrations.request_fields import VIDEO_CATALOG_FIELD_GAPS

    assert VIDEO_CATALOG_FIELD_GAPS, "claiming no gaps at all needs removing this test, not emptying the list"
    for field, reason in VIDEO_CATALOG_FIELD_GAPS.items():
        assert len(reason.split()) >= 12, f"{field}: {reason!r}"


def test_a_catalog_key_carrying_a_control_domain_is_not_quietly_shrugged_off():
    """A gap whose reason is real and a gap that was never noticed read identically.

    Every gap entry names why nothing reads that key. For the keys that publish a settable
    domain on some model -- a {min,max} object or a non-empty list -- the reason has to say
    something about routing, not merely that the field is unused, because those are the ones
    a user can see advertised and cannot reach.
    """
    from open_webui_openrouter_pipe.integrations.request_fields import VIDEO_CATALOG_FIELD_GAPS

    domained = {
        key
        for row in VIDEO_CATALOG.values()
        for key, value in row.items()
        if isinstance(value, dict) and {"min", "max"} <= set(value)
        or isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value)
    }
    unexplained = [
        key
        for key in sorted(domained & set(VIDEO_CATALOG_FIELD_GAPS))
        if not any(
            word in VIDEO_CATALOG_FIELD_GAPS[key].lower()
            for word in ("route", "request format", "passthrough", "send")
        )
    ]
    assert not unexplained, (
        f"these keys publish a control domain but their gap reason never says why it cannot "
        f"be sent: {unexplained}"
    )
