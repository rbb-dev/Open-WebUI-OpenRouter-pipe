"""Controls a user could not reach before, driven end to end.

Every test here loads the rendered filter the way Open WebUI loads one, runs its `inlet`,
and then puts the result through the adapter, because a control that writes into
`image_config` or into the request metadata and is dropped on the way out is worse than
no control: it reports a setting as in force that nothing carries.

Each is parametrised over two distinct values, so a production function that returned a
constant would satisfy at most one row.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe.filters.image_filter_renderer import (
    ALWAYS_ON_VALVE_NAMES,
    build_image_model_filter_spec,
    render_image_model_filter_source,
)
from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter
from open_webui_openrouter_pipe.integrations.image_types import (
    PASSTHROUGH_ENUMS,
    SCHEMA_ONLY_PARAMS,
    TOP_LEVEL_PARAMS,
)
from tests.test_image_api_path import (  # noqa: F401 - shared stubs, one definition
    BASE,
    _adapter,
    _Emitter,
    _KeyPipe,
    _posted,
    _StubResponsesBody,
    _StubValves,
    _user_turn_with_images,
)
from tests.test_image_generation import _load_filter_from_source

PIPE_META = "openrouter_pipe"

_RECORDED_CONTRACTS = sorted(
    (Path(__file__).parent / "fixtures").glob("openrouter_image_endpoints_*.json")
)

assert len(_RECORDED_CONTRACTS) > 30, (
    f"only {len(_RECORDED_CONTRACTS)} recorded contracts found; the fleet-wide sweeps "
    "below parametrise over this list, and an empty one collects no nodes and SKIPS "
    "rather than failing. Fails at collection so the drift cannot pass as a green run."
)

_RECRAFT = {
    "provider_slug": "recraft",
    "allowed_passthrough_parameters": ["style", "controls"],
    "supported_parameters": {
        "aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]},
        "n": {"type": "range", "min": 1, "max": 6},
        "input_references": {"type": "range", "min": 0, "max": 1},
    },
}

_OPENAI = {
    "provider_slug": "openai",
    "allowed_passthrough_parameters": ["moderation"],
    "supported_parameters": {
        "n": {"type": "range", "min": 1, "max": 4},
        "input_references": {"type": "range", "min": 0, "max": 16},
    },
}


def _filter(record: dict[str, Any], name: str, model_id: str = "m/x") -> Any:
    spec = build_image_model_filter_spec(model_id, {"id": model_id, "name": model_id}, [record])
    return _load_filter_from_source(render_image_model_filter_source(spec), name)


def _inlet(module: Any, valves: dict[str, Any], model_id: str = "m/x"):
    body: dict[str, Any] = {"model": model_id}
    metadata: dict[str, Any] = {}
    module.Filter().inlet(body, metadata, {"valves": module.Filter.UserValves(**valves)})
    return body, metadata


# =============================================================================
# 1a - a provider-options escape hatch on the image path
# =============================================================================


@pytest.mark.parametrize(
    ("slug", "option", "value"),
    [("recraft", "style", "realistic_image"), ("openai", "quality_hint", "sharp")],
)
def test_a_provider_option_the_contract_never_named_reaches_the_request(slug, option, value):
    """Six of the forty live models publish an empty option list, so their filters draw
    no option control at all. Nothing else on the image path can address a company."""
    module = _filter(_RECRAFT, f"hatch_{slug}")
    _body, metadata = _inlet(
        module, {"IMAGE_PROVIDER_OPTIONS_JSON": json.dumps({slug: {option: value}})}
    )

    assert metadata[PIPE_META]["provider"]["options"] == {slug: {option: value}}


@pytest.mark.parametrize(("kept", "added"), [("recraft", "openai"), ("fal", "azure")])
def test_the_escape_hatch_adds_to_the_routing_choice_rather_than_replacing_it(kept, added):
    """Provider routing writes the same place. Overwriting it would drop an operator's
    pin the moment any user typed one option."""
    module = _filter(_RECRAFT, f"hatch_merge_{kept}")
    body: dict[str, Any] = {"model": "m/x"}
    metadata: dict[str, Any] = {PIPE_META: {"provider": {"only": [kept], "options": {kept: {"a": 1}}}}}
    module.Filter().inlet(
        body,
        metadata,
        {"valves": module.Filter.UserValves(IMAGE_PROVIDER_OPTIONS_JSON=json.dumps({added: {"b": 2}}))},
    )

    provider = metadata[PIPE_META]["provider"]
    assert provider["only"] == [kept], "the routing pin must survive"
    assert provider["options"] == {kept: {"a": 1}, added: {"b": 2}}


@pytest.mark.parametrize("typed", ["{not json", '["a list"]'])
def test_a_provider_options_value_that_is_not_an_object_is_refused_by_name(typed):
    module = _filter(_RECRAFT, f"hatch_bad_{abs(hash(typed))}")
    with pytest.raises(Exception) as caught:
        _inlet(module, {"IMAGE_PROVIDER_OPTIONS_JSON": typed})
    assert "Provider options" in str(caught.value)


@pytest.mark.parametrize(
    ("slug", "option", "value"),
    [("recraft", "style", "vector_illustration"), ("recraft", "steps", "40")],
)
@pytest.mark.asyncio
async def test_the_escape_hatch_survives_the_adapter_and_reaches_the_wire(slug, option, value):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [_RECRAFT])

    result = await _posted(
        adapter,
        body={},
        responses_body=_user_turn_with_images(0),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        metadata={"chat_id": "c", PIPE_META: {"provider": {"options": {slug: {option: value}}}}},
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    assert result.payload["provider"]["options"][slug][option] == value


# =============================================================================
# 1b - a value only some of a model's providers accept
# =============================================================================


def _gemini_records() -> list[dict[str, Any]]:
    raw = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "openrouter_image_endpoints_google_gemini-3-pro-image.json"
        ).read_text()
    )
    return raw["endpoints"]


def test_a_tier_only_one_provider_publishes_is_offered_and_marked():
    """Recorded live: one of this model's two companies lists 4K and the other does not.
    Offering only the overlap left it unreachable with nothing saying so."""
    records = _gemini_records()
    spec = build_image_model_filter_spec("google/gemini-3-pro-image", None, records)

    assert dict(spec.narrowed) == {"resolution": ("4K",)}
    module = _load_filter_from_source(render_image_model_filter_source(spec), "narrowed_gemini")
    offered = module.Filter.UserValves.model_fields["IMAGE_RESOLUTION"]
    from typing import get_args

    assert set(get_args(offered.annotation)) == {"", "1K", "2K", "4K"}
    assert "4K" in (offered.description or ""), (
        "a value only some companies accept has to say so on the control, or it reads as "
        "one every provider takes"
    )


@pytest.mark.parametrize(
    ("slug", "wanted", "sent"),
    [("google-ai-studio/global", "4K", True), ("google-vertex/global", "4K", False)],
)
@pytest.mark.asyncio
async def test_a_narrowed_value_is_sent_or_reported_by_the_provider_that_serves(slug, wanted, sent):
    """The offer is safe because the adapter still fits the value to the record that
    serves the request, and says so when it does not fit."""
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), _gemini_records())

    result = await _posted(
        adapter,
        body={"image_config": {"resolution": wanted}},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}],
            provider={"only": [slug]},
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    notices = " ".join(str(call.get("content", "")) for call in result.events)
    assert (result.payload.get("resolution") == wanted) is sent
    assert ("resolution" in notices) is not sent


# =============================================================================
# 3 - a documented field no contract describes
# =============================================================================


def test_the_split_of_top_level_names_is_a_partition():
    from open_webui_openrouter_pipe.integrations.image_types import CONTRACT_GATED_PARAMS

    assert set(CONTRACT_GATED_PARAMS) | set(SCHEMA_ONLY_PARAMS) == set(TOP_LEVEL_PARAMS)

    # Anchored outside the partition. `CONTRACT_GATED_PARAMS` is *defined* as
    # TOP_LEVEL minus SCHEMA_ONLY, so asserting the two do not intersect is true for
    # every possible value of either and catches no mis-partition at all.
    recorded = json.loads(
        (Path(__file__).parent / "fixtures" / "openrouter_request_schema_fields.json").read_text()
    )["image"]["properties"]
    assert set(SCHEMA_ONLY_PARAMS) <= set(TOP_LEVEL_PARAMS)
    assert set(TOP_LEVEL_PARAMS) <= set(recorded), sorted(set(TOP_LEVEL_PARAMS) - set(recorded))


@pytest.mark.parametrize("record", [_RECRAFT, _OPENAI])
def test_every_schema_only_field_gets_a_control_whatever_the_contract_says(record):
    """Zero of the forty-four recorded contracts describe `size`, so a control gated on
    the contract is drawn on no model at all."""
    spec = build_image_model_filter_spec("m/x", None, [record])
    assert spec.schema_only == SCHEMA_ONLY_PARAMS

    module = _load_filter_from_source(
        render_image_model_filter_source(spec), f"schema_only_{record['provider_slug']}"
    )
    field = module.Filter.UserValves.model_fields["IMAGE_SIZE"]
    assert field.annotation is str
    assert "publishes no list" in (field.description or ""), (
        "a value that cannot be checked before it is sent must say so where it is set"
    )


@pytest.mark.parametrize("chosen", ["2048x2048", "4K"])
def test_a_schema_only_value_travels_from_the_control_into_the_request(chosen):
    module = _filter(_RECRAFT, f"size_travel_{chosen}")
    body, _metadata = _inlet(module, {"IMAGE_SIZE": chosen})
    assert body["image_config"]["size"] == chosen

    top_level, _provider, notes = ImageGenerationAdapter._split_image_config(
        body, allowed_passthrough=frozenset(_RECRAFT["allowed_passthrough_parameters"]), record=_RECRAFT
    )
    assert top_level["size"] == chosen
    assert not [note for note in notes if note.name == "size"], (
        "the model publishes no limits for this field, so there is nothing it can be "
        f"reported as failing. got {[n.text for n in notes]}"
    )


@pytest.mark.parametrize("chosen", ["1024x1024", "2K"])
@pytest.mark.asyncio
async def test_a_schema_only_value_reaches_the_wire(chosen):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [_RECRAFT])

    result = await _posted(
        adapter,
        body={"image_config": {"size": chosen}},
        responses_body=_user_turn_with_images(0),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )
    assert result.payload["size"] == chosen


# =============================================================================
# 4 - which images become references
# =============================================================================


@pytest.mark.parametrize(("mode", "kept"), [("latest-only", 1), ("none", 0)])
@pytest.mark.asyncio
async def test_the_reference_mode_decides_how_many_attachments_are_sent(mode, kept):
    """Sixteen of the forty-four recorded contracts accept a single reference, and the
    adapter picked which one by document order with no way to choose."""
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (
        time.monotonic(),
        [{**_RECRAFT, "supported_parameters": {**_RECRAFT["supported_parameters"],
                                               "input_references": {"type": "range", "min": 0, "max": 4}}}],
    )

    result = await _posted(
        adapter,
        body={},
        responses_body=_user_turn_with_images(3),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        metadata={"chat_id": "c", PIPE_META: {"image_generation": {"reference_mode": mode}}},
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    assert len(result.payload.get("input_references", [])) == kept
    if kept:
        assert result.payload["input_references"][0]["image_url"]["url"].endswith("/2/content"), (
            "latest-only means the most recent attachment, not the first"
        )


@pytest.mark.parametrize(
    "link", ["https://example.com/a.png", "https://example.org/b.jpg"]
)
@pytest.mark.asyncio
async def test_a_typed_link_is_sent_ahead_of_the_attachments(link):
    """On the sixteen single-reference models an explicit link must be the one that
    survives the cap, or supplying one would silently do nothing."""
    pipe = _KeyPipe("sk-x")
    cast(Any, pipe)._multimodal_handler = _AllowAll()
    adapter = _adapter(pipe)
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [_RECRAFT])

    result = await _posted(
        adapter,
        body={},
        responses_body=_user_turn_with_images(2),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        metadata={"chat_id": "c", PIPE_META: {"image_generation": {"reference_urls": [link]}}},
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    assert [ref["image_url"]["url"] for ref in result.payload["input_references"]] == [link]


class _AllowAll:
    async def _is_safe_url(self, url: str) -> bool:
        return True


class _RefuseAll:
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def _is_safe_url(self, url: str) -> bool:
        self.seen.append(url)
        return False


@pytest.mark.parametrize("link", ["http://169.254.169.254/latest", "http://10.0.0.1/x.png"])
@pytest.mark.asyncio
async def test_a_typed_link_goes_through_the_same_gate_as_every_other_fetched_url(link):
    """OpenRouter fetches whatever is listed, so an unchecked link here would be a
    request this deployment makes to an address its user chose."""
    import aiohttp
    from aioresponses import aioresponses

    handler = _RefuseAll()
    pipe = _KeyPipe("sk-x")
    cast(Any, pipe)._multimodal_handler = handler
    adapter = _adapter(pipe)
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [_RECRAFT])

    with aioresponses() as mocked:
        mocked.post(f"{BASE}/images", payload={"data": []})
        async with aiohttp.ClientSession() as session:
            content = await adapter.generate(
                body={},
                responses_body=_user_turn_with_images(0),
                valves=_StubValves("sk-x"),
                session=session,
                event_emitter=_Emitter(),
                metadata={"chat_id": "c", PIPE_META: {"image_generation": {"reference_urls": [link]}}},
                user=None,
                request=object(),
                user_obj=object(),
                normalized_model_id="m.x",
                api_model_id="m/x",
            )
        posts = [key for key in mocked.requests if key[1].path == "/api/v1/images"]

    assert handler.seen == [link], "the gate has to actually be asked about this link"
    assert not posts, "a refused link must stop the request, not generate without it"
    assert "https" in content


@pytest.mark.parametrize("published", [None, 40])
def test_no_request_carries_more_references_than_the_format_allows(published):
    """The request format caps the list at sixteen. An unreadable contract used to mean
    no cap at all, which is a rejected request rather than a large one."""
    record = (
        None
        if published is None
        else {"supported_parameters": {"input_references": {"type": "range", "min": 0, "max": published}}}
    )
    assert ImageGenerationAdapter._reference_limit(record) == published


@pytest.mark.parametrize("attached", [20, 30])
@pytest.mark.asyncio
async def test_an_unreadable_contract_still_caps_the_reference_list(attached):
    adapter = _adapter(_KeyPipe("sk-x"))
    notes: list[Any] = []
    refs = await adapter._reference_payload(
        _user_turn_with_images(attached), None, record=None, notes=notes
    )
    assert len(refs) == 16
    assert any("16" in note.text for note in notes)


# =============================================================================
# 5 - a provider option whose values are published
# =============================================================================


@pytest.mark.parametrize("chosen", ["auto", "low"])
def test_the_published_values_of_a_named_option_are_offered_as_choices(chosen):
    """Delivered on the six OpenAI models that name it, as free text, so a typo went out
    as a moderation setting and came back as a rejection about something else."""
    from typing import get_args

    module = _filter(_OPENAI, f"moderation_{chosen}", model_id="openai/gpt-image-2")
    field = module.Filter.UserValves.model_fields["IMAGE_MODERATION"]
    assert set(get_args(field.annotation)) == {"", *PASSTHROUGH_ENUMS["moderation"][0]}

    body, _metadata = _inlet(
        module, {"IMAGE_MODERATION": chosen}, model_id="openai/gpt-image-2"
    )
    assert body["image_config"]["moderation"] == chosen


def test_a_typed_option_replaces_the_free_text_one_rather_than_joining_it():
    """Two fields would collide on one name, the second silently dropped, so which one
    a user saw would depend on the order they were built in."""
    module = _filter(_OPENAI, "moderation_single", model_id="openai/gpt-image-2")
    fields = module.Filter.UserValves.model_fields
    assert len([name for name in fields if name.endswith("MODERATION")]) == 1


@pytest.mark.parametrize("chosen", ["auto", "low"])
@pytest.mark.asyncio
async def test_a_typed_option_still_lands_under_the_company_serving_the_request(chosen):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [_OPENAI])

    result = await _posted(
        adapter,
        body={"image_config": {"moderation": chosen}},
        responses_body=_user_turn_with_images(0),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )
    assert result.payload["provider"]["options"]["openai"]["moderation"] == chosen


# =============================================================================
# the always-on controls exist on every model, and only once
# =============================================================================


@pytest.mark.parametrize("path", _RECORDED_CONTRACTS, ids=lambda p: p.stem)
def test_the_controls_that_come_from_no_contract_are_drawn_on_every_model(path):
    """One node per recorded contract, read from inside the tree under test.

    This globbed an absolute path into `.external/`, which is gitignored, so on any
    checkout without the sweep -- CI on every run -- it skipped, and the fleet-wide claim
    it makes was measured on nothing. Run from a scratch copy it read the *other* tree.
    """
    raw = json.loads(path.read_text())
    spec = build_image_model_filter_spec(raw["id"], None, raw["endpoints"])
    module = _load_filter_from_source(
        render_image_model_filter_source(spec), f"always_on_{spec.dotted_id}"
    )
    assert ALWAYS_ON_VALVE_NAMES <= set(module.Filter.UserValves.model_fields), raw["id"]


@pytest.mark.parametrize("published", ["provider_options_json", "reference_urls"])
def test_a_published_name_cannot_take_over_a_reserved_control(published):
    """The field name decides three things that must agree, and two fields sharing one
    leaves the last definition standing with no warning."""
    record = {**_RECRAFT, "allowed_passthrough_parameters": [published]}
    spec = build_image_model_filter_spec("m/x", None, [record])
    assert published not in spec.passthrough

    module = _load_filter_from_source(
        render_image_model_filter_source(spec), f"reserved_{published}"
    )
    field = module.Filter.UserValves.model_fields[f"IMAGE_{published.upper()}"]
    assert field.title in {"Provider options", "Reference image links"}, (
        "the surviving field must be the reserved control, not a free-text box wearing "
        f"its name. got title={field.title!r}"
    )
