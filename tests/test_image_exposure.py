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
    spec = build_image_model_filter_spec(model_id, {"id": model_id, "name": model_id}, [record], dedicated_image_api=True)
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
    spec = build_image_model_filter_spec("google/gemini-3-pro-image", None, records, dedicated_image_api=True)

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
    spec = build_image_model_filter_spec("m/x", None, [record], dedicated_image_api=True)
    assert spec.schema_only == SCHEMA_ONLY_PARAMS

    module = _load_filter_from_source(
        render_image_model_filter_source(spec), f"schema_only_{record['provider_slug']}"
    )
    field = module.Filter.UserValves.model_fields["IMAGE_SIZE"]
    assert field.annotation is str
    described = field.description or ""
    assert "go out as typed" in described and "publishes a list of pixel sizes" in described, (
        "a value that cannot be checked before it is sent must say so where it is set; "
        f"got {described!r}"
    )
    for tier in ("512", "1K", "2K", "4K"):
        assert tier in described, (
            "OpenRouter's own schema says this field also takes a tier, and a control that "
            f"hides that tells the user not to type the thing that works; {tier} missing"
        )


def _publishes_only_1k_2k() -> dict[str, Any]:
    """The recorded Vertex record for Gemini 3 Pro Image: `resolution` is `1K, 2K`."""
    for record in _gemini_records():
        values = ((record.get("supported_parameters") or {}).get("resolution") or {}).get("values")
        if values == ["1K", "2K"]:
            return record
    raise AssertionError("the recorded contract no longer publishes a bounded resolution")


@pytest.mark.parametrize(
    ("contract", "chosen", "sent"),
    [
        ("no-resolution", "2048x2048", True),
        ("no-resolution", "4K", True),
        ("publishes-1K-2K", "2048x2048", True),
        ("publishes-1K-2K", "2K", True),
        ("publishes-1K-2K", "4K", False),
    ],
)
def test_a_schema_only_value_travels_from_the_control_into_the_request(contract, chosen, sent):
    """`size` is unconstrained except where it spells a tier the model rules out.

    OpenRouter words a tier `size` as equivalent to setting `resolution`, so on a model
    publishing `1K, 2K` the two spellings had opposite fates: `resolution="4K"` was
    refused with a note while `size="4K"` went out unchecked, and `IMAGE_SIZE` is free
    text on every model. An explicit pixel size stays unconstrained wherever it is set,
    because no contract describes pixel sizes -- so gating `size` on the mere presence of
    a `resolution` descriptor would refuse `2048x2048` on the nineteen endpoints that
    publish one.
    """
    record = _RECRAFT if contract == "no-resolution" else _publishes_only_1k_2k()
    published = ((record.get("supported_parameters") or {}).get("resolution") or {}).get("values")
    assert (published is None) == (contract == "no-resolution"), (
        f"the {contract} row is not the contract it names: resolution={published!r}"
    )

    module = _filter(record, f"size_travel_{contract}_{chosen}")
    body, _metadata = _inlet(module, {"IMAGE_SIZE": chosen})
    assert body["image_config"]["size"] == chosen, (
        "the control must write what was typed; the contract is the adapter's job"
    )

    top_level, _provider, notes = ImageGenerationAdapter._split_image_config(
        body,
        allowed_passthrough=frozenset(record.get("allowed_passthrough_parameters") or []),
        record=record,
    )
    reported = [note.text for note in notes if note.name == "size"]
    assert (top_level.get("size") == chosen) is sent, (
        f"{contract} + size={chosen!r}: top_level={top_level!r} notes={reported!r}"
    )
    assert bool(reported) is not sent, (
        f"{contract} + size={chosen!r}: a value that is not sent has to say so, and one "
        f"that is sent has nothing to report. got {reported!r}"
    )


@pytest.mark.parametrize(("tier", "kept"), [("2K", "resolution"), ("1K", "resolution")])
def test_a_refused_tier_size_does_not_evict_the_resolution_that_passed_the_contract(tier, kept):
    """The unchecked spelling used to win, and the user was told the checked one lost.

    With both set, `size` supersedes `resolution` because they set the same thing -- so
    an out-of-contract `size` deleted a `resolution` the model does publish and reported
    the wrong one of the two as withheld. Two in-contract tiers are asserted so a rule
    that always kept `resolution` regardless of what it held cannot satisfy both rows.
    """
    record = _publishes_only_1k_2k()
    top_level, _provider, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"resolution": tier, "size": "4K"}},
        allowed_passthrough=frozenset(record.get("allowed_passthrough_parameters") or []),
        record=record,
    )

    assert top_level == {kept: tier}, (
        f"the validated tier was thrown away for an unchecked one: {top_level!r}"
    )
    reported = " ".join(note.text for note in notes)
    assert "size=" in reported and "resolution" in reported, (
        f"the setting that was not sent has to be named as the one that was not sent: "
        f"{reported!r}"
    )
    assert f"resolution={tier!r}" not in reported, (
        f"the tier that passed the contract was reported as withheld: {reported!r}"
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
        assert result.payload["input_references"][0]["image_url"]["url"].endswith("att2"), (
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


class _Gate:
    def __init__(self, allow: set[str] | None = None) -> None:
        self.seen: list[str] = []
        self._allow = allow or set()

    async def _is_safe_url(self, url: str) -> bool:
        self.seen.append(url)
        return url in self._allow


def _turn_carrying(urls: list[str]) -> Any:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": "make it bluer"}]
    for url in urls:
        content.append({"type": "input_image", "image_url": {"url": url}, "detail": "auto"})
    return _StubResponsesBody([{"role": "user", "content": content}])


def _gated_adapter(allow: set[str] | None = None) -> tuple[Any, _Gate]:
    gate = _Gate(allow)
    pipe = _KeyPipe("sk-x")
    cast(Any, pipe)._multimodal_handler = gate
    adapter = _adapter(pipe)
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [_OPENAI])
    return adapter, gate


def _notifications(result: Any) -> str:
    return " ".join(
        str(event.get("content", ""))
        for event in result.events
        if event.get("type") == "notification"
    )


@pytest.mark.parametrize(
    ("blocked", "allowed"),
    [
        ("http://169.254.169.254/latest/meta-data/iam/security-credentials/",
         "https://cdn.example.com/keep-a.png"),
        ("https://10.0.0.1/admin/backup.png", "https://cdn.example.org/keep-b.png"),
    ],
)
@pytest.mark.asyncio
async def test_a_reference_the_chat_carries_meets_the_gate_the_typed_box_meets(blocked, allowed):
    """The same address was refused through the reference box and forwarded through a message.

    `_to_input_image` leaves a remote image URL in the conversation verbatim whenever its
    download or its storage upload fails, and every prior image in the chat becomes an
    `input_references` entry. OpenRouter fetches whatever is listed, so the unchecked
    route turned a link chosen by whoever could put one in the history into a request the
    deployment makes on its behalf -- the instance metadata endpoint among them.

    Two addresses and a surviving sibling each time, so neither "drop everything" nor
    "drop nothing" passes, and a constant verdict satisfies at most one row.
    """
    adapter, gate = _gated_adapter({allowed})

    result = await _posted(
        adapter,
        body={},
        responses_body=_turn_carrying([blocked, allowed]),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        metadata={"chat_id": "c"},
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    assert gate.seen == [blocked, allowed], "every reference OpenRouter would fetch is asked about"
    assert [ref["image_url"]["url"] for ref in result.payload["input_references"]] == [allowed]
    assert blocked not in json.dumps(result.payload)
    assert "dropped 1 reference image" in _notifications(result), (
        f"a reference that was not sent has to say so; the user saw {_notifications(result)!r}"
    )


@pytest.mark.parametrize("carried", [1, 3])
@pytest.mark.asyncio
async def test_an_inline_reference_is_never_put_to_the_gate(carried):
    """A data: URI is the payload itself, so there is no address to resolve and no fetch.

    The gate here refuses everything it is asked about: routing inlined attachments
    through it would empty `input_references` on every ordinary request, which is what
    every image edit in the product is made of.
    """
    adapter, gate = _gated_adapter()

    result = await _posted(
        adapter,
        body={},
        responses_body=_user_turn_with_images(carried),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        metadata={"chat_id": "c"},
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    assert gate.seen == [], "inline content is not a fetch and must not be resolved"
    assert len(result.payload["input_references"]) == carried


@pytest.mark.parametrize("repeats", [2, 4])
@pytest.mark.asyncio
async def test_one_address_is_resolved_once_however_many_turns_repeat_it(repeats):
    """Each check is a DNS resolution on a worker thread, and a long chat repeats an image.

    Resolving per occurrence also lets two occurrences of one address disagree when the
    record changes mid-request, which is a reference sent on the strength of a lookup that
    a later lookup contradicted.
    """
    link = "https://cdn.example.com/same.png"
    adapter, gate = _gated_adapter({link})

    result = await _posted(
        adapter,
        body={},
        responses_body=_turn_carrying([link] * repeats),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        metadata={"chat_id": "c"},
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    assert gate.seen == [link]
    assert len(result.payload["input_references"]) == repeats


@pytest.mark.parametrize("route", ["typed-into-the-box", "already-in-the-chat"])
@pytest.mark.parametrize(
    "blocked", ["http://169.254.169.254/latest/meta-data/", "https://192.168.0.5/x.png"]
)
@pytest.mark.asyncio
async def test_no_route_puts_a_refused_address_on_the_wire(route, blocked):
    """The property, stated over the routes rather than over one of them.

    Both rows use one gate and one address; only the door changes. A guard added to
    whichever door the reporter happened to try leaves the other one open.
    """
    import aiohttp
    from aioresponses import aioresponses

    adapter, gate = _gated_adapter()
    typed = route == "typed-into-the-box"
    metadata: dict[str, Any] = {"chat_id": "c"}
    if typed:
        metadata[PIPE_META] = {"image_generation": {"reference_urls": [blocked]}}

    with aioresponses() as mocked:
        mocked.post(f"{BASE}/images", payload={"data": []})
        async with aiohttp.ClientSession() as session:
            await adapter.generate(
                body={},
                responses_body=_turn_carrying([] if typed else [blocked]),
                valves=_StubValves("sk-x"),
                session=session,
                event_emitter=_Emitter(),
                metadata=metadata,
                user=None,
                request=object(),
                user_obj=object(),
                normalized_model_id="m.x",
                api_model_id="m/x",
            )
        sent = [
            json.dumps(call.kwargs.get("json"))
            for key, calls in mocked.requests.items()
            if key[1].path == "/api/v1/images"
            for call in calls
        ]

    assert gate.seen == [blocked], f"{route}: the gate was never asked about it"
    assert not any(blocked in body for body in sent), f"{route} put it on the wire: {sent}"


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
    spec = build_image_model_filter_spec(raw["id"], None, raw["endpoints"], dedicated_image_api=True)
    module = _load_filter_from_source(
        render_image_model_filter_source(spec), f"always_on_{spec.dotted_id}"
    )
    assert ALWAYS_ON_VALVE_NAMES <= set(module.Filter.UserValves.model_fields), raw["id"]


@pytest.mark.parametrize("published", ["provider_options_json", "reference_urls"])
def test_a_published_name_cannot_take_over_a_reserved_control(published):
    """The field name decides three things that must agree, and two fields sharing one
    leaves the last definition standing with no warning."""
    record = {**_RECRAFT, "allowed_passthrough_parameters": [published]}
    spec = build_image_model_filter_spec("m/x", None, [record], dedicated_image_api=True)
    assert published not in spec.passthrough

    module = _load_filter_from_source(
        render_image_model_filter_source(spec), f"reserved_{published}"
    )
    field = module.Filter.UserValves.model_fields[f"IMAGE_{published.upper()}"]
    assert field.title in {"Provider options", "Reference image links"}, (
        "the surviving field must be the reserved control, not a free-text box wearing "
        f"its name. got title={field.title!r}"
    )


@pytest.mark.parametrize(
    "unrenderable",
    [
        pytest.param(float("inf"), id="infinity"),
        pytest.param(float("nan"), id="not-a-number"),
        pytest.param(10**5000, id="past-the-digit-limit"),
    ],
)
@pytest.mark.parametrize(("low", "high"), [(1, 4), (2, 9)])
def test_a_bound_the_pipe_cannot_write_as_a_literal_draws_no_control(unrenderable, low, high):
    """A bound that cannot become a Python literal must read as absent, not raise.

    `json.loads` decodes `Infinity`, `NaN` and integers past the interpreter's own
    digit limit, and every one of them reached an f-string that writes `ge=`/`le=`
    into the generated source. The model then lost its whole filter, and `help` for
    it raised on a path with no handler above it.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
        render_image_model_filter_source,
    )

    def _spec(maximum):
        return build_image_model_filter_spec(
            "vendor/model",
            {"id": "vendor/model", "name": "Vendor: Model",
             "architecture": {"output_modalities": ["image"]}},
            [{"provider_slug": "p",
              "supported_parameters": {"n": {"type": "range", "min": low, "max": maximum}}}],
            dedicated_image_api=True,
        )

    good = render_image_model_filter_source(_spec(high))
    assert f"le={high}" in good, "a readable bound must still draw its control"
    assert ("IMAGE_N:" in good) is True

    rendered = render_image_model_filter_source(_spec(unrenderable))
    assert "IMAGE_N:" not in rendered, (
        "a bound the pipe cannot write must draw no control rather than an invented one"
    )
    compile(rendered, "<generated-filter>", "exec")


def test_the_server_tool_draws_one_control_per_parameter():
    """No parameter may be drawn twice by the generated server-tool filter.

    `moderation` is written into the template directly, with the values OpenRouter
    publishes for it, and was also reaching the contract-driven loop, so the filter
    carried two controls writing the same request key -- one of them free text
    captioned as though nothing were published.
    """
    import re

    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        ImageModelFilterSpec,
        render_image_gen_filter_source,
    )

    spec = ImageModelFilterSpec(
        model_id="vendor/model",
        display_name="Vendor: Model",
        function_id="openrouter_image_gen",
        marker="openrouter_image_gen:v1",
        contract_read=True,
    )
    source = render_image_gen_filter_source(spec, catalog_match=True, selected_model="vendor/model")
    written = re.findall(r"params\[[\"']([a-z_]+)[\"']\]\s*=", source)
    assert written, "the generated filter writes no request keys at all"
    assert len(written) == len(set(written)), (
        f"one request key is written by two controls: {sorted(written)}"
    )


async def _installed_user_valves(model_id: str, contract: list[dict[str, Any]]):
    """Render one model's filter the way the pipe installs it, and hand back its fields.

    The catalog row is registered through the same two entry points production uses --
    a text+image row arrives on `/models` through `ensure_loaded`, an image-only row
    through `register_image_models` -- so `image_model` is whatever the registry really
    holds for it, not a dict the test invented. The only thing stubbed is
    `_ensure_filter_installed`, one seam below the installer, so the source asserted on
    is the source the installer produced.
    """
    import logging
    from unittest.mock import AsyncMock, MagicMock

    import aiohttp
    from aioresponses import aioresponses

    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

    catalog = json.loads(
        (Path(__file__).parent / "fixtures" / "openrouter_image_models.json").read_text()
    )["data"]
    row = next(item for item in catalog if item["id"] == model_id)
    answers_with_text = "text" in ((row.get("architecture") or {}).get("output_modalities") or [])

    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry._last_fetch = 0.0
    OpenRouterModelRegistry._next_refresh_after = 0.0
    OpenRouterModelRegistry.set_image_endpoints({model_id: contract})
    if answers_with_text:
        with aioresponses() as http:
            http.get(
                "https://openrouter.ai/api/v1/models", payload={"data": [row]}, repeat=True
            )
            http.get(
                "https://openrouter.ai/api/v1/endpoints/zdr", payload={"data": []}, repeat=True
            )
            async with aiohttp.ClientSession() as session:
                await OpenRouterModelRegistry.ensure_loaded(
                    session,
                    base_url="https://openrouter.ai/api/v1",
                    api_key="sk-x",
                    cache_seconds=3600,
                    logger=logging.getLogger("test"),
                )
    else:
        OpenRouterModelRegistry.register_image_models([row])

    pipe = MagicMock()
    pipe.valves.AUTO_INSTALL_IMAGE_FILTERS = True
    pipe.valves.ENABLE_OPENROUTER_IMAGE_GENERATION = True
    manager = FilterManager(pipe=pipe, valves=pipe.valves, logger=MagicMock())
    rendered: dict[str, str] = {}

    async def _capture(**kwargs):
        rendered["source"] = kwargs["desired_source"]
        return kwargs["preferred_id"]

    manager._ensure_filter_installed = AsyncMock(side_effect=_capture)
    manager._retire_variant_image_filters = AsyncMock(return_value=None)
    manager._image_filter_exists = AsyncMock(return_value=False)

    await manager.ensure_openrouter_image_filter_function_ids(
        OpenRouterModelRegistry.list_models()
    )
    assert rendered.get("source"), f"the installer rendered no filter for {model_id}"
    module = _load_filter_from_source(
        rendered["source"], f"transport_{model_id.replace('/', '_').replace('.', '_')}"
    )
    return set(module.Filter.UserValves().model_fields_set) | set(
        module.Filter.UserValves.model_fields
    )


@pytest.mark.parametrize(
    ("model_id", "fixture", "transport"),
    [
        ("recraft/recraft-v3", "recraft_recraft-v3", "image"),
        ("google/gemini-3-pro-image", "google_gemini-3-pro-image", "chat"),
    ],
)
@pytest.mark.asyncio
async def test_no_control_is_drawn_that_its_own_transport_would_throw_away(
    model_id, fixture, transport
):
    """A control is drawn only where the request format that carries it has a home for it.

    `provider.options` is defined by the image request format and by nothing in
    `ProviderPreferences`, and reference mode and reference links are read only on the
    image path, so on a model that answers with text as well as a picture those three
    boxes were drawn, filled in, and then emptied on the way out. The rows are two real
    catalog models whose transports differ, registered through the registry, because the
    thin dict production actually hands the renderer carries no `architecture` at all --
    which is exactly why a test that built one by hand could not see this.
    """
    from open_webui_openrouter_pipe.integrations.provider_options import (
        TRANSPORT_PROVIDER_KEYS,
    )

    contract = json.loads(
        (Path(__file__).parent / "fixtures" / f"openrouter_image_endpoints_{fixture}.json").read_text()
    )["endpoints"]
    fields = await _installed_user_valves(model_id, contract)

    accepted = TRANSPORT_PROVIDER_KEYS[transport]
    assert ("IMAGE_PROVIDER_OPTIONS_JSON" in fields) == ("options" in accepted), (
        f"{model_id}: the panel draws the provider options box "
        f"{'IMAGE_PROVIDER_OPTIONS_JSON' in fields!r} while its transport accepts it "
        f"{('options' in accepted)!r}"
    )
    image_only = {"IMAGE_REFERENCE_MODE", "IMAGE_REFERENCE_URLS"}
    assert (image_only <= fields) == (transport == "image"), (
        f"{model_id}: the reference controls are read only on the image API, and the "
        f"panel drew {sorted(image_only & fields)} on the {transport} route"
    )
    assert "IMAGE_ASPECT_RATIO" in fields, (
        "the model's own published controls must be drawn on both transports"
    )


@pytest.mark.parametrize("accepting_endpoint_first", [True, False])
@pytest.mark.asyncio
async def test_a_value_one_provider_alone_accepts_is_pinned_to_a_provider_that_accepts_it(
    accepting_endpoint_first,
):
    """Whoever serves the request accepts every value in it, or the value was dropped.

    With no pin the validation set collapsed to whichever endpoint the catalog listed
    first, so the same request either sent `4K` unpinned -- free to be routed to the
    company that does not publish it -- or dropped it and blamed the model, purely on
    record order. Both orders are asserted, so a fix that only repairs one cannot pass.
    """
    def _publishes_4k(record):
        published = (record.get("supported_parameters") or {}).get("resolution") or {}
        return "4K" in (published.get("values") or ())

    records = sorted(_gemini_records(), key=_publishes_4k, reverse=accepting_endpoint_first)
    assert _publishes_4k(records[0]) is accepting_endpoint_first, (
        "the two orders this case is about were not produced"
    )
    accepting = sorted(str(record.get("provider_tag")) for record in records if _publishes_4k(record))
    assert 0 < len(accepting) < len(records), (
        "the fixture no longer has a value only some of its endpoints publish, so this "
        "case asserts nothing"
    )

    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), records)
    result = await _posted(
        adapter,
        body={"image_config": {"resolution": "4K"}},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    notices = " ".join(str(call.get("content", "")) for call in result.events)
    assert result.payload.get("resolution") == "4K", (
        "the panel offers 4K because one of this model's companies publishes it, so the "
        f"request must carry it; instead it was dropped with {notices!r}"
    )
    assert result.payload.get("provider", {}).get("only") == accepting, (
        "4K was sent unpinned, so OpenRouter may route it to the company that does not "
        f"publish it; payload provider={result.payload.get('provider')!r}"
    )


@pytest.mark.parametrize(
    ("tag", "expected"),
    [("northstar/eu", ["northstar/eu"]), ("northstar", ["northstar"])],
)
@pytest.mark.asyncio
async def test_the_image_path_pins_the_tag_and_keys_its_options_by_the_slug(tag, expected):
    """The two fields do different jobs and the pipe read one of them for both.

    OpenRouter documents `provider_slug` for `provider.options[slug]` and `provider_tag`
    for pinning, and the image request format enumerates about a hundred and twenty bare
    option slugs and no slashed ones -- so a block keyed by the tag would be silently
    dropped on the eight recorded endpoints whose slug carries a region. Both halves are
    asserted together because reading one field for both jobs passes either half alone.
    """
    records = [
        {
            "provider_slug": "southstar",
            "provider_tag": "southstar/us",
            "supported_parameters": {"resolution": {"type": "enum", "values": ["1K", "2K"]}},
            "allowed_passthrough_parameters": ["style"],
        },
        {
            "provider_slug": "northstar",
            "provider_tag": tag,
            "supported_parameters": {
                "resolution": {"type": "enum", "values": ["1K", "2K", "4K"]}
            },
            "allowed_passthrough_parameters": ["style"],
        },
    ]
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), records)
    result = await _posted(
        adapter,
        body={"image_config": {"resolution": "4K", "style": "realistic_image"}},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    provider = result.payload.get("provider") or {}
    assert provider.get("only") == expected, (
        f"the pin named {provider.get('only')!r}; `provider_tag` is the field OpenRouter "
        "documents for pinning a request to a provider"
    )
    assert set(provider.get("options") or {}) == {"southstar", "northstar"}, (
        f"the option block is keyed {sorted(provider.get('options') or {})}; the image "
        "request format enumerates bare slugs only and drops keys it does not recognise"
    )


@pytest.mark.parametrize(
    "inline",
    [
        "data:image/png;base64,iVBORw0KGgo=",
        "data:image/webp;base64,UklGRg==",
    ],
)
@pytest.mark.asyncio
async def test_a_reference_the_user_pasted_inline_is_sent_without_being_asked_of_the_gate(
    inline,
):
    """A `data:` reference names no host, so there is nothing for the SSRF gate to resolve.

    Asking about it anyway is a question with no right answer: the gate resolves host
    names, so a `data:` URL either fails to parse or is refused, and either way the
    picture the user pasted into the panel silently stops working. Only the http arm of
    this branch was ever driven, so the `data:` arm could be deleted and nothing said.
    """
    handler = _RefuseAll()
    pipe = _KeyPipe("sk-x")
    cast(Any, pipe)._multimodal_handler = handler
    adapter = _adapter(pipe)
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [_RECRAFT])

    result = await _posted(
        adapter,
        body={},
        responses_body=_user_turn_with_images(0),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        metadata={"chat_id": "c", PIPE_META: {"image_generation": {"reference_urls": [inline]}}},
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    assert handler.seen == [], (
        f"a pasted picture was put through the host gate, which refuses it: {handler.seen}"
    )
    assert [ref["image_url"]["url"] for ref in result.payload["input_references"]] == [inline]
