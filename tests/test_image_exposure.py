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
import re
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
    _event_data,
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

    notices = " ".join(str(_event_data(call).get("content", "")) for call in result.events)
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
    refused = "4K"
    top_level, _provider, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"resolution": tier, "size": refused}},
        allowed_passthrough=frozenset(record.get("allowed_passthrough_parameters") or []),
        record=record,
    )

    assert top_level == {kept: tier}, (
        f"the validated tier was thrown away for an unchecked one: {top_level!r}"
    )
    reported = " ".join(note.text for note in notes)
    assert re.search(rf"\bsize\b\)?={refused!r}", reported) and "resolution" in reported, (
        f"the setting that was not sent has to be named as the one that was not sent, "
        f"with the value it was refused for: {reported!r}"
    )
    assert not re.search(rf"\bresolution\b\)?={tier!r}", reported), (
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
    async def _is_safe_url(self, url: str, *, seconds: float = 5.0) -> bool:
        assert seconds > 0, "the address check was handed no time at all"
        return True


class _RefuseAll:
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def _is_safe_url(self, url: str, *, seconds: float = 5.0) -> bool:
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

    async def _is_safe_url(self, url: str, *, seconds: float = 5.0) -> bool:
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
        str(_event_data(event).get("content", ""))
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

    notices = " ".join(str(_event_data(call).get("content", "")) for call in result.events)
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


_NAMES_A_USER_SEES = {
    "n": {"type": "range", "min": 1, "max": 4},
    "aspect_ratio": {"type": "enum", "values": ["1:1"]},
}

_REFUSED_BY_THAT_CONTRACT = {"n": "lots", "aspect_ratio": "16:9"}


def _rendered_title(source: str, valve: str) -> str:
    block = source.split(f"        {valve}: ", 1)[1].split("\n        )", 1)[0]
    found = re.search(r"^\s+title=(?P<q>['\"])(?P<title>.*?)(?P=q),$", block, re.M)
    assert found, f"{valve} is drawn with no title at all:\n{block}"
    return found.group("title")


@pytest.mark.parametrize("published", sorted(_NAMES_A_USER_SEES), ids=sorted(_NAMES_A_USER_SEES))
def test_a_dropped_setting_is_named_the_way_its_own_control_is_labelled(published):
    """The toast has to name the control the reader used, not the wire name behind it.

    Someone who set a control labelled "Number of images" and was told `n` was not sent
    has been handed a name that appears nowhere on their screen; the documentation
    promises them the label. The wire name is kept alongside because the log record and
    any upstream rejection carry only that, and a message with one of the two makes the
    other unsearchable.

    The expected label is read out of the RENDERED filter -- the panel Open WebUI draws --
    rather than transcribed here, so renaming a control moves the panel and this guard in
    one commit instead of leaving a guard that passes against a label nobody sees.

    Two controls, because `aspect_ratio` -> "Aspect ratio" is what a hand-rolled
    underscore-to-space rule would also produce; `n` -> "Number of images" is not, so no
    such rule and no constant satisfies both rows.
    """
    record = {
        "provider_slug": "vendor",
        "provider_tag": "vendor",
        "allowed_passthrough_parameters": [],
        "supported_parameters": dict(_NAMES_A_USER_SEES),
    }
    spec = build_image_model_filter_spec(
        "vendor/model", {"id": "vendor/model"}, [record], dedicated_image_api=True
    )
    label = _rendered_title(render_image_model_filter_source(spec), f"IMAGE_{published.upper()}")
    assert label != published, (
        f"{published} is drawn under its own wire name, so this row proves nothing about "
        "which of the two the message carries"
    )

    _top_level, _provider, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {published: _REFUSED_BY_THAT_CONTRACT[published]}},
        allowed_passthrough=frozenset(),
        record=record,
        records=[record],
    )
    reported = " ".join(note.text for note in notes)

    assert reported, (
        f"{published} was refused by the contract and nothing was reported: notes={notes!r}"
    )
    assert label in reported, (
        f"the message names {published!r} but the control the reader used is labelled "
        f"{label!r}, which appears nowhere in it: {reported!r}"
    )
    assert re.search(rf"\b{re.escape(published)}\b", reported), (
        f"the wire name is gone, so the log record and the upstream rejection cannot be "
        f"matched to this message: {reported!r}"
    )


# =============================================================================
# 9 - the size box, and the reason a panel gives for being empty
# =============================================================================


def _tier_records(published: dict[str, tuple[str, ...]]) -> list[dict[str, Any]]:
    """One endpoint record per company, each publishing the tiers it was given."""
    return [
        {
            "provider_slug": slug,
            "provider_tag": slug,
            "supported_parameters": (
                {"resolution": {"type": "enum", "values": list(values)}} if values else {}
            ),
        }
        for slug, values in published.items()
    ]


def _empty_list_records(slugs: tuple[str, ...]) -> list[dict[str, Any]]:
    """A company publishing a tier list with nothing in it, which is not a tier list."""
    return [
        {
            "provider_slug": slug,
            "provider_tag": slug,
            "supported_parameters": {"resolution": {"type": "enum", "values": []}},
        }
        for slug in slugs
    ]


_TIER_CONTRACTS: dict[str, list[dict[str, Any]]] = {
    "one-list-they-all-share": _tier_records({"alpha": ("1K", "2K"), "beta": ("1K", "2K", "4K")}),
    "no-list-they-all-share": _tier_records({"alpha": ("1K", "2K"), "beta": ("4K",)}),
    "no-list-at-all": _tier_records({"alpha": ()}),
    "one-company-with-a-list": _tier_records({"alpha": ("1K", "2K")}),
    "one-list-with-nothing-in-it": _empty_list_records(("alpha",)),
    "two-lists-with-nothing-in-them": _empty_list_records(("alpha", "beta")),
}


def _routing_clauses() -> set[str]:
    """Every sentence the panel uses for tiers split across companies, as shipped.

    Taken from the meaning table by the state it is keyed under, so renaming the strings
    or reordering the table leaves this reading the same sentences.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import _SIZE_MEANING

    return {
        clause
        for (state, _ratio), (clause, _pixels) in _SIZE_MEANING.items()
        if state == "split"
    }


def _tiers_one_company_admits(record: dict[str, Any]) -> tuple[str, ...]:
    """The tiers this company's own published list lets through, measured by the adapter.

    A tier that survives without being measured against anything is not evidence of a
    published list, so the fit has to report both: a value that came back and the name it
    was measured as.
    """
    from open_webui_openrouter_pipe.integrations.image_types import SCHEMA_ENUMS, TIER_EQUIVALENT

    declared = record.get("supported_parameters") or {}
    admitted = []
    for tier in SCHEMA_ENUMS[TIER_EQUIVALENT["size"]]:
        fitted = ImageGenerationAdapter._fit_published(declared, "size", tier)
        if fitted.value is not None and fitted.measured_as:
            admitted.append(tier)
    return tuple(admitted)


@pytest.mark.parametrize("case", sorted(_TIER_CONTRACTS))
def test_the_size_box_describes_the_check_the_request_will_actually_make(case):
    """The box said a tier goes out unchecked while the request checked it and dropped it.

    The panel picked its sentence from the tiers every company shares and the request
    measures against the tiers any of them publishes. Those differ the moment two
    companies publish disjoint lists, and the reader was told 512 would be sent for a
    model that refuses it. The expectation here is read off the request path itself so it
    cannot be restated wrongly, and the contracts disagree about it, so no single sentence
    satisfies them all. A list with nothing in it is published and admits nothing, which
    is why the count is taken from what a tier gets through rather than from what a record
    carries -- and why the sentence promising the tier travels is owed to the request
    letting one through unmeasured rather than to no company publishing a list, which is
    also true of a company publishing one that admits nothing.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        IMAGE_KNOB_TITLES,
        image_knob_text,
        renders_control,
    )

    records = _TIER_CONTRACTS[case]
    spec = build_image_model_filter_spec("v/m", {"id": "v/m"}, records, dedicated_image_api=True)

    admitted = {
        str(record.get("provider_tag")): _tiers_one_company_admits(record) for record in records
    }
    publishing = [company for company, tiers in admitted.items() if tiers]

    assert spec.publishes_size_tiers is (len(publishing) > 1), (
        f"{case}: the spec says publishes_size_tiers={spec.publishes_size_tiers} while the "
        f"request path lets a tier through the published list of {publishing}: {admitted}"
    )

    _title, description = image_knob_text("size", spec)
    says_unchecked = any(clause in description for clause in _passage_clauses())
    _carried, unmeasured = _tiers_the_request_carries(records)
    assert says_unchecked is bool(unmeasured), (
        f"{case}: the box says the tier is checked against nothing but OpenRouter's four "
        f"names and then travels, and the request path lets {list(unmeasured)} of the four "
        f"through without measuring it against anything {publishing or 'this model'} "
        f"published: {description!r}"
    )
    tiered_title = IMAGE_KNOB_TITLES["resolution"][0]
    assert (tiered_title in description) is renders_control(spec, "resolution"), (
        f"{case}: the box names a {tiered_title} control this panel does not draw: "
        f"{description!r}"
    )


@pytest.mark.parametrize("case", sorted(_TIER_CONTRACTS))
def test_the_size_box_promises_routing_only_where_two_companies_publish_a_list(case):
    """The split wording promises a check across companies and a request routed to one.

    It was chosen for any record carrying anything under size or resolution, so a single
    company publishing a list nothing can satisfy -- an empty one, or a bound measured as
    a number -- got a sentence promising the request would be steered to a company that
    takes the tier, with no second company to steer it to. Whether the promise is true is
    read off the request path per company, and the drawn field decides whether the box is
    the one making it, so neither half of the expectation is restated here.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        _valve_name,
        image_knob_text,
    )
    from open_webui_openrouter_pipe.integrations.image_types import TIER_EQUIVALENT

    records = _TIER_CONTRACTS[case]
    spec = build_image_model_filter_spec("v/m", {"id": "v/m"}, records, dedicated_image_api=True)
    publishing = [record for record in records if _tiers_one_company_admits(record)]

    drawn = set(re.findall(r"^ {8}(IMAGE_[A-Z0-9_]+):", render_image_model_filter_source(spec), re.M))
    tiers_have_their_own_control = _valve_name(TIER_EQUIVALENT["size"]) in drawn

    _title, description = image_knob_text("size", spec)
    promises_routing = any(clause in description for clause in _routing_clauses())

    assert promises_routing is (len(publishing) > 1 and not tiers_have_their_own_control), (
        f"{case}: the box promises the request is steered to a company that takes the "
        f"tier, and {len(publishing)} of {len(records)} companies publish a list a tier "
        f"gets through (a control of their own is drawn: {tiers_have_their_own_control}): "
        f"{description!r}"
    )


@pytest.mark.parametrize(("tier", "sent"), [("4K", True), ("512", False)])
@pytest.mark.asyncio
async def test_a_tier_only_one_company_publishes_travels_as_the_size_box_promises(tier, sent):
    """The promise the split wording makes, put on the wire.

    ``4K`` is published by one of the two companies and ``512`` by neither. The box says a
    tier is checked against every tier they publish between them and that the request goes
    to a company that takes it, so one of these must be sent and pinned and the other
    refused. Both rows are asserted, so a request path that sent everything or nothing
    cannot pass.
    """
    records = _TIER_CONTRACTS["no-list-they-all-share"]
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), records)

    result = await _posted(
        adapter,
        body={"image_config": {"size": tier}},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    notices = " ".join(str(_event_data(call).get("content", "")) for call in result.events)
    assert (result.payload.get("size") == tier) is sent, (
        f"size={tier!r} was {'dropped' if sent else 'sent'} against the box's promise; "
        f"payload={result.payload!r} notices={notices!r}"
    )
    if sent:
        publishing = sorted(
            str(record["provider_tag"])
            for record in records
            if tier in ((record["supported_parameters"].get("resolution") or {}).get("values") or [])
        )
        assert result.payload.get("provider", {}).get("only") == publishing, (
            f"{tier!r} went out unpinned, so OpenRouter may route it to a company that does "
            f"not publish it; provider={result.payload.get('provider')!r}"
        )


_EMPTY_PANEL_CONTRACTS: dict[str, list[dict[str, Any]]] = {
    "same-setting-one-company-unnamed": [
        {"provider_slug": "acme", "provider_tag": "acme", "allowed_passthrough_parameters": ["steps"]},
        {"provider_tag": "brand", "allowed_passthrough_parameters": ["steps"]},
    ],
    "different-settings-both-companies-named": [
        {"provider_slug": "acme", "provider_tag": "acme", "allowed_passthrough_parameters": ["steps"]},
        {"provider_slug": "brand", "provider_tag": "brand", "allowed_passthrough_parameters": ["cfg"]},
    ],
}


_DISAGREEMENT_CONTRACTS: dict[str, list[dict[str, Any]]] = {
    "the-same-setting-the-panel-already-carries": [
        {
            "provider_slug": slug,
            "provider_tag": slug,
            "allowed_passthrough_parameters": ["reference_mode"],
        }
        for slug in ("acme", "brand")
    ],
    "settings-with-different-names": [
        {"provider_slug": "acme", "provider_tag": "acme", "allowed_passthrough_parameters": ["steps"]},
        {"provider_slug": "brand", "provider_tag": "brand", "allowed_passthrough_parameters": ["cfg"]},
    ],
    "one-setting-with-values-they-do-not-share": [
        {
            "provider_slug": slug,
            "provider_tag": slug,
            "supported_parameters": {"quality": {"type": "enum", "values": [value]}},
            "allowed_passthrough_parameters": [],
        }
        for slug, value in (("acme", "low"), ("brand", "high"))
    ],
}


@pytest.mark.parametrize("case", sorted(_DISAGREEMENT_CONTRACTS))
def test_companies_are_called_disagreeing_only_where_their_own_panels_differ(case):
    """A setting the panel already carries was counted as a disagreement about it.

    Whether a company's setting can be offered is one rule, and it was written twice: the
    copy driving this flag left out the check against names the panel supplies itself, so
    two companies publishing the identical setting were reported as wanting different
    things. The expectation is the filter each company would get on its own, rendered and
    compared -- a company that would be given the same panel is not disagreeing with
    anyone. One contract expects no disagreement and two expect one, and one of those two
    publishes the same name with values they do not share, which is a real disagreement
    that no comparison of names alone would catch.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        DISAGREED_SETTINGS,
        image_gen_model_note,
    )

    records = _DISAGREEMENT_CONTRACTS[case]
    spec = build_image_model_filter_spec("v/m", {"id": "v/m"}, records, dedicated_image_api=True)
    assert spec.knob_count == 0, f"{case} was meant to draw nothing, got {spec.knob_count}"

    alone = {
        render_image_model_filter_source(
            build_image_model_filter_spec("v/m", {"id": "v/m"}, [record], dedicated_image_api=True)
        )
        for record in records
    }
    would_be_given_different_panels = len(alone) > 1

    assert spec.providers_disagree is would_be_given_different_panels, (
        f"{case}: providers_disagree={spec.providers_disagree} while the companies would "
        f"be given {len(alone)} distinct panel(s) of their own"
    )
    note = image_gen_model_note(spec, catalog_match=True)
    blames_a_difference = DISAGREED_SETTINGS.format(named=spec.model_id) in note
    assert blames_a_difference is would_be_given_different_panels, (
        f"{case}: the note sends the reader looking for a difference between panels that "
        f"are the same: {note!r}"
    )


@pytest.mark.parametrize("case", sorted(_EMPTY_PANEL_CONTRACTS))
def test_an_empty_panel_blames_disagreement_only_where_the_companies_disagree(case):
    """Two companies advertising the identical setting were reported as disagreeing.

    The shared set is emptied by a record with no provider slug just as it is by a record
    naming something else, because a setting cannot be keyed to a company OpenRouter does
    not name. Only the second is a disagreement, and telling a reader the companies want
    different things sends them looking for a difference that is not there. The two
    contracts here expect opposite sentences on both surfaces, so neither sentence can be
    emitted unconditionally. Both sentences are imported from production rather than
    reduced to a fragment of themselves ("different settings", "does not name one of
    them"), so rewording either one moves the expectation instead of reddening the suite.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        DISAGREED_SETTINGS,
        UNKEYABLE_SETTINGS,
        _renderable_names,
        image_gen_model_note,
        named_settings,
    )
    from open_webui_openrouter_pipe.integrations.image_help import render_image_help

    records = _EMPTY_PANEL_CONTRACTS[case]
    spec = build_image_model_filter_spec("v/m", {"id": "v/m"}, records, dedicated_image_api=True)
    assert spec.knob_count == 0, f"{case} was meant to draw nothing, got {spec.knob_count}"

    apart_if_all_were_named = [
        _renderable_names([{**record, "provider_slug": record.get("provider_slug") or "named"}])
        for record in records
    ]
    would_render_differently = len({frozenset(names) for names in apart_if_all_were_named}) > 1

    assert spec.providers_disagree is would_render_differently, (
        f"{case}: providers_disagree={spec.providers_disagree} while the records would "
        f"render {apart_if_all_were_named} apart once every company is named"
    )
    assert spec.passthrough_unaddressable is not would_render_differently, (
        f"{case}: passthrough_unaddressable={spec.passthrough_unaddressable} does not match "
        f"what the records render apart: {apart_if_all_were_named}"
    )

    help_text = render_image_help("v/m", {"id": "v/m"}, endpoint_record=records, dedicated_image_api=True)
    note = image_gen_model_note(spec, catalog_match=True)
    listed = named_settings(spec)
    for surface, text, named in (("help", help_text, "this model"), ("model note", note, spec.model_id)):
        blame = DISAGREED_SETTINGS.format(named=named)
        unnamed = UNKEYABLE_SETTINGS.format(named=named, listed=listed)
        assert (blame in text) is would_render_differently, (
            f"{case}: the {surface} blames a disagreement that is not there: {text!r}"
        )
        assert (unnamed in text) is not would_render_differently, (
            f"{case}: the {surface} blames an unnamed company wrongly: {text!r}"
        )


_WITHHELD_CONTRACTS: dict[str, tuple[str, ...]] = {
    "one-setting": ("steps",),
    "two-settings": ("steps", "cfg_scale"),
}


@pytest.mark.parametrize("case", sorted(_WITHHELD_CONTRACTS))
def test_every_setting_withheld_for_want_of_a_named_company_is_named(case):
    """Two withheld settings were reported as one, and neither surface said which.

    Both messages were written for a single setting, so a contract sharing two told the
    reader one was missing and left them to guess which control they were looking for.
    The names are the ones the contract published, and the two rows publish different
    numbers of them, so no fixed sentence can name every one of both.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import image_gen_model_note
    from open_webui_openrouter_pipe.integrations.image_help import render_image_help

    published = _WITHHELD_CONTRACTS[case]
    records: list[dict[str, Any]] = [
        {"provider_tag": tag, "allowed_passthrough_parameters": list(published)}
        for tag in ("acme", "brand")
    ]
    spec = build_image_model_filter_spec("v/m", {"id": "v/m"}, records, dedicated_image_api=True)
    assert spec.passthrough_unaddressable and not spec.passthrough, (
        f"{case} was meant to reach the withheld branch: passthrough={spec.passthrough}"
    )

    surfaces = {
        "model note": image_gen_model_note(spec, catalog_match=True),
        "help card": render_image_help(
            "v/m", {"id": "v/m"}, endpoint_record=records, dedicated_image_api=True
        ),
    }
    vocabulary = {name for names in _WITHHELD_CONTRACTS.values() for name in names}
    for surface, text in surfaces.items():
        assert {name for name in vocabulary if name in text} == set(published), (
            f"{case}: the {surface} withholds {published} and names "
            f"{sorted(name for name in vocabulary if name in text)}, so a reader cannot "
            f"tell which control is missing: {text!r}"
        )


_ROUTED_CARD_CONTRACTS: dict[str, list[dict[str, Any]]] = {
    "companies-with-different-settings": [
        {
            "provider_slug": slug,
            "provider_tag": slug,
            "supported_parameters": {
                "resolution": {"type": "enum", "values": [tier]},
                "quality": {"type": "enum", "values": [quality]},
            },
        }
        for slug, tier, quality in (("alpha", "1K", "low"), ("beta", "4K", "high"))
    ],
    "companies-that-publish-nothing": [
        {"provider_slug": slug, "provider_tag": slug, "supported_parameters": {}}
        for slug in ("alpha", "beta")
    ],
}


@pytest.mark.parametrize("case", sorted(_ROUTED_CARD_CONTRACTS))
def test_a_card_never_both_routes_a_size_and_says_routing_is_impossible(case):
    """One card gave two accounts of the same mechanism, and one of them was false.

    The disagreed settings were said to be unofferable "without knowing which one will
    take the request", while the size bullet below promised the request goes to a company
    that takes the tier -- and the adapter does pin one, for any top-level value, not only
    a size. Whether it pins is measured here rather than assumed, and the account of the
    disagreement is read from the string both surfaces are built from, so restoring an
    impossibility claim is a divergence rather than a wording change. One contract routes
    and blames a disagreement and the other does neither, so a fixed card fails a row.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        DISAGREED_SETTINGS,
        image_gen_model_note,
    )
    from open_webui_openrouter_pipe.integrations.image_help import render_image_help

    records = _ROUTED_CARD_CONTRACTS[case]
    spec = build_image_model_filter_spec("v/m", {"id": "v/m"}, records, dedicated_image_api=True)
    assert not spec.knob_count, f"{case} was meant to draw nothing, got {spec.knob_count}"

    provider: dict[str, Any] = {}
    ImageGenerationAdapter._pin_accepting_providers(provider, records, {"size": "4K"})
    takes_it = sorted(
        str(record["provider_tag"])
        for record in records
        if "4K" in ((record["supported_parameters"].get("resolution") or {}).get("values") or ["4K"])
    )
    expected_pin = takes_it if 0 < len(takes_it) < len(records) else None
    assert provider.get("only") == expected_pin, (
        f"{case}: the request was steered to {provider.get('only')!r} where the companies "
        f"taking 4K are {takes_it}"
    )

    card = render_image_help(
        "v/m", {"id": "v/m"}, endpoint_record=records, dedicated_image_api=True
    )
    routes_the_pick = expected_pin is not None
    assert any(clause in card for clause in _routing_clauses()) is routes_the_pick, (
        f"{case}: the card promises the pick is routed while the adapter pinned "
        f"{provider.get('only')!r}:\n{card}"
    )
    for surface, text, named in (
        ("help card", card, "this model"),
        ("model note", image_gen_model_note(spec, catalog_match=True), spec.model_id),
    ):
        assert (DISAGREED_SETTINGS.format(named=named) in text) is spec.providers_disagree, (
            f"{case}: the {surface} accounts for the disagreement in its own words while "
            f"the same surface promises the request is routed to a company that accepts "
            f"the pick: {text!r}"
        )


@pytest.mark.parametrize(("slug", "keyed"), [("acme", True), (None, False)])
@pytest.mark.asyncio
async def test_a_setting_withheld_for_want_of_a_company_is_one_the_request_cannot_key(slug, keyed):
    """The panel's account of an unnamed company, checked against the request's own.

    The control is withheld because the adapter has no slug to key the provider block
    under and drops the value. Dropping the slug requirement so the control renders again
    would put a box on screen that nothing carries, so the two must be asserted together:
    where the panel offers the setting the request carries it, and where it does not the
    request says why.
    """
    record: dict[str, Any] = {
        "provider_tag": "acme",
        "allowed_passthrough_parameters": ["steps"],
    }
    if slug is not None:
        record["provider_slug"] = slug
    records = [record]

    spec = build_image_model_filter_spec("v/m", {"id": "v/m"}, records, dedicated_image_api=True)
    assert ("steps" in spec.passthrough) is keyed, (
        f"the panel offers steps={('steps' in spec.passthrough)} with provider_slug={slug!r}"
    )
    assert spec.passthrough_unaddressable is not keyed

    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), records)
    result = await _posted(
        adapter,
        body={"image_config": {"steps": 8}},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    options = (result.payload.get("provider") or {}).get("options") or {}
    carried = options.get("acme", {}).get("steps")
    notices = " ".join(str(_event_data(call).get("content", "")) for call in result.events)
    assert (carried == 8) is keyed, (
        f"provider_slug={slug!r} carried steps={carried!r}; payload={result.payload!r}"
    )
    assert ("does not name the company" in notices) is not keyed, (
        f"provider_slug={slug!r} produced notices {notices!r}"
    )


# =============================================================================
# 10 - the generation panel draws the same size control the model's own does
# =============================================================================


_SIZE_SHAPE_CONTRACTS: list[tuple[str, tuple[str, ...], str]] = [
    ("size", ("1024x1024", "512x512"), "999x999"),
    ("resolution", ("1K", "2K"), "3K"),
]


def _published_size_records(published: str, values: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "provider_slug": "alpha",
            "provider_tag": "alpha",
            "supported_parameters": {published: {"type": "enum", "values": list(values)}},
        }
    ]


def _carried_by_the_model_panel(spec: Any, valve: str, typed: str, tag: str) -> bool:
    """What the model's own filter puts into ``image_config`` for one typed value."""
    module = _load_filter_from_source(render_image_model_filter_source(spec), tag)
    body: dict[str, Any] = {"model": spec.model_id}
    module.Filter().inlet(
        body, {}, {"valves": module.Filter.UserValves(**{valve: typed})}
    )
    return typed in (body.get("image_config") or {}).values()


def _carried_by_the_gen_panel(spec: Any, valve: str, typed: str, tag: str) -> bool:
    """What the generation panel puts into the image_generation tool call."""
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        render_image_gen_filter_source,
    )

    module = _load_filter_from_source(
        render_image_gen_filter_source(spec, catalog_match=True, selected_model=spec.model_id),
        tag,
    )
    metadata: dict[str, Any] = {}
    module.Filter().inlet(
        {"model": spec.model_id},
        metadata,
        {"valves": module.Filter.UserValves(**{valve: typed})},
    )
    tools = (metadata.get(PIPE_META) or {}).get("server_tools") or {}
    return typed in (tools.get("image_generation") or {}).values()


@pytest.mark.parametrize(
    ("published", "values", "unpublished"),
    _SIZE_SHAPE_CONTRACTS,
    ids=[published for published, _values, _unpublished in _SIZE_SHAPE_CONTRACTS],
)
def test_the_generation_panel_draws_the_size_control_the_models_own_panel_draws(
    published, values, unpublished
):
    """One contract publishes ``size``; the other publishes ``resolution``.

    The generation panel picked its size control by reading the ``resolution`` entry
    alone, so a contract publishing ``size`` was invisible to it: the model's own panel
    drew a closed list of the sizes that model takes while the generation panel drew a
    free-text box whose help text offered four tiers the request path then refused,
    because a declared ``size`` is matched straight against the published list.

    The shape of each control is read off the RENDERED filters rather than asserted as a
    literal: both are loaded the way Open WebUI loads one and run, and a value the
    contract does not publish either survives into the request (free text) or does not
    (a closed list). A published value must survive both, so a panel that dropped
    everything could not pass. Two contracts, and only one of them was broken, so a
    constant answer satisfies at most one row.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import _valve_name

    records = _published_size_records(published, values)
    spec = build_image_model_filter_spec(
        "v/m", {"id": "v/m", "name": "v/m"}, records, dedicated_image_api=True
    )
    valve = _valve_name(published)
    tag = f"size_shape_{published}"

    kept = values[0]
    assert _carried_by_the_model_panel(spec, valve, kept, f"{tag}_own_kept"), (
        f"the model's own panel dropped {kept!r}, which {published} publishes, so neither "
        "panel carries anything and the comparison below proves nothing"
    )
    assert _carried_by_the_gen_panel(spec, valve, kept, f"{tag}_gen_kept"), (
        f"the generation panel dropped {kept!r}, which {published} publishes"
    )

    own_takes_anything = _carried_by_the_model_panel(
        spec, valve, unpublished, f"{tag}_own_free"
    )
    gen_takes_anything = _carried_by_the_gen_panel(spec, valve, unpublished, f"{tag}_gen_free")

    assert gen_takes_anything is own_takes_anything, (
        f"{published} is published as a list of {values}: the model's own panel carries "
        f"the unpublished {unpublished!r}={own_takes_anything} and the generation panel "
        f"carries it={gen_takes_anything}, so one of them offers a value the other -- and "
        "the request path -- refuses"
    )


# =============================================================================
# 11 - the size box against a tier descriptor published but unusable
# =============================================================================


def _one_company_publishing(declared: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"provider_slug": "alpha", "provider_tag": "alpha", "supported_parameters": declared}]


_UNUSABLE_TIER_CONTRACTS: dict[str, list[dict[str, Any]]] = {
    "an-empty-list": _one_company_publishing(
        {"resolution": {"type": "enum", "values": []}}
    ),
    "a-numeric-range": _one_company_publishing(
        {"resolution": {"type": "range", "min": 1, "max": 4}}
    ),
    "a-range-under-the-other-name": _one_company_publishing(
        {"size": {"type": "range", "min": 1, "max": 4}}
    ),
    "a-usable-list": _one_company_publishing(
        {"resolution": {"type": "enum", "values": ["1K", "2K"]}}
    ),
    "nothing-of-its-own": _one_company_publishing({}),
}


def _tiers_the_request_carries(records: list[dict[str, Any]]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The tiers this contract lets onto the wire, and those it never measured.

    Both come from the request path: the split the adapter performs on a request
    carrying one tier, and the fit that split consults for the name it measured against.
    """
    from open_webui_openrouter_pipe.integrations.image_types import SCHEMA_ENUMS, TIER_EQUIVALENT

    declared = ImageGenerationAdapter._union_declared(records)
    carried: list[str] = []
    unmeasured: list[str] = []
    for tier in SCHEMA_ENUMS[TIER_EQUIVALENT["size"]]:
        top_level, _provider, _notes = ImageGenerationAdapter._split_image_config(
            {"image_config": {"size": tier}},
            allowed_passthrough=frozenset(),
            record=records[0],
            records=records,
        )
        if top_level.get("size") != tier:
            continue
        carried.append(tier)
        if not ImageGenerationAdapter._fit_published(declared, "size", tier).measured_as:
            unmeasured.append(tier)
    return tuple(carried), tuple(unmeasured)


def _passage_clauses() -> set[str]:
    """Every sentence the panel uses for a model that measures a tier against nothing.

    Read out of the meaning table by the state it is keyed under, so renaming the
    constants or reordering the table leaves this reading the same sentences.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import _SIZE_MEANING

    return {clause for (state, _ratio), (clause, _pixels) in _SIZE_MEANING.items() if state == "none"}


def _tier_sentence(description: str) -> str:
    """The one sentence the box uses to say what a tier is measured against.

    The longest of the table's clauses that the box carries: the pair for a panel with
    an Aspect ratio control and one without share an opening, so the shorter is a prefix
    of the longer and matching alone would find both.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import _SIZE_MEANING

    carried = {
        clause
        for (_state, _ratio), (clause, _pixels) in _SIZE_MEANING.items()
        if clause in description
    }
    assert carried, f"the box says nothing about what a tier is measured against: {description!r}"
    return max(carried, key=len)


@pytest.mark.parametrize("case", sorted(_UNUSABLE_TIER_CONTRACTS))
def test_the_size_box_promises_a_tier_travels_only_where_one_actually_does(case):
    """A descriptor can be published and still admit nothing: an empty list, or a bound.

    The box read those as publishing nothing and promised the tier would go out for the
    company running the model to interpret, while the request path measured it against
    that very descriptor and dropped it -- so nothing went out, and the panel said it
    would. What each contract does is computed here by putting a request carrying one
    tier through the adapter, and the sentence that makes the promise is taken from the
    panel's own meaning table rather than named, so the two halves cannot be restated to
    agree with each other.
    """
    records = _UNUSABLE_TIER_CONTRACTS[case]
    spec = build_image_model_filter_spec("v/m", {"id": "v/m"}, records, dedicated_image_api=True)
    carried, unmeasured = _tiers_the_request_carries(records)

    from open_webui_openrouter_pipe.filters.image_filter_renderer import image_knob_text

    _title, description = image_knob_text("size", spec)
    promises_passage = any(clause in description for clause in _passage_clauses())

    assert promises_passage is bool(unmeasured), (
        f"{case}: the box promises a tier goes out for the company to interpret="
        f"{promises_passage}, and the request path lets {list(unmeasured)} through "
        f"without measuring it against anything this model published: {description!r}"
    )
    assert not promises_passage or carried, (
        f"{case}: the box promises the tier travels and the request path carries none of "
        f"the four: {description!r}"
    )


@pytest.mark.parametrize(
    "case", sorted(set(_UNUSABLE_TIER_CONTRACTS) - {"a-usable-list", "nothing-of-its-own"})
)
def test_a_tier_the_model_refuses_outright_is_described_as_measured_not_as_passed_on(case):
    """Three contracts the request path refuses every tier for, one sentence between them.

    Two publish a descriptor that yields no values a tier can match -- an empty list, and
    a bound the fit reads as a number -- and one publishes the bound under `size` rather
    than `resolution`. The adapter treats all three alike: it measures the tier against
    what was published and drops it. The box has to describe them alike too, and the
    sentence it must use is the one it already gives a model that publishes a list a tier
    is measured against, read off the RENDERED box for such a model rather than named.
    """
    from open_webui_openrouter_pipe.filters.image_filter_renderer import image_knob_text

    records = _UNUSABLE_TIER_CONTRACTS[case]
    spec = build_image_model_filter_spec("v/m", {"id": "v/m"}, records, dedicated_image_api=True)
    carried, _unmeasured = _tiers_the_request_carries(records)
    assert not carried, (
        f"{case}: the request path carries {list(carried)}, so this is not the contract "
        "this test was set up around"
    )

    measured = build_image_model_filter_spec(
        "v/m", {"id": "v/m"}, _UNUSABLE_TIER_CONTRACTS["a-usable-list"], dedicated_image_api=True
    )
    unmeasured_spec = build_image_model_filter_spec(
        "v/m",
        {"id": "v/m"},
        _UNUSABLE_TIER_CONTRACTS["nothing-of-its-own"],
        dedicated_image_api=True,
    )
    measured_sentence = _tier_sentence(image_knob_text("size", measured)[1])
    assert measured_sentence != _tier_sentence(image_knob_text("size", unmeasured_spec)[1]), (
        "a model that measures a tier and one that does not are given the same sentence, "
        "so the comparison below would hold whatever the box said"
    )

    assert _tier_sentence(image_knob_text("size", spec)[1]) == measured_sentence, (
        f"{case}: the request path measures a tier against what this model published and "
        f"drops it, exactly as it does for a published list, and the box describes the "
        f"two differently: {image_knob_text('size', spec)[1]!r}"
    )
