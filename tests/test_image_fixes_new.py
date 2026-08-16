"""New coverage for the image-path fixes (SYS, P1/P1b, P2, S2b, BCAST)."""
from __future__ import annotations

import time
from typing import Any

import pytest

from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter
from open_webui_openrouter_pipe.integrations.image_types import (
    pixel_size,
    prompt_with_system,
    reduced_ratio,
)
from open_webui_openrouter_pipe.integrations.provider_options import (
    bare_pins,
    fan_provider_options,
    options_key,
)
from open_webui_openrouter_pipe.requests.fusion_engine import item_text, latest_user_text

from tests.test_image_api_path import (  # noqa: F401
    _Emitter,
    _KeyPipe,
    _StubResponsesBody,
    _StubValves,
    _adapter,
    _posted,
    _posted_payload,
)


# ----------------------------------------------------------------- SYS


@pytest.mark.parametrize(
    ("house", "wanted"),
    [
        ("HOUSE STYLE: always cel-shaded, teal background", "a red mug"),
        ("HOUSE STYLE: always blueprint line art", "a green kettle"),
    ],
)
@pytest.mark.asyncio
async def test_the_models_own_system_prompt_reaches_the_image_prompt(house, wanted):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])
    payload = await _posted_payload(
        adapter,
        body={},
        responses_body=_StubResponsesBody([
            {"type": "message", "role": "system",
             "content": [{"type": "input_text", "text": house}]},
            {"role": "user", "content": [{"type": "input_text", "text": wanted}]},
        ]),
        valves=_StubValves("sk-x"),
        event_emitter=None,
        normalized_model_id="m.x",
        api_model_id="m/x",
    )
    prompt = payload["prompt"]
    assert house in prompt, f"the model's own system prompt was dropped; sent {prompt!r}"
    assert wanted in prompt
    assert prompt.index(house) < prompt.index(wanted), (
        f"the house style must lead the request it qualifies; sent {prompt!r}"
    )


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ([{"role": "system", "content": [{"type": "input_text", "text": "HOUSE"}]}], ""),
        ([{"role": "system", "content": [{"type": "input_text", "text": "HOUSE"}]},
          {"role": "user", "content": "   "}], "   "),
    ],
)
def test_a_request_with_no_user_text_stays_empty_so_the_prompt_guard_still_fires(items, expected):
    """A system turn alone is not a request.

    ``image.py`` refuses a blank prompt. Composing one out of the house style would turn an
    empty submit -- including an image attached with no text, which Open WebUI allows --
    into a billed generation nobody asked for.
    """
    assert prompt_with_system(items) == expected
    assert not prompt_with_system(items).strip()


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        ([{"text": ""}, {"text": "b"}], "\nb"),
        ([{"text": ""}, {"text": ""}], "\n"),
    ],
)
def test_item_text_keeps_an_empty_part_exactly_as_the_walk_it_replaced_did(parts, expected):
    """The extraction is of the existing behaviour, not of the video filter's.

    ``latest_user_text`` kept an empty-string part; the video half's ``_item_text`` dropped
    it. Adopting the video rule here silently changes what every caller of
    ``latest_user_text`` sends -- the fusion engine and the orchestrator among them.
    """
    item = {"role": "user", "content": parts}
    assert item_text(item) == expected
    assert latest_user_text([item]) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [([{"type": "input_image", "image_url": "u"}], None), ("", "")],
)
def test_item_text_separates_no_text_from_empty_text(content, expected):
    """``None`` means look past this item; ``""`` means it really is blank.

    Collapsing the two makes ``latest_user_text`` return ``""`` for an image-only turn
    instead of walking back to the last turn that carried words.
    """
    assert item_text({"role": "user", "content": content}) == expected


# ----------------------------------------------------------------- P1 / P1b


@pytest.mark.parametrize(
    ("sharded", "bare"),
    [
        ("black-forest-labs/us-3", "black-forest-labs"),
        ("google-ai-studio/global", "google-ai-studio"),
    ],
)
@pytest.mark.asyncio
async def test_a_region_sharded_slug_is_keyed_and_pinned_by_its_bare_name(sharded, bare):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{
        "provider_slug": sharded,
        "allowed_passthrough_parameters": ["style"],
        "supported_parameters": {"n": {"type": "range", "min": 1, "max": 4}},
    }])
    result = await _posted(
        adapter,
        body={"image_config": {"style": "digital_illustration", "n": 2}},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}],
            provider={"only": [bare]},
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )
    assert set(result.payload["provider"]["options"]) == {bare}
    assert result.payload["n"] == 2, "the contract was dropped, so n was withheld"
    notices = " ".join(str(e.get("content", "")) for e in result.events)
    assert "does not serve this model" not in notices
    assert result.payload["provider"]["only"] == [bare]


@pytest.mark.parametrize(
    ("pin", "expected"),
    [("google-ai-studio", "google-ai-studio/global"), ("google-vertex", "google-vertex/global")],
)
def test_a_bare_order_pin_reaches_the_record_it_names(pin, expected):
    """``order`` is the ordered field, and it was silently ignored.

    Comparing a bare pin against a sharded published slug never matched, so the fallback
    handed the request to whichever provider the catalog listed first.
    """
    records = [
        {"provider_slug": "google-vertex/global"},
        {"provider_slug": "google-ai-studio/global"},
    ]
    record, unserved = ImageGenerationAdapter._select_endpoint(records, {"order": [pin]})
    assert unserved == ""
    assert record is not None and record["provider_slug"] == expected


@pytest.mark.parametrize(
    ("ignored", "expected"),
    [("google-vertex", "google-ai-studio/global"), ("google-ai-studio", "google-vertex/global")],
)
def test_a_bare_ignore_pin_removes_the_record_it_names(ignored, expected):
    records = [
        {"provider_slug": "google-vertex/global"},
        {"provider_slug": "google-ai-studio/global"},
    ]
    record, unserved = ImageGenerationAdapter._select_endpoint(records, {"ignore": [ignored]})
    assert unserved == ""
    assert record is not None and record["provider_slug"] == expected


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_two_records_under_one_key_are_broken_by_sorted_slug_not_by_record_order(order):
    """Record order is not a decision anyone made; sorted slug at least reproduces."""
    shards = [{"provider_slug": "black-forest-labs/us-3"},
              {"provider_slug": "black-forest-labs/eu-1"}]
    records = [shards[order[0]], shards[order[1]]]
    record, _ = ImageGenerationAdapter._select_endpoint(records, {"only": ["black-forest-labs"]})
    assert record is not None and record["provider_slug"] == "black-forest-labs/eu-1"


@pytest.mark.parametrize(
    ("slug", "key"),
    [("black-forest-labs/us-3", "black-forest-labs"), ("google-ai-studio/global", "google-ai-studio")],
)
def test_options_key_reduces_a_shard_and_bare_pins_reduces_all_three_directives(slug, key):
    assert options_key(slug) == key
    assert options_key(key) == key
    assert options_key(None) == ""
    reduced = bare_pins({"only": [slug], "order": [slug], "ignore": [slug], "sort": "price"})
    assert reduced == {"only": [key], "order": [key], "ignore": [key], "sort": "price"}


@pytest.mark.parametrize(
    ("pin", "expected"),
    [("black-forest-labs", "black-forest-labs"), ("fal", "fal")],
)
def test_a_pin_no_record_carries_is_still_reported_as_unserved(pin, expected):
    records = [{"provider_slug": "black-forest-labs/us-3"}]
    record, unserved = ImageGenerationAdapter._select_endpoint(records, {"only": [pin]})
    if pin == "black-forest-labs":
        assert unserved == "" and record is not None
    else:
        assert record is None and unserved == expected


# ----------------------------------------------------------------- BCAST


@pytest.mark.parametrize(
    ("slugs", "bare"),
    [
        (["google-vertex/global", "google-ai-studio/global"], {"google-vertex", "google-ai-studio"}),
        (["black-forest-labs/us-3", "fal"], {"black-forest-labs", "fal"}),
    ],
)
@pytest.mark.asyncio
async def test_provider_options_are_fanned_to_every_carrier_and_directives_are_not(slugs, bare):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [
        {"provider_slug": slug, "allowed_passthrough_parameters": ["style"],
         "supported_parameters": {}}
        for slug in slugs
    ])
    result = await _posted(
        adapter,
        body={"image_config": {"style": "digital_illustration"}},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}],
            provider={"sort": "price"},
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )
    options = result.payload["provider"]["options"]
    assert set(options) == bare
    assert all(block == {"style": "digital_illustration"} for block in options.values())
    for directive in ("only", "order", "sort", "ignore"):
        assert not isinstance(result.payload["provider"].get(directive), dict)
    assert result.payload["provider"]["sort"] == "price"


@pytest.mark.parametrize(
    ("carriers", "expected"),
    [(["a", "b"], {"a", "b"}), (["c"], {"c"})],
)
def test_fan_provider_options_keeps_the_per_slug_deep_merge(carriers, expected):
    block = fan_provider_options(
        {"options": {"a": {"kept": 1}}, "only": ["a"]}, carriers, {"style": "x"}
    )
    assert set(block["options"]) >= expected
    assert block["only"] == ["a"]
    if "a" in expected:
        assert block["options"]["a"] == {"kept": 1, "style": "x"}


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([{"provider_slug": "google-vertex/global"}, {"provider_slug": "google-ai-studio/global"}],
         ["google-ai-studio", "google-vertex"]),
        ([{"provider_slug": "fal"}, {}], ["fal"]),
    ],
)
def test_option_carriers_reads_every_published_record(records, expected):
    assert ImageGenerationAdapter._option_carriers(records) == expected


# ----------------------------------------------------------------- S2b


@pytest.mark.parametrize(
    ("size", "ratio_survives"),
    [("1024x1024", False), ("1920x1080", True)],
)
def test_an_explicit_pixel_size_only_drops_a_ratio_it_provably_contradicts(size, ratio_survives):
    """``1920x1080`` *is* 16:9. Dropping it there would print a note claiming a 400 that
    the documented rule does not produce."""
    declared = {
        "supported_parameters": {
            name: {"type": "passthrough"} for name in ("size", "aspect_ratio", "resolution")
        }
    }
    params, _provider, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"size": size, "aspect_ratio": "16:9"}},
        allowed_passthrough=frozenset(),
        record=declared,
    )
    assert params["size"] == size
    assert ("aspect_ratio" in params) is ratio_survives
    assert bool([n for n in notes if n.name == "aspect_ratio"]) is not ratio_survives


@pytest.mark.parametrize(
    ("tier", "resolution", "kept"),
    [("2K", "2K", True), ("2K", "1K", False)],
)
def test_a_tier_size_collides_only_with_a_differing_resolution(tier, resolution, kept):
    declared = {
        "supported_parameters": {
            name: {"type": "passthrough"} for name in ("size", "aspect_ratio", "resolution")
        }
    }
    params, _provider, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"size": tier, "resolution": resolution, "aspect_ratio": "16:9"}},
        allowed_passthrough=frozenset(),
        record=declared,
    )
    assert params["aspect_ratio"] == "16:9", "a tier never collides with a ratio"
    assert ("resolution" in params) is kept
    assert bool(notes) is not kept


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2²x2", None), ("1024x1024", (1024, 1024))],
)
def test_pixel_size_refuses_a_digit_int_would_refuse(value, expected):
    """``"²".isdigit()`` is True and ``int("²")`` raises.

    Without the ASCII test the raise escapes ``_split_image_config`` and kills the whole
    request over one mistyped character.
    """
    assert pixel_size(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2²:1", None), ("1920:1080", (16, 9))],
)
def test_reduced_ratio_refuses_a_digit_int_would_refuse(value, expected):
    assert reduced_ratio(value) == expected
