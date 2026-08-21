"""The provider picker draws what the model's request format can carry, and no more.

The picker was written for chat completions and installed on whatever an operator listed.
On an image model eleven of its sixteen controls fed request fields that format does not
define, so the adapter dropped them after the fact -- a Zero Data Retention toggle that a
user could set, that read as in force, and that nothing enforced.

The expectation here is derived from the key sets the adapters themselves restrict on, so
a key added to one of them moves this gate with it. No number is written down.
"""

from __future__ import annotations

import ast
import re
import time
from typing import Any, get_args

import pytest

from open_webui_openrouter_pipe.filters.filter_manager import (
    _ROUTING_CONTROL_KEYS,
    FilterManager,
)
from open_webui_openrouter_pipe.integrations.provider_options import (
    CHAT_PROVIDER_KEYS,
    IMAGE_PROVIDER_KEYS,
    TRANSPORT_PROVIDER_KEYS,
    VIDEO_PROVIDER_KEYS,
    restrict_provider_block,
)
from tests.test_image_api_path import (  # noqa: F401 - shared stubs, one definition
    _adapter,
    _Emitter,
    _KeyPipe,
    _posted,
    _StubValves,
    _user_turn_with_images,
)
from tests.test_filters import _load_filter_from_source

RECORD = {
    "provider_slug": "recraft",
    "allowed_passthrough_parameters": [],
    "supported_parameters": {"n": {"type": "range", "min": 1, "max": 6}},
}


def _render(transport: str, visibility: str = "both") -> str:
    return FilterManager._render_provider_routing_filter_source(
        "vendor/model",
        ["recraft", "openai"],
        ["fp16"],
        visibility,
        short_name="Model",
        provider_names={"recraft": "Recraft", "openai": "OpenAI"},
        transport=transport,
    )


def _controls(source: str) -> set[str]:
    return set(re.findall(r"^        ([A-Z][A-Z_0-9]+): ", source, re.M))


def _emitted_keys(source: str) -> set[str]:
    return set(re.findall(r'provider\["([a-z_]+)"\]', source))


@pytest.mark.parametrize(
    ("transport", "accepted"),
    [("chat", CHAT_PROVIDER_KEYS), ("image", IMAGE_PROVIDER_KEYS), ("video", VIDEO_PROVIDER_KEYS)],
)
def test_a_picker_draws_exactly_the_controls_its_request_format_can_carry(transport, accepted):
    source = _render(transport)
    ast.parse(source)

    expected = {name for name, key in _ROUTING_CONTROL_KEYS.items() if key in accepted}
    assert _controls(source) == expected
    assert _emitted_keys(source) <= set(accepted)


@pytest.mark.parametrize(("transport", "accepted"), [("image", IMAGE_PROVIDER_KEYS), ("video", VIDEO_PROVIDER_KEYS)])
def test_the_hidden_controls_are_the_ones_that_would_have_been_dropped(transport, accepted):
    """Derived from the two key sets, never from a count: the set to hide is exactly the
    set the adapter would strip, so the two can never disagree about one key."""
    hidden = {name for name, key in _ROUTING_CONTROL_KEYS.items() if key not in accepted}
    assert hidden, "this row proves nothing unless the format really is narrower than chat"

    source = _render(transport)
    assert not (_controls(source) & hidden)

    would_have_been = {_ROUTING_CONTROL_KEYS[name] for name in hidden}
    kept, dropped = restrict_provider_block({key: "x" for key in would_have_been}, accepted)
    assert not kept and set(dropped) == would_have_been, (
        "the picker no longer draws them, and the adapter must still refuse them: a "
        "second producer of this metadata key would otherwise reach the wire unchecked"
    )


def test_every_control_names_a_request_field_the_inlet_actually_writes():
    """One table decides which controls exist and which fields they feed. Comparing it
    against the source the table itself generates is what stops the two drifting."""
    source = _render("chat")
    assert _emitted_keys(source) == set(_ROUTING_CONTROL_KEYS.values())
    assert set(TRANSPORT_PROVIDER_KEYS["chat"]) == _emitted_keys(source)


@pytest.mark.parametrize("visibility", ["admin", "user"])
def test_an_image_picker_still_offers_the_routing_controls_that_do_work(visibility):
    source = _render("image", visibility)
    module = _load_filter_from_source(source, f"routing_image_{visibility}")
    model = module.Filter.Valves if visibility == "admin" else module.Filter.UserValves

    assert {"ORDER", "ONLY", "IGNORE", "SORT", "ALLOW_FALLBACKS"} <= set(model.model_fields)
    assert "ZDR" not in model.model_fields, (
        "a retention control the request format has no field for is a promise nothing keeps"
    )


@pytest.mark.parametrize(("chosen", "slug"), [("Recraft", "recraft"), ("OpenAI", "openai")])
@pytest.mark.asyncio
async def test_a_control_the_image_picker_still_draws_reaches_the_image_request(chosen, slug):
    module = _load_filter_from_source(_render("image", "user"), f"routing_wire_{slug}")
    body: dict[str, Any] = {"model": "vendor/model"}
    metadata: dict[str, Any] = {}
    module.Filter().inlet(
        body, metadata, {"valves": module.Filter.UserValves(ONLY=chosen)}, None
    )

    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [RECORD])
    result = await _posted(
        adapter,
        body={},
        responses_body=_user_turn_with_images(0),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        metadata={"chat_id": "c", **metadata},
        normalized_model_id="m.x",
        api_model_id="m/x",
    )
    assert result.payload["provider"]["only"] == [slug]


@pytest.mark.parametrize("visibility", ["admin", "both"])
def test_a_format_that_carries_none_of_these_settings_gets_no_picker(visibility):
    """Every one of the sixteen controls fed a field the video request format does not
    define, so the whole picker was decoration."""
    assert FilterManager._routing_controls("video") == frozenset()
    assert _controls(_render("video", visibility)) == set()


@pytest.mark.parametrize(
    ("modalities", "features", "expected"),
    [
        (["image"], set(), "image"),
        (["image", "text"], set(), "chat"),
        (["text"], set(), "chat"),
        (["video"], {"video_generation"}, "video"),
    ],
)
def test_the_request_format_is_read_from_the_catalogue_not_the_name(
    modalities, features, expected
):
    """Everything else here hangs off this answer: get it wrong and an image model draws
    the whole chat picker again. A model that returns pictures AND words stays on chat,
    which no naming convention shows."""
    from open_webui_openrouter_pipe.models.registry import (
        ModelFamily,
        OpenRouterModelRegistry,
    )

    slug = "vendor/some-model"
    saved = OpenRouterModelRegistry._specs
    OpenRouterModelRegistry._specs = {
        ModelFamily.base_model(slug): {
            "features": features,
            "architecture": {"output_modalities": modalities},
        }
    }
    try:
        assert FilterManager.model_transport(slug) == expected
    finally:
        OpenRouterModelRegistry._specs = saved


def test_an_unknown_model_falls_back_to_the_widest_control_set():
    """A catalogue miss must not silently strip an operator's routing controls."""
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

    saved = OpenRouterModelRegistry._specs
    OpenRouterModelRegistry._specs = {}
    try:
        assert FilterManager.model_transport("vendor/never-seen") == "chat"
    finally:
        OpenRouterModelRegistry._specs = saved


@pytest.mark.parametrize("slug", ["vendor/one", "vendor/two"])
def test_the_state_hash_moves_when_a_model_changes_request_format(slug, monkeypatch):
    """A model that becomes image-only must have its picker rebuilt. The sync short-
    circuits on this hash, so a hash blind to the format would leave the old one drawn."""
    provider_map = {slug: {"providers": ["recraft"]}}

    monkeypatch.setattr(FilterManager, "model_transport", staticmethod(lambda _s: "chat"))
    as_chat = FilterManager.compute_provider_routing_hash(slug, "", provider_map)
    monkeypatch.setattr(FilterManager, "model_transport", staticmethod(lambda _s: "image"))
    as_image = FilterManager.compute_provider_routing_hash(slug, "", provider_map)

    assert as_chat != as_image


# =============================================================================
# 6 - the sort strategies and the object form
# =============================================================================


@pytest.mark.parametrize("transport", ["chat", "image"])
def test_the_published_sort_strategies_are_all_offered(transport):
    """OpenRouter's own schema lists four; its description names three, and the picker
    copied the description."""
    module = _load_filter_from_source(_render(transport, "user"), f"sort_offer_{transport}")
    offered = set(get_args(module.Filter.UserValves.model_fields["SORT"].annotation))
    assert {"price", "throughput", "latency", "exacto"} <= offered


@pytest.mark.parametrize(("strategy", "partition"), [("exacto", "none"), ("price", "model")])
def test_choosing_a_partition_sends_the_object_form_of_sort(strategy, partition):
    module = _load_filter_from_source(
        _render("chat", "user"), f"sort_object_{strategy}_{partition}"
    )
    body: dict[str, Any] = {"model": "vendor/model"}
    metadata: dict[str, Any] = {}
    module.Filter().inlet(
        body,
        metadata,
        {"valves": module.Filter.UserValves(SORT=strategy, SORT_PARTITION=partition)},
        None,
    )

    assert metadata["openrouter_pipe"]["provider"]["sort"] == {
        "by": strategy,
        "partition": partition,
    }


@pytest.mark.parametrize("strategy", ["exacto", "throughput"])
def test_a_strategy_with_no_partition_still_sends_the_bare_string(strategy):
    """The schema accepts either shape. Always sending the object would change what a
    request means for every operator who never asked about partitioning."""
    module = _load_filter_from_source(_render("chat", "user"), f"sort_bare_{strategy}")
    body: dict[str, Any] = {"model": "vendor/model"}
    metadata: dict[str, Any] = {}
    module.Filter().inlet(
        body, metadata, {"valves": module.Filter.UserValves(SORT=strategy)}, None
    )

    assert metadata["openrouter_pipe"]["provider"]["sort"] == strategy


@pytest.mark.parametrize("partition", ["model", "none"])
def test_a_partition_alone_is_sent_as_an_object_with_no_strategy(partition):
    module = _load_filter_from_source(_render("chat", "user"), f"sort_part_{partition}")
    body: dict[str, Any] = {"model": "vendor/model"}
    metadata: dict[str, Any] = {}
    module.Filter().inlet(
        body, metadata, {"valves": module.Filter.UserValves(SORT_PARTITION=partition)}, None
    )

    assert metadata["openrouter_pipe"]["provider"]["sort"] == {"partition": partition}


@pytest.mark.parametrize("visibility", ["admin", "user"])
def test_the_distillation_control_the_inlet_already_read_now_exists(visibility):
    """The generated inlet read this control and wrote its request field, and no class
    ever declared it, so the branch could not fire."""
    module = _load_filter_from_source(_render("chat", visibility), f"distill_{visibility}")
    model = module.Filter.Valves if visibility == "admin" else module.Filter.UserValves
    assert "ENFORCE_DISTILLABLE_TEXT" in model.model_fields

    body: dict[str, Any] = {"model": "vendor/model"}
    metadata: dict[str, Any] = {}
    filt = module.Filter()
    if visibility == "admin":
        filt.valves = model(ENFORCE_DISTILLABLE_TEXT=True)
        filt.inlet(body, metadata, None, None)
    else:
        filt.inlet(body, metadata, {"valves": model(ENFORCE_DISTILLABLE_TEXT=True)}, None)

    assert metadata["openrouter_pipe"]["provider"]["enforce_distillable_text"] is True


def test_the_routing_filter_adds_to_the_provider_block_rather_than_replacing_it():
    """Two filters write one provider block, and neither may erase the other's keys.

    Open WebUI runs filters in `(priority, id)` order and both default to priority 0, so
    `openrouter_image_filter_*` runs before `openrouter_provider_*`. The routing filter
    assigned the whole block, which threw away the provider options the image filter had
    just deep-merged into it -- silently, and only in that ordering.
    """
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    source = FilterManager._render_provider_routing_filter_source(
        model_slug="vendor_model",
        providers=["openai", "together"],
        quantizations=["fp16"],
        visibility="user",
    )
    module = _load_filter_from_source(source, "routing_filter_merge_probe")

    already_there = {"options": {"openai": {"labels": {"team": "x"}}}}
    metadata: dict = {"openrouter_pipe": {"provider": dict(already_there)}}
    instance = module.Filter()
    instance.inlet(
        {},
        __user__={"valves": module.Filter.UserValves(ONLY="Openai")},
        __metadata__=metadata,
    )

    written = metadata["openrouter_pipe"]["provider"]
    assert written.get("only") == ["openai"], (
        f"the routing filter did not write its own key: {written!r}"
    )
    assert written.get("options") == already_there["options"], (
        f"the routing filter erased another filter's provider options: {written!r}"
    )
