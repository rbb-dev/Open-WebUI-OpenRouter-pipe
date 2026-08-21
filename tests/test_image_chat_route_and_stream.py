"""The chat route's image settings, and the streamed form of the dedicated one.

Nine of the forty models in OpenRouter's image catalogue return text as well as pictures,
so they answer on chat completions and never reach the image adapter -- yet each one gets
a filter built from its published contract, and everything that filter writes used to be
forwarded to chat completions unread. These tests pin what the contract now decides there.

The second half pins the streamed form of the dedicated image request: which models it is
asked for, that the reply is read by what came back rather than by what was asked for, and
that all four of its events are handled -- including the one spelled ``error`` on its own.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, cast

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter
from open_webui_openrouter_pipe.integrations.image_client import (
    _IMAGE_STREAM_HANDLERS,
    OpenRouterImageClient,
)
from open_webui_openrouter_pipe.integrations.image_types import ImageGenerationError

BASE = "https://openrouter.ai/api/v1"

GEMINI = "google/gemini-3-pro-image"
GPT5_IMAGE = "openai/gpt-5-image"
GPT_IMAGE_2 = "openai/gpt-image-2"


def records(model_id: str) -> list[dict[str, Any]]:
    """The model's own published contract, as recorded from the live endpoints listing."""
    name = model_id.replace("/", "_")
    path = Path(__file__).resolve().parent / "fixtures" / f"openrouter_image_endpoints_{name}.json"
    return json.loads(path.read_text())["endpoints"]


class _Logger:
    def __getattr__(self, _name: str):
        def _noop(*_args: Any, **_kwargs: Any) -> None:
            return None

        return _noop

    def isEnabledFor(self, _level: int) -> bool:
        return False


class _EmitterHandler:
    def __init__(self) -> None:
        self.notices: list[str] = []
        self.statuses: list[str] = []

    async def _emit_notification(self, emitter, content="", *, level="info", **_kw):
        self.notices.append(str(content))

    async def _emit_status(self, emitter, description, done=False, **_kw):
        self.statuses.append(str(description))

    async def _emit_completion(self, emitter, content="", done=False, usage=None, **_kw):
        return None


class _Gateway:
    async def resolve_storage_context(self, request, user_obj):
        return request, user_obj

    async def upload_to_owui_storage(self, **_kwargs):
        return "file-1"


class _Valves:
    def __init__(self) -> None:
        self.key = "sk-x"
        self.BASE_URL = BASE
        self.HTTP_REFERER_OVERRIDE = ""
        self.MODEL_CATALOG_REFRESH_SECONDS = 3600
        self.BASE64_MAX_SIZE_MB = 50
        self.SHOW_FINAL_USAGE_STATUS = False
        self.FINAL_USAGE_STATUS_STYLE = "text"
        self.USAGE_STATUS_ICON_SET = ""
        self.COSTS_REDIS_DUMP = False


class _Pipe:
    def __init__(self) -> None:
        self._event_emitter_handler = _EmitterHandler()
        self._file_gateway = _Gateway()
        self.valves = _Valves()
        self.id = "orpipe"
        self.generations: list[dict[str, Any]] = []

    async def _dispatch_plugin_event(self, method, *args, **kwargs):
        if method == "dispatch_on_generation_complete":
            self.generations.append({"usage": args[0], "status": args[1], **kwargs})

    @staticmethod
    def _resolve_openrouter_api_key(valves) -> tuple[str | None, str | None]:
        return valves.key, None


def adapter(pipe: Any) -> ImageGenerationAdapter:
    return ImageGenerationAdapter(pipe=cast(Any, pipe), logger=cast(Any, _Logger()))


class _Body:
    """Only the two attributes the chat-route fitting reads off a request body."""

    def __init__(self, image_config: dict[str, Any] | None, provider: dict[str, Any] | None = None):
        self.image_config = image_config
        self.provider = provider
        self.input = [{"role": "user", "content": [{"type": "input_text", "text": "a red leaf"}]}]


def png(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


# The chat route: what a published contract decides about image_config


@pytest.mark.parametrize(("spelled", "value"), [("image_size", "2K"), ("resolution", "1K")])
@pytest.mark.asyncio
async def test_a_chat_models_size_tier_goes_out_under_the_name_its_contract_publishes(
    spelled, value
):
    """One knob, two spellings, and only one of them means anything to anybody.

    ``image_size`` is what the retired fixed-variant filter wrote; ``resolution`` is what
    every one of these models publishes and the only one named in any OpenRouter document.
    Both have to arrive as the published one or the two filters disagree on the wire.
    """
    pipe = _Pipe()
    body = _Body({spelled: value})
    await adapter(pipe).fit_chat_image_config(
        responses_body=body,
        published=records(GEMINI),
        metadata=None,
        event_emitter=object(),
        api_model_id=GEMINI,
    )
    assert body.image_config == {"resolution": value}


@pytest.mark.parametrize(("ratio", "published"), [("21:9", True), ("32:9", False)])
@pytest.mark.asyncio
async def test_a_shape_the_chat_model_does_not_publish_is_withheld_and_named(ratio, published):
    pipe = _Pipe()
    body = _Body({"aspect_ratio": ratio})
    await adapter(pipe).fit_chat_image_config(
        responses_body=body,
        published=records(GEMINI),
        metadata=None,
        event_emitter=object(),
        api_model_id=GEMINI,
    )
    if published:
        assert body.image_config == {"aspect_ratio": ratio}
        assert pipe._event_emitter_handler.notices == []
    else:
        assert body.image_config is None
        assert pipe._event_emitter_handler.notices, "a withheld setting has to be reported"
        assert "aspect_ratio" in pipe._event_emitter_handler.notices[0]


@pytest.mark.parametrize(
    ("model_id", "name", "value"),
    [
        (GEMINI, "cachedContent", "projects/p/locations/global/cachedContents/7"),
        (GPT5_IMAGE, "moderation", "low"),
    ],
)
@pytest.mark.asyncio
async def test_a_provider_option_the_contract_names_survives_the_chat_route(
    model_id, name, value
):
    """Chat completions has one field for image settings and a closed provider block.

    Its ``ProviderPreferences`` sets ``additionalProperties: false`` and defines no
    ``options``, so there is nowhere else on that request for a provider setting to go.
    It stays where OpenRouter documents the provider-specific block to be, and what
    changes is that the contract is now consulted before it is sent.
    """
    pipe = _Pipe()
    body = _Body({name: value, "zzz_not_published": "1"})
    await adapter(pipe).fit_chat_image_config(
        responses_body=body,
        published=records(model_id),
        metadata=None,
        event_emitter=object(),
        api_model_id=model_id,
    )
    assert body.image_config == {name: value}
    assert pipe._event_emitter_handler.notices
    assert "zzz_not_published" in pipe._event_emitter_handler.notices[0]


@pytest.mark.parametrize(
    "config", [{"resolution": "9K", "zzz_not_published": "1"}, {"aspect_ratio": "32:9"}]
)
@pytest.mark.parametrize("published", [None, []])
@pytest.mark.asyncio
async def test_a_chat_request_is_left_alone_when_no_contract_is_in_hand(config, published):
    """A contract that could not be read is not a contract that shrank.

    Fitting a request to an empty contract deletes every setting the user chose, which is
    the same failure as overwriting an installed filter with a knobless one.
    """
    pipe = _Pipe()
    body = _Body(dict(config))
    await adapter(pipe).fit_chat_image_config(
        responses_body=body,
        published=published,
        metadata=None,
        event_emitter=object(),
        api_model_id=GEMINI,
    )
    assert body.image_config == config
    assert pipe._event_emitter_handler.notices == []


@pytest.mark.parametrize(
    ("pinned", "expected"),
    [("google-ai-studio", {"resolution": "4K"}), ("google-vertex", None)],
)
@pytest.mark.asyncio
async def test_the_chat_route_reads_the_contract_of_the_provider_that_was_pinned(
    pinned, expected
):
    """The two records this model publishes do not offer the same tiers.

    Google AI Studio publishes ``4K`` and Google Vertex stops at ``2K``, so which record
    is read decides whether the request keeps the tier or loses it. Both slugs are
    region-sharded in the contract and bare in the pin, which is why they are matched on
    the bare key.
    """
    pipe = _Pipe()
    body = _Body({"resolution": "4K"}, provider={"only": [pinned]})
    published = records(GEMINI) + [
        {
            "provider_slug": "google-ai-studio/global",
            "supported_parameters": {
                "resolution": {"type": "enum", "values": ["1K", "2K", "4K"]}
            },
            "allowed_passthrough_parameters": ["cachedContent"],
        }
    ]
    await adapter(pipe).fit_chat_image_config(
        responses_body=body,
        published=published,
        metadata=None,
        event_emitter=object(),
        api_model_id=GEMINI,
    )
    assert body.image_config == expected


@pytest.mark.parametrize("tier", ["4K", "2K"])
@pytest.mark.parametrize("order", ["published", "reversed"])
@pytest.mark.asyncio
async def test_the_chat_route_answers_the_same_however_the_records_are_ordered(tier, order):
    """A published value may not be kept or dropped by the position of a record.

    This model's two companies disagree: Google Vertex publishes `1K, 2K` and Google AI
    Studio publishes `1K, 2K, 4K`, and the filter draws `4K` because one of them takes
    it. Gating on a single record made the answer depend on which one the listing
    happened to put first -- with Vertex first `4K` was unreachable and the note
    contradicted the control's own text, and with AI Studio first `4K` was sent with
    nothing stopping Vertex taking the request and rejecting a value it does not
    publish. Both tiers are asserted so a rule that always kept, or always dropped,
    satisfies at most one row.
    """
    published = records(GEMINI)
    assert len(published) == 2, "the recorded contract no longer has two disagreeing records"
    if order == "reversed":
        published = list(reversed(published))

    pipe = _Pipe()
    body = _Body({"resolution": tier})
    await adapter(pipe).fit_chat_image_config(
        responses_body=body,
        published=published,
        metadata=None,
        event_emitter=object(),
        api_model_id=GEMINI,
    )
    assert body.image_config == {"resolution": tier}, (
        f"{order} order dropped a tier the model publishes: {body.image_config!r}"
    )
    expected_pin = ["google-ai-studio/global"] if tier == "4K" else None
    assert (body.provider or {}).get("only") == expected_pin, (
        f"{order} order pinned {(body.provider or {}).get('only')!r} for {tier}; a tier "
        "only one company publishes has to restrict routing to that company, and one "
        "they all publish must restrict nothing"
    )
    assert pipe._event_emitter_handler.notices == [], (
        f"{order} order reported {pipe._event_emitter_handler.notices!r} for a tier this "
        "model publishes"
    )


@pytest.mark.parametrize("tier", ["4K", "2K"])
@pytest.mark.asyncio
async def test_the_chat_route_pin_survives_the_restriction_the_orchestrator_applies(tier):
    """A pin the chat request format then discards is a routing control nothing enforces.

    `fit_chat_image_config` writes into `responses_body.provider`, and the orchestrator
    reduces that block to `CHAT_PROVIDER_KEYS` immediately afterwards, so the write is
    only worth making if it survives that step.
    """
    from open_webui_openrouter_pipe.integrations.provider_options import (
        CHAT_PROVIDER_KEYS,
        restrict_provider_block,
    )

    pipe = _Pipe()
    body = _Body({"resolution": tier})
    await adapter(pipe).fit_chat_image_config(
        responses_body=body,
        published=records(GEMINI),
        metadata=None,
        event_emitter=object(),
        api_model_id=GEMINI,
    )
    kept, _dropped = restrict_provider_block(body.provider or {}, CHAT_PROVIDER_KEYS)
    assert kept == (body.provider or {}), (
        f"the chat restriction dropped {sorted(set(body.provider or {}) - set(kept))} of "
        "the block the fitting wrote"
    )


@pytest.mark.parametrize(
    ("tag", "expected"),
    [("google-ai-studio/global", ["google-ai-studio/global"]), ("northstar", ["northstar"])],
)
@pytest.mark.asyncio
async def test_the_routing_pin_is_the_field_openrouter_designates_for_pinning(tag, expected):
    """OpenRouter splits the two roles and the pipe read the wrong one.

    `provider_slug` is documented for `provider.options[slug]`; `provider_tag` is
    documented for pinning a request to a provider and is `null` where provider-level
    routing is unavailable. Every one of the forty-four recorded contracts happens to
    publish the same string for both, so the two are indistinguishable on live data --
    these records give them different values so the pin has to name the one it is for.
    """
    published = [
        {
            "provider_slug": "google-vertex",
            "provider_tag": "google-vertex/global",
            "supported_parameters": {"resolution": {"type": "enum", "values": ["1K", "2K"]}},
            "allowed_passthrough_parameters": [],
        },
        {
            "provider_slug": "google-ai-studio",
            "provider_tag": tag,
            "supported_parameters": {
                "resolution": {"type": "enum", "values": ["1K", "2K", "4K"]}
            },
            "allowed_passthrough_parameters": [],
        },
    ]
    pipe = _Pipe()
    body = _Body({"resolution": "4K"})
    await adapter(pipe).fit_chat_image_config(
        responses_body=body,
        published=published,
        metadata=None,
        event_emitter=object(),
        api_model_id=GEMINI,
    )
    assert body.image_config == {"resolution": "4K"}
    assert (body.provider or {}).get("only") == expected, (
        f"the pin named {(body.provider or {}).get('only')!r}; `provider_tag` is the "
        "field OpenRouter documents for pinning"
    )


@pytest.mark.parametrize("tags", [(None, None), ("", "   ")])
@pytest.mark.asyncio
async def test_no_pin_is_written_where_no_candidate_publishes_a_routing_tag(tags):
    """`provider_tag` is null where provider-level routing is unavailable.

    Falling back to `provider_slug` would send a pin naming something routing cannot
    select, and OpenRouter's `only` is an allow-set -- so the request would be restricted
    to a provider that matches nothing and could not be served at all.
    """
    published = [
        {
            "provider_slug": "google-vertex",
            "provider_tag": tags[0],
            "supported_parameters": {"resolution": {"type": "enum", "values": ["1K", "2K"]}},
            "allowed_passthrough_parameters": [],
        },
        {
            "provider_slug": "google-ai-studio",
            "provider_tag": tags[1],
            "supported_parameters": {
                "resolution": {"type": "enum", "values": ["1K", "2K", "4K"]}
            },
            "allowed_passthrough_parameters": [],
        },
    ]
    pipe = _Pipe()
    body = _Body({"resolution": "4K"})
    await adapter(pipe).fit_chat_image_config(
        responses_body=body,
        published=published,
        metadata=None,
        event_emitter=object(),
        api_model_id=GEMINI,
    )
    assert body.image_config == {"resolution": "4K"}
    assert "only" not in (body.provider or {}), (
        f"a pin was written from something other than a routing tag: {body.provider!r}"
    )


@pytest.mark.parametrize(("ratio", "kept"), [("16:9", True), ("32:9", False)])
@pytest.mark.asyncio
async def test_a_chat_image_models_settings_are_fitted_on_the_way_out_of_the_pipe(
    ratio, kept, monkeypatch
):
    """The whole way through, not the two ends of it.

    Everything above tests the fitting in isolation; this drives a real request from
    ``Pipe.pipe`` to the outbound payload, because a fitting nothing calls changes
    nothing.
    """
    from open_webui_openrouter_pipe import EncryptedStr, Pipe
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

    OpenRouterModelRegistry.set_image_endpoints({GEMINI: records(GEMINI)})
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
                body='data: {"type":"response.completed","response":{"output":[],"usage":{"input_tokens":1,"output_tokens":1}}}\n\n',
                headers={"Content-Type": "text/event-stream"},
            )

        with aioresponses() as mocked:
            mocked.post(f"{BASE}/responses", callback=_callback, repeat=True)
            mocked.post(f"{BASE}/chat/completions", callback=_callback, repeat=True)
            mocked.get(
                f"{BASE}/models",
                payload={"data": [{"id": GEMINI, "name": "Gemini 3 Pro Image",
                                   "architecture": {"output_modalities": ["image", "text"]}}]},
                repeat=True,
            )
            answer = await pipe.pipe(
                body={
                    "model": GEMINI,
                    "messages": [{"role": "user", "content": "a red maple leaf"}],
                    "stream": True,
                    "image_config": {"aspect_ratio": ratio},
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
                async for _chunk in stream:
                    pass
    finally:
        await pipe.close()
        OpenRouterModelRegistry.set_image_endpoints({})

    assert sent, "nothing was sent upstream"
    config = sent[0].get("image_config")
    assert config == ({"aspect_ratio": ratio} if kept else None)


# The dedicated image request: asking for a stream, and reading one


@pytest.mark.parametrize(
    ("published", "asked"),
    [
        ([{"supports_streaming": True}, {"supports_streaming": True}], True),
        ([{"supports_streaming": True}, {"supports_streaming": False}], False),
    ],
)
def test_a_stream_is_asked_for_only_when_every_candidate_endpoint_publishes_one(
    published, asked
):
    """The model-level flag is a union and this is not.

    Which endpoint serves a request is settled after it leaves, so a model whose one
    streaming endpoint is joined by a non-streaming one would be asked to stream on a
    request the other takes.
    """
    assert ImageGenerationAdapter._every_endpoint_publishes_streaming(published) is asked


@pytest.mark.parametrize(
    ("value", "asked"), [(True, True), ("true", False)]
)
def test_a_streaming_flag_that_is_not_a_published_true_is_not_one(value, asked):
    assert (
        ImageGenerationAdapter._every_endpoint_publishes_streaming([{"supports_streaming": value}])
        is asked
    )


def test_the_recorded_contracts_name_exactly_the_models_that_can_stream():
    """Read off the live recording rather than asserted, so a re-record moves the list."""
    assert ImageGenerationAdapter._every_endpoint_publishes_streaming(records(GPT_IMAGE_2)) is True
    assert ImageGenerationAdapter._every_endpoint_publishes_streaming(records(GEMINI)) is False


def test_the_four_stream_events_are_keyed_by_their_exact_discriminator():
    """The failure event is spelled ``error``, not ``image_generation.error``.

    Keying the dispatch on the family prefix matches the three that report progress and
    drops the one that says the generation failed.
    """
    assert set(_IMAGE_STREAM_HANDLERS) == {
        "image_generation.partial_image",
        "image_generation.text_chunk",
        "image_generation.completed",
        "error",
    }


def _sse(*events: dict[str, Any]) -> str:
    lines = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return lines + "data: [DONE]\n\n"


async def _streamed(body: str, on_progress: Any = None, content_type: str = "text/event-stream"):
    import aiohttp

    with aioresponses() as mocked:
        mocked.post(
            f"{BASE}/images", body=body, headers={"Content-Type": content_type}, status=200
        )
        async with aiohttp.ClientSession() as session:
            client = OpenRouterImageClient(
                session, base_url=BASE, api_key="sk-x", logger=cast(Any, _Logger())
            )
            return await client.generate(
                {"model": GPT_IMAGE_2, "prompt": "a leaf", "stream": True},
                on_progress=on_progress,
            )


@pytest.mark.parametrize(("width", "height"), [(8, 8), (16, 32)])
@pytest.mark.asyncio
async def test_a_streamed_generation_returns_the_same_finished_image_as_a_buffered_one(
    width, height
):
    """The stream is reduced to the shape every later step already reads.

    That is what keeps the answer a plain markdown image link, which is what an iterative
    edit re-reads out of the previous turn to build the next request.
    """
    result = await _streamed(
        _sse(
            {"type": "image_generation.partial_image", "partial_image_index": 0, "b64_json": b64(png(2, 2))},
            {
                "type": "image_generation.completed",
                "b64_json": b64(png(width, height)),
                "media_type": "image/png",
                "created": 1,
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "cost": 0.04},
            },
        )
    )
    assert len(result.images) == 1
    assert result.images[0].data == png(width, height)
    assert result.usage["cost"] == 0.04


@pytest.mark.parametrize(
    ("message", "code"), [("The upstream provider returned an error", "upstream_error"), ("Generation failed", "server_error")]
)
@pytest.mark.asyncio
async def test_a_mid_stream_failure_reaches_the_user_with_the_reason_it_gave(message, code):
    with pytest.raises(ImageGenerationError) as caught:
        await _streamed(
            _sse(
                {"type": "image_generation.partial_image", "partial_image_index": 0, "b64_json": b64(png(2, 2))},
                {"type": "error", "error": {"message": message, "code": code}},
            )
        )
    assert message in str(caught.value)


@pytest.mark.parametrize(
    ("phase", "reported"), [("content", True), ("reasoning", False)]
)
@pytest.mark.asyncio
async def test_only_the_drawing_phase_of_a_text_stream_is_reported_as_progress(phase, reported):
    """A vector model never emits a partial picture, so its text is the only sign of life.

    ``reasoning`` and ``draft`` are the provider thinking on the way there; saying the
    picture is being drawn while it is not would be wrong.
    """
    seen: list[str] = []

    async def on_progress(message: str) -> None:
        seen.append(message)

    await _streamed(
        _sse(
            {"type": "image_generation.text_chunk", "phase": phase, "text": "<svg"},
            {"type": "image_generation.completed", "b64_json": b64(png(4, 4)), "created": 1},
        ),
        on_progress=on_progress,
    )
    assert ("Drawing the image…" in seen) is reported


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.asyncio
async def test_every_preview_the_stream_delivers_is_reported_as_progress(count):
    seen: list[str] = []

    async def on_progress(message: str) -> None:
        seen.append(message)

    await _streamed(
        _sse(
            *[
                {"type": "image_generation.partial_image", "partial_image_index": index, "b64_json": b64(png(2, 2))}
                for index in range(count)
            ],
            {"type": "image_generation.completed", "b64_json": b64(png(4, 4)), "created": 1},
        ),
        on_progress=on_progress,
    )
    assert seen == [f"Generating image… preview {index + 1}" for index in range(count)]


@pytest.mark.parametrize(
    "body",
    [
        "data: {\"type\":\"image_generation.partial_image\",\"partial_image_index\":0,\"b64_json\":\"\"}\n\ndata: [DONE]\n\n",
        "data: [DONE]\n\n",
    ],
)
@pytest.mark.asyncio
async def test_a_stream_that_ends_without_the_finished_image_is_a_failure_that_cost_nothing(body):
    """Image generation is billed all or nothing, so there is nothing to salvage."""
    with pytest.raises(ImageGenerationError) as caught:
        await _streamed(body)
    assert "Nothing was billed" in str(caught.value)


@pytest.mark.parametrize(("width", "height"), [(8, 8), (16, 32)])
@pytest.mark.asyncio
async def test_a_provider_that_ignores_the_request_to_stream_is_read_as_what_it_sent(
    width, height
):
    """OpenRouter documents a non-streaming provider as ignoring the flag and answering
    whole, so branching on what was asked for would try to parse that answer as a stream.
    """
    result = await _streamed(
        json.dumps({"created": 1, "data": [{"b64_json": b64(png(width, height)), "media_type": "image/png"}]}),
        content_type="application/json",
    )
    assert len(result.images) == 1
    assert result.images[0].data == png(width, height)


@pytest.mark.parametrize(("model_id", "asked"), [(GPT_IMAGE_2, True), (GEMINI, False)])
@pytest.mark.asyncio
async def test_the_stream_flag_reaches_the_request_for_the_models_that_publish_one(
    model_id, asked
):
    import time

    import aiohttp

    pipe = _Pipe()
    made = adapter(pipe)
    made._endpoint_cache[model_id] = (time.monotonic(), records(model_id))

    with aioresponses() as mocked:
        mocked.post(
            f"{BASE}/images",
            payload={"created": 1, "data": [{"b64_json": b64(png(8, 8)), "media_type": "image/png"}]},
        )
        async with aiohttp.ClientSession() as session:
            await made.generate(
                body={},
                responses_body=_Body(None),
                valves=_Valves(),
                session=session,
                event_emitter=None,
                metadata={},
                user=None,
                request=object(),
                user_obj=object(),
                normalized_model_id=model_id.replace("/", "."),
                api_model_id=model_id,
            )
        posts = [
            call
            for key, calls in mocked.requests.items()
            if key[1].path == "/api/v1/images"
            for call in calls
        ]

    assert posts
    assert ("stream" in posts[0].kwargs["json"]) is asked


@pytest.mark.parametrize("typed", [{"style": "digital_illustration"}, {"style": "vector_illustration"}])
@pytest.mark.asyncio
async def test_the_filters_provider_options_survive_to_the_image_request(typed):
    """A request bound for the image API must not be cut down to the chat key set.

    `ProviderPreferences` defines no `options`, so the merged block was emptied of it
    before the transport was chosen -- and the image request format, which does define
    `options`, then received a block with the user's settings already gone. Driven from
    `Pipe.pipe` because the cut happens above the adapter, where an adapter-level test
    cannot see it.
    """
    from open_webui_openrouter_pipe import EncryptedStr, Pipe
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

    model_id = "recraft/recraft-v3"
    OpenRouterModelRegistry.set_image_endpoints({model_id: records(model_id)})
    pipe = Pipe()
    sent: list[dict[str, Any]] = []
    try:
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        pipe.valves.BASE_URL = BASE
        pipe.valves.ENABLE_OPENROUTER_IMAGE_GENERATION = True

        def _callback(url, **kwargs):
            from aioresponses import CallbackResult

            sent.append(kwargs["json"])
            return CallbackResult(status=200, payload={"data": [], "usage": {}})

        with aioresponses() as mocked:
            mocked.post(f"{BASE}/images", callback=_callback, repeat=True)
            mocked.get(
                f"{BASE}/models",
                payload={"data": [{"id": model_id, "name": "Recraft V3",
                                   "architecture": {"output_modalities": ["image"]}}]},
                repeat=True,
            )
            await pipe.pipe(
                body={"model": model_id, "messages": [{"role": "user", "content": "a red maple leaf"}]},
                __user__={"id": "user_1"},
                __request__=None,
                __event_emitter__=None,
                __event_call__=None,
                __metadata__={
                    "model": {"id": model_id},
                    "openrouter_pipe": {
                        "provider": {"only": ["recraft"], "options": {"recraft": typed}}
                    },
                },
                __tools__=None,
                __task__=None,
                __task_body__=None,
            )
    finally:
        await pipe.close()
        OpenRouterModelRegistry.set_image_endpoints({})

    assert sent, "nothing was sent upstream"
    provider = sent[0].get("provider") or {}
    assert provider.get("options", {}).get("recraft") == typed, (
        f"the user's provider options did not reach the request: provider={provider!r}"
    )
    assert provider.get("only") == ["recraft"], (
        f"the routing pin that triggered the cut was itself lost: provider={provider!r}"
    )


@pytest.mark.parametrize(
    ("ignored", "expected"),
    [("google-vertex", {"resolution": "4K"}), ("google-ai-studio", None)],
)
@pytest.mark.asyncio
async def test_the_chat_route_stops_reading_the_contract_of_a_provider_the_operator_excluded(
    ignored, expected
):
    """`ignore` narrows which endpoints can serve the request exactly as `only` does.

    Google Vertex stops at `2K` and Google AI Studio publishes `4K`, so excluding one or
    the other decides whether the tier survives. Only the `only` pin was ever driven
    through this path, so the `ignore` arm could be deleted and every excluded provider's
    limits would go on being applied to a request it can no longer serve.

    The two rows expect opposite answers, so a rule that ignores the pin satisfies one at
    most.
    """
    pipe = _Pipe()
    body = _Body({"resolution": "4K"}, provider={"ignore": [ignored]})
    await adapter(pipe).fit_chat_image_config(
        responses_body=body,
        published=records(GEMINI),
        metadata=None,
        event_emitter=object(),
        api_model_id=GEMINI,
    )
    assert body.image_config == expected


@pytest.mark.asyncio
async def test_a_pin_naming_a_provider_that_does_not_serve_this_model_says_so():
    """A pin nobody serves is an operator mistake, not an outage.

    The contract WAS read, so reporting nothing would leave the user watching a request
    take a provider they thought they had pinned away from, and reporting a lookup
    failure would send an operator to check OpenRouter's status page.
    """
    pipe = _Pipe()
    body = _Body({"resolution": "4K"}, provider={"only": ["some-other-company"]})

    await adapter(pipe).fit_chat_image_config(
        responses_body=body,
        published=records(GEMINI),
        metadata=None,
        event_emitter=object(),
        api_model_id=GEMINI,
    )

    said = " ".join(pipe._event_emitter_handler.notices)
    assert "some-other-company" in said, (
        f"the pin that does not match was never mentioned to anyone: "
        f"{pipe._event_emitter_handler.notices}"
    )
    assert "does not serve this model" in said, said
    assert body.image_config == {"resolution": "4K"}, (
        "an unmatched pin must not also silently drop the settings the user chose"
    )
