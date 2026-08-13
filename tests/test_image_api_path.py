from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any, cast

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter
from open_webui_openrouter_pipe.integrations.provider_options import (
    IMAGE_PROVIDER_KEYS,
    carrier_slug,
    merge_provider_options,
)
from open_webui_openrouter_pipe.integrations.image_client import OpenRouterImageClient
from open_webui_openrouter_pipe.storage.multimodal import _guess_image_mime_type
from open_webui_openrouter_pipe.integrations.image_types import (
    GeneratedImage,
    ImageGenerationError,
)
from open_webui_openrouter_pipe.models.registry import uses_dedicated_image_api

BASE = "https://openrouter.ai/api/v1"


def monkeypatch_client(adapter: ImageGenerationAdapter, client: Any) -> None:
    object.__setattr__(adapter, "_client", lambda *_a, **_k: client)


def _adapter(pipe: Any) -> ImageGenerationAdapter:
    return ImageGenerationAdapter(pipe=cast(Any, pipe), logger=cast(Any, _Logger()))


def _png(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


class _Logger:
    def __getattr__(self, _name: str):
        def _noop(*_args: Any, **_kwargs: Any) -> None:
            return None

        return _noop

    def isEnabledFor(self, _level: int) -> bool:
        return False


async def _client(session) -> OpenRouterImageClient:
    return OpenRouterImageClient(
        session, base_url=BASE, api_key="test-key", logger=_Logger()
    )


@pytest.mark.asyncio
async def test_generate_posts_to_the_dedicated_image_endpoint():
    import aiohttp

    with aioresponses() as mocked:
        mocked.post(
            f"{BASE}/images",
            payload={"created": 1, "data": [{"b64_json": _b64(_png(512, 1024)), "media_type": "image/png"}]},
        )
        async with aiohttp.ClientSession() as session:
            client = await _client(session)
            await client.generate({"model": "qwen/qwen-image-3", "prompt": "leaf"})

        requests = [key for key in mocked.requests if key[1].path == "/api/v1/images"]
        assert requests, f"no request to /api/v1/images; saw {[k[1].path for k in mocked.requests]}"


@pytest.mark.asyncio
async def test_generate_sends_the_payload_verbatim_at_the_top_level():
    import aiohttp

    with aioresponses() as mocked:
        mocked.post(
            f"{BASE}/images",
            payload={"created": 1, "data": [{"b64_json": _b64(_png(512, 1024))}]},
        )
        async with aiohttp.ClientSession() as session:
            client = await _client(session)
            await client.generate(
                {
                    "model": "qwen/qwen-image-3",
                    "prompt": "leaf",
                    "aspect_ratio": "1:2",
                    "resolution": "2K",
                    "n": 2,
                }
            )

        (_method, _url), calls = next(iter(mocked.requests.items()))
        sent = calls[0].kwargs["json"]
        assert sent["aspect_ratio"] == "1:2"
        assert sent["resolution"] == "2K"
        assert sent["n"] == 2
        assert "image_config" not in sent


@pytest.mark.asyncio
async def test_endpoints_reads_an_unwrapped_payload():
    import aiohttp

    with aioresponses() as mocked:
        mocked.get(
            f"{BASE}/images/models/recraft/recraft-v3/endpoints",
            payload={
                "id": "recraft/recraft-v3",
                "endpoints": [
                    {
                        "provider_slug": "recraft",
                        "allowed_passthrough_parameters": ["style"],
                        "supported_parameters": {},
                    }
                ],
            },
        )
        async with aiohttp.ClientSession() as session:
            client = await _client(session)
            records = await client.endpoints("recraft/recraft-v3")

    assert records[0]["provider_slug"] == "recraft"
    assert records[0]["allowed_passthrough_parameters"] == ["style"]


@pytest.mark.asyncio
async def test_generate_reports_an_envelope_without_inline_base64():
    import aiohttp

    with aioresponses() as mocked:
        mocked.post(f"{BASE}/images", payload={"created": 1, "data": [{"url": "https://x/y.png"}]})
        async with aiohttp.ClientSession() as session:
            client = await _client(session)
            with pytest.raises(ImageGenerationError, match="inline base64"):
                await client.generate({"model": "m", "prompt": "p"})


@pytest.mark.asyncio
async def test_generate_reports_undecodable_base64():
    import aiohttp

    with aioresponses() as mocked:
        mocked.post(f"{BASE}/images", payload={"data": [{"b64_json": "data:image/png;base64,zz"}]})
        async with aiohttp.ClientSession() as session:
            client = await _client(session)
            with pytest.raises(ImageGenerationError, match="base64"):
                await client.generate({"model": "m", "prompt": "p"})


@pytest.mark.parametrize(("ceiling_mb", "rejected"), [(1, True), (64, False)])
@pytest.mark.asyncio
async def test_the_valve_governs_the_decode_ceiling(ceiling_mb, rejected):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x", base64_max_size_mb=ceiling_mb),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        reply={"data": [{"b64_json": _b64(_png(4, 4) + b"\x00" * (4 * 1024 * 1024))}]},
    )

    if rejected:
        assert "BASE64_MAX_SIZE_MB" in result.content
        assert not result.calls
    else:
        assert result.content.startswith("![Generated image]")
        assert len(result.calls) == 1


@pytest.mark.asyncio
async def test_a_reply_whose_images_together_exceed_the_ceiling_is_rejected():
    import aiohttp

    entry = {"b64_json": _b64(_png(4, 4) + b"\x00" * (600 * 1024))}
    with aioresponses() as mocked:
        mocked.post(f"{BASE}/images", payload={"data": [entry, entry]})
        async with aiohttp.ClientSession() as session:
            client = await _client(session)
            with pytest.raises(ImageGenerationError, match="BASE64_MAX_SIZE_MB"):
                await client.generate({"model": "m", "prompt": "p"}, max_decoded_bytes=1024 * 1024)

    with aioresponses() as mocked:
        mocked.post(f"{BASE}/images", payload={"data": [entry]})
        async with aiohttp.ClientSession() as session:
            client = await _client(session)
            result = await client.generate({"model": "m", "prompt": "p"}, max_decoded_bytes=1024 * 1024)
    assert len(result.images) == 1


@pytest.mark.asyncio
async def test_a_reply_that_is_not_an_image_is_reported_not_stored():
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        reply={"data": [{"b64_json": _b64(b'{"error":"quota"}'), "media_type": "application/json"}]},
    )

    assert not result.calls, "junk bytes must not reach Open WebUI storage"
    assert "Image generation failed" in result.content


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("image/avif", "image/avif"),
        ("IMAGE/AVIF", "image/avif"),
        ("image/AVIF; codecs=av01", "image/avif"),
        ("image/jp2", None),
        ("image/vnd.adobe.photoshop", None),
        ("image/x-quicktime", None),
    ],
)
def test_an_unknown_image_subtype_is_never_substituted_for_a_known_one(declared, expected):
    from open_webui_openrouter_pipe.storage.multimodal import canonical_image_mime

    assert canonical_image_mime(declared) == expected, (
        "the declaration reaches a stored filename and a content-type header, so an "
        "unrecognised subtype must be refused rather than quietly become png: serving "
        "jp2 bytes as image/png is a lie the browser acts on"
    )


@pytest.mark.parametrize("declared", ["image/avif", "IMAGE/AVIF", "image/AVIF; codecs=av01"])
@pytest.mark.asyncio
async def test_an_unrecognised_container_with_an_image_media_type_is_kept(declared):
    import aiohttp

    with aioresponses() as mocked:
        mocked.post(
            f"{BASE}/images",
            payload={"data": [{"b64_json": _b64(b"\x00\x00\x00 ftypavif"), "media_type": declared}]},
        )
        async with aiohttp.ClientSession() as session:
            client = await _client(session)
            result = await client.generate({"model": "m", "prompt": "p"})

    assert len(result.images) == 1
    assert result.images[0].extension == "avif", (
        "mime_type becomes the stored filename's extension; a parameterised or upper-case "
        f"declaration must not reach the filesystem verbatim. got {result.images[0].mime_type!r}"
    )
    assert result.images[0].mime_type == "image/avif", (
        "the sniffer not knowing a format is not evidence the payload is junk; the declared "
        "media type is the discriminator"
    )


@pytest.mark.asyncio
async def test_a_turn_with_an_image_and_no_caption_is_reported_not_raised():
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])
    emitter = _Emitter()

    image_only = _StubResponsesBody(
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "/api/v1/files/a/content"}
                ],
            }
        ]
    )

    content = await adapter.generate(
        body={},
        responses_body=image_only,
        valves=_StubValves("sk-x"),
        session=None,
        event_emitter=emitter,
        metadata={},
        user=None,
        request=object(),
        user_obj=object(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    assert "ImageGenerationError" not in content, (
        "the user reads this; a class name prefixed onto the message is a stack-trace leak, "
        f"not an explanation. got {content!r}"
    )
    assert isinstance(content, str) and "prompt" in content, (
        "attaching an image with no caption is the ordinary way into image editing; it must "
        f"not escape as an exception. got {content!r}"
    )
    assert any(event.get("done") for event in emitter.statuses)


@pytest.mark.parametrize(
    "exc",
    [__import__("asyncio").TimeoutError(), RuntimeError("disk on fire")],
)
@pytest.mark.asyncio
async def test_a_transport_failure_never_renders_a_dangling_reason(exc, monkeypatch):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    class _Boom:
        async def generate(self, *_a, **_k):
            raise exc

    monkeypatch.setattr(adapter, "_client", lambda *_a, **_k: _Boom())
    content = await adapter.generate(
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        session=None,
        event_emitter=_Emitter(),
        metadata={},
        user=None,
        request=object(),
        user_obj=object(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    reason = content.split("\n\n")[-1].strip()
    assert reason and not reason.endswith(":"), (
        f"several aiohttp/asyncio failures have an empty str(); rendered {content!r}"
    )


@pytest.mark.asyncio
async def test_the_requests_own_provider_block_survives():
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["recraft/recraft-v3"] = (time.monotonic(), [RECRAFT_RECORD])
    responses_body = _user_turn_with_images(
        0,
        provider={
            "only": ["recraft"],
            "zdr": True,
            "sort": "price",
            "options": {"recraft": {"controls": {"colors": []}}},
        },
    )

    result = await _posted(
        adapter,
        body={"image_config": {"style": "digital_illustration"}},
        responses_body=responses_body,
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="recraft.recraft-v3",
        api_model_id="recraft/recraft-v3",
    )

    provider = result.payload["provider"]
    assert provider["only"] == ["recraft"]
    assert provider["sort"] == "price"
    assert "zdr" not in provider, (
        "ImageGenerationProviderPreferences defines allow_fallbacks/ignore/only/options/"
        "order/sort and sets no additionalProperties:false, so zdr would be accepted and "
        "ignored - a privacy control that reads as in force while nothing enforces it"
    )
    assert provider["options"]["recraft"] == {
        "controls": {"colors": []},
        "style": "digital_illustration",
    }


@pytest.mark.parametrize(("ceiling_mb", "raises"), [(1, True), (64, False)])
@pytest.mark.asyncio
async def test_a_reply_over_the_base64_ceiling_is_rejected_before_decoding(ceiling_mb, raises):
    import aiohttp

    oversized = _b64(_png(4, 4) + b"\x00" * (4 * 1024 * 1024))

    with aioresponses() as mocked:
        mocked.post(f"{BASE}/images", payload={"data": [{"b64_json": oversized}]})
        async with aiohttp.ClientSession() as session:
            client = await _client(session)
            if raises:
                with pytest.raises(ImageGenerationError, match="BASE64_MAX_SIZE_MB"):
                    await client.generate(
                        {"model": "m", "prompt": "p"},
                        max_decoded_bytes=ceiling_mb * 1024 * 1024,
                    )
            else:
                result = await client.generate(
                    {"model": "m", "prompt": "p"},
                    max_decoded_bytes=ceiling_mb * 1024 * 1024,
                )
                assert len(result.images) == 1


@pytest.mark.asyncio
async def test_the_configured_ceiling_reaches_the_client_from_the_valve():
    import aiohttp

    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])
    valves = _StubValves("sk-x", base64_max_size_mb=1)
    emitter = _Emitter()

    with aioresponses() as mocked:
        mocked.post(
            f"{BASE}/images",
            payload={"data": [{"b64_json": _b64(_png(4, 4) + b"\x00" * (4 * 1024 * 1024))}]},
        )
        async with aiohttp.ClientSession() as session:
            content = await adapter.generate(
                body={},
                responses_body=_StubResponsesBody(
                    [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
                ),
                valves=valves,
                session=session,
                event_emitter=emitter,
                metadata={},
                user=None,
                request=object(),
                user_obj=object(),
                normalized_model_id="m.x",
                api_model_id="m/x",
            )

    assert "BASE64_MAX_SIZE_MB" in content, (
        "the valve's ceiling must reach the decode site; a hardcoded default would let a "
        "tightened valve be ignored"
    )


@pytest.mark.parametrize("status", [400, 402, 429, 500])
@pytest.mark.asyncio
async def test_generate_raises_a_routable_api_error_carrying_the_status(status):
    import aiohttp
    from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError

    with aioresponses() as mocked:
        mocked.post(
            f"{BASE}/images",
            status=status,
            payload={"error": {"message": "Unsupported aspect ratio 99:1", "code": status}},
        )
        async with aiohttp.ClientSession() as session:
            client = await _client(session)
            with pytest.raises(OpenRouterAPIError) as caught:
                await client.generate({"model": "m", "prompt": "p"})

    assert caught.value.status == status
    assert "99:1" in str(caught.value) or "99:1" in (caught.value.raw_body or "")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (_png(4, 4), "image/png"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"\xff\xd8\xff\xe0", "image/jpeg"),
        (b"GIF89a", "image/gif"),
        (b"<svg xmlns=", "image/svg+xml"),
        (b"\x00\x00\x01\x00", "image/x-icon"),
        (b"<?xml version='1.0'?><rss><channel/></rss>", None),
        (b"\x00\x01\x02\x03", None),
    ],
)
def test_the_package_has_one_magic_byte_classifier(raw, expected):
    assert _guess_image_mime_type("", "", raw) == expected


@pytest.mark.parametrize(
    ("mime", "extension"),
    [
        ("image/png", "png"),
        ("image/jpeg", "jpeg"),
        ("image/jpg", "jpeg"),
        ("image/webp", "webp"),
        ("image/svg+xml", "svg"),
        ("image/x-icon", "x-icon"),
        ("jpg", "png"),
        ("binary", "png"),
        ("", "png"),
    ],
)
def test_the_extension_matches_every_other_persistence_path(mime, extension):
    assert GeneratedImage(data=b"", mime_type=mime).extension == extension


def _register_live_image_catalog():
    import json
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

    catalog = json.loads(
        (Path(__file__).resolve().parent / "fixtures" / "openrouter_image_models.json").read_text()
    )["data"]
    OpenRouterModelRegistry.register_image_models(catalog)
    return {entry["id"]: entry for entry in catalog}


def test_only_image_only_models_take_the_dedicated_api():
    catalog = _register_live_image_catalog()
    routed, kept = set(), set()
    for model_id, entry in catalog.items():
        spec = {"architecture": entry.get("architecture") or {}}
        (routed if uses_dedicated_image_api(spec) else kept).add(model_id)

    expected_routed = {
        model_id
        for model_id, entry in catalog.items()
        if (entry.get("architecture") or {}).get("output_modalities") == ["image"]
    }
    assert routed == expected_routed
    assert routed and kept
    assert not any(
        "text" in ((catalog[m].get("architecture") or {}).get("output_modalities") or [])
        for m in routed
    )


def test_the_auto_routers_are_excluded_by_their_text_modality():
    catalog = _register_live_image_catalog()
    for router in ("openrouter/auto", "openrouter/auto-beta"):
        entry = catalog.get(router)
        if entry is None:
            continue
        assert "text" in (entry.get("architecture") or {}).get("output_modalities", [])
        assert uses_dedicated_image_api({"architecture": entry["architecture"]}) is False


@pytest.mark.parametrize(
    ("modalities", "expected"),
    [
        (["image"], True),
        (["image", "text"], False),
        (["text", "image"], False),
        (["text"], False),
        ([], False),
    ],
)
def test_routing_is_decided_by_output_modalities(modalities, expected):
    assert uses_dedicated_image_api({"architecture": {"output_modalities": modalities}}) is expected


def test_routing_ignores_a_missing_spec():
    assert uses_dedicated_image_api(None) is False
    assert uses_dedicated_image_api({}) is False
    assert uses_dedicated_image_api({"architecture": None}) is False
    assert uses_dedicated_image_api({"architecture": {"output_modalities": None}}) is False


def test_every_top_level_param_is_promoted_out_of_image_config():
    config = {
        "aspect_ratio": "16:9",
        "image_size": "2K",
        "size": "1024x1024",
        "n": 3,
        "seed": 7,
        "quality": "high",
        "background": "transparent",
        "output_format": "png",
        "output_compression": 80,
    }
    published = {
        "supported_parameters": {
            name: {"type": "passthrough"}
            for name in (
                "aspect_ratio", "resolution", "size", "n", "seed", "quality",
                "background", "output_format", "output_compression",
            )
        }
    }
    params, provider, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": config}, allowed_passthrough=frozenset(), record=published
    )
    assert params == {
        "aspect_ratio": "16:9",
        "resolution": "2K",
        "size": "1024x1024",
        "n": 3,
        "seed": 7,
        "quality": "high",
        "background": "transparent",
        "output_format": "png",
        "output_compression": 80,
    }
    assert provider == {}
    assert notes == []


def test_blank_and_unknown_image_config_keys_are_not_forwarded():
    params, provider, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"aspect_ratio": "", "font_inputs": ["x"], "resolution": "1K"}},
        allowed_passthrough=frozenset(),
        record={"supported_parameters": {"resolution": {"type": "passthrough"}}},
    )
    assert params == {"resolution": "1K"}
    assert provider == {}
    assert any("font_inputs" in note.text for note in notes), (
        "a knob the pipe's own filter renders was dropped; the user must be told, not left "
        f"wondering why it had no effect. notes were {notes!r}"
    )


class _Body:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.input = items


def test_prior_images_become_input_references():
    refs = ImageGenerationAdapter._input_references(
        _Body(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "make it bluer"},
                        {"type": "input_image", "image_url": "/api/v1/files/abc/content"},
                    ],
                }
            ]
        )
    )
    assert refs == [{"type": "image_url", "image_url": {"url": "/api/v1/files/abc/content"}}], (
        "requests/transformer.py::_to_input_image emits image_url as a bare string; a test "
        "that only drives the nested-dict form validates a shape production never produces"
    )


def test_the_nested_image_url_form_is_still_accepted():
    refs = ImageGenerationAdapter._input_references(
        _Body(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "/api/v1/files/xyz/content"}}
                    ],
                }
            ]
        )
    )
    assert refs == [{"type": "image_url", "image_url": {"url": "/api/v1/files/xyz/content"}}]


def test_text_only_history_yields_no_input_references():
    refs = ImageGenerationAdapter._input_references(
        _Body([{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}])
    )
    assert refs == []


def _recorded_endpoint(name: str) -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "fixtures" / f"openrouter_image_endpoints_{name}.json"
    return json.loads(path.read_text())["endpoints"][0]


RECRAFT_RECORD = _recorded_endpoint("recraft_recraft-v3")
QWEN_RECORD = _recorded_endpoint("qwen_qwen-image-3")


def test_provider_knobs_go_to_provider_options_not_the_top_level():
    top_level, provider, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"aspect_ratio": "1:1", "style": "digital_illustration"}},
        allowed_passthrough=frozenset(RECRAFT_RECORD["allowed_passthrough_parameters"]),
        record=RECRAFT_RECORD,
    )
    assert top_level == {"aspect_ratio": "1:1"}
    assert provider == {"style": "digital_illustration"}
    assert notes == []


def test_a_knob_the_endpoint_does_not_advertise_is_not_sent():
    _, provider, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"font_inputs": ["x"]}},
        allowed_passthrough=frozenset(RECRAFT_RECORD["allowed_passthrough_parameters"]),
        record=RECRAFT_RECORD,
    )
    assert provider == {}
    assert any("font_inputs" in note.text for note in notes)


@pytest.mark.parametrize(
    ("record", "sent", "dropped"),
    [
        (RECRAFT_RECORD, {"aspect_ratio": "1:1"}, "resolution"),
        (QWEN_RECORD, {"aspect_ratio": "1:1", "resolution": "2K"}, ""),
    ],
)
def test_top_level_knobs_are_gated_on_what_the_endpoint_advertises(record, sent, dropped):
    top_level, _, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"aspect_ratio": "1:1", "image_size": "2K"}},
        allowed_passthrough=frozenset(),
        record=record,
    )
    assert top_level == sent, (
        "the endpoint record is the authority on which top-level knobs this model takes; "
        f"recraft-v3 does not advertise resolution and qwen-image-3 does. got {top_level!r}"
    )
    if dropped:
        assert any(dropped in note.text for note in notes)
    else:
        assert notes == []


def test_a_value_outside_the_published_enum_is_reported_not_sent():
    top_level, _, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"aspect_ratio": "21:9"}},
        allowed_passthrough=frozenset(),
        record=QWEN_RECORD,
    )
    assert "aspect_ratio" not in top_level
    assert any("21:9" in note.text for note in notes)


@pytest.mark.parametrize(("asked", "expected"), [(99, 6), (2, 2)])
def test_a_count_is_capped_to_the_published_range(asked, expected):
    top_level, _, _ = ImageGenerationAdapter._split_image_config(
        {"image_config": {"n": asked}}, allowed_passthrough=frozenset(), record=QWEN_RECORD
    )
    assert top_level["n"] == expected, (
        "n is a billing multiplier and image_config is a first-class request field, so the "
        "endpoint's published max is the only bound on the wire"
    )


@pytest.mark.parametrize("asked", [99, 2])
def test_an_unreadable_contract_withholds_the_billing_multiplier_and_says_so(asked):
    top_level, provider, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"aspect_ratio": "21:9", "n": asked, "style": "x"}},
        allowed_passthrough=frozenset(),
        record=None,
    )
    assert "n" not in top_level, (
        "n multiplies what the request costs; with no published maximum there is nothing to "
        f"bound it, so an upstream blip would let n={asked} through at the requester's word"
    )
    assert top_level == {"aspect_ratio": "21:9"}, (
        "the other knobs must still work during an endpoint outage; withholding them all "
        "would trade one silent failure for another"
    )
    assert provider == {}
    assert any("style" in note.text for note in notes), (
        "a lookup failure must not silently change the payload shape without telling the user"
    )
    assert any("n" in note.text for note in notes if note.kind == "unbounded-multiplier"), (
        "the requester asked for n and did not get it; that must not be silent"
    )
    assert any(note.kind == "unvalidated" and "aspect_ratio" in note.text for note in notes), (
        "a value sent without a bound applied to it must be reported as unchecked, or the "
        "user cannot tell a validated request from an unvalidated one"
    )


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"supported_parameters": {"input_references": {"type": "range", "min": 0, "max": 1}}}, 1),
        ({"supported_parameters": {"input_references": {"type": "range", "min": 0, "max": 16}}}, 16),
        ({"supported_parameters": {}}, 0),
        ({"supported_parameters": {"input_references": "nonsense"}}, 0),
        ({}, None),
        ({"supported_parameters": {"input_references": {"type": "range", "max": 4.0}}}, 4),
        ({"supported_parameters": {"input_references": {"type": "range", "max": 1.0}}}, 1),
        ({"supported_parameters": {"input_references": {"type": "range", "max": -1}}}, None),
        ({"supported_parameters": {"input_references": {"type": "range", "max": "16"}}}, None),
        ({"supported_parameters": {"input_references": {"type": "range", "max": True}}}, None),
    ],
)
def test_reference_limit_distinguishes_unsupported_from_unknown(record, expected):
    assert ImageGenerationAdapter._reference_limit(record) == expected


@pytest.mark.asyncio
async def test_references_are_omitted_when_the_endpoint_does_not_support_them():
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{"supported_parameters": {}}])

    payload = await _posted_payload(
        adapter,
        body={},
        responses_body=_user_turn_with_images(2),
        valves=_StubValves("sk-x"),
        event_emitter=None,
        normalized_model_id="m.x",
        api_model_id="m/x",
    )
    assert "input_references" not in payload


class _Emitter:
    def __init__(self):
        self.statuses = []

    async def __call__(self, event):
        self.statuses.append(event)


class _StubEmitterHandler:
    async def _emit_status(self, emitter, description, done=False, **_kw):
        if emitter is not None:
            await emitter({"type": "status", "description": description, "done": done})

    async def _emit_notification(self, emitter, content="", *, level="info", **_kw):
        if emitter is not None:
            await emitter({"type": "notification", "content": content, "level": level})

    async def _emit_completion(self, emitter, content="", done=False, usage=None, **_kw):
        if emitter is not None:
            await emitter(
                {"type": "completion", "content": content, "done": done, "usage": usage}
            )


class _StubGateway:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def resolve_storage_context(self, request, user_obj):
        return request, user_obj

    async def upload_to_owui_storage(self, **kwargs):
        self.calls.append(kwargs)
        return f"file-{len(self.calls)}"


class _StubResponsesBody:
    def __init__(self, items, provider: dict[str, Any] | None = None):
        self.input = items
        self.provider = provider


def _user_turn_with_images(count, provider: dict[str, Any] | None = None):
    content: list[dict[str, Any]] = [{"type": "input_text", "text": "make it bluer"}]
    for index in range(count):
        content.append({"type": "input_image", "image_url": f"/api/v1/files/{index}/content", "detail": "auto"})
    return _StubResponsesBody([{"role": "user", "content": content}], provider=provider)


class _Posted:
    def __init__(
        self, payload, content, calls, statuses, events, generations=None, headers=None,
        requests=None,
    ):
        self.generations = generations or []
        self.headers = headers or {}
        self.requests = requests or {}
        self.payload = payload
        self.content = content
        self.calls = calls
        self.statuses = statuses
        self.events = events


async def _posted(
    adapter: ImageGenerationAdapter,
    *,
    body: dict[str, Any],
    responses_body: Any,
    valves: Any,
    event_emitter: Any,
    normalized_model_id: str,
    api_model_id: str,
    metadata: dict[str, Any] | None = None,
    reply: dict[str, Any] | None = None,
    user: Any = None,
    user_obj: Any = None,
    show_usage: bool | None = None,
) -> _Posted:
    import aiohttp

    if show_usage is not None:
        valves.SHOW_FINAL_USAGE_STATUS = show_usage

    with aioresponses() as mocked:
        mocked.get(
            f"{BASE}/images/models/{api_model_id}/endpoints",
            payload={"endpoints": [{}]},
        )
        mocked.post(
            f"{BASE}/images",
            payload=reply
            or {
                "created": 1,
                "data": [{"b64_json": _b64(_png(8, 8)), "media_type": "image/png"}],
            },
        )
        async with aiohttp.ClientSession() as session:
            content = await adapter.generate(
                body=body,
                responses_body=responses_body,
                valves=valves,
                session=session,
                event_emitter=event_emitter,
                metadata={"chat_id": "chat-1", "message_id": "msg-1"}
                if metadata is None
                else metadata,
                user=user,
                request=object(),
                user_obj=user_obj if user_obj is not None else object(),
                normalized_model_id=normalized_model_id,
                api_model_id=api_model_id,
            )

        posts = [
            call
            for key, calls in mocked.requests.items()
            if key[1].path == "/api/v1/images"
            for call in calls
        ]
        assert posts, f"nothing was POSTed to /api/v1/images; saw {[k[1].path for k in mocked.requests]}"
        gateway = cast(Any, adapter._pipe)._file_gateway
        statuses = getattr(event_emitter, "statuses", [])
        return _Posted(
            posts[0].kwargs["json"],
            content,
            list(getattr(gateway, "calls", [])),
            [str(item.get("description", "")) for item in statuses],
            list(statuses),
            list(getattr(cast(Any, adapter._pipe), "generations", [])),
            dict(posts[0].kwargs.get("headers") or {}),
            dict(mocked.requests),
        )


async def _posted_payload(adapter, **kwargs) -> dict[str, Any]:
    return (await _posted(adapter, **kwargs)).payload


@pytest.mark.parametrize(
    ("model_id", "record", "supplied", "kept", "dropped"),
    [
        ("recraft/recraft-v3", RECRAFT_RECORD, 3, 1, 2),
        ("qwen/qwen-image-3", QWEN_RECORD, 5, 4, 1),
    ],
)
@pytest.mark.asyncio
async def test_references_are_capped_to_the_endpoint_maximum(
    model_id, record, supplied, kept, dropped
):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache[model_id] = (time.monotonic(), [record])
    emitter = _Emitter()

    result = await _posted(
        adapter,
        body={"image_config": {"aspect_ratio": "1:1"}},
        responses_body=_user_turn_with_images(supplied),
        valves=_StubValves("sk-x"),
        event_emitter=emitter,
        normalized_model_id=model_id.replace("/", "."),
        api_model_id=model_id,
    )

    assert len(result.payload["input_references"]) == kept
    notes = [
        str(event.get("content", ""))
        for event in result.events
        if event.get("type") == "notification"
    ]
    assert any(f"dropped {dropped} reference image" in note for note in notes), (
        f"the user must be told how many were dropped; notifications were {notes!r}"
    )


@pytest.mark.parametrize(
    ("model_id", "record", "slug"),
    [
        ("recraft/recraft-v3", RECRAFT_RECORD, "recraft"),
        (
            "qwen/qwen-image-3",
            {**QWEN_RECORD, "allowed_passthrough_parameters": ["style"]},
            "alibaba",
        ),
    ],
)
@pytest.mark.asyncio
async def test_provider_options_are_keyed_by_the_endpoint_slug_not_the_model_prefix(
    model_id, record, slug
):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache[model_id] = (time.monotonic(), [record])
    knob = record["allowed_passthrough_parameters"][0]

    result = await _posted(
        adapter,
        body={"image_config": {knob: "value-x"}},
        responses_body=_user_turn_with_images(0),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id=model_id.replace("/", "."),
        api_model_id=model_id,
    )

    assert set(result.payload["provider"]["options"]) == {slug}, (
        "qwen/qwen-image-3 is served by 'alibaba' - a slug derived from the model id would be "
        f"wrong for it. got {result.payload['provider']['options']!r}"
    )
    assert knob not in result.payload


@pytest.mark.parametrize(
    ("prompt", "api_model_id"),
    [("a red leaf", "qwen/qwen-image-3"), ("a blue cube", "recraft/recraft-v3")],
)
@pytest.mark.asyncio
async def test_the_users_prompt_and_the_resolved_model_are_what_reach_the_request(
    prompt, api_model_id
):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache[api_model_id] = (time.monotonic(), [{}])

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id=api_model_id.replace("/", "."),
        api_model_id=api_model_id,
    )

    assert result.payload["prompt"] == prompt
    assert result.payload["model"] == api_model_id, (
        "the dotted OWUI form is not a model OpenRouter knows; shipping it rejects every "
        f"image generation. got {result.payload.get('model')!r}"
    )


@pytest.mark.parametrize(
    ("raw", "declared", "extension", "mime"),
    [
        (_png(8, 8), "image/png", "png", "image/png"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/png", "webp", "image/webp"),
    ],
)
@pytest.mark.asyncio
async def test_the_saved_file_carries_the_sniffed_container_not_the_declared_one(
    raw, declared, extension, mime
):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        reply={"data": [{"b64_json": _b64(raw), "media_type": declared}]},
    )

    assert len(result.calls) == 1
    call = result.calls[0]
    assert call["file_data"] == raw
    assert call["mime_type"] == mime
    assert call["filename"].endswith(f".{extension}")
    assert call["chat_id"] == "chat-1"
    assert call["message_id"] == "msg-1"
    assert result.content == "![Generated image](/api/v1/files/file-1/content)"
    completions = [e for e in result.events if e.get("type") == "completion"]
    assert completions and completions[-1]["content"] == result.content


class _StubValves:
    def __init__(self, key, base64_max_size_mb: int = 50, show_usage: bool = True):
        self.key = key
        self.BASE_URL = BASE
        self.HTTP_REFERER_OVERRIDE = ""
        self.MODEL_CATALOG_REFRESH_SECONDS = 3600
        self.BASE64_MAX_SIZE_MB = base64_max_size_mb
        self.SHOW_FINAL_USAGE_STATUS = show_usage
        self.FINAL_USAGE_STATUS_STYLE = "text"
        self.USAGE_STATUS_ICON_SET = ""
        self.COSTS_REDIS_DUMP = False


class _ReportRecorder:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def _report_openrouter_error(self, exc, **kwargs):
        self.calls.append({"exc": exc, **kwargs})


class _KeyPipe:
    def __init__(self, key, record_errors: bool = False):
        self._key = key
        self._event_emitter_handler = _StubEmitterHandler()
        self._file_gateway = _StubGateway()
        self.valves = _StubValves(key)
        self.id = "orpipe"
        self.reports = _ReportRecorder() if record_errors else None
        self.generations: list[dict[str, Any]] = []

    async def _dispatch_plugin_event(self, method, *args, **kwargs):
        if method == "dispatch_on_generation_complete":
            self.generations.append({"usage": args[0], "status": args[1], **kwargs})

    def _ensure_error_formatter(self):
        from open_webui_openrouter_pipe.core.error_formatter import ErrorFormatter
        import logging as _logging

        if self.reports is not None:
            return self.reports
        return ErrorFormatter(
            cast(Any, self), cast(Any, self._event_emitter_handler), _logging.getLogger("test.fmt")
        )

    @staticmethod
    def _resolve_openrouter_api_key(valves) -> tuple[str | None, str | None]:
        return valves.key, None


class _MissingKeyPipe(_KeyPipe):
    @staticmethod
    def _resolve_openrouter_api_key(valves) -> tuple[str | None, str | None]:
        return None, "OpenRouter API key is not configured."


@pytest.mark.parametrize("api_key", ["sk-or-first-key", "sk-or-second-key"])
@pytest.mark.asyncio
async def test_the_configured_api_key_reaches_the_authorization_header(api_key):
    import aiohttp

    adapter = _adapter(_KeyPipe(api_key))
    adapter._endpoint_cache["qwen/qwen-image-3"] = (time.monotonic(), [{}])

    with aioresponses() as mocked:
        mocked.post(
            f"{BASE}/images",
            payload={"created": 1, "data": [{"b64_json": _b64(_png(8, 8)), "media_type": "image/png"}]},
        )
        async with aiohttp.ClientSession() as session:
            await adapter.generate(
                body={},
                responses_body=_StubResponsesBody(
                    [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
                ),
                valves=_StubValves(api_key),
                session=session,
                event_emitter=None,
                metadata={},
                user=None,
                request=object(),
                user_obj=object(),
                normalized_model_id="qwen.qwen-image-3",
                api_model_id="qwen/qwen-image-3",
            )

        (_key, calls) = next(iter(mocked.requests.items()))
        assert calls[0].kwargs["headers"]["Authorization"] == f"Bearer {api_key}"


@pytest.mark.asyncio
async def test_a_missing_api_key_is_reported_to_the_user_not_swallowed():
    adapter = _adapter(_MissingKeyPipe(""))
    adapter._endpoint_cache["qwen/qwen-image-3"] = (time.monotonic(), [{}])
    emitter = _Emitter()

    result = await adapter.generate(
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves(""),
        session=None,
        event_emitter=emitter,
        metadata={},
        user=None,
        request=object(),
        user_obj=object(),
        normalized_model_id="qwen.qwen-image-3",
        api_model_id="qwen/qwen-image-3",
    )

    assert isinstance(result, str)
    assert "API key" in result
    assert any(e.get("done") for e in emitter.statuses)


@pytest.mark.asyncio
async def test_the_endpoint_record_is_fetched_once_and_reused_within_the_ttl():
    import aiohttp

    adapter = _adapter(_KeyPipe("sk-x"))
    valves = _StubValves("sk-x")

    with aioresponses() as mocked:
        mocked.get(
            f"{BASE}/images/models/recraft/recraft-v3/endpoints",
            payload={"endpoints": [RECRAFT_RECORD]},
        )
        async with aiohttp.ClientSession() as session:
            first, _ = await adapter._endpoint_record(session, valves, "recraft/recraft-v3")
            second, _ = await adapter._endpoint_record(session, valves, "recraft/recraft-v3")

        gets = [
            key for key in mocked.requests if key[1].path.endswith("/endpoints")
        ]
    assert first is not None and first["provider_slug"] == "recraft"
    assert second == first
    assert len(gets) == 1, "the record must be cached, not re-fetched on every generation"


@pytest.mark.asyncio
async def test_a_failed_endpoint_lookup_yields_unknown_not_an_empty_contract():
    import aiohttp

    adapter = _adapter(_KeyPipe("sk-x"))
    valves = _StubValves("sk-x")

    with aioresponses() as mocked:
        mocked.get(f"{BASE}/images/models/m/x/endpoints", status=500, payload={"error": "boom"})
        async with aiohttp.ClientSession() as session:
            record, _unserved = await adapter._endpoint_record(session, valves, "m/x")

    assert record is None, (
        "an empty dict means 'this model advertises nothing', which silently drops every "
        "provider knob; a failed lookup must be distinguishable from that"
    )
    assert "m/x" not in adapter._endpoint_cache, (
        "caching a failed lookup freezes the degradation for the whole TTL"
    )


@pytest.mark.asyncio
async def test_a_reshaped_endpoints_envelope_is_not_cached_as_an_empty_contract():
    import aiohttp

    adapter = _adapter(_KeyPipe("sk-x"))
    valves = _StubValves("sk-x")

    with aioresponses() as mocked:
        mocked.get(
            f"{BASE}/images/models/m/x/endpoints", payload={"data": {"endpoints": [RECRAFT_RECORD]}}
        )
        async with aiohttp.ClientSession() as session:
            record, _unserved = await adapter._endpoint_record(session, valves, "m/x")

    assert record is None
    assert "m/x" not in adapter._endpoint_cache


@pytest.mark.asyncio
async def test_cancellation_during_the_endpoint_lookup_is_not_swallowed(monkeypatch):
    import asyncio

    adapter = _adapter(_KeyPipe("sk-x"))

    class _Cancels:
        async def endpoints(self, _model_id):
            raise asyncio.CancelledError()

    monkeypatch.setattr(adapter, "_client", lambda *_a, **_k: _Cancels())
    with pytest.raises(asyncio.CancelledError):
        await adapter._endpoint_record(None, _StubValves("sk-x"), "m/x")


@pytest.mark.parametrize("referer", ["https://first.example", "https://second.example"])
@pytest.mark.asyncio
async def test_the_configured_referer_and_base_url_are_the_ones_used(referer):
    import aiohttp

    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])
    valves = _StubValves("sk-x")
    valves.HTTP_REFERER_OVERRIDE = referer

    with aioresponses() as mocked:
        mocked.post(f"{BASE}/images", payload={"data": [{"b64_json": _b64(_png(8, 8))}]})
        async with aiohttp.ClientSession() as session:
            await adapter.generate(
                body={},
                responses_body=_StubResponsesBody(
                    [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
                ),
                valves=valves,
                session=session,
                event_emitter=None,
                metadata={},
                user=None,
                request=object(),
                user_obj=object(),
                normalized_model_id="m.x",
                api_model_id="m/x",
            )
        (_key, calls) = next(iter(mocked.requests.items()))

    headers = calls[0].kwargs["headers"]
    from open_webui_openrouter_pipe.core.config import (
        _OPENROUTER_CATEGORIES,
        _OPENROUTER_TITLE,
    )

    assert headers["HTTP-Referer"] == referer
    assert headers["X-OpenRouter-Title"] == _OPENROUTER_TITLE
    assert headers["X-OpenRouter-Categories"] == _OPENROUTER_CATEGORIES


@pytest.mark.asyncio
async def test_usage_from_the_response_reaches_the_completion_event():
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        reply={
            "data": [{"b64_json": _b64(_png(8, 8))}],
            "usage": {"cost": 0.04, "prompt_tokens": 16, "completion_tokens": 4175, "total_tokens": 4191},
        },
    )

    completions = [e for e in result.events if e.get("type") == "completion"]
    assert completions and completions[-1].get("usage") == {
        "input_tokens": 16,
        "output_tokens": 4175,
        "total_tokens": 4191,
        "cost": 0.04,
    }, (
        "/api/v1/images answers in the chat-completions spelling; every consumer in this pipe "
        "reads the responses spelling, so the translation must happen at the transport boundary"
    )


@pytest.mark.asyncio
async def test_base64_outside_the_alphabet_is_rejected():
    import aiohttp

    with aioresponses() as mocked:
        mocked.post(f"{BASE}/images", payload={"data": [{"b64_json": "AAAA!!!!"}]})
        async with aiohttp.ClientSession() as session:
            client = await _client(session)
            with pytest.raises(ImageGenerationError, match="base64"):
                await client.generate({"model": "m", "prompt": "p"})


@pytest.mark.asyncio
async def test_a_blank_model_id_never_reaches_the_network():
    import aiohttp

    with aioresponses() as mocked:
        async with aiohttp.ClientSession() as session:
            client = await _client(session)
            with pytest.raises(ImageGenerationError):
                await client.endpoints("  ")
        assert not mocked.requests


def _also_emits_text(entry: dict) -> bool:
    """Whether a model answers with text as well as images.

    Read from the model's own declared modalities rather than a hand-kept list of ids.
    The list this replaces named six models and went stale the moment OpenRouter added
    a seventh, at which point the test reported a routing bug that was not one.
    """
    modalities = (entry.get("architecture") or {}).get("output_modalities") or []
    return "text" in modalities


def _recorded_image_api_roster() -> list[str]:
    path = Path(__file__).resolve().parent / "fixtures" / "openrouter_image_models.json"
    payload = json.loads(path.read_text())
    data = payload.get("data") if isinstance(payload, dict) else payload
    return [entry["id"] for entry in (data or []) if isinstance(entry, dict) and "id" in entry]


def test_every_recorded_image_api_model_routes_there_unless_it_also_emits_text():
    catalog = _register_live_image_catalog()
    roster = _recorded_image_api_roster()
    assert roster, "the recorded /images/models roster is empty"

    misrouted = []
    checked = 0
    for model_id in roster:
        entry = catalog.get(model_id)
        if entry is None:
            continue
        checked += 1
        routed = uses_dedicated_image_api({"architecture": entry.get("architecture") or {}})
        if not routed and not _also_emits_text(entry):
            misrouted.append(model_id)

    assert checked >= 20, (
        f"only {checked} of {len(roster)} recorded models are present in the routing catalog "
        "fixture, so this test is mostly skipping; re-record the catalog fixture"
    )
    assert not misrouted, (
        "these models are served by the dedicated image API but the pipe sends them to "
        f"chat/completions: {misrouted}"
    )


def test_the_routing_rule_is_exactly_does_this_model_also_emit_text():
    """One rule decides the transport, and it is the model's own declared modalities.

    This replaces a hand-listed set of exempt ids. That list named six models and went
    stale as soon as OpenRouter published a seventh, at which point the roster test
    above reported a routing bug that was not one.
    """
    catalog = _register_live_image_catalog()
    assert catalog, "the recorded catalogue must not be empty"

    disagreements = []
    both_kinds = set()
    for model_id, entry in catalog.items():
        architecture = entry.get("architecture") or {}
        routed = uses_dedicated_image_api({"architecture": architecture})
        emits_text = _also_emits_text(entry)
        both_kinds.add(emits_text)
        if routed == emits_text:
            disagreements.append((model_id, routed, emits_text))

    assert not disagreements, (
        "a model must take the dedicated image API if and only if it does not also emit "
        f"text; these disagree: {disagreements}"
    )
    assert both_kinds == {True, False}, (
        "the catalogue must contain models of both kinds or this proves nothing about "
        f"the rule; saw only {both_kinds}"
    )


def _routing_filter_provider_keys() -> set[str]:
    import re as _re
    from pathlib import Path as _Path

    source = (
        _Path(__file__).resolve().parents[1]
        / "open_webui_openrouter_pipe"
        / "filters"
        / "filter_manager.py"
    ).read_text()
    return set(_re.findall(r'provider\["([a-z_]+)"\]\s*=', source))


@pytest.mark.asyncio
async def test_every_provider_routing_key_the_filter_emits_reaches_the_wire():
    emitted = _routing_filter_provider_keys()
    assert len(emitted) >= 10, f"only found {sorted(emitted)} in the routing filter source"

    requested: dict[str, Any] = {key: "x" for key in emitted}
    requested["options"] = {"recraft": {"controls": {}}}
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["recraft/recraft-v3"] = (time.monotonic(), [RECRAFT_RECORD])

    result = await _posted(
        adapter,
        body={"image_config": {"style": "digital_illustration"}},
        responses_body=_user_turn_with_images(0, provider=requested),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="recraft.recraft-v3",
        api_model_id="recraft/recraft-v3",
    )

    sent = set(result.payload["provider"])
    assert sent == (emitted | {"options"}) & IMAGE_PROVIDER_KEYS, (
        "the image API documents exactly six provider keys; every other key the routing "
        f"filter writes must be withheld and reported, not sent. sent={sorted(sent)}"
    )
    undeliverable = emitted - IMAGE_PROVIDER_KEYS
    assert undeliverable, "this test is vacuous unless the filter writes a chat-only key"
    notice = " ".join(str(call.get("content", "")) for call in result.events)
    for key in sorted(undeliverable):
        assert key in notice, (
            f"{key} was withheld but the operator was never told; silence is what makes a "
            "dropped privacy control indistinguishable from an honoured one"
        )


@pytest.mark.parametrize(("uid", "meta"), [("u-real", "u-task"), ("u-second", "u-other")])
def test_the_user_object_outranks_the_metadata_user_id(uid, meta):
    assert ImageGenerationAdapter._requester_id({"id": f"  {uid}  "}, {"user_id": meta}) == uid, (
        "this id attributes the OWUI upload and the Redis cost row; when the two sources "
        "disagree the wrong account is billed"
    )


@pytest.mark.parametrize("uid", ["u-alpha", "u-beta"])
@pytest.mark.asyncio
async def test_the_requester_id_reaches_the_upload(uid):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        user={"id": uid},
    )

    assert result.calls[0]["owui_user_id"] == uid, (
        "without it a None user_obj resolves to the fallback storage account and the file is "
        "never linked to the chat"
    )


@pytest.mark.parametrize(("show", "expected"), [(True, "0.04"), (False, "")])
@pytest.mark.asyncio
async def test_the_final_status_honours_the_usage_valve(show, expected):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])
    valves = _StubValves("sk-x")

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=valves,
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        reply={"data": [{"b64_json": _b64(_png(8, 8))}], "usage": {"cost": 0.04, "total_tokens": 9}},
        show_usage=show,
    )

    final = [d for d in result.statuses if d][-1]
    if expected:
        assert expected in final, f"SHOW_FINAL_USAGE_STATUS is on but the cost is absent: {final!r}"
    else:
        assert "0.04" not in final
    assert "Saved base64 image to storage" not in final, (
        "that is the input-side progress message, not the terminal status of a generation"
    )


@pytest.mark.parametrize(
    "reply",
    [
        {"data": [{"url": "https://x/y.png"}]},
        {"data": []},
    ],
)
@pytest.mark.asyncio
async def test_an_upstream_contract_failure_is_visible_at_the_default_log_level(reply, caplog):
    import aiohttp
    import logging as _logging

    adapter = ImageGenerationAdapter(
        pipe=cast(Any, _KeyPipe("sk-x")), logger=_logging.getLogger("test.image.contract")
    )
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    with caplog.at_level(_logging.WARNING), aioresponses() as mocked:
        mocked.post(f"{BASE}/images", payload=reply)
        async with aiohttp.ClientSession() as session:
            await adapter.generate(
                body={},
                responses_body=_StubResponsesBody(
                    [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
                ),
                valves=_StubValves("sk-x"),
                session=session,
                event_emitter=_Emitter(),
                metadata={},
                user=None,
                request=object(),
                user_obj=object(),
                normalized_model_id="m.x",
                api_model_id="m/x",
            )

    warnings = [r for r in caplog.records if r.levelno >= _logging.WARNING]
    assert warnings, (
        "an operator running at the default level must be able to tell 'OpenRouter changed the "
        "/images envelope and every request is failing' from 'nobody generated an image today'"
    )
    assert any("m/x" in r.getMessage() for r in warnings)


@pytest.mark.parametrize("seam", ["endpoint_record", "record_cost", "generate"])
@pytest.mark.asyncio
async def test_cancellation_is_never_swallowed_by_any_handler(seam, monkeypatch):
    adapter = _adapter(_KeyPipe("sk-x"))

    async def _cancel(*_a, **_k):
        raise asyncio.CancelledError

    if seam == "endpoint_record":
        class _Cancels:
            async def endpoints(self, _model_id):
                raise asyncio.CancelledError

        monkeypatch_client(adapter, _Cancels())
        with pytest.raises(asyncio.CancelledError):
            await adapter._endpoint_record(None, _StubValves("sk-x"), "m/x")
        return

    if seam == "record_cost":
        from open_webui_openrouter_pipe.integrations import image as image_module

        monkeypatch.setattr(image_module, "maybe_dump_costs_snapshot", _cancel)
        with pytest.raises(asyncio.CancelledError):
            await adapter._record_cost(
                _StubValves("sk-x"),
                {"input_tokens": 1},
                user={"id": "u"},
                metadata={},
                user_obj=None,
                api_model_id="m/x",
            )
        return

    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    class _CancelsGenerate:
        async def generate(self, _payload, **_k):
            raise asyncio.CancelledError

    monkeypatch_client(adapter, _CancelsGenerate())
    with pytest.raises(asyncio.CancelledError):
        await adapter.generate(
            body={},
            responses_body=_StubResponsesBody(
                [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
            ),
            valves=_StubValves("sk-x"),
            session=object(),
            event_emitter=None,
            metadata={},
            user=None,
            request=object(),
            user_obj=object(),
            normalized_model_id="m.x",
            api_model_id="m/x",
        )


@pytest.mark.parametrize(
    ("record", "provider", "label"),
    [
        ({"allowed_passthrough_parameters": ["style", {"nope": 1}], "provider_slug": "recraft"},
         {}, "a non-string passthrough entry"),
        ({"provider_slug": "", "allowed_passthrough_parameters": ["style"]},
         {}, "an empty provider slug"),
        ({"supported_parameters": {"n": {"type": "enum", "values": "square"}}},
         {}, "a non-list enum values"),
        ({}, "not-a-dict", "a non-dict provider block"),
        ({}, {"options": "junk"}, "a non-dict options value"),
        ({"supported_parameters": "nonsense"}, {}, "a non-dict supported_parameters"),
    ],
)
@pytest.mark.asyncio
async def test_a_malformed_contract_degrades_instead_of_failing_the_generation(
    record, provider, label
):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [record])

    result = await _posted(
        adapter,
        body={"image_config": {"style": "digital_illustration", "n": 2}},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}],
            provider=provider,
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    assert result.payload["prompt"] == "a leaf", (
        f"{label} in the contract must degrade to 'unknown' and still reach POST /images; "
        "raising here turns an upstream shape drift into a hard failure of every request "
        "for that model, which is the opposite of the degrade-and-warn posture the "
        "adapter's own logging promises"
    )


@pytest.mark.parametrize("payload_size", [5000, 40000])
@pytest.mark.parametrize("source", ["image_config", "provider", "unkeyable-passthrough"])
@pytest.mark.asyncio
async def test_request_text_cannot_size_the_log_line_or_the_notification(
    source, payload_size, caplog
):
    import logging as _logging

    key = "z" * payload_size
    adapter = _adapter(_KeyPipe("sk-x"))
    emitter = _Emitter()
    body: dict[str, Any] = {}
    provider: dict[str, Any] | None = None

    if source == "image_config":
        adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])
        body = {"image_config": {key: "v" * payload_size}}
    elif source == "provider":
        adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])
        provider = {key: "on", "zdr": True}
    else:
        adapter._endpoint_cache["m/x"] = (
            time.monotonic(),
            [{"allowed_passthrough_parameters": [key]}],
        )
        body = {"image_config": {key: "on"}}

    with caplog.at_level(_logging.DEBUG):
        await _posted(
            adapter,
            body=body,
            responses_body=_StubResponsesBody(
                [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}],
                provider=provider,
            ),
            valves=_StubValves("sk-x"),
            event_emitter=emitter,
            normalized_model_id="m.x",
            api_model_id="m/x",
        )

    notified = max(
        (len(str(event.get("content", ""))) for event in emitter.statuses), default=0
    )
    logged = max((len(r.getMessage()) for r in caplog.records), default=0)
    assert notified < 2000 and logged < 2000, (
        "every text-bearing field of the request arrives from the client verbatim, so a key "
        "interpolated at full length sizes the browser event and the log record to whatever "
        f"was sent. source={source}, notification={notified} bytes, log={logged} bytes"
    )


@pytest.mark.parametrize(
    ("entries", "delivered"),
    [
        (["good", "good", "good", "good"], 4),
        (["good", "good", "good", "no-blob"], 3),
        (["good", "corrupt", "good", "not-an-image"], 2),
        (["no-blob", "good"], 1),
    ],
)
@pytest.mark.asyncio
async def test_an_unusable_entry_removes_only_itself_from_a_billed_batch(entries, delivered):
    shapes = {
        "good": {"b64_json": _b64(_png(8, 8)), "media_type": "image/png"},
        "no-blob": {"url": "https://example.invalid/x.png", "media_type": "image/png"},
        "corrupt": {"b64_json": "!!!not base64!!!", "media_type": "image/png"},
        "not-an-image": {"b64_json": _b64(b"plain text bytes"), "media_type": "text/plain"},
    }
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        reply={"data": [shapes[name] for name in entries]},
    )

    assert result.content.count("/api/v1/files/") == delivered, (
        "the batch was billed as a whole; an entry the pipe cannot read must remove exactly "
        f"itself, not the images decoded beside it. got {result.content!r}"
    )


@pytest.mark.parametrize("digits", [401, 60])
@pytest.mark.asyncio
async def test_a_paid_generation_is_not_lost_to_a_status_line(digits):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])
    pipe = cast(Any, adapter._pipe)

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        reply={
            "data": [{"b64_json": _b64(_png(8, 8)), "media_type": "image/png"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": int("1" + "0" * digits)},
        },
    )

    assert "/api/v1/files/" in result.content, (
        "the image was generated, billed and persisted; rendering the status line is "
        f"decoration and must not be able to turn it into a failure. got {result.content!r}"
    )
    statuses = [g.get("status") for g in pipe.generations]
    assert statuses == ["ok"], (
        f"exactly one generation event per request, carrying the 200's usage. got {statuses}"
    )


@pytest.mark.parametrize("seam", ["upload-raises", "no-storage-context"])
@pytest.mark.asyncio
async def test_a_failure_after_the_billed_call_still_reports_what_it_cost(seam, monkeypatch):
    from open_webui_openrouter_pipe.integrations import image as image_module

    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])
    pipe = cast(Any, adapter._pipe)

    snapshots: list[dict[str, Any]] = []

    async def _snapshot(_pipe, _valves, **kwargs):
        snapshots.append(kwargs)

    monkeypatch.setattr(image_module, "maybe_dump_costs_snapshot", _snapshot)

    async def _explode(*_a, **_k):
        raise OSError("[Errno 28] No space left on device")

    async def _no_context(request, user_obj):
        return None, None

    if seam == "upload-raises":
        cast(Any, pipe)._file_gateway.upload_to_owui_storage = _explode
    else:
        cast(Any, pipe)._file_gateway.resolve_storage_context = _no_context

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        reply={
            "data": [{"b64_json": _b64(_png(8, 8)), "media_type": "image/png"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "cost": 0.42},
        },
    )

    assert "failed" in result.content.lower()
    assert len(pipe.generations) == 1, (
        f"one request must produce one generation event. got {pipe.generations!r}"
    )
    reported = pipe.generations[0].get("usage") or {}
    assert reported.get("cost") == 0.42, (
        "the upstream call was billed the moment it returned 200; a local failure afterwards "
        f"must not record it as free. got {reported!r}"
    )
    assert [(snap.get("usage") or {}).get("cost") for snap in snapshots] == [0.42], (
        "one 200 is one charge, so it reaches the cost ledger exactly once. _settle closes a "
        f"failed request out; it must not record a second time. got {snapshots!r}"
    )


@pytest.mark.parametrize("pin_length", [8, 5000])
@pytest.mark.asyncio
async def test_an_unserved_pin_cannot_size_the_note_it_appears_in(pin_length):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (
        time.monotonic(),
        [{"provider_slug": "alpha", "supported_parameters": {}}],
    )

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}],
            provider={"only": ["z" * pin_length]},
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    notice = " ".join(str(event.get("content", "")) for event in result.events)
    assert len(notice) < 1000, (
        "the pin comes from the request, so a note built around it must be bounded by a "
        f"pipe-local constant rather than growing with what was sent. got {len(notice)} bytes"
    )


@pytest.mark.parametrize("pin", ["fal", "not-a-provider"])
@pytest.mark.asyncio
async def test_a_pin_we_hold_no_contract_for_never_borrows_another_providers_contract(pin):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (
        time.monotonic(),
        [
            {
                "provider_slug": "alpha",
                "allowed_passthrough_parameters": ["style"],
                "supported_parameters": {"n": {"type": "range", "min": 1, "max": 4}},
            }
        ],
    )

    result = await _posted(
        adapter,
        body={"image_config": {"n": 7}},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}],
            provider={"only": [pin]},
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    options = (result.payload.get("provider") or {}).get("options") or {}
    assert "alpha" not in options, (
        "routing was pinned away from alpha, so options keyed to alpha are dropped by "
        f"OpenRouter. got {options!r}"
    )
    notice = " ".join(str(e.get("content", "")) for e in result.events)
    assert "capped at 4" not in notice, (
        "alpha's published maximum belongs to a provider that will not serve this request; "
        f"quoting it tells the user something untrue about their own request. got {notice!r}"
    )


@pytest.mark.asyncio
async def test_one_request_never_dispatches_two_contradictory_generation_events():
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])
    pipe = cast(Any, adapter._pipe)

    class _FailsOnCompletion:
        def __init__(self):
            self.statuses = []
            self.fired = False

        async def __call__(self, event):
            self.statuses.append(event)
            if event.get("type") == "completion" and not self.fired:
                self.fired = True
                raise RuntimeError("the browser connection dropped")

    await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_FailsOnCompletion(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        reply={
            "data": [{"b64_json": _b64(_png(8, 8)), "media_type": "image/png"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "cost": 0.1},
        },
    )

    statuses = [g.get("status") for g in pipe.generations]
    assert statuses == ["ok"], (
        "the generation succeeded and was reported ok; a later delivery failure must not "
        f"dispatch a contradicting 'failed' for the same request. got {statuses}"
    )


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("image/png", "png"),
        ("image/jpeg", "jpeg"),
        ("image/svg+xml", "svg"),
        ("image/exe", "png"),
        ("image/" + "Z" * 400, "png"),
    ],
)
def test_the_extension_never_escapes_its_known_vocabulary(declared, expected):
    from open_webui_openrouter_pipe.storage.multimodal import image_extension_for_mime

    assert image_extension_for_mime(declared) == expected, (
        "the extension becomes a filename handed to Open WebUI storage; an upstream-declared "
        "media type must not be able to choose it"
    )


@pytest.mark.asyncio
async def test_a_lookup_failure_says_whether_a_stale_contract_is_in_play(caplog):
    import logging as _logging

    from open_webui_openrouter_pipe.integrations import image as image_module

    image_module._warned_image_endpoints.clear()

    class _Raises:
        async def endpoints(self, _model_id):
            raise RuntimeError("upstream said no")

    cold = ImageGenerationAdapter(
        pipe=cast(Any, _KeyPipe("sk-x")), logger=_logging.getLogger("test.image.cold")
    )
    monkeypatch_client(cold, _Raises())
    warm = ImageGenerationAdapter(
        pipe=cast(Any, _KeyPipe("sk-x")), logger=_logging.getLogger("test.image.warm")
    )
    warm._endpoint_cache["m/y"] = (time.monotonic() - 99999.0, [{"provider_slug": "alpha"}])
    monkeypatch_client(warm, _Raises())

    with caplog.at_level(_logging.DEBUG):
        await cold._endpoint_record(None, _StubValves("sk-x"), "m/x")
        cold_text = " ".join(r.getMessage() for r in caplog.records)
        caplog.clear()
        await warm._endpoint_record(None, _StubValves("sk-x"), "m/y")
        warm_text = " ".join(r.getMessage() for r in caplog.records)

    assert "unvalidated" in cold_text and "unvalidated" not in warm_text, (
        "with no cache the knobs really do go out unvalidated; with a stale cache they are "
        f"gated against it. cold={cold_text!r} warm={warm_text!r}"
    )
    assert "stale" in warm_text, (
        "an operator chasing a knob gated against limits the model no longer publishes needs "
        "to be told a stale contract is in play"
    )


@pytest.mark.parametrize(("entry_count", "key_len"), [(64, 200), (1, 20000), (2, 9000)])
@pytest.mark.asyncio
async def test_upstream_text_cannot_size_the_failure_message(entry_count, key_len, caplog):
    import logging as _logging

    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    with caplog.at_level(_logging.WARNING):
        result = await _posted(
            adapter,
            body={},
            responses_body=_StubResponsesBody(
                [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
            ),
            valves=_StubValves("sk-x"),
            event_emitter=_Emitter(),
            normalized_model_id="m.x",
            api_model_id="m/x",
            reply={"data": [{"z" * key_len: 1} for _ in range(entry_count)]},
        )

    logged = max((len(r.getMessage()) for r in caplog.records), default=0)
    assert len(result.content) < 2000 and logged < 2000, (
        "the reply is upstream-supplied, so both bounds have to hold: each rejection reason "
        "is clamped where it is built, and the join that assembles them is capped. one "
        f"enormous entry defeats the cap alone. content={len(result.content)} bytes, "
        f"longest log={logged} bytes"
    )








@pytest.mark.parametrize("mode", ["empty", "raises"])
@pytest.mark.asyncio
async def test_a_failing_endpoint_lookup_warns_once_per_model(mode, caplog):
    import logging as _logging
    from open_webui_openrouter_pipe.integrations import image as image_module

    image_module._warned_image_endpoints.clear()
    adapter = ImageGenerationAdapter(
        pipe=cast(Any, _KeyPipe("sk-x")), logger=_logging.getLogger("test.image.endpoints")
    )

    class _Empty:
        async def endpoints(self, _model_id):
            return []

    class _Raises:
        async def endpoints(self, _model_id):
            raise RuntimeError("upstream said no")

    monkeypatch_client(adapter, _Empty() if mode == "empty" else _Raises())
    with caplog.at_level(_logging.DEBUG):
        await adapter._endpoint_record(None, _StubValves("sk-x"), "m/x")
        await adapter._endpoint_record(None, _StubValves("sk-x"), "m/x")
        await adapter._endpoint_record(None, _StubValves("sk-x"), "m/y")

    needle = "no endpoint record" if mode == "empty" else "lookup failed"
    levels = [r.levelno for r in caplog.records if needle in r.getMessage()]
    assert levels == [_logging.WARNING, _logging.DEBUG, _logging.WARNING], (
        f"expected warn-once-per-model then DEBUG; got {levels!r}"
    )


@pytest.mark.parametrize(
    "config",
    [
        {"aspect_ratio": "1:2", "image_size": "2K", "n": 3, "seed": 11},
        {"aspect_ratio": "16:9", "image_size": "1K", "n": 1, "seed": 7},
    ],
)
@pytest.mark.asyncio
async def test_the_knobs_the_endpoint_advertises_reach_the_top_level_of_the_request(config):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["qwen/qwen-image-3"] = (time.monotonic(), [QWEN_RECORD])

    result = await _posted(
        adapter,
        body={"image_config": dict(config)},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="qwen.qwen-image-3",
        api_model_id="qwen/qwen-image-3",
    )

    assert result.payload["aspect_ratio"] == config["aspect_ratio"]
    assert result.payload["resolution"] == config["image_size"], (
        "image_size is the filter's spelling; /api/v1/images calls it resolution"
    )
    assert result.payload["n"] == config["n"]
    assert result.payload["seed"] == config["seed"]
    assert "image_config" not in result.payload, "the raw block must not reach the wire"


@pytest.mark.parametrize("status", [402, 429])
@pytest.mark.asyncio
async def test_an_upstream_status_is_reported_through_the_shared_error_formatter(status):
    import aiohttp

    pipe = _KeyPipe("sk-x", record_errors=True)
    adapter = _adapter(pipe)
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])
    emitter = _Emitter()

    with aioresponses() as mocked:
        mocked.post(f"{BASE}/images", status=status, payload={"error": {"message": "nope"}})
        async with aiohttp.ClientSession() as session:
            content = await adapter.generate(
                body={},
                responses_body=_StubResponsesBody(
                    [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
                ),
                valves=_StubValves("sk-x"),
                session=session,
                event_emitter=emitter,
                metadata={},
                user=None,
                request=object(),
                user_obj=object(),
                normalized_model_id="m.x",
                api_model_id="m/x",
            )

    assert isinstance(content, str)
    assert pipe.reports is not None and len(pipe.reports.calls) == 1, (
        "a 4xx must reach the shared error formatter so it renders the right template"
    )
    call = pipe.reports.calls[0]
    assert call["exc"].status == status
    assert call["api_model_id"] == "m/x"
    assert any(event.get("done") for event in emitter.statuses)


@pytest.mark.parametrize("order", [("first", "second"), ("second", "first")])
@pytest.mark.asyncio
async def test_the_first_endpoint_in_the_array_is_the_one_used(order):
    import aiohttp

    adapter = _adapter(_KeyPipe("sk-x"))
    endpoints = [{**RECRAFT_RECORD, "provider_slug": slug} for slug in order]

    with aioresponses() as mocked:
        mocked.get(f"{BASE}/images/models/m/x/endpoints", payload={"endpoints": endpoints})
        async with aiohttp.ClientSession() as session:
            record, _unserved = await adapter._endpoint_record(session, _StubValves("sk-x"), "m/x")

    assert record is not None and record["provider_slug"] == order[0]


@pytest.mark.parametrize("reply", [{"status": 500}, {"payload": {"endpoints": []}}])
@pytest.mark.asyncio
async def test_a_failed_refresh_keeps_the_previously_cached_contract(reply):
    import aiohttp

    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic() - 10_000, [RECRAFT_RECORD])

    with aioresponses() as mocked:
        mocked.get(f"{BASE}/images/models/m/x/endpoints", **reply)
        async with aiohttp.ClientSession() as session:
            record, _unserved = await adapter._endpoint_record(session, _StubValves("sk-x"), "m/x")

    assert record is RECRAFT_RECORD, (
        "a transient failure must not throw away a good contract and silently drop every "
        "provider knob until the next success"
    )


@pytest.mark.parametrize(("refresh_seconds", "expected_gets"), [(10_000, 0), (60, 1)])
@pytest.mark.asyncio
async def test_the_refresh_valve_governs_the_endpoint_cache_window(refresh_seconds, expected_gets):
    import aiohttp

    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic() - 3_600, [RECRAFT_RECORD])
    valves = _StubValves("sk-x")
    valves.MODEL_CATALOG_REFRESH_SECONDS = refresh_seconds

    with aioresponses() as mocked:
        mocked.get(f"{BASE}/images/models/m/x/endpoints", payload={"endpoints": [RECRAFT_RECORD]})
        mocked.get(f"{BASE}/images/models/m/x/endpoints", payload={"endpoints": [RECRAFT_RECORD]})
        async with aiohttp.ClientSession() as session:
            await adapter._endpoint_record(session, valves, "m/x")
            await adapter._endpoint_record(session, valves, "m/x")
        gets = sum(
            len(calls) for key, calls in mocked.requests.items() if key[1].path.endswith("/endpoints")
        )

    assert gets == expected_gets, (
        f"MODEL_CATALOG_REFRESH_SECONDS={refresh_seconds} must decide the window; saw {gets} fetches"
    )


@pytest.mark.parametrize("failing_index", [1, 2])
@pytest.mark.asyncio
async def test_a_partly_persisted_multi_image_reply_keeps_the_survivors_and_says_so(failing_index):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])
    gateway = cast(Any, adapter._pipe)._file_gateway
    original = gateway.upload_to_owui_storage

    async def _flaky(**kwargs):
        file_id = await original(**kwargs)
        return None if len(gateway.calls) == failing_index else file_id

    gateway.upload_to_owui_storage = _flaky
    entry = {"b64_json": _b64(_png(8, 8)), "media_type": "image/png"}

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        reply={"data": [entry, entry, entry]},
    )

    links = [line for line in result.content.split("\n\n") if line.startswith("![")]
    assert len(links) == 2, f"two of three persisted; content was {result.content!r}"
    assert "Generated image 1" in result.content or "Generated image 2" in result.content
    assert "1 generated image(s) could not be saved" in result.content


@pytest.mark.asyncio
async def test_nothing_is_rendered_when_storage_has_no_context():
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    async def _no_context(request, user_obj):
        return None, None

    cast(Any, adapter._pipe)._file_gateway.resolve_storage_context = _no_context

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    assert not result.calls, "no storage context means no upload may be attempted"
    assert "could not be saved to Open WebUI storage" in result.content


def test_an_unreadable_contract_reads_differently_from_a_rejected_knob():
    unknown = ImageGenerationAdapter._split_image_config(
        {"image_config": {"style": "x"}}, allowed_passthrough=frozenset(), record=None
    )[2]
    known = ImageGenerationAdapter._split_image_config(
        {"image_config": {"style": "x"}}, allowed_passthrough=frozenset(), record=RECRAFT_RECORD
    )[2]

    assert "could not be read" in " ".join(n.text for n in unknown)
    assert "could not be read" not in " ".join(n.text for n in known), (
        "a transient lookup failure and a model that genuinely rejects the knob need "
        "different remedies, so they must read differently"
    )
    assert "not offered" in " ".join(n.text for n in known)
    assert "not offered" not in " ".join(n.text for n in unknown)


def test_a_published_maximum_of_zero_means_none_not_unknown():
    assert (
        ImageGenerationAdapter._reference_limit(
            {"supported_parameters": {"input_references": {"type": "range", "min": 0, "max": 0}}}
        )
        == 0
    ), "None means 'contract unknown, do not cap'; a published 0 means 'send none'"


def test_the_recorded_ceiling_is_what_caps_the_billing_multiplier():
    recorded = RECRAFT_RECORD["supported_parameters"]["n"]["max"]

    top_level, _, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"n": recorded + 4}},
        allowed_passthrough=frozenset(),
        record=RECRAFT_RECORD,
    )

    assert top_level["n"] == recorded, (
        "n is what the request is billed per image, and this recording is the only statement "
        f"of what the model actually accepts. reading the ceiling from the fixture means a "
        f"re-recording propagates instead of silently disagreeing. got {top_level.get('n')!r}"
    )
    assert any(str(recorded) in note.text for note in notes)


@pytest.mark.parametrize(
    ("asked", "fitted", "note"),
    [
        (0, 1, "raised to 1"),
        (99, 6, "capped at 6"),
        (2, 2, ""),
        ("2K", None, "expects a number"),
        (True, None, "expects a number"),
        (False, None, "expects a number"),
    ],
)
def test_a_range_knob_fits_numbers_and_refuses_everything_else(asked, fitted, note):
    assert ImageGenerationAdapter._fit_descriptor(
        {"type": "range", "min": 1, "max": 6}, asked
    ) == (fitted, note), (
        "a non-number must be dropped with a note the user can act on, never raised out of "
        "_split_image_config as a bare TypeError"
    )


@pytest.mark.parametrize(("asked", "expected"), [(0, 1), (99, 6), (2, 2)])
def test_a_count_is_fitted_to_both_ends_of_the_published_range(asked, expected):
    top_level, _, _ = ImageGenerationAdapter._split_image_config(
        {"image_config": {"n": asked}}, allowed_passthrough=frozenset(), record=QWEN_RECORD
    )
    assert top_level["n"] == expected


@pytest.mark.parametrize("prompt", ["a chat-shaped leaf", "a chat-shaped cube"])
@pytest.mark.asyncio
async def test_the_prompt_falls_back_to_the_chat_messages(prompt):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    result = await _posted(
        adapter,
        body={"messages": [{"role": "user", "content": prompt}]},
        responses_body=_StubResponsesBody([]),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    assert result.payload["prompt"] == prompt


@pytest.mark.parametrize(
    ("responses_text", "chat_text"),
    [("a transformed leaf", "a raw leaf"), ("a transformed cube", "a raw cube")],
)
@pytest.mark.asyncio
async def test_the_transformed_prompt_outranks_the_raw_chat_messages(responses_text, chat_text):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    result = await _posted(
        adapter,
        body={"messages": [{"role": "user", "content": chat_text}]},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": responses_text}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    assert result.payload["prompt"] == responses_text, (
        "body['messages'] is the untransformed request; the responses body is what the "
        "transformer produced, and it is the only one that has resolved attachments"
    )


@pytest.mark.asyncio
async def test_the_provider_block_is_read_from_pipe_metadata_when_the_request_has_none():
    from open_webui_openrouter_pipe.core.config import _PIPE_METADATA_KEY

    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["recraft/recraft-v3"] = (time.monotonic(), [RECRAFT_RECORD])

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}], provider=None
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="recraft.recraft-v3",
        api_model_id="recraft/recraft-v3",
        metadata={
            "chat_id": "c",
            "message_id": "m",
            _PIPE_METADATA_KEY: {"provider": {"only": ["recraft"], "data_collection": "deny"}},
        },
    )

    assert result.payload["provider"]["only"] == ["recraft"]
    assert "data_collection" not in result.payload["provider"], (
        "data_collection is not part of ImageGenerationProviderPreferences; forwarding it "
        "would be accepted and ignored"
    )


@pytest.mark.parametrize(
    ("requested", "candidates", "expected"),
    [
        ({"only": ["fal"]}, ["alpha", "beta"], "fal"),
        ({"only": ["openai", "recraft"]}, ["recraft"], "recraft"),
        ({"only": ["openai", "recraft"]}, ["recraft", "openai"], "openai"),
        ({"ignore": ["provider-a"]}, ["provider-a", "provider-b"], "provider-b"),
        ({"ignore": ["provider-a", "provider-b"]}, ["provider-a", "provider-b"], "provider-a"),
        ({"only": ["beta"], "ignore": ["beta"]}, ["alpha", "beta"], "beta"),
        ({"order": ["fal"]}, ["alpha", "beta"], "alpha"),
        ({"order": ["beta"]}, ["alpha", "beta"], "beta"),
        ({"only": ["beta"], "order": ["alpha"]}, ["alpha", "beta"], "beta"),
        ({}, ["alpha", "beta"], "alpha"),
        ({"only": ["fal"]}, [], ""),
    ],
)
def test_a_pin_decides_which_slug_carries_the_options(requested, candidates, expected):
    assert carrier_slug(requested, candidates) == expected, (
        "only is an allow-set, not a precedence order, so when several allowed providers "
        "serve the model the carrier is the allowed one with a published record; a pin the "
        "catalog has not caught up with still carries; ignore removes a provider entirely, "
        "because options keyed to an excluded provider are guaranteed to be dropped"
    )


@pytest.mark.parametrize(
    ("requested", "slug", "params", "expected"),
    [
        ({"only": ["a"], "options": {}}, "a", {"k": 1}, {"only": ["a"], "options": {"a": {"k": 1}}}),
        ({"only": ["a"], "options": {}}, None, {}, {"only": ["a"]}),
        ({"only": ["a"], "options": "junk"}, None, {}, {"only": ["a"]}),
        ({"only": ["a"]}, "a", {}, {"only": ["a"]}),
    ],
)
def test_the_merged_block_never_re_emits_the_callers_options_value(
    requested, slug, params, expected
):
    assert merge_provider_options(requested, slug, params) == expected, (
        "an empty or malformed options value must be dropped, not passed through; the image "
        "adapter gates on `if provider:` so a stray {'options': {}} ships an empty block"
    )


def test_building_the_provider_block_does_not_mutate_the_callers_object():
    import copy

    requested = {"only": ["recraft"], "options": {"recraft": {"controls": {"colors": []}}}}
    before = copy.deepcopy(requested)

    block = merge_provider_options(requested, "other", {"style": "digital_illustration"})
    assert block["options"]["recraft"] is not requested["options"]["recraft"], (
        "a shared inner dict lets a later stage's edit reach back into the request object"
    )

    merge_provider_options(requested, "recraft", {"style": "digital_illustration"})

    assert requested == before, (
        "the derived params must not leak back into the request object later stages read and log"
    )


@pytest.mark.parametrize(
    ("decoded_bytes", "rejected"), [(1024 * 1024 - 4096, False), (1024 * 1024 + 4096, True)]
)
@pytest.mark.asyncio
async def test_the_ceiling_is_measured_in_decoded_bytes(decoded_bytes, rejected):
    import aiohttp

    raw = _png(4, 4) + b"\x00" * (decoded_bytes - len(_png(4, 4)))
    with aioresponses() as mocked:
        mocked.post(f"{BASE}/images", payload={"data": [{"b64_json": _b64(raw)}]})
        async with aiohttp.ClientSession() as session:
            client = await _client(session)
            if rejected:
                with pytest.raises(ImageGenerationError, match="BASE64_MAX_SIZE_MB"):
                    await client.generate({"model": "m", "prompt": "p"}, max_decoded_bytes=1024 * 1024)
            else:
                result = await client.generate(
                    {"model": "m", "prompt": "p"}, max_decoded_bytes=1024 * 1024
                )
                assert len(result.images) == 1


@pytest.mark.parametrize(("uid", "chat"), [("u-alpha", "chat-a"), ("u-beta", "chat-b")])
@pytest.mark.asyncio
async def test_the_requesting_user_is_identified_on_the_wire(uid, chat, monkeypatch):
    from types import SimpleNamespace

    from open_webui_openrouter_pipe.core import config

    def _include(headers, user):
        headers = dict(headers)
        headers["X-OpenWebUI-User-Id"] = str(getattr(user, "id", ""))
        headers["X-OpenWebUI-User-Email"] = str(getattr(user, "email", ""))
        return headers

    monkeypatch.setattr(
        config, "_owui_env", SimpleNamespace(ENABLE_FORWARD_USER_INFO_HEADERS=True)
    )
    monkeypatch.setattr(config, "_owui_include_user_info_headers", _include)
    adapter = _adapter(_KeyPipe("sk-x"))

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        metadata={"chat_id": chat, "message_id": "m1"},
        user_obj=SimpleNamespace(id=uid, email=f"{uid}@example.test", name=uid, role="user"),
    )

    for (method, url), calls in result.requests.items():
        sent = calls[0].kwargs.get("headers") or {}
        assert sent.get("X-OpenWebUI-User-Id") == uid, (
            f"{method} {url} went out unattributed; every outbound request on this path "
            "carries the same bearer token, so a per-user gateway must see all of them"
        )

    headers = result.headers
    assert headers.get("X-OpenWebUI-User-Id") == uid, (
        "every other OpenRouter transport in this package stamps the requester; a gateway that "
        f"authorizes or quotas per user sees image generation unattributed. got {sorted(headers)}"
    )
    assert headers.get("X-OpenWebUI-Chat-Id") == chat


@pytest.mark.parametrize(
    "base_url",
    [
        "https://openrouter.ai/api/v1",
        "https://gateway.internal/openrouter/v1",
        "https://gateway.internal/openrouter/v1/",
    ],
)
@pytest.mark.asyncio
async def test_the_base_url_valve_decides_where_the_request_goes(base_url):
    import aiohttp

    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])
    valves = _StubValves("sk-x")
    valves.BASE_URL = base_url

    with aioresponses() as mocked:
        mocked.post(f"{base_url}/images", payload={"data": [{"b64_json": _b64(_png(8, 8))}]})
        async with aiohttp.ClientSession() as session:
            await adapter.generate(
                body={},
                responses_body=_StubResponsesBody(
                    [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
                ),
                valves=valves,
                session=session,
                event_emitter=None,
                metadata={},
                user=None,
                request=object(),
                user_obj=object(),
                normalized_model_id="m.x",
                api_model_id="m/x",
            )
        urls = [str(key[1]) for key in mocked.requests]

    assert urls == [f"{base_url.rstrip('/')}/images"], (
        "an operator on a self-hosted gateway must not have image generation bypass "
        "BASE_URL, and a trailing slash they typed must not produce a doubled path "
        f"separator; requests went to {urls}"
    )


@pytest.mark.asyncio
async def test_a_blank_api_key_never_reaches_the_network():
    import aiohttp

    with aioresponses() as mocked:
        async with aiohttp.ClientSession() as session:
            client = OpenRouterImageClient(
                session, base_url=BASE, api_key="", logger=_Logger()
            )
            with pytest.raises(RuntimeError):
                await client.list_models()

    assert not mocked.requests, (
        "an empty bearer token must be caught before a request is issued; a 401 from the "
        "wire is classified as a routine transport failure and stamps the attempt TTL"
    )


@pytest.mark.parametrize("knob", ["style", "text_layout"])
@pytest.mark.asyncio
async def test_a_knob_that_cannot_be_keyed_is_reported_not_swallowed(knob):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (
        time.monotonic(),
        [{"allowed_passthrough_parameters": [knob], "supported_parameters": {}}],
    )

    result = await _posted(
        adapter,
        body={"image_config": {knob: "value-x"}},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    assert "provider" not in result.payload
    notes = [
        str(event.get("content", ""))
        for event in result.events
        if event.get("type") == "notification"
    ]
    assert any(knob in note for note in notes), (
        "the endpoint advertised the knob but published no provider slug to key it under; "
        f"dropping it silently is the one hole every other rejection path closes. got {notes!r}"
    )


@pytest.mark.parametrize("image_count", [1, 2])
@pytest.mark.asyncio
async def test_losing_every_image_is_not_quieter_than_losing_some(image_count, caplog):
    import logging as _logging

    adapter = ImageGenerationAdapter(
        pipe=cast(Any, _KeyPipe("sk-x")), logger=_logging.getLogger("test.image.totalloss")
    )
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    async def _no_context(request, user_obj):
        return None, None

    cast(Any, adapter._pipe)._file_gateway.resolve_storage_context = _no_context
    entry = {"b64_json": _b64(_png(8, 8)), "media_type": "image/png"}

    with caplog.at_level(_logging.WARNING):
        result = await _posted(
            adapter,
            body={},
            responses_body=_StubResponsesBody(
                [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
            ),
            valves=_StubValves("sk-x"),
            event_emitter=_Emitter(),
            normalized_model_id="m.x",
            api_model_id="m/x",
            reply={"data": [entry] * image_count},
        )

    warnings = [r for r in caplog.records if r.levelno >= _logging.WARNING]
    assert warnings, (
        "the generation was paid for and every byte was thrown away; an operator at the default "
        "level must not be the last to know"
    )
    assert any(str(image_count) in r.getMessage() and "m/x" in r.getMessage() for r in warnings)
    assert "could not be saved" in result.content


@pytest.mark.asyncio
async def test_a_successful_generation_reports_its_cost_to_the_plugin_layer():
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        reply={"data": [{"b64_json": _b64(_png(8, 8))}], "usage": {"cost": 0.03, "total_tokens": 9}},
    )

    assert len(result.generations) == 1, (
        "without this the usage store writes cost=0 for a request OpenRouter charged for"
    )
    assert result.generations[0]["status"] == "ok"
    assert (result.generations[0]["usage"] or {}).get("cost") == 0.03


@pytest.mark.asyncio
async def test_a_failed_generation_is_reported_as_failed():
    import aiohttp

    pipe = _KeyPipe("sk-x")
    adapter = _adapter(pipe)
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    with aioresponses() as mocked:
        mocked.post(f"{BASE}/images", payload={"data": []})
        async with aiohttp.ClientSession() as session:
            await adapter.generate(
                body={},
                responses_body=_StubResponsesBody(
                    [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
                ),
                valves=_StubValves("sk-x"),
                session=session,
                event_emitter=_Emitter(),
                metadata={},
                user=None,
                request=object(),
                user_obj=object(),
                normalized_model_id="m.x",
                api_model_id="m/x",
            )

    assert [g["status"] for g in pipe.generations] == ["failed"], (
        "a failed generation recorded as ok inflates the success rate the dashboard shows"
    )


@pytest.mark.parametrize("cost", [0.03, 0.11])
@pytest.mark.asyncio
async def test_an_image_generation_reaches_the_redis_cost_export(cost, monkeypatch):
    from types import SimpleNamespace

    import open_webui_openrouter_pipe.integrations.image as image_module

    written: dict[str, Any] = {}

    async def _snapshot(pipe, valves, **kwargs):
        written.update(kwargs)

    monkeypatch.setattr(image_module, "maybe_dump_costs_snapshot", _snapshot)
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        user={"id": "u-alpha"},
        user_obj=SimpleNamespace(id="u-alpha", email="u@example.test", name="u"),
        reply={"data": [{"b64_json": _b64(_png(8, 8))}], "usage": {"cost": cost, "total_tokens": 9}},
    )

    assert (written.get("usage") or {}).get("cost") == cost, (
        "three other billable paths write a cost snapshot; without this one an operator's "
        f"Redis export under-reports by exactly the image spend. got {written!r}"
    )
    assert written.get("model_id") == "m/x"
    assert written.get("user_id") == "u-alpha"


@pytest.mark.parametrize("cost", [0.03, 0.11])
@pytest.mark.asyncio
async def test_a_generation_that_could_not_be_stored_is_reported_as_failed_with_its_cost(cost):
    pipe = _KeyPipe("sk-x")
    adapter = _adapter(pipe)
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    async def _no_context(request, user_obj):
        return None, None

    cast(Any, adapter._pipe)._file_gateway.resolve_storage_context = _no_context

    await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        reply={"data": [{"b64_json": _b64(_png(8, 8))}], "usage": {"cost": cost}},
    )

    assert [(g["status"], (g["usage"] or {}).get("cost")) for g in pipe.generations] == [
        ("failed", cost)
    ], (
        "the pipe-level backstop reports ok/None, so a paid generation whose bytes were all "
        "lost would be recorded as a free success"
    )


@pytest.mark.parametrize("knob", ["image_size", "aspect_ratio"])
@pytest.mark.asyncio
async def test_a_dropped_image_knob_is_findable_in_the_log_without_an_emitter(knob, caplog):
    import logging as _logging

    adapter = ImageGenerationAdapter(
        pipe=cast(Any, _KeyPipe("sk-x")), logger=_logging.getLogger("test.image.drops")
    )
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{"supported_parameters": {}}])

    with caplog.at_level(_logging.DEBUG):
        await _posted(
            adapter,
            body={"image_config": {knob: "2K" if knob == "image_size" else "21:9"}},
            responses_body=_StubResponsesBody(
                [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
            ),
            valves=_StubValves("sk-x"),
            event_emitter=None,
            normalized_model_id="m.x",
            api_model_id="m/x",
        )

    expected = "resolution" if knob == "image_size" else "aspect_ratio"
    records = [r for r in caplog.records if "not sent" in r.getMessage()]
    assert any(expected in r.getMessage() and "m/x" in r.getMessage() for r in records), (
        "the notification channel vanishes when there is no emitter; an operator asked why a "
        f"knob never applied has nothing to read. records were {[r.getMessage() for r in records]}"
    )


@pytest.mark.asyncio
async def test_the_endpoints_envelope_is_read_from_one_key():
    import aiohttp

    with aioresponses() as mocked:
        mocked.get(
            f"{BASE}/images/models/m/x/endpoints",
            payload={"data": [{"provider_slug": "alpha"}]},
        )
        async with aiohttp.ClientSession() as session:
            client = await _client(session)
            records = await client.endpoints("m/x")

    assert records == [], (
        "reading a second key would make a reshaped envelope look like a real contract; an "
        "unrecognised shape must read as 'unknown' so knobs are sent unvalidated, not gated "
        "against an invented record"
    )


@pytest.mark.parametrize("state", ["fresh-fetch", "cache-hit", "stale-refresh-raises", "stale-refresh-empty"])
@pytest.mark.parametrize("pin", ["alpha", "beta"])
@pytest.mark.asyncio
async def test_the_endpoint_contract_follows_the_operator_pin(pin, state):
    records = [
        {"provider_slug": "alpha", "allowed_passthrough_parameters": ["style"], "supported_parameters": {}},
        {"provider_slug": "beta", "allowed_passthrough_parameters": ["style"], "supported_parameters": {}},
    ]

    class _Serving:
        async def endpoints(self, _model_id):
            return records

    class _Raises:
        async def endpoints(self, _model_id):
            raise RuntimeError("upstream said no")

    class _Empty:
        async def endpoints(self, _model_id):
            return []

    adapter = _adapter(_KeyPipe("sk-x"))
    monkeypatch_client(adapter, _Serving())
    if state == "cache-hit":
        await adapter._endpoint_record(None, _StubValves("sk-x"), "m/x")
    elif state != "fresh-fetch":
        adapter._endpoint_cache["m/x"] = (time.monotonic() - 99999.0, records)
        monkeypatch_client(adapter, _Raises() if state == "stale-refresh-raises" else _Empty())

    record, _unserved = await adapter._endpoint_record(
        None, _StubValves("sk-x"), "m/x", requested={"only": [pin]}
    )

    assert record is not None and record["provider_slug"] == pin, (
        "validating knobs against one provider's contract and keying options to it while "
        "routing goes to another means the pinned provider receives nothing. every return "
        f"path must select by the pin, not only the fresh fetch; state={state}"
    )


@pytest.mark.parametrize("order", [["beta", "alpha"], ["alpha", "beta"]])
@pytest.mark.asyncio
async def test_the_endpoint_contract_follows_provider_order_when_there_is_no_only(order):
    import aiohttp

    adapter = _adapter(_KeyPipe("sk-x"))
    records = [
        {"provider_slug": "alpha", "supported_parameters": {}},
        {"provider_slug": "beta", "supported_parameters": {}},
    ]

    with aioresponses() as mocked:
        mocked.get(f"{BASE}/images/models/m/x/endpoints", payload={"endpoints": records})
        async with aiohttp.ClientSession() as session:
            record, _unserved = await adapter._endpoint_record(
                session, _StubValves("sk-x"), "m/x", requested={"order": order}
            )

    assert record is not None and record["provider_slug"] == order[0], (
        "provider.order expresses a preference routing honours; the contract must follow it"
    )


@pytest.mark.asyncio
async def test_the_first_endpoint_wins_when_nothing_constrains_routing():
    import aiohttp

    adapter = _adapter(_KeyPipe("sk-x"))
    records = [
        {"supported_parameters": {"n": {"type": "range", "min": 1, "max": 4}}},
        {"supported_parameters": {"n": {"type": "range", "min": 1, "max": 9}}},
    ]

    with aioresponses() as mocked:
        mocked.get(f"{BASE}/images/models/m/x/endpoints", payload={"endpoints": records})
        async with aiohttp.ClientSession() as session:
            record, _unserved = await adapter._endpoint_record(session, _StubValves("sk-x"), "m/x")

    assert record is not None and record["supported_parameters"]["n"]["max"] == 4, (
        "with nothing to select on, the first published endpoint is the contract; taking the "
        "last would validate every knob against a different provider's limits"
    )


@pytest.mark.asyncio
async def test_an_empty_endpoint_reply_leaves_the_cache_untouched():
    import aiohttp

    adapter = _adapter(_KeyPipe("sk-x"))

    with aioresponses() as mocked:
        mocked.get(f"{BASE}/images/models/m/x/endpoints", payload={"endpoints": []})
        async with aiohttp.ClientSession() as session:
            record, _unserved = await adapter._endpoint_record(session, _StubValves("sk-x"), "m/x")

    assert record is None
    assert "m/x" not in adapter._endpoint_cache, (
        "caching an empty reply freezes 'this model advertises nothing' for the whole TTL, so "
        "every provider knob is dropped until it expires"
    )


@pytest.mark.asyncio
async def test_the_ceiling_stops_the_loop_before_it_decodes(monkeypatch):
    import aiohttp
    import base64 as _base64

    import open_webui_openrouter_pipe.integrations.image_client as client_module

    decoded: list[int] = []
    real = _base64.b64decode

    def _spy(blob, **kwargs):
        decoded.append(len(blob))
        return real(blob, **kwargs)

    monkeypatch.setattr(client_module.base64, "b64decode", _spy)
    entry = {"b64_json": _b64(_png(4, 4) + b"\x00" * (600 * 1024))}

    with aioresponses() as mocked:
        mocked.post(f"{BASE}/images", payload={"data": [entry, entry]})
        async with aiohttp.ClientSession() as session:
            client = await _client(session)
            with pytest.raises(ImageGenerationError, match="BASE64_MAX_SIZE_MB"):
                await client.generate({"model": "m", "prompt": "p"}, max_decoded_bytes=1024 * 1024)

    assert len(decoded) == 1, (
        "the pre-decode estimate exists so an oversized reply is refused without materialising "
        f"every blob; {len(decoded)} entries were decoded"
    )


@pytest.mark.asyncio
async def test_an_http_failure_on_the_endpoint_lookup_reads_as_a_transport_failure(caplog):
    import aiohttp
    import logging as _logging

    adapter = ImageGenerationAdapter(
        pipe=cast(Any, _KeyPipe("sk-x")), logger=_logging.getLogger("test.image.transport")
    )

    with caplog.at_level(_logging.DEBUG), aioresponses() as mocked:
        mocked.get(f"{BASE}/images/models/m/x/endpoints", status=500, payload={"error": "boom"})
        async with aiohttp.ClientSession() as session:
            await adapter._endpoint_record(session, _StubValves("sk-x"), "m/x")

    assert "lookup failed" in caplog.text, (
        "an operator debugging a 401 must not be told the model advertises nothing"
    )
    assert "no endpoint record" not in caplog.text


@pytest.mark.asyncio
async def test_a_reply_whose_entries_are_all_unusable_blames_the_envelope_not_storage():
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        reply={"data": ["oops"]},
    )

    assert "no usable images" in result.content, (
        "the two failure modes must not be interchangeable; an operator needs to know whether "
        "OpenRouter sent nothing usable or Open WebUI storage refused it"
    )
    assert "could not be saved" not in result.content, (
        "an envelope the pipe cannot parse is an upstream contract break, not a storage failure"
    )


@pytest.mark.parametrize("status", [402, 429])
@pytest.mark.asyncio
async def test_the_error_card_names_the_model_that_was_requested(status):
    import aiohttp

    pipe = _KeyPipe("sk-x", record_errors=True)
    adapter = _adapter(pipe)
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    with aioresponses() as mocked:
        mocked.post(f"{BASE}/images", status=status, payload={"error": {"message": "nope"}})
        async with aiohttp.ClientSession() as session:
            await adapter.generate(
                body={},
                responses_body=_StubResponsesBody(
                    [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
                ),
                valves=_StubValves("sk-x"),
                session=session,
                event_emitter=_Emitter(),
                metadata={},
                user=None,
                request=object(),
                user_obj=object(),
                normalized_model_id="m.x",
                api_model_id="m/x",
            )

    assert pipe.reports is not None
    assert pipe.reports.calls[0]["exc"].requested_model == "m/x"


@pytest.mark.parametrize("elapsed_is_zero", [False])
@pytest.mark.asyncio
async def test_the_final_status_reports_a_real_duration(elapsed_is_zero, monkeypatch):
    import open_webui_openrouter_pipe.integrations.image as image_module

    ticks = iter([100.0, 100.0, 142.5, 142.5, 142.5])
    monkeypatch.setattr(image_module.time, "monotonic", lambda: next(ticks, 142.5))
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (142.5, [{}])

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    final = [d for d in result.statuses if d][-1]
    assert "0.00s" not in final, f"a 42s generation must not report zero elapsed. got {final!r}"


@pytest.mark.asyncio
async def test_the_user_is_told_generation_started():
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    assert any("Generating image" in d for d in result.statuses), (
        "image generation is not streamed; without a progress status the UI is silent for the "
        "whole call"
    )


@pytest.mark.asyncio
async def test_a_dropped_knob_notice_arrives_as_a_warning():
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{"supported_parameters": {}}])

    result = await _posted(
        adapter,
        body={"image_config": {"aspect_ratio": "21:9"}},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    notices = [e for e in result.events if e.get("type") == "notification"]
    assert notices and notices[0].get("level") == "warning", (
        "an ignored setting is not informational; it is something the user asked for and did "
        f"not get. got {notices!r}"
    )


@pytest.mark.parametrize("uid", ["u-meta-alpha", "u-meta-beta"])
@pytest.mark.asyncio
async def test_the_requester_id_falls_back_to_the_metadata_user_id(uid):
    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        normalized_model_id="m.x",
        api_model_id="m/x",
        metadata={"chat_id": "c", "message_id": "m", "user_id": uid},
        user=None,
    )

    assert result.calls[0]["owui_user_id"] == uid


@pytest.mark.asyncio
async def test_one_model_never_inherits_another_models_contract():
    import aiohttp

    adapter = _adapter(_KeyPipe("sk-x"))
    adapter._endpoint_cache["a/one"] = (
        time.monotonic(),
        [{"provider_slug": "alpha", "allowed_passthrough_parameters": ["style"]}],
    )

    with aioresponses() as mocked:
        mocked.get(f"{BASE}/images/models/b/two/endpoints", status=500, payload={})
        async with aiohttp.ClientSession() as session:
            record, _unserved = await adapter._endpoint_record(session, _StubValves("sk-x"), "b/two")

    assert record is None, (
        "a cache miss must not fall back to whatever another model published; that would send "
        "b/two's request under alpha's slug"
    )


@pytest.mark.parametrize(
    ("mime", "payload", "ext"),
    [("image/svg+xml", "PHN2Zy8+", "svg"), ("image/png", "iVBORw0KGgo=", "png")],
)
@pytest.mark.asyncio
async def test_an_inbound_data_url_is_stored_with_the_shared_extension(mime, payload, ext):
    import contextlib

    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.core.errors import RequiredInternalFileError
    from open_webui_openrouter_pipe.requests.transformer import transform_messages_to_input

    _register_live_image_catalog()
    pipe = Pipe()
    saved: list[str] = []

    async def _upload(**kwargs):
        saved.append(str(kwargs.get("filename")))
        return "file-1"

    async def _context(request, user_obj):
        return object(), object()

    cast(Any, pipe)._file_gateway.upload_to_owui_storage = _upload
    cast(Any, pipe)._file_gateway.resolve_storage_context = _context

    try:
        with contextlib.suppress(RequiredInternalFileError):
            await transform_messages_to_input(
                pipe,
                [
                    {
                        "role": "user",
                        "id": "m1",
                        "content": [
                            {"type": "text", "text": "here"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{payload}"},
                            },
                        ],
                    }
                ],
                model_id="recraft/recraft-v3",
                openwebui_model_id="recraft/recraft-v3",
                capability_model_id="recraft/recraft-v3",
                valves=pipe.valves,
            )
    finally:
        await pipe.close()

    assert saved, "the attachment was never persisted, so this asserts nothing"
    assert saved[0].endswith(f".{ext}"), (
        "a naive mime split yields 'image-<hex>.svg+xml' here; every persistence path must "
        f"derive the extension from the shared helper. got {saved[0]!r}"
    )


@pytest.mark.parametrize(
    ("mime", "ext"), [("image/jpg", "jpeg"), ("image/svg+xml", "svg"), ("image/png", "png")]
)
@pytest.mark.asyncio
async def test_a_downloaded_image_is_stored_with_the_shared_extension(mime, ext):
    import contextlib

    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.core.errors import RequiredInternalFileError
    from open_webui_openrouter_pipe.requests.transformer import transform_messages_to_input

    _register_live_image_catalog()
    pipe = Pipe()
    saved: list[str] = []

    async def _upload(**kwargs):
        saved.append(str(kwargs.get("filename")))
        return "file-1"

    async def _context(request, user_obj):
        return object(), object()

    async def _download(_url):
        return {"data": _png(8, 8), "mime_type": mime}

    cast(Any, pipe)._file_gateway.upload_to_owui_storage = _upload
    cast(Any, pipe)._file_gateway.resolve_storage_context = _context
    cast(Any, pipe)._multimodal_handler._download_remote_url = _download

    try:
        with contextlib.suppress(RequiredInternalFileError):
            await transform_messages_to_input(
                pipe,
                [
                    {
                        "role": "user",
                        "id": "m1",
                        "content": [
                            {"type": "text", "text": "here"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.invalid/pictures/leaf"},
                            },
                        ],
                    }
                ],
                model_id="recraft/recraft-v3",
                openwebui_model_id="recraft/recraft-v3",
                capability_model_id="recraft/recraft-v3",
                valves=pipe.valves,
            )
    finally:
        await pipe.close()

    assert saved, "the download was never persisted, so this asserts nothing"
    assert saved[0].endswith(f".{ext}"), (
        "the download branch is the second call site the changeset redirected to the shared "
        f"helper; a naive mime split reverts it silently. got {saved[0]!r}"
    )


@pytest.mark.asyncio
async def test_the_dropped_knob_latch_does_not_widen_with_the_rejected_value(caplog):
    import logging as _logging

    from open_webui_openrouter_pipe.integrations import image as image_module

    image_module._warned_dropped_image_param.clear()
    adapter = ImageGenerationAdapter(
        pipe=cast(Any, _KeyPipe("sk-x")), logger=_logging.getLogger("test.image.latchkey")
    )
    adapter._endpoint_cache["qwen/qwen-image-3"] = (time.monotonic(), [QWEN_RECORD])

    with caplog.at_level(_logging.DEBUG):
        for ratio in ("21:9", "99:1", "7:3", "5:2"):
            await _posted(
                adapter,
                body={"image_config": {"aspect_ratio": ratio}},
                responses_body=_StubResponsesBody(
                    [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
                ),
                valves=_StubValves("sk-x"),
                event_emitter=_Emitter(),
                normalized_model_id="qwen.qwen-image-3",
                api_model_id="qwen/qwen-image-3",
            )

    assert len(image_module._warned_dropped_image_param) == 1, (
        "the latch key must be drawn from a closed set; keying it on the rendered note means "
        "the user's rejected value widens it, so warn-once never latches and the set grows "
        f"without bound. entries: {sorted(image_module._warned_dropped_image_param)}"
    )
    levels = [r.levelno for r in caplog.records if "not sent" in r.getMessage()]
    assert levels == [
        _logging.WARNING,
        _logging.DEBUG,
        _logging.DEBUG,
        _logging.DEBUG,
    ], f"one knob on one model must warn once, then drop to DEBUG. got {levels!r}"


@pytest.mark.parametrize("junk_keys", [200, 500])
@pytest.mark.parametrize("advertised", [0, 3000])
def test_one_request_cannot_generate_unbounded_notes(junk_keys, advertised):
    config = {f"junk_{index}": "x" for index in range(junk_keys)}
    _, _, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": config},
        allowed_passthrough=frozenset(f"p{index}" for index in range(advertised)),
        record=RECRAFT_RECORD,
    )

    from open_webui_openrouter_pipe.integrations.image import (
        _PROVIDER_KEY_REPORT_LIMIT,
        _TOP_LEVEL_PARAMS,
    )

    ceiling = len(_TOP_LEVEL_PARAMS) + _PROVIDER_KEY_REPORT_LIMIT + 1
    assert len(notes) <= ceiling, (
        "image_config is an extra-allow request field, so an uncapped note per key lets one "
        "request arm one latch entry per key for the life of the worker. The ceiling must "
        "also be a pipe-local constant: deriving it from the reply's advertised passthrough "
        f"list lets an upstream widen it. advertised={advertised}, got {len(notes)}"
    )
    assert any("further image_config key" in note.text for note in notes), (
        "the remainder must still be reported as a count, not silently dropped"
    )


@pytest.mark.parametrize("cost", [0.03, 0.11])
@pytest.mark.asyncio
async def test_a_billed_reply_the_pipe_cannot_decode_is_still_costed(cost, monkeypatch):
    import aiohttp
    from types import SimpleNamespace

    import open_webui_openrouter_pipe.integrations.image as image_module

    snapshots: list[dict[str, Any]] = []

    async def _snapshot(pipe, valves, **kwargs):
        snapshots.append(kwargs)

    monkeypatch.setattr(image_module, "maybe_dump_costs_snapshot", _snapshot)
    pipe = _KeyPipe("sk-x")
    adapter = _adapter(pipe)
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{}])
    oversized = _b64(_png(4, 4) + b"\x00" * (4 * 1024 * 1024))

    with aioresponses() as mocked:
        mocked.post(
            f"{BASE}/images",
            payload={"data": [{"b64_json": oversized}], "usage": {"cost": cost, "total_tokens": 9}},
        )
        async with aiohttp.ClientSession() as session:
            await adapter.generate(
                body={},
                responses_body=_StubResponsesBody(
                    [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
                ),
                valves=_StubValves("sk-x", base64_max_size_mb=1),
                session=session,
                event_emitter=_Emitter(),
                metadata={},
                user={"id": "u1"},
                request=object(),
                user_obj=SimpleNamespace(id="u1", email="u@example.test", name="u"),
                normalized_model_id="m.x",
                api_model_id="m/x",
            )

    assert [(g["status"], (g["usage"] or {}).get("cost")) for g in pipe.generations] == [
        ("failed", cost)
    ], (
        "OpenRouter answered 200 and charged for the image; the pipe failing to decode it does "
        "not make the request free"
    )
    assert snapshots and (snapshots[0].get("usage") or {}).get("cost") == cost, (
        "the Redis cost export must carry the billed cost of a request that failed after 200"
    )


@pytest.mark.asyncio
async def test_the_latch_key_does_not_widen_with_an_unknown_request_key():
    import logging as _logging

    from open_webui_openrouter_pipe.integrations import image as image_module

    image_module._warned_dropped_image_param.clear()
    adapter = ImageGenerationAdapter(
        pipe=cast(Any, _KeyPipe("sk-x")), logger=_logging.getLogger("test.image.latchkeys")
    )
    adapter._endpoint_cache["m/x"] = (time.monotonic(), [{"supported_parameters": {}}])

    for suffix in ("alpha", "beta", "gamma"):
        await _posted(
            adapter,
            body={"image_config": {f"unknown_{suffix}": "x"}},
            responses_body=_StubResponsesBody(
                [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
            ),
            valves=_StubValves("sk-x"),
            event_emitter=_Emitter(),
            normalized_model_id="m.x",
            api_model_id="m/x",
        )

    assert len(image_module._warned_dropped_image_param) == 1, (
        "image_config is extra-allow, so a key drawn from the request widens the latch domain "
        "without limit across requests even when each single request is budgeted. entries: "
        f"{sorted(image_module._warned_dropped_image_param)}"
    )


@pytest.mark.parametrize(
    "advertised",
    [["style"], ["style", "text_layout"], ["style", "controls", "text_layout"]],
)
def test_help_advertises_only_what_the_request_path_will_actually_send(advertised):
    """Help is a model's only in-product documentation.

    Naming a control the very next message withholds sends the user to something that
    cannot work. Both surfaces read the same record, so the check is that everything help
    names survives the adapter's split rather than being reported back as not offered.
    """
    from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter
    from open_webui_openrouter_pipe.integrations.image_help import render_image_help

    record = {
        "provider_slug": "recraft",
        "supported_parameters": {"aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]}},
        "allowed_passthrough_parameters": advertised,
    }
    model = {"id": "recraft/recraft-v3", "name": "Recraft V3"}
    rendered = render_image_help("recraft/recraft-v3", model, endpoint_record=record)

    controls = rendered.split("## Controls", 1)[1]
    named = [
        line.split("**")[1]
        for line in controls.splitlines()
        if line.startswith("- **") and "**" in line[4:]
    ]
    assert named, "the model publishes controls, so help must list them"

    for passthrough in advertised:
        assert passthrough in named, f"{passthrough} is published and must be named"

    # Everything help named must survive the split the request path performs.
    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )

    spec = build_image_model_filter_spec("recraft/recraft-v3", model, record)
    config = {name: values[0] for name, values in spec.enums}
    config.update({name: "a_value" for name in spec.passthrough})
    top_level, provider, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": config}, allowed_passthrough=tuple(advertised), record=record
    )
    assert notes == [], f"help named something the request path refuses: {notes}"
    assert set(top_level) | set(provider) == set(config), (
        f"every named control must arrive somewhere; top={top_level} provider={provider}"
    )


@pytest.mark.parametrize(
    ("descriptor", "asked", "fitted", "note"),
    [
        ({"type": "boolean"}, 12345, 12345, ""),
        ({"type": "boolean"}, 0, 0, ""),
        ({"type": "boolean"}, True, None, "expects a number"),
        ({"type": "boolean"}, False, None, "expects a number"),
        ({"type": "boolean"}, "abc", None, "expects a number"),
    ],
)
def test_a_supported_parameter_carries_a_number(descriptor, asked, fitted, note):
    """OpenRouter writes `boolean` to say the model SUPPORTS a parameter.

    Their own model schema words it "whether the model supports deterministic generation
    via seed parameter", and such a descriptor never carries a domain. The request takes
    a number, so anything else is refused rather than forwarded unexamined.
    """
    from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter

    got, got_note = ImageGenerationAdapter._fit_descriptor(descriptor, asked)
    assert got == fitted, f"{asked!r} should fit to {fitted!r}, got {got!r}"
    assert got_note == note


@pytest.mark.asyncio
async def test_help_is_given_every_published_record_not_the_one_a_request_would_use():
    """Describing a model and routing a request are different questions.

    `_select_endpoint` picks one provider because a request has to go somewhere. Handing
    that single record to help would list controls built from one provider's superset,
    while the model's filter is built from the intersection -- so help would name
    controls the chat UI does not draw.
    """
    import aiohttp

    from open_webui_openrouter_pipe.filters.image_filter_renderer import (
        build_image_model_filter_spec,
    )

    wide = {
        "provider_slug": "a",
        "supported_parameters": {"aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]}},
    }
    narrow = {
        "provider_slug": "b",
        "supported_parameters": {"aspect_ratio": {"type": "enum", "values": ["4:3"]}},
    }

    adapter = _adapter(_KeyPipe("sk-x"))
    with aioresponses() as mocked:
        mocked.get(
            f"{BASE}/images/models/vendor/model/endpoints",
            payload={"endpoints": [wide, narrow]},
        )
        async with aiohttp.ClientSession() as session:
            records = await adapter._published_records(
                session, _StubValves("sk-x"), "vendor/model"
            )

    assert len(records) == 2, (
        f"every published record must reach help, not just the routed one; got {records}"
    )

    spec = build_image_model_filter_spec(
        "vendor/model", {"id": "vendor/model", "name": "M"}, records
    )
    assert spec.knob_count == 0, (
        "the providers disagree, so the filter offers nothing -- and help built from the "
        "same records must say the same"
    )

    from_first_only = build_image_model_filter_spec(
        "vendor/model", {"id": "vendor/model", "name": "M"}, records[:1]
    )
    assert from_first_only.knob_count == 1, (
        "sanity: one record alone would offer a control, which is what makes the "
        "difference between the two shapes observable"
    )

