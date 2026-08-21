"""The OpenRouter video HTTP client, exercised over a real aiohttp session.

Every method here reaches the network, and none of it was executed by any test: a
mutation pass typo'd the submit URL to ``/vidoes``, deleted the ``status`` error
branch, and returned an empty dict from ``bearer_header`` -- all three at once -- and
the whole suite stayed green. A typo'd URL sends every video job to a 404, a missing
error branch parses a 500 body as a job status, and an empty auth header makes the
content download anonymous.

Asserted against the request that actually goes out and the response the client
actually returns, not against the strings the source happens to contain.
"""

from __future__ import annotations

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from open_webui_openrouter_pipe.integrations.video_client import (
    OpenRouterVideoClient,
    extension_for_video_mime,
)
from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError
from open_webui_openrouter_pipe.integrations.video_types import VideoGenerationError

BASE = "https://openrouter.ai/api/v1"
# BASE_URL is an operator setting whose own description is "Override this if you are
# using a gateway or proxy", and the key is resolved per request. With one value of each,
# a client that ignored both and hardcoded openrouter.ai under a fixed token passed this
# whole file -- so every URL and header assertion runs against two of each.
GATEWAY = "https://gateway.internal/openrouter/v1"


@pytest.fixture(params=[BASE, GATEWAY], ids=["direct", "gateway"])
def base_url(request) -> str:
    return request.param


async def _client(session, base_url: str = BASE, **kwargs) -> OpenRouterVideoClient:
    import logging

    return OpenRouterVideoClient(
        session,
        base_url=base_url,
        api_key=kwargs.pop("api_key", "sk-test-key"),
        logger=logging.getLogger("video-client-test"),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_submit_posts_to_the_videos_endpoint_and_returns_the_job(base_url):
    """The URL is the whole contract: a typo here 404s every generation."""
    async with aiohttp.ClientSession() as session:
        client = await _client(session, base_url=base_url)
        with aioresponses() as http:
            http.post(f"{base_url}/videos", payload={"id": "job-1", "status": "queued"})
            data = await client.submit({"model": "test/video", "prompt": "a cat"})

            requests = [(m, str(u)) for (m, u) in http.requests]
            assert requests == [("POST", f"{base_url}/videos")], (
                f"submit sent {requests!r}; anything other than POST {base_url}/videos "
                "never reaches the operator's configured host"
            )
        assert data == {"id": "job-1", "status": "queued"}


@pytest.mark.asyncio
async def test_submit_raises_with_the_providers_message_on_an_error_status():
    """A 4xx body carries the reason, and it has to arrive in the shape the templates read.

    The error blocks users see -- heading, error id, provider message, request id -- are
    rendered from an `OpenRouterAPIError`. Raising a bare `VideoGenerationError` meant every
    media failure bypassed those templates and dumped whatever string came off the wire into
    the chat, so the type is part of the contract, not an implementation detail.
    """
    async with aiohttp.ClientSession() as session:
        client = await _client(session)
        with aioresponses() as http:
            http.post(
                f"{BASE}/videos",
                status=402,
                payload={"error": {"message": "Insufficient credits"}},
            )
            with pytest.raises(OpenRouterAPIError) as excinfo:
                await client.submit({"model": "test/video"})
    assert excinfo.value.status == 402
    assert "Insufficient credits" in str(excinfo.value.openrouter_message)


@pytest.mark.asyncio
async def test_submit_rejects_a_non_object_response():
    async with aiohttp.ClientSession() as session:
        client = await _client(session)
        with aioresponses() as http:
            http.post(f"{BASE}/videos", payload=["not", "an", "object"])
            with pytest.raises(VideoGenerationError):
                await client.submit({"model": "test/video"})


@pytest.mark.asyncio
async def test_status_raises_rather_than_parsing_an_error_body_as_a_job(base_url):
    """Without the error branch a 500 body is read back as a job state.

    That is the dangerous direction: the caller polls, sees no terminal status in the
    error payload, and keeps polling a job that failed.
    """
    async with aiohttp.ClientSession() as session:
        client = await _client(session, base_url=base_url)
        with aioresponses() as http:
            http.get(
                f"{base_url}/videos/job-1",
                status=500,
                payload={"error": {"message": "upstream exploded"}},
            )
            with pytest.raises(OpenRouterAPIError) as excinfo:
                await client.status("job-1")
    assert excinfo.value.status == 500
    assert "upstream exploded" in str(excinfo.value.openrouter_message)


@pytest.mark.asyncio
async def test_status_returns_the_job_state_on_success(base_url):
    async with aiohttp.ClientSession() as session:
        client = await _client(session, base_url=base_url)
        with aioresponses() as http:
            http.get(f"{base_url}/videos/job-1", payload={"id": "job-1", "status": "completed"})
            data = await client.status("job-1")
    assert data == {"id": "job-1", "status": "completed"}


@pytest.mark.asyncio
async def test_status_refuses_an_empty_job_id():
    async with aiohttp.ClientSession() as session:
        client = await _client(session)
        with pytest.raises(VideoGenerationError):
            await client.status("   ")


@pytest.mark.parametrize("api_key", ["sk-test-key", "sk-second-distinct-key"])
@pytest.mark.asyncio
async def test_submit_sends_the_configured_key_on_the_wire(api_key):
    """`bearer_header()` is not what submit sends -- `_headers()` is, and nothing read it.

    A key hardcoded inside `_headers` passed the whole file: the only Authorization
    assertion was on `bearer_header()`, which the content download uses and submit does
    not. Read off the outbound request instead, over two distinct keys.
    """
    seen: dict = {}

    def _capture(url, **kwargs):
        seen.update(kwargs.get("headers") or {})
        return CallbackResult(status=200, payload={"id": "job-1", "status": "queued"})

    async with aiohttp.ClientSession() as session:
        client = await _client(session, api_key=api_key)
        with aioresponses() as http:
            http.post(f"{BASE}/videos", callback=_capture)
            await client.submit({"model": "test/video", "prompt": "a cat"})

    assert seen.get("Authorization") == f"Bearer {api_key}", (
        f"submit sent Authorization={seen.get('Authorization')!r} for key {api_key!r}; "
        "every video job would go out under the wrong credential"
    )


@pytest.mark.parametrize("api_key", ["sk-test-key", "sk-second-distinct-key"])
@pytest.mark.asyncio
async def test_the_bearer_header_actually_carries_the_key(api_key):
    """The content download is authorised by this header alone.

    Returning an empty dict here makes that request anonymous, which OpenRouter
    rejects -- and no test noticed, because nothing read the header.
    """
    async with aiohttp.ClientSession() as session:
        client = await _client(session, api_key=api_key)
        headers = client.bearer_header()
    assert headers.get("Authorization") == f"Bearer {api_key}", (
        f"bearer_header produced {headers!r} for key {api_key!r}; the video content "
        "download would go out unauthenticated, or under somebody else's key"
    )


@pytest.mark.asyncio
async def test_a_missing_api_key_is_refused_before_any_request():
    async with aiohttp.ClientSession() as session:
        client = await _client(session, api_key="")
        with pytest.raises(VideoGenerationError):
            client.bearer_header()
        with pytest.raises(VideoGenerationError):
            await client.submit({"model": "test/video"})


@pytest.mark.parametrize("job_id", ["job-1", "vid_29f4c0"])
@pytest.mark.asyncio
async def test_the_content_url_addresses_the_jobs_own_content(job_id, base_url):
    """Two ids AND two bases, because one of each is satisfied by a constant.

    Verified: with a single id, hardcoding `f"{base}/videos/job-1/content"` passed; with
    a single base, hardcoding the whole openrouter.ai URL passed both ids. This URL
    fetches the finished render, so a constant serves one job's video to everyone, and an
    operator behind a gateway has every download go to the wrong host.
    """
    async with aiohttp.ClientSession() as session:
        client = await _client(session, base_url=base_url)
        assert client.content_url(job_id) == f"{base_url}/videos/{job_id}/content", (
            f"content_url({job_id!r}) produced {client.content_url(job_id)!r} against "
            f"base {base_url!r}; a URL that does not carry both the caller's job id and "
            "the configured base downloads from the wrong place"
        )


@pytest.mark.parametrize(
    ("mime", "expected"),
    [
        ("video/webm", ".webm"),
        ("video/webm; codecs=vp9", ".webm"),
        ("VIDEO/WEBM", ".webm"),
        ("video/mp4", ".mp4"),
        ("", ".mp4"),
    ],
)
def test_the_extension_follows_the_mime_type(mime, expected):
    assert extension_for_video_mime(mime) == expected



# ============================================================================
# REGFIX: video client -- index, poll_url, output_count
# ============================================================================


@pytest.mark.parametrize(("index", "suffix"), [(0, ""), (1, "?index=1"), (3, "?index=3")])
@pytest.mark.parametrize("job_id", ["job-1", "vid_29f4c0"])
@pytest.mark.asyncio
async def test_the_content_url_addresses_each_output_by_index(job_id, base_url, index, suffix):
    async with aiohttp.ClientSession() as session:
        client = await _client(session, base_url=base_url)
        assert client.content_url(job_id, index) == (
            f"{base_url}/videos/{job_id}/content{suffix}"
        )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"unsigned_urls": ["a", "b", "c"]}, 3),
        ({"unsigned_urls": ["a"]}, 1),
        ({"unsigned_urls": []}, 1),
        ({}, 1),
        ({"unsigned_urls": "nope"}, 1),
        ({"unsigned_urls": ["a", "", None, "b"]}, 2),
    ],
)
def test_the_output_count_comes_from_the_urls_the_api_returned(payload, expected):
    assert OpenRouterVideoClient.output_count(payload) == expected


@pytest.mark.parametrize(
    ("polling_url", "expected_path"),
    [
        ("/api/v1/videos/job-9", "/api/v1/videos/job-9"),
        ("/api/v2/videos/job-9/state", "/api/v2/videos/job-9/state"),
        ("https://evil.example/steal", None),
        ("", None),
        (None, None),
        ("<base>/videos/job-9", "<base>/videos/job-9"),
        ("<base>", "<base>"),
        ("<base>x/videos/job-9", None),
    ],
)
@pytest.mark.asyncio
async def test_the_poll_url_follows_the_api_but_never_leaves_the_configured_origin(
    base_url, polling_url, expected_path
):
    async with aiohttp.ClientSession() as session:
        client = await _client(session, base_url=base_url)
        wanted = polling_url
        if isinstance(wanted, str) and wanted.startswith("<base>"):
            wanted = base_url + wanted[len("<base>"):]
        resolved = client.poll_url("job-9", wanted)

        if expected_path is None:
            assert resolved == f"{base_url}/videos/job-9"
        elif expected_path.startswith("<base>"):
            assert resolved == base_url + expected_path[len("<base>"):]
        else:
            origin = "/".join(base_url.split("/", 3)[:3])
            assert resolved == f"{origin}{expected_path}"


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        (["image", "text"], ["image", "text"]),
        (["video", "image", "audio", "text"], ["video", "image", "audio", "text"]),
    ],
)
@pytest.mark.asyncio
async def test_the_accepted_input_kinds_are_read_off_the_endpoints_contract(
    base_url, declared, expected
):
    """The declared-modality gate's only source, parsed off the wire the API actually sends.

    `/videos/models` publishes no `architecture` block, so `GET /models/{id}/endpoints`
    is the sole place a video model states which reference kinds it accepts, and the
    gates that hide reference controls read nothing else. Every test of those gates fed
    the list in by hand or replaced the whole client, so `model_modalities` never ran:
    reading `output_modalities` instead, or returning a constant, was invisible.

    Two different declarations, so a constant cannot satisfy both.
    """
    async with aiohttp.ClientSession() as session:
        client = await _client(session, base_url=base_url)
        with aioresponses() as http:
            http.get(
                f"{base_url}/models/vendor/clip-1/endpoints",
                payload={
                    "data": {
                        "id": "vendor/clip-1",
                        "architecture": {
                            "input_modalities": declared,
                            "output_modalities": ["video"],
                        },
                        "endpoints": [{"provider_name": "Vendor"}],
                    }
                },
            )
            found = await client.model_modalities("vendor/clip-1")

            requests = [(m, str(u)) for (m, u) in http.requests]
            assert requests == [("GET", f"{base_url}/models/vendor/clip-1/endpoints")], (
                f"the kinds were read from {requests!r}, which is not the endpoint that "
                "publishes them"
            )
        assert found == expected


@pytest.mark.parametrize("status", [404, 500])
@pytest.mark.asyncio
async def test_a_contract_that_cannot_be_read_declares_nothing_rather_than_refusing(status):
    """An unreadable contract must not read as "this model accepts no references".

    The gate treats an empty list as "nothing declared, offer every control", so a bad
    afternoon on one endpoint must not silently strip reference controls fleet-wide --
    and it must not raise, because this runs inside the catalog load for every model.
    """
    async with aiohttp.ClientSession() as session:
        client = await _client(session)
        with aioresponses() as http:
            http.get(f"{BASE}/models/vendor/clip-1/endpoints", status=status, payload={})
            assert await client.model_modalities("vendor/clip-1") == []


@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"architecture": {"input_modalities": "image"}}},
        {"data": {"architecture": {}}},
        {"data": {}},
        {},
    ],
)
@pytest.mark.asyncio
async def test_a_contract_missing_the_declaration_is_not_read_as_an_empty_one(payload):
    """Four shapes the endpoint can send, none of which states the kinds."""
    async with aiohttp.ClientSession() as session:
        client = await _client(session)
        with aioresponses() as http:
            http.get(f"{BASE}/models/vendor/clip-1/endpoints", payload=payload)
            assert await client.model_modalities("vendor/clip-1") == []


@pytest.mark.asyncio
async def test_a_blank_model_id_is_never_turned_into_a_request():
    """A blank id would address `/models//endpoints`, which is somebody else's resource."""
    async with aiohttp.ClientSession() as session:
        client = await _client(session)
        with aioresponses() as http:
            assert await client.model_modalities("  ") == []
            assert list(http.requests) == [], "a blank id still reached the network"


@pytest.mark.parametrize(
    ("published", "expected_ids"),
    [
        ([{"id": "vendor/a"}, {"id": "vendor/b"}], ["vendor/a", "vendor/b"]),
        ([{"id": "vendor/c"}], ["vendor/c"]),
    ],
)
@pytest.mark.asyncio
async def test_the_video_catalog_is_read_from_the_videos_models_endpoint(
    base_url, published, expected_ids
):
    """The catalog every video filter is built from, taken off the wire.

    Two different catalogs, so a hardcoded list cannot satisfy both.
    """
    async with aiohttp.ClientSession() as session:
        client = await _client(session, base_url=base_url)
        with aioresponses() as http:
            http.get(f"{base_url}/videos/models", payload={"data": published})
            models = await client.list_models()

            requests = [(m, str(u)) for (m, u) in http.requests]
            assert requests == [("GET", f"{base_url}/videos/models")], (
                f"the catalog was fetched from {requests!r}"
            )
        assert [entry["id"] for entry in models] == expected_ids


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"data": [{"id": "vendor/a"}, "not-a-model", 7]}, [{"id": "vendor/a"}]),
        ({"data": "not-a-list"}, []),
        ({}, []),
    ],
)
@pytest.mark.asyncio
async def test_only_the_catalog_entries_that_are_objects_reach_the_registry(payload, expected):
    """Anything that is not an object would be registered as a model nobody can call."""
    async with aiohttp.ClientSession() as session:
        client = await _client(session)
        with aioresponses() as http:
            http.get(f"{BASE}/videos/models", payload=payload)
            assert await client.list_models() == expected


@pytest.mark.asyncio
async def test_a_catalog_fetch_that_fails_raises_rather_than_registering_nothing():
    """An empty catalog and a failed fetch are different events for the loader above."""
    async with aiohttp.ClientSession() as session:
        client = await _client(session)
        with aioresponses() as http:
            http.get(f"{BASE}/videos/models", status=500, payload={"error": {"message": "down"}})
            with pytest.raises(aiohttp.ClientResponseError):
                await client.list_models()
