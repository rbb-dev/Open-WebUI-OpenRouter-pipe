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
    """A 4xx body carries the reason; swallowing it strands the user on a generic error."""
    async with aiohttp.ClientSession() as session:
        client = await _client(session)
        with aioresponses() as http:
            http.post(
                f"{BASE}/videos",
                status=402,
                payload={"error": {"message": "Insufficient credits"}},
            )
            with pytest.raises(VideoGenerationError) as excinfo:
                await client.submit({"model": "test/video"})
    assert "Insufficient credits" in str(excinfo.value)


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
            with pytest.raises(VideoGenerationError) as excinfo:
                await client.status("job-1")
    assert "upstream exploded" in str(excinfo.value)


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
