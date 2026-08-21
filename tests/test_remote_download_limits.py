"""The size cap and MIME allowlist are enforced in exactly one place.

``_download_remote_url_streaming`` is the sole enforcement point for the
``REMOTE_VIDEO_MAX_SIZE_MB`` and ``VIDEO_OUTPUT_MIME_ALLOWLIST`` valves, and none of it
was ever executed by a test: a mutation pass disabled both -- the size comparison
replaced with ``if False`` and the allowlist check inverted -- and the whole suite
stayed green. Two operator-facing controls could be turned off without a single
assertion noticing.

These drive the real function with a stubbed transport, so what is asserted is the
decision the function made, not the presence of a comparison in the source.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

import pytest

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.core.config import EncryptedStr

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _FakeResponse:
    def __init__(self, chunks: list[bytes], headers: dict[str, str], status: int = 200):
        self._chunks = chunks
        self.headers = headers
        self.status_code = status

    def raise_for_status(self) -> None:
        """Real behaviour, because a no-op here hid the whole retry decision.

        With this returning None unconditionally, no test could ever reach
        `_classify_retryable_http_error` on the streaming path -- so both arms of the
        retry decision for VIDEO downloads were unexercised while the image path's
        equivalent was covered.
        """
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("GET", "https://example.test/asset"),
                response=httpx.Response(self.status_code, headers=self.headers),
            )

    async def aiter_bytes(self, chunk_size: int = 0):
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _FakeClient:
    def __init__(self, chunks: list[bytes], headers: dict[str, str], status: int = 200):
        self._chunks = chunks
        self._headers = headers
        self._status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def stream(self, _method, _url, **_kwargs):
        return _FakeResponse(self._chunks, self._headers, self._status)


@pytest.fixture
def transport(monkeypatch):
    """Replace only the network, leaving every decision in the function under test."""
    from open_webui_openrouter_pipe.storage import multimodal as mm

    def _install(chunks: list[bytes], headers: dict[str, str] | None = None, status: int = 200):
        hdrs = headers or {}
        monkeypatch.setattr(
            mm.httpx, "AsyncClient", lambda **_kw: _FakeClient(chunks, hdrs, status), raising=True
        )

    return _install


async def _download(pipe, tmp_path, **kwargs) -> Any:
    handler = pipe._multimodal_handler
    async def _pinned(url):
        return (url, {}, {})

    handler._prepare_pinned_request = _pinned  # type: ignore[method-assign]
    return await handler._download_remote_url_streaming(
        "https://example.test/asset", tmp_path / "out.bin", **kwargs
    )


@pytest.mark.asyncio
async def test_a_stream_over_the_size_cap_is_aborted(pipe_instance_async, tmp_path, transport):
    """Enforced per chunk, because a server need not send Content-Length.

    Without this the pipe writes the whole body to disk regardless of the operator's
    limit -- the valve becomes decorative.
    """
    transport([b"x" * 600, b"x" * 600], {"content-type": "video/mp4"})
    dest = tmp_path / "out.bin"

    result = await _download(pipe_instance_async, tmp_path, max_size_bytes=1000)

    assert result is None, (
        "a 1200-byte body was accepted under a 1000-byte cap; REMOTE_VIDEO_MAX_SIZE_MB "
        "is not being enforced"
    )
    written = dest.stat().st_size if dest.exists() else 0
    assert written <= 1000, (
        f"{written} bytes reached disk under a 1000-byte cap. The download was refused "
        "but not bounded, so the limit does not protect the filesystem."
    )


@pytest.mark.asyncio
async def test_a_declared_content_length_over_the_cap_is_refused_before_the_body(
    pipe_instance_async, tmp_path, transport
):
    """The cheap check: refuse before downloading anything at all."""
    transport([b"x" * 10], {"content-type": "video/mp4", "content-length": "999999"})

    result = await _download(pipe_instance_async, tmp_path, max_size_bytes=1000)

    assert result is None, "a Content-Length far over the cap was downloaded anyway"


@pytest.mark.asyncio
async def test_a_declared_content_length_under_the_cap_is_not_refused(
    pipe_instance_async, tmp_path, transport
):
    """The other direction, and the one that was missing.

    The over-limit test above is satisfied by `if True:` -- refuse everything that
    declares a length at all. Essentially every real server sends Content-Length, so
    that mutation means no image URL is ever fetched and no generated video is ever
    downloaded, each returning nothing with a warning naming a size UNDER the limit.
    The existing control sends no Content-Length header, so it never enters this branch.
    """
    transport([b"x" * 100], {"content-type": "video/mp4", "content-length": "100"})

    result = await _download(pipe_instance_async, tmp_path, max_size_bytes=1000)

    assert result is not None, (
        "a Content-Length of 100 under a 1000-byte cap was refused. The pre-check is "
        "rejecting on the presence of the header rather than on its value."
    )


@pytest.mark.asyncio
async def test_a_body_within_the_cap_is_kept(pipe_instance_async, tmp_path, transport):
    """The control. Without it the two tests above pass on a function that always fails."""
    transport([b"x" * 100], {"content-type": "video/mp4"})

    result = await _download(pipe_instance_async, tmp_path, max_size_bytes=1000)

    assert result is not None, "a body inside the cap was rejected"
    assert result["path"].read_bytes() == b"x" * 100


@pytest.mark.asyncio
async def test_a_mime_outside_the_allowlist_is_refused(
    pipe_instance_async, tmp_path, transport
):
    """VIDEO_OUTPUT_MIME_ALLOWLIST decides what may be persisted and served back."""
    transport([b"x" * 10], {"content-type": "application/x-msdownload"})

    result = await _download(
        pipe_instance_async, tmp_path, mime_allowlist={"video/mp4", "video/webm"}
    )

    assert result is None, (
        "a MIME outside the allowlist was accepted; VIDEO_OUTPUT_MIME_ALLOWLIST is not "
        "being enforced"
    )


@pytest.mark.asyncio
async def test_a_mime_inside_the_allowlist_is_kept(
    pipe_instance_async, tmp_path, transport
):
    transport([b"x" * 10], {"content-type": "video/mp4"})

    result = await _download(
        pipe_instance_async, tmp_path, mime_allowlist={"video/mp4", "video/webm"}
    )

    assert result is not None, "an allowed MIME was rejected"
    assert result["mime_type"] == "video/mp4"


@pytest.mark.asyncio
async def test_an_unhelpful_content_type_is_sniffed_before_the_allowlist_decides(
    pipe_instance_async, tmp_path, transport
):
    """``application/octet-stream`` is what a server sends when it will not commit.

    Taking it at face value would reject every such response even when the bytes are
    an allowed type, so the allowlist is applied to the sniffed type instead.
    """
    transport([PNG_BYTES], {"content-type": "application/octet-stream"})

    result = await _download(pipe_instance_async, tmp_path, mime_allowlist={"image/png"})

    assert result is not None, (
        "a PNG served as application/octet-stream was refused; the sniffed type is not "
        "reaching the allowlist"
    )
    assert result["mime_type"] == "image/png"


@pytest.mark.asyncio
async def test_no_allowlist_means_no_mime_restriction(
    pipe_instance_async, tmp_path, transport
):
    """None and the empty set are different: one disables the check, one forbids all."""
    transport([b"x" * 10], {"content-type": "application/x-msdownload"})

    result = await _download(pipe_instance_async, tmp_path, mime_allowlist=None)

    assert result is not None, "passing no allowlist unexpectedly filtered by MIME"


@pytest.mark.asyncio
async def test_the_video_path_passes_both_valves_to_the_downloader(monkeypatch):
    """Reads the values the downloader received, driven through the real lifecycle.

    The previous version scanned video.py's AST and asserted the valve NAME appeared in
    the expression bound to each kwarg. Two mutations kept every character it looked
    for and destroyed both limits: `* 1024 * 1024 * 1000` multiplies the operator's cap
    by a thousand, and `_csv_set(valves.VIDEO_OUTPUT_MIME_ALLOWLIST) and None` still
    mentions the valve while evaluating to None -- which the downloader documents as
    "no MIME restriction", so an executable would download, persist into Open WebUI
    Files and serve back. Both passed. It was also skipped under a bundle, so the
    shipped artifacts had no guard here at all.

    The valve values below are ones no default could produce, so a wired-up-but-ignored
    path cannot coincide with them.
    """
    import time

    from open_webui_openrouter_pipe.integrations.video_types import VideoLifecycleResult

    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test")
    pipe.valves.VIDEO_INITIAL_POLL_DELAY_SECONDS = 0
    pipe.valves.VIDEO_POLL_INTERVAL_SECONDS = 0
    pipe.valves.VIDEO_POLL_INTERVAL_MAX_SECONDS = 0
    pipe.valves.REMOTE_VIDEO_MAX_SIZE_MB = 7
    pipe.valves.VIDEO_OUTPUT_MIME_ALLOWLIST = "video/x-probe"
    adapter = pipe._ensure_video_generation_adapter()

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def status(self, _job_id, polling_url=None):
            return {"status": "completed", "usage": {"cost": 0.1}}

        def content_url(self, job_id, index=0):
            return f"https://example.test/videos/{job_id}/content"

        def bearer_header(self):
            return {"Authorization": "Bearer test"}

    received: list[dict[str, Any]] = []

    async def recording_download(url, dest_path, **kwargs):
        received.append(kwargs)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"\x00" * 16)
        return {"path": dest_path, "mime_type": "video/mp4", "url": url, "size_bytes": 16}

    async def fake_upload_from_path(*_args, **_kwargs):
        return "file-1"

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", FakeClient
    )
    monkeypatch.setattr(
        pipe._multimodal_handler, "_download_remote_url_streaming", recording_download
    )
    monkeypatch.setattr(
        pipe._file_gateway, "upload_to_owui_storage_from_path", fake_upload_from_path
    )

    async def emitter(_event):
        return None

    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    message_lock = asyncio.Lock()
    await message_lock.acquire()

    result = await adapter._run_lifecycle_after_submit(
        key=("chat-1", "msg-1"),
        job_id="job-1",
        api_model_id="openai/sora-2-pro",
        normalized_model_id="openai.sora-2-pro",
        valves=pipe.valves,
        event_emitter=emitter,
        user={"id": "user-1"},
        user_obj={"id": "user-1"},
        chat_id="chat-1",
        message_id="msg-1",
        request=None,
        user_id="user-1",
        global_semaphore=semaphore,
        message_lock=message_lock,
        started_at=time.monotonic(),
    )
    assert isinstance(result, VideoLifecycleResult)

    assert received, (
        "the lifecycle completed without calling the downloader, so this test observed "
        "nothing about either valve"
    )
    kwargs = received[0]
    assert kwargs.get("max_size_bytes") == 7 * 1024 * 1024, (
        f"the downloader was given max_size_bytes={kwargs.get('max_size_bytes')!r} for "
        "REMOTE_VIDEO_MAX_SIZE_MB=7. The operator's cap is not the cap being enforced."
    )
    assert kwargs.get("extra_headers") == {"Authorization": "Bearer test"}, (
        f"the download was issued with extra_headers={kwargs.get('extra_headers')!r}. "
        "The OpenRouter video content endpoint is authorised by this header alone, so "
        "every generated video would be fetched anonymously and rejected. This test "
        "already captured the value and never read it."
    )
    assert kwargs.get("mime_allowlist") == {"video/x-probe"}, (
        f"the downloader was given mime_allowlist={kwargs.get('mime_allowlist')!r} for "
        'VIDEO_OUTPUT_MIME_ALLOWLIST="video/x-probe". None means no MIME restriction at '
        "all, so any content type would be stored and served."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expect_retries"),
    [(429, True), (503, True), (404, False), (403, False)],
    ids=["rate-limited", "server-busy", "not-found", "forbidden"],
)
async def test_a_failing_video_download_retries_only_what_is_worth_retrying(
    pipe_instance_async, tmp_path, monkeypatch, status, expect_retries
):
    """The streaming path's retry decision, which nothing had ever reached.

    `_FakeResponse.raise_for_status` returned None unconditionally, so no test could
    drive an HTTP error status into `_download_remote_url_streaming` at all -- both arms
    of `if retryable:` were unexercised. That path fetches GENERATED VIDEOS: the slowest
    and most expensive thing a user waits for, and the one where OpenRouter is most
    likely to answer "busy, try again". The image path's equivalent was covered; this
    one was not.

    Asserted on the number of attempts, not on an exception: the function's contract is
    `None` on failure, and the retry classification is internal. What a user actually
    experiences is whether a 429 gets another go -- and whether a 404 wastes their time
    being retried when it never will succeed.

    Both directions over four statuses, so neither "always retry" nor "never retry"
    satisfies it.
    """
    from open_webui_openrouter_pipe.storage import multimodal as mm

    attempts = {"n": 0}
    real_client = _FakeClient

    class _CountingClient(real_client):
        def __init__(self, chunks, headers, st=status):
            super().__init__(chunks, headers, st)

        def stream(self, *a, **kw):
            attempts["n"] += 1
            return super().stream(*a, **kw)

    monkeypatch.setattr(
        mm.httpx, "AsyncClient",
        lambda **_kw: _CountingClient([b"x" * 10], {"content-type": "video/mp4"}),
        raising=True,
    )

    # Small and fast: the point is whether a retry happens at all, not how long it waits.
    pipe_instance_async.valves.REMOTE_DOWNLOAD_MAX_RETRIES = 2
    pipe_instance_async.valves.REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS = 1
    pipe_instance_async.valves.REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS = 5

    result = await _download(pipe_instance_async, tmp_path, max_size_bytes=100_000)

    assert result is None, f"HTTP {status} produced a result dict; the download failed"
    if expect_retries:
        assert attempts["n"] > 1, (
            f"HTTP {status} was attempted once and abandoned. OpenRouter says 'busy, "
            "try again' with this status -- giving up loses a video the user already "
            "paid to generate."
        )
    else:
        assert attempts["n"] == 1, (
            f"HTTP {status} was attempted {attempts['n']} times. It will never succeed, "
            "so every extra attempt is the user waiting for nothing."
        )


@pytest.mark.parametrize("via_file_host", [False, True])
@pytest.mark.parametrize("cap_mb", [7, 23])
@pytest.mark.asyncio
async def test_the_generated_video_cap_is_not_described_as_bounding_an_attachment(
    monkeypatch, via_file_host, cap_mb
):
    """The admin text for a valve names the thing it actually bounds.

    ``REMOTE_VIDEO_MAX_SIZE_MB`` was documented as also capping each attached clip or
    sound file. It does not: an attachment is measured against
    ``VIDEO_FRAME_IMAGE_MAX_BYTES`` when it goes inline, or against
    ``MEDIA_FILE_HOST_MAX_SIZE_MB`` when it goes to a public host, and a clip or sound
    file has no inline route at all. An admin lowering the wrong valve to bound what
    users upload changes nothing about what users upload.

    Measured by watching the byte cap the reader is actually handed, over two distinct
    valve values so a coincidence cannot pass, and over both routes so neither is
    assumed.
    """
    import base64
    import logging
    from types import SimpleNamespace

    from open_webui_openrouter_pipe.integrations import video as video_module
    from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter

    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test")
    try:
        adapter = VideoGenerationAdapter(pipe=pipe, logger=logging.getLogger("cap-probe"))
        pipe.valves.REMOTE_VIDEO_MAX_SIZE_MB = cap_mb
        generated_video_cap = cap_mb * 1024 * 1024

        async def stored_record(file_id, _logger):
            family = file_id.split("-", 1)[0]
            return SimpleNamespace(
                id=file_id,
                filename=f"{file_id}.bin",
                meta={"content_type": {"image": "image/png", "audio": "audio/mpeg", "video": "video/mp4"}[family]},
                user_id="user-1",
                data={},
                path=None,
            )

        handed: list[tuple[str, int]] = []

        async def reader(file_obj, _chunk, max_bytes, **_kwargs):
            handed.append((file_obj.id, max_bytes))
            return base64.b64encode(b"x" * 16).decode()

        async def relay(*_args, **_kwargs):
            return "https://example.test/relayed.bin"

        monkeypatch.setattr(video_module, "relay_to_public_url", relay)
        monkeypatch.setattr(video_module, "get_file_by_id", stored_record)
        monkeypatch.setattr(video_module, "authorize_file_publication", lambda *_a, **_k: True)
        monkeypatch.setattr(pipe._file_gateway, "read_file_record_base64", reader)
        monkeypatch.setattr(adapter, "_file_host_wanted", lambda *_a, **_k: via_file_host)

        async def emitter(_event):
            return None

        await adapter._encode_input_references(
            {
                "input_references": [{"id": "image-1"}, {"id": "audio-1"}, {"id": "video-1"}],
                "model_id": "alibaba/wan-2.7",
            },
            pipe.valves,
            withheld=[],
            user_obj={"id": "user-1"},
            video_model={
                "id": "alibaba/wan-2.7",
                "input_modalities": ["text", "image", "audio", "video"],
            },
            event_emitter=emitter,
        )
    finally:
        await pipe.close()

    assert handed, "no attachment was read at all, so the caps were never exercised"
    assert generated_video_cap not in [cap for _name, cap in handed], (
        f"REMOTE_VIDEO_MAX_SIZE_MB={cap_mb} reached an attachment as {generated_video_cap} "
        f"bytes; the caps actually handed out were {handed}"
    )


def test_the_admin_is_sent_to_the_valves_that_really_bound_an_attachment():
    """The other half of the same property, in the surface an administrator reads.

    Split from the behavioural test above so that one keeps running in the bundles
    built without plugins, where the configuration screen's metadata does not exist.
    """
    pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META

    detail = CONFIG_META["REMOTE_VIDEO_MAX_SIZE_MB"]["detail"]
    for real in ("VIDEO_FRAME_IMAGE_MAX_BYTES", "MEDIA_FILE_HOST_MAX_SIZE_MB"):
        assert CONFIG_META[real]["title"] in detail, (
            f"{real} is what really bounds an attachment and the admin is not sent to it: "
            f"{detail}"
        )
    assert "clip or sound file attached" not in detail, (
        f"this valve bounds the generated video coming back, not anything attached: {detail}"
    )
