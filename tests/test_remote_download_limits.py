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


@pytest.mark.parametrize(
    "declared",
    ["application/octet-stream", "binary/octet-stream", "application/binary"],
)
@pytest.mark.asyncio
async def test_a_content_type_outside_the_allowlist_is_sniffed_before_it_decides(
    pipe_instance_async, tmp_path, transport, declared
):
    """A server that will not commit to a type must not be taken at face value.

    This enumerated the generic types it would look past, listing only
    ``application/octet-stream``. A delivery server sending ``binary/octet-stream``
    matched neither the allowlist nor the enumeration, so the bytes were never consulted
    and the clip was downloaded in full and then discarded -- after the generation had
    been paid for. Parametrising over a third value that no enumeration would have
    contained keeps the general rule in place: where the header does not clear the
    allowlist and will not name a type, the bytes decide.

    Which spelling OpenRouter actually sends is not settled and this rule does not
    depend on it. Its two published specs for the same content endpoint disagree --
    one documents `video/mp4`, the other `application/octet-stream`. A recorded live
    capture shows `video/mp4`; the corpus is not unanimous, so treat neither spelling as
    settled. Every generic spelling is handled here,
    and an honest `video/mp4` is taken at its word by the allowlist short-circuit.
    """
    transport([PNG_BYTES], {"content-type": declared})

    result = await _download(pipe_instance_async, tmp_path, mime_allowlist={"image/png"})

    assert result is not None, (
        f"a PNG served as {declared!r} was refused; the sniffed type is not reaching "
        "the allowlist"
    )
    assert result["mime_type"] == "image/png"


@pytest.mark.asyncio
async def test_a_specific_but_wrong_content_type_does_not_veto_the_bytes(
    pipe_instance_async, tmp_path, transport
):
    """The header is a claim by the far end; the magic bytes are the evidence."""
    transport([PNG_BYTES], {"content-type": "text/plain"})

    result = await _download(pipe_instance_async, tmp_path, mime_allowlist={"image/png"})

    assert result is not None, (
        "a PNG mislabelled as text/plain was refused, so a wrong header can still veto "
        "content the allowlist permits"
    )
    assert result["mime_type"] == "image/png"


@pytest.mark.asyncio
async def test_bytes_outside_the_allowlist_are_refused_whatever_the_header_says(
    pipe_instance_async, tmp_path, transport
):
    """The control. Without it the two above are satisfied by deleting the check."""
    transport([b"MZ\x90\x00" + b"\x00" * 40], {"content-type": "binary/octet-stream"})

    result = await _download(pipe_instance_async, tmp_path, mime_allowlist={"image/png"})

    assert result is None, (
        "an unrecognised payload declared as a generic type was accepted; sniffing now "
        "widens the allowlist instead of resolving against it"
    )


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


def _ftyp(major: bytes, compatible: bytes = b"") -> bytes:
    body = b"ftyp" + major + bytes([0, 0, 2, 0]) + compatible
    return bytes([0, 0, 0, len(body) + 4]) + body + bytes(32)


@pytest.mark.asyncio
async def test_a_generated_mp4_arriving_in_small_chunks_is_still_identified(
    pipe_instance_async, tmp_path, transport
):
    """The case this whole path exists for, driven end to end.

    What this pins is the buffer's ACCUMULATION across chunks, not its size: the
    payload is delivered in four-byte pieces, and a buffer that captured only the first
    chunk leaves four bytes, which cannot reach the `ftyp` marker at all. The window
    size is pinned elsewhere, by the test whose compatible brand sits past byte 32.
    The allowlist is read from the valve so this also fails if the shipped default
    stops admitting what the sniffer returns.
    """
    from open_webui_openrouter_pipe.integrations.video import _csv_set

    payload = _ftyp(b"mp42", b"isomiso2")
    transport([payload[i : i + 4] for i in range(0, len(payload), 4)],
              {"content-type": "binary/octet-stream"})

    result = await _download(
        pipe_instance_async,
        tmp_path,
        mime_allowlist=_csv_set(pipe_instance_async.valves.VIDEO_OUTPUT_MIME_ALLOWLIST),
    )

    assert result is not None, "a generated MP4 was discarded after being paid for"
    assert result["mime_type"] == "video/mp4"


@pytest.mark.parametrize(
    ("major", "compatible", "declared"),
    [
        (b"qt  ", b"", "video/quicktime"),
        (b"M4A ", b"mp42isom", "audio/mp4"),
        (b"heic", b"mif1", "image/heic"),
    ],
)
@pytest.mark.asyncio
async def test_a_container_the_operator_excluded_is_not_relabelled_as_mp4(
    pipe_instance_async, tmp_path, transport, major, compatible, declared
):
    """QuickTime, m4a and HEIC all carry an `ftyp` box exactly as an MP4 does.

    Answering `video/mp4` for the whole family lets each of them overrule an honest
    declaration and clear an allowlist naming only `video/mp4` -- a format the operator
    deliberately excluded, then stored with a .mp4 extension and served back as video.
    The m4a case is the pointed one: it lists `mp42` among its compatible brands, so a
    lookup that consulted those first would still get this wrong.
    """
    transport([_ftyp(major, compatible)], {"content-type": declared})

    result = await _download(pipe_instance_async, tmp_path, mime_allowlist={"video/mp4"})

    assert result is None, (
        f"{declared} was accepted against a video/mp4-only allowlist; the sniffer is "
        "relabelling it rather than identifying it"
    )


@pytest.mark.asyncio
async def test_an_unusual_major_brand_is_rescued_by_its_compatible_brands(
    pipe_instance_async, tmp_path, transport
):
    """Refusing every brand the table does not name would cost paid generations.

    An MP4 whose major brand is unrecognised almost always still lists `isom` or an
    `mp4x` among its compatible brands, and that is enough to identify it. Without this
    the strict reading of "refuse what you cannot name" throws away a clip that is
    plainly an MP4.
    """
    transport([_ftyp(b"zzzz", b"isomiso2")], {"content-type": "binary/octet-stream"})

    result = await _download(pipe_instance_async, tmp_path, mime_allowlist={"video/mp4"})

    assert result is not None, "an MP4 with an unusual major brand was discarded"
    assert result["mime_type"] == "video/mp4"


@pytest.mark.asyncio
async def test_an_unrecognised_brand_is_still_treated_as_a_container(
    pipe_instance_async, tmp_path, transport
):
    """The brand table upgrades a container; it does not decide membership.

    Refusing every brand nobody enumerated costs paid generations: `av01`, the
    registered AV1-in-MP4 brand, is not in the table, and an earlier version of this
    code discarded such a clip after the user had been charged for it. Where a server
    will not name a type at all the bytes are the only evidence there is -- an `ftyp`
    box we cannot name more precisely is still an ISO-BMFF container, and `video/mp4`
    is the honest generic answer.

    Refusing here would not have bought safety. A payload with no recognisable signature
    at all is accepted whenever the header names an allowlisted type, so the brand table
    can only ever turn away well-formed media containers. It is a format policy.
    """
    from open_webui_openrouter_pipe.storage.multimodal import _ISO_BMFF_BRANDS

    assert _ISO_BMFF_BRANDS.get(b"av01") is None and _ISO_BMFF_BRANDS.get(b"av1m") is None, (
        "this payload must be unknown end to end, or the compatible-brand scan answers "
        "before the fallback and this test stops covering it"
    )
    transport([_ftyp(b"av01", b"av01av1m")], {"content-type": "binary/octet-stream"})

    result = await _download(pipe_instance_async, tmp_path, mime_allowlist={"video/mp4"})

    assert result is not None, "an AV1 clip was discarded after being paid for"
    assert result["mime_type"] == "video/mp4"


@pytest.mark.parametrize("box_size", [32, 1, 0], ids=["explicit", "largesize-64bit", "to-eof"])
def test_the_brand_scan_reads_every_size_encoding_the_format_allows(box_size):
    """ISO/IEC 14496-12 gives `size` three meanings, and two of them are not a length.

    `size == 1` means a 64-bit `largesize` follows; `size == 0` means the box runs to end
    of file. Clamping the scan to `size` verbatim made the `size == 1` range empty, so a
    legal MP4 fell through to the generic family answer -- and a family answer loses to
    any committal declaration, so a clip an operator had excluded by header was discarded
    after being paid for. Three encodings of the same box, one expected answer.
    """
    from open_webui_openrouter_pipe.storage.multimodal import Confidence, _sniff_evidence

    body = b"ftyp" + b"zzzz" + bytes([0, 0, 2, 0]) + b"isom"
    payload = box_size.to_bytes(4, "big") + body + bytes(32)

    evidence = _sniff_evidence(payload)

    assert evidence is not None and evidence.confidence is Confidence.IDENTIFIED, (
        f"a box declaring size={box_size} hid its own compatible brand, so a real MP4 "
        f"resolves to the generic family answer: {evidence}"
    )


@pytest.mark.asyncio
async def test_the_compatible_brand_scan_stays_inside_the_ftyp_box(
    pipe_instance_async, tmp_path, transport
):
    """A 16-byte `ftyp` box is legal and is followed immediately by the next box.

    Scanning the whole sniff buffer for compatible brands reads that next box's type as
    though it were a brand, so a file could be identified from bytes that are not part
    of its `ftyp` box at all. Here the following box is typed `qt  `; without the box
    length clamp the download is reported as QuickTime and refused.
    """
    payload = b"\x00\x00\x00\x10ftypzzzz" + bytes([0, 0, 2, 0]) + b"\x00\x00\x00\x08qt  " + bytes(40)
    transport([payload], {"content-type": "binary/octet-stream"})

    result = await _download(pipe_instance_async, tmp_path, mime_allowlist={"video/mp4"})

    assert result is not None, "a brand was read from beyond the ftyp box"
    assert result["mime_type"] == "video/mp4"


@pytest.mark.parametrize(
    ("major", "compatible", "expected"),
    [
        (b"qt  ", b"", "video/quicktime"),
        (b"M4A ", b"mp42isom", "audio/mp4"),
        (b"heic", b"mif1", "image/heic"),
    ],
)
@pytest.mark.asyncio
async def test_a_container_the_operator_allowed_is_identified_by_name(
    pipe_instance_async, tmp_path, transport, major, compatible, expected
):
    """The mirror of the exclusion test, which on its own constrains nothing.

    Refusal is satisfied just as well by a sniffer that identifies *nothing* -- the
    declared type then fails the allowlist and the answer is the same. Only asserting
    the returned value pins the table's non-mp4 rows; without this, deleting every one
    of them leaves the suite green. The generic header is essential for the same
    reason: with an honest one the assertion passes on the fallback.
    """
    transport([_ftyp(major, compatible)], {"content-type": "binary/octet-stream"})

    result = await _download(
        pipe_instance_async, tmp_path, mime_allowlist={"video/mp4", expected}
    )

    assert result is not None, f"a {expected} container was refused by its own allowlist"
    assert result["mime_type"] == expected


@pytest.mark.parametrize(
    ("major", "declared"),
    [(b"qt  ", "video/mp4"), (b"M4V ", "video/mp4"), (b"3gp4", "video/webm")],
)
@pytest.mark.asyncio
async def test_a_declaration_the_operator_allowed_is_not_second_guessed(
    pipe_instance_async, tmp_path, transport, major, declared
):
    """OpenRouter's own spec declares a real media type for the content endpoint.

    The brand table is a guess about what a container is; the operator's allowlist is a
    statement about what may be stored. Where the header already satisfies that list the
    guess has nothing to add, and letting it veto discards a clip that has been generated
    and billed. Eighteen of the table's rows resolve to something other than `video/mp4`,
    so every one of them was a way to lose paid output.

    Two distinct declared values, so a constant cannot satisfy both rows.
    """
    transport([_ftyp(major)], {"content-type": declared})

    result = await _download(
        pipe_instance_async, tmp_path, mime_allowlist={"video/mp4", "video/webm"}
    )

    assert result is not None, (
        f"a clip declared {declared!r} -- a type the operator allows -- was discarded "
        f"after being paid for, because its brand is {major!r}"
    )
    assert result["mime_type"] == declared


@pytest.mark.asyncio
async def test_an_unnameable_container_does_not_overrule_an_honest_declaration(
    pipe_instance_async, tmp_path, transport
):
    """`Confidence.FAMILY` is the weakest evidence the sniffer produces.

    An `ftyp` box whose brand is in no table says only "some ISO-BMFF container". That
    must not promote itself over a header that names a type the operator excluded, or
    the allowlist stops being a filter. Without this row the noncommittal guard is dead
    weight: replacing it with a bare `if evidence is not None` passes the whole suite.
    """
    transport([_ftyp(b"zzzz")], {"content-type": "video/quicktime"})

    result = await _download(pipe_instance_async, tmp_path, mime_allowlist={"video/mp4"})

    assert result is None, (
        "a container nobody could name was promoted to video/mp4 over a committal "
        "declaration the operator had excluded"
    )


@pytest.mark.asyncio
async def test_a_compatible_brand_can_answer_something_other_than_mp4(
    pipe_instance_async, tmp_path, transport
):
    """Otherwise the scan is indistinguishable from the fallback.

    Every other case that exercises the compatible-brand loop expects `video/mp4`,
    which is also what the function returns when the loop finds nothing -- so deleting
    the loop entirely leaves those tests green. Here the compatible brand resolves to
    audio, which the fallback can never produce, and it sits past byte 32 so the
    64-byte sniff window is load-bearing too.
    """
    payload = _ftyp(b"zzzz", b"XXXXYYYYZZZZWWWWM4A ")

    transport([payload], {"content-type": "binary/octet-stream"})

    result = await _download(
        pipe_instance_async, tmp_path, mime_allowlist={"video/mp4", "audio/mp4"}
    )

    assert result is not None and result["mime_type"] == "audio/mp4", (
        f"the compatible-brand scan did not reach offset 32: {result}"
    )
