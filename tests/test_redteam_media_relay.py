"""Regression tests for the media relay findings. These FAIL on today's code."""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from open_webui_openrouter_pipe.core.config import Valves
from open_webui_openrouter_pipe.integrations import video as video_module
from open_webui_openrouter_pipe.integrations.media_relay import _ENDPOINTS
from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter
from open_webui_openrouter_pipe.integrations.video_types import VideoGenerationError
from open_webui_openrouter_pipe.storage.owui_files import infer_file_mime_type

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


async def _a_listening_chat(_event):
    """A chat whose socket is attached, which is the precondition for any upload."""
    return None

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32


class _Ctx:
    def __init__(self, session): self._session = session
    async def __aenter__(self): return self._session
    async def __aexit__(self, *_exc): return False


class _Harness:
    """Real relay, real MIME inference; only the storage read and the socket are stubbed."""

    def __init__(self, blobs, records, host="litterbox"):
        self.blobs, self.records, self.host = blobs, records, host
        self.posts: list[dict] = []
        self.notices: list[str] = []
        self.timeline: list[str] = []
        self.the_chat_hears_it = True
        self.second_post_raises: BaseException | None = None
        self.second_post_status = 200
        self.pipe = MagicMock()
        self.pipe.logger = logging.getLogger("redteam")

        async def _read(file_obj, _chunk, _cap, user=None):
            return base64.b64encode(self.blobs[file_obj.id]).decode()

        async def _notify(_emitter, content, *, level="info"):
            self.notices.append(content)
            self.timeline.append("told")
            return self.the_chat_hears_it

        self.pipe._file_gateway.read_file_record_base64 = _read
        self.pipe._event_emitter_handler._emit_notification = _notify
        self.adapter = VideoGenerationAdapter(pipe=self.pipe, logger=self.pipe.logger)

    def _wire(self, url, **kwargs):
        form = kwargs["data"]()
        body = bytearray()

        class _Sink:
            async def write(self, chunk): body.extend(chunk)

        asyncio.run(form.write(_Sink())) if False else None
        self.posts.append({"url": str(url), "form": form})
        self.timeline.append("published")
        if len(self.posts) > 1:
            if self.second_post_raises is not None:
                raise self.second_post_raises
            if self.second_post_status != 200:
                return CallbackResult(status=self.second_post_status, body="no")
        return CallbackResult(status=200, body="https://files.catbox.moe/PUBLISHED.mp4")

    async def encode(
        self, refs, valves, *, relayed=None, withheld=None, model=None, event_emitter=None
    ):
        async def _get_file(file_id, _logger):
            return self.records[file_id]

        saved_get_file = video_module.get_file_by_id
        saved_infer = video_module.infer_file_mime_type
        video_module.get_file_by_id = _get_file
        video_module.infer_file_mime_type = infer_file_mime_type
        try:
            async with aiohttp.ClientSession() as session:
                self.pipe._create_http_session = lambda *_a, **_k: _Ctx(session)
                with aioresponses() as mocked:
                    mocked.post(_ENDPOINTS[self.host][0], callback=self._wire, repeat=True)
                    return await self.adapter._encode_input_references(
                        {"input_references": refs, "model_id": "runway/aleph-2"},
                        valves,
                        withheld=withheld if withheld is not None else [],
                        user_obj=SimpleNamespace(id="bob", role="user"),
                        video_model=model or {"id": "runway/aleph-2",
                                              "input_modalities": ["video", "image", "audio"]},
                        relayed=relayed if relayed is not None else set(),
                        companions=True,
                        event_emitter=event_emitter,
                    )
        finally:
            video_module.get_file_by_id = saved_get_file
            video_module.infer_file_mime_type = saved_infer


def _record(file_id, stored_mime, filename="clip.mp4"):
    return SimpleNamespace(id=file_id, filename=filename, user_id="bob",
                           mime_type=None, content_type=None,
                           meta={"content_type": stored_mime})


# ---------------------------------------------------------------- FINDING 3 --
@pytest.mark.parametrize(
    ("declared", "stored_mime"),
    [("video/mp4", ";"), ("audio/mpeg", "; charset=utf-8")],
)
@pytest.mark.asyncio
async def test_the_kind_a_file_is_relayed_as_comes_from_the_record_not_the_request(
    declared, stored_mime
):
    """`body["files"][i]["content_type"]` is browser input. It must never decide
    which family a stored file belongs to, nor reach the upload's Content-Type.

    Parametrised over two declared types so a production `return "video/mp4"`
    cannot satisfy both.
    """
    harness = _Harness({"f1": PNG}, {"f1": _record("f1", stored_mime)})
    valves = Valves()
    valves.SEND_MEDIA_VIA_FILE_HOST = True
    valves.SEND_IMAGES_VIA_FILE_HOST = False

    await harness.encode([{"id": "f1", "content_type": declared}], valves)

    assert harness.posts == [], (
        f"a file whose stored type is {stored_mime!r} was uploaded as {declared!r} "
        f"because the request said so"
    )


@pytest.mark.parametrize("hostile", ["video/mp4\r\nX-Injected: yes", "video/mp4\nX: y"])
@pytest.mark.asyncio
async def test_a_request_declared_type_cannot_reach_the_upload_headers(hostile):
    """aiohttp refuses CRLF in a header value by raising ValueError, which nothing
    in the relay catches. The pipe must reject the type itself, not crash on it."""
    harness = _Harness({"f1": MP4}, {"f1": _record("f1", ";")})
    valves = Valves()
    valves.SEND_MEDIA_VIA_FILE_HOST = True
    withheld: list[tuple[str, str]] = []

    encoded = await harness.encode(
        [{"id": "f1", "content_type": hostile}], valves, withheld=withheld
    )
    assert encoded == []
    assert withheld, "the file was neither sent nor reported"


# ---------------------------------------------------------------- FINDING 1 --
@pytest.mark.parametrize("interrupted_by", ["stop-pressed", "the-next-upload-refused"])
@pytest.mark.asyncio
async def test_a_file_already_published_is_disclosed_even_when_the_request_fails(
    interrupted_by,
):
    """The notice is the only thing that tells a user their media left the server, so it
    has to be out before the bytes are -- not on a later line that a failure can skip.

    Two ways the request dies after the first file is already public: Stop, which raises
    CancelledError, and the next upload being refused, which raises VideoGenerationError.
    A disclosure written into an `except Exception` handler covers the second and misses
    the first; one written into `finally` misses a hard cancel. Only disclosing before
    the upload survives both, which is why both are exercised here.
    """
    harness = _Harness({"f1": MP4, "f2": MP4},
                       {"f1": _record("f1", "video/mp4"), "f2": _record("f2", "video/mp4")})
    if interrupted_by == "stop-pressed":
        harness.second_post_raises = asyncio.CancelledError()
    else:
        harness.second_post_status = 400
    valves = Valves()
    valves.SEND_MEDIA_VIA_FILE_HOST = True
    relayed: set[tuple[str, str]] = set()
    seen: list[dict] = []

    async def emitter(event):
        seen.append(event)

    with pytest.raises(BaseException):
        await harness.encode(
            [{"id": "f1"}, {"id": "f2"}], valves, relayed=relayed, event_emitter=emitter
        )

    assert harness.posts, "precondition: the first file must have been uploaded"
    assert harness.timeline and harness.timeline[0] == "told", (
        f"bytes reached the host before anything was said: {harness.timeline}"
    )
    said = harness.notices[0]
    assert valves.MEDIA_FILE_HOST in said, f"the notice does not name the host: {said}"
    assert "hour" in said, f"the notice does not say how long it stays: {said}"


# ---------------------------------------------------------------- FINDING 5 --
@pytest.mark.parametrize("count", [40, 400])
@pytest.mark.asyncio
async def test_a_single_request_cannot_publish_an_unbounded_number_of_files(count):
    """Every accepted reference is held in memory as base64 before the first upload
    starts, and the deployment-wide semaphore is taken only afterwards."""
    blobs = {f"f{i}": MP4 for i in range(count)}
    records = {f"f{i}": _record(f"f{i}", "video/mp4") for i in range(count)}
    harness = _Harness(blobs, records)
    valves = Valves()
    valves.SEND_MEDIA_VIA_FILE_HOST = True

    encoded = await harness.encode(
        [{"id": f"f{i}"} for i in range(count)], valves, event_emitter=_a_listening_chat
    )

    assert len(harness.posts) <= 16, (
        f"{len(harness.posts)} files were published from one request "
        f"({len(encoded)} references accepted; the image adapter caps at 16)"
    )


# ---------------------------------------------------------------- FINDING 4 --
@pytest.mark.parametrize(
    "answer",
    ["https://attacker.example/beacon.mp4", "http://169.254.169.254/latest/meta-data/"],
)
@pytest.mark.asyncio
async def test_the_link_forwarded_is_the_one_the_chosen_host_serves(answer):
    """Whatever the anonymous host replies becomes a URL this pipe hands to
    OpenRouter with no origin or scheme check."""
    pipe = MagicMock()
    pipe.logger = logging.getLogger("redteam")
    adapter = VideoGenerationAdapter(pipe=pipe, logger=pipe.logger)
    valves = Valves()
    valves.SEND_MEDIA_VIA_FILE_HOST = True

    async with aiohttp.ClientSession() as session:
        pipe._create_http_session = lambda *_a, **_k: _Ctx(session)
        with aioresponses() as mocked:
            mocked.post(
                _ENDPOINTS["litterbox"][0],
                callback=lambda url, **kw: CallbackResult(status=200, body=answer),
                repeat=True,
            )
            with pytest.raises(VideoGenerationError) as refused:
                await adapter._relay_reference(
                    valves, base64.b64encode(MP4).decode(),
                    filename="a.mp4", mime="video/mp4", family="video",
                    deadline=time.monotonic() + 30.0,
                )

    said = str(refused.value)
    assert "litterbox" in said, f"the refusal does not name the host: {said}"
    assert answer not in said and "://" not in said, (
        f"the address the host answered with came back as an address: {said}"
    )
