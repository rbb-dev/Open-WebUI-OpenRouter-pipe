"""What the model-icon download is allowed to receive, and what it costs to refuse.

`_fetch_image_as_data_url` turns a catalog-supplied icon URL into a data URL during
catalog sync, once per distinct URL, ten at a time behind a semaphore. Four properties
are pinned here:

- The 2 MiB cap bounds what the process RECEIVES, not merely what it uses. A body far
  larger than the cap must never be fully buffered, and the source must stop being read
  once the cap is passed -- measured as peak traced allocation and as bytes actually
  pulled off the response, never as a log line.
- The byte cap does not bound the decoded bitmap, so a pixel budget does. A 258 KiB PNG
  declaring 9000x9000 costs 310 MB of resident memory once decoded, which is why the
  budget is checked against the header before `Image.load` is reached -- and why the
  refusal has to read the same way whether or not `PIL.Image.MAX_IMAGE_PIXELS` has
  already been lowered by another module in this process.
- Decoding does not stall the event loop. Ten budget-sized icons held it for 2.9s.
- Nothing holds the call open indefinitely. `HTTP_TOTAL_TIMEOUT_SECONDS` defaults to
  disabled and `sock_read` restarts on every byte, so a server dribbling one byte per
  half second used to be unbounded.

The double sits one seam below the subject: these tests drive the real
`_fetch_image_as_data_url` body and replace only the aiohttp response it reads from.
Nothing here patches the function under test.

`_Body.iter_chunked` honours the size it is handed. An earlier version returned its own
64 KiB block whenever the caller asked for more, which meant `served` measured the test
file's literal rather than production's read granularity: raising the production chunk
size to 16 MiB left every assertion green while the process pulled 16 MiB off a body it
had capped at 2 MiB. For the same reason the over-read bounds below are ABSOLUTE
allowances rather than `CAP + <the production constant>`, which would move with the
thing being measured.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import threading
import time
import tracemalloc
from typing import Any

import aiohttp
import pytest
from PIL import Image
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from open_webui_openrouter_pipe.core.config import (
    _MAX_MODEL_PROFILE_IMAGE_BYTES,
    _MAX_MODEL_PROFILE_IMAGE_PIXELS,
)
from open_webui_openrouter_pipe.storage import multimodal as mm
from tests.vetting_helpers import counted_dns

CAP = _MAX_MODEL_PROFILE_IMAGE_BYTES

# What a caller is allowed to over-read past the cap before stopping, and what it is
# allowed to hold while doing so. Absolute, so neither tracks the production constant
# they are meant to bound. Measured on this tree: a faithful double reports 1.03xCAP
# served and 1.12xCAP peak; a full read of the smallest oversized body here is 4xCAP.
OVER_READ_ALLOWANCE = 256 * 1024
PEAK_ALLOWANCE = 1024 * 1024


class _Body:
    """A response body that is generated, never materialised.

    `served` is the honest measure of what the code pulled off the wire: the whole point
    is that a caller which stops early must be visibly cheaper than one that does not.
    `read()` is provided so an implementation that buffers the lot still runs against
    this double -- it is what makes the must-fail mutation observable rather than an
    import error.
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self.served = 0

    async def iter_chunked(self, size: int):
        while self.served < self.total:
            take = min(size, self.total - self.served)
            self.served += take
            yield b"\x00" * take

    async def read(self) -> bytes:
        self.served = self.total
        return b"\x00" * self.total


class _LiteralBody(_Body):
    """A body whose bytes are given, for the cases that must decode to a real image."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(len(payload))
        self._payload = payload

    async def iter_chunked(self, size: int):
        for start in range(0, len(self._payload), size):
            piece = self._payload[start : start + size]
            self.served += len(piece)
            yield piece

    async def read(self) -> bytes:
        self.served = self.total
        return self._payload


class _Response:
    status = 200

    def __init__(
        self,
        body: _Body,
        headers: dict[str, str],
        raise_exc: BaseException | None = None,
    ) -> None:
        self.content = body
        self.headers = headers
        self._raise_exc = raise_exc

    def raise_for_status(self) -> None:
        if self._raise_exc is not None:
            raise self._raise_exc

    async def read(self) -> bytes:
        return await self.content.read()

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *_exc: Any) -> bool:
        return False


class _Session:
    """Records every URL it is asked for, so 'no request was issued' is checkable.

    Installed as `handler._vetted_http_session`, because the transport owns the session
    rather than taking one: the connector that session carries is what does the vetting,
    so a caller cannot supply an unvetted one by accident.
    """

    closed = False

    def __init__(self, response: _Response | None) -> None:
        self._response = response
        self.requested: list[str] = []

    def get(self, url: str, **_kwargs: Any) -> _Response:
        self.requested.append(url)
        if self._response is None:
            raise AssertionError(f"an outbound request was issued for {url!r}")
        return self._response

    async def close(self) -> None:
        self.closed = True


class _Recorder(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _record_logs(handler: Any) -> _Recorder:
    recorder = _Recorder()
    handler.logger.addHandler(recorder)
    handler.logger.setLevel(logging.DEBUG)
    return recorder


@pytest.fixture()
def icon_handler(pipe_instance_async):
    """The pipe's real handler.

    `session.requested` reports the URL the caller passed, because the transport no
    longer rewrites the host to the validated IP -- it validates inside the connector's
    resolver instead, which is what keeps the connection pool keyed on the hostname.
    What the transport does with an address is pinned in `test_vetted_fetch_transport.py`
    against real sockets; these tests are about the body, the cap and the conversion.
    """
    return pipe_instance_async._multimodal_handler


def _png(side: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (side, side), (200, 30, 40)).save(buf, format="PNG")
    return buf.getvalue()


def _decode(data_url: str) -> bytes:
    assert data_url.startswith("data:image/png;base64,"), data_url
    return base64.b64decode(data_url.split(",", 1)[1])


@pytest.mark.asyncio
@pytest.mark.parametrize("side", [4, 7])
async def test_small_png_icon_still_becomes_a_data_url(icon_handler, side):
    """The ordinary path must keep working, and must carry THIS image through.

    Two sizes: a hardcoded data URL cannot answer both, so the assertion is about the
    bytes that travelled rather than about the shape of the return value.
    """
    from PIL import Image

    body = _LiteralBody(_png(side))
    session = _Session(_Response(body, {"Content-Type": "image/png"}))
    icon_handler._vetted_http_session = session

    result = await icon_handler._fetch_image_as_data_url(
        "https://cdn.example.com/icon.png"
    )

    assert result is not None
    with Image.open(io.BytesIO(_decode(result))) as image:
        assert image.size == (side, side)
    assert session.requested == ["https://cdn.example.com/icon.png"]


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [True, False])
async def test_icon_exactly_at_the_cap_is_not_rejected(icon_handler, declared):
    """The cap is a ceiling, not a fencepost: a body of exactly CAP bytes is allowed.

    Parametrised over the Content-Length header because production compares twice --
    once against the declaration and once against what arrives -- and a body with no
    declaration exercises only the second. A declared length is the common CDN case, so
    a fencepost slip in the header gate would otherwise ship unnoticed.
    """
    from PIL import Image

    payload = _png(5)
    payload = payload + b"\x00" * (CAP - len(payload))
    assert len(payload) == CAP
    headers = {"Content-Type": "image/png"}
    if declared:
        headers["Content-Length"] = str(CAP)
    session = _Session(_Response(_LiteralBody(payload), headers))
    icon_handler._vetted_http_session = session

    result = await icon_handler._fetch_image_as_data_url(
        "https://cdn.example.com/big.png"
    )

    assert result is not None
    with Image.open(io.BytesIO(_decode(result))) as image:
        assert image.size == (5, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("total", [8 * 1024 * 1024, 32 * 1024 * 1024])
async def test_oversized_icon_is_never_fully_received(icon_handler, total):
    """The cap must bound what is READ, not only what is used.

    Both measurements are of the process, not of a log line: how many bytes the response
    was asked to hand over, and the peak traced allocation across the call. Parametrised
    over two body sizes because the defect is precisely that both numbers track the body
    size; the fix makes them track the cap instead.
    """
    body = _Body(total)
    session = _Session(_Response(body, {"Content-Type": "image/png"}))
    icon_handler._vetted_http_session = session

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        result = await icon_handler._fetch_image_as_data_url(
            "https://cdn.example.com/huge.png"
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result is None
    assert body.served <= CAP + OVER_READ_ALLOWANCE, (
        f"pulled {body.served} bytes off a {total}-byte body"
    )
    assert peak <= CAP + PEAK_ALLOWANCE, f"peak {peak} bytes for a {total}-byte body"


@pytest.mark.asyncio
async def test_oversized_icon_cost_does_not_track_body_size(icon_handler):
    """A body four times larger must not cost four times as much to refuse."""
    measured: dict[int, tuple[int, int]] = {}

    for total in (8 * 1024 * 1024, 32 * 1024 * 1024):
        body = _Body(total)
        session = _Session(_Response(body, {"Content-Type": "image/png"}))
        icon_handler._vetted_http_session = session
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            await icon_handler._fetch_image_as_data_url(
                "https://cdn.example.com/huge.png"
            )
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        measured[total] = (body.served, peak)

    small_served, small_peak = measured[8 * 1024 * 1024]
    large_served, large_peak = measured[32 * 1024 * 1024]
    assert large_served - small_served <= OVER_READ_ALLOWANCE, measured
    assert large_peak - small_peak <= PEAK_ALLOWANCE, measured


@pytest.mark.asyncio
async def test_declared_content_length_over_the_cap_reads_no_body(icon_handler):
    """A truthful oversize declaration is refused before a single byte is pulled."""
    body = _Body(64 * 1024 * 1024)
    session = _Session(
        _Response(
            body,
            {"Content-Type": "image/png", "Content-Length": str(64 * 1024 * 1024)},
        )
    )
    icon_handler._vetted_http_session = session

    result = await icon_handler._fetch_image_as_data_url(
        "https://cdn.example.com/declared.png"
    )

    assert result is None
    assert body.served == 0


@pytest.mark.asyncio
async def test_lying_content_length_does_not_defeat_the_cap(icon_handler):
    """Content-Length is a cheap hint, never the enforcement.

    The header claims one kilobyte and the body is 32 MiB. If the header were load
    bearing this would buffer the lot.
    """
    body = _Body(32 * 1024 * 1024)
    session = _Session(
        _Response(body, {"Content-Type": "image/png", "Content-Length": "1024"})
    )
    icon_handler._vetted_http_session = session

    result = await icon_handler._fetch_image_as_data_url(
        "https://cdn.example.com/liar.png"
    )

    assert result is None
    assert body.served <= CAP + OVER_READ_ALLOWANCE, body.served


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/icon.png",
        "https://10.0.0.5/icon.png",
        "https://[::1]/icon.png",
    ],
)
async def test_an_address_literal_in_a_private_range_issues_no_request(
    pipe_instance_async, url
):
    """The vetting has to happen BEFORE the socket, so 'returned None' is not enough.

    The session raises if it is asked for anything, which is the only assertion that
    separates a real pre-flight check from a post-hoc discard of the response. aiohttp
    does not consult a resolver for a host that is already an address, so these three
    cannot be left to the connector and are decided before the request is built.
    """
    handler = pipe_instance_async._multimodal_handler
    session = _Session(None)
    handler._vetted_http_session = session

    result = await handler._fetch_image_as_data_url(url)

    assert result is None
    assert session.requested == []


@pytest.mark.asyncio
async def test_a_name_resolving_privately_never_reaches_a_socket(pipe_instance_async):
    """A NAME is refused inside the connector, which is below `session.get`.

    So the assertion cannot be `requested == []` any more -- the request is built, and
    the refusal happens while the connection is being established. The evidence has to
    come from the layer that would have been reached: no socket is opened at all, to
    loopback or anywhere else. That is strictly more than the old assertion proved.
    """
    import socket as _socket

    handler = pipe_instance_async._multimodal_handler
    dialled: list[Any] = []
    installed = _socket.socket.connect

    def _record(self, address):
        dialled.append(address)
        return installed(self, address)

    _socket.socket.connect = _record
    try:
        result = await handler._fetch_image_as_data_url("https://localhost/icon.png")
    finally:
        _socket.socket.connect = installed
        await handler.aclose()

    assert result is None
    assert dialled == [], dialled


@pytest.mark.asyncio
async def test_relative_icon_path_is_vetted_after_it_is_absolutised(pipe_instance_async):
    """A bare path becomes an openrouter.ai URL; the vetting must see the resolved form."""
    handler = pipe_instance_async._multimodal_handler
    seen: list[str] = []
    original = handler._hop_is_refused

    def _spy(target: str) -> bool:
        seen.append(target)
        return original(target)

    handler._hop_is_refused = _spy
    try:
        session = _Session(_Response(_LiteralBody(_png(4)), {"Content-Type": "image/png"}))
        handler._vetted_http_session = session
        await handler._fetch_image_as_data_url("images/icons/x.png")
    finally:
        handler._hop_is_refused = original

    assert seen == ["https://openrouter.ai/images/icons/x.png"]
    assert session.requested == seen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "data:image/png;base64,AAAA",
        "data:image/gif;base64,R0lGODlhAQABAAAAACw=",
    ],
)
async def test_data_url_icon_needs_no_address_check(pipe_instance_async, url):
    """An inline data URL has no address, so it must not pay for a resolution.

    Two distinct URLs, and the assertion is `result == url`: a constant return value
    satisfies one row and fails the other.

    The RESOURCE is counted -- `getaddrinfo` calls -- rather than a named method, because
    the previous version spied on `_prepare_pinned_request`, which this path stopped
    calling when the transport moved the gate into the connector's resolver. `calls` was
    empty for every input, so the assertion could not fail: inserting a blocking
    resolution ahead of the data-URL shortcut left the file green. A count of the
    syscall survives every rename of the gate. `_hop_is_refused` is spied as well, since
    an address-shaped host reaches that and never reaches a resolver at all.
    """
    handler = pipe_instance_async._multimodal_handler
    hops: list[str] = []
    original = handler._hop_is_refused

    def _spy(target: str) -> bool:
        hops.append(target)
        return original(target)

    handler._hop_is_refused = _spy
    session = _Session(None)
    handler._vetted_http_session = session

    try:
        with counted_dns() as lookups:
            result = await handler._fetch_image_as_data_url(url)
    finally:
        handler._hop_is_refused = original

    assert result == url
    assert lookups == [], lookups
    assert hops == [], hops
    assert session.requested == [], session.requested


@pytest.mark.asyncio
async def test_upstream_http_error_is_logged_without_a_traceback(icon_handler):
    """A 404 on an icon is a routine outcome; it must not print a stack."""
    recorder = _record_logs(icon_handler)
    error = aiohttp.ClientResponseError(
        request_info=aiohttp.RequestInfo(
            url=URL("https://cdn.example.com/gone.png"),
            method="GET",
            headers=CIMultiDictProxy(CIMultiDict()),
            real_url=URL("https://cdn.example.com/gone.png"),
        ),
        history=(),
        status=404,
        message="Not Found",
    )
    session = _Session(_Response(_Body(0), {"Content-Type": "image/png"}, raise_exc=error))
    icon_handler._vetted_http_session = session

    try:
        result = await icon_handler._fetch_image_as_data_url(
            "https://cdn.example.com/gone.png"
        )
    finally:
        icon_handler.logger.removeHandler(recorder)

    assert result is None
    matching = [r for r in recorder.records if "gone.png" in r.getMessage()]
    assert matching, [r.getMessage() for r in recorder.records]
    assert all(r.exc_info is None for r in matching)
    assert any("404" in r.getMessage() or "Not Found" in r.getMessage() for r in matching)


@pytest.mark.asyncio
async def test_unexpected_download_failure_keeps_its_traceback(icon_handler):
    """Dropping the stack is scoped to the expected case, not to every failure."""
    recorder = _record_logs(icon_handler)
    session = _Session(
        _Response(_Body(0), {"Content-Type": "image/png"}, raise_exc=RuntimeError("boom"))
    )
    icon_handler._vetted_http_session = session

    try:
        result = await icon_handler._fetch_image_as_data_url(
            "https://cdn.example.com/odd.png"
        )
    finally:
        icon_handler.logger.removeHandler(recorder)

    assert result is None
    matching = [r for r in recorder.records if "odd.png" in r.getMessage()]
    assert matching, [r.getMessage() for r in recorder.records]
    assert any(r.exc_info is not None for r in matching)


# ── the pixel budget ─────────────────────────────────────────────────────────

# PIL's own ceiling, `int(1024 * 1024 * 1024 // 4 // 3)`, applies before anything in
# this package runs; `media/thumbnail.py` and `media/frame_extraction.py` lower it to
# 25_000_000 at import time, and `integrations/video.py` imports `..media`. A worker
# that has built the video adapter therefore decodes icons in a different global state
# from one that has not, and `None` disables the check entirely. All three must reach
# the same verdict and say the same thing about it.
PIL_DEFAULT_MAX_PIXELS = int(1024 * 1024 * 1024 // 4 // 3)
MEDIA_MAX_PIXELS = 25_000_000
IMPORT_STATES = [PIL_DEFAULT_MAX_PIXELS, MEDIA_MAX_PIXELS, None]


def test_the_media_modules_really_do_lower_the_process_wide_ceiling():
    """Why MEDIA_MAX_PIXELS above is a real state and not a number someone invented.

    The import is the subject, and its side effect is process-wide and permanent, so the
    ceiling is put back: a later test that reads `Image.MAX_IMAGE_PIXELS` would otherwise
    see a different value depending on whether this file ran first.
    """
    from PIL import Image

    previous = Image.MAX_IMAGE_PIXELS
    try:
        from open_webui_openrouter_pipe.media import thumbnail  # noqa: F401

        assert Image.MAX_IMAGE_PIXELS == MEDIA_MAX_PIXELS
    finally:
        Image.MAX_IMAGE_PIXELS = previous


@pytest.fixture()
def pil_ceiling(request):
    from PIL import Image

    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = request.param
    try:
        yield request.param
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def _sized(side: int, fmt: str) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("P" if fmt == "GIF" else "RGB", (side, side)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
@pytest.mark.asyncio
@pytest.mark.parametrize("pil_ceiling", IMPORT_STATES, indirect=True)
@pytest.mark.parametrize(("fmt", "mime"), [("PNG", "image/png"), ("GIF", "image/gif")])
@pytest.mark.parametrize(("side", "kept"), [(4000, True), (6000, False)])
async def test_a_bitmap_over_the_pixel_budget_is_refused_and_one_under_it_is_kept(
    icon_handler, pil_ceiling, fmt, mime, side, kept
):
    """The byte cap does not bound the bitmap; a separate pixel budget has to.

    Both bodies here are a small fraction of the byte cap -- a solid-colour image
    compresses to well under 6% of it at either size -- so whatever decides the outcome,
    it is provably not `_MAX_MODEL_PROFILE_IMAGE_BYTES`. One side is under the pixel
    budget and one is over, and they must come out opposite; a constant answer, in
    either direction, fails one of the two rows. Run in all three global states,
    because the verdict must not depend on which modules a worker happens to have
    imported.

    Two FORMATS, because the budget has two enforcement points and only one was covered.
    `image_pixel_size` reads PNG, JPEG, WebP and ICO headers, and returns None for GIF,
    BMP and TIFF -- those reach `Image.open` unsized and the check AFTER it is the whole
    of their budget. Deleting that check left the file at 38 passed. A 6000x6000 P-mode
    GIF is 26,837 bytes and costs 56 -> 193 MB decoded, so it is the cheap way to spend
    the budget and there was nothing pinning the refusal.
    """
    from PIL import Image

    payload = _sized(side, fmt)
    assert len(payload) < CAP // 8, len(payload)
    assert (side * side > _MAX_MODEL_PROFILE_IMAGE_PIXELS) is not kept
    assert (mm.image_pixel_size(payload) is None) is (fmt == "GIF"), (
        f"{fmt} changed which enforcement point it exercises, so one of these rows is no "
        "longer covering the branch it was chosen for"
    )
    session = _Session(_Response(_LiteralBody(payload), {"Content-Type": mime}))
    icon_handler._vetted_http_session = session

    result = await icon_handler._fetch_image_as_data_url(
        f"https://cdn.example.com/logo.{fmt.lower()}"
    )

    if not kept:
        assert result is None
        return
    assert result is not None
    with Image.open(io.BytesIO(_decode(result))) as image:
        assert image.size == (side, side)


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_the_header_sizer_declines_these_formats_so_the_post_open_check_is_their_budget(
    fmt,
):
    """Named, so the reason the row above exists cannot quietly stop being true.

    If `image_pixel_size` grows a GIF reader, the parametrised budget test stops covering
    the post-open path and nothing else would say so. BMP and TIFF are listed because
    they are the other two the sizer declines, and the refusal they get is the same one.
    """
    assert mm.image_pixel_size(_sized(64, fmt)) is None
    assert not mm._decodes_during_open(_sized(64, fmt)), (
        f"{fmt} is opened lazily, so refusing everything the sizer declines would drop "
        "it for no reason"
    )


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
@pytest.mark.asyncio
@pytest.mark.parametrize("pil_ceiling", IMPORT_STATES, indirect=True)
async def test_an_oversized_bitmap_is_refused_the_same_way_in_every_import_state(
    icon_handler, pil_ceiling
):
    """One branch, one log shape.

    With PIL's default ceiling an 8000x8000 icon was refused by the package's own budget
    and logged without a stack; with the ceiling media/ installs, PIL raised first and
    the same icon was logged as a conversion failure WITH a traceback. Two shapes for
    one decision, chosen by an unrelated import. The refusal is the same either way, so
    the record has to be too -- compared as the format string, the level and whether a
    stack was attached, which is what a log search and an operator actually key on.
    """
    recorder = _record_logs(icon_handler)
    session = _Session(
        _Response(_LiteralBody(_png(8000)), {"Content-Type": "image/png"})
    )
    icon_handler._vetted_http_session = session
    try:
        result = await icon_handler._fetch_image_as_data_url(
            "https://cdn.example.com/bomb.png"
        )
    finally:
        icon_handler.logger.removeHandler(recorder)

    assert result is None
    shapes = [
        (r.msg, r.levelno, r.exc_info is not None)
        for r in recorder.records
        if "bomb.png" in r.getMessage()
    ]
    assert shapes == [
        (
            "Skipping model icon not provably within the %d pixel budget "
            "(%d bytes, url=%s)",
            logging.DEBUG,
            False,
        )
    ], shapes


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
@pytest.mark.asyncio
@pytest.mark.parametrize(("side", "decoded"), [(4000, True), (6000, False)])
async def test_an_oversized_bitmap_is_refused_before_it_is_decoded(
    icon_handler, side, decoded
):
    """Refusing after the decode would still hold the bitmap, which is the whole cost.

    `Image.open` reads the header only, so the header dimensions are available before a
    single pixel is allocated. This records whether `Image.load` ran at all -- the
    collaborator two seams below the subject, never the subject -- so 'refused' and
    'refused cheaply' are distinguishable. Parametrised because a spy that never fires
    would otherwise pass for the wrong reason.
    """
    from PIL import Image

    loads: list[tuple[int, int]] = []
    original = Image.Image.load

    def _spy(self, *args: Any, **kwargs: Any):
        loads.append(self.size)
        return original(self, *args, **kwargs)

    session = _Session(_Response(_LiteralBody(_png(side)), {"Content-Type": "image/png"}))
    icon_handler._vetted_http_session = session
    Image.Image.load = _spy
    try:
        await icon_handler._fetch_image_as_data_url("https://cdn.example.com/logo.png")
    finally:
        Image.Image.load = original

    assert ((side, side) in loads) is decoded, loads


# ── the loop, and the clock ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_converting_icons_does_not_stall_the_event_loop(icon_handler):
    """Ten budget-sized icons held the loop for 2.9s, so nothing else in the worker ran.

    The threshold is calibrated in-test against one decode rather than hard-coded: the
    same conversion is timed synchronously first, and the loop's worst stall across ten
    of them must come in under half of that. On this tree a single decode is ~0.26s and
    the worst stall drops from ~0.28s to ~0.014s, so the margin is twenty-fold -- but a
    machine slow enough to change the decode time changes the bound with it.
    """
    payload = _png(4900)
    started = time.perf_counter()
    mm._icon_png_bytes(payload)
    one_decode = time.perf_counter() - started

    stalls: list[float] = []
    running = True

    async def _tick() -> None:
        previous = time.perf_counter()
        while running:
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            stalls.append(now - previous)
            previous = now

    ticker = asyncio.create_task(_tick())
    await asyncio.sleep(0.05)
    stalls.clear()
    try:
        for _ in range(10):
            session = _Session(
                _Response(_LiteralBody(payload), {"Content-Type": "image/png"})
            )
            icon_handler._vetted_http_session = session
            assert await icon_handler._fetch_image_as_data_url(
                "https://cdn.example.com/i.png"
            )
    finally:
        running = False
        await asyncio.sleep(0.02)
        ticker.cancel()

    assert stalls, "the ticker never ran"
    assert max(stalls) < one_decode / 2, (
        f"the loop stalled for {max(stalls):.3f}s while a single decode costs "
        f"{one_decode:.3f}s, so the conversion is running on the loop"
    )


@pytest.mark.asyncio
async def test_a_dribbling_icon_server_cannot_hold_a_semaphore_slot(icon_handler, monkeypatch):
    """One byte per interval, for ever: the cap can never fire and the slot never frees.

    Ten such URLs fill the catalog sync's `Semaphore(10)` and the refresh never
    completes. The deadline is shortened here so the test costs a fraction of a second;
    what is being pinned is that there is one at all.
    """
    monkeypatch.setattr(mm, "_ICON_FETCH_TIMEOUT_SECONDS", 0.3)

    class _Dribble(_Body):
        """Endless, and NOT sized -- a total here is a live grenade.

        `_Body.read` materialises `b"\\x00" * self.total`, so the mutation this test
        exists to catch (going back to `await resp.read()`) would allocate whatever total
        this double declares. At `1 << 40` that is a 1 TiB synchronous C allocation: the
        outer `asyncio.wait_for` cannot interrupt it, and the run reached 38 GB RSS and
        was still climbing after two minutes. Total 0 plus a named refusal turns the same
        mutation into a one-line failure.
        """

        def __init__(self) -> None:
            super().__init__(0)

        async def iter_chunked(self, size: int):
            while True:
                await asyncio.sleep(0.05)
                self.served += 1
                yield b"\x00"[:size]

        async def read(self) -> bytes:
            raise AssertionError(
                "the icon download buffered the whole body instead of streaming it; "
                "an endless source has no length to read"
            )

    body = _Dribble()
    session = _Session(_Response(body, {"Content-Type": "image/png"}))
    icon_handler._vetted_http_session = session

    # The outer wait_for is a backstop, never the mechanism: without it a regression
    # here hangs the suite instead of failing it. `elapsed` is what tells the two apart.
    started = time.perf_counter()
    result = await asyncio.wait_for(
        icon_handler._fetch_image_as_data_url("https://cdn.example.com/slow.png"),
        timeout=5.0,
    )
    elapsed = time.perf_counter() - started

    assert result is None
    assert elapsed < 3.0, (
        f"the fetch ran for {elapsed:.1f}s, so nothing in production bounded it"
    )
    assert body.served < 60, body.served
    assert body.served > 0, (
        "nothing was streamed off the body at all, so the deadline was not what ended "
        "the exchange"
    )


# ── the one container Pillow decodes while opening it ────────────────────────


def _ico_wrapping(png: bytes, claimed: int = 1, frames: int = 1) -> bytes:
    """An ICO whose directory claims `claimed`x`claimed` over the real frame.

    A directory entry stores width and height in ONE BYTE each, so 256x256 is the largest
    size it can even express and 0 means 256. The embedded frame carries the real ones.
    An ICO that lies is therefore not exotic: it is the only way to describe a frame
    bigger than 256 on a side.
    """
    import struct

    header = struct.pack("<HHH", 0, 1, frames)
    offset = 6 + 16 * frames
    entries = b"".join(
        struct.pack(
            "<BBBBHHII", claimed & 0xFF, claimed & 0xFF, 0, 0, 1, 32, len(png), offset
        )
        for _ in range(frames)
    )
    return header + entries + png


def _ico_with_bmp_frame(
    width: int, height: int, header: int = 12, payload: int = 64
) -> bytes:
    """An ICO whose frame is a BMP, with the header fields written literally.

    The frames the other helper builds are PNGs, so the two BMP branches of the sizer --
    where the signed/unsigned defect lived -- had no coverage at all. `header` selects
    BITMAPCOREHEADER (12, 16-bit fields) or BITMAPINFOHEADER (40, 32-bit fields), which
    are the two Pillow reads at different offsets and different widths.
    """
    import struct

    if header == 12:
        block = struct.pack("<I", 12) + struct.pack("<HHHH", width, height, 1, 32)
    else:
        block = (
            struct.pack("<I", 40)
            + width.to_bytes(4, "little")
            + height.to_bytes(4, "little")
            + struct.pack("<HHIIIIII", 1, 32, 0, 0, 0, 0, 0, 0)
        )
    body = block + b"\x00" * payload
    entry = struct.pack("<BBBBHHII", 0, 0, 0, 0, 1, 32, len(body), 22)
    return struct.pack("<HHH", 0, 1, 1) + entry + body


def _pillow_ico_frame_size(raw: bytes) -> tuple[int, int]:
    """The size Pillow's own reader arrives at for that frame, as ICO uses it.

    `BmpImagePlugin._bitmap` reads the header and `IcoFile.frame` halves the height,
    because an ICO BMP frame stores the XOR and AND masks stacked. Read from Pillow
    rather than restated, so "the two readers agree" is a comparison and not two copies
    of the same belief.
    """
    from PIL import BmpImagePlugin

    with BmpImagePlugin.DibImageFile(io.BytesIO(raw[22:])) as dib:
        return (dib.size[0], int(dib.size[1] / 2))


@pytest.mark.parametrize(
    ("header", "width", "height"),
    [
        (12, 0x0010, 0x0020),
        (12, 0xFFFF, 0x0AAA),
        (12, 0x8000, 0x0004),
        (12, 0x0004, 0x8000),
        (40, 0x00000010, 0x00000020),
        (40, 0xFFFFFFF0, 0x00000010),
        (40, 0x00000010, 0xFF000010),
        (40, 0x80000000, 0x00000010),
    ],
    ids=[
        "core-small", "core-width-ff", "core-width-8000", "core-height-8000",
        "info-small", "info-width-huge", "info-height-flipped", "info-width-msb",
    ],
)
def test_the_ico_pre_check_reads_the_fields_the_way_pillow_does(header, width, height):
    """Two readers, one file. Anywhere they disagree the budget stops bounding anything.

    The pre-check read BITMAPCOREHEADER width and height as SIGNED and took `abs`;
    `BmpImagePlugin._bitmap` reads them unsigned, and takes `2**32 - value` rather than
    `abs` for a v3+ height whose top byte is 0xFF. `abs(int.from_bytes(b'\xff\xff',
    'little', signed=True))` is 1 and the unsigned read is 65535, so a 98-byte file
    measured as 1,365 pixels against a frame Pillow decodes at 89,455,275 -- and for ICO
    the pre-check is the ONLY gate, because `IcoImageFile._open` decodes.

    Every row carries a value at or above 0x8000 in one field or the other, except the
    two controls: the existing ICO tests all use dimensions that read identically signed
    or unsigned, which is why they were green on the defect.
    """
    raw = _ico_with_bmp_frame(width, height, header=header)

    assert mm.image_pixel_size(raw) == _pillow_ico_frame_size(raw)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.asyncio
@pytest.mark.parametrize("pil_ceiling", IMPORT_STATES, indirect=True)
async def test_an_ico_whose_header_only_reads_large_unsigned_is_refused(
    icon_handler, pil_ceiling
):
    """The defect end to end: measured under the budget, decoded far over it.

    Width 0xFFFF reads as 1 signed and 65535 unsigned; the height is identical either
    way. So the pre-check saw 682 pixels and let it through to `Image.open`, which is
    where an ICO decodes.

    Parametrised over the three `Image.MAX_IMAGE_PIXELS` states a worker can be in
    because Pillow's own bomb check MASKS this at the 25M ceiling `media/` installs --
    a test run only in that state passes on the defective code.
    """
    from PIL import Image

    over = _ico_with_bmp_frame(0xFFFF, 1364)
    signed_reading = abs(int.from_bytes(b"\xff\xff", "little", signed=True))
    assert signed_reading * (1364 // 2) < _MAX_MODEL_PROFILE_IMAGE_PIXELS
    assert _pillow_ico_frame_size(over)[0] * _pillow_ico_frame_size(over)[1] > (
        _MAX_MODEL_PROFILE_IMAGE_PIXELS
    )

    opened: list[int] = []
    installed = Image.open

    def _spy(fp, *args, **kwargs):
        opened.append(1)
        return installed(fp, *args, **kwargs)

    Image.open = _spy
    try:
        session = _Session(_Response(_LiteralBody(over), {"Content-Type": "image/x-icon"}))
        icon_handler._vetted_http_session = session
        refused = await icon_handler._fetch_image_as_data_url(
            "https://cdn.example.com/signed.ico"
        )
        over_opens = len(opened)

        small = _ico_with_bmp_frame(16, 32)
        session = _Session(_Response(_LiteralBody(small), {"Content-Type": "image/x-icon"}))
        icon_handler._vetted_http_session = session
        under = await icon_handler._fetch_image_as_data_url(
            "https://cdn.example.com/small.ico"
        )
        under_opens = len(opened) - over_opens
    finally:
        Image.open = installed

    assert refused is None
    assert over_opens == 0, "PIL was handed an ICO the pre-check mis-measured"
    assert under_opens == 1, (
        "a small BMP-framed ICO never reached PIL either, so the refusal above is a "
        "blanket one and says nothing about the header read"
    )
    del under


def test_an_ico_is_sized_from_its_frame_and_not_from_its_directory():
    """`Image.open` is where an ICO decodes, so the size has to come from the bytes.

    Every other Pillow plugin's `_open` reads a header and stops, which is what makes a
    check placed after `Image.open` free. `IcoImageFile._open` calls `self.load()`, and it
    is the only one that does -- so by the time a post-open check can read the corrected
    size, the frame is already resident. Measured in a clean process: a 236 KB ICO wrapping
    a 9000x9000 PNG costs 315 MiB inside `Image.open`; the same PNG on its own costs 2.
    """
    from open_webui_openrouter_pipe.storage.multimodal import image_pixel_size

    big = _ico_wrapping(_png(300))
    small = _ico_wrapping(_png(40))

    assert image_pixel_size(big) == (300, 300)
    assert image_pixel_size(small) == (40, 40)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.asyncio
async def test_an_ico_over_the_budget_is_refused_without_being_opened(icon_handler):
    """The verdict was already right; what it cost was not.

    The post-open check DOES refuse this, because `Image.open` has by then loaded the
    frame and corrected the size -- so a test asserting only the verdict passes on the
    defective version. `Image.open` is therefore spied on: the property is that Pillow
    never sees the bytes at all. Parametrised against a small ICO so a guard that refuses
    every ICO fails.
    """
    from PIL import Image

    opened: list[int] = []
    installed = Image.open

    def _spy(fp, *args, **kwargs):
        opened.append(1)
        return installed(fp, *args, **kwargs)

    over = _ico_wrapping(_png(6000))
    assert 6000 * 6000 > _MAX_MODEL_PROFILE_IMAGE_PIXELS
    assert len(over) < CAP, len(over)

    Image.open = _spy
    try:
        session = _Session(_Response(_LiteralBody(over), {"Content-Type": "image/x-icon"}))
        icon_handler._vetted_http_session = session
        refused = await icon_handler._fetch_image_as_data_url(
            "https://cdn.example.com/big.ico"
        )
        over_opens = len(opened)

        session = _Session(
            _Response(_LiteralBody(_ico_wrapping(_png(40))), {"Content-Type": "image/x-icon"})
        )
        icon_handler._vetted_http_session = session
        kept = await icon_handler._fetch_image_as_data_url(
            "https://cdn.example.com/small.ico"
        )
    finally:
        Image.open = installed

    assert refused is None
    assert over_opens == 0, (
        "PIL was handed an ICO over the budget; IcoImageFile._open decodes the frame, so "
        "the refusal is correct and the memory is already spent"
    )
    assert kept is not None, "a small ICO must still convert"


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.asyncio
async def test_an_ico_whose_size_cannot_be_read_is_refused(icon_handler):
    """The deliberate answer for a container this cannot size.

    Refusing everything `image_pixel_size` declines would break GIF, BMP and TIFF icons,
    whose `Image.open` is lazy and for whom the post-open check is sound. ICO is the one
    format where being unable to size the bytes means being unable to bound the decode,
    so it is the one where 'cannot tell' has to mean 'no'.

    The subject is an ICO Pillow opens PERFECTLY WELL -- 65 tiny frames, past the frame
    ceiling this sizer will walk. A truncated file would not do: Pillow refuses that one
    too, so the refusal would come from somewhere else and the decision recorded here
    would be untested.
    """
    from open_webui_openrouter_pipe.storage.multimodal import image_pixel_size

    unsizable = _ico_wrapping(_png(8), frames=65)
    assert image_pixel_size(unsizable) is None, "the sizer has to decline this one"
    with Image.open(io.BytesIO(unsizable)) as probe:
        assert probe.size == (8, 8), "Pillow opens it, so nothing else refuses it"

    session = _Session(
        _Response(_LiteralBody(unsizable), {"Content-Type": "image/x-icon"})
    )
    icon_handler._vetted_http_session = session

    assert await icon_handler._fetch_image_as_data_url(
        "https://cdn.example.com/many.ico"
    ) is None


def test_ico_is_the_only_pillow_plugin_that_decodes_while_opening():
    """Why the guard is one format and not a blanket refusal.

    If a second plugin ever loads inside `_open`, the post-open check stops bounding it
    and this fails rather than the budget quietly ceasing to apply.

    Iterated over `Image.OPEN`, which is the table `Image.open` actually dispatches
    through. Walking `Image.EXTENSION` reached it only via the plugins that register a
    file extension, and four -- IMT, MCIDAS, SPIDER and XVTHUMB -- register none, so the
    claim in this docstring was broader than the loop underneath it.
    """
    import inspect

    from PIL import Image

    Image.init()
    eager = []
    for plugin_id in sorted(Image.OPEN):
        factory = Image.OPEN.get(plugin_id)
        cls = getattr(factory[0], "__self__", None) or factory[0] if factory else None
        opener = getattr(cls, "_open", None)
        if opener is None:
            continue
        try:
            source = inspect.getsource(opener)
        except (OSError, TypeError):  # pragma: no cover - built-in plugin
            continue
        if "self.load()" in source:
            eager.append(plugin_id)

    assert sorted(set(eager)) == ["ICO"], sorted(set(eager))
    assert {"IMT", "MCIDAS", "SPIDER", "XVTHUMB"} <= set(Image.OPEN), (
        "these four openers register no extension, so a loop over Image.EXTENSION never "
        "reaches them; if they have gone, check what else this table no longer covers"
    )


# ── what bounds the aggregate cost of decoding ───────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_icon_decodes_are_bounded_by_the_pool_and_not_the_semaphore(
    icon_handler,
):
    """Peak is workers x budget, not callers x budget.

    Decoding used to run on the caller's thread, so the GIL serialised it and the
    aggregate was bounded by accident. Moving it to a thread removed that: ten concurrent
    budget-sized icons went from +313 MiB to +2700 MiB. `asyncio.to_thread` uses the
    default executor, whose width is `min(32, cpu_count + 4)` -- and address resolution
    shares it. A pool of its own with a fixed width is what puts a number on both.

    The width is asserted absolutely first: a check written only against the constant is
    satisfied by raising it.
    """
    from open_webui_openrouter_pipe.storage.multimodal import _ICON_DECODE_WORKERS

    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def _slow(_data: bytes) -> bytes:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        return _png(4)

    await asyncio.gather(
        *(icon_handler._run_decode(_slow, b"x") for _ in range(20))
    )

    assert peak <= 4, peak
    assert peak <= _ICON_DECODE_WORKERS, (
        f"{peak} decodes ran at once against a pool of {_ICON_DECODE_WORKERS}"
    )


@pytest.mark.asyncio
async def test_the_decode_pool_is_not_the_executor_address_resolution_uses(icon_handler):
    """A budget-sized decode must not occupy the threads a resolution needs.

    `_is_safe_url` and the pinned-download path both hand `getaddrinfo` to
    `asyncio.to_thread`, whose worker cannot be interrupted -- the budget frees the
    caller, not the thread. Sharing that executor with image decoding means a fan-out of
    slow decodes delays every address check behind them.
    """
    names: list[str] = []

    def _name(_data: bytes) -> bytes:
        names.append(threading.current_thread().name)
        return _png(4)

    await icon_handler._run_decode(_name, b"x")
    default_names: list[str] = []

    def _default() -> None:
        default_names.append(threading.current_thread().name)

    await asyncio.to_thread(_default)

    assert names and default_names
    assert not (set(names) & set(default_names)), (names, default_names)
    assert all(n.startswith("or-icon-decode") for n in names), names
