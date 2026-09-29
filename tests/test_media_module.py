"""Unit tests for the media package (frame extraction, thumbnails, image conv)."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from open_webui_openrouter_pipe.media import (
    FrameExtractionError,
    Thumbnail,
    VideoMetadata,
    composite_on_white,
    extract_frame,
    make_thumbnail,
    normalise_mime,
    probe_video,
)
import imageio.v3 as iio
from typing import Any, cast


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_mp4(tmp_path_factory) -> Path:
    """Generate a tiny synthetic mp4 for tests via ffmpeg.

    Skips the test module if ffmpeg is unavailable.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg  # type: ignore[import-untyped]
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pytest.skip("ffmpeg not available")
    out = tmp_path_factory.mktemp("video") / "sample.mp4"
    cmd = [
        ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=red:size=64x64:rate=24:duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        pytest.skip(f"ffmpeg fixture generation failed: {result.stderr.decode(errors='replace')[:200]}")
    return out


def _ffmpeg_binary() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pytest.skip("ffmpeg not available")


@pytest.fixture
def red_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (200, 100), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def rgba_png_bytes() -> bytes:
    img = Image.new("RGBA", (100, 100), color=(0, 255, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _dqt_sum(jpeg_bytes: bytes) -> int:
    """The sum of every quantisation table in a JPEG, read by walking the markers.

    Byte length is the readable way to see that `quality` reached the encoder, but
    it is also what a *smaller* image produces, so on its own it cannot separate
    "the knob did nothing" from "the encoder chose". The quantisation tables are
    the encoder's own record of the knob: a coarser table holds larger
    coefficients, so the sum falls as quality rises. A JPEG carries one table per
    component, so all of them are summed rather than the first, and the walk stops
    at the start-of-scan marker, past which the rest of the file is entropy-coded
    samples that happen to contain the same two bytes.
    """
    index = 2
    total = 0
    tables = 0
    while index + 4 <= len(jpeg_bytes):
        if jpeg_bytes[index] != 0xFF:
            index += 1
            continue
        marker = jpeg_bytes[index + 1]
        if marker == 0xDA:
            break
        segment_length = int.from_bytes(jpeg_bytes[index + 2:index + 4], "big")
        if marker == 0xDB:
            payload = jpeg_bytes[index + 4:index + 2 + segment_length]
            total += sum(payload[1:])
            tables += 1
        index += 2 + segment_length
    assert tables, "the encoded bytes carry no quantisation table to compare"
    return total


# -----------------------------------------------------------------------------
# image_conversion
# -----------------------------------------------------------------------------

class TestNormaliseMime:
    def test_lowercase_strips_params(self):
        assert normalise_mime("IMAGE/JPEG; charset=utf-8") == "image/jpeg"

    def test_empty_returns_empty(self):
        assert normalise_mime("") == ""
        assert normalise_mime(None) == ""

    def test_already_clean(self):
        assert normalise_mime("image/png") == "image/png"

    def test_strips_whitespace(self):
        assert normalise_mime("  image/webp  ") == "image/webp"

class TestCompositeOnWhite:
    def test_rgba_flattened(self, rgba_png_bytes):
        """A half-transparent pixel is a BLEND with the canvas, not its own colour.

        Asserting only the mode is a tautology here: a function that dropped alpha
        altogether would satisfy it, and the picture would come back as a solid
        block of the source colour, which is the defect this pins. There is no
        encoder in this path, so the blend is exact rather than approximate.
        """
        img = Image.open(io.BytesIO(rgba_png_bytes))
        result = composite_on_white(img)
        assert result.mode == "RGB"
        assert _rgb_at(result, (50, 50)) == (127, 255, 127)

    def test_rgb_passthrough(self, red_jpeg_bytes):
        img = Image.open(io.BytesIO(red_jpeg_bytes))
        result = composite_on_white(img)
        assert result.mode == "RGB"

    def test_grayscale_converted_to_rgb(self):
        img = Image.new("L", (10, 10), 128)
        result = composite_on_white(img)
        assert result.mode == "RGB"

# -----------------------------------------------------------------------------
# thumbnail
# -----------------------------------------------------------------------------

def _rgb_at(img: Image.Image, xy: tuple[int, int]) -> tuple[int, ...]:
    """The pixel at `xy` as integers, with every channel the image carries.

    Pillow types `getpixel` as a union -- a float for a greyscale image, a tuple
    for a colour one, and None for a pixel outside the picture -- so every caller
    here has to know the mode to use the result at all. This asserts that once,
    at the point of reading, so a probe that lands off the canvas is a failure
    with a message rather than a `TypeError` three lines later. Three channels
    unless the image carries a fourth, so an alpha read is legal.
    """
    pixel = img.getpixel(xy)
    assert isinstance(pixel, tuple) and len(pixel) in (3, 4), (
        f"{xy} on a {img.mode} image is {pixel!r}, not a colour; the fixture is "
        f"not the mode this test assumes"
    )
    return tuple(int(channel) for channel in pixel)


def _content_bbox(out: Image.Image) -> tuple[int, int]:
    """The size of the drawn picture, measured off the decoded canvas.

    The dataclass's `width`/`height` are the canvas, which `make_thumbnail`
    assigns from its own argument, so reading them says nothing about the picture
    and everything about the caller. A stretch, a mis-centred paste and a blank
    canvas all satisfy them. This measures the non-white region instead, and the
    `is not None` precondition is load-bearing: a canvas with no picture on it has
    no bounding box to unpack, and without the guard the mutant dies on a
    `TypeError` instead of on the property.
    """
    from PIL import ImageOps

    bbox = ImageOps.invert(out.convert("L")).getbbox()
    assert bbox is not None, (
        "the thumbnail is entirely white: a canvas with no picture on it passes a "
        "test that only looks at the bars"
    )
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


class TestMakeThumbnail:
    def test_returns_jpeg_at_target_size(self, red_jpeg_bytes):
        thumb = make_thumbnail(red_jpeg_bytes)
        assert isinstance(thumb, Thumbnail)
        assert thumb.mime_type == "image/jpeg"
        assert thumb.width == 256
        assert thumb.height == 256

    def test_aspect_preserved_with_letterbox_wide(self):
        """A 400x100 picture on a 256x256 canvas is 256x64, centred, with white bars.

        The content runs along the horizontal axis, so the bar is above and below
        and the vertical bar probe is on the left edge, where the blue channel
        carries the picture. The channel follows the fixture's colour, not the axis
        it happens to mirror: the tall fixture below is green, and the same probe
        on it reads channel 1.
        """
        img = Image.new("RGB", (400, 100), color=(0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        thumb = make_thumbnail(buf.getvalue())
        out = Image.open(io.BytesIO(thumb.image_bytes))

        assert _rgb_at(out, (128, 4)) == (255, 255, 255), (
            "there is no white bar above the picture, so the canvas is not "
            "letterboxing it"
        )
        assert _rgb_at(out, (4, 128))[2] > 200, (
            f"the left edge is {_rgb_at(out, (4, 128))}, so the picture is not "
            f"centred: it runs into the left bar"
        )
        width, height = _content_bbox(out)
        assert abs(width - 256) <= 2 and abs(height - 64) <= 2, (
            f"the content measures {width}x{height}, so the source's 4:1 shape was "
            f"not preserved (the measured picture is 66 tall: JPEG ringing on the "
            f"blue/white edge, which is why the tolerance is 2 and not 0)"
        )

    def test_aspect_preserved_with_letterbox_tall(self):
        """The mirror image, and the mirror image is where the green channel is read.

        A 100x400 picture on a 256x256 canvas is 64x256, so the bars are left and
        right and the horizontal bar probe reads the picture's own row. The fixture
        is green, so the probe at the top reads channel 1, not the blue channel the
        wide fixture's probe reads.
        """
        img = Image.new("RGB", (100, 400), color=(0, 255, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        thumb = make_thumbnail(buf.getvalue())
        out = Image.open(io.BytesIO(thumb.image_bytes))

        assert _rgb_at(out, (128, 4))[1] > 200, (
            f"the top edge is {_rgb_at(out, (128, 4))}, so the picture is not "
            f"centred: it runs into the top bar"
        )
        assert _rgb_at(out, (4, 128)) == (255, 255, 255), (
            "there is no white bar left of the picture, so the canvas is not "
            "letterboxing it"
        )
        width, height = _content_bbox(out)
        assert abs(width - 64) <= 2 and abs(height - 256) <= 2, (
            f"the content measures {width}x{height}, so the source's 1:4 shape was "
            f"not preserved"
        )

    def test_composites_rgba_on_white(self, rgba_png_bytes):
        """A 50%-alpha green thumbnail is pale green, and it is a JPEG at that.

        The mode says no alpha, which a function that dropped alpha also says, so
        the pixel is the assertion. The source is 100px and the fit scales it up to
        256 on its long side, so the picture covers the canvas: the coordinate below
        is read off the picture the function actually drew, not computed from the
        source size, so it cannot drift onto the white bar if the fixture or the fit
        is ever changed.

        Deliberate divergence from Open WebUI, which normalises a user's own upload
        with a bare `image.convert('RGB')` -- structurally the "drop the alpha"
        mutant this test exists to kill. There the drop is invisible, because the
        result goes back to the same user as an edit of their own picture; here the
        thumbnail is drawn on a white canvas inside a disclosure block other text
        sits against, so an un-matted pixel reads as a hole in the card.
        """
        source = Image.open(io.BytesIO(rgba_png_bytes))
        thumb = make_thumbnail(rgba_png_bytes)
        result_img = Image.open(io.BytesIO(thumb.image_bytes))
        from PIL import ImageOps

        drawn = ImageOps.invert(result_img.convert("L")).getbbox()
        assert drawn is not None, (
            "the thumbnail is entirely white, so there is no picture to sample and "
            "the probe below would be reading the letterbox"
        )
        centre = ((drawn[0] + drawn[2]) // 2, (drawn[1] + drawn[3]) // 2)

        thumb = make_thumbnail(rgba_png_bytes)
        # Decode resulting JPEG and check it's RGB (no alpha)
        result_img = Image.open(io.BytesIO(thumb.image_bytes))
        assert result_img.mode == "RGB"
        red, green, blue = _rgb_at(result_img, centre)
        # The nearest wrong matte (a black one) is 126 away, so a tolerance of 3
        # survives JPEG rounding and still kills it.
        assert abs(green - 255) <= 3 and abs(red - 126) <= 3 and abs(blue - 128) <= 3, (
            f"the centre pixel is {_rgb_at(result_img, centre)}: a half-transparent "
            f"green matted onto white is about (126, 255, 128), and anything far "
            f"from that is the alpha having been dropped or the matte being black"
        )

    def test_handles_grayscale(self):
        img = Image.new("L", (200, 200), 100)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        thumb = make_thumbnail(buf.getvalue())
        assert thumb.width == 256

    def test_quality_param_affects_size(self, red_jpeg_bytes):
        thumb_high = make_thumbnail(red_jpeg_bytes, quality=95)
        thumb_low = make_thumbnail(red_jpeg_bytes, quality=10)
        assert len(thumb_low.image_bytes) < len(thumb_high.image_bytes), (
            f"quality=10 produced {len(thumb_low.image_bytes)} bytes against "
            f"{len(thumb_high.image_bytes)} at quality=95, so the argument did not "
            f"reach the encoder"
        )
        assert _dqt_sum(thumb_low.image_bytes) > _dqt_sum(thumb_high.image_bytes), (
            f"quality=10 summed to {_dqt_sum(thumb_low.image_bytes)} against "
            f"{_dqt_sum(thumb_high.image_bytes)} at quality=95: a coarser table holds "
            f"larger coefficients, so this is the direction a real knob moves in"
        )

    def test_rejects_zero_byte_input(self):
        with pytest.raises(ValueError, match="empty"):
            make_thumbnail(b"")

# -----------------------------------------------------------------------------
# frame_extraction
# -----------------------------------------------------------------------------

class TestProbeVideo:
    @pytest.mark.asyncio
    async def test_returns_metadata(self, synthetic_mp4):
        meta = await probe_video(synthetic_mp4)
        assert meta.duration_seconds > 1.5  # ~2s video
        assert meta.width == 64
        assert meta.height == 64
        assert meta.fps > 0

    @pytest.mark.asyncio
    async def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FrameExtractionError):
            await probe_video(tmp_path / "nope.mp4")

class TestExtractFrame:
    @pytest.mark.asyncio
    async def test_first_frame(self, synthetic_mp4):
        frame = await extract_frame(
            synthetic_mp4, target="first_frame", logger=logging.getLogger("test"),
        )
        assert frame.actual_timestamp_seconds == 0.0
        assert len(frame.image_bytes) > 0
        assert frame.width == 64

    @pytest.mark.asyncio
    async def test_last_frame(self, synthetic_mp4):
        frame = await extract_frame(
            synthetic_mp4, target="last_frame", logger=logging.getLogger("test"),
        )
        assert frame.actual_timestamp_seconds > 0.0
        assert len(frame.image_bytes) > 0

    @pytest.mark.asyncio
    async def test_at_timestamp_within_duration(self, synthetic_mp4):
        frame = await extract_frame(
            synthetic_mp4, target="at_timestamp", timestamp_seconds=1.0,
            logger=logging.getLogger("test"),
        )
        assert frame.actual_timestamp_seconds == 1.0
        assert frame.requested_timestamp_seconds == 1.0
        assert frame.downgrade_note == ""

    @pytest.mark.asyncio
    async def test_at_timestamp_overshoot_downgrades(self, synthetic_mp4):
        frame = await extract_frame(
            synthetic_mp4, target="at_timestamp", timestamp_seconds=999.0,
            fallback_to_last_on_overshoot=True,
            logger=logging.getLogger("test"),
        )
        assert frame.requested_timestamp_seconds == 999.0
        assert frame.actual_timestamp_seconds < 999.0
        assert frame.downgrade_note == "timestamp_past_video_end_used_last_frame"

    @pytest.mark.asyncio
    async def test_at_timestamp_overshoot_raises_when_disabled(self, synthetic_mp4):
        with pytest.raises(FrameExtractionError):
            await extract_frame(
                synthetic_mp4, target="at_timestamp", timestamp_seconds=999.0,
                fallback_to_last_on_overshoot=False,
                logger=logging.getLogger("test"),
            )

    @pytest.mark.asyncio
    async def test_last_frame_when_probe_fails_uses_end_seek(self, synthetic_mp4, monkeypatch):
        """When probe_video fails (no duration), last_frame must still extract a
        real frame via end-seek — not fail with empty output from a bad -ss
        sentinel past EOF."""
        import open_webui_openrouter_pipe.media.frame_extraction as fe

        async def _failing_probe(_path):
            raise FrameExtractionError("probe failed")

        monkeypatch.setattr(fe, "probe_video", _failing_probe)
        frame = await extract_frame(
            synthetic_mp4, target="last_frame", logger=logging.getLogger("test"),
        )
        assert len(frame.image_bytes) > 0

    @pytest.mark.asyncio
    async def test_at_timestamp_past_last_frame_within_duration(self, synthetic_mp4):
        """A timestamp within the reported duration but at/past the last frame's
        PTS must still return a frame (falls back to the last frame) rather than
        raising empty-output."""
        meta = await probe_video(synthetic_mp4)
        ts = max(0.0, meta.duration_seconds - 0.001)
        frame = await extract_frame(
            synthetic_mp4, target="at_timestamp", timestamp_seconds=ts,
            logger=logging.getLogger("test"),
        )
        assert len(frame.image_bytes) > 0

    @pytest.mark.asyncio
    async def test_at_timestamp_nonzero_exit_falls_back_to_last_frame(self, synthetic_mp4, monkeypatch):
        """A past-EOF seek can also fail with a NON-ZERO ffmpeg exit code (not
        just exit-0 empty output); that failure mode must equally fall back to
        the end-seek last frame, report the true last-frame timestamp, and emit
        a coded downgrade note instead of prose.

        The scripted failure below raises the EMPTY-OUTPUT marker, not a non-zero
        exit, so what this actually pins is that the empty-output arm -- the one
        that has always walked to the end-seek ladder -- keeps walking. The
        non-zero-exit class is pinned against a real ffmpeg on a real truncated
        file further down; it is not something a hand-written double can produce
        honestly, because the exit code and its meaning come from ffmpeg itself.
        """
        import open_webui_openrouter_pipe.media.frame_extraction as fe

        meta = await probe_video(synthetic_mp4)
        ts = max(0.0, meta.duration_seconds - 0.001)
        real_ffmpeg = fe._extract_frame_ffmpeg
        calls: list[bool] = []

        async def _scripted_ffmpeg(path, *, timestamp_seconds, logger, from_end=False,
                                   saw_damage=None):
            calls.append(from_end)
            if not from_end:
                raise FrameExtractionError(
                    "ffmpeg returned 1: Output file is empty", no_frame=True,
                )
            return await real_ffmpeg(
                path, timestamp_seconds=timestamp_seconds, logger=logger, from_end=True,
            )

        monkeypatch.setattr(fe, "_extract_frame_ffmpeg", _scripted_ffmpeg)
        frame = await extract_frame(
            synthetic_mp4, target="at_timestamp", timestamp_seconds=ts,
            logger=logging.getLogger("test"),
        )
        assert calls == [False, True], "must retry exactly once with from_end=True"
        assert len(frame.image_bytes) > 0
        assert frame.downgrade_note == "frame_seek_failed_used_last_frame"
        expected_last = max(0.0, meta.duration_seconds - max(1.0 / meta.fps, 0.04))
        assert frame.actual_timestamp_seconds == pytest.approx(expected_last, abs=0.05)
        assert frame.requested_timestamp_seconds == pytest.approx(ts, abs=1e-6)

    @pytest.mark.asyncio
    async def test_negative_timestamp_raises(self, synthetic_mp4):
        with pytest.raises(FrameExtractionError):
            await extract_frame(
                synthetic_mp4, target="at_timestamp", timestamp_seconds=-1.0,
                logger=logging.getLogger("test"),
            )

    @pytest.mark.asyncio
    async def test_at_timestamp_without_value_raises(self, synthetic_mp4):
        with pytest.raises(FrameExtractionError):
            await extract_frame(
                synthetic_mp4, target="at_timestamp", timestamp_seconds=None,
                logger=logging.getLogger("test"),
            )

    @pytest.mark.asyncio
    async def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FrameExtractionError):
            await extract_frame(
                tmp_path / "nope.mp4", target="first_frame",
                logger=logging.getLogger("test"),
            )

    @pytest.mark.asyncio
    async def test_unknown_target_raises(self, synthetic_mp4):
        with pytest.raises(FrameExtractionError):
            await extract_frame(
                synthetic_mp4, target="middle",  # type: ignore[arg-type]
                logger=logging.getLogger("test"),
            )

# -----------------------------------------------------------------------------
# image_pixel_size -- the reader every pixel-based rule in the pipe asks
# -----------------------------------------------------------------------------


def _encoded(width: int, height: int, fmt: str, **options) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (12, 34, 56)).save(buffer, fmt, **options)
    return buffer.getvalue()


@pytest.mark.parametrize(("width", "height"), [(320, 140), (257, 4096)])
@pytest.mark.parametrize(
    ("fmt", "options"),
    [
        ("PNG", {}),
        ("JPEG", {}),
        ("WEBP", {"lossless": True}),
        ("WEBP", {"quality": 80}),
    ],
)
def test_the_dimensions_are_read_from_what_an_encoder_actually_writes(
    width, height, fmt, options
):
    """The size of a picture is read off the bytes the encoder actually wrote.

    Reading that from a hand-built header proves the test author can write a header.
    These bytes come from a real encoder, and the two sizes are distinct in both axes
    so a parser returning a constant, or swapping width for height, fails.
    """
    from open_webui_openrouter_pipe.storage.multimodal import image_pixel_size

    assert image_pixel_size(_encoded(width, height, fmt, **options)) == (width, height)


def test_lossy_and_lossless_webp_are_both_read_though_their_headers_differ():
    """One RIFF container, two chunk layouts -- VP8L packs the size into a bitfield."""
    from open_webui_openrouter_pipe.storage.multimodal import image_pixel_size

    lossy = _encoded(300, 200, "WEBP", quality=80)
    lossless = _encoded(300, 200, "WEBP", lossless=True)

    assert lossy[12:16] == b"VP8 "
    assert lossless[12:16] == b"VP8L"
    assert image_pixel_size(lossy) == image_pixel_size(lossless) == (300, 200)


def test_an_extended_webp_declares_its_size_one_less_than_it_is():
    """VP8X stores width-1 in three little-endian bytes; off by one is off by a pixel.

    Pillow does not emit VP8X for a plain RGB image, so the container is assembled to
    the specification here rather than encoded.
    """
    from open_webui_openrouter_pipe.storage.multimodal import image_pixel_size

    payload = b"VP8X" + (10).to_bytes(4, "little") + b"\x00" * 4
    payload += (640 - 1).to_bytes(3, "little") + (480 - 1).to_bytes(3, "little")
    raw = b"RIFF" + (len(payload) + 4).to_bytes(4, "little") + b"WEBP" + payload

    assert image_pixel_size(raw) == (640, 480)


@pytest.mark.parametrize(
    "raw",
    [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 4, b"RIFF" + b"\x00" * 20],
)
def test_bytes_that_are_not_a_picture_are_declined_rather_than_guessed(raw):
    """A guessed size would refuse a legal upload or pass an illegal one."""
    from open_webui_openrouter_pipe.storage.multimodal import image_pixel_size

    assert image_pixel_size(raw) is None


def _jpeg_frame_header(width: int, height: int) -> bytes:
    return (
        b"\xff\xc0\x00\x11\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    )


@pytest.mark.parametrize(
    ("name", "raw", "expected"),
    [
        (
            "bytes-between-two-markers",
            b"\xff\xd8" + b"\xff\xe0\x00\x04AB" + b"CD" + _jpeg_frame_header(640, 480),
            (640, 480),
        ),
        (
            "markers-that-carry-no-segment",
            b"\xff\xd8" + b"\xff\xd0" + b"\xff\xd7" + _jpeg_frame_header(321, 123),
            (321, 123),
        ),
        ("a-segment-shorter-than-its-own-length-field", b"\xff\xd8\xff\xe0\x00\x01" + b"\x00" * 64, None),
        ("a-segment-declaring-no-length", b"\xff\xd8\xff" + b"\x00" * 64, None),
    ],
)
def test_a_jpeg_the_scanner_has_to_walk_is_measured_or_declined_never_guessed(
    name, raw, expected
):
    """The reference-image gate is a size, so a wrong one refuses a picture that is fine.

    These four streams are written by hand because no encoder produces them: the
    scanner's resync arm, its standalone-marker arm and its short-segment guard are
    reached only by a stream that is damaged or padded, which is exactly the stream a
    user's re-encoded upload can be. All three arms were unreachable from the suite, so
    the scanner could walk off the end or read a length as a size and nothing said so.

    The two readable rows carry different sizes and are not square, so a parser that
    returns a constant or transposes the axes fails.
    """
    from open_webui_openrouter_pipe.storage.multimodal import image_pixel_size

    assert image_pixel_size(raw) == expected, name


# -----------------------------------------------------------------------------
# The video stream's length, measured with ffmpeg, on a host with no ffprobe
# -----------------------------------------------------------------------------

_STREAM_SHAPES: list[tuple[Path, float, float, float]] = []

@pytest.fixture(scope="module")
def shapes(tmp_path_factory):
    """Twelve sources whose PICTURE does not fill the container, each pinning one reading.

    The oracle is the `expected_end` below -- the picture's own length, except for
    `offset.mp4`, where the picture starts 2.0s into the container and the value the
    code measures is where it ENDS. Neither `probe_video` nor ffprobe is the oracle:
    `probe_video` is the code under test, and ffprobe is absent on the deployment this
    ships for, so an oracle built from either would let a deleted fix pass.

    Each row yields (path, expected_end, picture_s, fps). `expected_end` is what the
    measurement must read and `picture_s` is where the picture really stops, and the
    two are not the same number on the matroska rows: their `DURATION` tag carries a
    small container-clock offset over the nominal picture (3.0 becomes 3.003), which
    is why a row cannot state one number for both.

    Every shape is built at test time from explicit lavfi durations, so the split
    between the picture and the bed is true by construction. The nine older ones
    cover both containers, both stream orders, a file whose video is neither the
    first nor the last stream, a picture that does not start at zero, a still-image
    stream the measurement must skip, two video tracks where the longer one is the
    answer, the abbreviated time base an NTSC frame rate produces, and both
    matroska rows. The four newer ones are the dimensions a first-match reading and
    an ungated one cannot tell apart, and each is load-bearing in a direction the
    others are not:

    - `longcover.mkv` -- a still LONGER than the picture, on the matroska stage,
      which is the only arm that pins the still-image exclusion there.
    - `twovid.mp4` -- two video tracks on the mp4 path, so the second one is the
      answer; the still-image rows cannot separate that from a gated reading.
    - `stillshorter.mp4` -- a still SHORTER than the picture, at a file rate the
      still itself sets (8.0), on the mp4 path.
    - `stilllonger.mp4` -- a still longer than the picture, on the mp4 path. A short
      still cannot separate the gate on from the gate off (a `max` over
      `[0.125, 4.0]` is 4.0 either way), so this direction carries the mp4 gate on
      its own.

    `pytest.fail`, not `pytest.skip`: a fixture that gives up here ships twelve
    skipped arms and a green gate, which is the shape this class exists to prevent.
    """
    if _STREAM_SHAPES:
        return _STREAM_SHAPES
    ffmpeg = _ffmpeg_binary()
    out = tmp_path_factory.mktemp("shapes")
    testsrc = "testsrc=size=160x120:rate={rate}:duration={d}"
    sine = "sine=frequency=440:duration={d}"
    cover = out / "cover.png"
    Image.new("RGB", (64, 64), (0, 0, 255)).save(cover)
    red = out / "red.png"
    Image.new("RGB", (64, 64), (255, 0, 0)).save(red)
    rows = [
        ("v3in60.mp4", 3.0, 3.0, 24.0,
         ["-f", "lavfi", "-i", testsrc.format(rate=24, d=3),
          "-f", "lavfi", "-i", sine.format(d=60),
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-map", "0:v", "-map", "1:a"]),
        ("v5in40.mp4", 5.0, 5.0, 24.0,
         ["-f", "lavfi", "-i", testsrc.format(rate=24, d=5),
          "-f", "lavfi", "-i", sine.format(d=40),
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-map", "1:a", "-map", "0:v"]),
        ("three.mp4", 5.0, 5.0, 24.0,
         ["-f", "lavfi", "-i", sine.format(d=40),
          "-f", "lavfi", "-i", testsrc.format(rate=24, d=5),
          "-f", "lavfi", "-i", "sine=frequency=880:duration=40",
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-map", "0:a", "-map", "1:v", "-map", "2:a"]),
        ("offset.mp4", 8.0, 8.0, 24.0,
         ["-itsoffset", "2.0",
          "-f", "lavfi", "-i", testsrc.format(rate=24, d=6),
          "-f", "lavfi", "-i", sine.format(d=12),
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-map", "0:v", "-map", "1:a"]),
        ("coverart2.mkv", 4.083, 4.0, 24.0,
         ["-loop", "1", "-t", "0.125", "-i", str(cover),
          "-f", "lavfi", "-i", testsrc.format(rate=24, d=4),
          "-f", "lavfi", "-i", sine.format(d=40),
          "-map", "0:v", "-map", "1:v", "-map", "2:a",
          "-c:v:0", "png", "-disposition:v:0", "attached_pic",
          "-c:v:1", "libx264", "-pix_fmt", "yuv420p",
          "-c:a", "libvorbis"]),
        ("twovid.mkv", 9.0, 9.0, 24.0,
         ["-f", "lavfi", "-i", testsrc.format(rate=24, d=4),
          "-f", "lavfi",
          "-i", "testsrc=size=160x120:rate=24:duration=9",
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-map", "0:v", "-map", "1:v"]),
        ("ntsc4in40.mp4", 4.004, 4.0, 24000 / 1001,
         ["-f", "lavfi", "-i", testsrc.format(rate="24000/1001", d=4),
          "-f", "lavfi", "-i", sine.format(d=40),
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-map", "0:v", "-map", "1:a"]),
        ("v3in60.mkv", 3.003, 3.0, 24.0,
         ["-f", "lavfi", "-i", testsrc.format(rate=24, d=3),
          "-f", "lavfi", "-i", sine.format(d=60),
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-map", "0:v", "-map", "1:a", "-c:a", "libvorbis"]),
        ("a5in40.mkv", 5.003, 5.0, 24.0,
         ["-f", "lavfi", "-i", testsrc.format(rate=24, d=5),
          "-f", "lavfi", "-i", sine.format(d=40),
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-map", "1:a", "-map", "0:v", "-c:a", "libvorbis"]),
        ("longcover.mkv", 1.083, 1.0, 24.0,
         ["-loop", "1", "-t", "5", "-i", str(cover),
          "-f", "lavfi", "-i", testsrc.format(rate=24, d=1),
          "-map", "0:v", "-map", "1:v",
          "-c:v:0", "png", "-c:v:1", "libx264", "-pix_fmt:v:1", "yuv420p"]),
        ("twovid.mp4", 9.0, 9.0, 24.0,
         ["-loop", "1", "-t", "2", "-i", str(red),
          "-f", "lavfi", "-i", testsrc.format(rate=24, d=9),
          "-map", "0:v", "-map", "1:v", "-c:v", "libx264"]),
        ("stillshorter.mp4", 4.0, 4.0, 8.0,
         ["-loop", "1", "-framerate", "8", "-t", "0.125", "-i", str(cover),
          "-f", "lavfi", "-i", testsrc.format(rate=24, d=4),
          "-f", "lavfi", "-i", sine.format(d=40),
          "-map", "0:v", "-map", "1:v", "-map", "2:a",
          "-c:v:0", "mjpeg", "-pix_fmt:v:0", "yuvj420p", "-color_range", "pc",
          "-c:v:1", "libx264", "-pix_fmt:v:1", "yuv420p", "-c:a", "aac"]),
        ("stilllonger.mp4", 1.0, 1.0, 24.0,
         ["-loop", "1", "-t", "5", "-i", str(cover),
          "-f", "lavfi", "-i", testsrc.format(rate=24, d=1),
          "-map", "0:v", "-map", "1:v",
          "-c:v:0", "mjpeg", "-pix_fmt:v:0", "yuvj420p", "-color_range", "pc",
          "-c:v:1", "libx264", "-pix_fmt:v:1", "yuv420p"]),
    ]
    built: list[tuple[Path, float, float, float]] = []
    for name, expected_end, picture_s, fps, cmd in rows:
        path = out / name
        result = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", *cmd, str(path)],
            capture_output=True, check=False,
        )
        if result.returncode != 0:
            pytest.fail(
                f"ffmpeg could not build the {name} shape: "
                f"{result.stderr.decode(errors='replace')[:200]}. A skipped shape is a "
                "silently unpinned one, so this is a failure rather than a skip"
            )
        built.append((path, expected_end, picture_s, fps))
    _STREAM_SHAPES[:] = built
    return _STREAM_SHAPES


def _declared(path: Path) -> tuple[object, ...] | None:
    """The size the container's header claims, read straight from imageio."""
    size = iio.immeta(str(path), exclude_applied=False).get("size")
    return tuple(size) if isinstance(size, (list, tuple)) else None


_REQUEST = object()
_STORAGE_FALLBACK_USER = SimpleNamespace(id="fallback-storage-user")
