"""Unit tests for the media package (frame extraction, thumbnails, image conv)."""
from __future__ import annotations

import io
import logging
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from open_webui_openrouter_pipe.media import (
    FrameExtractionError,
    Thumbnail,
    composite_on_white,
    extract_frame,
    make_thumbnail,
    normalise_mime,
    probe_video,
)


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
        img = Image.open(io.BytesIO(rgba_png_bytes))
        result = composite_on_white(img)
        assert result.mode == "RGB"

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

class TestMakeThumbnail:
    def test_returns_jpeg_at_target_size(self, red_jpeg_bytes):
        thumb = make_thumbnail(red_jpeg_bytes)
        assert isinstance(thumb, Thumbnail)
        assert thumb.mime_type == "image/jpeg"
        assert thumb.width == 256
        assert thumb.height == 256

    def test_aspect_preserved_with_letterbox_wide(self):
        img = Image.new("RGB", (400, 100), color=(0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        thumb = make_thumbnail(buf.getvalue())
        assert thumb.width == 256 and thumb.height == 256

    def test_aspect_preserved_with_letterbox_tall(self):
        img = Image.new("RGB", (100, 400), color=(0, 255, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        thumb = make_thumbnail(buf.getvalue())
        assert thumb.width == 256 and thumb.height == 256

    def test_composites_rgba_on_white(self, rgba_png_bytes):
        thumb = make_thumbnail(rgba_png_bytes)
        # Decode resulting JPEG and check it's RGB (no alpha)
        result_img = Image.open(io.BytesIO(thumb.image_bytes))
        assert result_img.mode == "RGB"

    def test_handles_grayscale(self):
        img = Image.new("L", (200, 200), 100)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        thumb = make_thumbnail(buf.getvalue())
        assert thumb.width == 256

    def test_quality_param_affects_size(self, red_jpeg_bytes):
        thumb_high = make_thumbnail(red_jpeg_bytes, quality=95)
        thumb_low = make_thumbnail(red_jpeg_bytes, quality=10)
        # Lower quality should be smaller (or equal — for tiny images both can be small)
        assert len(thumb_low.image_bytes) <= len(thumb_high.image_bytes)

    def test_rejects_zero_byte_input(self):
        with pytest.raises(Exception):
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
        assert frame.downgrade_note != ""

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
        a coded downgrade note instead of prose."""
        import open_webui_openrouter_pipe.media.frame_extraction as fe

        meta = await probe_video(synthetic_mp4)
        ts = max(0.0, meta.duration_seconds - 0.001)
        real_ffmpeg = fe._extract_frame_ffmpeg
        calls: list[bool] = []

        async def _scripted_ffmpeg(path, *, timestamp_seconds, logger, from_end=False):
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
        assert frame.downgrade_note == "frame_past_eof_used_last_frame"
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
# image_pixel_size -- the gate that keeps a reference image inside 256..5760 px
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
    """OpenRouter rejects a reference image outside 256..5760 px on either side.

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
