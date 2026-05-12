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
