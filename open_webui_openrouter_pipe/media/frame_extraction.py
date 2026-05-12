"""Video frame extraction at first / last / arbitrary timestamp.

PIL+imageio first; ffmpeg subprocess as fallback for codecs imageio can't handle.
Async-wrapped via run_in_threadpool to avoid event-loop stalls on blocking IO.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import imageio.v3 as iio  # type: ignore[import-untyped]
from PIL import Image

Image.MAX_IMAGE_PIXELS = 25_000_000

_MAX_FRAME_PIXELS = 25_000_000
_FFMPEG_TIMEOUT_S = 30.0


class FrameExtractionError(Exception):
    """Raised when a frame cannot be extracted from a video file."""


@dataclass
class VideoMetadata:
    duration_seconds: float
    width: int
    height: int
    fps: float
    has_audio: bool


@dataclass
class ExtractedFrame:
    image_bytes: bytes
    """PNG-encoded image bytes."""
    width: int
    height: int
    actual_timestamp_seconds: float
    requested_timestamp_seconds: Optional[float]
    downgrade_note: str = ""


# -----------------------------------------------------------------------------
# Probe
# -----------------------------------------------------------------------------

def _probe_video_sync(path: Path) -> VideoMetadata:
    """Blocking video probe. Caller wraps in to_thread."""
    try:
        meta = iio.immeta(str(path), exclude_applied=False)  # type: ignore[no-any-return]
        duration = float(meta.get("duration", 0.0) or 0.0)
        fps_raw = meta.get("fps") or meta.get("fps_in_av") or 0.0
        fps = float(fps_raw) if fps_raw else 24.0
        size = meta.get("size") or (0, 0)
        width = int(size[0]) if isinstance(size, (list, tuple)) and len(size) >= 1 else 0
        height = int(size[1]) if isinstance(size, (list, tuple)) and len(size) >= 2 else 0
        has_audio = bool(meta.get("audio_codec"))
        return VideoMetadata(
            duration_seconds=duration,
            width=width,
            height=height,
            fps=fps if fps > 0 else 24.0,
            has_audio=has_audio,
        )
    except Exception as exc:
        raise FrameExtractionError(f"probe_video failed: {exc}") from exc


async def probe_video(path: Path) -> VideoMetadata:
    """Async wrapper around imageio video probe.

    Raises FrameExtractionError on any failure (corrupt file, unsupported codec).
    """
    return await asyncio.to_thread(_probe_video_sync, path)


# -----------------------------------------------------------------------------
# Frame extraction
# -----------------------------------------------------------------------------

def _extract_frame_imageio_sync(
    path: Path, *, frame_index: int
) -> tuple[bytes, int, int]:
    """Extract a single frame at the given index via imageio. Returns
    (png_bytes, width, height). Raises FrameExtractionError on failure or
    on decompression-bomb-sized output."""
    try:
        arr = iio.imread(str(path), index=frame_index)
        if arr is None or len(arr.shape) < 2:
            raise FrameExtractionError("imageio returned empty frame")
        h = int(arr.shape[0])
        w = int(arr.shape[1])
        if w * h > _MAX_FRAME_PIXELS:
            raise FrameExtractionError(
                f"frame too large: {w}x{h} exceeds {_MAX_FRAME_PIXELS} pixel cap",
            )
        img = Image.fromarray(arr)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), img.width, img.height
    except FrameExtractionError:
        raise
    except Exception as exc:
        raise FrameExtractionError(f"imageio extract failed: {exc}") from exc


async def _extract_frame_ffmpeg(
    path: Path, *, timestamp_seconds: float, logger: logging.Logger
) -> tuple[bytes, int, int]:
    """Fallback frame extraction via ffmpeg subprocess.

    Pipes a single PNG-encoded frame to stdout. Returns (png_bytes, w, h).
    Hardened: rejects path starting with `-` (argv injection), restricts
    ffmpeg to local file protocol, scales output to bound memory, applies
    a 30s timeout, and kills the subprocess on cancellation.
    """
    del logger
    path_str = str(path)
    if path_str.startswith("-"):
        raise FrameExtractionError("refusing path starting with '-' (argv injection guard)")

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        try:
            import imageio_ffmpeg  # type: ignore[import-untyped]
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise FrameExtractionError(f"ffmpeg unavailable: {exc}") from exc

    cmd = [
        ffmpeg_bin,
        "-protocol_whitelist", "file",
        "-ss", str(max(0.0, timestamp_seconds)),
        "-i", path_str,
        "-frames:v", "1",
        "-vf", "scale='min(1920,iw)':-2",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-loglevel", "error",
        "-",
    ]
    proc: Optional[asyncio.subprocess.Process] = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_FFMPEG_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                proc.kill()
                await proc.wait()
            raise FrameExtractionError(
                f"ffmpeg timed out after {_FFMPEG_TIMEOUT_S}s",
            ) from None
        if proc.returncode != 0:
            raise FrameExtractionError(
                f"ffmpeg returned {proc.returncode}: {stderr.decode('utf-8', errors='replace')[:200]}"
            )
        if not stdout:
            raise FrameExtractionError("ffmpeg produced empty output")
        img = Image.open(io.BytesIO(stdout))
        img.load()
        if img.width * img.height > _MAX_FRAME_PIXELS:
            raise FrameExtractionError(
                f"ffmpeg output {img.width}x{img.height} exceeds pixel cap",
            )
        return stdout, img.width, img.height
    except asyncio.CancelledError:
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.kill()
                await proc.wait()
        raise
    except FrameExtractionError:
        raise
    except Exception as exc:
        raise FrameExtractionError(f"ffmpeg extract failed: {exc}") from exc


async def extract_frame(
    path: Path,
    *,
    target: Literal["first_frame", "last_frame", "at_timestamp"],
    timestamp_seconds: Optional[float] = None,
    fallback_to_last_on_overshoot: bool = True,
    overshoot_fallback_index: Literal["first", "last"] = "last",
    logger: Optional[logging.Logger] = None,
) -> ExtractedFrame:
    """Extract a frame from a video file.

    - first_frame: frame at index 0
    - last_frame: last frame (probes duration to compute timestamp)
    - at_timestamp: frame at requested seconds; if > duration AND
      fallback_to_last_on_overshoot, downgrades to either the first frame
      (overshoot_fallback_index="first") or the last frame ("last", default)
      and sets `downgrade_note`.

    PIL+imageio first; ffmpeg subprocess fallback if imageio fails.

    Raises FrameExtractionError on any failure that can't be downgraded.
    """
    logger = logger or logging.getLogger(__name__)
    if not path.exists():
        raise FrameExtractionError(f"video file not found: {path}")
    if target == "at_timestamp" and (timestamp_seconds is None or timestamp_seconds < 0):
        raise FrameExtractionError("at_timestamp requires non-negative timestamp_seconds")

    downgrade_note = ""
    requested_ts = timestamp_seconds if target == "at_timestamp" else None

    if target == "first_frame":
        actual_ts = 0.0
    elif target in ("last_frame", "at_timestamp"):
        try:
            meta = await probe_video(path)
        except FrameExtractionError:
            meta = None
        if target == "last_frame":
            if meta and meta.duration_seconds > 0:
                actual_ts = max(0.0, meta.duration_seconds - max(1.0 / meta.fps, 0.04))
            else:
                actual_ts = 9_999_999.0  # sentinel — ffmpeg clamps to last frame
        else:
            assert timestamp_seconds is not None
            if meta and timestamp_seconds > meta.duration_seconds:
                if not fallback_to_last_on_overshoot:
                    raise FrameExtractionError(
                        f"timestamp {timestamp_seconds}s exceeds video duration {meta.duration_seconds}s"
                    )
                if overshoot_fallback_index == "first":
                    actual_ts = 0.0
                    fallback_word = "first"
                else:
                    actual_ts = max(
                        0.0, meta.duration_seconds - max(1.0 / meta.fps, 0.04)
                    )
                    fallback_word = "last"
                downgrade_note = (
                    f"Requested frame at {timestamp_seconds:.1f}s but the previous video "
                    f"is only {meta.duration_seconds:.1f}s. Using {fallback_word} frame instead."
                )
            else:
                actual_ts = float(timestamp_seconds)
    else:
        raise FrameExtractionError(f"unknown target: {target}")

    if target == "first_frame":
        try:
            png_bytes, w, h = await asyncio.to_thread(
                _extract_frame_imageio_sync, path, frame_index=0,
            )
            return ExtractedFrame(
                image_bytes=png_bytes, width=w, height=h,
                actual_timestamp_seconds=0.0,
                requested_timestamp_seconds=requested_ts,
                downgrade_note=downgrade_note,
            )
        except FrameExtractionError as exc:
            logger.debug("imageio first_frame failed; falling through to ffmpeg: %s", exc)

    png_bytes, w, h = await _extract_frame_ffmpeg(
        path, timestamp_seconds=actual_ts, logger=logger,
    )
    return ExtractedFrame(
        image_bytes=png_bytes, width=w, height=h,
        actual_timestamp_seconds=actual_ts,
        requested_timestamp_seconds=requested_ts,
        downgrade_note=downgrade_note,
    )
