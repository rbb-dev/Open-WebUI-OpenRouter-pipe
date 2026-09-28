"""Video frame extraction at first / last / arbitrary timestamp.

PIL+imageio first; ffmpeg subprocess as fallback for codecs imageio can't handle.
Async-wrapped via run_in_threadpool to avoid event-loop stalls on blocking IO.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import imageio.v3 as iio  # type: ignore[import-untyped]
from PIL import Image

from .image_conversion import composite_on_white

_MAX_FRAME_PIXELS = 25_000_000
_MAX_FRAME_WIDTH = 1920
_FFMPEG_TIMEOUT_S = 30.0
_PROBE_TIMEOUT_S = 10.0
_END_SEEK_WINDOWS = ("-1", "-5", "-30")
_RETRYABLE_FFMPEG_EXITS = frozenset({69})


class FrameExtractionError(Exception):
    """Raised when a frame cannot be extracted from a video file.
    """

    def __init__(self, message: str, *, no_frame: bool = False,
                 returncode: int | None = None, pixel_cap: bool = False) -> None:
        super().__init__(message)
        self.no_frame = no_frame
        self.returncode = returncode
        self.pixel_cap = pixel_cap


def _over_pixel_cap(width: int, height: int) -> bool:
    return width * height > _MAX_FRAME_PIXELS


def _pixel_cap_refusal(width: int, height: int) -> FrameExtractionError:
    return FrameExtractionError(
        f"frame too large: {width}x{height} exceeds {_MAX_FRAME_PIXELS} pixel cap",
        pixel_cap=True,
    )


def _ffmpeg_pixel_cap_refusal(width: int, height: int) -> FrameExtractionError:
    return FrameExtractionError(
        f"ffmpeg output {width}x{height} exceeds pixel cap",
        pixel_cap=True,
    )


@dataclass
class VideoMetadata:
    duration_seconds: float
    width: int
    height: int
    fps: float
    has_audio: bool
    duration_is_stream: bool = False


@dataclass
class ExtractedFrame:
    image_bytes: bytes
    """PNG-encoded image bytes."""
    width: int
    height: int
    actual_timestamp_seconds: float
    requested_timestamp_seconds: float | None
    downgrade_note: str = ""
    resolved_target: Literal["first_frame", "last_frame", "at_timestamp"] = "at_timestamp"


def _index_end(meta: VideoMetadata, index: Literal["first", "last"]) -> float:
    if index == "first":
        return 0.0
    return max(0.0, meta.duration_seconds - (max(1.0 / meta.fps, 0.04) if meta.fps > 0 else 0.04))


# -----------------------------------------------------------------------------
# Probe
# -----------------------------------------------------------------------------

def _ffprobe_stream_duration(path: Path) -> float | None:
    binary = shutil.which("ffprobe")
    if binary is None:
        return None
    try:
        proc = subprocess.run(
            [binary, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration", "-of", "csv=p=0",
             "-protocol_whitelist", "file", str(path)],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        value = float((proc.stdout or "").strip())
    except ValueError:
        return None
    return value if value > 0 else None


def _probe_video_sync(path: Path) -> VideoMetadata:
    """Blocking video probe. Caller wraps in to_thread."""
    if str(path).startswith("-"):
        raise FrameExtractionError("refusing path starting with '-' (argv injection guard)")
    try:
        meta = iio.immeta(str(path), exclude_applied=False)  # type: ignore[no-any-return]
        duration = float(meta.get("duration", 0.0) or 0.0)
        duration_is_stream = False
        if duration > 0:
            probed = _ffprobe_stream_duration(path)
            if probed is not None:
                duration = probed
                duration_is_stream = True
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
            duration_is_stream=duration_is_stream,
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

def _normalise_png_mode(img: Image.Image) -> bytes:
    if img.mode != "RGB":
        img = composite_on_white(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _scale_to_max_width(img: Image.Image) -> Image.Image:
    if img.width <= _MAX_FRAME_WIDTH:
        return img
    height = max(
        2, ((_MAX_FRAME_WIDTH * img.height + img.width) // (2 * img.width)) * 2,
    )
    return img.resize((_MAX_FRAME_WIDTH, height), Image.Resampling.LANCZOS)


def _declared_size(path: Path) -> tuple[int, int] | None:
    try:
        meta = iio.immeta(str(path), exclude_applied=False)
        size = meta.get("size")
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return int(size[0]), int(size[1])
    except Exception:  # noqa: BLE001 - a header we cannot read is a header we lack
        return None
    return None


def _normalise_frame_sync(stdout: bytes) -> tuple[bytes, int, int]:
    img = Image.open(io.BytesIO(stdout))
    if _over_pixel_cap(img.width, img.height):
        raise _ffmpeg_pixel_cap_refusal(img.width, img.height)
    img.load()
    if img.mode != "RGB":
        stdout = _normalise_png_mode(img)
    return stdout, img.width, img.height


def _extract_frame_imageio_sync(
    path: Path, *, frame_index: int
) -> tuple[bytes, int, int]:
    """Extract a single frame at the given index via imageio. Returns
    (png_bytes, width, height). Raises FrameExtractionError on failure or
    on decompression-bomb-sized output."""
    try:
        declared = _declared_size(path)
        if declared is not None and _over_pixel_cap(*declared):
            raise _pixel_cap_refusal(*declared)
        arr = iio.imread(str(path), index=frame_index)
        if arr is None or len(arr.shape) < 2:
            raise FrameExtractionError("imageio returned empty frame")
        h = int(arr.shape[0])
        w = int(arr.shape[1])
        if _over_pixel_cap(w, h):
            raise _pixel_cap_refusal(w, h)
        img = _scale_to_max_width(Image.fromarray(arr))
        return _normalise_png_mode(img), img.width, img.height
    except FrameExtractionError:
        raise
    except Exception as exc:
        raise FrameExtractionError(f"imageio extract failed: {exc}") from exc


async def _extract_frame_ffmpeg(
    path: Path, *, timestamp_seconds: float, logger: logging.Logger,
    from_end: bool = False, saw_damage: list[bool] | None = None
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

    if from_end:
        # Input-seeking with -ss past the last frame returns 0 bytes, so to grab
        # the true last frame we seek a short window before EOF, scale, then
        # reverse it — frame 1 of the reversed tail is the last decodable frame.
        seek_arg_sets = [["-sseof", window] for window in _END_SEEK_WINDOWS]
        vf = f"scale='min({_MAX_FRAME_WIDTH},iw)':-2,reverse"
    else:
        seek_arg_sets = [["-ss", str(max(0.0, timestamp_seconds))]]
        vf = f"scale='min({_MAX_FRAME_WIDTH},iw)':-2"
    last_no_frame: FrameExtractionError | None = None
    walked_past_damage = False
    for seek_args in seek_arg_sets:
        cmd = [
            ffmpeg_bin,
            "-protocol_whitelist", "file",
            *seek_args,
            "-i", path_str,
            "-frames:v", "1",
            "-vf", vf,
            "-f", "image2pipe",
            "-vcodec", "png",
            "-loglevel", "error",
            "-",
        ]
        proc: asyncio.subprocess.Process | None = None
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
            except TimeoutError:
                with contextlib.suppress(Exception):
                    proc.kill()
                    await proc.wait()
                raise FrameExtractionError(
                    f"ffmpeg timed out after {_FFMPEG_TIMEOUT_S}s",
                ) from None
            if proc.returncode != 0:
                raise FrameExtractionError(
                    f"ffmpeg returned {proc.returncode}: {stderr.decode('utf-8', errors='replace')[:200]}",
                    returncode=proc.returncode,
                )
            if not stdout:
                raise FrameExtractionError("ffmpeg produced empty output", no_frame=True)
            stdout, width, height = await asyncio.to_thread(_normalise_frame_sync, stdout)
            if saw_damage is not None:
                saw_damage.append(walked_past_damage)
            return stdout, width, height
        except asyncio.CancelledError:
            if proc is not None:
                with contextlib.suppress(Exception):
                    proc.kill()
                    await proc.wait()
            raise
        except FrameExtractionError as exc:
            if not exc.no_frame and exc.returncode not in _RETRYABLE_FFMPEG_EXITS:
                raise
            if exc.returncode in _RETRYABLE_FFMPEG_EXITS:
                walked_past_damage = True
            last_no_frame = exc
        except Exception as exc:
            if proc is not None:
                with contextlib.suppress(Exception):
                    proc.kill()
                    await proc.wait()
            raise FrameExtractionError(f"ffmpeg extract failed: {exc}") from exc
    if saw_damage is not None:
        saw_damage.append(walked_past_damage)
    if last_no_frame is not None:
        raise last_no_frame
    raise FrameExtractionError("ffmpeg extract failed: no seek attempted")


async def _imageio_last_resort(
    path: Path, *, requested_ts: float | None,
    downgrade_note: str, logger: logging.Logger,
) -> ExtractedFrame:
    png_bytes, w, h = await asyncio.to_thread(
        _extract_frame_imageio_sync, path, frame_index=0,
    )
    if not downgrade_note:
        downgrade_note = "frame_damaged_used_first_frame"
    logger.debug("ffmpeg produced no frame; the file's first frame is the last resort")
    return ExtractedFrame(
        image_bytes=png_bytes, width=w, height=h,
        actual_timestamp_seconds=0.0,
        requested_timestamp_seconds=requested_ts,
        downgrade_note=downgrade_note,
        resolved_target="first_frame",
    )


def _is_finite_non_negative(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return math.isfinite(value) and value >= 0


async def extract_frame(
    path: Path,
    *,
    target: Literal["first_frame", "last_frame", "at_timestamp"],
    timestamp_seconds: float | None = None,
    fallback_to_last_on_overshoot: bool = True,
    reused_frame_index: Literal["first", "last"] = "last",
    logger: logging.Logger | None = None,
) -> ExtractedFrame:
    """Extract a frame from a video file.

    - first_frame: frame at index 0
    - last_frame: last frame (probes duration to compute timestamp)

    PIL+imageio first; ffmpeg subprocess fallback if imageio fails.

    Raises FrameExtractionError on any failure that can't be downgraded.
    """
    logger = logger or logging.getLogger(__name__)
    if not path.exists():
        raise FrameExtractionError(f"video file not found: {path}")
    if target == "at_timestamp" and not _is_finite_non_negative(timestamp_seconds):
        raise FrameExtractionError(
            "at_timestamp requires a finite non-negative timestamp_seconds"
        )

    downgrade_note = ""
    requested_ts = timestamp_seconds if target == "at_timestamp" else None
    use_end_seek = False
    meta: VideoMetadata | None = None
    overshoot_measured = False
    overshoot_downgrade = ""

    if target == "first_frame":
        actual_ts = 0.0
        resolved_target = "first_frame"
    elif target in ("last_frame", "at_timestamp"):
        try:
            meta = await probe_video(path)
        except FrameExtractionError:
            meta = None
        if target == "last_frame":
            if meta and meta.duration_seconds > 0:
                actual_ts = _index_end(meta, "last")
            else:
                # No probe -> can't compute a duration-based timestamp. Input
                # seeking past EOF returns 0 bytes, so seek from the end instead
                # (an -ss sentinel would just produce an empty frame and fail).
                actual_ts = 0.0
                use_end_seek = True
            resolved_target = "last_frame"
        else:
            assert timestamp_seconds is not None
            overshoot_measured = (
                meta is not None
                and meta.duration_seconds > 0
                and timestamp_seconds > meta.duration_seconds
            )
            if overshoot_measured:
                assert meta is not None
                if not fallback_to_last_on_overshoot:
                    raise FrameExtractionError(
                        f"timestamp {timestamp_seconds}s exceeds video duration {meta.duration_seconds}s"
                    )
                actual_ts = _index_end(meta, reused_frame_index)
                fallback_word = "first" if reused_frame_index == "first" else "last"
                overshoot_downgrade = (
                    f"timestamp_past_video_end_used_{fallback_word}_frame"
                )
                resolved_target = (
                    "first_frame" if reused_frame_index == "first" else "last_frame"
                )
            else:
                actual_ts = float(timestamp_seconds)
                resolved_target = "at_timestamp"
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
                resolved_target=resolved_target,
            )
        except FrameExtractionError as exc:
            if getattr(exc, "pixel_cap", False):
                raise
            logger.debug("imageio first_frame failed; falling through to ffmpeg: %s", exc)

    direct_saw_damage: list[bool] = []
    try:
        png_bytes, w, h = await _extract_frame_ffmpeg(
            path, timestamp_seconds=actual_ts, logger=logger, from_end=use_end_seek,
            saw_damage=direct_saw_damage,
        )
        if not downgrade_note and direct_saw_damage and direct_saw_damage[0]:
            downgrade_note = "frame_damaged_used_last_decodable_frame"
        elif not downgrade_note and overshoot_measured:
            downgrade_note = overshoot_downgrade
    except FrameExtractionError as exc:
        if use_end_seek and exc.no_frame:
            return await _imageio_last_resort(
                path, requested_ts=requested_ts, downgrade_note=downgrade_note,
                logger=logger,
            )
        if use_end_seek or target == "first_frame" or (
            not exc.no_frame and exc.returncode not in _RETRYABLE_FFMPEG_EXITS
        ) or (
            target == "at_timestamp" and not fallback_to_last_on_overshoot
        ):
            raise
        rescue_first = target == "at_timestamp" and reused_frame_index == "first"
        logger.debug(
            "ffmpeg seek to %.3fs produced no frame; falling back to %s frame", actual_ts,
            "first" if rescue_first else "last",
        )
        ladder_saw_damage: list[bool] = []
        try:
            png_bytes, w, h = await _extract_frame_ffmpeg(
                path, timestamp_seconds=0.0, logger=logger,
                from_end=not rescue_first, saw_damage=ladder_saw_damage,
            )
        except FrameExtractionError:
            png_bytes, w, h = await asyncio.to_thread(
                _extract_frame_imageio_sync, path, frame_index=0,
            )
            actual_ts = 0.0
            if not downgrade_note:
                downgrade_note = "frame_damaged_used_first_frame"
            return ExtractedFrame(
                image_bytes=png_bytes, width=w, height=h,
                actual_timestamp_seconds=actual_ts,
                requested_timestamp_seconds=requested_ts,
                downgrade_note=downgrade_note,
                resolved_target="first_frame",
            )
        use_end_seek = not rescue_first
        if rescue_first:
            actual_ts = 0.0
            resolved_target = "first_frame"
        else:
            resolved_target = "last_frame"
            if meta is not None and meta.duration_seconds > 0 and (
                meta.duration_is_stream or not meta.has_audio
            ):
                actual_ts = _index_end(meta, "last")
            else:
                actual_ts = float("nan")
                logger.debug(
                    "probe failed or duration unmeasurable: rescue frame position is "
                    "unmeasurable",
                )
        walked_past_damage = bool(ladder_saw_damage and ladder_saw_damage[0])
        if walked_past_damage and not rescue_first and not downgrade_note:
            downgrade_note = "frame_damaged_used_last_decodable_frame"
        elif not downgrade_note and target == "at_timestamp" and overshoot_measured:
            downgrade_note = ("frame_past_eof_used_first_frame" if rescue_first
                             else "frame_past_eof_used_last_frame")
        elif not downgrade_note and target == "at_timestamp":
            downgrade_note = ("frame_seek_failed_used_first_frame" if rescue_first
                             else "frame_seek_failed_used_last_frame")
    return ExtractedFrame(
        image_bytes=png_bytes, width=w, height=h,
        actual_timestamp_seconds=actual_ts,
        requested_timestamp_seconds=requested_ts,
        downgrade_note=downgrade_note,
        resolved_target=resolved_target,
    )
