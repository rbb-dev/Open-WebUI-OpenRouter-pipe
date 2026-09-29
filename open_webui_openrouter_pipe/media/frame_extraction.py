"""Video frame extraction at first / last / arbitrary timestamp.

PIL+imageio first; ffmpeg subprocess as fallback for codecs imageio can't handle.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import math
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import imageio.v3 as iio  # type: ignore[import-untyped]
from PIL import Image

from .image_conversion import composite_on_white

_MAX_FRAME_PIXELS = 25_000_000
_MAX_FRAME_WIDTH = 1920
_MAX_CONCURRENT_EXTRACTIONS = 4
_FFMPEG_TIMEOUT_S = 30.0
_PROBE_TIMEOUT_S = 10.0
_END_SEEK_WINDOWS = ("-1", "-5", "-30")
_END_SEEK_HOP_SECONDS = 1.0
_END_SEEK_BUDGET_SECONDS = 2 * _FFMPEG_TIMEOUT_S
_RETRYABLE_FFMPEG_EXITS = frozenset({69})
_TRACK_LENGTH_TOLERANCE_S = 0.5
_MAX_SEEK_SECONDS = 1e12
_VIDEO_HEAD = re.compile(r"Video:")
_STREAM_INDEX = re.compile(r"Stream #\d+:(\d+)")
_MATROSKA_DURATION = re.compile(r"DURATION\s*:\s*(\d+):(\d+):([\d.]+)")
_MOV_DURATION = re.compile(r"Processing st:\s*(\d+),[^\n]*duration:\s*(\d+)")
_TIME_BASE = re.compile(r"1/(\d+)\s*:")
_PICTURE_CODEC = re.compile(r"Video:\s*(?:png|mjpeg|bmp|gif|webp|tiff)\b")
_INPUT_DEMUXER: dict[str, str] = {
    ".mp4": "mov",
    ".m4v": "mov",
    ".mov": "mov",
    ".mkv": "matroska",
    ".webm": "matroska",
    ".avi": "avi",
    ".h264": "h264",
    ".264": "h264",
    ".h265": "hevc",
    ".265": "hevc",
    ".hevc": "hevc",
}
_PLAYLIST_MAGIC: tuple[bytes, ...] = (
    b"ffconcat version",
    b"#EXTM3U",
    b"#EXT-X-",
    b"<?xml",
)
_PLAYLIST_HEAD_BYTES = 64

_extraction_semaphore: asyncio.Semaphore | None = None
_ABANDONED = "extraction abandoned: the awaiting task was cancelled"


def _cancelled(cancel: threading.Event | None) -> bool:
    return cancel is not None and cancel.is_set()


async def _abandonable(func, /, *args, **kwargs):
    cancel = threading.Event()
    try:
        return await asyncio.to_thread(func, *args, cancel=cancel, **kwargs)
    except asyncio.CancelledError:
        cancel.set()
        raise


def _ensure_extraction_semaphore() -> asyncio.Semaphore:
    global _extraction_semaphore
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    sem = _extraction_semaphore
    if sem is not None:
        try:
            sem_loop = getattr(sem, "_get_loop", lambda: None)()
        except RuntimeError:
            sem_loop = None
        if current_loop is not None and sem_loop is not current_loop:
            _extraction_semaphore = None
    if _extraction_semaphore is None:
        _extraction_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_EXTRACTIONS)
    return _extraction_semaphore


class FrameExtractionError(Exception):
    """Raised when a frame cannot be extracted from a video file.
    """

    def __init__(self, message: str, *, no_frame: bool = False,
                 returncode: int | None = None, pixel_cap: bool = False,
                 byte_budget: bool = False) -> None:
        super().__init__(message)
        self.no_frame = no_frame
        self.returncode = returncode
        self.pixel_cap = pixel_cap
        self.byte_budget = byte_budget


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


def _refuse_unsafe_input(path: Path) -> str:
    if str(path).startswith("-"):
        raise FrameExtractionError("refusing path starting with '-' (argv injection guard)")
    try:
        with path.open("rb") as handle:
            head = handle.read(_PLAYLIST_HEAD_BYTES)
    except OSError:
        head = b""
    for magic in _PLAYLIST_MAGIC:
        if head.startswith(magic):
            raise FrameExtractionError(
                "refusing input that is a playlist naming a second file"
            )
    return _INPUT_DEMUXER.get(path.suffix.lower(), "")


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


def _seek_seconds(timestamp_seconds: float) -> str:
    if not math.isfinite(timestamp_seconds):
        return "0"
    return f"{min(max(0.0, timestamp_seconds), _MAX_SEEK_SECONDS):.6f}"


def _end_seek_hop_argv() -> list[list[str]]:
    reach = max(abs(float(w)) for w in _END_SEEK_WINDOWS)
    hops = math.ceil(reach / _END_SEEK_HOP_SECONDS)
    return [
        ["-sseof", f"-{_seek_seconds(k * _END_SEEK_HOP_SECONDS)}",
         "-t", _seek_seconds(_END_SEEK_HOP_SECONDS)]
        for k in range(1, hops + 1)
    ]


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


def _ffmpeg_binary() -> str | None:
    binary = shutil.which("ffmpeg")
    if binary is not None:
        return binary
    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - a missing wheel is a host with no ffmpeg
        return None


def _ffmpeg_video_stream_duration(path: Path, binary: str) -> float | None:
    def _run(level: str | None) -> str | None:
        level_arg = ["-loglevel", level] if level is not None else []
        try:
            proc = subprocess.run(
                [binary, "-hide_banner", "-nostats", "-protocol_whitelist", "file",
                 *level_arg, "-i", str(path)],
                capture_output=True, text=True, errors="replace",
                timeout=_PROBE_TIMEOUT_S, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stderr or ""

    stderr = _run(None)
    if stderr is not None:
        values: list[float] = []
        in_video = False
        for line in stderr.splitlines():
            if "Stream #" in line:
                in_video = (
                    _VIDEO_HEAD.search(line) is not None
                    and _PICTURE_CODEC.search(line) is None
                )
            elif in_video and "Metadata:" not in line:
                found = _MATROSKA_DURATION.search(line)
                if found is not None:
                    hours, minutes, seconds = found.groups()
                    value = int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)
                    if value > 0:
                        values.append(value)
        if values:
            return max(values)
    stderr = _run("debug")
    if stderr is None:
        return None
    found_values: list[float] = []
    for line in stderr.splitlines():
        if "Stream #" not in line or not _VIDEO_HEAD.search(line):
            continue
        if _PICTURE_CODEC.search(line):
            continue
        position = _STREAM_INDEX.search(line)
        base = _TIME_BASE.search(line.split("Video:")[0])
        stream_index = position.group(1) if position else None
        time_base = int(base.group(1)) if base else None
        if stream_index is None or not time_base:
            continue
        value = None
        for found in _MOV_DURATION.finditer(stderr):
            if found.group(1) == stream_index:
                value = int(found.group(2)) / time_base
        if value is not None and value > 0:
            found_values.append(value)
    if found_values:
        return max(found_values)
    return None


def _last_time_seconds(stderr: str) -> float | None:
    for line in reversed(stderr.splitlines()):
        marker = line.rfind("time=")
        if marker < 0:
            continue
        token = line[marker + len("time="):].split(" ", 1)[0].strip()
        sign = -1.0 if token.startswith("-") else 1.0
        parts = token.lstrip("-").split(":")
        if len(parts) != 3:
            continue
        try:
            hours, minutes, seconds = (float(part) for part in parts)
        except ValueError:
            continue
        total = hours * 3600.0 + minutes * 60.0 + seconds
        if math.isfinite(total):
            return sign * total
    return None


def _video_track_seconds_sync(path: Path) -> float | None:
    try:
        _refuse_unsafe_input(path)
    except FrameExtractionError:
        return None
    binary = _ffmpeg_binary()
    if binary is None:
        return None
    try:
        proc = subprocess.run(
            [binary, "-hide_banner", "-nostats", "-protocol_whitelist", "file",
             "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    measured = _last_time_seconds(proc.stderr or "")
    return measured if measured is not None and measured > 0 else None


async def _video_track_seconds(path: Path) -> float | None:
    return await asyncio.to_thread(_video_track_seconds_sync, path)


def _probe_video_sync(path: Path, cancel: threading.Event | None = None) -> VideoMetadata:
    """Blocking video probe. Caller wraps in to_thread."""
    if _cancelled(cancel):
        raise FrameExtractionError(_ABANDONED)
    _refuse_unsafe_input(path)
    try:
        meta = iio.immeta(str(path), exclude_applied=False)  # type: ignore[no-any-return]
        if _cancelled(cancel):
            raise FrameExtractionError(_ABANDONED)
        duration = float(meta.get("duration", 0.0) or 0.0)
        duration_is_stream = False
        if duration > 0:
            probed = _ffprobe_stream_duration(path)
            if _cancelled(cancel):
                raise FrameExtractionError(_ABANDONED)
            if probed is None:
                binary = _ffmpeg_binary()
                if binary is not None:
                    probed = _ffmpeg_video_stream_duration(path, binary)
                    if _cancelled(cancel):
                        raise FrameExtractionError(_ABANDONED)
            if probed is not None and probed > 0:
                duration = probed
                duration_is_stream = True
        fps_raw = meta.get("fps") or 0.0
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
    return await _abandonable(_probe_video_sync, path)


async def _container_length_is_the_pictures(meta: VideoMetadata, path: Path) -> bool:
    if meta.duration_is_stream or not meta.has_audio:
        return True
    video_s = await _video_track_seconds(path)
    if video_s is None:
        return True
    return meta.duration_seconds - video_s <= _TRACK_LENGTH_TOLERANCE_S


# -----------------------------------------------------------------------------
# Frame extraction
# -----------------------------------------------------------------------------

def _check_frame_bytes(data: bytes, max_frame_bytes: int) -> None:
    if 0 < max_frame_bytes < len(data):
        raise FrameExtractionError(
            f"frame too large: {len(data)} bytes exceeds the {max_frame_bytes} byte "
            f"frame budget",
            byte_budget=True,
        )


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
    _refuse_unsafe_input(path)
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
    path: Path, *, frame_index: int, cancel: threading.Event | None = None
) -> tuple[bytes, int, int]:
    """Extract a single frame at the given index via imageio. Returns
    (png_bytes, width, height). Raises FrameExtractionError on failure or
    on decompression-bomb-sized output."""
    if _cancelled(cancel):
        raise FrameExtractionError(_ABANDONED)
    _refuse_unsafe_input(path)
    try:
        declared = _declared_size(path)
        if _cancelled(cancel):
            raise FrameExtractionError(_ABANDONED)
        if declared is not None and _over_pixel_cap(*declared):
            raise _pixel_cap_refusal(*declared)
        read_kwargs: dict[str, Any] = {}
        if declared is not None and declared[0] > _MAX_FRAME_WIDTH:
            read_kwargs["output_params"] = ["-vf", f"scale={_MAX_FRAME_WIDTH}:-2"]
        try:
            arr = iio.imread(str(path), index=frame_index, **read_kwargs)
        except TypeError:
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
    input_format = _refuse_unsafe_input(path)

    ffmpeg_bin = await asyncio.to_thread(_ffmpeg_binary)
    if ffmpeg_bin is None:
        raise FrameExtractionError("ffmpeg unavailable")

    if from_end:
        # Input-seeking with -ss past the last frame returns 0 bytes, so to grab
        # the true last frame we seek a short window before EOF, scale, then
        # reverse it — frame 1 of the reversed tail is the last decodable frame.
        seek_arg_sets = _end_seek_hop_argv()
        deadline = time.monotonic() + _END_SEEK_BUDGET_SECONDS
        vf = f"scale='min({_MAX_FRAME_WIDTH},iw)':-2,reverse"
    else:
        seek_arg_sets = [["-ss", _seek_seconds(timestamp_seconds)]]
        deadline = None
        vf = f"scale='min({_MAX_FRAME_WIDTH},iw)':-2"
    last_no_frame: FrameExtractionError | None = None
    walked_past_damage = False
    for seek_args in seek_arg_sets:
        if deadline is not None and time.monotonic() >= deadline:
            break
        cmd = [
            ffmpeg_bin,
            "-protocol_whitelist", "file",
            *seek_args,
            *(("-f", input_format) if input_format else ()),
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
    downgrade_note: str, logger: logging.Logger, max_frame_bytes: int = 0,
) -> ExtractedFrame:
    png_bytes, w, h = await _abandonable(
        _extract_frame_imageio_sync, path, frame_index=0,
    )
    if not downgrade_note:
        downgrade_note = "frame_damaged_used_first_frame"
    logger.debug("ffmpeg produced no frame; the file's first frame is the last resort")
    _check_frame_bytes(png_bytes, max_frame_bytes)
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
    max_frame_bytes: int = 0,
    meta: VideoMetadata | None = None,
) -> ExtractedFrame:
    return await _extract_frame_with_budget(
        path,
        target=target,
        timestamp_seconds=timestamp_seconds,
        fallback_to_last_on_overshoot=fallback_to_last_on_overshoot,
        reused_frame_index=reused_frame_index,
        logger=logger,
        max_frame_bytes=max_frame_bytes,
        meta=meta,
    )


async def _extract_frame_with_budget(
    path: Path,
    *,
    target: Literal["first_frame", "last_frame", "at_timestamp"],
    timestamp_seconds: float | None = None,
    fallback_to_last_on_overshoot: bool = True,
    reused_frame_index: Literal["first", "last"] = "last",
    logger: logging.Logger | None = None,
    max_frame_bytes: int = 0,
    meta: VideoMetadata | None = None,
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
    _refuse_unsafe_input(path)
    if target == "at_timestamp" and not _is_finite_non_negative(timestamp_seconds):
        raise FrameExtractionError(
            "at_timestamp requires a finite non-negative timestamp_seconds"
        )

    async with _ensure_extraction_semaphore():
        downgrade_note = ""
        requested_ts = timestamp_seconds if target == "at_timestamp" else None
        use_end_seek = False
        probed_meta: VideoMetadata | None = meta
        overshoot_measured = False
        overshoot_downgrade = ""

        if target == "first_frame":
            actual_ts = 0.0
            resolved_target = "first_frame"
        elif target in ("last_frame", "at_timestamp"):
            if probed_meta is None:
                try:
                    probed_meta = await probe_video(path)
                except FrameExtractionError:
                    probed_meta = None
            if target == "last_frame":
                if probed_meta and probed_meta.duration_seconds > 0:
                    actual_ts = _index_end(probed_meta, "last")
                else:
                    # No probe -> can't compute a duration-based timestamp. Input
                    # seeking past EOF returns 0 bytes, so seek from the end instead
                    # (an -ss sentinel would just produce an empty frame and fail).
                    actual_ts = float("nan")
                    use_end_seek = True
                resolved_target = "last_frame"
            else:
                assert timestamp_seconds is not None
                overshoot_measured = (
                    probed_meta is not None
                    and probed_meta.duration_seconds > 0
                    and timestamp_seconds > probed_meta.duration_seconds
                )
                if overshoot_measured:
                    assert probed_meta is not None
                    if not fallback_to_last_on_overshoot:
                        raise FrameExtractionError(
                            f"timestamp {timestamp_seconds}s exceeds video duration {probed_meta.duration_seconds}s"
                        )
                    actual_ts = _index_end(probed_meta, reused_frame_index)
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
            first_frame: tuple[bytes, int, int] | None = None
            try:
                first_frame = await _abandonable(
                    _extract_frame_imageio_sync, path, frame_index=0,
                )
            except FrameExtractionError as exc:
                if getattr(exc, "pixel_cap", False):
                    raise
                logger.debug("imageio first_frame failed; falling through to ffmpeg: %s", exc)
            if first_frame is not None:
                first_frame_bytes, w, h = first_frame
                _check_frame_bytes(first_frame_bytes, max_frame_bytes)
                return ExtractedFrame(
                    image_bytes=first_frame_bytes, width=w, height=h,
                    actual_timestamp_seconds=0.0,
                    requested_timestamp_seconds=requested_ts,
                    downgrade_note=downgrade_note,
                    resolved_target=resolved_target,
                )

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
            if use_end_seek and (exc.no_frame or exc.returncode in _RETRYABLE_FFMPEG_EXITS):
                return await _imageio_last_resort(
                    path, requested_ts=requested_ts, downgrade_note=downgrade_note,
                    logger=logger, max_frame_bytes=max_frame_bytes,
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
                if (
                    target == "at_timestamp"
                    and not rescue_first
                    and probed_meta is not None
                    and probed_meta.duration_seconds > 0
                ):
                    last_index_end = _index_end(probed_meta, "last")
                    try:
                        png_bytes, w, h = await _extract_frame_ffmpeg(
                            path, timestamp_seconds=last_index_end,
                            logger=logger, from_end=False,
                        )
                    except FrameExtractionError:
                        pass
                    else:
                        return ExtractedFrame(
                            image_bytes=png_bytes, width=w, height=h,
                            actual_timestamp_seconds=last_index_end,
                            requested_timestamp_seconds=requested_ts,
                            downgrade_note=downgrade_note or "frame_seek_failed_used_last_frame",
                            resolved_target="last_frame",
                        )
                png_bytes, w, h = await _abandonable(
                    _extract_frame_imageio_sync, path, frame_index=0,
                )
                _check_frame_bytes(png_bytes, max_frame_bytes)
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
                if probed_meta is not None and probed_meta.duration_seconds > 0 and (
                    await _container_length_is_the_pictures(probed_meta, path)
                ):
                    actual_ts = _index_end(probed_meta, "last")
                else:
                    actual_ts = float("nan")
                    logger.debug(
                        "probe failed or duration unmeasurable: rescue frame position is "
                        "unmeasurable",
                    )
            walked_past_damage = bool(
                (direct_saw_damage and direct_saw_damage[0])
                or (ladder_saw_damage and ladder_saw_damage[0])
            )
            if walked_past_damage and not rescue_first and not downgrade_note:
                downgrade_note = "frame_damaged_used_last_decodable_frame"
            elif not downgrade_note and target == "at_timestamp" and overshoot_measured:
                downgrade_note = ("frame_past_eof_used_first_frame" if rescue_first
                                 else "frame_past_eof_used_last_frame")
            elif not downgrade_note and target == "at_timestamp":
                downgrade_note = ("frame_seek_failed_used_first_frame" if rescue_first
                                 else "frame_seek_failed_used_last_frame")
        _check_frame_bytes(png_bytes, max_frame_bytes)
        return ExtractedFrame(
            image_bytes=png_bytes, width=w, height=h,
            actual_timestamp_seconds=actual_ts,
            requested_timestamp_seconds=requested_ts,
            downgrade_note=downgrade_note,
            resolved_target=resolved_target,
        )
