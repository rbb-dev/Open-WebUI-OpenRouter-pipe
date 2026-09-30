"""Video frame extraction at first / last / arbitrary timestamp.

PIL+imageio first; ffmpeg subprocess as fallback for codecs imageio can't handle.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import io
import logging
import math
import re
import shutil
import subprocess
import threading
import time
import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Self

import imageio.v3 as iio  # type: ignore[import-untyped]
from PIL import Image

from .image_conversion import composite_on_white

_MAX_FRAME_PIXELS = 25_000_000
_MAX_FRAME_WIDTH = 1920
_MAX_CONCURRENT_EXTRACTIONS = 4
_FFMPEG_TIMEOUT_S = 30.0
_PROBE_TIMEOUT_S = 10.0
_IMAGEIO_TIMEOUT_S = 30.0
_PROBE_DEADLINE_S = _PROBE_TIMEOUT_S * 3 + _IMAGEIO_TIMEOUT_S
_END_SEEK_WINDOWS = ("-1", "-5", "-30")
_END_SEEK_HOP_SECONDS = 1.0
_END_SEEK_BUDGET_SECONDS = 2 * _FFMPEG_TIMEOUT_S
_RETRYABLE_FFMPEG_EXITS = frozenset({69})
_TRACK_LENGTH_TOLERANCE_S = 0.5
_MAX_SEEK_SECONDS = 1e12
_FRAME_READ_CHUNK_BYTES = 64 * 1024
_FRAME_READ_SLACK_BYTES = 256 * 1024
_VIDEO_HEAD = re.compile(r"Video:")
_STREAM_INDEX = re.compile(r"Stream #\d+:(\d+)")
_MATROSKA_DURATION = re.compile(r"DURATION\s*:\s*(\d+):(\d+):([\d.]+)")
_MOV_DURATION = re.compile(r"Processing st:\s*(\d+),[^\n]*duration:\s*(\d+)")
_TIME_BASE = re.compile(r"1/(\d+)\s*:")
_PICTURE_CODEC = re.compile(r"Video:\s*(?:png|mjpeg|bmp|gif|webp|tiff)\b")
_VIDEO_STREAM = re.compile(r"Stream #\d+:\d+.*Video:")
_VIDEO_STREAM_SIZE = re.compile(r"(?<=\s)(\d{1,5})x(\d{1,5})(?![0-9x])")
_VIDEO_STREAM_RATE = re.compile(r"(\d+(?:\.\d+)?)\s+(?:fps|tbr)\b")
_AUDIO_STREAM = re.compile(r"Stream #\d+:\d+.*Audio:")
_CONTAINER_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_INPUT_DEMUXER: dict[str, str] = {
    ".mp4": "mov",
    ".m4v": "mov",
    ".mov": "mov",
    ".3gp": "mov",
    ".3g2": "mov",
    ".mkv": "matroska",
    ".webm": "matroska",
    ".avi": "avi",
    ".ogv": "ogg",
    ".mpeg": "mpegvideo",
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
    b"<MPD",
)
_PLAYLIST_HEAD_BYTES = 64

_extraction_semaphores: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_ACTIVE_SLOT: contextvars.ContextVar = contextvars.ContextVar("extraction_slot", default=None)
_ABANDONED = "extraction abandoned: the awaiting task was cancelled"


def _cancelled(cancel: threading.Event | None) -> bool:
    return cancel is not None and cancel.is_set()


class _ExtractionSlot:
    def __init__(self, sem: asyncio.Semaphore, loop: asyncio.AbstractEventLoop) -> None:
        self._sem = sem
        self._loop = loop
        self._lock = threading.Lock()
        self._pending = 0
        self._closed = False
        self._given_back = False

    async def __aenter__(self) -> Self:
        await self._sem.acquire()
        _ACTIVE_SLOT.set(self)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        _ACTIVE_SLOT.set(None)
        with self._lock:
            self._closed = True
            if self._given_back or self._pending:
                return
            self._given_back = True
        self._sem.release()

    def spend(self) -> _ExtractionSpend:
        with self._lock:
            self._pending += 1
        return _ExtractionSpend(self)

    def reclaim(self) -> None:
        with self._lock:
            self._pending -= 1
            if self._given_back or not self._closed or self._pending:
                return
            self._given_back = True
        try:
            self._loop.call_soon_threadsafe(self._sem.release)
        except RuntimeError:
            pass


class _ExtractionSpend:
    def __init__(self, slot: _ExtractionSlot) -> None:
        self._slot = slot
        self._lock = threading.Lock()
        self._entered = False
        self._abandoned = False

    def enter(self) -> bool:
        with self._lock:
            if self._abandoned:
                return False
            self._entered = True
            return True

    def abandon(self) -> None:
        with self._lock:
            if self._entered or self._abandoned:
                return
            self._abandoned = True
        self._slot.reclaim()

    def finish(self) -> None:
        self._slot.reclaim()


def _extraction_slot() -> _ExtractionSlot:
    return _ExtractionSlot(_ensure_extraction_semaphore(), asyncio.get_running_loop())


async def _abandonable(
    func, /, *args, deadline: float | None = None, label: str = "", **kwargs
) -> Any:
    cancel = threading.Event()
    slot = _ACTIVE_SLOT.get()
    spend: _ExtractionSpend | None = None
    if slot is None:
        offload = asyncio.to_thread(func, *args, cancel=cancel, **kwargs)
    else:
        charge = slot.spend()

        def _run(charge: _ExtractionSpend = charge):
            if not charge.enter():
                return None
            try:
                return func(*args, cancel=cancel, **kwargs)
            finally:
                charge.finish()

        spend = charge
        offload = asyncio.to_thread(_run)
    try:
        if deadline is None:
            return await offload
        return await asyncio.wait_for(offload, timeout=deadline)
    except asyncio.CancelledError:
        cancel.set()
        if spend is not None:
            spend.abandon()
        raise
    except TimeoutError:
        cancel.set()
        if spend is not None:
            spend.abandon()
        raise FrameExtractionError(f"{label} timed out after {deadline}s") from None


def _ensure_extraction_semaphore() -> asyncio.Semaphore:
    try:
        loop: Any = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.Semaphore(_MAX_CONCURRENT_EXTRACTIONS)
    sem = _extraction_semaphores.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(_MAX_CONCURRENT_EXTRACTIONS)
        _extraction_semaphores[loop] = sem
    return sem


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
    demuxer = _INPUT_DEMUXER.get(path.suffix.lower())
    if not demuxer:
        raise FrameExtractionError(
            "refusing input whose container is not one the pipe names"
        )
    return demuxer


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

def _usable_duration(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _ffprobe_stream_duration(path: Path) -> float | None:
    binary = shutil.which("ffprobe")
    if binary is None:
        return None
    try:
        input_format = _refuse_unsafe_input(path)
    except FrameExtractionError:
        return None
    try:
        proc = subprocess.run(
            [binary, "-v", "error", "-f", input_format,
             "-select_streams", "v:0",
             "-show_entries", "stream=duration", "-of", "csv=p=0",
             "-protocol_whitelist", "file", str(path)],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return _usable_duration((proc.stdout or "").strip())


def _ffmpeg_binary(cancel: threading.Event | None = None) -> str | None:
    if _cancelled(cancel):
        return None
    binary = shutil.which("ffmpeg")
    if binary is not None:
        return binary
    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - a missing wheel is a host with no ffmpeg
        return None


def _matroska_track_seconds(stderr: str) -> float | None:
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
    return None


def _mov_stream_seconds(stderr: str) -> float | None:
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


def _ffmpeg_video_stream_duration(path: Path, binary: str) -> float | None:
    try:
        input_format = _refuse_unsafe_input(path)
    except FrameExtractionError:
        return None
    try:
        proc = subprocess.run(
            [binary, "-hide_banner", "-nostats", "-protocol_whitelist", "file",
             "-f", input_format,
             "-loglevel", "debug", "-i", str(path)],
            capture_output=True, text=True, errors="replace",
            timeout=_PROBE_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    stderr = proc.stderr or ""
    return _matroska_track_seconds(stderr) or _mov_stream_seconds(stderr)


def _pinned_probe(path: Path) -> dict[str, Any]:
    input_format = _refuse_unsafe_input(path)
    binary = _ffmpeg_binary()
    if binary is None:
        raise FrameExtractionError("ffmpeg unavailable")
    try:
        proc = subprocess.run(
            [binary, "-hide_banner", "-nostats", "-f", input_format,
             "-protocol_whitelist", "file", "-i", str(path)],
            capture_output=True, text=True, errors="replace",
            timeout=_PROBE_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FrameExtractionError(f"header read failed: {exc}") from exc
    stderr = proc.stderr or ""
    width, height = 0, 0
    fps = 0.0
    seen_video = False
    for line in stderr.splitlines():
        if _VIDEO_STREAM.search(line):
            seen_video = True
            size = _VIDEO_STREAM_SIZE.search(line)
            if size is not None:
                width, height = int(size.group(1)), int(size.group(2))
            rate = _VIDEO_STREAM_RATE.search(line)
            if rate is not None:
                fps = float(rate.group(1))
            break
    if not seen_video:
        raise FrameExtractionError("header read found no video stream to measure")
    duration = 0.0
    found = _CONTAINER_DURATION.search(stderr)
    if found is not None:
        hours, minutes, seconds = found.groups()
        duration = int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)
    return {
        "duration": duration,
        "fps": fps,
        "size": (width, height),
        "audio_codec": "audio" if _AUDIO_STREAM.search(stderr) else None,
    }


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


def _video_track_seconds_sync(
    path: Path, cancel: threading.Event | None = None,
) -> float | None:
    if _cancelled(cancel):
        return None
    try:
        input_format = _refuse_unsafe_input(path)
    except FrameExtractionError:
        return None
    binary = _ffmpeg_binary()
    if binary is None:
        return None
    try:
        proc = subprocess.run(
            [binary, "-hide_banner", "-nostats", "-protocol_whitelist", "file",
             "-f", input_format,
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
    return await _abandonable(
        _video_track_seconds_sync, path, label="video_track_seconds",
    )


def _probe_video_sync(path: Path, cancel: threading.Event | None = None) -> VideoMetadata:
    """Blocking video probe. Caller wraps in to_thread."""
    if _cancelled(cancel):
        raise FrameExtractionError(_ABANDONED)
    _refuse_unsafe_input(path)
    try:
        meta = _pinned_probe(path)
        if _cancelled(cancel):
            raise FrameExtractionError(_ABANDONED)
        duration = _usable_duration(meta.get("duration")) or 0.0
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
            probed = _usable_duration(probed)
            if probed is not None:
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
    return await _abandonable(
        _probe_video_sync, path, deadline=_PROBE_DEADLINE_S, label="probe_video",
    )


async def _pictures_own_length(meta: VideoMetadata, path: Path) -> float:
    if meta.duration_is_stream or not meta.has_audio:
        return meta.duration_seconds
    video_s = await _video_track_seconds(path)
    if video_s is None:
        return meta.duration_seconds
    if meta.duration_seconds - video_s <= _TRACK_LENGTH_TOLERANCE_S:
        return meta.duration_seconds
    return video_s


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
        meta = _pinned_probe(path)
        size = meta.get("size")
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return int(size[0]), int(size[1])
    except Exception:  # noqa: BLE001 - a header we cannot read is a header we lack
        return None
    return None


def _scaled_frame_size(width: int, height: int) -> tuple[int, int]:
    out_w = min(_MAX_FRAME_WIDTH, width)
    return out_w, max(2, int(height * out_w / width / 2.0 + 0.5) * 2)


def _read_cap(max_frame_bytes: int) -> int:
    if max_frame_bytes <= 0:
        return 0
    return max_frame_bytes + _FRAME_READ_SLACK_BYTES


async def _read_bounded(stream: Any, cap: int, budget: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await stream.read(_FRAME_READ_CHUNK_BYTES)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if 0 < cap < total:
            raise FrameExtractionError(
                f"frame too large: past the {cap} byte read cap on the way over "
                f"the {budget} byte frame budget ({total} bytes read)",
                byte_budget=True,
            )
        chunks.append(chunk)


async def _stop_child(
    proc: asyncio.subprocess.Process, stderr_task: asyncio.Task[bytes] | None
) -> None:
    with contextlib.suppress(Exception):
        proc.kill()
        await proc.wait()
    if stderr_task is not None:
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await stderr_task


def _normalise_frame_sync(
    stdout: bytes, cancel: threading.Event | None = None,
) -> tuple[bytes, int, int]:
    if _cancelled(cancel):
        raise FrameExtractionError(_ABANDONED)
    img = Image.open(io.BytesIO(stdout))
    if _over_pixel_cap(img.width, img.height):
        raise _ffmpeg_pixel_cap_refusal(img.width, img.height)
    img.load()
    if img.mode != "RGB":
        stdout = _normalise_png_mode(img)
    return stdout, img.width, img.height


def _extract_frame_imageio_sync(
    path: Path, *, frame_index: int, cancel: threading.Event | None = None,
    declared_size: tuple[int, int] | None = None,
) -> tuple[bytes, int, int]:
    """Extract a single frame at the given index via imageio. Returns
    (png_bytes, width, height). Raises FrameExtractionError on failure or
    on decompression-bomb-sized output."""
    if _cancelled(cancel):
        raise FrameExtractionError(_ABANDONED)
    input_format = _refuse_unsafe_input(path)
    try:
        declared = declared_size if declared_size is not None else _declared_size(path)
        if _cancelled(cancel):
            raise FrameExtractionError(_ABANDONED)
        if declared is not None and _over_pixel_cap(*declared):
            raise _pixel_cap_refusal(*declared)
        read_kwargs: dict[str, Any] = {
            "plugin": "FFMPEG",
            "input_params": ["-f", input_format, "-protocol_whitelist", "file"],
        }
        if declared is not None and declared[0] > _MAX_FRAME_WIDTH:
            read_kwargs["output_params"] = ["-vf", f"scale={_MAX_FRAME_WIDTH}:-2"]
        arr = iio.imread(str(path), index=frame_index, **read_kwargs)
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
    from_end: bool = False, saw_damage: list[bool] | None = None,
    max_frame_bytes: int = 0, hop_index: list[int] | None = None,
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

    ffmpeg_bin = await _abandonable(_ffmpeg_binary, label="ffmpeg_binary")
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
    for hop, seek_args in enumerate(seek_arg_sets):
        if deadline is not None and time.monotonic() >= deadline:
            break
        cmd = [
            ffmpeg_bin,
            "-protocol_whitelist", "file",
            *seek_args,
            "-f", input_format,
            "-i", path_str,
            "-frames:v", "1",
            "-vf", vf,
            "-f", "image2pipe",
            "-vcodec", "png",
            "-loglevel", "error",
            "-",
        ]
        proc: asyncio.subprocess.Process | None = None
        stderr_task: asyncio.Task[bytes] | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert proc.stdout is not None and proc.stderr is not None
            stdout_stream = proc.stdout
            stderr_task = asyncio.create_task(proc.stderr.read())
            try:
                stdout = await asyncio.wait_for(
                    _read_bounded(
                        stdout_stream, _read_cap(max_frame_bytes), max_frame_bytes,
                    ),
                    timeout=_FFMPEG_TIMEOUT_S,
                )
            except TimeoutError:
                await _stop_child(proc, stderr_task)
                raise FrameExtractionError(
                    f"ffmpeg timed out after {_FFMPEG_TIMEOUT_S}s",
                ) from None
            except FrameExtractionError:
                await _stop_child(proc, stderr_task)
                raise
            await proc.wait()
            stderr = await stderr_task
            if proc.returncode != 0:
                raise FrameExtractionError(
                    f"ffmpeg returned {proc.returncode}: {stderr.decode('utf-8', errors='replace')[:200]}",
                    returncode=proc.returncode,
                )
            if not stdout:
                raise FrameExtractionError("ffmpeg produced empty output", no_frame=True)
            stdout, width, height = await _abandonable(
                _normalise_frame_sync, stdout, label="normalise",
            )
            if saw_damage is not None:
                saw_damage.append(walked_past_damage)
            if hop_index is not None:
                hop_index.append(hop)
            return stdout, width, height
        except asyncio.CancelledError:
            if proc is not None:
                with contextlib.suppress(Exception):
                    proc.kill()
                    await proc.wait()
            if stderr_task is not None and not stderr_task.done():
                stderr_task.cancel()
            raise
        except FrameExtractionError as exc:
            if not exc.no_frame and exc.returncode not in _RETRYABLE_FFMPEG_EXITS:
                if stderr_task is not None and not stderr_task.done():
                    stderr_task.cancel()
                raise
            if exc.returncode in _RETRYABLE_FFMPEG_EXITS:
                walked_past_damage = True
            last_no_frame = exc
        except Exception as exc:
            if proc is not None:
                with contextlib.suppress(Exception):
                    proc.kill()
                    await proc.wait()
            if stderr_task is not None and not stderr_task.done():
                stderr_task.cancel()
            raise FrameExtractionError(f"ffmpeg extract failed: {exc}") from exc
    if saw_damage is not None:
        saw_damage.append(walked_past_damage)
    if hop_index is not None:
        hop_index.append(len(seek_arg_sets))
    if last_no_frame is not None:
        raise last_no_frame
    raise FrameExtractionError("ffmpeg extract failed: no seek attempted")


def _ladder_note(damage_seen: bool) -> str:
    return (
        "frame_damaged_used_last_decodable_frame" if damage_seen
        else "frame_seek_missed_used_nearest_decodable_frame"
    )


def _give_up_note(damage_seen: bool) -> str:
    return (
        "frame_damaged_used_first_frame" if damage_seen
        else "frame_end_unreadable_used_first_frame"
    )


async def _imageio_last_resort(
    path: Path, *, requested_ts: float | None,
    downgrade_note: str, logger: logging.Logger, max_frame_bytes: int = 0,
    declared_size: tuple[int, int] | None = None,
    damage_seen: bool = False,
) -> ExtractedFrame:
    png_bytes, w, h = await _abandonable(
        _extract_frame_imageio_sync, path, frame_index=0,
        deadline=_IMAGEIO_TIMEOUT_S, label="imageio",
        declared_size=declared_size,
    )
    if not downgrade_note:
        downgrade_note = _give_up_note(damage_seen)
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
    probe_failed: bool = False,
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
        probe_failed=probe_failed,
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
    probe_failed: bool = False,
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

    async with _extraction_slot():
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
            if probed_meta is None and not probe_failed:
                try:
                    probed_meta = await probe_video(path)
                except FrameExtractionError:
                    probed_meta = None
                    probe_failed = True
            if target == "last_frame":
                if probed_meta and probed_meta.duration_seconds > 0:
                    actual_ts = _index_end(
                        replace(
                            probed_meta,
                            duration_seconds=await _pictures_own_length(
                                probed_meta, path,
                            ),
                        ),
                        "last",
                    )
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
                    deadline=_IMAGEIO_TIMEOUT_S, label="imageio",
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

        display_size: tuple[int, int] | None = None
        if probed_meta is not None and probed_meta.width > 0 and probed_meta.height > 0:
            display_size = (probed_meta.width, probed_meta.height)
        elif not probe_failed:
            try:
                display_size = await asyncio.wait_for(
                    asyncio.to_thread(_declared_size, path), timeout=_PROBE_DEADLINE_S,
                )
            except TimeoutError:
                display_size = None
        if display_size is not None and display_size[0] > 0 and display_size[1] > 0:
            out_w, out_h = _scaled_frame_size(*display_size)
            if _over_pixel_cap(out_w, out_h):
                raise _ffmpeg_pixel_cap_refusal(out_w, out_h)

        direct_saw_damage: list[bool] = []
        direct_hop: list[int] = []
        try:
            png_bytes, w, h = await _extract_frame_ffmpeg(
                path, timestamp_seconds=actual_ts, logger=logger, from_end=use_end_seek,
                saw_damage=direct_saw_damage, max_frame_bytes=max_frame_bytes,
                hop_index=direct_hop,
            )
            walked_past_first_hop = bool(direct_hop and direct_hop[0] > 0)
            if (
                not downgrade_note
                and direct_saw_damage
                and (
                    direct_saw_damage[0]
                    or (use_end_seek and target == "last_frame" and walked_past_first_hop)
                )
            ):
                downgrade_note = _ladder_note(direct_saw_damage[0])
            elif not downgrade_note and overshoot_measured:
                downgrade_note = overshoot_downgrade
        except FrameExtractionError as exc:
            if use_end_seek and (exc.no_frame or exc.returncode in _RETRYABLE_FFMPEG_EXITS):
                return await _imageio_last_resort(
                    path, requested_ts=requested_ts, downgrade_note=downgrade_note,
                    logger=logger, max_frame_bytes=max_frame_bytes,
                    declared_size=(
                        (probed_meta.width, probed_meta.height)
                        if probed_meta is not None else None
                    ),
                    damage_seen=bool(direct_saw_damage and direct_saw_damage[0]),
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
                    max_frame_bytes=max_frame_bytes,
                )
            except FrameExtractionError:
                if (
                    target == "at_timestamp"
                    and not rescue_first
                    and probed_meta is not None
                    and probed_meta.duration_seconds > 0
                ):
                    last_index_end = _index_end(
                        replace(
                            probed_meta,
                            duration_seconds=await _pictures_own_length(
                                probed_meta, path,
                            ),
                        ),
                        "last",
                    )
                    try:
                        png_bytes, w, h = await _extract_frame_ffmpeg(
                            path, timestamp_seconds=last_index_end,
                            logger=logger, from_end=False,
                            max_frame_bytes=max_frame_bytes,
                        )
                    except FrameExtractionError:
                        pass
                    else:
                        _check_frame_bytes(png_bytes, max_frame_bytes)
                        walked_past_damage = bool(
                            (direct_saw_damage and direct_saw_damage[0])
                            or (ladder_saw_damage and ladder_saw_damage[0])
                        )
                        if walked_past_damage and not downgrade_note:
                            downgrade_note = "frame_damaged_used_last_decodable_frame"
                        elif not downgrade_note and overshoot_measured:
                            downgrade_note = overshoot_downgrade
                        return ExtractedFrame(
                            image_bytes=png_bytes, width=w, height=h,
                            actual_timestamp_seconds=last_index_end,
                            requested_timestamp_seconds=requested_ts,
                            downgrade_note=downgrade_note or "frame_seek_failed_used_last_frame",
                            resolved_target="last_frame",
                        )
                png_bytes, w, h = await _abandonable(
                    _extract_frame_imageio_sync, path, frame_index=0,
                    deadline=_IMAGEIO_TIMEOUT_S, label="imageio",
                    declared_size=(
                        (probed_meta.width, probed_meta.height)
                        if probed_meta is not None else None
                    ),
                )
                _check_frame_bytes(png_bytes, max_frame_bytes)
                actual_ts = 0.0
                if not downgrade_note:
                    downgrade_note = _give_up_note(
                        bool(
                            (direct_saw_damage and direct_saw_damage[0])
                            or (ladder_saw_damage and ladder_saw_damage[0])
                        )
                    )
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
                actual_ts = float("nan")
            damage_seen = bool(
                (direct_saw_damage and direct_saw_damage[0])
                or (ladder_saw_damage and ladder_saw_damage[0])
            )
            if (
                not rescue_first
                and not downgrade_note
                and (damage_seen or target == "last_frame")
            ):
                downgrade_note = _ladder_note(damage_seen)
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
