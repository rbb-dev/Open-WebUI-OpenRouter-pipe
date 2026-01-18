"""Logging system with session-based log capture and archival.

This module handles all logging-related functionality:
- SessionLogger: Per-request logger with context-aware buffering
- Log event classification and formatting
- Async log queue processing
- Session log archival (encrypted zip files)
- Automatic cleanup of stale sessions

The SessionLogger uses contextvars to track request_id and session_id,
enabling per-request log isolation and structured event capture.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import logging
import os
import sys
import threading
import time
import traceback
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import _sanitize_path_component
from ..core.timing_logger import timed

try:
    import pyzipper  # type: ignore[import-untyped]
except ImportError:
    pyzipper = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Session Log Archive Job
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class _SessionLogArchiveJob:
    """Represents a single session log archive request destined for the writer thread."""

    base_dir: str
    zip_password: bytes
    zip_compression: str
    zip_compresslevel: Optional[int]
    user_id: str
    session_id: str
    chat_id: str
    message_id: str
    request_id: str
    created_at: float
    log_format: str
    log_events: list[dict[str, Any]]


# -----------------------------------------------------------------------------
# SessionLogger Class
# -----------------------------------------------------------------------------

class SessionLogger:
    """Per-request logger that captures console output and an in-memory log buffer.

    The logger tracks two identifiers via contextvars:
    - session_id: Open WebUI session identifier (for status/errors/debug).
    - request_id: Per-request unique id used to key the in-memory log buffer.

    Cleanup is intentional and explicit: request handlers call ``cleanup`` once
    they finish streaming so there is no background task silently pruning logs.

    Attributes:
        session_id: ContextVar storing the Open WebUI session id.
        request_id: ContextVar storing the per-request buffer key.
        log_level:  ContextVar storing the minimum level to emit for this request.
        logs:       Map of request_id -> fixed-size deque of structured log events (dicts).
    """

    session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
    request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
    user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
    log_level: ContextVar[int] = ContextVar("log_level", default=logging.INFO)
    max_lines: int = 2000
    logs: Dict[str, deque[dict[str, Any]]] = {}
    _session_last_seen: Dict[str, float] = {}
    log_queue: asyncio.Queue[logging.LogRecord] | None = None
    _main_loop: asyncio.AbstractEventLoop | None = None
    _state_lock = threading.Lock()
    _console_formatter = logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _memory_formatter = logging.Formatter("%(asctime)s [%(levelname)s] [user=%(user_id)s] %(message)s")

    @staticmethod
    @timed
    def _classify_event_type(message: str) -> str:
        msg = (message or "").lstrip()
        if msg.startswith("OpenRouter request headers:"):
            return "openrouter.request.headers"
        if msg.startswith("OpenRouter request payload:"):
            return "openrouter.request.payload"
        if msg.startswith("OpenRouter payload:"):
            return "openrouter.sse.event"
        if msg.startswith("Tool ") or msg.startswith("🔧") or msg.startswith("Skipping "):
            return "pipe.tools"
        return "pipe"

    @classmethod
    @timed
    def _build_event(cls, record: logging.LogRecord) -> dict[str, Any]:
        """Return a structured session log event extracted from a LogRecord."""
        try:
            message = record.getMessage()
        except Exception:
            message = str(getattr(record, "msg", "") or "")

        event_type = cls._classify_event_type(message)

        event: dict[str, Any] = {
            "created": float(getattr(record, "created", time.time())),
            "level": str(getattr(record, "levelname", "INFO") or "INFO"),
            "logger": str(getattr(record, "name", "") or ""),
            "request_id": getattr(record, "request_id", None),
            "session_id": getattr(record, "session_id", None),
            "user_id": getattr(record, "user_id", None),
            "event_type": event_type,
            "module": str(getattr(record, "module", "") or ""),
            "func": str(getattr(record, "funcName", "") or ""),
            "lineno": int(getattr(record, "lineno", 0) or 0),
        }

        try:
            exc_info = getattr(record, "exc_info", None)
            exc_text = getattr(record, "exc_text", None)
            if exc_text:
                event["exception"] = {"text": str(exc_text)}
            elif exc_info:
                event["exception"] = {"text": "".join(traceback.format_exception(*exc_info))}
        except Exception:
            pass

        event["message"] = message
        return event

    @classmethod
    @timed
    def format_event_as_text(cls, event: dict[str, Any]) -> str:
        """Best-effort text rendering for debug dumps and optional logs.txt archives."""
        created_raw = event.get("created")
        try:
            created = float(created_raw) if created_raw is not None else time.time()
        except Exception:
            created = time.time()
        try:
            base = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created))
            msecs = int((created - int(created)) * 1000)
            asctime = f"{base},{msecs:03d}"
        except Exception:
            asctime = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S,000")
        level = str(event.get("level") or "INFO")
        uid = str(event.get("user_id") or "-")
        message = event.get("message")
        try:
            message_str = str(message) if message is not None else ""
        except Exception:
            message_str = ""
        return f"{asctime} [{level}] [user={uid}] {message_str}"

    @classmethod
    @timed
    def get_logger(cls, name=__name__):
        """Create a logger wired to the current SessionLogger context.

        Args:
            name: Logger name; defaults to the current module name.

        Returns:
            logging.Logger: A configured logger that writes both to stdout and
            the in-memory `SessionLogger.logs` buffer. The buffer is keyed by
            the current `SessionLogger.request_id`.
        """
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.filters.clear()
        logger.setLevel(logging.DEBUG)
        root_logger = logging.getLogger()
        if not any(isinstance(handler, logging.NullHandler) for handler in root_logger.handlers):
            root_logger.addHandler(logging.NullHandler())
        logger.propagate = True

        # Single combined filter: attach session_id and respect per-session level.
        @timed
        def filter(record):
            """Attach session metadata and capture the per-request console log level."""
            try:
                sid = cls.session_id.get()
                rid = cls.request_id.get()
                uid = cls.user_id.get()
                record.session_id = sid
                record.request_id = rid
                record.user_id = uid or "-"
                record.session_log_level = cls.log_level.get()
                if rid:
                    with cls._state_lock:
                        cls._session_last_seen[rid] = time.time()
            except Exception:
                # Logging must never break request handling.
                pass
            return True

        logger.addFilter(filter)

        async_handler = logging.Handler()

        @timed
        def _emit(record: logging.LogRecord) -> None:
            cls._enqueue(record)

        async_handler.emit = _emit  # type: ignore[assignment]
        logger.addHandler(async_handler)

        return logger

    @classmethod
    @timed
    def set_log_queue(cls, queue: asyncio.Queue[logging.LogRecord] | None) -> None:
        cls.log_queue = queue
    @classmethod
    @timed
    def set_main_loop(cls, loop: asyncio.AbstractEventLoop | None) -> None:
        cls._main_loop = loop

    @classmethod
    @timed
    def set_max_lines(cls, value: int) -> None:
        """Set the maximum in-memory lines retained per request (best effort)."""
        try:
            value_int = int(value)
        except Exception:
            return
        value_int = max(100, min(200000, value_int))
        cls.max_lines = value_int

    @classmethod
    @timed
    def _enqueue(cls, record: logging.LogRecord) -> None:
        queue = cls.log_queue
        if queue is None:
            cls.process_record(record)
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop and running_loop is cls._main_loop:
            cls._safe_put(queue, record)
            return

        main_loop = cls._main_loop
        if main_loop and not main_loop.is_closed():
            main_loop.call_soon_threadsafe(cls._safe_put, queue, record)
        else:
            cls.process_record(record)

    @classmethod
    @timed
    def _safe_put(cls, queue: asyncio.Queue[logging.LogRecord], record: logging.LogRecord) -> None:
        try:
            queue.put_nowait(record)
        except asyncio.QueueFull:
            cls.process_record(record)

    @classmethod
    @timed
    def process_record(cls, record: logging.LogRecord) -> None:
        try:
            session_log_level = getattr(record, "session_log_level", logging.INFO)
            if record.levelno >= int(session_log_level):
                try:
                    console_line = cls._console_formatter.format(record)
                    sys.stdout.write(console_line + "\n")
                    sys.stdout.flush()
                except Exception:
                    pass
            request_id = getattr(record, "request_id", None)
            if request_id:
                try:
                    event = cls._build_event(record)
                except Exception:
                    event = {
                        "created": time.time(),
                        "level": str(getattr(record, "levelname", "INFO") or "INFO"),
                        "logger": str(getattr(record, "name", "") or ""),
                        "request_id": request_id,
                        "session_id": getattr(record, "session_id", None),
                        "user_id": getattr(record, "user_id", None),
                        "event_type": "pipe",
                        "module": str(getattr(record, "module", "") or ""),
                        "func": str(getattr(record, "funcName", "") or ""),
                        "lineno": int(getattr(record, "lineno", 0) or 0),
                        "message": str(getattr(record, "msg", "") or ""),
                    }
                with cls._state_lock:
                    buffer = cls.logs.get(request_id)
                    if buffer is None or buffer.maxlen != cls.max_lines:
                        buffer = deque(maxlen=cls.max_lines)
                        cls.logs[request_id] = buffer
                    try:
                        buffer.append(event)
                        cls._session_last_seen[request_id] = time.time()
                    except Exception:
                        pass
        except Exception:
            # Never raise from logging hooks.
            return

    @classmethod
    @timed
    def cleanup(cls, max_age_seconds: float = 3600) -> None:
        """Remove stale session logs to avoid unbounded growth."""
        cutoff = time.time() - max_age_seconds
        with cls._state_lock:
            stale = [sid for sid, ts in cls._session_last_seen.items() if ts < cutoff]
            for sid in stale:
                cls.logs.pop(sid, None)
                cls._session_last_seen.pop(sid, None)


# -----------------------------------------------------------------------------
# Session Log Archive Writer
# -----------------------------------------------------------------------------

@timed
def write_session_log_archive(job: _SessionLogArchiveJob) -> None:
    """Write a single encrypted zip archive containing session logs + metadata.

    Args:
        job: Archive job containing all configuration and log events

    The archive contains:
    - meta.json: Metadata including timestamps, IDs, and configuration
    - logs.txt: Text-formatted logs (if log_format is "text" or "both")
    - logs.jsonl: JSONL-formatted logs (if log_format is "jsonl" or "both")

    All files are encrypted using AES encryption with the provided password.
    Atomic file replacement is used to prevent partial writes.
    """
    if pyzipper is None:
        return
    base_dir = (job.base_dir or "").strip()
    if not base_dir:
        return

    user_id = _sanitize_path_component(job.user_id, fallback="user")
    chat_id = _sanitize_path_component(job.chat_id, fallback="chat")
    message_id = _sanitize_path_component(job.message_id, fallback="message")
    session_id = str(job.session_id or "")

    root = Path(base_dir).expanduser()
    out_dir = root / user_id / chat_id
    out_path = out_dir / f"{message_id}.zip"
    tmp_path = out_dir / f"{message_id}.zip.tmp"

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    compression_map = {
        "stored": pyzipper.ZIP_STORED,
        "deflated": pyzipper.ZIP_DEFLATED,
        "bzip2": pyzipper.ZIP_BZIP2,
        "lzma": pyzipper.ZIP_LZMA,
    }
    compression = compression_map.get((job.zip_compression or "lzma").lower(), pyzipper.ZIP_LZMA)

    meta = {
        "created_at": datetime.datetime.fromtimestamp(job.created_at, tz=datetime.timezone.utc).isoformat(),
        "ids": {
            "user_id": str(job.user_id or ""),
            "session_id": str(session_id),
            "chat_id": str(job.chat_id or ""),
            "message_id": str(job.message_id or ""),
        },
        "request_id": str(job.request_id or ""),
        "log_format": str(job.log_format or ""),
    }
    meta_json = json.dumps(meta, ensure_ascii=False, indent=2)

    log_format = (job.log_format or "jsonl").strip().lower()
    if log_format not in {"jsonl", "text", "both"}:
        log_format = "jsonl"
    write_text = log_format in {"text", "both"}
    write_jsonl = log_format in {"jsonl", "both"}

    @timed
    def _format_asctime_local(created: float) -> str:
        try:
            base = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created))
            msecs = int((created - int(created)) * 1000)
            return f"{base},{msecs:03d}"
        except Exception:
            return datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S,000")

    @timed
    def _format_event_as_text(event: dict[str, Any]) -> str:
        created = event.get("created")
        try:
            created_val = float(created) if created is not None else time.time()
        except Exception:
            created_val = time.time()
        level = str(event.get("level") or "INFO")
        uid = str(event.get("user_id") or job.user_id or "-")
        message = event.get("message")
        try:
            message_str = str(message) if message is not None else ""
        except Exception:
            message_str = ""
        return f"{_format_asctime_local(created_val)} [{level}] [user={uid}] {message_str}"

    @timed
    def _format_iso_utc(created: float) -> str:
        try:
            ts = datetime.datetime.fromtimestamp(created, tz=datetime.timezone.utc).isoformat(timespec="milliseconds")
            return ts.replace("+00:00", "Z")
        except Exception:
            ts = datetime.datetime.fromtimestamp(time.time(), tz=datetime.timezone.utc).isoformat(timespec="milliseconds")
            return ts.replace("+00:00", "Z")

    @timed
    def _coerce_event(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        try:
            msg = str(raw)
        except Exception:
            msg = ""
        return {
            "created": time.time(),
            "level": "INFO",
            "logger": "",
            "request_id": job.request_id,
            "session_id": job.session_id,
            "user_id": job.user_id,
            "event_type": "pipe",
            "module": "",
            "func": "",
            "lineno": 0,
            "message": msg,
        }

    @timed
    def _build_jsonl_record(event: dict[str, Any]) -> dict[str, Any]:
        created_raw = event.get("created")
        try:
            created_val = float(created_raw) if created_raw is not None else time.time()
        except Exception:
            created_val = time.time()

        exception_block = event.get("exception")
        exception_out = exception_block if isinstance(exception_block, dict) else None

        record_out: dict[str, Any] = {
            "ts": _format_iso_utc(created_val),
            "level": str(event.get("level") or "INFO"),
            "logger": str(event.get("logger") or ""),
            "request_id": str(job.request_id or ""),
            "user_id": str(job.user_id or ""),
            "session_id": str(job.session_id or ""),
            "chat_id": str(job.chat_id or ""),
            "message_id": str(job.message_id or ""),
            "event_type": str(event.get("event_type") or "pipe"),
            "module": str(event.get("module") or ""),
            "func": str(event.get("func") or ""),
            "lineno": int(event.get("lineno") or 0),
        }
        if exception_out:
            record_out["exception"] = exception_out

        message = event.get("message")
        try:
            record_out["message"] = str(message) if message is not None else ""
        except Exception:
            record_out["message"] = ""
        return record_out

    logs_payload = ""
    if write_text:
        try:
            text_lines = [_format_event_as_text(_coerce_event(evt)) for evt in (job.log_events or [])]
            logs_payload = "\n".join([line.rstrip("\n") for line in text_lines])
            if logs_payload and not logs_payload.endswith("\n"):
                logs_payload += "\n"
        except Exception:
            logs_payload = ""

    jsonl_payload = ""
    if write_jsonl:
        try:
            jsonl_lines: list[str] = []
            for raw_evt in (job.log_events or []):
                evt = _coerce_event(raw_evt)
                record_out = _build_jsonl_record(evt)
                try:
                    jsonl_lines.append(json.dumps(record_out, ensure_ascii=False, separators=(",", ":")))
                except Exception:
                    fallback = {"ts": record_out.get("ts"), "level": record_out.get("level"), "message": "<<failed to encode log record>>"}
                    jsonl_lines.append(json.dumps(fallback, ensure_ascii=False, separators=(",", ":")))
            jsonl_payload = "\n".join(jsonl_lines)
            if jsonl_payload and not jsonl_payload.endswith("\n"):
                jsonl_payload += "\n"
        except Exception:
            jsonl_payload = ""

    zip_kwargs: dict[str, Any] = {
        "mode": "w",
        "compression": compression,
        "encryption": pyzipper.WZ_AES,
    }
    if job.zip_compresslevel is not None and compression in {pyzipper.ZIP_DEFLATED, pyzipper.ZIP_BZIP2}:
        zip_kwargs["compresslevel"] = int(job.zip_compresslevel)

    # NOTE: Timing events are written directly to TIMING_LOG_FILE valve path,
    # not to session archives. See timing_logger.py for direct file output.

    try:
        with pyzipper.AESZipFile(tmp_path, **zip_kwargs) as zf:
            zf.setpassword(job.zip_password or b"")
            zf.writestr("meta.json", meta_json)
            if write_text:
                zf.writestr("logs.txt", logs_payload)
            if write_jsonl:
                zf.writestr("logs.jsonl", jsonl_payload)
    except Exception:
        with contextlib.suppress(Exception):
            tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        return

    try:
        os.replace(tmp_path, out_path)
    except Exception:
        with contextlib.suppress(Exception):
            tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
