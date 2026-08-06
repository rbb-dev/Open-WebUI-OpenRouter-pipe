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
from typing import Any, ClassVar

from .utils import _sanitize_path_component

try:
    import pyzipper  # type: ignore[import-untyped]
except ImportError:
    pyzipper = None  # type: ignore[assignment]

# Session Log Archive Job

@dataclass(slots=True)
class _SessionLogArchiveJob:
    """Represents a single session log archive request destined for the writer thread."""

    base_dir: str
    zip_password: bytes
    zip_compression: str
    zip_compresslevel: int | None
    user_id: str
    session_id: str
    chat_id: str
    message_id: str
    request_id: str
    created_at: float
    log_format: str
    log_events: list[dict[str, Any]]


def _safe_message(record: logging.LogRecord) -> str:
    """Message text for a record whose ``getMessage()`` failed.

    Never invokes user ``__str__``/``__repr__``: a hostile message object would
    raise again out of the fallback and drop the record from the buffer entirely.
    """
    msg = getattr(record, "msg", "")
    if isinstance(msg, str):
        return msg
    if msg is None:
        return ""
    return object.__repr__(msg)


# SessionLogger Class

def _render_exception_suffix(event: dict[str, Any]) -> str:
    """Render a captured traceback for the text outputs.

    Both text renderers emit only ``message``. Logging an error via
    ``logger.exception(...)`` / ``exc_info=`` moves the error detail out of the
    message and into ``event["exception"]``, so without this the human-readable
    logs.txt and the error-log citation dump show the headline and silently drop
    what actually went wrong. The jsonl output keeps it either way.
    """
    block = event.get("exception")
    if not isinstance(block, dict):
        return ""
    try:
        text = str(block.get("text") or "").rstrip()
    except Exception:  # noqa: BLE001 - render helper; a bad payload must not lose the line
        return "\n<<unrenderable exception>>"
    return f"\n{text}" if text else ""


def resolve_level(name: str | None, fallback: int) -> int:
    """A real logging level, or *fallback*.

    `getattr(logging, name, fallback)` returns ANY attribute of the module, so a typo
    that happens to match one -- GLOBAL_LOG_LEVEL=BASIC_FORMAT -- yields a format
    string where an int is expected. The default only covers absent, never wrong-kind.
    Bare `getattr(logging, name)` is worse again: it raises, and the sites that used it
    run per request.

    NOTSET is rejected rather than returned as 0. It is not a threshold -- as one it
    admits everything -- so accepting it made GLOBAL_LOG_LEVEL=NOTSET mean "emit
    everything" to the process floor while the valve fell back to INFO.

    The single resolver for a level name in this package: `core.config` derives the
    LOG_LEVEL valve default from this function, so the valve and the process floor
    cannot disagree about what a given GLOBAL_LOG_LEVEL means.
    """
    resolved = logging.getLevelName((name or "").strip().upper())
    if not isinstance(resolved, int) or resolved <= logging.NOTSET:
        return fallback
    return resolved




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
        log_level:  ContextVar holding this request's minimum level, or None when
                    no request is in scope. Resolve it with effective_log_level().
        logs:       Map of request_id -> fixed-size deque of structured log events (dicts).
    """

    session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
    request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
    user_id: ContextVar[str | None] = ContextVar("user_id", default=None)
    log_level: ContextVar[int | None] = ContextVar("log_level", default=None)
    process_log_level: int = resolve_level(os.getenv("GLOBAL_LOG_LEVEL"), logging.INFO)
    SESSION_LOG_MAX_LINES: int = 20000
    logs: ClassVar[dict[str, deque[dict[str, Any]]]] = {}
    _session_last_seen: ClassVar[dict[str, float]] = {}
    log_queue: asyncio.Queue[logging.LogRecord] | None = None
    _main_loop: asyncio.AbstractEventLoop | None = None
    _state_lock = threading.Lock()
    _console_formatter = logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _classify_event_type(message: str) -> str:
        msg = (message or "").lstrip()
        if msg.startswith("OpenRouter request headers:"):
            return "openrouter.request.headers"
        if msg.startswith("OpenRouter request payload:"):
            return "openrouter.request.payload"
        if msg.startswith("OpenRouter payload:"):
            return "openrouter.sse.event"
        if msg.startswith(("Tool ", "🔧", "Skipping ")):
            return "pipe.tools"
        return "pipe"

    @classmethod
    def _build_event(cls, record: logging.LogRecord) -> dict[str, Any]:
        """Return a structured session log event extracted from a LogRecord."""
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - capture path: self-log recurses; in-band fallback
            message = _safe_message(record)

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
        except Exception:  # noqa: BLE001 - capture path: self-log recurses; in-band sentinel
            event["exception"] = {"text": "<<failed to format exception>>"}

        event["message"] = message
        return event

    @classmethod
    def format_event_as_text(cls, event: dict[str, Any]) -> str:
        """Best-effort text rendering for debug dumps and optional logs.txt archives."""
        created_raw = event.get("created")
        try:
            created = float(created_raw) if created_raw is not None else time.time()
        except (TypeError, ValueError, OverflowError):
            created = time.time()
        try:
            base = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created))
            msecs = int((created - int(created)) * 1000)
            asctime = f"{base},{msecs:03d}"
        except (ValueError, OverflowError, OSError):
            asctime = datetime.datetime.fromtimestamp(time.time(), tz=datetime.UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S,000")
        level = str(event.get("level") or "INFO")
        uid = str(event.get("user_id") or "-")
        message = event.get("message")
        try:
            message_str = str(message) if message is not None else ""
        except Exception:  # noqa: BLE001 - str-coercion guard in render helper; sentinel fallback
            message_str = "<<unrenderable message>>"
        return (
            f"{asctime} [{level}] [user={uid}] {message_str}"
            + _render_exception_suffix(event)
        )

    class _HostForwardHandler(logging.Handler):
        """Re-emit to the host's handlers, honouring the session log level.

        Replaces propagation. ``logging.Logger.callHandlers`` walks ancestor
        *handlers* and consults their levels, never the ancestor loggers' -- so with
        this logger at DEBUG every record reached Open WebUI's level-less root
        StreamHandler regardless of any level anyone configured. Same walk here, with
        the pipe's own threshold applied first.
        """

        def __init__(self, owner: logging.Logger) -> None:
            super().__init__(level=logging.DEBUG)
            self._owner = owner

        def emit(self, record: logging.LogRecord) -> None:
            if not SessionLogger._passes_threshold(record):
                return
            node: logging.Logger | None = self._owner.parent
            while node is not None:
                for handler in list(node.handlers):
                    if record.levelno >= handler.level:
                        try:
                            handler.handle(record)
                        except Exception:  # noqa: BLE001, S112 - one bad host handler must not stop the others
                            continue
                if not node.propagate:
                    break
                node = node.parent

    @classmethod
    def effective_log_level(cls) -> int:
        """The minimum level in force: this request's, else the process-wide one.

        Records emitted outside a request -- startup, `pipes()`, plugin wiring, the
        Redis listener -- never had the ContextVar set, so reading it directly pinned
        them to its default regardless of how the operator had configured logging.
        """
        level = cls.log_level.get()
        return cls.process_log_level if level is None else int(level)

    @classmethod
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
        logger.propagate = False

        def filter(record):
            """Attach session metadata and capture the per-request console log level."""
            try:
                sid = cls.session_id.get()
                rid = cls.request_id.get()
                uid = cls.user_id.get()
                record.session_id = sid
                record.request_id = rid
                record.user_id = uid or "-"
                if rid:
                    with cls._state_lock:
                        cls._session_last_seen[rid] = time.time()
            except Exception:  # noqa: BLE001, S110 - logging Filter: self-log re-enters Logger.handle
                pass
            return True

        async_handler = logging.Handler()
        async_handler.addFilter(filter)

        def _emit(record: logging.LogRecord) -> None:
            cls._enqueue(record)

        async_handler.emit = _emit  # type: ignore[assignment]
        logger.addHandler(async_handler)
        logger.addHandler(cls._HostForwardHandler(logger))
        cls._ensure_console_owner(logger)


        return logger

    @classmethod
    def set_log_queue(cls, queue: asyncio.Queue[logging.LogRecord] | None) -> None:
        cls.log_queue = queue

    @classmethod
    def set_main_loop(cls, loop: asyncio.AbstractEventLoop | None) -> None:
        cls._main_loop = loop

    @classmethod
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
    def _safe_put(cls, queue: asyncio.Queue[logging.LogRecord], record: logging.LogRecord) -> None:
        try:
            queue.put_nowait(record)
        except asyncio.QueueFull:
            cls.process_record(record)

    @classmethod
    def _passes_threshold(cls, record: logging.LogRecord) -> bool:
        """The one decision about whether a record is shown, for every console sink.

        The package logger is pinned at DEBUG so the session archive captures
        everything, which means neither sink can rely on the logger's own level -- each
        has to ask. When only one of them did, `LOG_LEVEL` was inert on any host with no
        root handler, and every request payload went to stdout.

        Evaluated per record rather than set once as a handler level: the threshold moves
        under us twice over. `_refresh_process_log_level` rewrites the process floor when
        the valve changes, and each request sets its own via the `log_level` ContextVar --
        and `get_logger` runs from `Pipe.__init__`, before either has happened, so a
        handler level snapshotted there would freeze INFO for the worker's life.
        """
        try:
            threshold = int(cls.effective_log_level())
        except Exception:  # noqa: BLE001 - this runs inside Logger.handle, so anything
            threshold = logging.INFO
        return record.levelno >= threshold

    @classmethod
    def _ensure_console_owner(cls, logger: logging.Logger) -> None:
        """Exactly one thing prints a record, and it is the host's chain where there is one.

        This used to write its own formatted line to stdout AND forward the record to the
        host's handlers, so with Open WebUI's usual wiring every pipe record appeared
        twice, in two different formats. Worse under `LOG_FORMAT=json`: the private line
        is not JSON, so it corrupted the operator's structured log stream.

        Open WebUI installs a root handler only when `GLOBAL_LOG_LEVEL` names a real
        level (`env.py`: `if GLOBAL_LOG_LEVEL in logging.getLevelNamesMapping()`), so a
        host that does not set it has nowhere to print. A NullHandler is deliberately not
        counted as somewhere: `get_logger` adds one to the root itself, to keep the
        "no handlers could be found" warning quiet, and treating it as a sink would make
        the pipe silent on exactly the hosts that need it to print.
        """
        node: logging.Logger | None = logger.parent
        while node is not None:
            for handler in node.handlers:
                if not isinstance(handler, logging.NullHandler):
                    return
            node = node.parent if node.propagate else None

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(cls._console_formatter)
        console.addFilter(cls._passes_threshold)
        logger.addHandler(console)

    @classmethod
    def process_record(cls, record: logging.LogRecord) -> None:
        try:
            request_id = getattr(record, "request_id", None)
            if request_id:
                try:
                    event = cls._build_event(record)
                except Exception:  # noqa: BLE001 - capture path: self-log recurses; fallback event preserves record
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
                        "message": _safe_message(record),
                    }
                with cls._state_lock:
                    buffer = cls.logs.get(request_id)
                    if buffer is None:
                        buffer = deque(maxlen=cls.SESSION_LOG_MAX_LINES)
                        cls.logs[request_id] = buffer
                    elif buffer.maxlen != cls.SESSION_LOG_MAX_LINES:
                        buffer = deque(buffer, maxlen=cls.SESSION_LOG_MAX_LINES)
                        cls.logs[request_id] = buffer
                    buffer.append(event)
                    cls._session_last_seen[request_id] = time.time()
        except Exception:  # noqa: BLE001 - never raise from logging hooks
            # Never raise from logging hooks.
            return

    @classmethod
    def cleanup(cls, max_age_seconds: float = 3600) -> None:
        """Remove stale session logs to avoid unbounded growth."""
        cutoff = time.time() - max_age_seconds
        with cls._state_lock:
            stale = [sid for sid, ts in cls._session_last_seen.items() if ts < cutoff]
            for sid in stale:
                cls.logs.pop(sid, None)
                cls._session_last_seen.pop(sid, None)


# Session Log Archive Writer

def write_session_log_archive(job: _SessionLogArchiveJob) -> None:
    """Write a single encrypted zip archive containing session logs + metadata.

    Args:
        job: Archive job containing all configuration and log events

    The archive contains:
    - meta.json: Metadata including timestamps, IDs, and configuration
    - logs.txt: Text-formatted logs (if log_format is "text" or "both")
    - logs.jsonl: JSONL-formatted logs — ALWAYS written as the canonical,
      machine-readable record so re-assembly can merge/dedup prior events even
      when log_format is "text" (read_archive_events reads logs.jsonl only).

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
    except OSError as exc:
        sys.stderr.write(f"session log archive: mkdir {out_dir} failed: {exc}\n")
        return

    compression_map = {
        "stored": pyzipper.ZIP_STORED,
        "deflated": pyzipper.ZIP_DEFLATED,
        "bzip2": pyzipper.ZIP_BZIP2,
        "lzma": pyzipper.ZIP_LZMA,
    }
    compression = compression_map.get((job.zip_compression or "lzma").lower(), pyzipper.ZIP_LZMA)

    request_ids: list[str] = []
    try:
        ids: set[str] = set()
        for evt in (job.log_events or []):
            if not isinstance(evt, dict):
                continue
            rid = evt.get("request_id")
            if isinstance(rid, str) and rid.strip():
                ids.add(rid.strip())
        request_ids = sorted(ids)
    except TypeError:
        sys.stderr.write("session log archive: request-id enrichment skipped (log_events not iterable)\n")
        request_ids = []

    meta = {
        "created_at": datetime.datetime.fromtimestamp(job.created_at, tz=datetime.UTC).isoformat(),
        "ids": {
            "user_id": str(job.user_id or ""),
            "session_id": str(session_id),
            "chat_id": str(job.chat_id or ""),
            "message_id": str(job.message_id or ""),
        },
        "request_id": str(job.request_id or ""),
        **({"request_ids": request_ids} if request_ids else {}),
        "log_format": str(job.log_format or ""),
    }
    meta_json = json.dumps(meta, ensure_ascii=False, indent=2)

    log_format = (job.log_format or "jsonl").strip().lower()
    if log_format not in {"jsonl", "text", "both"}:
        log_format = "jsonl"
    write_text = log_format in {"text", "both"}
    write_jsonl = True

    def _format_asctime_local(created: float) -> str:
        try:
            base = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created))
            msecs = int((created - int(created)) * 1000)
            return f"{base},{msecs:03d}"
        except (ValueError, OverflowError, OSError):
            return datetime.datetime.fromtimestamp(time.time(), tz=datetime.UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S,000")

    def _format_event_as_text(event: dict[str, Any]) -> str:
        created = event.get("created")
        try:
            created_val = float(created) if created is not None else time.time()
        except (TypeError, ValueError, OverflowError):
            created_val = time.time()
        level = str(event.get("level") or "INFO")
        uid = str(event.get("user_id") or job.user_id or "-")
        message = event.get("message")
        try:
            message_str = str(message) if message is not None else ""
        except Exception:  # noqa: BLE001 - per-event str-coercion guard; sentinel fallback
            message_str = "<<unrenderable message>>"
        return (
            f"{_format_asctime_local(created_val)} [{level}] [user={uid}] {message_str}"
            + _render_exception_suffix(event)
        )

    def _format_iso_utc(created: float) -> str:
        try:
            ts = datetime.datetime.fromtimestamp(created, tz=datetime.UTC).isoformat(timespec="milliseconds")
            return ts.replace("+00:00", "Z")
        except (ValueError, OverflowError, OSError):
            ts = datetime.datetime.fromtimestamp(time.time(), tz=datetime.UTC).isoformat(timespec="milliseconds")
            return ts.replace("+00:00", "Z")

    def _coerce_event(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        try:
            msg = str(raw)
        except Exception:  # noqa: BLE001 - per-event str-coercion guard; sentinel fallback
            msg = "<<unrenderable event>>"
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

    def _build_jsonl_record(event: dict[str, Any]) -> dict[str, Any]:
        created_raw = event.get("created")
        try:
            created_val = float(created_raw) if created_raw is not None else time.time()
        except (TypeError, ValueError, OverflowError):
            created_val = time.time()

        exception_block = event.get("exception")
        exception_out = exception_block if isinstance(exception_block, dict) else None

        event_request_id = event.get("request_id")
        request_id_out = str(event_request_id) if event_request_id else str(job.request_id or "")
        record_out: dict[str, Any] = {
            "ts": _format_iso_utc(created_val),
            "level": str(event.get("level") or "INFO"),
            "logger": str(event.get("logger") or ""),
            "request_id": request_id_out,
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
        except Exception:  # noqa: BLE001 - per-event str-coercion guard; sentinel fallback
            record_out["message"] = "<<unrenderable message>>"
        return record_out

    logs_payload = ""
    if write_text:
        try:
            text_lines = [_format_event_as_text(_coerce_event(evt)) for evt in (job.log_events or [])]
            logs_payload = "\n".join([line.rstrip("\n") for line in text_lines])
            if logs_payload and not logs_payload.endswith("\n"):
                logs_payload += "\n"
        except Exception as exc:  # noqa: BLE001 - logs.txt build net; surfaced via stderr; logging avoided in archive writer
            sys.stderr.write(f"session log archive: logs.txt build failed: {exc}\n")
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
                except (TypeError, ValueError, RecursionError):
                    fallback = {"ts": record_out.get("ts"), "level": record_out.get("level"), "message": "<<failed to encode log record>>"}
                    jsonl_lines.append(json.dumps(fallback, ensure_ascii=False, separators=(",", ":")))
            jsonl_payload = "\n".join(jsonl_lines)
            if jsonl_payload and not jsonl_payload.endswith("\n"):
                jsonl_payload += "\n"
        except Exception as exc:  # noqa: BLE001 - logs.jsonl build net; surfaced via stderr; logging avoided in archive writer
            sys.stderr.write(f"session log archive: logs.jsonl build failed: {exc}\n")
            jsonl_payload = ""

    zip_kwargs: dict[str, Any] = {
        "mode": "w",
        "compression": compression,
        "encryption": pyzipper.WZ_AES,
    }
    if job.zip_compresslevel is not None and compression in {pyzipper.ZIP_DEFLATED, pyzipper.ZIP_BZIP2}:
        zip_kwargs["compresslevel"] = int(job.zip_compresslevel)


    try:
        with pyzipper.AESZipFile(tmp_path, **zip_kwargs) as zf:
            zf.setpassword(job.zip_password or b"")
            zf.writestr("meta.json", meta_json)
            if write_text:
                zf.writestr("logs.txt", logs_payload)
            if write_jsonl:
                zf.writestr("logs.jsonl", jsonl_payload)
    except Exception as exc:  # noqa: BLE001 - zip write net; mixed pyzipper error types; surfaced via stderr
        sys.stderr.write(f"session log archive: zip write {tmp_path} failed: {exc}\n")
        with contextlib.suppress(Exception):
            tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        return

    try:
        os.replace(tmp_path, out_path)
    except OSError as exc:
        sys.stderr.write(f"session log archive: publish {out_path} failed: {exc}\n")
        with contextlib.suppress(Exception):
            tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
