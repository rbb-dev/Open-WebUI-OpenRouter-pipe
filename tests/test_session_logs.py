from __future__ import annotations

import json
import os

import pytest

pyzipper = pytest.importorskip("pyzipper")

import open_webui_openrouter_pipe.pipe as pipe_module
from open_webui_openrouter_pipe import (
    EncryptedStr,
    Pipe,
    SessionLogger,
    _SessionLogArchiveJob,
    _sanitize_path_component,
)


def test_sanitize_path_component_blocks_traversal() -> None:
    assert _sanitize_path_component("../etc/passwd", fallback="x") != "../etc/passwd"
    assert "/" not in _sanitize_path_component("a/b", fallback="x")
    assert _sanitize_path_component("", fallback="x") == "x"


def test_write_session_log_archive_creates_encrypted_zip(tmp_path, pipe_instance) -> None:
    pipe = pipe_instance
    password = b"correct horse battery staple"
    job = _SessionLogArchiveJob(
        base_dir=str(tmp_path),
        zip_password=password,
        zip_compression="lzma",
        zip_compresslevel=None,
        user_id="714affbb-d092-4e39-af42-0c59ee82ab8d",
        session_id="Ix-n1ptqDgmpL-8gAAAp",
        chat_id="8ec723e5-6553-41bd-a0c6-2ab885c1bcbc",
        message_id="2e5b92d9-04b6-42ae-a1c7-602a3efdb1d2",
        request_id="deadbeef",
        created_at=1_700_000_000.0,
        log_format="jsonl",
        log_events=[
            {
                "created": 1_700_000_000.0,
                "level": "INFO",
                "logger": "session-log-test",
                "request_id": "deadbeef",
                "session_id": "Ix-n1ptqDgmpL-8gAAAp",
                "user_id": "714affbb-d092-4e39-af42-0c59ee82ab8d",
                "event_type": "pipe",
                "module": "test_session_log_storage",
                "func": "test_write_session_log_archive_creates_encrypted_zip",
                "lineno": 1,
                "message": "hello",
            },
            {
                "created": 1_700_000_000.1,
                "level": "INFO",
                "logger": "session-log-test",
                "request_id": "deadbeef",
                "session_id": "Ix-n1ptqDgmpL-8gAAAp",
                "user_id": "714affbb-d092-4e39-af42-0c59ee82ab8d",
                "event_type": "pipe",
                "module": "test_session_log_storage",
                "func": "test_write_session_log_archive_creates_encrypted_zip",
                "lineno": 1,
                "message": "world",
            },
        ],
    )

    pipe._write_session_log_archive(job)

    out_path = (
        tmp_path
        / job.user_id
        / job.chat_id
        / f"{job.message_id}.zip"
    )
    assert out_path.exists()

    with pyzipper.AESZipFile(out_path) as zf:
        zf.setpassword(password)
        names = set(zf.namelist())
        assert {"meta.json", "logs.jsonl"} <= names
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        assert meta["ids"]["user_id"] == job.user_id
        assert meta["ids"]["session_id"] == job.session_id
        assert meta["ids"]["chat_id"] == job.chat_id
        assert meta["ids"]["message_id"] == job.message_id
        assert meta["request_id"] == job.request_id
        assert meta["request_ids"] == [job.request_id]
        logs = zf.read("logs.jsonl").decode("utf-8").splitlines()
        records = [json.loads(line) for line in logs if line.strip()]
        assert any(rec.get("message") == "hello" for rec in records)
        assert any(rec.get("message") == "world" for rec in records)


def test_write_session_log_archive_preserves_per_event_request_id(tmp_path, pipe_instance) -> None:
    pipe = pipe_instance
    password = b"correct horse battery staple"
    job = _SessionLogArchiveJob(
        base_dir=str(tmp_path),
        zip_password=password,
        zip_compression="lzma",
        zip_compresslevel=None,
        user_id="user",
        session_id="sid",
        chat_id="chat",
        message_id="message",
        request_id="bundle",
        created_at=1_700_000_000.0,
        log_format="jsonl",
        log_events=[
            {"created": 1_700_000_000.0, "level": "INFO", "logger": "t", "request_id": "r1", "message": "one"},
            {"created": 1_700_000_000.1, "level": "INFO", "logger": "t", "request_id": "r2", "message": "two"},
        ],
    )

    pipe._write_session_log_archive(job)

    out_path = tmp_path / job.user_id / job.chat_id / f"{job.message_id}.zip"
    with pyzipper.AESZipFile(out_path) as zf:
        zf.setpassword(password)
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        assert meta["request_id"] == "bundle"
        assert meta["request_ids"] == ["r1", "r2"]
        records = [json.loads(line) for line in zf.read("logs.jsonl").decode("utf-8").splitlines() if line.strip()]
        assert [rec.get("request_id") for rec in records] == ["r1", "r2"]


def test_enqueue_session_log_archive_skips_when_missing_ids(tmp_path, monkeypatch, pipe_instance) -> None:
    pipe = pipe_instance

    valves = pipe.Valves(
        SESSION_LOG_STORE_ENABLED=True,
        SESSION_LOG_DIR=str(tmp_path),
        SESSION_LOG_ZIP_PASSWORD=EncryptedStr("secret"),
    )

    called = {"started": False}

    def _noop_start() -> None:
        called["started"] = True

    monkeypatch.setattr(pipe, "_maybe_start_session_log_workers", _noop_start)
    monkeypatch.setattr(pipe, "_session_log_queue", None)

    pipe._enqueue_session_log_archive(
        valves,
        user_id="",
        session_id="Ix-n1ptqDgmpL-8gAAAp",
        chat_id="c",
        message_id="m",
        request_id="r",
        log_events=[{"created": 1_700_000_000.0, "level": "INFO", "logger": "t", "message": "x"}],
    )

    assert called["started"] is False


def test_session_log_buffer_captures_debug_even_when_console_level_warning(capsys) -> None:
    request_id = "req-test-debug-capture"
    tokens = [
        (SessionLogger.request_id, SessionLogger.request_id.set(request_id)),
        (SessionLogger.session_id, SessionLogger.session_id.set("sess-1")),
        (SessionLogger.user_id, SessionLogger.user_id.set("user-1")),
        (SessionLogger.log_level, SessionLogger.log_level.set(30)),  # WARNING
    ]
    try:
        logger = SessionLogger.get_logger("session-log-test")
        logger.debug("DEBUG_ONLY_TOKEN")
        logger.warning("WARNING_TOKEN")
    finally:
        for var, token in reversed(tokens):
            try:
                var.reset(token)
            except Exception:
                pass

    stdout = capsys.readouterr().out
    assert "WARNING_TOKEN" in stdout
    assert "DEBUG_ONLY_TOKEN" not in stdout

    lines = list(SessionLogger.logs.get(request_id, []))
    assert any("DEBUG_ONLY_TOKEN" in (line.get("message") or "") for line in lines if isinstance(line, dict))
    assert any("WARNING_TOKEN" in (line.get("message") or "") for line in lines if isinstance(line, dict))
    SessionLogger.logs.pop(request_id, None)


class TestSessionLogArchiveEdgeCases:
    """Additional edge case tests for session log archiving."""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

    def test_archive_with_missing_ids_skipped(self, tmp_path, pipe_instance) -> None:
        """Archive not written when user_id/session_id/chat_id/message_id missing."""
        pipe = pipe_instance
        password = b"correct horse battery staple"

        # Test with missing user_id
        job_missing_user = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=password,
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="",  # Missing
            session_id="Ix-n1ptqDgmpL-8gAAAp",
            chat_id="8ec723e5-6553-41bd-a0c6-2ab885c1bcbc",
            message_id="2e5b92d9-04b6-42ae-a1c7-602a3efdb1d2",
            request_id="deadbeef",
            created_at=1_700_000_000.0,
            log_format="jsonl",
            log_events=[{"created": 1_700_000_000.0, "level": "INFO", "logger": "t", "message": "hello"}],
        )

        # Should not create archive when user_id is missing
        pipe._write_session_log_archive(job_missing_user)

        # Verify no archive was created
        out_path = tmp_path / job_missing_user.user_id / job_missing_user.chat_id / f"{job_missing_user.message_id}.zip"
        assert not out_path.exists()

    def test_archive_password_encryption(self, tmp_path, pipe_instance) -> None:
        """Zip archive encrypted with AES and correct password."""
        pipe = pipe_instance
        password = b"correct horse battery staple"

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=password,
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="714affbb-d092-4e39-af42-0c59ee82ab8d",
            session_id="Ix-n1ptqDgmpL-8gAAAp",
            chat_id="8ec723e5-6553-41bd-a0c6-2ab885c1bcbc",
            message_id="2e5b92d9-04b6-42ae-a1c7-602a3efdb1d2",
            request_id="deadbeef",
            created_at=1_700_000_000.0,
            log_format="jsonl",
            log_events=[{"created": 1_700_000_000.0, "level": "INFO", "logger": "t", "message": "hello"}],
        )

        pipe._write_session_log_archive(job)

        out_path = (
            tmp_path
            / job.user_id
            / job.chat_id
            / f"{job.message_id}.zip"
        )
        assert out_path.exists()

        # Verify archive is encrypted by trying to open with wrong password
        with pyzipper.AESZipFile(out_path) as zf:
            # Should fail with wrong password
            zf.setpassword(b"wrong_password")
            with pytest.raises(RuntimeError):
                zf.read("meta.json")

        # Should succeed with correct password
        with pyzipper.AESZipFile(out_path) as zf:
            zf.setpassword(password)
            meta = json.loads(zf.read("meta.json").decode("utf-8"))
            assert meta["ids"]["user_id"] == job.user_id

    def test_archive_compression_options(self, tmp_path, pipe_instance) -> None:
        """Different compression algorithms (stored, deflated, bzip2, lzma)."""
        pipe = pipe_instance
        password = b"correct horse battery staple"

        compression_methods = ["stored", "deflated", "bzip2", "lzma"]

        for compression in compression_methods:
            job = _SessionLogArchiveJob(
                base_dir=str(tmp_path),
                zip_password=password,
                zip_compression=compression,
                zip_compresslevel=None,
                user_id=f"user_{compression}",
                session_id="Ix-n1ptqDgmpL-8gAAAp",
                chat_id="8ec723e5-6553-41bd-a0c6-2ab885c1bcbc",
                message_id=f"msg_{compression}",
                request_id="deadbeef",
                created_at=1_700_000_000.0,
                log_format="jsonl",
                log_events=[{"created": 1_700_000_000.0, "level": "INFO", "logger": "t", "message": "hello"}],
            )

            pipe._write_session_log_archive(job)

            out_path = (
                tmp_path
                / job.user_id
                / job.chat_id
                / f"{job.message_id}.zip"
            )
            assert out_path.exists()

            # Verify archive can be opened
            with pyzipper.AESZipFile(out_path) as zf:
                zf.setpassword(password)
                meta = json.loads(zf.read("meta.json").decode("utf-8"))
                assert meta["ids"]["user_id"] == job.user_id

    def test_cleanup_deletes_old_archives(self, tmp_path, monkeypatch, pipe_instance) -> None:
        """Archives older than retention window deleted."""
        pipe = pipe_instance

        # Set retention to 1 day for testing
        valves = pipe.Valves(SESSION_LOG_RETENTION_DAYS=1)
        fixed_now = 1_700_000_000.0
        monkeypatch.setattr(pipe_module.time, "time", lambda: fixed_now)

        # Create old archive (2 days ago)
        old_timestamp = fixed_now - (2 * 24 * 60 * 60)
        old_job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"secret",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user_old",
            session_id="Ix-n1ptqDgmpL-8gAAAp",
            chat_id="8ec723e5-6553-41bd-a0c6-2ab885c1bcbc",
            message_id="msg_old",
            request_id="deadbeef",
            created_at=old_timestamp,
            log_format="jsonl",
            log_events=[{"created": old_timestamp, "level": "INFO", "logger": "t", "message": "old"}],
        )

        pipe._write_session_log_archive(old_job)

        # Create new archive (today)
        new_timestamp = fixed_now
        new_job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"secret",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user_new",
            session_id="Ix-n1ptqDgmpL-8gAAAp",
            chat_id="8ec723e5-6553-41bd-a0c6-2ab885c1bcbc",
            message_id="msg_new",
            request_id="deadbeef",
            created_at=new_timestamp,
            log_format="jsonl",
            log_events=[{"created": new_timestamp, "level": "INFO", "logger": "t", "message": "new"}],
        )

        pipe._write_session_log_archive(new_job)

        # Verify both archives exist
        old_path = tmp_path / "user_old" / old_job.chat_id / f"{old_job.message_id}.zip"
        new_path = tmp_path / "user_new" / new_job.chat_id / f"{new_job.message_id}.zip"
        assert old_path.exists()
        assert new_path.exists()

        # Set mtimes to match the job created_at so cleanup uses deterministic cutoff comparisons.
        os.utime(old_path, (old_timestamp, old_timestamp))
        os.utime(new_path, (new_timestamp, new_timestamp))

        pipe._session_log_dirs = {str(tmp_path)}  # type: ignore[attr-defined]
        pipe._session_log_retention_days = valves.SESSION_LOG_RETENTION_DAYS  # type: ignore[attr-defined]
        pipe._cleanup_session_log_archives()

        assert not old_path.exists()
        assert new_path.exists()
        # Old directories should be pruned if emptied.
        assert not (tmp_path / "user_old").exists()

    def test_cleanup_prunes_empty_directories(self, tmp_path, monkeypatch, pipe_instance) -> None:
        """Empty user/chat directories removed after archive deletion."""
        pipe = pipe_instance

        # Create archive
        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"secret",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user_test",
            session_id="Ix-n1ptqDgmpL-8gAAAp",
            chat_id="8ec723e5-6553-41bd-a0c6-2ab885c1bcbc",
            message_id="msg_test",
            request_id="deadbeef",
            created_at=1_700_000_000.0,
            log_format="jsonl",
            log_events=[{"created": 1_700_000_000.0, "level": "INFO", "logger": "t", "message": "test"}],
        )

        pipe._write_session_log_archive(job)

        # Verify directory structure exists
        user_dir = tmp_path / job.user_id
        chat_dir = user_dir / job.chat_id
        archive_path = chat_dir / f"{job.message_id}.zip"

        assert user_dir.exists()
        assert chat_dir.exists()
        assert archive_path.exists()

        # Age the archive and run the real cleanup, which should delete and prune empties.
        fixed_now = 1_700_000_000.0
        old_timestamp = fixed_now - 10
        os.utime(archive_path, (old_timestamp, old_timestamp))
        pipe._session_log_dirs = {str(tmp_path)}  # type: ignore[attr-defined]
        pipe._session_log_retention_days = 0  # type: ignore[attr-defined]
        monkeypatch.setattr(pipe_module.time, "time", lambda: fixed_now)
        pipe._cleanup_session_log_archives()

        assert not archive_path.exists()
        assert not chat_dir.exists()
        assert not user_dir.exists()


# ===== From test_session_log_merge.py =====

"""Tests for session log archive merging functionality.

These tests verify the merge helpers that prevent data loss when multiple
pipe invocations share the same message_id (main response, title generation,
tool calls, etc.).
"""


import io
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from open_webui_openrouter_pipe import Pipe


class TestConvertJsonlToInternal:
    """Tests for _convert_jsonl_to_internal helper."""

    def test_converts_iso_timestamp_to_epoch(self, pipe_instance) -> None:
        """ISO 'ts' field is converted to 'created' epoch float."""
        pipe = pipe_instance
        evt = {
            "ts": "2025-01-20T10:30:45.123000+00:00",
            "level": "INFO",
            "message": "test",
        }
        result = pipe._convert_jsonl_to_internal(evt)

        assert "created" in result
        assert "ts" not in result  # ts is removed after conversion
        assert isinstance(result["created"], float)
        # Verify approximate timestamp (2025-01-20 10:30:45 UTC)
        assert 1737369045 <= result["created"] <= 1737369046

    def test_converts_z_suffix_timestamp(self, pipe_instance) -> None:
        """'Z' suffix (Zulu time) is properly handled."""
        pipe = pipe_instance
        evt = {"ts": "2025-01-20T10:30:45.123Z", "message": "test"}
        result = pipe._convert_jsonl_to_internal(evt)

        assert "created" in result
        assert isinstance(result["created"], float)

    def test_preserves_existing_created_field(self, pipe_instance) -> None:
        """If 'created' already exists, 'ts' is ignored."""
        pipe = pipe_instance
        evt = {
            "ts": "2025-01-20T10:30:45.123Z",
            "created": 1234567890.0,
            "message": "test",
        }
        result = pipe._convert_jsonl_to_internal(evt)

        # created should be unchanged, ts should remain
        assert result["created"] == 1234567890.0
        assert "ts" in result  # ts is NOT removed when created exists

    def test_handles_missing_ts_field(self, pipe_instance) -> None:
        """Events without 'ts' field pass through unchanged."""
        pipe = pipe_instance
        evt = {"created": 1234567890.0, "message": "test"}
        result = pipe._convert_jsonl_to_internal(evt)

        assert result == evt

    def test_handles_invalid_timestamp_gracefully(self, pipe_instance) -> None:
        """Invalid timestamp falls back to current time."""
        pipe = pipe_instance
        before = time.time()
        evt = {"ts": "not-a-valid-timestamp", "message": "test"}
        result = pipe._convert_jsonl_to_internal(evt)
        after = time.time()

        assert "created" in result
        assert before <= result["created"] <= after

    def test_does_not_mutate_original_event(self, pipe_instance) -> None:
        """Original event dict is not modified."""
        pipe = pipe_instance
        original = {"ts": "2025-01-20T10:30:45.123Z", "message": "test"}
        original_copy = dict(original)
        pipe._convert_jsonl_to_internal(original)

        assert original == original_copy


class TestDedupeSessionLogEvents:
    """Tests for _dedupe_session_log_events helper."""

    def test_removes_exact_duplicates(self, pipe_instance) -> None:
        """Identical events are deduplicated."""
        pipe = pipe_instance
        events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "hello"},
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "hello"},
        ]
        result = pipe._dedupe_session_log_events(events)

        assert len(result) == 1
        assert result[0]["message"] == "hello"

    def test_keeps_events_with_different_timestamps(self, pipe_instance) -> None:
        """Events with different timestamps are kept."""
        pipe = pipe_instance
        events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "hello"},
            {"created": 1001.0, "request_id": "req1", "lineno": 42, "message": "hello"},
        ]
        result = pipe._dedupe_session_log_events(events)

        assert len(result) == 2

    def test_keeps_events_with_different_request_ids(self, pipe_instance) -> None:
        """Events from different request_ids are kept."""
        pipe = pipe_instance
        events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "hello"},
            {"created": 1000.0, "request_id": "req2", "lineno": 42, "message": "hello"},
        ]
        result = pipe._dedupe_session_log_events(events)

        assert len(result) == 2

    def test_keeps_events_with_different_line_numbers(self, pipe_instance) -> None:
        """Events from different line numbers are kept."""
        pipe = pipe_instance
        events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "hello"},
            {"created": 1000.0, "request_id": "req1", "lineno": 43, "message": "hello"},
        ]
        result = pipe._dedupe_session_log_events(events)

        assert len(result) == 2

    def test_keeps_events_with_different_messages(self, pipe_instance) -> None:
        """Events with different messages are kept."""
        pipe = pipe_instance
        events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "hello"},
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "world"},
        ]
        result = pipe._dedupe_session_log_events(events)

        assert len(result) == 2

    def test_preserves_order_of_first_occurrence(self, pipe_instance) -> None:
        """First occurrence of duplicate is kept."""
        pipe = pipe_instance
        events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "first", "extra": "A"},
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "first", "extra": "B"},
        ]
        result = pipe._dedupe_session_log_events(events)

        assert len(result) == 1
        assert result[0]["extra"] == "A"  # First occurrence kept

    def test_handles_missing_fields_gracefully(self, pipe_instance) -> None:
        """Events with missing fields are handled."""
        pipe = pipe_instance
        events = [
            {"message": "no created or request_id"},
            {"created": 1000.0, "message": "no request_id"},
            {"request_id": "req1", "message": "no created"},
        ]
        result = pipe._dedupe_session_log_events(events)

        # All are unique because of different content
        assert len(result) == 3

    def test_handles_empty_list(self, pipe_instance) -> None:
        """Empty event list returns empty list."""
        pipe = pipe_instance
        result = pipe._dedupe_session_log_events([])
        assert result == []


class TestReadSessionLogArchiveEvents:
    """Tests for _read_session_log_archive_events helper."""

    def test_reads_events_from_encrypted_zip(self, pipe_instance) -> None:
        """Events are read from an AES-encrypted zip archive."""
        import pyzipper

        pipe = pipe_instance

        # Create a test archive
        events = [
            {"ts": "2025-01-20T10:30:45.123Z", "level": "INFO", "message": "event 1"},
            {"ts": "2025-01-20T10:30:46.456Z", "level": "DEBUG", "message": "event 2"},
        ]
        jsonl_content = "\n".join(json.dumps(e) for e in events)

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = Path(f.name)

        try:
            password = b"test-password-123"
            with pyzipper.AESZipFile(
                zip_path, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES
            ) as zf:
                zf.setpassword(password)
                zf.writestr("logs.jsonl", jsonl_content.encode("utf-8"))

            settings = ("/tmp", password, "deflated", 6)
            result = pipe._read_session_log_archive_events(zip_path, settings)

            assert len(result) == 2
            # Verify conversion happened
            assert "created" in result[0]
            assert "created" in result[1]
            assert result[0]["message"] == "event 1"
            assert result[1]["message"] == "event 2"
        finally:
            zip_path.unlink(missing_ok=True)

    def test_returns_empty_list_for_missing_jsonl(self, pipe_instance) -> None:
        """Archive without logs.jsonl returns empty list."""
        import pyzipper

        pipe = pipe_instance

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = Path(f.name)

        try:
            password = b"test-password"
            with pyzipper.AESZipFile(
                zip_path, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES
            ) as zf:
                zf.setpassword(password)
                zf.writestr("other.txt", b"not jsonl")

            settings = ("/tmp", password, "deflated", 6)
            result = pipe._read_session_log_archive_events(zip_path, settings)

            assert result == []
        finally:
            zip_path.unlink(missing_ok=True)

    def test_skips_malformed_json_lines(self, pipe_instance) -> None:
        """Malformed JSON lines are skipped, valid ones are returned."""
        import pyzipper

        pipe = pipe_instance

        jsonl_content = (
            '{"ts": "2025-01-20T10:30:45.123Z", "message": "valid"}\n'
            "not valid json\n"
            '{"ts": "2025-01-20T10:30:46.456Z", "message": "also valid"}\n'
        )

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = Path(f.name)

        try:
            password = b"test-password"
            with pyzipper.AESZipFile(
                zip_path, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES
            ) as zf:
                zf.setpassword(password)
                zf.writestr("logs.jsonl", jsonl_content.encode("utf-8"))

            settings = ("/tmp", password, "deflated", 6)
            result = pipe._read_session_log_archive_events(zip_path, settings)

            assert len(result) == 2
            assert result[0]["message"] == "valid"
            assert result[1]["message"] == "also valid"
        finally:
            zip_path.unlink(missing_ok=True)

    def test_handles_empty_lines(self, pipe_instance) -> None:
        """Empty lines in JSONL are skipped."""
        import pyzipper

        pipe = pipe_instance

        jsonl_content = (
            '{"ts": "2025-01-20T10:30:45.123Z", "message": "one"}\n'
            "\n"
            "   \n"
            '{"ts": "2025-01-20T10:30:46.456Z", "message": "two"}\n'
        )

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = Path(f.name)

        try:
            password = b"test-password"
            with pyzipper.AESZipFile(
                zip_path, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES
            ) as zf:
                zf.setpassword(password)
                zf.writestr("logs.jsonl", jsonl_content.encode("utf-8"))

            settings = ("/tmp", password, "deflated", 6)
            result = pipe._read_session_log_archive_events(zip_path, settings)

            assert len(result) == 2
        finally:
            zip_path.unlink(missing_ok=True)


class TestMergeIntegration:
    """Integration tests for the full merge flow."""

    def test_merge_combines_existing_and_new_events(self, pipe_instance) -> None:
        """When archive exists, its events are merged with new DB events."""
        # This is a higher-level test that verifies the merge logic works together
        pipe = pipe_instance

        # Create some "existing" events (as if from archive)
        existing = [
            {"created": 1000.0, "request_id": "req1", "lineno": 10, "message": "existing 1"},
            {"created": 1001.0, "request_id": "req1", "lineno": 20, "message": "existing 2"},
        ]

        # Create some "new" events (as if from DB)
        new_events = [
            {"created": 1002.0, "request_id": "req2", "lineno": 30, "message": "new 1"},
            {"created": 1003.0, "request_id": "req2", "lineno": 40, "message": "new 2"},
        ]

        # Merge them
        merged = existing + new_events
        merged = pipe._dedupe_session_log_events(merged)
        merged.sort(key=lambda e: e.get("created", 0))

        assert len(merged) == 4
        assert merged[0]["message"] == "existing 1"
        assert merged[1]["message"] == "existing 2"
        assert merged[2]["message"] == "new 1"
        assert merged[3]["message"] == "new 2"

    def test_merge_dedupes_overlapping_events(self, pipe_instance) -> None:
        """Duplicate events between archive and DB are deduplicated."""
        pipe = pipe_instance

        # Create overlapping events
        existing = [
            {"created": 1000.0, "request_id": "req1", "lineno": 10, "message": "same event"},
        ]
        new_events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 10, "message": "same event"},
            {"created": 1001.0, "request_id": "req1", "lineno": 20, "message": "unique"},
        ]

        merged = existing + new_events
        merged = pipe._dedupe_session_log_events(merged)
        merged.sort(key=lambda e: e.get("created", 0))

        assert len(merged) == 2
        assert merged[0]["message"] == "same event"
        assert merged[1]["message"] == "unique"

    def test_sort_is_stable_for_equal_timestamps(self, pipe_instance) -> None:
        """Events with equal timestamps maintain relative order (stable sort)."""
        pipe = pipe_instance

        events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 1, "message": "A"},
            {"created": 1000.0, "request_id": "req1", "lineno": 2, "message": "B"},
            {"created": 1000.0, "request_id": "req1", "lineno": 3, "message": "C"},
        ]

        # Sort should preserve insertion order for equal keys
        events.sort(key=lambda e: e.get("created", 0))

        assert events[0]["message"] == "A"
        assert events[1]["message"] == "B"
        assert events[2]["message"] == "C"


# ===== From test_session_logger_thread_safety.py =====

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from open_webui_openrouter_pipe import SessionLogger


def test_session_logger_thread_safety_filter_and_cleanup() -> None:
    """Test that SessionLogger filter and cleanup don't race with RuntimeError."""
    SessionLogger.logs.clear()
    SessionLogger._session_last_seen.clear()

    request_id = "test-request-001"
    num_iterations = 100
    runtime_errors: list[RuntimeError] = []

    def cleanup_worker() -> None:
        """Repeatedly call cleanup in a background thread."""
        for _ in range(num_iterations):
            try:
                SessionLogger.cleanup(max_age_seconds=0)
            except RuntimeError as e:
                if "dictionary changed size during iteration" in str(e):
                    runtime_errors.append(e)
                raise

    def logging_worker() -> None:
        """Repeatedly log with the same request_id in a background thread."""
        logger = SessionLogger.get_logger(__name__)
        for _ in range(num_iterations):
            SessionLogger.request_id.set(request_id)
            SessionLogger.session_id.set("test-session")
            SessionLogger.user_id.set("test-user")
            logger.info(f"Test log message {time.time()}")
            time.sleep(0.001)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        futures.append(executor.submit(cleanup_worker))
        futures.append(executor.submit(logging_worker))
        futures.append(executor.submit(logging_worker))

        for f in futures:
            f.result()

    assert len(runtime_errors) == 0, (
        f"Got {len(runtime_errors)} RuntimeError(s): {runtime_errors}"
    )

    assert request_id in SessionLogger.logs
    assert request_id in SessionLogger._session_last_seen
    assert isinstance(SessionLogger.logs[request_id], deque)
    assert len(SessionLogger.logs[request_id]) > 0

    SessionLogger.logs.clear()
    SessionLogger._session_last_seen.clear()


def test_session_logger_cleanup_removes_stale_entries() -> None:
    """Test that cleanup removes entries older than max_age_seconds."""
    SessionLogger.logs.clear()
    SessionLogger._session_last_seen.clear()

    request_id_old = "old-request"
    request_id_fresh = "fresh-request"

    record_old = logging.LogRecord(
        name=__name__,
        level=logging.INFO,
        pathname="",
        lineno=1,
        msg="Old message",
        args=(),
        exc_info=None,
    )
    record_old.request_id = request_id_old
    record_old.user_id = "test-user"
    record_old.session_log_level = logging.INFO
    SessionLogger.process_record(record_old)

    old_ts = SessionLogger._session_last_seen.get(request_id_old)
    assert old_ts is not None

    # Make the first record stale without sleeping.
    with SessionLogger._state_lock:
        SessionLogger._session_last_seen[request_id_old] = time.time() - 3600

    record_fresh = logging.LogRecord(
        name=__name__,
        level=logging.INFO,
        pathname="",
        lineno=1,
        msg="Fresh message",
        args=(),
        exc_info=None,
    )
    record_fresh.request_id = request_id_fresh
    record_fresh.user_id = "test-user"
    record_fresh.session_log_level = logging.INFO
    SessionLogger.process_record(record_fresh)

    fresh_ts = SessionLogger._session_last_seen.get(request_id_fresh)
    assert fresh_ts is not None

    SessionLogger.cleanup(max_age_seconds=0.05)

    assert request_id_old not in SessionLogger.logs
    assert request_id_old not in SessionLogger._session_last_seen
    assert request_id_fresh in SessionLogger.logs
    assert request_id_fresh in SessionLogger._session_last_seen

    SessionLogger.logs.clear()
    SessionLogger._session_last_seen.clear()


def test_session_logger_concurrent_writes_same_request() -> None:
    """Test multiple threads logging to the same request_id don't corrupt."""
    SessionLogger.logs.clear()
    SessionLogger._session_last_seen.clear()

    request_id = "concurrent-request"
    num_threads = 8
    messages_per_thread = 50

    def worker(idx: int) -> None:
        logger = SessionLogger.get_logger(__name__)
        for i in range(messages_per_thread):
            SessionLogger.request_id.set(request_id)
            SessionLogger.session_id.set(f"session-{idx}")
            SessionLogger.user_id.set(f"user-{idx}")
            logger.info(f"Worker {idx} message {i}")
            time.sleep(0.0001)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for f in futures:
            f.result()

    assert request_id in SessionLogger.logs
    buffer = SessionLogger.logs[request_id]
    expected_min_messages = min(
        num_threads * messages_per_thread, SessionLogger.max_lines
    )
    assert len(buffer) == expected_min_messages

    SessionLogger.logs.clear()
    SessionLogger._session_last_seen.clear()
