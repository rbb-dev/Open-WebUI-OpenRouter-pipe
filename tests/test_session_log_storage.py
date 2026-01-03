from __future__ import annotations

import json
import os

import pytest

pyzipper = pytest.importorskip("pyzipper")

import open_webui_openrouter_pipe.open_webui_openrouter_pipe as pipe_module
from open_webui_openrouter_pipe.open_webui_openrouter_pipe import (
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


def test_write_session_log_archive_creates_encrypted_zip(tmp_path) -> None:
    pipe = Pipe()
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
        logs = zf.read("logs.jsonl").decode("utf-8").splitlines()
        records = [json.loads(line) for line in logs if line.strip()]
        assert any(rec.get("message") == "hello" for rec in records)
        assert any(rec.get("message") == "world" for rec in records)


def test_enqueue_session_log_archive_skips_when_missing_ids(tmp_path, monkeypatch) -> None:
    pipe = Pipe()

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

    def test_archive_with_missing_ids_skipped(self, tmp_path) -> None:
        """Archive not written when user_id/session_id/chat_id/message_id missing."""
        pipe = Pipe()
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

    def test_archive_password_encryption(self, tmp_path) -> None:
        """Zip archive encrypted with AES and correct password."""
        pipe = Pipe()
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

    def test_archive_compression_options(self, tmp_path) -> None:
        """Different compression algorithms (stored, deflated, bzip2, lzma)."""
        pipe = Pipe()
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

    def test_cleanup_deletes_old_archives(self, tmp_path, monkeypatch) -> None:
        """Archives older than retention window deleted."""
        pipe = Pipe()

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

    def test_cleanup_prunes_empty_directories(self, tmp_path, monkeypatch) -> None:
        """Empty user/chat directories removed after archive deletion."""
        pipe = Pipe()

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
