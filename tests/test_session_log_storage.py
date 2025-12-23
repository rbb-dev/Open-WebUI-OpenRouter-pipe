from __future__ import annotations

import json

import pyzipper

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import Pipe, SessionLogger, _SessionLogArchiveJob, _sanitize_path_component


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
        log_lines=["hello", "world"],
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
        assert {"meta.json", "logs.txt"} <= names
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        assert meta["ids"]["user_id"] == job.user_id
        assert meta["ids"]["session_id"] == job.session_id
        assert meta["ids"]["chat_id"] == job.chat_id
        assert meta["ids"]["message_id"] == job.message_id
        assert meta["request_id"] == job.request_id
        logs = zf.read("logs.txt").decode("utf-8")
        assert "hello" in logs
        assert "world" in logs


def test_enqueue_session_log_archive_skips_when_missing_ids(tmp_path, monkeypatch) -> None:
    pipe = Pipe()

    valves = pipe.Valves(
        SESSION_LOG_STORE_ENABLED=True,
        SESSION_LOG_DIR=str(tmp_path),
        SESSION_LOG_ZIP_PASSWORD="secret",
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
        log_lines=["x"],
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
    assert any("DEBUG_ONLY_TOKEN" in line for line in lines)
    assert any("WARNING_TOKEN" in line for line in lines)
    SessionLogger.logs.pop(request_id, None)
