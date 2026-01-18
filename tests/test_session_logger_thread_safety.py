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
