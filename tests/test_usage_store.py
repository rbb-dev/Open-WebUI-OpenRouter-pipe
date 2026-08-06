"""UsageStore tests: table creation, writer thread, drain-on-stop, purge
lock, overload drops, and disabled-mode no-ops — against a real sqlite DB."""

from __future__ import annotations

import datetime
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, inspect as sa_inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard.usage_store import UsageStore
from open_webui_openrouter_pipe.storage.persistence import ArtifactStore


def _make_store_host() -> Any:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    host = SimpleNamespace(
        _engine=engine,
        _session_factory=sessionmaker(bind=engine),
        logger=Mock(),
        _item_model=None,
        _db_executor=None,
        table_suffix=lambda: "tpipe_ab12cd34",
        _is_table_exists_error=ArtifactStore._is_table_exists_error,
        _maybe_heal_index_conflict=lambda *a, **k: False,
    )
    guard: Any = ArtifactStore._create_table_with_race_guard
    host._create_table_with_race_guard = (
        lambda table, eng, name: guard(host, table, eng, name)
    )
    return host


def _row(**over: Any) -> dict[str, Any]:
    now = datetime.datetime.now(datetime.UTC).astimezone().replace(tzinfo=None)
    base: dict[str, Any] = {
        "ts": now,
        "started_at": now,
        "kind": "chat",
        "user_id": "u1",
        "user_name": "sam",
        "chat_id": "c1",
        "session_id": "s1",
        "model_id": "anthropic/claude-sonnet-4.6",
        "task_name": None,
        "status": "ok",
        "duration_ms": 1200,
        "tokens_in": 100,
        "tokens_out": 20,
        "tokens_reasoning": 0,
        "tokens_cached": 40,
        "tools_ok": 1,
        "tools_failed": 0,
        "retries": 0,
        "cost": 0.01,
        "cache_savings": 0.001,
        "worker_pid": 123,
    }
    base.update(over)
    return base


def _count_rows(usage: UsageStore) -> int:
    factory = usage._store._session_factory
    with factory() as session:
        return session.query(usage._model).count()


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_ensure_creates_table_idempotently():
    host = _make_store_host()
    usage = UsageStore()
    assert usage.ensure(host) is True
    assert usage.table_name == "dashboard_tpipe_ab12cd34"
    assert sa_inspect(host._engine).has_table("dashboard_tpipe_ab12cd34")
    index_names = {ix["name"] for ix in sa_inspect(host._engine).get_indexes("dashboard_tpipe_ab12cd34")}
    assert any("ts" in name for name in index_names)
    assert usage.ensure(host) is True


def test_ensure_disabled_without_engine():
    host = SimpleNamespace(_engine=None, _session_factory=None)
    usage = UsageStore()
    assert usage.ensure(host) is False
    assert usage.enabled is False
    usage.record(_row())
    assert usage.stop() is None


def test_record_persists_through_thread():
    host = _make_store_host()
    usage = UsageStore()
    assert usage.ensure(host)
    usage.record(_row())
    assert usage._thread is not None and usage._thread.is_alive()
    usage.stop()
    assert _count_rows(usage) == 1


def test_stop_drains_pending_rows():
    host = _make_store_host()
    usage = UsageStore()
    assert usage.ensure(host)
    for i in range(5):
        usage.record(_row(chat_id=f"c{i}"))
    usage.stop()
    assert _count_rows(usage) == 5


def test_signal_stop_is_nonblocking_join_writer_drains():
    """signal_stop() must not join (loop-safe); the off-loop join_writer()
    performs the drain-and-exit."""
    host = _make_store_host()
    usage = UsageStore()
    assert usage.ensure(host)
    for i in range(4):
        usage.record(_row(chat_id=f"c{i}"))
    usage.signal_stop()
    assert usage._stop_event.is_set()
    usage.join_writer()
    assert usage._thread is None
    assert _count_rows(usage) == 4


def test_queue_overload_drops_oldest_and_counts():
    host = _make_store_host()
    usage = UsageStore(queue_max=2)
    assert usage.ensure(host)
    usage._start_thread = lambda: None
    usage.record(_row(chat_id="a"))
    usage.record(_row(chat_id="b"))
    assert usage.dropped == 0
    usage.record(_row(chat_id="c"))
    assert usage.dropped == 1
    queued = []
    while not usage._queue.empty():
        item = usage._queue.get_nowait()
        if item is not None:
            queued.append(item["chat_id"])
    assert queued == ["b", "c"]


def test_purge_deletes_only_older_than_cutoff():
    host = _make_store_host()
    usage = UsageStore()
    assert usage.ensure(host)
    old = _row(ts=datetime.datetime.now(datetime.UTC).astimezone().replace(tzinfo=None) - datetime.timedelta(days=40), chat_id="old")
    new = _row(ts=datetime.datetime.now(datetime.UTC).astimezone().replace(tzinfo=None), chat_id="new")
    usage._persist_sync([old, new])
    assert _count_rows(usage) == 2
    usage._retention_days_fn = lambda: 30
    usage._purge_sync(usage._purge_cutoff())
    factory = host._session_factory
    with factory() as session:
        remaining = [r.chat_id for r in session.query(usage._model).all()]
    assert remaining == ["new"]


def test_purge_skips_when_lock_held():
    host = _make_store_host()
    usage = UsageStore()
    assert usage.ensure(host)
    usage._persist_sync([_row(ts=datetime.datetime.now(datetime.UTC).astimezone().replace(tzinfo=None) - datetime.timedelta(days=40))])
    host._item_model = object()
    usage._acquire_purge_lock = lambda store, item_model, lock_id: False
    usage._purge_sync(datetime.datetime.now(datetime.UTC).astimezone().replace(tzinfo=None))
    assert _count_rows(usage) == 1


def test_purge_releases_lock_after_delete():
    host = _make_store_host()
    usage = UsageStore()
    assert usage.ensure(host)
    host._item_model = object()
    host._try_acquire_lock_sync = Mock(return_value=True)
    host._delete_artifacts_sync = Mock()
    usage._reap_stale_lock = lambda *a, **k: None
    usage._purge_sync(datetime.datetime.now(datetime.UTC).astimezone().replace(tzinfo=None))
    assert host._try_acquire_lock_sync.called
    assert host._delete_artifacts_sync.called


def test_retention_read_live_and_clamped():
    usage = UsageStore()
    usage._retention_days_fn = lambda: 0
    cutoff = usage._purge_cutoff()
    assert cutoff <= datetime.datetime.now(datetime.UTC).astimezone().replace(tzinfo=None) - datetime.timedelta(days=1) + datetime.timedelta(seconds=5)
    usage._retention_days_fn = lambda: (_ for _ in ()).throw(RuntimeError())
    assert usage._purge_cutoff() is not None


def test_table_info_counts_records():
    host = _make_store_host()
    usage = UsageStore()
    assert usage.ensure(host)
    usage._persist_sync([_row(), _row(chat_id="c2")])
    info = usage._table_info_sync()
    assert info["records"] == 2
    assert info["approx_bytes"] is None or isinstance(info["approx_bytes"], int)


@pytest.mark.asyncio
async def test_start_and_stop_purge_task():
    host = _make_store_host()
    usage = UsageStore()
    assert usage.ensure(host)
    usage.start_purge_task(lambda: 30)
    assert usage._purge_task is not None
    returned = usage.stop()
    assert returned is not None
    import asyncio

    with pytest.raises(asyncio.CancelledError):
        await returned


@pytest.mark.parametrize(
    "tz",
    ["UTC", "Australia/Brisbane", "America/New_York", "Asia/Kolkata"],
)
def test_purge_cutoff_is_in_the_same_naive_frame_as_stored_timestamps(tz, monkeypatch):
    """The purge cutoff must live in the naive-LOCAL frame, not naive-UTC.

    SessionTracker.db_row stores ``ts`` as naive local time. A cutoff built in any
    other frame is skewed by the UTC offset: east of UTC the naive-UTC cutoff falls
    earlier, so a UTC+10 deployment retains 10 hours more than the retention window
    allows, and west of UTC it falls later, so a UTC-5 deployment purges 5 hours of
    history early. CI runs in UTC, where the two frames coincide, so only a non-UTC
    zone can see the bug.
    """
    monkeypatch.setenv("TZ", tz)
    time.tzset()
    try:
        usage = UsageStore()
        usage._retention_days_fn = lambda: 7

        expected = datetime.datetime.fromtimestamp(time.time() - 7 * 86400)
        cutoff = usage._purge_cutoff()

        assert cutoff.tzinfo is None, "stored timestamps are naive; the cutoff must match"
        assert abs((cutoff - expected).total_seconds()) < 5, (
            f"cutoff is {(cutoff - expected).total_seconds() / 3600:.1f}h away from "
            f"naive-local now-7d under {tz}"
        )
    finally:
        monkeypatch.undo()
        time.tzset()


@pytest.mark.parametrize(
    "tz",
    ["UTC", "Australia/Brisbane", "America/New_York", "Asia/Kolkata"],
)
def test_the_writer_and_the_decoder_agree_on_the_stored_frame(tz, monkeypatch):
    """A row written for an instant must read back as that same instant.

    The test above anchors the purge cutoff to naive-local computed independently, so
    it catches the frame moving. It cannot catch the writer and the decoder moving
    TOGETHER -- a round trip through a self-consistent pair holds in any frame. That
    matters because the pair is now one owner: an edit touches both at once.

    So this asserts both halves. The round trip pins writer against decoder; the
    comparison with an independently built naive-local value pins the pair against the
    frame the column actually holds. Driven through SessionTracker.db_row, which is the
    real writer, not through the helper alone.
    """
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.session_tracker import (
        SessionTracker,
    )
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.usage_store import (
        epoch_from_usage_ts,
    )

    monkeypatch.setenv("TZ", tz)
    time.tzset()
    try:
        started = time.time() - 300.0
        done = time.time()
        row = SessionTracker().db_row({"started": started, "done": done, "kind": "chat"})

        assert row["ts"].tzinfo is None and row["started_at"].tzinfo is None, (
            "the column is DateTime with no timezone; an aware value is stored with its "
            "offset silently dropped"
        )
        assert abs(epoch_from_usage_ts(row["ts"]) - done) < 1e-3, (
            f"a row written for {done} decodes as {epoch_from_usage_ts(row['ts'])} under "
            f"{tz}; the writer and the decoder are in different frames, so every "
            "dashboard timestamp is off by the difference"
        )
        assert abs(epoch_from_usage_ts(row["started_at"]) - started) < 1e-3

        assert abs((row["ts"] - datetime.datetime.fromtimestamp(done)).total_seconds()) < 1, (
            f"the stored value is not naive-local under {tz}. The purge cutoff is built "
            "in naive-local, so a writer in any other frame deletes rows from a window "
            "shifted by the UTC offset -- silently, and permanently."
        )
    finally:
        monkeypatch.undo()
        time.tzset()
