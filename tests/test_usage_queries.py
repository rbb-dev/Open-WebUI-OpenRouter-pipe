"""usage_queries tests: cards, tz bucketing, task model rows, top-N users, memo,
retention clamp — against a real seeded sqlite usage table."""

from __future__ import annotations

import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard import usage_queries as uq
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
        table_suffix=lambda: "qpipe_ab12cd34",
        _is_table_exists_error=ArtifactStore._is_table_exists_error,
        _maybe_heal_index_conflict=lambda *a, **k: False,
    )
    guard: Any = ArtifactStore._create_table_with_race_guard
    host._create_table_with_race_guard = (
        lambda table, eng, name: guard(host, table, eng, name)
    )
    return host


def _row(ts: float, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ts": datetime.datetime.fromtimestamp(ts),
        "started_at": datetime.datetime.fromtimestamp(ts - 5),
        "kind": "chat",
        "user_id": "u1",
        "user_name": "sam",
        "chat_id": "c1",
        "session_id": "s1",
        "model_id": "vendor/model-a",
        "task_name": None,
        "status": "ok",
        "duration_ms": 5000,
        "tokens_in": 100,
        "tokens_out": 10,
        "tokens_reasoning": 2,
        "tokens_cached": 40,
        "tools_ok": 1,
        "tools_failed": 0,
        "retries": 0,
        "cost": 0.01,
        "cache_savings": 0.001,
        "worker_pid": 1,
    }
    base.update(over)
    return base


@pytest.fixture()
def seeded():
    host = _make_store_host()
    usage = UsageStore()
    assert usage.ensure(host)
    return host, usage


def _query(usage: UsageStore, host: Any, now: float, **kw: Any) -> dict[str, Any]:
    params: dict[str, Any] = {"range_key": "24h", "tz_offset_min": 0, "include_tasks": True, "name_fn": None}
    params.update(kw)
    return uq.query_usage_stats(usage._model, host._session_factory, now=now, **params)


def test_cards_current_vs_prev_and_totals(seeded):
    host, usage = seeded
    now = time.time()
    usage._persist_sync([
        _row(now - 100),
        _row(now - 200, status="failed", retries=2, cost=0.02),
        _row(now - 90000, cost=0.5),
        _row(now - 200000, cost=1.0),
    ])
    result = _query(usage, host, now)
    cards = result["cards"]
    assert cards["sessions"] == {"count": 2, "failed": 1, "cancelled": 0, "retried": 1}
    assert cards["tokens"]["input"] == 200
    assert cards["tokens"]["cached"] == 80
    assert cards["cost"]["total"] == pytest.approx(0.03)
    assert cards["errors"]["rate"] == pytest.approx(0.5)
    assert cards["cached"]["pct"] == pytest.approx(0.4)
    assert result["prev"] is not None
    assert result["prev"]["sessions"]["count"] == 1
    assert result["totals"]["sessions"] == 4
    assert result["totals"]["cost"] == pytest.approx(1.53)
    assert result["meta"]["bucket_s"] == 300


def test_prev_omitted_without_full_prior_window(seeded):
    host, usage = seeded
    now = time.time()
    usage._persist_sync([_row(now - 100)])
    result = _query(usage, host, now)
    assert result["prev"] is None


def test_bucket_alignment_respects_tz_offset(seeded):
    host, usage = seeded
    off_min = 600
    off = off_min * 60
    # Deterministic base: (base + off) aligned to a 5-min bucket boundary so the
    # first two events land in one bucket and the third in a later one.
    base = (1783231200 // 300) * 300 - off
    now = base + 5000
    usage._persist_sync([_row(base), _row(base + 100), _row(base + 3700)])
    result = _query(usage, host, now, tz_offset_min=off_min)
    buckets = result["buckets"]
    assert len(buckets) == 2
    for bucket in buckets:
        assert (bucket["t"] + off) % 300 == 0
    assert buckets[0]["sessions"] == 2 and buckets[1]["sessions"] == 1


def test_include_tasks_toggle(seeded):
    host, usage = seeded
    now = time.time()
    usage._persist_sync([
        _row(now - 100),
        _row(now - 50, kind="task", task_name="title_generation", model_id="google/flash",
             chat_id="c1", cost=0.004, tokens_in=10, tokens_out=1, tools_ok=0),
    ])
    with_tasks = _query(usage, host, now)
    # A task is not a session even when tasks are included; its cost still counts.
    assert with_tasks["cards"]["sessions"]["count"] == 1
    assert with_tasks["cards"]["cost"]["task_portion"] == pytest.approx(0.004)
    without = _query(usage, host, now, include_tasks=False)
    assert without["cards"]["sessions"]["count"] == 1
    assert without["cards"]["cost"]["task_portion"] == 0.0
    # Totals respect the toggle too: task spend is included only when tasks are on;
    # sessions never count tasks either way.
    assert with_tasks["totals"]["cost"] == pytest.approx(0.014)
    assert with_tasks["totals"]["sessions"] == 1
    assert without["totals"]["cost"] == pytest.approx(0.01)
    assert without["totals"]["sessions"] == 1


def test_task_only_window_zero_sessions_but_counts_cost(seeded):
    host, usage = seeded
    now = time.time()
    usage._persist_sync([
        _row(now - 100, kind="task", task_name="title_generation", model_id="google/flash",
             chat_id="c1", cost=0.0005, tokens_in=500, tokens_out=100, tools_ok=0, tools_failed=0),
        _row(now - 90, kind="task", task_name="tags_generation", model_id="google/flash",
             chat_id="c2", status="failed", retries=1, cost=0.0002, tokens_in=400, tokens_out=80, tools_ok=0),
    ])
    result = _query(usage, host, now)
    cards = result["cards"]
    # Tasks are background requests, never sessions — no place counts them as one.
    assert cards["sessions"]["count"] == 0
    assert cards["sessions"]["failed"] == 0
    assert cards["sessions"]["retried"] == 0
    assert result["totals"]["sessions"] == 0
    assert cards["errors"]["rate"] == 0.0
    assert cards["cost"]["avg_per_session"] == 0.0
    assert all(b["sessions"] == 0 for b in result["buckets"])
    # ...but their spend IS counted.
    assert cards["cost"]["total"] == pytest.approx(0.0007)
    assert result["totals"]["cost"] == pytest.approx(0.0007)
    assert cards["tokens"]["input"] == 900


def test_sessions_exclude_tasks_but_chat_failures_count(seeded):
    host, usage = seeded
    now = time.time()
    usage._persist_sync([
        _row(now - 100, status="failed", cost=0.01),
        _row(now - 90, cost=0.02),
        _row(now - 80, kind="task", task_name="title", model_id="google/flash",
             chat_id="c9", status="failed", retries=2, cost=0.001),
    ])
    cards = _query(usage, host, now)["cards"]
    assert cards["sessions"]["count"] == 2          # 2 chats; the task is not a session
    assert cards["sessions"]["failed"] == 1         # only the failed chat
    assert cards["sessions"]["retried"] == 0        # task retries don't count
    assert cards["errors"]["rate"] == 0.5           # 1 failed / 2 sessions
    assert cards["cost"]["avg_per_session"] == pytest.approx(0.031 / 2)  # task cost in numerator, chats in denominator
    assert cards["cost"]["total"] == pytest.approx(0.031)


def test_task_rows_get_their_own_model_row_with_count(seeded):
    """Task executions get their own (tasks) model row with real cost, not folded into the parent."""
    host, usage = seeded
    now = time.time()
    usage._persist_sync([
        _row(now - 100, chat_id="c1", model_id="vendor/model-a", cost=0.10),
        _row(now - 50, kind="task", task_name="title_generation", chat_id="c1",
             model_id="google/flash", cost=0.004, tokens_in=10, tokens_out=1, tools_ok=0),
        _row(now - 40, kind="task", task_name="tags_generation", chat_id="c-orphan",
             model_id="google/flash", cost=0.002, tokens_in=5, tokens_out=1, tools_ok=0),
    ])
    result = _query(usage, host, now, name_fn=lambda mid: mid.upper())
    rows = {row["model_name"]: row for row in result["by_model"]}
    # The chat model keeps ONLY its own cost — task cost no longer folds in.
    chat = rows["VENDOR/MODEL-A"]
    assert chat["sessions"] == 1
    assert chat["cost"] == pytest.approx(0.10)
    assert chat["tokens_cached"] == 40
    # Both task executions (different types, one orphan) land on the (tasks) row.
    tasks = rows["GOOGLE/FLASH (tasks)"]
    assert tasks["sessions"] == 2
    assert tasks["cost"] == pytest.approx(0.006)
    assert tasks["tokens_cached"] == 80
    assert tasks["avg_cost"] == pytest.approx(0.003)
    # Tasks still never inflate the headline session count.
    assert result["cards"]["sessions"]["count"] == 1
    assert sum(row["share_pct"] for row in result["by_model"]) == pytest.approx(100.0, abs=0.2)


def test_by_user_returns_all_sorted_by_cost(seeded):
    """All users returned sorted by cost; top-N + 'others' is now client-side."""
    host, usage = seeded
    now = time.time()
    rows = [
        _row(now - 100 - i, user_id=f"u{i}", user_name=f"user{i}", cost=1.0 - i * 0.05,
             tools_ok=1, tools_failed=(1 if i == 0 else 0))
        for i in range(12)
    ]
    usage._persist_sync(rows)
    users = _query(usage, host, now)["by_user"]
    assert len(users) == 12
    assert [u["user_id"] for u in users] == [f"u{i}" for i in range(12)]
    assert users[0]["tokens_cached"] == 40 and users[0]["tools"] == 2 and users[0]["tools_failed"] == 1
    assert users[0]["last_active"] is not None
    assert all("others" not in u["user_name"] for u in users)


@pytest.mark.asyncio
async def test_memo_returns_copy_and_quantizes_tz(seeded, monkeypatch):
    host, usage = seeded
    host._db_executor = ThreadPoolExecutor(max_workers=1)
    usage._persist_sync([_row(time.time() - 100)])
    plugin = SimpleNamespace(
        ctx=SimpleNamespace(valves=SimpleNamespace(
            PIPE_DASHBOARD_USAGE_COLLECT=True, PIPE_DASHBOARD_USAGE_RETENTION_DAYS=30,
        )),
        _usage_store=usage,
    )
    pipe = SimpleNamespace(_artifact_store=host)
    monkeypatch.setattr(uq, "_UQ_MEMO", {})

    r1 = await uq.run_usage_query(plugin, pipe, {"range": "24h", "tz_offset_min": 607, "include_tasks": True})
    r1["totals"]["cost"] = 999.0  # mutate the returned dict
    r2 = await uq.run_usage_query(plugin, pipe, {"range": "24h", "tz_offset_min": 601, "include_tasks": True})
    # 607 and 601 both quantize to 600 → same memo key; the cache must not be
    # poisoned by the r1 mutation.
    assert r2["totals"]["cost"] != 999.0
    assert list(uq._UQ_MEMO.keys()) == [("24h", True, 600)]


@pytest.mark.asyncio
async def test_run_usage_query_clamps_and_memoizes(seeded, monkeypatch):
    host, usage = seeded
    host._db_executor = ThreadPoolExecutor(max_workers=1)
    usage._persist_sync([_row(time.time() - 100)])

    plugin = SimpleNamespace(
        ctx=SimpleNamespace(valves=SimpleNamespace(
            PIPE_DASHBOARD_USAGE_COLLECT=True, PIPE_DASHBOARD_USAGE_RETENTION_DAYS=7,
        )),
        _usage_store=usage,
    )
    pipe = SimpleNamespace(_artifact_store=host)
    monkeypatch.setattr(uq, "_UQ_MEMO", {})

    denied = await uq.run_usage_query(plugin, pipe, {"range": "30d", "tz_offset_min": 0, "include_tasks": True})
    assert denied["available"] is False and "retention" in denied["reason"]

    unknown = await uq.run_usage_query(plugin, pipe, {"range": "9y", "tz_offset_min": 0, "include_tasks": True})
    assert unknown["available"] is False

    calls = {"n": 0}
    real = uq.query_usage_stats

    def _counting(*a: Any, **kw: Any):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(uq, "query_usage_stats", _counting)
    args = {"range": "24h", "tz_offset_min": 0, "include_tasks": True}
    first = await uq.run_usage_query(plugin, pipe, args)
    second = await uq.run_usage_query(plugin, pipe, args)
    assert first["available"] is True
    assert first["meta"]["records"] == 1
    assert first["meta"]["collect_on"] is True
    # Memo returns an independent deep copy each hit (no shared-mutable-dict
    # poisoning), but the underlying query ran only once.
    assert second == first
    assert second is not first
    assert calls["n"] == 1

    no_store = await uq.run_usage_query(plugin, SimpleNamespace(_artifact_store=None), args)
    assert no_store["available"] is False


@pytest.mark.asyncio
async def test_run_usage_query_warms_cold_store(seeded, monkeypatch):
    host, usage = seeded
    real_exec = ThreadPoolExecutor(max_workers=1)
    host._db_executor = real_exec
    usage._persist_sync([_row(time.time() - 100)])

    # Cold worker: DB layer fully uninitialized (session factory AND executor)
    # and usage model unbuilt. _ensure_artifact_store restores them, as the real
    # lazy init does on the first chat request.
    real_sf = host._session_factory
    host._session_factory = None
    host._db_executor = None
    usage._model = None
    warm_calls = {"n": 0}

    def _warm(v: Any, pid: str) -> None:
        warm_calls["n"] += 1
        host._session_factory = real_sf
        host._db_executor = real_exec

    host._ensure_artifact_store = _warm
    plugin = SimpleNamespace(
        ctx=SimpleNamespace(valves=SimpleNamespace(
            PIPE_DASHBOARD_USAGE_COLLECT=True, PIPE_DASHBOARD_USAGE_RETENTION_DAYS=30,
        )),
        _usage_store=usage,
    )
    pipe = SimpleNamespace(_artifact_store=host, valves=SimpleNamespace(), id="pipe_x")
    monkeypatch.setattr(uq, "_UQ_MEMO", {})

    result = await uq.run_usage_query(plugin, pipe, {"range": "24h", "tz_offset_min": 0, "include_tasks": True})
    assert warm_calls["n"] == 1
    assert result["available"] is True
    assert result["meta"]["records"] == 1


@pytest.mark.asyncio
async def test_run_usage_query_reports_warm_failure_reason(seeded, monkeypatch):
    host, usage = seeded
    host._session_factory = None
    usage._model = None

    def _warm(v: Any, pid: str) -> None:
        raise RuntimeError("db locked")

    host._ensure_artifact_store = _warm
    plugin = SimpleNamespace(
        ctx=SimpleNamespace(valves=SimpleNamespace(
            PIPE_DASHBOARD_USAGE_COLLECT=True, PIPE_DASHBOARD_USAGE_RETENTION_DAYS=30,
        )),
        _usage_store=usage,
    )
    pipe = SimpleNamespace(_artifact_store=host, valves=SimpleNamespace(), id="pipe_x")
    monkeypatch.setattr(uq, "_UQ_MEMO", {})

    result = await uq.run_usage_query(plugin, pipe, {"range": "24h", "tz_offset_min": 0, "include_tasks": True})
    assert result["available"] is False
    assert "storage unavailable" in result["reason"]
    assert "RuntimeError" in result["reason"]


@pytest.mark.asyncio
async def test_run_usage_query_surfaces_warm_error_via_second_guard(seeded, monkeypatch):
    # Model built earlier (enabled) but the store's DB layer later went cold, so
    # the first bail is skipped and control reaches the second guard; a raising
    # re-warm must still surface the exception type there.
    host, usage = seeded
    host._session_factory = None
    host._db_executor = None

    def _warm(v: Any, pid: str) -> None:
        raise RuntimeError("db gone")

    host._ensure_artifact_store = _warm
    plugin = SimpleNamespace(
        ctx=SimpleNamespace(valves=SimpleNamespace(
            PIPE_DASHBOARD_USAGE_COLLECT=True, PIPE_DASHBOARD_USAGE_RETENTION_DAYS=30,
        )),
        _usage_store=usage,
    )
    pipe = SimpleNamespace(_artifact_store=host, valves=SimpleNamespace(), id="pipe_x")
    monkeypatch.setattr(uq, "_UQ_MEMO", {})

    result = await uq.run_usage_query(plugin, pipe, {"range": "24h", "tz_offset_min": 0, "include_tasks": True})
    assert result["available"] is False
    assert "RuntimeError" in result["reason"]
