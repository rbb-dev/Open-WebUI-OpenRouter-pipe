"""SessionTracker lifecycle tests + PipeDashboardPlugin capture wiring."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard.session_tracker import (
    SessionTracker,
    _usage_numbers,
)


def _usage(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "input_tokens": 100,
        "output_tokens": 20,
        "input_tokens_details": {"cached_tokens": 40},
        "output_tokens_details": {"reasoning_tokens": 5},
        "cost": 0.01,
    }
    base.update(over)
    return base


def _start(tracker: SessionTracker, rid: str = "r1", **over: Any) -> None:
    kwargs: dict[str, Any] = {
        "body": {"model": "anthropic/claude-sonnet-4.6"},
        "user": {"id": "u1", "name": "sam"},
        "metadata": {"chat_id": "c1", "session_id": "s1", "user_id": "u1"},
        "task": None,
    }
    kwargs.update(over)
    tracker.start(rid, **kwargs)


def test_usage_numbers_parses_both_shapes():
    numbers = _usage_numbers(_usage())
    assert numbers == {"tin": 100, "tout": 20, "treason": 5, "tcached": 40, "cost": 0.01, "discount": 0.0}
    chat_shape = _usage_numbers({"prompt_tokens": 7, "completion_tokens": 3, "cache_discount": -0.002})
    assert chat_shape["tin"] == 7 and chat_shape["tout"] == 3 and chat_shape["discount"] == -0.002
    assert _usage_numbers(None)["tin"] == 0


def test_full_lifecycle_running_total_then_final_overwrite():
    tracker = SessionTracker()
    _start(tracker)
    tracker.update_usage("r1", _usage(cost=0.05))
    rows = tracker.live_sessions()
    assert rows[0]["status"] == "streaming"
    assert rows[0]["cost"] == 0.05
    tracker.tool_started("r1", "web_search")
    assert tracker.live_sessions()[0]["status"] == "tool:web_search"
    tracker.tool_result("r1", "completed")
    tracker.tool_result("r1", "failed")
    tracker.finalize("r1", _usage(cost=0.18), "ok")
    rows = tracker.live_sessions()
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed"
    assert row["cost"] == 0.18
    assert row["tools_ok"] == 1 and row["tools_failed"] == 1
    assert row["done"] is not None


def test_finalize_is_idempotent_backstop_safe():
    """The _execute_pipe_job backstop calls finalize again for every request;
    a real terminal that already finalized must make the second call a no-op
    (one row, no double-count)."""
    tracker = SessionTracker()
    rows: list[dict[str, Any]] = []
    tracker.on_finalize = rows.append
    _start(tracker)
    tracker.finalize("r1", _usage(cost=0.05), "ok")
    tracker.finalize("r1", None, "failed")  # backstop after a real terminal
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["cost"] == 0.05
    live = tracker.live_sessions()
    assert len(live) == 1 and live[0]["status"] == "completed"


def test_backstop_finalizes_never_terminated_session():
    """A session that never got a real terminal (early return / exception in
    the request path) is finalized by the backstop instead of leaking."""
    tracker = SessionTracker()
    rows: list[dict[str, Any]] = []
    tracker.on_finalize = rows.append
    _start(tracker)
    tracker.finalize("r1", None, "failed")  # backstop only
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    assert tracker.live_sessions()[0]["status"] == "failed"


def test_status_mapping_failed_and_cancelled():
    tracker = SessionTracker()
    _start(tracker, rid="rf")
    tracker.finalize("rf", None, "failed")
    _start(tracker, rid="rc")
    tracker.finalize("rc", None, "cancelled")
    statuses = {row["status"] for row in tracker.live_sessions()}
    assert statuses == {"failed", "cancelled"}


def test_task_cost_folds_into_parent_and_task_row_hidden():
    tracker = SessionTracker()
    _start(tracker, rid="chat1")
    tracker.finalize("chat1", _usage(cost=0.10), "ok")
    _start(tracker, rid="task1", body={"model": "google/gemini-3-flash"}, task="title_generation")
    tracker.finalize("task1", _usage(cost=0.004, input_tokens=10, output_tokens=2), "ok")
    rows = tracker.live_sessions()
    # Task rows fold into the parent chat — they must NOT appear as their own
    # live row (else the summary strip double-counts their cost).
    assert all(row["kind"] != "task" for row in rows)
    chat = next(row for row in rows if row["kind"] == "chat")
    assert chat["cost"] == pytest.approx(0.104)
    assert chat["task_cost"] == pytest.approx(0.004)


def test_task_fold_survives_parent_finalize_overwrite():
    """Task finalizes BEFORE its parent (parent still active): the fold must
    survive the parent's own finalize overwriting its usage cost."""
    tracker = SessionTracker()
    _start(tracker, rid="chat1")
    _start(tracker, rid="task1", body={"model": "google/gemini-3-flash"}, task="tags_generation")
    tracker.finalize("task1", _usage(cost=0.004, input_tokens=10, output_tokens=2), "ok")
    tracker.finalize("chat1", _usage(cost=0.10), "ok")
    chat = next(row for row in tracker.live_sessions() if row["kind"] == "chat")
    assert chat["cost"] == pytest.approx(0.104)
    assert chat["task_cost"] == pytest.approx(0.004)


def test_db_row_excludes_folded_task_cost():
    """DB rows carry the session's own cost only; task rows persist separately."""
    tracker = SessionTracker()
    captured: list[dict[str, Any]] = []
    tracker.on_finalize = captured.append
    _start(tracker, rid="chat1")
    _start(tracker, rid="task1", task="title_generation")
    tracker.finalize("task1", _usage(cost=0.004), "ok")
    tracker.finalize("chat1", _usage(cost=0.10), "ok")
    chat_entry = next(e for e in captured if e.get("kind") == "chat")
    assert tracker.db_row(chat_entry)["cost"] == pytest.approx(0.10)


def test_cache_savings_prefers_discount_then_pricing():
    tracker = SessionTracker(pricing_fn=lambda mid: {"prompt": "0.000002", "input_cache_read": "0.0000002"})
    _start(tracker, rid="r1")
    tracker.finalize("r1", _usage(cache_discount=-0.0031), "ok")
    finalized: list[dict[str, Any]] = []
    tracker2 = SessionTracker(pricing_fn=lambda mid: {"prompt": "0.000002", "input_cache_read": "0.0000002"})
    tracker2.on_finalize = finalized.append
    _start(tracker2, rid="r2")
    tracker2.finalize("r2", _usage(input_tokens_details={"cached_tokens": 1000}), "ok")
    assert finalized[0]["savings"] == pytest.approx(0.0018)
    tracker3 = SessionTracker(pricing_fn=lambda mid: None)
    _start(tracker3, rid="r3")
    tracker3.finalize("r3", _usage(input_tokens_details={"cached_tokens": 1000}), "ok")


def test_db_row_mapping():
    tracker = SessionTracker()
    captured: list[dict[str, Any]] = []
    tracker.on_finalize = captured.append
    _start(tracker)
    tracker.retry("r1")
    tracker.finalize("r1", _usage(), "ok")
    row = tracker.db_row(captured[0])
    assert row["status"] == "ok"
    assert row["kind"] == "chat"
    assert row["user_name"] == "sam"
    assert row["tokens_in"] == 100 and row["tokens_cached"] == 40
    assert row["retries"] == 1
    assert row["duration_ms"] >= 0
    assert row["worker_pid"] > 0


def test_sweep_abandons_stale_sessions():
    tracker = SessionTracker()
    finalized: list[dict[str, Any]] = []
    tracker.on_finalize = finalized.append
    _start(tracker)
    with tracker._lock:
        tracker._active["r1"]["started"] = time.time() - 8000
    tracker.sweep()
    assert finalized and finalized[0]["status"] == "failed"
    assert tracker.live_sessions()[0]["status"] == "failed"


def test_finalize_callback_exception_isolated():
    tracker = SessionTracker()
    tracker.on_finalize = Mock(side_effect=RuntimeError("boom"))
    _start(tracker)
    tracker.finalize("r1", None, "ok")
    assert tracker.live_sessions()[0]["status"] == "completed"


def test_empty_request_id_noop_and_unknown_ids_safe():
    tracker = SessionTracker()
    tracker.start("")
    tracker.update_usage("missing", _usage())
    tracker.tool_result("missing", "failed")
    tracker.retry("missing")
    tracker.finalize("missing", None, "ok")
    assert tracker.live_sessions() == []


def test_live_sessions_caps_and_fields():
    tracker = SessionTracker()
    for i in range(40):
        _start(tracker, rid=f"r{i}")
    rows = tracker.live_sessions()
    assert len(rows) == 30
    sample = rows[0]
    assert set(sample) >= {"user", "model_id", "model_name", "kind", "status", "started",
                           "elapsed_s", "tokens_in", "tokens_cached", "tokens_out", "tools_ok", "tools_failed",
                           "cost", "worker_pid", "chat_id"}


def test_task_costs_by_chat_excludes_same_worker_folded():
    tracker = SessionTracker()
    # chat + its task on the SAME worker (both chat_id="c1") fold locally, so the
    # task is NOT reported for cross-worker folding.
    _start(tracker, rid="chat1")
    tracker.finalize("chat1", _usage(cost=0.10), "ok")
    _start(tracker, rid="task1", task="title_generation")
    tracker.finalize("task1", _usage(cost=0.004), "ok")
    # a task whose parent chat is NOT on this worker (different chat_id, no chat entry)
    _start(tracker, rid="task2", task="tags_generation",
           metadata={"chat_id": "c-other", "session_id": "s1", "user_id": "u1"})
    tracker.finalize("task2", _usage(cost=0.002), "ok")
    tc = tracker.task_costs_by_chat()
    assert list(tc.keys()) == ["c-other"]
    assert tc["c-other"] == pytest.approx(0.002)


def test_live_snapshot_returns_atomic_sessions_and_task_costs():
    tracker = SessionTracker()
    _start(tracker, rid="chat1")
    tracker.finalize("chat1", _usage(cost=0.10), "ok")
    _start(tracker, rid="task2", task="tags_generation",
           metadata={"chat_id": "c-other", "session_id": "s1", "user_id": "u1"})
    tracker.finalize("task2", _usage(cost=0.002), "ok")
    rows, task_costs = tracker.live_snapshot()
    assert isinstance(rows, list) and isinstance(task_costs, dict)
    assert [r["kind"] for r in rows] == ["chat"]
    assert rows[0]["chat_id"] == "c1"
    assert task_costs == {"c-other": pytest.approx(0.002)}
    # the single-lock snapshot returns the same content as the two façade getters
    assert rows == tracker.live_sessions()
    assert task_costs == tracker.task_costs_by_chat()


# ── Plugin wiring ──


def _make_plugin():
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.plugin import PipeDashboardPlugin

    plugin = PipeDashboardPlugin()
    plugin.ctx = Mock()
    plugin.ctx.valves.PIPE_DASHBOARD_ENABLE = True
    plugin.ctx.valves.PIPE_DASHBOARD_USAGE_COLLECT = False
    plugin.ctx.valves.PIPE_DASHBOARD_USAGE_RETENTION_DAYS = 30
    plugin._get_pipe = lambda: None
    return plugin


@pytest.mark.asyncio
async def test_plugin_on_request_tracks_only_foreign_models():
    plugin = _make_plugin()
    plugin._tracker = Mock()
    result = await plugin.on_request(
        {"model": "anthropic/claude-sonnet-4.6"}, {"id": "u"}, {"chat_id": "c"}, None, None,
        request_id="rid-1",
    )
    assert result is None
    assert plugin._tracker.start.call_args.args[0] == "rid-1"

    plugin._tracker.reset_mock()
    plugin.ctx.build_response = Mock(return_value={"choices": []})
    await plugin.on_request({"model": "pipe-dashboard"}, {"id": "u"}, {}, None, "title_generation", request_id="rid-2")
    assert not plugin._tracker.start.called


@pytest.mark.asyncio
async def test_plugin_emitter_wrap_observes_usage_and_tool_start():
    plugin = _make_plugin()
    inner = AsyncMock(return_value="sent")
    wrapped = await plugin.on_emitter_wrap(inner, job_metadata={"request_id": "rid-9"})
    assert wrapped is not None
    plugin._tracker.start("rid-9", body={"model": "m"}, user={}, metadata={})

    assert await wrapped({"type": "chat:completion", "data": {"usage": _usage(cost=0.02)}}) == "sent"
    assert plugin._tracker.live_sessions()[0]["cost"] == 0.02

    await wrapped({"type": "response.output_item.added",
                   "item": {"type": "function_call", "name": "web_search", "status": "in_progress"}})
    assert plugin._tracker.live_sessions()[0]["status"] == "tool:web_search"

    assert await wrapped("not-a-dict") == "sent"
    assert inner.await_count == 3

    assert await plugin.on_emitter_wrap(inner, job_metadata={}) is None


@pytest.mark.asyncio
async def test_plugin_generation_complete_finalizes_and_respects_valve():
    plugin = _make_plugin()
    plugin._usage_store = Mock()
    plugin._usage_store.enabled = True
    plugin._tracker.start("rid-3", body={"model": "m"}, user={"id": "u", "name": "n"}, metadata={"chat_id": "c"})
    await plugin.on_generation_complete(_usage(cost=0.07), "ok", request_id="rid-3")
    assert plugin._tracker.live_sessions()[0]["status"] == "completed"
    assert not plugin._usage_store.record.called

    plugin.ctx.valves.PIPE_DASHBOARD_USAGE_COLLECT = True
    pipe = Mock()
    plugin._get_pipe = lambda: pipe
    plugin._tracker.start("rid-4", body={"model": "m"}, user={"id": "u", "name": "n"}, metadata={"chat_id": "c"})
    await plugin.on_generation_complete(_usage(cost=0.07), "failed", request_id="rid-4")
    assert plugin._usage_store.record.called
    recorded = plugin._usage_store.record.call_args.args[0]
    assert recorded["status"] == "failed"
    assert recorded["cost"] == 0.07


def test_plugin_live_snapshot_sweeps_and_returns_tuple():
    plugin = _make_plugin()
    plugin._tracker.start("chat1", body={"model": "m"}, user={"id": "u", "name": "sam"},
                          metadata={"chat_id": "c1"})
    plugin._tracker.finalize("chat1", _usage(cost=0.05), "ok")
    rows, task_costs = plugin._live_snapshot()
    assert isinstance(rows, list) and isinstance(task_costs, dict)
    assert rows[0]["chat_id"] == "c1"
    assert rows[0]["cost"] == pytest.approx(0.05)
    # a tracker that raises during sweep must never propagate — empty snapshot instead
    plugin._tracker = Mock()
    plugin._tracker.sweep.side_effect = RuntimeError("boom")
    assert plugin._live_snapshot() == ([], {})


@pytest.mark.asyncio
async def test_plugin_tool_and_retry_hooks_update_counters():
    plugin = _make_plugin()
    plugin._tracker.start("rid-5", body={"model": "m"}, user={}, metadata={})
    await plugin.on_tool_result("web_search", "completed", request_id="rid-5")
    await plugin.on_tool_result("web_search", "failed", request_id="rid-5")
    await plugin.on_request_retry("signature", request_id="rid-5")
    row = plugin._tracker.live_sessions()[0]
    assert row["tools_ok"] == 1 and row["tools_failed"] == 1
    with plugin._tracker._lock:
        assert plugin._tracker._active["rid-5"]["retries"] == 1


@pytest.mark.asyncio
async def test_plugin_on_shutdown_combines_publisher_and_purge():
    plugin = _make_plugin()
    loop = asyncio.get_running_loop()
    publisher = loop.create_task(asyncio.sleep(60))
    purge = loop.create_task(asyncio.sleep(60))
    plugin._publisher_task = publisher
    plugin._usage_store = Mock()
    plugin._usage_store.signal_stop = Mock(return_value=purge)
    plugin._usage_store.join_writer = Mock()
    purge.cancel()
    returned = plugin.on_shutdown()
    assert returned is not None
    await returned
    assert publisher.cancelled()
    # writer join happens off the event loop (via asyncio.to_thread)
    assert plugin._usage_store.join_writer.called
