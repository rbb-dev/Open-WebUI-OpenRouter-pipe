"""Live-sessions slice: collect → expand → aggregate round-trip."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard import dashboard_publisher as sp


@pytest.fixture(autouse=True)
def _reset_getter():
    original = sp._pd_snapshot_getter
    yield
    sp._pd_snapshot_getter = original


def _row(started: float, done: float | None = None, pid: int = 1) -> dict:
    return {
        "user": "sam",
        "model_id": "m",
        "model_name": "M",
        "kind": "chat",
        "status": "completed" if done else "streaming",
        "started": started,
        "done": done,
        "elapsed_s": 1.0,
        "tokens_in": 10,
        "tokens_out": 2,
        "tools_ok": 0,
        "tools_failed": 0,
        "cost": 0.01,
        "task_cost": 0.0,
        "worker_pid": pid,
    }


def test_worker_payload_includes_sessions_from_getter():
    sp.set_snapshot_getter(lambda: ([_row(100.0)], {"c9": 0.003}))
    pipe = Mock()
    pipe._active_pipes_calls = 0
    pipe._global_semaphore = None
    pipe._semaphore_limit = 0
    pipe._tool_global_semaphore = None
    pipe._tool_global_limit = 0
    pipe.valves = None
    pipe._request_queue = None
    pipe._QUEUE_MAXSIZE = 1000
    pipe._log_queue = None
    pipe._session_log_manager = None
    pipe._circuit_breaker = None
    pipe._video_global_semaphore = None
    pipe._video_global_limit = 0
    pipe._video_active_tasks = {}
    payload = sp._collect_worker_payload(pipe)
    assert payload["sl"] == [_row(100.0)]
    assert payload["tc"] == {"c9": 0.003}


def test_snapshot_getter_failure_yields_empty_snapshot():
    sp.set_snapshot_getter(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert sp._snapshot_safe() == ([], {})
    sp.set_snapshot_getter(lambda: "not-a-tuple")
    assert sp._snapshot_safe() == ([], {})
    sp.set_snapshot_getter(None)
    assert sp._snapshot_safe() == ([], {})


def test_expand_passes_sessions_through():
    expanded = sp.expand_worker_payload({"pid": 1, "sl": [_row(5.0)]})
    assert expanded["sessions_live"] == [_row(5.0)]
    assert sp.expand_worker_payload({})["sessions_live"] == []
    assert sp.expand_worker_payload({"sl": "junk"})["sessions_live"] == []


def test_expand_passes_task_costs_through():
    assert sp.expand_worker_payload({"pid": 1, "tc": {"c1": 0.004}})["task_costs"] == {"c1": 0.004}
    assert sp.expand_worker_payload({})["task_costs"] == {}
    assert sp.expand_worker_payload({"tc": "junk"})["task_costs"] == {}


def test_aggregate_concats_sorts_and_caps_sessions():
    w1 = {"pid": 1, "uptime_s": 1.0, "sessions": {"in_flight": 0},
          "sessions_live": [_row(100.0, done=200.0, pid=1), _row(300.0, pid=1)]}
    w2 = {"pid": 2, "uptime_s": 1.0, "sessions": {"in_flight": 0},
          "sessions_live": [_row(400.0, pid=2)]}
    result = sp.aggregate_worker_payloads([w1, w2])
    rows = result["sessions_live"]
    assert [r["started"] for r in rows] == [400.0, 300.0, 100.0]
    assert rows[-1]["done"] == 200.0

    big = {"pid": 3, "uptime_s": 1.0, "sessions": {"in_flight": 0},
           "sessions_live": [_row(float(i), pid=3) for i in range(sp._PD_SESSIONS_CAP + 50)]}
    capped = sp.aggregate_worker_payloads([big])
    assert len(capped["sessions_live"]) == sp._PD_SESSIONS_CAP


def test_aggregate_folds_cross_worker_task_costs():
    # Worker 1 holds the parent chat; worker 2 holds only the task cost for it
    # (title/tag task fired on a different worker) plus a truly-orphan task cost.
    w1 = {"pid": 1, "uptime_s": 1.0, "sessions": {"in_flight": 0},
          "sessions_live": [dict(_row(300.0), chat_id="c1", cost=0.10, task_cost=0.0)],
          "task_costs": {}}
    w2 = {"pid": 2, "uptime_s": 1.0, "sessions": {"in_flight": 0},
          "sessions_live": [],
          "task_costs": {"c1": 0.004, "c-orphan": 0.002}}
    rows = sp.aggregate_worker_payloads([w1, w2])["sessions_live"]
    assert len(rows) == 1                              # truly-orphan cost dropped, no synthetic row
    assert rows[0]["cost"] == pytest.approx(0.104)     # 0.10 + cross-worker folded 0.004
    assert rows[0]["task_cost"] == pytest.approx(0.004)
    assert "chat_id" not in rows[0]                    # stripped before emit
