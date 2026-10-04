"""usage_queries tests: cards, tz bucketing, task model rows, top-N users, memo,
retention clamp — against a real seeded sqlite usage table."""

from __future__ import annotations

import datetime
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from sqlalchemy import (
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    create_engine,
)
from sqlalchemy.engine import Connection
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard import usage_queries as uq
from open_webui_openrouter_pipe.plugins.pipe_dashboard.usage_store import USAGE_ROW_FIELDS, UsageStore
from open_webui_openrouter_pipe.storage.persistence import ArtifactStore
from tests.test_usage_store import _install_persisted_collect_row


def _make_store_host(*, monkeypatch) -> Any:
    """This suite's own host, with the persisted collect valve carried as production does.

    `UsageStore._persist_sync` gates every batch on the persisted
    `PIPE_DASHBOARD_USAGE_COLLECT` row, read from the same engine it writes through --
    which on a deployment is Open WebUI's, where the `function` table lives. A stub
    engine without it is a shape no deployment has, and every aggregate below would be
    computed over an empty table because the writer refused rather than because the
    query is wrong.

    WARNING -- this twin helper is still on `StaticPool`, and `tests/test_usage_store.py`
    moved off it for a reason. `StaticPool` hands every thread the SAME DBAPI connection,
    and `_persist_sync` writes each row inside its own `begin_nested()`, so a row is a
    SAVEPOINT on that connection. A reader here that checks out and returns the
    connection issues a ROLLBACK that destroys a savepoint the live writer has open; the
    release raises "no such savepoint", the row is reported rejected and dropped, and the
    arm fails as a lost write. Nothing in THIS file fires that path -- the seeds below are
    written on the test's own thread, so no writer thread is live while a reader polls --
    which is exactly why the trap is still armed here. Do not add an arm that polls
    `_rows_in` while a `UsageStore` is committing; copy the file-backed engine from
    `tests/test_usage_store.py::_make_store_host` if you need to.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    host = SimpleNamespace(
        id="openrouter",
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
    # The guard delegates to these, so a stub host must carry them bound to itself.
    host._create_table_best_effort = MethodType(ArtifactStore._create_table_best_effort, host)
    host._create_declared_indexes = MethodType(ArtifactStore._create_declared_indexes, host)
    host._drop_superseded_indexes = MethodType(ArtifactStore._drop_superseded_indexes, host)
    _install_persisted_collect_row(host, True, monkeypatch)
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
def seeded(monkeypatch):
    host = _make_store_host(monkeypatch=monkeypatch)
    usage = UsageStore()
    assert usage.ensure(host)
    return host, usage


class _FakeClock:
    """A monotonic clock the test moves by hand, so no test ever really waits."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
