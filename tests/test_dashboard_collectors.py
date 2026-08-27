"""Tests for the tiered runtime stats collectors and worker payload aggregation.

Relocated from test_ephemeral_keys.py when the ephemeral-key SSE transport
was replaced by OWUI socket.io (the collector/aggregation code is unchanged).
"""

from __future__ import annotations

import time
from collections import deque
from unittest.mock import Mock, patch

import pytest
pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard.runtime_metrics import (
    collect_fast_stats,
    collect_identity,
    collect_medium_stats,
    collect_slow_stats,
)
from open_webui_openrouter_pipe.plugins.pipe_dashboard.dashboard_publisher import (
    _collect_worker_payload,
    aggregate_worker_payloads,
    expand_worker_payload,
)
from open_webui_openrouter_pipe.plugins.registry import PluginRegistry


# ── Runtime Metrics Tests ──


def _mock_registry(**attrs):
    """Mock OpenRouterModelRegistry with every field the collector reads."""
    reg = Mock()
    reg._models = attrs.pop("models", [])
    reg._specs = attrs.pop("specs", {})
    reg._consecutive_failures = attrs.pop("failures", 0)
    reg._last_fetch = attrs.pop("last_fetch", 0.0)
    reg._last_error = attrs.pop("last_error", None)
    reg._last_error_time = attrs.pop("last_error_time", 0.0)
    reg._zdr_model_ids = attrs.pop("zdr", None)
    reg._last_video_fetch = attrs.pop("video_fetch", 0.0)
    reg._last_image_fetch = attrs.pop("image_fetch", 0.0)
    reg._last_video_attempt = attrs.pop("video_attempt", 0.0)
    reg._last_image_attempt = attrs.pop("image_attempt", 0.0)
    for key, value in attrs.items():
        setattr(reg, key, value)
    return reg


def _make_mock_pipe():
    """Create a mock pipe with all runtime subsystems for tiered collectors."""
    pipe = Mock()
    pipe.id = "test-pipe"

    # Concurrency
    pipe._global_semaphore = Mock()
    pipe._global_semaphore._value = 45
    pipe._semaphore_limit = 50
    pipe._tool_global_semaphore = Mock()
    pipe._tool_global_semaphore._value = 8
    pipe._tool_global_limit = 10

    # Queues
    pipe._request_queue = Mock()
    pipe._request_queue.qsize.return_value = 3
    pipe._QUEUE_MAXSIZE = 1000
    pipe._log_queue = Mock()
    pipe._log_queue.qsize.return_value = 7

    # Session Log Manager
    pipe._session_log_manager = Mock()
    pipe._session_log_manager._worker_thread = Mock()
    pipe._session_log_manager._worker_thread.is_alive.return_value = True
    pipe._session_log_manager._queue = Mock()
    pipe._session_log_manager._queue.qsize.return_value = 2
    pipe._session_log_manager._retention_days = 30

    # Circuit Breaker
    pipe._circuit_breaker = Mock()
    pipe._circuit_breaker._threshold = 5
    pipe._circuit_breaker._window_seconds = 60.0
    pipe._circuit_breaker._breaker_records = {}
    pipe._circuit_breaker._tool_breakers = {}

    # Health
    pipe._initialized = True
    pipe._startup_checks_complete = True
    pipe._warmup_failed = False
    pipe._multimodal_handler = Mock()
    pipe._multimodal_handler.transport_session_state.return_value = "active"
    pipe._redis_enabled = False
    pipe._redis_client = None

    # Valves
    pipe.valves = Mock()
    pipe.valves.SESSION_LOG_STORE_ENABLED = True
    pipe.valves.DEFAULT_LLM_ENDPOINT = "https://openrouter.ai/api/v1"
    pipe.valves.BREAKER_MAX_FAILURES = 5
    pipe.valves.BREAKER_WINDOW_SECONDS = 60
    pipe.valves.ENABLE_TIMING_LOG = False
    pipe.valves.ARTIFACT_CLEANUP_INTERVAL_HOURS = 24
    pipe.valves.ARTIFACT_CLEANUP_DAYS = 30
    pipe.valves.SESSION_LOG_RETENTION_DAYS = 14
    pipe.valves.REDIS_CACHE_TTL_SECONDS = 300
    pipe.valves.STREAMING_IDLE_FLUSH_MS = 100

    # Artifact store
    pipe._artifact_store = None

    # Plugin registry
    pipe._plugin_registry = None

    # In-flight gauge + video pool
    pipe._active_pipes_calls = 0
    pipe._video_global_semaphore = None
    pipe._video_global_limit = 0
    pipe._video_active_tasks = {}

    return pipe


class TestCollectIdentity:
    def test_returns_identity_key(self):
        pipe = _make_mock_pipe()
        result = collect_identity(pipe)
        assert "identity" in result
        ident = result["identity"]
        assert "version" in ident
        assert ident["pipe_id"] == "test-pipe"
        assert ident["worker_count"] == 1

    def test_identity_custom_worker_count(self):
        pipe = _make_mock_pipe()
        result = collect_identity(pipe, worker_count=4)
        assert result["identity"]["worker_count"] == 4


class TestCollectFastStats:
    def test_basic_fields(self):
        pipe = _make_mock_pipe()
        stats = collect_fast_stats(pipe)
        assert "uptime_s" in stats
        assert "pid" in stats

    def test_concurrency_stats(self):
        pipe = _make_mock_pipe()
        stats = collect_fast_stats(pipe)
        c = stats["concurrency"]
        assert c["active_requests"] == 5  # 50 - 45
        assert c["max_requests"] == 50
        assert c["active_tools"] == 2  # 10 - 8
        assert c["max_tools"] == 10

    def test_queue_stats(self):
        pipe = _make_mock_pipe()
        stats = collect_fast_stats(pipe)
        q = stats["queues"]
        assert q["requests"] == 3
        assert q["requests_max"] == 1000
        assert q["logs"] == 7
        assert q["archive"] == 2

    def test_rate_limits_empty(self):
        pipe = _make_mock_pipe()
        stats = collect_fast_stats(pipe)
        rl = stats["rate_limits"]
        assert rl["tracked_users"] == 0
        assert rl["users_with_failures"] == 0
        assert rl["tripped_users"] == 0
        assert rl["threshold"] == 5
        assert rl["tool_tracked"] == 0
        assert rl["tool_tripped"] == 0

    def test_rate_limits_with_request_failures(self):
        pipe = _make_mock_pipe()
        now = time.time()
        # Two users with recent failures; one over threshold
        pipe._circuit_breaker._breaker_records = {
            "user1": deque([now - 10, now - 5, now - 2, now - 1, now]),  # 5 = threshold
            "user2": deque([now - 3]),  # 1 failure
            "user3": deque([now - 200]),  # Outside window (60s)
        }
        stats = collect_fast_stats(pipe)
        rl = stats["rate_limits"]
        assert rl["tracked_users"] == 3
        assert rl["users_with_failures"] == 2  # user1 + user2 have recent
        assert rl["tripped_users"] == 1  # user1 at threshold

    def test_rate_limits_with_tool_breakers(self):
        pipe = _make_mock_pipe()
        now = time.time()
        pipe._circuit_breaker._tool_breakers = {
            "user1": {
                "tool_a": deque([now - 1, now - 2, now - 3, now - 4, now - 5]),  # 5 = threshold
                "tool_b": deque([now - 10]),  # 1 failure
            },
        }
        stats = collect_fast_stats(pipe)
        rl = stats["rate_limits"]
        assert rl["tool_tracked"] == 2
        assert rl["tool_tripped"] == 1  # tool_a at threshold

    def test_rate_limits_auth_failures(self):
        pipe = _make_mock_pipe()
        now = time.time()
        from open_webui_openrouter_pipe.core.circuit_breaker import CircuitBreaker
        # Temporarily inject auth failures
        original = dict(CircuitBreaker._AUTH_FAILURE_UNTIL)
        try:
            CircuitBreaker._AUTH_FAILURE_UNTIL["user_x"] = now + 100  # active
            CircuitBreaker._AUTH_FAILURE_UNTIL["user_y"] = now - 10   # expired
            stats = collect_fast_stats(pipe)
            rl = stats["rate_limits"]
            assert rl["auth_failures_active"] >= 1  # at least user_x
        finally:
            CircuitBreaker._AUTH_FAILURE_UNTIL.clear()
            CircuitBreaker._AUTH_FAILURE_UNTIL.update(original)

    def test_sessions_stat(self):
        pipe = _make_mock_pipe()
        pipe._active_pipes_calls = 4
        stats = collect_fast_stats(pipe)
        assert stats["sessions"]["in_flight"] == 4

    def test_no_circuit_breaker(self):
        pipe = _make_mock_pipe()
        pipe._circuit_breaker = None
        stats = collect_fast_stats(pipe)
        rl = stats["rate_limits"]
        assert rl["tracked_users"] == 0
        assert rl["tripped_users"] == 0


class TestCollectMediumStats:
    def test_health_logging_enabled_from_valve(self):
        """BUG FIX: logging_enabled should come from valve, not queue existence."""
        pipe = _make_mock_pipe()
        pipe.valves.SESSION_LOG_STORE_ENABLED = True
        pipe._session_log_manager._queue = None  # queue is None but valve says enabled
        stats = collect_medium_stats(pipe)
        assert stats["health"]["logging_enabled"] is True

    def test_health_logging_disabled_from_valve(self):
        pipe = _make_mock_pipe()
        pipe.valves.SESSION_LOG_STORE_ENABLED = False
        pipe._session_log_manager._queue = Mock()  # queue exists but valve says disabled
        stats = collect_medium_stats(pipe)
        assert stats["health"]["logging_enabled"] is False

    def test_health_fields(self):
        pipe = _make_mock_pipe()
        stats = collect_medium_stats(pipe)
        h = stats["health"]
        assert h["initialized"] is True
        assert h["startup_complete"] is True
        assert h["warmup_failed"] is False
        assert h["http_session"] == "active"
        assert h["log_worker"] == "active"
        assert isinstance(h["log_buffers"], int)
        assert isinstance(h["log_events_buffered"], int)
        assert h["redis_enabled"] is False
        assert h["redis_connected"] is False

    def test_log_worker_tri_state(self):
        pipe = _make_mock_pipe()
        pipe.valves.SESSION_LOG_STORE_ENABLED = False
        assert collect_medium_stats(pipe)["health"]["log_worker"] == "disabled"
        pipe.valves.SESSION_LOG_STORE_ENABLED = True
        pipe._session_log_manager._worker_thread = None
        assert collect_medium_stats(pipe)["health"]["log_worker"] == "idle"
        pipe._session_log_manager._worker_thread = Mock()
        pipe._session_log_manager._worker_thread.is_alive.return_value = False
        assert collect_medium_stats(pipe)["health"]["log_worker"] == "stopped"

    def test_models_collect_failure_is_logged_not_silent(self, caplog):
        """A broken registry read must surface a warning, not a silently blank panel."""
        import logging as _logging

        pipe = _make_mock_pipe()
        with patch(
            "open_webui_openrouter_pipe.plugins.pipe_dashboard.runtime_metrics.collect_model_registry",
            side_effect=RuntimeError("registry exploded"),
        ), caplog.at_level(_logging.WARNING):
            stats = collect_medium_stats(pipe)
        assert stats["models"]["status"] == "unknown"
        warnings = [r for r in caplog.records if "model-registry stats unavailable" in r.getMessage()]
        assert len(warnings) == 1
        assert warnings[0].exc_info is not None

    def _patch_registry(self, **attrs):
        """Patch OpenRouterModelRegistry with given class attributes."""
        return patch(
            "open_webui_openrouter_pipe.models.registry.OpenRouterModelRegistry",
            **{f"_{k}" if not k.startswith("_") else k: v for k, v in attrs.items()},
        )

    def test_models_status_healthy(self):
        pipe = _make_mock_pipe()
        mock_reg = _mock_registry(models=[{"norm_id": f"m{i}"} for i in range(10)],
                                  last_fetch=time.time() - 30)
        with patch("open_webui_openrouter_pipe.models.registry.OpenRouterModelRegistry", mock_reg):
            stats = collect_medium_stats(pipe)
            assert stats["models"]["status"] == "healthy"
            assert stats["models"]["loaded"] == 10
            assert stats["models"]["text"] == 10
            assert stats["models"]["image"] == 0
            assert stats["models"]["video"] == 0

    def test_models_status_degraded(self):
        pipe = _make_mock_pipe()
        mock_reg = _mock_registry(models=[{"norm_id": f"m{i}"} for i in range(10)],
                                  failures=1, last_fetch=time.time() - 30, last_error="timeout")
        with patch("open_webui_openrouter_pipe.models.registry.OpenRouterModelRegistry", mock_reg):
            stats = collect_medium_stats(pipe)
            assert stats["models"]["status"] == "degraded"

    def test_models_status_failing(self):
        pipe = _make_mock_pipe()
        mock_reg = _mock_registry(models=[{"norm_id": f"m{i}"} for i in range(10)],
                                  failures=3, last_fetch=time.time() - 300,
                                  last_error="connection refused")
        with patch("open_webui_openrouter_pipe.models.registry.OpenRouterModelRegistry", mock_reg):
            stats = collect_medium_stats(pipe)
            assert stats["models"]["status"] == "failing"

    def test_models_status_pending_cold_boot(self):
        """Cold boot: no models loaded, no failures, no fetch yet → pending."""
        pipe = _make_mock_pipe()
        mock_reg = _mock_registry()
        with patch("open_webui_openrouter_pipe.models.registry.OpenRouterModelRegistry", mock_reg):
            stats = collect_medium_stats(pipe)
            assert stats["models"]["status"] == "pending"

    def test_models_status_failing_no_models_after_fetch(self):
        """No models loaded after a fetch attempt → failing (not pending)."""
        pipe = _make_mock_pipe()
        mock_reg = _mock_registry(last_fetch=time.time() - 60)
        with patch("open_webui_openrouter_pipe.models.registry.OpenRouterModelRegistry", mock_reg):
            stats = collect_medium_stats(pipe)
            assert stats["models"]["status"] == "failing"

    def test_models_type_breakdown_sums_to_loaded(self):
        """text+image+video == loaded; classification follows spec features."""
        pipe = _make_mock_pipe()
        models = [{"norm_id": "t1"}, {"norm_id": "i1"}, {"norm_id": "v1"}, {"norm_id": "t2"}]
        specs = {
            "i1": {"features": ["image_output"], "architecture": {"output_modalities": ["image"]}},
            "v1": {"features": ["video_generation"], "architecture": {}},
            "t2": {"features": ["image_output"], "architecture": {"output_modalities": ["text", "image"]}},
        }
        mock_reg = _mock_registry(models=models, specs=specs,
                                  last_fetch=time.time() - 30, zdr={"t1", "v1"})
        with patch("open_webui_openrouter_pipe.models.registry.OpenRouterModelRegistry", mock_reg):
            stats = collect_medium_stats(pipe)
            m = stats["models"]
            assert m["loaded"] == 4
            assert m["video"] == 1
            assert m["image"] == 1
            assert m["text"] == 2
            assert m["text"] + m["image"] + m["video"] == m["loaded"]
            assert m["zdr"] == 2

    def test_models_zdr_unavailable_is_none(self):
        pipe = _make_mock_pipe()
        mock_reg = _mock_registry(models=[{"norm_id": "m"}], last_fetch=time.time() - 5, zdr=None)
        with patch("open_webui_openrouter_pipe.models.registry.OpenRouterModelRegistry", mock_reg):
            assert collect_medium_stats(pipe)["models"]["zdr"] is None


class TestCollectSlowStats:
    def test_config_fields(self):
        pipe = _make_mock_pipe()
        stats = collect_slow_stats(pipe)
        cfg = stats["config"]
        assert "endpoint" in cfg
        assert "breaker" in cfg
        assert "timing_log" in cfg

    def test_plugins_empty(self):
        pipe = _make_mock_pipe()
        pipe._plugin_registry = None
        with patch.object(PluginRegistry, "_plugin_classes", []):
            stats = collect_slow_stats(pipe)
            assert stats["plugins"] == []

    def test_plugins_listed_from_class_registry(self):
        """Plugins come from the class-level registry (populated at import on
        every worker), not the lazily-created per-instance registry."""
        pipe = _make_mock_pipe()
        pipe._plugin_registry = None

        class _P:
            plugin_name = "Pipe Dashboard"
            plugin_id = "pipe-dashboard"
            plugin_version = "1.0"

        with patch.object(PluginRegistry, "_plugin_classes", [_P]):
            stats = collect_slow_stats(pipe)
            assert len(stats["plugins"]) == 1
            assert stats["plugins"][0]["name"] == "Pipe Dashboard"

    def test_plugins_instance_fallback(self):
        pipe = _make_mock_pipe()
        pr = Mock()
        p1 = Mock()
        p1.plugin_name = "Fallback Plugin"
        p1.plugin_id = "fb"
        p1.plugin_version = "0.1"
        pr._plugins = [p1]
        pipe._plugin_registry = pr
        with patch.object(PluginRegistry, "_plugin_classes", []):
            stats = collect_slow_stats(pipe)
            assert stats["plugins"][0]["name"] == "Fallback Plugin"

    def test_storage_not_connected(self):
        pipe = _make_mock_pipe()
        pipe._artifact_store = None
        stats = collect_slow_stats(pipe)
        assert stats["storage"]["connected"] is False

    def test_storage_connected_path_real_sqlite(self):
        """The connected-DB path against a real in-memory database — total,
        size, encrypted %, by_type and by_model must all populate."""
        import datetime

        from sqlalchemy import JSON, Boolean, Column, DateTime, String, create_engine
        from sqlalchemy.orm import declarative_base, sessionmaker

        base = declarative_base()

        class Item(base):
            __tablename__ = "artifacts_test"
            id = Column(String(26), primary_key=True)
            chat_id = Column(String(64))
            model_id = Column(String(128))
            item_type = Column(String(64))
            payload = Column(JSON)
            is_encrypted = Column(Boolean, default=False)
            created_at = Column(DateTime)

        engine = create_engine("sqlite:///:memory:")
        base.metadata.create_all(engine)
        sf = sessionmaker(bind=engine)
        now = datetime.datetime.now()
        with sf() as session:
            session.add(Item(id="a", chat_id="c1", model_id="m1", item_type="response",
                             payload={"x": "y"}, is_encrypted=True, created_at=now))
            session.add(Item(id="b", chat_id="c1", model_id="m1", item_type="response",
                             payload={"x": "zz"}, is_encrypted=False, created_at=now))
            session.add(Item(id="c", chat_id="c2", model_id="m2", item_type="upload",
                             payload={"k": 1}, is_encrypted=False, created_at=now))
            session.commit()

        store = Mock()
        store._session_factory = sf
        store._item_model = Item
        store._artifact_table_name = "artifacts_test"
        store._encryption_key = "k"
        store._encrypt_all = False
        store._compression_enabled = False
        store._compression_min_bytes = 0
        store._redis_enabled = False
        store._engine = engine
        pipe = _make_mock_pipe()
        pipe._artifact_store = store

        stats = collect_slow_stats(pipe)
        s = stats["storage"]
        assert s["connected"] is True
        assert s["state"] == "connected"
        assert s["total_items"] == "3"
        assert s["total_size"] != "-"
        assert s["encrypted_count"].startswith("1 ")
        assert len(s["by_type"]) == 2
        assert len(s["by_model"]) == 2
        assert s["encryption_mode"] == "Sensitive only"

    def test_storage_partial_columns_isolate_failures(self):
        """A model missing optional columns blanks only those fields —
        total_items still populates (per-query isolation)."""
        import datetime

        from sqlalchemy import Column, DateTime, String, create_engine
        from sqlalchemy.orm import declarative_base, sessionmaker

        base = declarative_base()

        class Item(base):
            __tablename__ = "artifacts_partial"
            id = Column(String(26), primary_key=True)
            item_type = Column(String(64))
            created_at = Column(DateTime)

        engine = create_engine("sqlite:///:memory:")
        base.metadata.create_all(engine)
        sf = sessionmaker(bind=engine)
        with sf() as session:
            session.add(Item(id="a", item_type="response", created_at=datetime.datetime.now()))
            session.commit()

        store = Mock()
        store._session_factory = sf
        store._item_model = Item
        store._artifact_table_name = "artifacts_partial"
        store._encryption_key = ""
        store._encrypt_all = False
        store._compression_enabled = False
        store._compression_min_bytes = 0
        store._redis_enabled = False
        store._engine = engine
        pipe = _make_mock_pipe()
        pipe._artifact_store = store

        s = collect_slow_stats(pipe)["storage"]
        assert s["total_items"] == "1"
        assert s["total_size"] == "-"
        assert s["encrypted_count"] == "-"
        assert s["state"] == "connected"

    def test_storage_ensure_called_when_not_connected(self):
        """The collector wires the store itself on an idle worker (system
        truth) instead of reporting a false 'Not connected'."""
        pipe = _make_mock_pipe()
        store = Mock()
        store._session_factory = None
        store._item_model = None
        store._encryption_key = ""
        store._encrypt_all = False
        store._compression_enabled = False
        store._compression_min_bytes = 0
        store._redis_enabled = False
        store._artifact_table_name = None
        pipe._artifact_store = store
        collect_slow_stats(pipe)
        store._ensure_artifact_store.assert_called_once()

    def test_storage_encryption_all_requires_key(self):
        pipe = _make_mock_pipe()
        store = Mock()
        store._session_factory = None
        store._item_model = None
        store._encryption_key = ""
        store._encrypt_all = True
        store._compression_enabled = False
        store._compression_min_bytes = 0
        store._redis_enabled = False
        store._artifact_table_name = None
        pipe._artifact_store = store
        s = collect_slow_stats(pipe)["storage"]
        assert s["encryption_mode"] == "Disabled"

    def test_storage_connected_no_db(self):
        pipe = _make_mock_pipe()
        store = Mock()
        store._session_factory = None
        store._item_model = None
        store._encryption_key = ""
        store._encrypt_all = False
        store._compression_enabled = False
        store._compression_min_bytes = 1024
        store._redis_enabled = False
        store._artifact_table_name = "artifacts"
        pipe._artifact_store = store
        stats = collect_slow_stats(pipe)
        s = stats["storage"]
        assert s["connected"] is False
        assert s["encryption_mode"] == "Disabled"
        assert s["compression_mode"] == "Disabled"


class TestGracefulNoneSubsystems:
    """All subsystems None should produce safe defaults across all tiers."""

    def test_fast_tier_none_subsystems(self):
        pipe = Mock()
        pipe._global_semaphore = None
        pipe._semaphore_limit = 0
        pipe._tool_global_semaphore = None
        pipe._tool_global_limit = 0
        pipe.valves = None  # No valves either — fallback returns 0
        pipe._request_queue = None
        pipe._QUEUE_MAXSIZE = 1000
        pipe._log_queue = None
        pipe._circuit_breaker = None
        pipe._session_log_manager = None
        stats = collect_fast_stats(pipe)
        assert stats["concurrency"]["active_requests"] == 0
        assert stats["concurrency"]["max_requests"] == 0
        assert stats["queues"]["requests"] == 0
        assert stats["rate_limits"]["tripped_users"] == 0

    def test_medium_tier_none_subsystems(self):
        pipe = Mock()
        pipe.valves = None
        pipe._session_log_manager = None
        pipe._multimodal_handler = None
        pipe._initialized = False
        pipe._startup_checks_complete = False
        pipe._warmup_failed = False
        pipe._redis_enabled = False
        pipe._redis_client = None
        stats = collect_medium_stats(pipe)
        assert stats["health"]["initialized"] is False
        assert stats["health"]["logging_enabled"] is False
        assert stats["health"]["log_worker"] == "disabled"

    def test_slow_tier_none_subsystems(self):
        pipe = Mock()
        pipe._artifact_store = None
        pipe.valves = None
        pipe._plugin_registry = None
        with patch.object(PluginRegistry, "_plugin_classes", []):
            stats = collect_slow_stats(pipe)
        assert stats["storage"]["connected"] is False
        assert stats["storage"]["state"] == "unavailable"
        assert stats["config"] == {}
        assert stats["plugins"] == []






class TestCollectWorkerPayload:
    def test_basic_fields(self):
        pipe = _make_mock_pipe()
        payload = _collect_worker_payload(pipe)
        assert "pid" in payload
        assert "up" in payload
        assert "c" in payload
        assert "q" in payload
        assert "rl" in payload
        assert "s" in payload

    def test_concurrency_compact_keys(self):
        pipe = _make_mock_pipe()
        payload = _collect_worker_payload(pipe)
        c = payload["c"]
        assert c["ar"] == 5  # 50 - 45
        assert c["mr"] == 50
        assert c["at"] == 2  # 10 - 8
        assert c["mt"] == 10

    def test_queues_compact_keys(self):
        pipe = _make_mock_pipe()
        payload = _collect_worker_payload(pipe)
        q = payload["q"]
        assert q["rq"] == 3
        assert q["rm"] == 1000
        assert q["lq"] == 7
        assert q["aq"] == 2

    def test_rate_limits_compact_keys(self):
        pipe = _make_mock_pipe()
        payload = _collect_worker_payload(pipe)
        rl = payload["rl"]
        assert rl["th"] == 5
        assert rl["ws"] == 60.0
        assert rl["tu"] == 0

    def test_none_subsystems(self):
        pipe = Mock()
        pipe._global_semaphore = None
        pipe._semaphore_limit = 0
        pipe._tool_global_semaphore = None
        pipe._tool_global_limit = 0
        pipe.valves = None  # No valves — fallback returns 0
        pipe._request_queue = None
        pipe._QUEUE_MAXSIZE = 1000
        pipe._log_queue = None
        pipe._session_log_manager = None
        pipe._circuit_breaker = None
        payload = _collect_worker_payload(pipe)
        assert payload["c"]["ar"] == 0
        assert payload["c"]["mr"] == 0
        assert payload["q"]["rq"] == 0
        assert payload["rl"]["tu"] == 0


class TestExpandWorkerPayload:
    def test_expands_all_fields(self):
        compact = {
            "pid": 1234,
            "up": 120.5,
            "c": {"ar": 3, "mr": 50, "at": 1, "mt": 10},
            "q": {"rq": 2, "rm": 1000, "lq": 5, "aq": 0},
            "rl": {"tu": 10, "fu": 3, "tr": 1, "th": 5, "ws": 60.0, "tt": 2, "tp": 0, "aa": 0},
            "s": 4,
        }
        expanded = expand_worker_payload(compact)
        assert expanded["pid"] == 1234
        assert expanded["uptime_s"] == 120.5
        assert expanded["concurrency"]["active_requests"] == 3
        assert expanded["concurrency"]["max_requests"] == 50
        assert expanded["queues"]["requests"] == 2
        assert expanded["queues"]["logs"] == 5
        assert expanded["rate_limits"]["tracked_users"] == 10
        assert expanded["rate_limits"]["tripped_users"] == 1
        assert expanded["sessions"]["in_flight"] == 4

    def test_handles_empty_payload(self):
        expanded = expand_worker_payload({})
        assert expanded["pid"] == 0
        assert expanded["uptime_s"] == 0
        assert expanded["concurrency"] == {}
        assert expanded["sessions"]["in_flight"] == 0
        assert expanded["videos"] == {}
        assert expanded["worker_health"] == {}

    def test_round_trips_new_compact_keys(self):
        compact = {
            "pid": 9, "up": 5.0, "ls": 1000,
            "q": {"rq": 1, "rm": 1000, "w": 7, "tw": 2, "lq": 0, "lm": 1000, "aq": 0, "am": 500},
            "rl": {"tf": 3},
            "v": {"a": 2, "m": 4},
            "s": 6,
            "h": {"init": 1, "wf": 0, "http": 1, "r": 0},
        }
        expanded = expand_worker_payload(compact)
        assert expanded["queues"]["waiting"] == 7
        assert expanded["queues"]["tool_waiting"] == 2
        assert expanded["queues"]["logs_max"] == 1000
        assert expanded["queues"]["archive_max"] == 500
        assert expanded["rate_limits"]["tool_with_failures"] == 3
        assert expanded["videos"] == {"active": 2, "max": 4}
        assert expanded["sessions"]["in_flight"] == 6
        assert expanded["worker_health"]["init"] == 1
        assert expanded["last_seen"] == 1000


class TestAggregateWorkerPayloads:
    def _make_worker(self, pid, active_req=0, active_tools=0, sessions=0, uptime=100.0):
        return {
            "pid": pid,
            "uptime_s": uptime,
            "concurrency": {
                "active_requests": active_req,
                "max_requests": 50,
                "active_tools": active_tools,
                "max_tools": 10,
            },
            "queues": {"requests": 1, "requests_max": 1000, "logs": 2, "archive": 0},
            "rate_limits": {
                "tracked_users": 5, "users_with_failures": 1, "tripped_users": 0,
                "threshold": 5, "window_s": 60.0,
                "tool_tracked": 2, "tool_tripped": 0, "auth_failures_active": 0,
            },
            "sessions": {"in_flight": sessions},
        }

    def test_empty_payloads(self):
        assert aggregate_worker_payloads([]) == {}

    def test_single_worker(self):
        w = self._make_worker(100, active_req=3, sessions=2)
        result = aggregate_worker_payloads([w])
        assert result["concurrency"]["active_requests"] == 3
        assert result["concurrency"]["max_requests"] == 50  # sum over 1 worker
        assert result["sessions"]["in_flight"] == 2
        assert len(result["workers"]) == 1
        assert result["workers"][0]["pid"] == 100

    def test_multi_worker_sums(self):
        w1 = self._make_worker(100, active_req=3, active_tools=1, sessions=2, uptime=500.0)
        w2 = self._make_worker(200, active_req=5, active_tools=2, sessions=1, uptime=300.0)
        w3 = self._make_worker(300, active_req=0, active_tools=0, sessions=0, uptime=100.0)
        result = aggregate_worker_payloads([w1, w2, w3])

        # Concurrency sums both actives and per-worker capacity limits
        assert result["concurrency"]["active_requests"] == 8  # 3+5+0
        assert result["concurrency"]["max_requests"] == 150  # 50+50+50
        assert result["concurrency"]["active_tools"] == 3  # 1+2+0
        assert result["concurrency"]["max_tools"] == 30  # 10+10+10

        # Queues sum across workers (depths and bounds)
        assert result["queues"]["requests"] == 3  # 1+1+1
        assert result["queues"]["requests_max"] == 3000  # 1000+1000+1000
        assert result["queues"]["logs"] == 6  # 2+2+2

        # Sessions sum
        assert result["sessions"]["in_flight"] == 3  # 2+1+0

        # Uptime = max across workers
        assert result["uptime_s"] == 500.0

        # Workers table sorted by PID
        assert len(result["workers"]) == 3
        assert result["workers"][0]["pid"] == 100
        assert result["workers"][1]["pid"] == 200
        assert result["workers"][2]["pid"] == 300
        assert result["workers"][0]["uptime_s"] == 500.0

    def test_rate_limits_sum(self):
        w1 = self._make_worker(100)
        w2 = self._make_worker(200)
        w1["rate_limits"]["tracked_users"] = 10
        w2["rate_limits"]["tracked_users"] = 8
        w1["rate_limits"]["tripped_users"] = 1
        w2["rate_limits"]["tripped_users"] = 2
        result = aggregate_worker_payloads([w1, w2])
        assert result["rate_limits"]["tracked_users"] == 18
        assert result["rate_limits"]["tripped_users"] == 3
        # Threshold and window from first worker
        assert result["rate_limits"]["threshold"] == 5
        assert result["rate_limits"]["window_s"] == 60.0

    def test_uptime_uses_max(self):
        """Aggregated uptime should use the longest-running worker."""
        w1 = self._make_worker(100, uptime=600.0)
        w2 = self._make_worker(200, uptime=120.0)
        result = aggregate_worker_payloads([w1, w2])
        assert result["uptime_s"] == 600.0

    def test_heterogeneous_skew_sums_and_max_config(self):
        """A worker on a different limit contributes its own value (sum, not
        first×n); a slice missing whole sections is skipped, not zeroed; and
        breaker config takes the max so an old-schema slice can't zero it."""
        w1 = self._make_worker(100)
        w2 = self._make_worker(200)
        w2["concurrency"]["max_requests"] = 80
        w2["rate_limits"]["threshold"] = 0
        w2["rate_limits"]["window_s"] = 0
        w3 = {"pid": 300, "uptime_s": 10.0}
        result = aggregate_worker_payloads([w1, w2, w3])
        assert result["concurrency"]["max_requests"] == 130  # 50+80, w3 skipped
        assert result["rate_limits"]["threshold"] == 5  # max, not first/zero
        assert result["rate_limits"]["window_s"] == 60.0
        assert len(result["workers"]) == 3

    def test_workers_carry_health_and_last_seen(self):
        w1 = self._make_worker(100, active_req=2)
        w1["last_seen"] = time.time() - 4
        w1["worker_health"] = {"init": 1, "wf": 0, "http": 1, "r": 1}
        result = aggregate_worker_payloads([w1])
        worker = result["workers"][0]
        assert worker["active_requests"] == 2
        assert worker["health"] == {"init": 1, "wf": 0, "http": 1, "r": 1}
        assert worker["last_seen_age"] is not None
        assert 3 <= worker["last_seen_age"] <= 6


class TestValveFallbackConcurrency:
    """When semaphores aren't initialized, collectors read valve config."""

    def test_fast_stats_reads_valve_limits(self):
        pipe = Mock()
        pipe._global_semaphore = None
        pipe._semaphore_limit = 0  # Not yet initialized
        pipe._tool_global_semaphore = None
        pipe._tool_global_limit = 0
        pipe.valves = Mock()
        pipe.valves.MAX_CONCURRENT_REQUESTS = 50
        pipe.valves.MAX_PARALLEL_TOOLS_GLOBAL = 10
        pipe._request_queue = None
        pipe._QUEUE_MAXSIZE = 1000
        pipe._log_queue = None
        pipe._circuit_breaker = None
        pipe._session_log_manager = None
        stats = collect_fast_stats(pipe)
        assert stats["concurrency"]["max_requests"] == 50
        assert stats["concurrency"]["max_tools"] == 10
        assert stats["concurrency"]["active_requests"] == 0

    def test_worker_payload_reads_valve_limits(self):
        pipe = Mock()
        pipe._global_semaphore = None
        pipe._semaphore_limit = 0
        pipe._tool_global_semaphore = None
        pipe._tool_global_limit = 0
        pipe.valves = Mock()
        pipe.valves.MAX_CONCURRENT_REQUESTS = 50
        pipe.valves.MAX_PARALLEL_TOOLS_GLOBAL = 10
        pipe._request_queue = None
        pipe._QUEUE_MAXSIZE = 1000
        pipe._log_queue = None
        pipe._circuit_breaker = None
        pipe._session_log_manager = None
        payload = _collect_worker_payload(pipe)
        assert payload["c"]["mr"] == 50
        assert payload["c"]["mt"] == 10


def test_unreadable_semaphore_does_not_blank_the_whole_fast_tier(pipe_instance, caplog):
    """A bad semaphore must cost one metric, not every panel on the dashboard."""
    import logging as _logging

    from open_webui_openrouter_pipe.plugins.pipe_dashboard import _collectors
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.runtime_metrics import (
        collect_fast_stats,
    )

    _collectors._warned_collectors.clear()
    pipe = pipe_instance

    class _Hostile:
        @property
        def _value(self):
            raise AttributeError("semaphore internals moved")

    pipe._global_semaphore = _Hostile()
    pipe._semaphore_limit = 4

    with caplog.at_level(_logging.WARNING):
        stats = collect_fast_stats(pipe)

    assert stats, "the entire fast tier was lost to one unreadable semaphore"
    assert "queues" in stats and "sessions" in stats, stats
    assert stats["concurrency"]["active_requests"] == 0
    assert stats["concurrency"]["max_requests"] == 4
    assert any("cannot read semaphore usage" in m for m in caplog.messages), caplog.messages


def test_collector_failures_warn_only_once(pipe_instance, caplog):
    """The collectors run every couple of seconds; the warning must not flood.

    Both halves of the latch, because only the pair pins the behaviour: one WARNING says
    the fault was reported, four DEBUGs say the repeats stayed visible to the operator who
    raises the log level to diagnose them. Counting warnings alone passes just as well
    when the repeat path goes silent, which is the defect `warn_level` exists to end.
    """
    import logging as _logging

    from open_webui_openrouter_pipe.plugins.pipe_dashboard import _collectors

    from tests.log_capture import emitted

    _collectors._warned_collectors.clear()

    class _Hostile:
        @property
        def _value(self):
            raise AttributeError("semaphore internals moved")

    with caplog.at_level(_logging.DEBUG):
        for _ in range(5):
            _collectors._semaphore_active(_Hostile(), 4)

    reported = "cannot read semaphore usage"
    warnings = emitted(caplog, min_level=_logging.WARNING, containing=reported)
    repeats = emitted(caplog, level=_logging.DEBUG, containing=reported)
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert len(repeats) == 4, [r.getMessage() for r in repeats]


class _HostileWaiters:
    """A semaphore whose `_waiters` raises rather than being absent."""

    @property
    def _waiters(self):
        raise TypeError("waiters unavailable")


class _HostileValue:
    """Not an int, and refuses to become one."""

    def __int__(self):
        raise ValueError("not a number")


class _HostileVideoPipe:
    _video_global_limit = 0
    _video_global_semaphore = None

    @property
    def _video_active_tasks(self):
        raise TypeError("task map unavailable")


class _HostileTransportHandler:
    """A multimodal handler whose transport state cannot be read."""

    def transport_session_state(self):
        raise RuntimeError("transport state unavailable")


class _HostileTransportPipe:
    _multimodal_handler = _HostileTransportHandler()


class _UnknownStateTransportHandler:
    """A handler reporting a state that is outside the dashboard's vocabulary."""

    def __init__(self, state: str = "idle") -> None:
        self.state = state

    def transport_session_state(self):
        return self.state


class _UnknownStateTransportPipe:
    def __init__(self, state: str = "idle") -> None:
        self._multimodal_handler = _UnknownStateTransportHandler(state)


@pytest.mark.asyncio
async def test_the_health_indicator_reads_the_session_the_process_actually_has(
    pipe_instance_async,
):
    """Three states, off the transport that exists.

    Both readers keyed on `pipe._http_session`, which is only ever assigned inside
    `_ensure_async_subsystems_initialized` -- a method with no non-test caller. So the
    dashboard's HTTP indicator was a constant on every worker: "none" in the panel and 0
    in the published slice, whatever the process was doing. The transport that DOES exist
    is the vetted one, a pool of its own that both readers were blind to.

    Driven through the real handler in all three states rather than asserted against a
    mock, because the defect was precisely that the attribute being read was never
    written. "Absent" is kept distinct from "closed": a worker that has not fetched
    anything yet is idle, not broken.
    """
    import contextlib

    from open_webui_openrouter_pipe.plugins.pipe_dashboard.dashboard_publisher import (
        _worker_health,
    )
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.runtime_metrics import (
        collect_medium_stats,
    )

    pipe = pipe_instance_async

    def _states():
        return (
            collect_medium_stats(pipe)["health"]["http_session"],
            _worker_health(pipe)["http"],
        )

    assert _states() == ("none", "none")

    with contextlib.suppress(Exception):
        await pipe._multimodal_handler._fetch_image_as_data_url(
            "https://cdn.example.com/icon.png"
        )
    assert _states() == ("active", "active"), _states()

    await pipe._multimodal_handler.aclose()
    assert _states() == ("closed", "closed"), _states()


@pytest.mark.parametrize("state", ["idle", "retiring"])
def test_a_state_the_dashboard_cannot_render_is_substituted_and_said_out_loud(
    state, caplog
):
    """`TRANSPORT_SESSION_STATES` is a second copy of a vocabulary the handler owns.

    Nothing links the two: `transport_session_state` gaining a fourth state would have
    been filtered to "none" here, and the panel would have rendered Idle for a worker
    that was doing something else, with no line in any log to notice it by. The filter
    itself has to stay -- `_worker_health(pipe)["http"]` is `json.dumps`-ed inside an
    `except Exception: logger.debug(...)`, so passing an unserialisable value through
    would silently drop the whole worker's slice -- so the fix is to announce the
    substitution, not to stop substituting.

    Two distinct unknown states, so a hardcoded "none" in the collector satisfies the
    return value and still fails the record. The offending value is asserted in the
    record's text rather than the message, because it travels as the raised value: a
    diagnostic that says a state was unknown without saying which one cannot be acted
    on.
    """
    import logging

    from open_webui_openrouter_pipe.plugins.pipe_dashboard import _collectors

    _collectors._warned_collectors.clear()
    try:
        with caplog.at_level(logging.WARNING, logger=_collectors.logger.name):
            verdict = _collectors.collect_transport_session_state(
                _UnknownStateTransportPipe(state)
            )
    finally:
        _collectors._warned_collectors.clear()

    records = [r for r in caplog.records if r.name == _collectors.logger.name]

    assert verdict == "none", (
        f"{state!r} reached the dashboard payload; a value outside the vocabulary is "
        "what json.dumps has to be protected from"
    )
    assert len(records) == 1, records
    assert state in caplog.text, (
        "the warning does not say which state was unknown, so nobody can tell whether "
        "the handler grew a state or the reader is broken"
    )


def _collector_causes() -> tuple[set[str], list[str]]:
    """Every `warn_level(_warned_collectors, ...)` cause key, read from the module's AST.

    Discovered rather than listed. The hand-written table held four of five: the fifth,
    `auth_failures`, was added by the same commit and guarded by nothing -- deleting its
    diagnostic outright left the whole suite green, and coverage showed its lines were
    executed by zero of 5862 tests. A table only makes an omission visible if something
    compares it to reality -- and only if the comparison can SEE every site, which is
    why the extractor is shared and reports the shapes it cannot reduce.
    """
    from open_webui_openrouter_pipe.plugins.pipe_dashboard import _collectors

    from tests.warn_latch_census import warn_level_causes

    return warn_level_causes(_collectors, "_warned_collectors")


# (driver, the value the collector substitutes when the read fails). The fallback is
# carried rather than assumed to be 0: `transport_session_state` reports a STRING, and a
# table hardcoding 0 could only cover it by leaving it out -- which is the omission this
# table exists to make visible.
_COLLECTOR_DRIVERS = {
    "safe_int": (lambda c: c._safe_int(_HostileValue()), 0),
    "waiter_count": (lambda c: c._waiter_count(_HostileWaiters()), 0),
    "semaphore_active": (lambda c: c._semaphore_active(_HostileValue(), 4), 0),
    "video_active": (lambda c: c.collect_video_pool(_HostileVideoPipe())["active"], 0),
    "auth_failures": (lambda c: _drive_hostile_auth_failures(c), 0),
    "transport_session_state": (
        lambda c: c.collect_transport_session_state(_HostileTransportPipe()),
        "none",
    ),
    "transport_session_domain": (
        lambda c: c.collect_transport_session_state(_UnknownStateTransportPipe()),
        "none",
    ),
}


def _drive_hostile_auth_failures(collectors):
    """Make the auth-failure read raise the way a corrupted breaker state would.

    `now < until` against a value that does not compare to a float raises TypeError,
    which is one of the four the collector catches. Restored on the way out because the
    breaker's tracking dict is a CLASS attribute shared by every test after this one.
    """
    from types import SimpleNamespace

    from open_webui_openrouter_pipe.core.circuit_breaker import CircuitBreaker

    # A real breaker: collect_rate_limits returns a zeroed dict and never reaches the
    # auth block when `_circuit_breaker` is falsy, so a hostile stub would drive nothing.
    pipe = SimpleNamespace(_circuit_breaker=CircuitBreaker(threshold=3, window_seconds=60.0))
    original = dict(CircuitBreaker._AUTH_FAILURE_UNTIL)
    CircuitBreaker._AUTH_FAILURE_UNTIL.clear()
    CircuitBreaker._AUTH_FAILURE_UNTIL["user"] = object()  # type: ignore[assignment]
    try:
        return collectors.collect_rate_limits(pipe)["auth_failures_active"]
    finally:
        CircuitBreaker._AUTH_FAILURE_UNTIL.clear()
        CircuitBreaker._AUTH_FAILURE_UNTIL.update(original)


def test_every_collector_fallback_has_a_driver():
    """A new collector without a driver is a red build, not a silent omission.

    Checked in both directions: a discovered cause with no driver is uncovered, and a
    driver naming a cause that no longer exists is a stale entry pointing at nothing.
    """
    from tests.warn_latch_census import UNRESOLVABLE_MESSAGE

    discovered, unresolvable = _collector_causes()
    assert not unresolvable, UNRESOLVABLE_MESSAGE + "\n  ".join(unresolvable)
    missing = sorted(discovered - set(_COLLECTOR_DRIVERS))
    assert not missing, (
        f"these collector fallbacks have no driver, so deleting their diagnostic is "
        f"invisible: {missing}. Add a driver rather than narrowing the table."
    )
    stale = sorted(set(_COLLECTOR_DRIVERS) - discovered)
    assert not stale, f"the driver table names causes that no longer exist: {stale}"


@pytest.mark.parametrize("label", sorted(_COLLECTOR_DRIVERS))
def test_each_collector_fallback_is_reported_once(label, caplog):
    """Substituting 0 for a metric that could not be read must be announced.

    One of these four was covered; the other three were written the same way in the
    same commit and were not. Deleting each warning block outright left the whole suite
    green, so the dashboard would report 0 for a broken metric with nothing anywhere
    saying the number is a placeholder.

    Parametrised rather than copied four times, because the defect IS that three
    siblings were left out of a pattern the fourth follows -- a table makes adding a
    fifth collector without its guard a visible omission.

    Asserted on records, not on message text: pinning the prose makes the test a
    spell-checker that goes red on a reword and still cannot see a latch that never
    lets the warning fire.
    """
    import logging

    from open_webui_openrouter_pipe.plugins.pipe_dashboard import _collectors

    call, fallback = _COLLECTOR_DRIVERS[label]
    _collectors._warned_collectors.clear()

    def _warnings():
        return [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == _collectors.logger.name
        ]

    try:
        with caplog.at_level(logging.WARNING, logger=_collectors.logger.name):
            assert call(_collectors) == fallback, (
                f"{label} did not fall back to {fallback!r}, so this test never reached "
                "the diagnostic it exists to check"
            )
            first = list(_warnings())
            for _ in range(4):
                call(_collectors)
            total = list(_warnings())
    finally:
        _collectors._warned_collectors.clear()

    assert len(first) == 1, (
        f"{label} substituted {fallback!r} and emitted {len(first)} warnings; the "
        "dashboard would show a placeholder with no indication it is one"
    )
    assert len(total) == 1, (
        f"{label} warned {len(total)} times over five calls. The latch is not holding, "
        "so a persistently unreadable metric floods the log on every dashboard tick."
    )
    assert first[0].exc_info is not None, (
        f"{label}'s warning carries no traceback, so it says a metric failed without "
        "saying why"
    )


def test_an_unreadable_data_volume_is_reported_once_and_names_the_path(caplog, monkeypatch):
    """Polled every ~16s while a viewer is connected, so it must not flood.

    And it must say WHICH volume failed. The first version interpolated
    `out.get("disk_path")`, which is assigned only after the read it reports on
    succeeds, so every message read "for the data directory" and named nothing.
    """
    import logging as _logging

    from open_webui_openrouter_pipe.plugins.pipe_dashboard import runtime_metrics as rm

    monkeypatch.setattr(rm.os, "getcwd", lambda: "/nonexistent-volume-xyz")

    def _boom(_path):
        raise FileNotFoundError("volume is not mounted")

    monkeypatch.setattr(rm.shutil, "disk_usage", _boom)

    out: dict = {}
    with caplog.at_level(_logging.DEBUG, logger=rm.logger.name):
        for _ in range(4):
            out = rm.collect_system_resources()

    disk = [r for r in caplog.records if "cannot read disk usage" in r.getMessage()]
    assert [r.levelno for r in disk] == [
        _logging.WARNING, _logging.DEBUG, _logging.DEBUG, _logging.DEBUG,
    ], [(r.levelname, r.getMessage()) for r in disk]
    assert "/nonexistent-volume-xyz" in disk[0].getMessage(), (
        f"the message named no path: {disk[0].getMessage()!r}"
    )
    assert disk[0].exc_info is not None
    assert not any("failed to import" in r.getMessage() for r in caplog.records)
    assert "disk_free" not in out
