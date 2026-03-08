"""Tests for the EphemeralKeyStore and tiered runtime stats collectors."""

from __future__ import annotations

import time
from collections import deque
from unittest.mock import Mock, patch

import pytest

from open_webui_openrouter_pipe.plugins.pipe_stats.ephemeral_keys import (
    EphemeralKeyStore,
    _PS_EK_DEFAULT_TTL,
    _PS_EK_MAX_KEYS,
)
from open_webui_openrouter_pipe.plugins.pipe_stats.runtime_stats import (
    collect_fast_stats,
    collect_identity,
    collect_medium_stats,
    collect_slow_stats,
)


# ── EphemeralKeyStore Tests ──


class TestEphemeralKeyStore:
    def test_generate_returns_hex_key(self):
        store = EphemeralKeyStore()
        key = store.generate()
        assert isinstance(key, str)
        assert len(key) == 64
        # Must be valid hex
        int(key, 16)

    def test_validate_valid_key(self):
        store = EphemeralKeyStore()
        key = store.generate()
        assert store.validate(key) is True

    def test_validate_invalid_key(self):
        store = EphemeralKeyStore()
        assert store.validate("nonexistent") is False

    def test_validate_touches_ttl(self):
        """Validating a key should refresh its last-access timestamp (keep-alive)."""
        store = EphemeralKeyStore(ttl=100)
        key = store.generate()
        # Record initial timestamp
        ts_before = store._keys[key]
        # Advance time slightly
        with patch("open_webui_openrouter_pipe.plugins.pipe_stats.ephemeral_keys.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 10
            store.validate(key)
            # Timestamp should be updated to the new time
            assert store._keys[key] == mock_time.monotonic.return_value

    def test_revoke(self):
        store = EphemeralKeyStore()
        key = store.generate()
        store.revoke(key)
        assert store.validate(key) is False

    def test_revoke_nonexistent_key_is_noop(self):
        store = EphemeralKeyStore()
        store.revoke("not-a-key")  # Should not raise

    def test_expiry(self):
        """Keys idle longer than TTL should be purged."""
        store = EphemeralKeyStore(ttl=1.0)
        key = store.generate()
        # Manually set timestamp to far in the past
        store._keys[key] = time.monotonic() - 10
        assert store.validate(key) is False

    def test_active_count(self):
        store = EphemeralKeyStore()
        assert store.active_count == 0
        store.generate()
        assert store.active_count == 1
        store.generate()
        assert store.active_count == 2

    def test_max_keys_eviction(self):
        """When at capacity, oldest key should be evicted."""
        store = EphemeralKeyStore()
        keys = [store.generate() for _ in range(_PS_EK_MAX_KEYS)]
        assert store.active_count == _PS_EK_MAX_KEYS
        # Generate one more — oldest should be evicted
        new_key = store.generate()
        assert store.active_count == _PS_EK_MAX_KEYS
        assert store.validate(new_key) is True
        assert store.validate(keys[0]) is False  # Oldest was evicted

    def test_cleanup_purges_expired(self):
        """Expired keys should be cleaned up automatically."""
        store = EphemeralKeyStore(ttl=1.0)
        k1 = store.generate()
        k2 = store.generate()
        # Expire k1
        store._keys[k1] = time.monotonic() - 10
        # validate() triggers cleanup
        assert store.validate(k2) is True
        assert k1 not in store._keys

    def test_default_ttl(self):
        store = EphemeralKeyStore()
        assert store._ttl == _PS_EK_DEFAULT_TTL

    def test_minimum_ttl_is_one(self):
        store = EphemeralKeyStore(ttl=0.1)
        assert store._ttl == 1.0

    def test_unique_keys(self):
        """Each generated key should be unique."""
        store = EphemeralKeyStore()
        keys = {store.generate() for _ in range(10)}
        assert len(keys) == 10


# ── Runtime Stats Tests ──


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
    pipe._http_session = Mock()
    pipe._http_session.closed = False
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
        mock_sl = Mock()
        mock_sl.logs = {"s1": Mock(), "s2": Mock()}
        with patch("open_webui_openrouter_pipe.core.logging_system.SessionLogger", mock_sl):
            stats = collect_fast_stats(pipe)
            assert stats["sessions"]["active"] == 2

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
        assert h["log_worker_alive"] is True
        assert h["redis_enabled"] is False
        assert h["redis_connected"] is False

    def _patch_registry(self, **attrs):
        """Patch OpenRouterModelRegistry with given class attributes."""
        return patch(
            "open_webui_openrouter_pipe.models.registry.OpenRouterModelRegistry",
            **{f"_{k}" if not k.startswith("_") else k: v for k, v in attrs.items()},
        )

    def test_models_status_healthy(self):
        pipe = _make_mock_pipe()
        mock_reg = Mock()
        mock_reg._models = [Mock()] * 10
        mock_reg._consecutive_failures = 0
        mock_reg._last_fetch = time.time() - 30
        mock_reg._last_error = None
        mock_reg._specs = {}
        with patch("open_webui_openrouter_pipe.models.registry.OpenRouterModelRegistry", mock_reg):
            stats = collect_medium_stats(pipe)
            assert stats["models"]["status"] == "healthy"
            assert stats["models"]["loaded"] == 10

    def test_models_status_degraded(self):
        pipe = _make_mock_pipe()
        mock_reg = Mock()
        mock_reg._models = [Mock()] * 10
        mock_reg._consecutive_failures = 1
        mock_reg._last_fetch = time.time() - 30
        mock_reg._last_error = "timeout"
        mock_reg._specs = {}
        with patch("open_webui_openrouter_pipe.models.registry.OpenRouterModelRegistry", mock_reg):
            stats = collect_medium_stats(pipe)
            assert stats["models"]["status"] == "degraded"

    def test_models_status_failing(self):
        pipe = _make_mock_pipe()
        mock_reg = Mock()
        mock_reg._models = [Mock()] * 10
        mock_reg._consecutive_failures = 3
        mock_reg._last_fetch = time.time() - 300
        mock_reg._last_error = "connection refused"
        mock_reg._specs = {}
        with patch("open_webui_openrouter_pipe.models.registry.OpenRouterModelRegistry", mock_reg):
            stats = collect_medium_stats(pipe)
            assert stats["models"]["status"] == "failing"

    def test_models_status_pending_cold_boot(self):
        """Cold boot: no models loaded, no failures, no fetch yet → pending."""
        pipe = _make_mock_pipe()
        mock_reg = Mock()
        mock_reg._models = []
        mock_reg._consecutive_failures = 0
        mock_reg._last_fetch = 0.0
        mock_reg._last_error = None
        mock_reg._specs = {}
        with patch("open_webui_openrouter_pipe.models.registry.OpenRouterModelRegistry", mock_reg):
            stats = collect_medium_stats(pipe)
            assert stats["models"]["status"] == "pending"

    def test_models_status_failing_no_models_after_fetch(self):
        """No models loaded after a fetch attempt → failing (not pending)."""
        pipe = _make_mock_pipe()
        mock_reg = Mock()
        mock_reg._models = []
        mock_reg._consecutive_failures = 0
        mock_reg._last_fetch = time.time() - 60
        mock_reg._last_error = None
        mock_reg._specs = {}
        with patch("open_webui_openrouter_pipe.models.registry.OpenRouterModelRegistry", mock_reg):
            stats = collect_medium_stats(pipe)
            assert stats["models"]["status"] == "failing"


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
        stats = collect_slow_stats(pipe)
        assert stats["plugins"] == []

    def test_plugins_listed(self):
        pipe = _make_mock_pipe()
        pr = Mock()
        p1 = Mock()
        p1.plugin_name = "Pipe Stats Dashboard"
        p1.plugin_id = "pipe-stats"
        p1.plugin_version = "1.0"
        pr._plugins = [p1]
        pipe._plugin_registry = pr
        stats = collect_slow_stats(pipe)
        assert len(stats["plugins"]) == 1
        assert stats["plugins"][0]["name"] == "Pipe Stats Dashboard"

    def test_storage_not_connected(self):
        pipe = _make_mock_pipe()
        pipe._artifact_store = None
        stats = collect_slow_stats(pipe)
        assert stats["storage"]["connected"] is False

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
        pipe._http_session = None
        pipe._initialized = False
        pipe._startup_checks_complete = False
        pipe._warmup_failed = False
        pipe._redis_enabled = False
        pipe._redis_client = None
        stats = collect_medium_stats(pipe)
        assert stats["health"]["initialized"] is False
        assert stats["health"]["logging_enabled"] is False
        assert stats["health"]["log_worker_alive"] is False

    def test_slow_tier_none_subsystems(self):
        pipe = Mock()
        pipe._artifact_store = None
        pipe.valves = None
        pipe._plugin_registry = None
        stats = collect_slow_stats(pipe)
        assert stats["storage"]["connected"] is False
        assert stats["config"] == {}
        assert stats["plugins"] == []


# ── Stats Publisher Tests ──


from open_webui_openrouter_pipe.plugins.pipe_stats.stats_publisher import (
    _collect_worker_payload,
    aggregate_worker_payloads,
    expand_worker_payload,
)


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
        assert expanded["sessions"]["active"] == 4

    def test_handles_empty_payload(self):
        expanded = expand_worker_payload({})
        assert expanded["pid"] == 0
        assert expanded["uptime_s"] == 0
        assert expanded["concurrency"] == {}
        assert expanded["sessions"]["active"] == 0


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
            "sessions": {"active": sessions},
        }

    def test_empty_payloads(self):
        assert aggregate_worker_payloads([]) == {}

    def test_single_worker(self):
        w = self._make_worker(100, active_req=3, sessions=2)
        result = aggregate_worker_payloads([w])
        assert result["concurrency"]["active_requests"] == 3
        assert result["concurrency"]["max_requests"] == 50  # 50 × 1 worker
        assert result["sessions"]["active"] == 2
        assert len(result["workers"]) == 1
        assert result["workers"][0]["pid"] == 100

    def test_multi_worker_sums(self):
        w1 = self._make_worker(100, active_req=3, active_tools=1, sessions=2, uptime=500.0)
        w2 = self._make_worker(200, active_req=5, active_tools=2, sessions=1, uptime=300.0)
        w3 = self._make_worker(300, active_req=0, active_tools=0, sessions=0, uptime=100.0)
        result = aggregate_worker_payloads([w1, w2, w3])

        # Concurrency sums active, max × worker count
        assert result["concurrency"]["active_requests"] == 8  # 3+5+0
        assert result["concurrency"]["max_requests"] == 150  # 50 × 3 workers
        assert result["concurrency"]["active_tools"] == 3  # 1+2+0
        assert result["concurrency"]["max_tools"] == 30  # 10 × 3 workers

        # Queues sum across workers, max × worker count
        assert result["queues"]["requests"] == 3  # 1+1+1
        assert result["queues"]["requests_max"] == 3000  # 1000 × 3 workers
        assert result["queues"]["logs"] == 6  # 2+2+2

        # Sessions sum
        assert result["sessions"]["active"] == 3  # 2+1+0

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
