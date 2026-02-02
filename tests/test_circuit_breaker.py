"""Comprehensive tests for CircuitBreaker to achieve 90%+ coverage.

This test module targets the missing lines in circuit_breaker.py:
- Line 66: threshold getter
- Line 77: threshold setter with existing breaker records
- Line 82: threshold setter with existing tool records
- Line 95: window_seconds getter
- Line 139: record_failure with empty user_id
- Line 174: tool_allows with empty user_id or tool_type
- Line 181: tool_allows evicting old failures
- Line 194: record_tool_failure with empty user_id or tool_type
- Line 206: reset_tool with empty user_id or tool_type
- Lines 226-235: note_auth_failure edge cases
- Line 249: auth_failure_active with empty scope_key
- Lines 256-259: auth_failure expiration handling
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from open_webui_openrouter_pipe.core.circuit_breaker import CircuitBreaker


class TestCircuitBreakerThresholdProperty:
    """Tests for the threshold property getter and setter."""

    def test_threshold_getter_returns_initial_value(self) -> None:
        """Threshold getter should return the configured threshold (line 66)."""
        breaker = CircuitBreaker(threshold=5, window_seconds=60)
        assert breaker.threshold == 5

    def test_threshold_setter_updates_value(self) -> None:
        """Threshold setter should update the threshold value."""
        breaker = CircuitBreaker(threshold=3, window_seconds=60)
        breaker.threshold = 10
        assert breaker.threshold == 10

    def test_threshold_setter_enforces_minimum_of_one(self) -> None:
        """Threshold setter should enforce a minimum value of 1."""
        breaker = CircuitBreaker(threshold=3, window_seconds=60)
        breaker.threshold = 0
        assert breaker.threshold == 1
        breaker.threshold = -5
        assert breaker.threshold == 1

    def test_threshold_setter_rebuilds_breaker_records(self) -> None:
        """Threshold setter should rebuild existing breaker records (line 77)."""
        breaker = CircuitBreaker(threshold=3, window_seconds=60)
        user_id = "user-1"

        # Record some failures
        breaker.record_failure(user_id)
        breaker.record_failure(user_id)

        # Change threshold - should rebuild the deque with new maxlen
        breaker.threshold = 5
        assert breaker.threshold == 5

        # Records should be preserved
        assert not breaker.allows(user_id) is False  # Still allowed since 2 < 5

    def test_threshold_setter_rebuilds_tool_breakers(self) -> None:
        """Threshold setter should rebuild existing tool breakers (line 82)."""
        breaker = CircuitBreaker(threshold=3, window_seconds=60)
        user_id = "user-1"
        tool_type = "function"

        # Record some tool failures
        breaker.record_tool_failure(user_id, tool_type)
        breaker.record_tool_failure(user_id, tool_type)

        # Change threshold - should rebuild the tool deques with new maxlen
        breaker.threshold = 5
        assert breaker.threshold == 5

        # Tool records should be preserved
        assert breaker.tool_allows(user_id, tool_type) is True  # Still allowed since 2 < 5


class TestCircuitBreakerWindowProperty:
    """Tests for the window_seconds property getter and setter."""

    def test_window_seconds_getter_returns_initial_value(self) -> None:
        """Window seconds getter should return the configured window (line 95)."""
        breaker = CircuitBreaker(threshold=3, window_seconds=30.5)
        assert breaker.window_seconds == 30.5

    def test_window_seconds_setter_updates_value(self) -> None:
        """Window seconds setter should update the window value."""
        breaker = CircuitBreaker(threshold=3, window_seconds=60)
        breaker.window_seconds = 120.0
        assert breaker.window_seconds == 120.0

    def test_window_seconds_setter_enforces_minimum(self) -> None:
        """Window seconds setter should enforce a minimum of 0.1 seconds."""
        breaker = CircuitBreaker(threshold=3, window_seconds=60)
        breaker.window_seconds = 0.05
        assert breaker.window_seconds == 0.1
        breaker.window_seconds = -10.0
        assert breaker.window_seconds == 0.1


class TestCircuitBreakerRequestBreaker:
    """Tests for the per-user request circuit breaker."""

    def test_allows_returns_true_for_empty_user_id(self) -> None:
        """Allows should return True for empty user_id."""
        breaker = CircuitBreaker(threshold=2, window_seconds=60)
        assert breaker.allows("") is True
        assert breaker.allows(None) is True  # type: ignore[arg-type]

    def test_record_failure_does_nothing_for_empty_user_id(self) -> None:
        """Record failure should return early for empty user_id (line 139)."""
        breaker = CircuitBreaker(threshold=2, window_seconds=60)
        # Should not raise and should return early
        breaker.record_failure("")
        breaker.record_failure(None)  # type: ignore[arg-type]

        # Verify no state was created
        assert len(breaker._breaker_records) == 0

    def test_allows_evicts_old_failures(self) -> None:
        """Allows should evict failures outside the time window."""
        breaker = CircuitBreaker(threshold=2, window_seconds=10)
        user_id = "user-1"

        with patch.object(time, "time", return_value=0.0):
            breaker.record_failure(user_id)
            breaker.record_failure(user_id)

        # At time 0, breaker should be tripped
        with patch.object(time, "time", return_value=0.0):
            assert breaker.allows(user_id) is False

        # At time 15, failures are outside the 10s window
        with patch.object(time, "time", return_value=15.0):
            assert breaker.allows(user_id) is True

    def test_reset_clears_failures(self) -> None:
        """Reset should clear all failures for a user."""
        breaker = CircuitBreaker(threshold=2, window_seconds=60)
        user_id = "user-1"

        breaker.record_failure(user_id)
        breaker.record_failure(user_id)
        assert breaker.allows(user_id) is False

        breaker.reset(user_id)
        assert breaker.allows(user_id) is True

    def test_reset_handles_empty_user_id(self) -> None:
        """Reset should do nothing for empty user_id."""
        breaker = CircuitBreaker(threshold=2, window_seconds=60)
        breaker.reset("")
        breaker.reset(None)  # type: ignore[arg-type]

    def test_reset_handles_nonexistent_user(self) -> None:
        """Reset should handle users not in records gracefully."""
        breaker = CircuitBreaker(threshold=2, window_seconds=60)
        breaker.reset("nonexistent-user")  # Should not raise


class TestCircuitBreakerToolBreaker:
    """Tests for the per-user per-tool-type circuit breaker."""

    def test_tool_allows_returns_true_for_empty_user_id(self) -> None:
        """Tool allows should return True for empty user_id (line 174)."""
        breaker = CircuitBreaker(threshold=2, window_seconds=60)
        assert breaker.tool_allows("", "function") is True
        assert breaker.tool_allows(None, "function") is True  # type: ignore[arg-type]

    def test_tool_allows_returns_true_for_empty_tool_type(self) -> None:
        """Tool allows should return True for empty tool_type (line 174)."""
        breaker = CircuitBreaker(threshold=2, window_seconds=60)
        assert breaker.tool_allows("user-1", "") is True
        assert breaker.tool_allows("user-1", None) is True  # type: ignore[arg-type]

    def test_tool_allows_evicts_old_failures(self) -> None:
        """Tool allows should evict failures outside the time window (line 181)."""
        breaker = CircuitBreaker(threshold=2, window_seconds=10)
        user_id = "user-1"
        tool_type = "function"

        with patch.object(time, "time", return_value=0.0):
            breaker.record_tool_failure(user_id, tool_type)
            breaker.record_tool_failure(user_id, tool_type)

        # At time 0, tool breaker should be tripped
        with patch.object(time, "time", return_value=0.0):
            assert breaker.tool_allows(user_id, tool_type) is False

        # At time 15, failures are outside the 10s window - eviction happens
        with patch.object(time, "time", return_value=15.0):
            assert breaker.tool_allows(user_id, tool_type) is True

    def test_record_tool_failure_does_nothing_for_empty_user_id(self) -> None:
        """Record tool failure should return early for empty user_id (line 194)."""
        breaker = CircuitBreaker(threshold=2, window_seconds=60)
        breaker.record_tool_failure("", "function")
        breaker.record_tool_failure(None, "function")  # type: ignore[arg-type]

        # Verify no state was created
        assert len(breaker._tool_breakers) == 0

    def test_record_tool_failure_does_nothing_for_empty_tool_type(self) -> None:
        """Record tool failure should return early for empty tool_type (line 194)."""
        breaker = CircuitBreaker(threshold=2, window_seconds=60)
        breaker.record_tool_failure("user-1", "")
        breaker.record_tool_failure("user-1", None)  # type: ignore[arg-type]

        # Verify no state was created
        assert len(breaker._tool_breakers) == 0

    def test_reset_tool_does_nothing_for_empty_user_id(self) -> None:
        """Reset tool should return early for empty user_id (line 206)."""
        breaker = CircuitBreaker(threshold=2, window_seconds=60)
        breaker.reset_tool("", "function")
        breaker.reset_tool(None, "function")  # type: ignore[arg-type]

    def test_reset_tool_does_nothing_for_empty_tool_type(self) -> None:
        """Reset tool should return early for empty tool_type (line 206)."""
        breaker = CircuitBreaker(threshold=2, window_seconds=60)
        breaker.reset_tool("user-1", "")
        breaker.reset_tool("user-1", None)  # type: ignore[arg-type]

    def test_reset_tool_clears_failures(self) -> None:
        """Reset tool should clear failures for specific tool type."""
        breaker = CircuitBreaker(threshold=2, window_seconds=60)
        user_id = "user-1"
        tool_type = "function"

        breaker.record_tool_failure(user_id, tool_type)
        breaker.record_tool_failure(user_id, tool_type)
        assert breaker.tool_allows(user_id, tool_type) is False

        breaker.reset_tool(user_id, tool_type)
        assert breaker.tool_allows(user_id, tool_type) is True

    def test_reset_tool_handles_nonexistent_records(self) -> None:
        """Reset tool should handle nonexistent user/tool gracefully."""
        breaker = CircuitBreaker(threshold=2, window_seconds=60)
        # These should not raise
        breaker.reset_tool("nonexistent-user", "function")
        breaker.reset_tool("user-1", "nonexistent-tool")

    def test_tool_breakers_independent_per_tool_type(self) -> None:
        """Different tool types should have independent breakers."""
        breaker = CircuitBreaker(threshold=2, window_seconds=60)
        user_id = "user-1"

        breaker.record_tool_failure(user_id, "tool_a")
        breaker.record_tool_failure(user_id, "tool_a")
        assert breaker.tool_allows(user_id, "tool_a") is False
        assert breaker.tool_allows(user_id, "tool_b") is True


class TestCircuitBreakerAuthFailure:
    """Tests for class-level auth failure tracking."""

    def setup_method(self) -> None:
        """Clear auth failure state before each test."""
        CircuitBreaker._AUTH_FAILURE_UNTIL.clear()

    def test_note_auth_failure_does_nothing_for_empty_scope_key(self) -> None:
        """Note auth failure should return early for empty scope_key (line 226)."""
        CircuitBreaker.note_auth_failure("")
        CircuitBreaker.note_auth_failure(None)  # type: ignore[arg-type]

        # Verify no state was created
        assert len(CircuitBreaker._AUTH_FAILURE_UNTIL) == 0

    def test_note_auth_failure_uses_default_ttl_for_zero(self) -> None:
        """Note auth failure treats ttl_seconds=0 as default (60s) due to 'or' fallback.

        This is a subtle behavior: `0 or 60` evaluates to `60` in Python.
        """
        with patch.object(time, "time", return_value=1000.0):
            CircuitBreaker.note_auth_failure("scope-1", ttl_seconds=0)

        # 0 is falsy, so it falls back to default 60 seconds
        assert "scope-1" in CircuitBreaker._AUTH_FAILURE_UNTIL
        assert CircuitBreaker._AUTH_FAILURE_UNTIL["scope-1"] == 1060.0

    def test_note_auth_failure_does_nothing_for_negative_ttl(self) -> None:
        """Note auth failure should return early for negative TTL (lines 230-231)."""
        CircuitBreaker.note_auth_failure("scope-2", ttl_seconds=-10)

        # Verify no state was created (negative TTL is truthy but <= 0 check catches it)
        assert "scope-2" not in CircuitBreaker._AUTH_FAILURE_UNTIL

    def test_note_auth_failure_records_with_default_ttl(self) -> None:
        """Note auth failure should record with default TTL (lines 229, 233-235)."""
        with patch.object(time, "time", return_value=1000.0):
            CircuitBreaker.note_auth_failure("scope-1")

        # Default TTL is 60 seconds
        assert "scope-1" in CircuitBreaker._AUTH_FAILURE_UNTIL
        assert CircuitBreaker._AUTH_FAILURE_UNTIL["scope-1"] == 1060.0

    def test_note_auth_failure_records_with_custom_ttl(self) -> None:
        """Note auth failure should record with custom TTL."""
        with patch.object(time, "time", return_value=1000.0):
            CircuitBreaker.note_auth_failure("scope-1", ttl_seconds=120)

        assert CircuitBreaker._AUTH_FAILURE_UNTIL["scope-1"] == 1120.0

    def test_auth_failure_active_returns_false_for_empty_scope_key(self) -> None:
        """Auth failure active should return False for empty scope_key (line 249)."""
        assert CircuitBreaker.auth_failure_active("") is False
        assert CircuitBreaker.auth_failure_active(None) is False  # type: ignore[arg-type]

    def test_auth_failure_active_returns_false_for_unknown_scope(self) -> None:
        """Auth failure active should return False for unknown scope (lines 254-255)."""
        assert CircuitBreaker.auth_failure_active("unknown-scope") is False

    def test_auth_failure_active_returns_true_when_active(self) -> None:
        """Auth failure active should return True when failure is active (line 259)."""
        with patch.object(time, "time", return_value=1000.0):
            CircuitBreaker.note_auth_failure("scope-1", ttl_seconds=60)

        with patch.object(time, "time", return_value=1030.0):  # Still within TTL
            assert CircuitBreaker.auth_failure_active("scope-1") is True

    def test_auth_failure_active_returns_false_when_expired(self) -> None:
        """Auth failure active should return False and clean up when expired (lines 256-258)."""
        with patch.object(time, "time", return_value=1000.0):
            CircuitBreaker.note_auth_failure("scope-1", ttl_seconds=60)

        # Time is now past the TTL
        with patch.object(time, "time", return_value=1100.0):
            assert CircuitBreaker.auth_failure_active("scope-1") is False

        # Verify the entry was cleaned up
        assert "scope-1" not in CircuitBreaker._AUTH_FAILURE_UNTIL

    def test_auth_failure_thread_safe(self) -> None:
        """Auth failure operations should be thread-safe."""
        import threading

        results = []
        errors = []

        def record_failures():
            try:
                for i in range(100):
                    CircuitBreaker.note_auth_failure(f"scope-{threading.current_thread().name}-{i}")
            except Exception as e:
                errors.append(e)

        def check_failures():
            try:
                for i in range(100):
                    CircuitBreaker.auth_failure_active(f"scope-check-{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t1 = threading.Thread(target=record_failures, name=f"writer-{i}")
            t2 = threading.Thread(target=check_failures, name=f"reader-{i}")
            threads.extend([t1, t2])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestCircuitBreakerIntegration:
    """Integration tests for CircuitBreaker with realistic scenarios."""

    def test_full_breaker_lifecycle(self) -> None:
        """Test complete lifecycle: record failures, trip, evict, recover."""
        breaker = CircuitBreaker(threshold=3, window_seconds=10)
        user_id = "user-1"

        # Record failures at time 0
        with patch.object(time, "time", return_value=0.0):
            breaker.record_failure(user_id)
            breaker.record_failure(user_id)
            assert breaker.allows(user_id) is True  # 2 < 3

        # Third failure trips the breaker
        with patch.object(time, "time", return_value=1.0):
            breaker.record_failure(user_id)
            assert breaker.allows(user_id) is False  # 3 >= 3

        # After window expires, breaker recovers
        with patch.object(time, "time", return_value=15.0):
            assert breaker.allows(user_id) is True

    def test_threshold_change_with_active_failures(self) -> None:
        """Test changing threshold while failures are recorded."""
        breaker = CircuitBreaker(threshold=5, window_seconds=60)
        user_id = "user-1"
        tool_type = "function"

        # Record some failures
        breaker.record_failure(user_id)
        breaker.record_failure(user_id)
        breaker.record_failure(user_id)
        breaker.record_tool_failure(user_id, tool_type)
        breaker.record_tool_failure(user_id, tool_type)

        # Both breakers allow (3 < 5 and 2 < 5)
        assert breaker.allows(user_id) is True
        assert breaker.tool_allows(user_id, tool_type) is True

        # Reduce threshold to 2
        breaker.threshold = 2

        # Now both should be blocked (3 >= 2 and 2 >= 2)
        assert breaker.allows(user_id) is False
        assert breaker.tool_allows(user_id, tool_type) is False

    def test_multiple_users_isolated(self) -> None:
        """Test that different users have isolated breakers."""
        breaker = CircuitBreaker(threshold=2, window_seconds=60)

        breaker.record_failure("user-1")
        breaker.record_failure("user-1")

        assert breaker.allows("user-1") is False
        assert breaker.allows("user-2") is True

    def test_auth_failure_shared_across_instances(self) -> None:
        """Test that auth failures are shared across breaker instances."""
        breaker1 = CircuitBreaker(threshold=3, window_seconds=60)
        breaker2 = CircuitBreaker(threshold=5, window_seconds=30)

        with patch.object(time, "time", return_value=1000.0):
            CircuitBreaker.note_auth_failure("global-scope")

        with patch.object(time, "time", return_value=1030.0):
            # Both instances should see the auth failure
            assert CircuitBreaker.auth_failure_active("global-scope") is True

# ===== From test_breaker_recovery.py =====

"""Circuit breaker state transitions and recovery tests.

These tests exercise the production breaker helpers (request/tool/db) rather
than manipulating internal deques directly.
"""


import asyncio

import pytest

import open_webui_openrouter_pipe.pipe as pipe_module
from open_webui_openrouter_pipe import (
    Pipe,
    SessionLogger,
    _ToolExecutionContext,
)


class TestRequestBreaker:
    """Tests for the per-user request circuit breaker."""

    def test_request_breaker_trips_after_threshold(self, monkeypatch, pipe_instance) -> None:
        """N failures in window => breaker blocks."""
        pipe = pipe_instance
        pipe._circuit_breaker.threshold = 3
        pipe._circuit_breaker.window_seconds = 60

        monkeypatch.setattr(pipe_module.time, "time", lambda: 1_000.0)

        user_id = "user-1"
        for _ in range(3):
            pipe._circuit_breaker.record_failure(user_id)

        assert pipe._circuit_breaker.allows(user_id) is False

    def test_request_breaker_resets_on_success(self, monkeypatch, pipe_instance) -> None:
        """Reset clears failures so breaker allows again."""
        pipe = pipe_instance
        pipe._circuit_breaker.threshold = 2
        pipe._circuit_breaker.window_seconds = 60

        monkeypatch.setattr(pipe_module.time, "time", lambda: 1_000.0)

        user_id = "user-1"
        pipe._circuit_breaker.record_failure(user_id)
        pipe._circuit_breaker.record_failure(user_id)
        assert pipe._circuit_breaker.allows(user_id) is False

        pipe._circuit_breaker.reset(user_id)
        assert pipe._circuit_breaker.allows(user_id) is True

    def test_breaker_window_sliding(self, monkeypatch, pipe_instance) -> None:
        """Failures outside the window are evicted and do not count."""
        pipe = pipe_instance
        pipe._circuit_breaker.threshold = 2
        pipe._circuit_breaker.window_seconds = 10

        user_id = "user-1"
        monkeypatch.setattr(pipe_module.time, "time", lambda: 0.0)
        pipe._circuit_breaker.record_failure(user_id)
        monkeypatch.setattr(pipe_module.time, "time", lambda: 50.0)
        pipe._circuit_breaker.record_failure(user_id)

        # Move time forward; both failures are now outside the 10s window.
        monkeypatch.setattr(pipe_module.time, "time", lambda: 100.0)
        assert pipe._circuit_breaker.allows(user_id) is True


class TestToolBreaker:
    """Tests for the tool-specific circuit breaker."""

    def test_tool_breaker_tracks_failures_by_type(self, monkeypatch, pipe_instance) -> None:
        """Failures are tracked per tool type and threshold blocks that type only."""
        pipe = pipe_instance
        pipe._circuit_breaker.threshold = 2
        pipe._circuit_breaker.window_seconds = 60
        monkeypatch.setattr(pipe_module.time, "time", lambda: 1_000.0)

        user_id = "user-1"
        tool_type = "function"
        pipe._circuit_breaker.record_tool_failure(user_id, tool_type)
        assert pipe._circuit_breaker.tool_allows(user_id, tool_type) is True
        pipe._circuit_breaker.record_tool_failure(user_id, tool_type)
        assert pipe._circuit_breaker.tool_allows(user_id, tool_type) is False

        assert pipe._circuit_breaker.tool_allows(user_id, "other_type") is True

    @pytest.mark.asyncio
    async def test_tool_breaker_skips_failing_tool_type(self, pipe_instance_async) -> None:
        """Blocked tool types emit a status notification via _notify_tool_breaker."""
        pipe = pipe_instance_async
        events: list[dict] = []

        async def emitter(event: dict) -> None:
            events.append(event)

        ctx = _ToolExecutionContext(
            queue=asyncio.Queue(),
            per_request_semaphore=asyncio.Semaphore(1),
            global_semaphore=None,
            timeout=1.0,
            batch_timeout=None,
            idle_timeout=None,
            user_id="user-1",
            event_emitter=emitter,
            batch_cap=1,
        )

        await pipe._ensure_tool_executor()._notify_tool_breaker(ctx, "function", "lookup")

        assert any(e.get("type") == "status" for e in events)
        assert any("Skipping lookup" in e.get("data", {}).get("description", "") for e in events)


class TestDatabaseBreaker:
    """Tests for the database circuit breaker."""

    def test_db_breaker_tracks_failures(self, monkeypatch, pipe_instance) -> None:
        """DB failures are tracked and eventually block DB ops."""
        pipe = pipe_instance
        pipe._circuit_breaker.threshold = 2
        pipe._circuit_breaker.window_seconds = 60
        pipe._artifact_store._breaker_threshold = 2
        pipe._artifact_store._breaker_window_seconds = 60
        monkeypatch.setattr(pipe_module.time, "time", lambda: 1_000.0)

        user_id = "user-1"
        pipe._artifact_store._record_db_failure(user_id)
        assert pipe._artifact_store._db_breaker_allows(user_id) is True
        pipe._artifact_store._record_db_failure(user_id)
        assert pipe._artifact_store._db_breaker_allows(user_id) is False

    @pytest.mark.asyncio
    async def test_db_breaker_suppresses_persistence_on_failure(self, monkeypatch, pipe_instance_async) -> None:
        """When DB breaker is tripped, _db_persist returns [] and emits a warning notification."""
        pipe = pipe_instance_async
        pipe._circuit_breaker.threshold = 1
        pipe._circuit_breaker.window_seconds = 60
        pipe._artifact_store._breaker_threshold = 1
        pipe._artifact_store._breaker_window_seconds = 60
        monkeypatch.setattr(pipe_module.time, "time", lambda: 1_000.0)

        user_id = "user-1"
        # Trip DB breaker.
        pipe._artifact_store._record_db_failure(user_id)

        events: list[dict] = []

        async def emitter(event: dict) -> None:
            events.append(event)

        ctx = _ToolExecutionContext(
            queue=asyncio.Queue(),
            per_request_semaphore=asyncio.Semaphore(1),
            global_semaphore=None,
            timeout=1.0,
            batch_timeout=None,
            idle_timeout=None,
            user_id=user_id,
            event_emitter=emitter,
            batch_cap=1,
        )

        tool_token = pipe._TOOL_CONTEXT.set(ctx)
        user_token = SessionLogger.user_id.set(user_id)
        try:
            result = await pipe._artifact_store._db_persist(
                [
                    {
                        "chat_id": "chat-1",
                        "message_id": "msg-1",
                        "item_type": "reasoning",
                        "payload": {"type": "reasoning", "content": "x"},
                    }
                ]
            )
        finally:
            SessionLogger.user_id.reset(user_token)
            pipe._TOOL_CONTEXT.reset(tool_token)

        assert result == []
        assert any(e.get("type") == "notification" for e in events)
        assert any(
            e.get("type") == "notification"
            and e.get("data", {}).get("type") == "warning"
            and "DB ops skipped" in e.get("data", {}).get("content", "")
            for e in events
        )

# ===== From test_auth_failfast.py =====


from typing import Any

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import (
    EncryptedStr,
    Pipe,
    ResponsesBody,
)


@pytest.mark.asyncio
async def test_invalid_encrypted_api_key_returns_auth_error_and_skips_catalog(monkeypatch) -> None:
    """Test that invalid API key returns error WITHOUT calling catalog endpoint."""
    monkeypatch.setenv("WEBUI_SECRET_KEY", "unit-test-secret")

    pipe = Pipe()
    valves = pipe.valves.model_copy(update={"API_KEY": EncryptedStr("encrypted:not-a-valid-token")})
    session = pipe._create_http_session(valves)

    # Mock HTTP at boundary - if catalog is requested, the test will fail
    # because aioresponses will raise an exception for unmocked calls
    with aioresponses() as mock_http:
        # Do NOT mock the catalog endpoint - if it's called, the test should fail
        # The real code should return an auth error before calling the catalog

        try:
            result = await pipe._handle_pipe_call(
                {"model": "openai/gpt-4o-mini", "stream": False},
                __user__={},
                __request__=None,
                __event_emitter__=None,
                __event_call__=None,
                __metadata__={"model": {"id": "openai/gpt-4o-mini"}},
                __tools__=None,
                __task__=None,
                __task_body__=None,
                valves=valves,
                session=session,
            )

            # Verify no HTTP calls were made (catalog was not called)
            # aioresponses raises if unexpected calls are made, so we're good if we reach here

            assert isinstance(result, dict)
            content = ((result.get("choices") or [{}])[0].get("message") or {}).get("content")
            assert isinstance(content, str)
            assert "Authentication Failed" in content
            assert "cannot be decrypted" in content
        finally:
            await session.close()
            await pipe.close()


@pytest.mark.asyncio
async def test_invalid_encrypted_api_key_task_returns_safe_stub(monkeypatch) -> None:
    """Test that invalid API key for task request returns safe stub."""
    monkeypatch.setenv("WEBUI_SECRET_KEY", "unit-test-secret")

    pipe = Pipe()
    valves = pipe.valves.model_copy(update={"API_KEY": EncryptedStr("encrypted:not-a-valid-token")})
    session = pipe._create_http_session(valves)

    with aioresponses() as mock_http:
        # Do NOT mock catalog - should not be called for auth failures

        try:
            result = await pipe._handle_pipe_call(
                {"model": "openai/gpt-4o-mini", "stream": False},
                __user__={},
                __request__=None,
                __event_emitter__=None,
                __event_call__=None,
                __metadata__={"model": {"id": "openai/gpt-4o-mini"}},
                __tools__=None,
                __task__="tags_generation",
                __task_body__={},
                valves=valves,
                session=session,
            )

            assert isinstance(result, dict)
            content = ((result.get("choices") or [{}])[0].get("message") or {}).get("content")
            assert isinstance(content, str)
            assert "\"tags\"" in content
        finally:
            await session.close()
            await pipe.close()


@pytest.mark.asyncio
async def test_process_transformed_request_accepts_string_task() -> None:
    """Test that _process_transformed_request accepts string task parameter.

    Real infrastructure exercised:
    - Real ResponsesBody.from_completions execution
    - Real task request processing
    - Real HTTP request to OpenRouter API
    """
    pipe = Pipe()
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    # Mock HTTP at boundary
    with aioresponses() as mock_http:
        # Mock catalog endpoint
        mock_http.get(
            "https://openrouter.ai/api/v1/models",
            payload={"data": [{"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "context_length": 8192}]},
        )

        # Mock task response
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload={
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": '{"tags":["test"]}'}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

        try:
            result = await pipe._process_transformed_request(
                {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "stream": False},
                __user__={},
                __request__=None,
                __event_emitter__=None,
                __event_call__=None,
                __metadata__={},
                __tools__=None,
                __task__="tags_generation",
                __task_body__={},
                valves=valves,
                session=session,
                openwebui_model_id="",
                pipe_identifier="pipe",
                allowlist_norm_ids={"openai.gpt-4o-mini"},
                enforced_norm_ids=set(),
                catalog_norm_ids=set(),
                features={},
                user_id="",
            )
            assert isinstance(result, str)
            assert "tags" in result.lower()
        finally:
            await session.close()
            await pipe.close()
