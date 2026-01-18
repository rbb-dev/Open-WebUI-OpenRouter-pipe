"""Circuit breaker state transitions and recovery tests.

These tests exercise the production breaker helpers (request/tool/db) rather
than manipulating internal deques directly.
"""

from __future__ import annotations

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
        pipe._breaker_threshold = 3
        pipe._breaker_window_seconds = 60

        monkeypatch.setattr(pipe_module.time, "time", lambda: 1_000.0)

        user_id = "user-1"
        for _ in range(3):
            pipe._record_failure(user_id)

        assert pipe._breaker_allows(user_id) is False

    def test_request_breaker_resets_on_success(self, monkeypatch, pipe_instance) -> None:
        """Reset clears failures so breaker allows again."""
        pipe = pipe_instance
        pipe._breaker_threshold = 2
        pipe._breaker_window_seconds = 60

        monkeypatch.setattr(pipe_module.time, "time", lambda: 1_000.0)

        user_id = "user-1"
        pipe._record_failure(user_id)
        pipe._record_failure(user_id)
        assert pipe._breaker_allows(user_id) is False

        pipe._reset_failure_counter(user_id)
        assert pipe._breaker_allows(user_id) is True

    def test_breaker_window_sliding(self, monkeypatch, pipe_instance) -> None:
        """Failures outside the window are evicted and do not count."""
        pipe = pipe_instance
        pipe._breaker_threshold = 2
        pipe._breaker_window_seconds = 10

        user_id = "user-1"
        monkeypatch.setattr(pipe_module.time, "time", lambda: 0.0)
        pipe._record_failure(user_id)
        monkeypatch.setattr(pipe_module.time, "time", lambda: 50.0)
        pipe._record_failure(user_id)

        # Move time forward; both failures are now outside the 10s window.
        monkeypatch.setattr(pipe_module.time, "time", lambda: 100.0)
        assert pipe._breaker_allows(user_id) is True


class TestToolBreaker:
    """Tests for the tool-specific circuit breaker."""

    def test_tool_breaker_tracks_failures_by_type(self, monkeypatch, pipe_instance) -> None:
        """Failures are tracked per tool type and threshold blocks that type only."""
        pipe = pipe_instance
        pipe._breaker_threshold = 2
        pipe._breaker_window_seconds = 60
        monkeypatch.setattr(pipe_module.time, "time", lambda: 1_000.0)

        user_id = "user-1"
        tool_type = "function"
        pipe._record_tool_failure_type(user_id, tool_type)
        assert pipe._tool_type_allows(user_id, tool_type) is True
        pipe._record_tool_failure_type(user_id, tool_type)
        assert pipe._tool_type_allows(user_id, tool_type) is False

        assert pipe._tool_type_allows(user_id, "other_type") is True

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

        await pipe._notify_tool_breaker(ctx, "function", "lookup")

        assert any(e.get("type") == "status" for e in events)
        assert any("Skipping lookup" in e.get("data", {}).get("description", "") for e in events)


class TestDatabaseBreaker:
    """Tests for the database circuit breaker."""

    def test_db_breaker_tracks_failures(self, monkeypatch, pipe_instance) -> None:
        """DB failures are tracked and eventually block DB ops."""
        pipe = pipe_instance
        pipe._breaker_threshold = 2
        pipe._breaker_window_seconds = 60
        monkeypatch.setattr(pipe_module.time, "time", lambda: 1_000.0)

        user_id = "user-1"
        pipe._record_db_failure(user_id)
        assert pipe._db_breaker_allows(user_id) is True
        pipe._record_db_failure(user_id)
        assert pipe._db_breaker_allows(user_id) is False

    @pytest.mark.asyncio
    async def test_db_breaker_suppresses_persistence_on_failure(self, monkeypatch, pipe_instance_async) -> None:
        """When DB breaker is tripped, _db_persist returns [] and emits a warning notification."""
        pipe = pipe_instance_async
        pipe._breaker_threshold = 1
        pipe._breaker_window_seconds = 60
        monkeypatch.setattr(pipe_module.time, "time", lambda: 1_000.0)

        user_id = "user-1"
        # Trip DB breaker.
        pipe._record_db_failure(user_id)

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
            result = await pipe._db_persist(
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
