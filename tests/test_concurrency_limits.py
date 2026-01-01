"""Concurrency limits and semaphore tests."""

from __future__ import annotations

import contextlib
import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import Pipe, _PipeJob


class TestRequestQueueLimits:
    """Tests for request queue limits."""

    @pytest.mark.asyncio
    async def test_request_queue_full_rejects_enqueue(self) -> None:
        """When the internal request queue is full, _enqueue_job returns False."""
        pipe = Pipe()

        # Ensure queue is initialized
        await pipe._ensure_concurrency_controls(pipe.valves)

        # Get the queue
        queue = type(pipe)._request_queue
        assert queue is not None

        # Fill queue to capacity (500)
        max_size = queue.maxsize or 500

        # Fill the queue
        loop = asyncio.get_running_loop()
        for i in range(max_size):
            job = _PipeJob(
                pipe=pipe,
                body={"model": "test", "messages": []},
                user={"id": f"user_{i}"},
                request=None,
                event_emitter=None,
                event_call=None,
                metadata={},
                tools=None,
                task=None,
                task_body=None,
                valves=pipe.valves,
                future=loop.create_future(),
            )
            queue.put_nowait(job)

        # Verify queue is full
        assert queue.full()

        # Next job should be rejected
        job = _PipeJob(
            pipe=pipe,
            body={"model": "test", "messages": []},
            user={"id": "new_user"},
            request=None,
            event_emitter=None,
            event_call=None,
            metadata={},
            tools=None,
            task=None,
            task_body=None,
            valves=pipe.valves,
            future=loop.create_future(),
        )

        # Queue should reject the job
        result = pipe._enqueue_job(job)
        assert result is False

        # Cleanup - drain the queue
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    @pytest.mark.asyncio
    async def test_global_semaphore_limits_parallel_requests(self) -> None:
        """MAX_CONCURRENT_REQUESTS blocks when all slots are held."""
        pipe = Pipe()

        cls = type(pipe)
        original_semaphore = cls._global_semaphore
        original_limit = cls._semaphore_limit
        blocked_task: asyncio.Task[bool] | None = None
        try:
            cls._global_semaphore = None
            cls._semaphore_limit = 0

            valves = pipe.valves.model_copy(update={"MAX_CONCURRENT_REQUESTS": 2})
            await pipe._ensure_concurrency_controls(valves)
            semaphore = cls._global_semaphore
            assert semaphore is not None
            semaphore = cast(asyncio.Semaphore, semaphore)

            # Hold both permits.
            await semaphore.acquire()
            await semaphore.acquire()

            # A third acquire must block until we release a permit.
            blocked_task = asyncio.create_task(semaphore.acquire())
            await asyncio.sleep(0)
            assert not blocked_task.done()

            semaphore.release()
            await asyncio.wait_for(blocked_task, timeout=1.0)

            # Release the two remaining held permits (one from the initial pair, one from blocked_task).
            semaphore.release()
            semaphore.release()
        finally:
            if blocked_task is not None and not blocked_task.done():
                blocked_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await blocked_task
            cls._global_semaphore = original_semaphore
            cls._semaphore_limit = original_limit


class TestToolSemaphore:
    """Tests for tool execution semaphores."""

    @pytest.mark.asyncio
    async def test_global_tool_semaphore_limits_parallel_tools(self) -> None:
        """MAX_PARALLEL_TOOLS_GLOBAL blocks when all tool slots are held."""
        pipe = Pipe()
        cls = type(pipe)
        original_semaphore = cls._tool_global_semaphore
        original_limit = cls._tool_global_limit
        blocked_task: asyncio.Task[bool] | None = None
        try:
            cls._tool_global_semaphore = None
            cls._tool_global_limit = 0

            valves = pipe.valves.model_copy(update={"MAX_PARALLEL_TOOLS_GLOBAL": 2})
            await pipe._ensure_concurrency_controls(valves)
            semaphore = cls._tool_global_semaphore
            assert semaphore is not None
            semaphore = cast(asyncio.Semaphore, semaphore)

            await semaphore.acquire()
            await semaphore.acquire()

            blocked_task = asyncio.create_task(semaphore.acquire())
            await asyncio.sleep(0)
            assert not blocked_task.done()

            semaphore.release()
            await asyncio.wait_for(blocked_task, timeout=1.0)

            semaphore.release()
            semaphore.release()
        finally:
            if blocked_task is not None and not blocked_task.done():
                blocked_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await blocked_task
            cls._tool_global_semaphore = original_semaphore
            cls._tool_global_limit = original_limit

    @pytest.mark.asyncio
    async def test_per_request_tool_semaphore_is_configured_from_valves(self, monkeypatch) -> None:
        """_execute_pipe_job wires MAX_PARALLEL_TOOLS_PER_REQUEST into the tool context semaphore."""
        pipe = Pipe()
        loop = asyncio.get_running_loop()

        observed: dict[str, Any] = {}

        async def fake_handle(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            ctx = pipe._TOOL_CONTEXT.get()
            assert ctx is not None
            sem = ctx.per_request_semaphore
            limit = cast(int, observed["limit"])

            # Acquire up to the limit.
            for _ in range(limit):
                await asyncio.wait_for(sem.acquire(), timeout=1.0)

            # Next acquire must block (timeout).
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(sem.acquire(), timeout=0.05)

            for _ in range(limit):
                sem.release()
            return {"ok": True}

        def fake_create_http_session(_valves):  # type: ignore[no-untyped-def]
            session = AsyncMock()
            session.close = AsyncMock()
            return session

        monkeypatch.setattr(pipe, "_handle_pipe_call", fake_handle)
        monkeypatch.setattr(pipe, "_create_http_session", fake_create_http_session)

        valves = pipe.valves.model_copy(update={"MAX_CONCURRENT_REQUESTS": 10, "MAX_PARALLEL_TOOLS_PER_REQUEST": 2})
        observed["limit"] = valves.MAX_PARALLEL_TOOLS_PER_REQUEST
        await pipe._ensure_concurrency_controls(valves)

        job = _PipeJob(
            pipe=pipe,
            body={"model": "openai/gpt-5", "input": []},
            user={"id": "user_1"},
            request=None,
            event_emitter=None,
            event_call=None,
            metadata={},
            tools=None,
            task=None,
            task_body=None,
            valves=valves,
            future=loop.create_future(),
        )
        await pipe._execute_pipe_job(job)
        assert job.future.done()
        assert job.future.result() == {"ok": True}


class TestSemaphoreRuntimeUpdate:
    """Tests for dynamic semaphore limit updates."""

    @pytest.mark.asyncio
    async def test_semaphore_limit_increase_at_runtime(self) -> None:
        """Increasing MAX_CONCURRENT_REQUESTS releases additional permits immediately."""
        pipe = Pipe()
        cls = type(pipe)
        original_semaphore = cls._global_semaphore
        original_limit = cls._semaphore_limit
        try:
            cls._global_semaphore = None
            cls._semaphore_limit = 0

            valves_small = pipe.valves.model_copy(update={"MAX_CONCURRENT_REQUESTS": 1})
            await pipe._ensure_concurrency_controls(valves_small)
            semaphore = cls._global_semaphore
            assert semaphore is not None
            semaphore = cast(asyncio.Semaphore, semaphore)

            await semaphore.acquire()
            blocked = asyncio.create_task(semaphore.acquire())
            await asyncio.sleep(0)
            assert not blocked.done()

            valves_big = pipe.valves.model_copy(update={"MAX_CONCURRENT_REQUESTS": 2})
            await pipe._ensure_concurrency_controls(valves_big)

            # The increase should have released one extra permit, allowing blocked to complete.
            await asyncio.wait_for(blocked, timeout=1.0)

            semaphore.release()
            semaphore.release()
        finally:
            cls._global_semaphore = original_semaphore
            cls._semaphore_limit = original_limit
