"""Concurrency limits and semaphore tests."""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

from __future__ import annotations

import contextlib
import asyncio
from typing import Any, cast

import pytest

from aioresponses import aioresponses

from open_webui_openrouter_pipe import Pipe, _PipeJob
from open_webui_openrouter_pipe.core.config import EncryptedStr


class TestRequestQueueLimits:
    """Tests for request queue limits."""

    @pytest.mark.asyncio
    async def test_request_queue_full_rejects_enqueue(self, pipe_instance_async) -> None:
        """When the internal request queue is full, _enqueue_job returns False."""
        pipe = pipe_instance_async

        # Ensure queue is initialized
        await pipe._ensure_concurrency_controls(pipe.valves)

        queue = pipe._request_queue
        assert queue is not None

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
    async def test_global_semaphore_limits_parallel_requests(self, pipe_instance_async) -> None:
        """MAX_CONCURRENT_REQUESTS blocks when all slots are held."""
        pipe = pipe_instance_async

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
            cls._global_semaphore = original_semaphore
            cls._semaphore_limit = original_limit


class TestToolSemaphore:
    """Tests for tool execution semaphores."""

    @pytest.mark.asyncio
    async def test_global_tool_semaphore_limits_parallel_tools(self, pipe_instance_async) -> None:
        """MAX_PARALLEL_TOOLS_GLOBAL blocks when all tool slots are held."""
        pipe = pipe_instance_async
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
    async def test_per_request_tool_semaphore_is_configured_from_valves(
        self, monkeypatch
    ) -> None:
        """Drives pipe() and reads the semaphore the pipe itself constructed.

        Two earlier versions of this test failed the same way: both installed a way to
        observe production and then asserted on an object the test built. The second one
        even appended to a `captured` list and never read it. If this test does not
        touch `captured`, it is not testing anything.
        """
        import open_webui_openrouter_pipe.pipe as pipe_module

        pipe = Pipe()
        pipe.valves.API_KEY = EncryptedStr("sk-test-key")
        pipe.valves.MAX_PARALLEL_TOOLS_PER_REQUEST = 3

        captured: list[Any] = []
        real_ctx = pipe_module._ToolExecutionContext

        def _capturing(*args, **kwargs):
            ctx = real_ctx(*args, **kwargs)
            captured.append(ctx)
            return ctx

        monkeypatch.setattr(pipe_module, "_ToolExecutionContext", _capturing)

        async def _streaming(self, session, request_body, **_kwargs):
            yield {"type": "response.completed", "response": {"output": [], "usage": {}}}

        monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _streaming)

        try:
            with aioresponses() as mock_http:
                mock_http.get(
                    "https://openrouter.ai/api/v1/models", payload={"data": []}, repeat=True
                )
                result = await pipe.pipe(
                    body={"model": "openrouter.test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                    __user__={"id": "u1", "role": "user"},
                    __request__=None,
                    __event_emitter__=None,
                    __event_call__=None,
                    __metadata__={},
                    __tools__={"lookup": {"callable": lambda **_k: "ok"}},
                )
                if hasattr(result, "__aiter__"):
                    async for _ in result:
                        pass
        finally:
            await pipe.close()

        assert captured, (
            "pipe() never constructed a _ToolExecutionContext, so this test observed "
            "nothing. That is the failure mode both previous versions had."
        )
        assert captured[0].per_request_semaphore._value == 3, (
            f"the pipe built its tool semaphore with "
            f"{captured[0].per_request_semaphore._value}, not the configured "
            "MAX_PARALLEL_TOOLS_PER_REQUEST of 3"
        )

    @pytest.mark.asyncio
    async def test_no_more_than_the_configured_number_of_tools_run_at_once(
        self, monkeypatch
    ) -> None:
        """Observes the peak that production's own semaphore permitted.

        Three earlier versions of this check failed three different ways: one built its
        own Semaphore and asserted CPython bounds it; one asserted on a context object
        the test constructed; one scanned the source for an `async with` and passed when
        the body was emptied out. Concurrency is a runtime property and no source shape
        implies it, so this runs tools and counts.

        Nothing here may construct a semaphore.
        """
        limit = 2
        pipe = Pipe()
        pipe.valves.API_KEY = EncryptedStr("sk-test-key")
        pipe.valves.MAX_PARALLEL_TOOLS_PER_REQUEST = limit
        pipe.valves.TOOL_EXECUTION_MODE = "Pipeline"

        live = 0
        peak = 0

        async def _slow(**_kwargs):
            nonlocal live, peak
            ran.append("x")
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.02)
            live -= 1
            return "ok"

        ran: list[str] = []
        calls = [
            {"type": "function_call", "call_id": f"c{i}", "name": "t", "arguments": "{}"}
            for i in range(limit * 3)
        ]
        rounds = [
            [{"type": "response.completed", "response": {"output": calls, "usage": {}}}],
            [
                {"type": "response.output_text.delta", "delta": "done"},
                {"type": "response.completed", "response": {"output": [], "usage": {}}},
            ],
        ]
        idx = 0

        async def _streaming(self, session, request_body, **_kwargs):
            nonlocal idx
            i = min(idx, len(rounds) - 1)
            idx += 1
            for event in rounds[i]:
                yield event

        monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _streaming)

        try:
            with aioresponses() as mock_http:
                mock_http.get(
                    "https://openrouter.ai/api/v1/models",
                    payload={
                        "data": [
                            {
                                "id": "openrouter/test",
                                "name": "T",
                                "supported_parameters": ["tools", "tool_choice"],
                                "architecture": {
                                    "input_modalities": ["text"],
                                    "output_modalities": ["text"],
                                },
                                "pricing": {"prompt": "0", "completion": "0"},
                                "context_length": 8192,
                            }
                        ]
                    },
                    repeat=True,
                )
                result = await pipe.pipe(
                    body={
                        "model": "openrouter.test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                    __user__={"id": "u1", "role": "user"},
                    __request__=None,
                    __event_emitter__=None,
                    __event_call__=None,
                    __metadata__={},
                    __tools__={
                        "t": {
                            "callable": _slow,
                            "spec": {
                                "name": "t",
                                "description": "d",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    },
                )
                if hasattr(result, "__aiter__"):
                    async for _ in result:
                        pass
        finally:
            await pipe.close()

        assert len(ran) == len(calls), (
            f"{len(ran)} of {len(calls)} tool calls reached the callable. With fewer "
            "running than the limit the peak below can never exceed it, so the check "
            "would pass on a build that executes nothing."
        )
        assert peak > 0, (
            "no tool ever ran, so this test observed nothing. That is the vacuity trap "
            "the previous versions fell into from the other direction."
        )
        assert peak <= limit, (
            f"{peak} tools ran concurrently under MAX_PARALLEL_TOOLS_PER_REQUEST={limit}; "
            "the per-request semaphore is not bounding execution"
        )

    @pytest.mark.asyncio
    async def test_semaphore_limit_increase_at_runtime(self, pipe_instance_async) -> None:
        """Increasing MAX_CONCURRENT_REQUESTS releases additional permits immediately."""
        pipe = pipe_instance_async
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

            await asyncio.wait_for(blocked, timeout=1.0)

            semaphore.release()
            semaphore.release()
        finally:
            cls._global_semaphore = original_semaphore
            cls._semaphore_limit = original_limit
