
import asyncio
import math
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from open_webui_openrouter_pipe.core.circuit_breaker import CircuitBreaker
from open_webui_openrouter_pipe.tools.tool_executor import _ToolExecutionContext, _resolved_user_obj


async def _echo_tool(**kwargs: Any) -> str:
    return "echo-ok"


async def _boom_tool(**kwargs: Any) -> str:
    raise RuntimeError("tool exploded")


def _registry(fn) -> dict[str, dict[str, Any]]:
    return {"mytool": {"type": "function", "callable": fn, "spec": {"parameters": {}}}}


def _calls(n: int) -> list[dict]:
    return [{"name": "mytool", "call_id": f"c{i}", "arguments": "{}"} for i in range(n)]


class _CtxHarness:
    def __init__(self, pipe, *, fusion_inner=False, tool_call_budget=None, tool_breaker=None, batch_timeout=5.0):
        self.pipe = pipe
        self.fusion_inner = fusion_inner
        self.tool_call_budget = tool_call_budget
        if fusion_inner and tool_breaker is None:
            tool_breaker = CircuitBreaker(threshold=pipe.valves.BREAKER_MAX_FAILURES, window_seconds=math.inf)
        self.tool_breaker = tool_breaker
        self.batch_timeout = batch_timeout
        self.ctx: Any = None
        self.token: Any = None

    async def __aenter__(self):
        self.ctx = _ToolExecutionContext(
            queue=asyncio.Queue(maxsize=10),
            per_request_semaphore=asyncio.Semaphore(2),
            global_semaphore=None,
            timeout=5.0,
            batch_timeout=self.batch_timeout,
            idle_timeout=None,
            user_id="u1",
            event_emitter=None,
            batch_cap=1,
            fusion_inner=self.fusion_inner,
            tool_breaker=self.tool_breaker,
            tool_call_budget=self.tool_call_budget,
        )
        executor = self.pipe._ensure_tool_executor()
        self.ctx.workers.append(asyncio.create_task(executor._tool_worker_loop(self.ctx)))
        self.token = self.pipe._TOOL_CONTEXT.set(self.ctx)
        return self.ctx

    async def __aexit__(self, *exc):
        self.pipe._TOOL_CONTEXT.reset(self.token)
        for w in self.ctx.workers:
            w.cancel()
        await asyncio.gather(*self.ctx.workers, return_exceptions=True)


class TestInnerToolBreakerSuppression:
    @pytest.mark.asyncio
    async def test_open_tool_breaker_does_not_block_inner_calls(self, monkeypatch, pipe_instance_async):
        pipe = pipe_instance_async
        monkeypatch.setattr(pipe._circuit_breaker, "tool_allows", lambda *a, **k: False)
        async with _CtxHarness(pipe, fusion_inner=True):
            outputs = await pipe._ensure_tool_executor()._execute_function_calls(
                _calls(1), _registry(_echo_tool)
            )
        assert len(outputs) == 1
        assert "echo-ok" in str(outputs[0])

    @pytest.mark.asyncio
    async def test_open_tool_breaker_still_blocks_normal_calls(self, monkeypatch, pipe_instance_async):
        pipe = pipe_instance_async
        monkeypatch.setattr(pipe._circuit_breaker, "tool_allows", lambda *a, **k: False)
        async with _CtxHarness(pipe, fusion_inner=False):
            outputs = await pipe._ensure_tool_executor()._execute_function_calls(
                _calls(1), _registry(_echo_tool)
            )
        assert len(outputs) == 1
        assert "skipped" in str(outputs[0])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("threshold", [1, 2])
    async def test_inner_tool_failures_not_recorded(self, monkeypatch, pipe_instance_async, threshold):
        pipe = pipe_instance_async
        recorded: list[Any] = []
        monkeypatch.setattr(
            pipe._circuit_breaker, "record_tool_failure",
            lambda *a, **k: recorded.append(a),
        )
        attempts: list[dict[str, Any]] = []

        async def _counting_boom_tool(**kwargs: Any) -> str:
            attempts.append(kwargs)
            raise RuntimeError("tool exploded")

        run_breaker = CircuitBreaker(threshold=threshold, window_seconds=math.inf)
        async with _CtxHarness(pipe, fusion_inner=True, tool_breaker=run_breaker):
            for _ in range(threshold):
                await pipe._ensure_tool_executor()._execute_function_calls(
                    _calls(1), _registry(_counting_boom_tool)
                )
        assert len(attempts) == threshold
        assert run_breaker.tool_allows("u1", "function", "mytool") is False
        assert recorded == []
        assert pipe._circuit_breaker.tool_allows("u1", "function", "mytool") is True

    @pytest.mark.asyncio
    async def test_normal_tool_failures_still_recorded(self, monkeypatch, pipe_instance_async):
        pipe = pipe_instance_async
        recorded: list[Any] = []
        monkeypatch.setattr(
            pipe._circuit_breaker, "record_tool_failure",
            lambda *a, **k: recorded.append(a),
        )
        async with _CtxHarness(pipe, fusion_inner=False):
            await pipe._ensure_tool_executor()._execute_function_calls(
                _calls(1), _registry(_boom_tool)
            )
        assert recorded


class TestInnerToolBudget:
    @pytest.mark.asyncio
    async def test_budget_exhaustion_skips_excess_calls(self, pipe_instance_async):
        pipe = pipe_instance_async
        async with _CtxHarness(pipe, fusion_inner=True, tool_call_budget=1) as ctx:
            outputs = await pipe._ensure_tool_executor()._execute_function_calls(
                _calls(3), _registry(_echo_tool)
            )
            assert ctx.tool_call_budget == 0
        texts = [str(o) for o in outputs]
        assert len(outputs) == 3
        assert sum("echo-ok" in s for s in texts) == 1
        assert sum("budget" in s for s in texts) == 2
        # A member's budget is spent by the first call, so the other two are refused inside
        # the first loop -- and the round comes back in the order the model asked for, not
        # with the two refusals up front.
        assert [o["call_id"] for o in outputs] == ["c0", "c1", "c2"]

    @pytest.mark.asyncio
    async def test_no_budget_means_unlimited(self, pipe_instance_async):
        pipe = pipe_instance_async
        async with _CtxHarness(pipe, fusion_inner=True, tool_call_budget=None):
            outputs = await pipe._ensure_tool_executor()._execute_function_calls(
                _calls(3), _registry(_echo_tool)
            )
        assert sum("echo-ok" in str(o) for o in outputs) == 3


class TestInnerSuccessLeavesTheUsersFailures:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("threshold", [2, 3])
    async def test_a_fusion_members_successful_call_leaves_the_users_own_tool_failures_alone(
        self, pipe_instance_async, threshold
    ):
        pipe = pipe_instance_async
        pipe._circuit_breaker.threshold = threshold
        for _ in range(threshold - 1):
            pipe._circuit_breaker.record_tool_failure("u1", "function", "mytool")
        async with _CtxHarness(pipe, fusion_inner=True):
            outputs = await pipe._ensure_tool_executor()._execute_function_calls(_calls(1), _registry(_echo_tool))
        assert "echo-ok" in str(outputs[0])

        # The user's own failures are still on record: one more trips the tool for the user.
        pipe._circuit_breaker.record_tool_failure("u1", "function", "mytool")
        assert pipe._circuit_breaker.tool_allows("u1", "function", "mytool") is False


async def _never_returns(**kwargs: Any) -> str:
    await asyncio.sleep(3600)
    return "never"


class TestInnerBatchLimitCountsForTheTurn:
    @pytest.mark.asyncio
    async def test_a_fusion_members_call_cut_by_the_batch_limit_counts_for_the_turn_not_the_user(self, pipe_instance_async):
        from open_webui_openrouter_pipe.core.circuit_breaker import CircuitBreaker

        pipe = pipe_instance_async
        pipe._circuit_breaker.threshold = 1
        turn_breaker = CircuitBreaker(threshold=1, window_seconds=60)
        async with _CtxHarness(pipe, fusion_inner=True, tool_breaker=turn_breaker, batch_timeout=0.05):
            outputs = await pipe._ensure_tool_executor()._execute_function_calls(_calls(1), _registry(_never_returns))

        assert "exceeded" in str(outputs[0])
        assert turn_breaker.tool_allows("u1", "function", "mytool") is False
        assert pipe._circuit_breaker.tool_allows("u1", "function", "mytool") is True

    @pytest.mark.asyncio
    async def test_a_users_own_call_cut_by_the_batch_limit_counts_for_the_user(self, pipe_instance_async):
        pipe = pipe_instance_async
        pipe._circuit_breaker.threshold = 1
        async with _CtxHarness(pipe, fusion_inner=False, batch_timeout=0.05):
            outputs = await pipe._ensure_tool_executor()._execute_function_calls(_calls(1), _registry(_never_returns))

        assert "exceeded" in str(outputs[0])
        assert pipe._circuit_breaker.tool_allows("u1", "function", "mytool") is False
