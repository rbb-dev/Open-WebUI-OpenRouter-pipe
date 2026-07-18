
import asyncio
from typing import Any

import pytest

from open_webui_openrouter_pipe.tools.tool_executor import _ToolExecutionContext


async def _echo_tool(**kwargs: Any) -> str:
    return "echo-ok"


async def _boom_tool(**kwargs: Any) -> str:
    raise RuntimeError("tool exploded")


def _registry(fn) -> dict[str, dict[str, Any]]:
    return {"mytool": {"type": "function", "callable": fn, "spec": {"parameters": {}}}}


def _calls(n: int) -> list[dict]:
    return [{"name": "mytool", "call_id": f"c{i}", "arguments": "{}"} for i in range(n)]


class _CtxHarness:
    def __init__(self, pipe, *, fusion_inner=False, tool_call_budget=None):
        self.pipe = pipe
        self.fusion_inner = fusion_inner
        self.tool_call_budget = tool_call_budget
        self.ctx: Any = None
        self.token: Any = None

    async def __aenter__(self):
        self.ctx = _ToolExecutionContext(
            queue=asyncio.Queue(maxsize=10),
            per_request_semaphore=asyncio.Semaphore(2),
            global_semaphore=None,
            timeout=5.0,
            batch_timeout=5.0,
            idle_timeout=None,
            user_id="u1",
            event_emitter=None,
            batch_cap=1,
            fusion_inner=self.fusion_inner,
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
    async def test_inner_tool_failures_not_recorded(self, monkeypatch, pipe_instance_async):
        pipe = pipe_instance_async
        recorded: list[Any] = []
        monkeypatch.setattr(
            pipe._circuit_breaker, "record_tool_failure",
            lambda *a, **k: recorded.append(a),
        )
        async with _CtxHarness(pipe, fusion_inner=True):
            await pipe._ensure_tool_executor()._execute_function_calls(
                _calls(1), _registry(_boom_tool)
            )
        assert recorded == []

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

    @pytest.mark.asyncio
    async def test_no_budget_means_unlimited(self, pipe_instance_async):
        pipe = pipe_instance_async
        async with _CtxHarness(pipe, fusion_inner=True, tool_call_budget=None):
            outputs = await pipe._ensure_tool_executor()._execute_function_calls(
                _calls(3), _registry(_echo_tool)
            )
        assert sum("echo-ok" in str(o) for o in outputs) == 3
