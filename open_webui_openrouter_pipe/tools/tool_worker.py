"""
Tool worker execution logic.

This module contains the worker loop and batching logic for tool execution.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .tool_executor import _QueuedToolCall, _ToolExecutionContext
from ..core.timing_logger import timed


@timed
async def _tool_worker_loop(self, context: _ToolExecutionContext) -> None:
    """Process queued tool calls with batching/timeouts."""
    pending: list[tuple[_QueuedToolCall | None, bool]] = []
    batch_cap = context.batch_cap
    idle_timeout = context.idle_timeout
    try:
        while True:
            if pending:
                item, from_queue = pending.pop(0)
            else:
                try:
                    get_coro = context.queue.get()
                    queued = (
                        await get_coro
                        if idle_timeout is None
                        else await asyncio.wait_for(get_coro, timeout=idle_timeout)
                    )
                except asyncio.TimeoutError:
                    message = (
                        f"Tool queue idle for {idle_timeout:.0f}s; cancelling pending work."
                        if idle_timeout
                        else "Tool queue idle timeout triggered."
                    )
                    context.timeout_error = context.timeout_error or message
                    self.logger.warning("%s", message)
                    break
                item, from_queue = queued, True
            if item is None:
                if from_queue:
                    context.queue.task_done()
                if pending:
                    continue
                break

            batch: list[tuple[_QueuedToolCall, bool]] = [(item, from_queue)]
            if item.allow_batch:
                while len(batch) < batch_cap:
                    try:
                        nxt = context.queue.get_nowait()
                        from_queue_next = True
                    except asyncio.QueueEmpty:
                        break
                    if nxt is None:
                        pending.insert(0, (None, True))
                        break
                    if nxt.allow_batch and self._can_batch_tool_calls(batch[0][0], nxt):
                        batch.append((nxt, from_queue_next))
                    else:
                        pending.insert(0, (nxt, from_queue_next))
                        break

            await self._execute_tool_batch([itm for itm, _ in batch], context)
            for _, flag in batch:
                if flag:
                    context.queue.task_done()
    finally:
        # Resolve remaining futures if the worker is stopping unexpectedly
        while pending:
            leftover, from_queue = pending.pop(0)
            if from_queue:
                context.queue.task_done()
            if leftover is None:
                continue
            if not leftover.future.done():
                error_msg = context.timeout_error or "Tool execution cancelled"
                leftover.future.set_result(
                    self._build_tool_output(
                        leftover.call,
                        error_msg,
                        status="cancelled",
                    )
                )


@timed
def _can_batch_tool_calls(self, first: _QueuedToolCall, candidate: _QueuedToolCall) -> bool:
    if first.call.get("name") != candidate.call.get("name"):
        return False

    dep_keys = {"depends_on", "_depends_on", "sequential", "no_batch"}
    if any(key in first.args or key in candidate.args for key in dep_keys):
        return False

    first_id = first.call.get("call_id")
    candidate_id = candidate.call.get("call_id")
    if first_id and self._args_reference_call(candidate.args, first_id):
        return False
    if candidate_id and self._args_reference_call(first.args, candidate_id):
        return False
    return True


@timed
def _args_reference_call(self, args: Any, call_id: str) -> bool:
    if isinstance(args, str):
        return call_id in args
    if isinstance(args, dict):
        return any(self._args_reference_call(value, call_id) for value in args.values())
    if isinstance(args, list):
        return any(self._args_reference_call(item, call_id) for item in args)
    return False
