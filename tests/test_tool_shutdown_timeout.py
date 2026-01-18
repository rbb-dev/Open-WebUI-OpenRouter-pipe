import asyncio
import logging

import pytest

from open_webui_openrouter_pipe import Pipe, _ToolExecutionContext


@pytest.mark.asyncio
async def test_shutdown_tool_context_times_out_and_cancels(caplog, pipe_instance_async) -> None:
    pipe = pipe_instance_async
    pipe.logger = logging.getLogger("tests.tool_shutdown")
    pipe.valves.TOOL_SHUTDOWN_TIMEOUT_SECONDS = 0.01

    queue: asyncio.Queue = asyncio.Queue()
    context = _ToolExecutionContext(
        queue=queue,
        per_request_semaphore=asyncio.Semaphore(1),
        global_semaphore=None,
        timeout=1.0,
        batch_timeout=None,
        idle_timeout=None,
        user_id="u",
        event_emitter=None,
        batch_cap=1,
    )

    blocker = asyncio.Event()

    async def stuck_worker() -> None:
        await blocker.wait()

    task = asyncio.create_task(stuck_worker())
    context.workers.append(task)

    caplog.set_level(logging.WARNING, logger="tests.tool_shutdown")
    await pipe._shutdown_tool_context(context)

    assert task.cancelled() or task.done()
    assert any("Tool shutdown exceeded" in r.getMessage() for r in caplog.records)

