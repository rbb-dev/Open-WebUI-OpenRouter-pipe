import asyncio
import logging

import pytest

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import Pipe, _QueuedToolCall, _ToolExecutionContext


@pytest.mark.asyncio
async def test_tool_exception_logs_stack_trace(caplog):
    pipe = Pipe()
    pipe.logger = logging.getLogger("tests.tool_exception_logging")

    def _boom() -> str:
        raise ValueError("boom")

    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    item = _QueuedToolCall(
        call={"name": "boom", "call_id": "call-1"},
        args={},
        tool_cfg={"type": "function", "callable": _boom},
        future=future,
        allow_batch=True,
    )
    context = _ToolExecutionContext(
        queue=asyncio.Queue(),
        per_request_semaphore=asyncio.Semaphore(1),
        global_semaphore=None,
        timeout=1.0,
        batch_timeout=None,
        idle_timeout=None,
        user_id="user-1",
        event_emitter=None,
        batch_cap=1,
    )

    caplog.set_level(logging.DEBUG, logger="tests.tool_exception_logging")
    await pipe._execute_tool_batch([item], context)

    result = future.result()
    assert isinstance(result, dict)
    assert "Tool error:" in (result.get("output") or "")

    debug_records = [
        record
        for record in caplog.records
        if record.levelno == logging.DEBUG
        and "Tool execution raised exception" in record.getMessage()
    ]
    assert debug_records, "Expected debug log with exc_info for tool exception"
    assert debug_records[0].exc_info is not None
