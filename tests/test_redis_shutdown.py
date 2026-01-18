import asyncio
import logging

import pytest

from open_webui_openrouter_pipe import Pipe


@pytest.mark.asyncio
async def test_stop_redis_cancels_tasks_and_closes_client(pipe_instance_async) -> None:
    pipe = pipe_instance_async
    pipe.logger = logging.getLogger("tests.redis_shutdown")

    class FakeRedis:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    blocker = asyncio.Event()

    async def _stuck() -> None:
        await blocker.wait()

    pipe._redis_listener_task = asyncio.create_task(_stuck())
    pipe._redis_flush_task = asyncio.create_task(_stuck())
    pipe._redis_ready_task = asyncio.create_task(_stuck())

    client = FakeRedis()
    pipe._redis_client = client  # type: ignore[assignment]
    pipe._redis_enabled = True  # type: ignore[attr-defined]

    await pipe._stop_redis()

    assert pipe._redis_enabled is False  # type: ignore[attr-defined]
    assert pipe._redis_listener_task is None
    assert pipe._redis_flush_task is None
    assert pipe._redis_ready_task is None
    assert pipe._redis_client is None
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_stop_redis_logs_close_failure(caplog, pipe_instance_async) -> None:
    pipe = pipe_instance_async
    pipe.logger = logging.getLogger("tests.redis_shutdown")

    class FakeRedis:
        async def close(self) -> None:
            raise RuntimeError("boom")

    pipe._redis_client = FakeRedis()  # type: ignore[assignment]
    pipe._redis_enabled = True  # type: ignore[attr-defined]

    caplog.set_level(logging.DEBUG, logger="tests.redis_shutdown")
    await pipe._stop_redis()

    assert pipe._redis_client is None
    assert any("Failed to close Redis client" in r.getMessage() for r in caplog.records)
