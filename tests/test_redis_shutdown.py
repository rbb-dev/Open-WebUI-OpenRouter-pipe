import logging

import pytest

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import Pipe


@pytest.mark.asyncio
async def test_stop_redis_logs_close_failure(caplog) -> None:
    pipe = Pipe()
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

