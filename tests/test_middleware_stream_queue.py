import asyncio

import pytest
from typing import Any, cast

from open_webui_openrouter_pipe import Pipe


def test_middleware_stream_queue_valves_defaults(pipe_instance) -> None:
    pipe = pipe_instance
    assert pipe.valves.MIDDLEWARE_STREAM_QUEUE_MAXSIZE == 0
    assert pipe.valves.MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS == 1.0


@pytest.mark.asyncio
async def test_try_put_middleware_stream_nowait_does_not_raise_when_full(pipe_instance_async) -> None:
    pipe = pipe_instance_async
    stream_queue: asyncio.Queue[dict | str | None] = asyncio.Queue(maxsize=1)
    stream_queue.put_nowait({"event": {"type": "notification", "data": {}}})

    pipe._try_put_middleware_stream_nowait(stream_queue, None)

    assert stream_queue.qsize() == 1


@pytest.mark.asyncio
async def test_put_middleware_stream_item_times_out_when_full(pipe_instance_async) -> None:
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self) -> None:
            self.request_id = "req-test"
            self.valves = pipe.Valves(
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=1,
                MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS=0.05,
            )
            self.future = asyncio.get_running_loop().create_future()

    job = _FakeJob()

    stream_queue: asyncio.Queue[dict | str | None] = asyncio.Queue(maxsize=1)
    stream_queue.put_nowait({"event": {"type": "notification", "data": {}}})

    with pytest.raises(asyncio.TimeoutError):
        await pipe._put_middleware_stream_item(
            cast(Any, job),
            stream_queue,
            {"event": {"type": "status"}},
        )
