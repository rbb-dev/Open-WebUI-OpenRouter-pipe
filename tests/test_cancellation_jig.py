"""OWUI stop-button cancellation jig.

Reproduces the exact Open WebUI stop flow against the real pipe stack with a
real local SSE server, so cancellation semantics (task cancellation, generator
teardown, aiohttp request abort) are exercised for real instead of mocked:

1. pipe() returns the middleware async generator (streaming mode).
2. An "OWUI process_chat" task consumes it (this is the task OWUI registers
   and cancels when the user presses stop).
3. The jig cancels that task and asserts the cancellation propagates all the
   way upstream: the local SSE server must observe the client disconnect.

If these tests pass, the pipe-side stop chain is sound and a dead stop button
points upstream (frontend task bookkeeping / OWUI task registry).
"""

import asyncio
import json
from typing import Any, cast

import pytest
from aiohttp import web

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.core.config import EncryptedStr


class _UpstreamRecorder:
    def __init__(self) -> None:
        self.request_started = asyncio.Event()
        self.disconnected = asyncio.Event()
        self.chunks_sent = 0


async def _start_sse_upstream() -> tuple[str, _UpstreamRecorder, web.AppRunner]:
    recorder = _UpstreamRecorder()

    async def models_handler(_request: web.Request) -> web.Response:
        return web.json_response({"data": [{"id": "openrouter/test-model"}]})

    async def responses_handler(request: web.Request) -> web.StreamResponse:
        recorder.request_started.set()
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        await response.prepare(request)

        def _frame(event_type: str, payload: dict) -> bytes:
            return (
                f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()
            )

        try:
            await response.write(
                _frame(
                    "response.created",
                    {"type": "response.created", "response": {"id": "resp_jig_1"}},
                )
            )
            await response.write(
                _frame(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": "item_jig_1",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                )
            )
            index = 0
            while True:
                await response.write(
                    _frame(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": "item_jig_1",
                            "output_index": 0,
                            "content_index": 0,
                            "delta": f"chunk{index} ",
                        },
                    )
                )
                recorder.chunks_sent += 1
                index += 1
                await asyncio.sleep(0.02)
        except BaseException:
            recorder.disconnected.set()
            raise
        return response

    app = web.Application()
    app.router.add_get("/models", models_handler)
    app.router.add_post("/responses", responses_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = cast(Any, site._server).sockets[0].getsockname()[1]
    return f"http://127.0.0.1:{port}", recorder, runner


def _streaming_body() -> dict:
    return {
        "model": "openrouter/test-model",
        "messages": [{"role": "user", "content": "stream forever"}],
        "stream": True,
    }


async def _build_pipe(base_url: str) -> Pipe:
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr(EncryptedStr.encrypt("test-api-key"))
    pipe.valves.BASE_URL = base_url
    return pipe


@pytest.mark.asyncio
async def test_owui_stop_task_cancel_aborts_upstream_request():
    base_url, recorder, runner = await _start_sse_upstream()
    pipe = await _build_pipe(base_url)
    consumed: list = []
    try:
        stream = await pipe.pipe(
            body=_streaming_body(),
            __user__={"id": "jig-user", "valves": {}},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__={"chat_id": "jig-chat", "message_id": "jig-msg"},
            __tools__=None,
        )
        assert hasattr(stream, "__anext__"), f"expected async generator, got {type(stream)}: {stream!r}"
        stream = cast(Any, stream)

        async def _owui_process_chat() -> None:
            async for item in stream:
                consumed.append(item)

        consumer_task = asyncio.create_task(_owui_process_chat())

        await asyncio.wait_for(recorder.request_started.wait(), timeout=10)
        for _ in range(200):
            if recorder.chunks_sent >= 5 and consumed:
                break
            await asyncio.sleep(0.02)
        assert recorder.chunks_sent >= 5, "upstream never streamed"
        assert consumed, "middleware generator never yielded"

        consumer_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consumer_task, timeout=5)

        await asyncio.wait_for(recorder.disconnected.wait(), timeout=5)

        chunks_at_disconnect = recorder.chunks_sent
        await asyncio.sleep(0.3)
        assert recorder.chunks_sent == chunks_at_disconnect, (
            "upstream kept streaming after disconnect"
        )
    finally:
        await pipe.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_owui_generator_close_aborts_upstream_request():
    base_url, recorder, runner = await _start_sse_upstream()
    pipe = await _build_pipe(base_url)
    consumed: list = []
    try:
        stream = await pipe.pipe(
            body=_streaming_body(),
            __user__={"id": "jig-user", "valves": {}},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__={"chat_id": "jig-chat-2", "message_id": "jig-msg-2"},
            __tools__=None,
        )
        assert hasattr(stream, "__anext__")
        stream = cast(Any, stream)

        await asyncio.wait_for(recorder.request_started.wait(), timeout=10)
        for _ in range(200):
            if recorder.chunks_sent >= 5:
                break
            await asyncio.sleep(0.02)
        item = await asyncio.wait_for(stream.__anext__(), timeout=5)
        consumed.append(item)

        await asyncio.wait_for(stream.aclose(), timeout=5)

        await asyncio.wait_for(recorder.disconnected.wait(), timeout=5)
    finally:
        await pipe.close()
        await runner.cleanup()
