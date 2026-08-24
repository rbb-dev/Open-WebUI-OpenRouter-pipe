"""Answers produced whole must arrive on the channel Open WebUI accumulates.

Open WebUI builds the assistant message it renders and persists from streamed
``delta.content`` chunks alone. A ``chat:completion`` frame is forwarded to the browser
out of band and is never counted as answer text, so a producer that emits only one ends
the turn with ``output == []``; the browser then overwrites the rendered message with the
empty string and the chat row is saved blank. Image generation shipped exactly that: the
picture rendered for a few milliseconds and vanished.

Every test here stubs one seam BELOW the emitter -- the middleware stream queue that the
pipe hands to Open WebUI -- so what is asserted is the bytes Open WebUI would accumulate,
not the shape of a test double. Each is parametrised over two distinct answers, so a
hardcoded constant in production cannot satisfy both rows.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import EncryptedStr, Pipe
from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter
from open_webui_openrouter_pipe.integrations.image_types import ImageGenerationError
from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter
from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry
from open_webui_openrouter_pipe.plugins.base import PluginBase

BASE = "https://openrouter.ai/api/v1"
IMAGE_MODEL = "openai/gpt-image-2"


def _logger() -> logging.Logger:
    return logging.getLogger("tests.unstreamed_answer_delivery")


def _png() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + (8).to_bytes(4, "big")
        + (8).to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )


def _stream_emitter(pipe: Pipe, queue: asyncio.Queue[Any]) -> Any:
    job = SimpleNamespace(
        request_id="req-unstreamed",
        metadata={"model": {"id": IMAGE_MODEL}},
        body={"model": IMAGE_MODEL},
        valves=pipe.valves,
        future=asyncio.get_running_loop().create_future(),
        event_emitter=None,
        stream_queue=queue,
    )
    return pipe._event_emitter_handler._make_middleware_stream_emitter(cast(Any, job), queue)


def _delta_text(items: list[Any]) -> str:
    """Concatenate exactly what Open WebUI's own accumulator would see."""
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        choices = item.get("choices")
        if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)):
            continue
        delta = choices[0].get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            parts.append(delta["content"])
    return "".join(parts)


def _shown_completion_text(items: list[Any]) -> str:
    """The content of the terminal frame the browser renders but never accumulates."""
    shown = ""
    for item in items:
        if not isinstance(item, dict):
            continue
        event = item.get("event")
        if not isinstance(event, dict) or event.get("type") != "chat:completion":
            continue
        data = event.get("data")
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            shown = data["content"]
    return shown


def _drain(queue: asyncio.Queue[Any]) -> list[Any]:
    items: list[Any] = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def _image_pipe() -> Pipe:
    pipe = Pipe()
    pipe.valves.BASE_URL = BASE
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.ENABLE_OPENROUTER_IMAGE_GENERATION = True
    return pipe


def _hand_back_file_id(pipe: Pipe, file_id: str) -> list[dict[str, Any]]:
    """Stand in for Open WebUI storage, spelling out the call production really makes.

    The signature mirrors ``OwuiFileGateway.upload_to_owui_storage`` keyword for keyword
    rather than swallowing ``**kwargs``, so a call site that drops or renames an argument
    fails here instead of being silently absorbed by a permissive double.
    """
    gateway = cast(Any, pipe)._file_gateway
    uploads: list[dict[str, Any]] = []

    async def _resolve(request: Any, user_obj: Any) -> tuple[Any, Any]:
        return request or SimpleNamespace(), user_obj or SimpleNamespace()

    async def _upload(
        *,
        request: Any,
        user: Any,
        file_data: bytes,
        filename: str,
        mime_type: str,
        chat_id: str | None = None,
        message_id: str | None = None,
        owui_user_id: str | None = None,
    ) -> str:
        uploads.append(
            {
                "request": request,
                "user": user,
                "file_data": file_data,
                "filename": filename,
                "mime_type": mime_type,
                "chat_id": chat_id,
                "message_id": message_id,
                "owui_user_id": owui_user_id,
            }
        )
        return file_id

    gateway.resolve_storage_context = _resolve
    gateway.upload_to_owui_storage = _upload
    return uploads


@pytest.mark.parametrize("file_id", ["file-alpha", "file-beta"])
@pytest.mark.asyncio
async def test_a_generated_image_reaches_open_webui_on_the_channel_it_accumulates(file_id):
    pipe = _image_pipe()
    try:
        uploads = _hand_back_file_id(pipe, file_id)
        adapter = ImageGenerationAdapter(pipe=cast(Any, pipe), logger=_logger())
        adapter._endpoint_cache[IMAGE_MODEL] = (time.monotonic(), [{}])
        queue: asyncio.Queue[Any] = asyncio.Queue()
        emitter = _stream_emitter(pipe, queue)

        import aiohttp

        with aioresponses() as mocked:
            mocked.post(
                f"{BASE}/images",
                payload={
                    "created": 1,
                    "data": [
                        {
                            "b64_json": base64.b64encode(_png()).decode(),
                            "media_type": "image/png",
                        }
                    ],
                },
            )
            async with aiohttp.ClientSession() as session:
                content = await adapter.generate(
                    body={},
                    responses_body=SimpleNamespace(
                        input=[
                            {
                                "role": "user",
                                "content": [{"type": "input_text", "text": "a leaf"}],
                            }
                        ],
                        provider=None,
                    ),
                    valves=pipe.valves,
                    session=session,
                    event_emitter=emitter,
                    metadata={"chat_id": "chat-1", "message_id": "msg-1"},
                    user={"id": "user-1"},
                    request=object(),
                    user_obj=object(),
                    normalized_model_id=IMAGE_MODEL.replace("/", "."),
                    api_model_id=IMAGE_MODEL,
                )
    finally:
        await pipe.close()

    assert [
        (call["mime_type"], call["file_data"], call["chat_id"], call["message_id"])
        for call in uploads
    ] == [("image/png", _png(), "chat-1", "msg-1")], (
        f"the answer must name a file the generation really stored: {uploads!r}"
    )
    assert file_id in content, (
        f"the fixture must reach the answer for this to distinguish anything: {content!r}"
    )
    assert _delta_text(_drain(queue)) == content, (
        "Open WebUI builds the stored message from delta chunks alone; an image delivered "
        "only on the completion frame is rendered, then overwritten with the empty string"
    )


@pytest.mark.parametrize(
    "reason",
    ["the provider refused the prompt", "the upstream ran out of capacity"],
)
@pytest.mark.asyncio
async def test_a_failed_image_generation_reaches_open_webui_on_the_channel_it_accumulates(reason):
    pipe = _image_pipe()
    try:
        adapter = ImageGenerationAdapter(pipe=cast(Any, pipe), logger=_logger())
        adapter._endpoint_cache[IMAGE_MODEL] = (time.monotonic(), [{}])

        built: list[dict[str, Any]] = []
        attempted: list[dict[str, Any]] = []

        class _RefusingClient:
            async def generate(
                self,
                payload: dict[str, Any],
                *,
                max_decoded_bytes: int = 0,
                on_progress: Any = None,
            ) -> Any:
                attempted.append(
                    {
                        "payload": payload,
                        "max_decoded_bytes": max_decoded_bytes,
                        "on_progress": on_progress,
                    }
                )
                raise ImageGenerationError(reason)

        def _refuse(
            session: Any,
            valves: Any,
            *,
            user: Any = None,
            owui_chat_id: str | None = None,
        ) -> Any:
            built.append(
                {
                    "session": session,
                    "valves": valves,
                    "user": user,
                    "owui_chat_id": owui_chat_id,
                }
            )
            return _RefusingClient()

        cast(Any, adapter)._client = _refuse
        queue: asyncio.Queue[Any] = asyncio.Queue()
        emitter = _stream_emitter(pipe, queue)

        content = await adapter.generate(
            body={},
            responses_body=SimpleNamespace(
                input=[
                    {"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}
                ],
                provider=None,
            ),
            valves=pipe.valves,
            session=object(),
            event_emitter=emitter,
            metadata={"chat_id": "chat-1", "message_id": "msg-1"},
            user={"id": "user-1"},
            request=object(),
            user_obj=object(),
            normalized_model_id=IMAGE_MODEL.replace("/", "."),
            api_model_id=IMAGE_MODEL,
        )
    finally:
        await pipe.close()

    assert [call["payload"]["model"] for call in attempted] == [IMAGE_MODEL], (
        "the card must describe a refused generation, not an early bail-out that never "
        f"reached the transport. client builds={built!r} attempts={attempted!r}"
    )
    assert reason in content, (
        f"the fixture must reach the answer for this to distinguish anything: {content!r}"
    )
    assert _delta_text(_drain(queue)) == content, (
        "a failure card delivered only on the completion frame is shown and then wiped, so "
        "the user is left with a blank turn and no idea what went wrong"
    )


@pytest.mark.parametrize(
    "content",
    [
        "<video controls src='/api/v1/files/vid-one/content'></video>\n",
        "<video controls src='/api/v1/files/vid-two/content'></video>\n",
    ],
)
@pytest.mark.asyncio
async def test_a_generated_video_reaches_open_webui_on_the_channel_it_accumulates(content):
    pipe = Pipe()
    try:
        adapter = VideoGenerationAdapter(pipe=pipe, logger=_logger())
        queue: asyncio.Queue[Any] = asyncio.Queue()
        emitter = _stream_emitter(pipe, queue)

        await adapter._emit_completion(emitter, content, usage={"cost": 0.25})
    finally:
        await pipe.close()

    items = _drain(queue)
    assert _delta_text(items) == content, (
        "video is the route that already worked; converging it onto the shared helper must "
        "not move its answer off the channel Open WebUI accumulates"
    )
    assert _shown_completion_text(items) == content


@pytest.mark.parametrize(
    "usage",
    [None, {}],
)
@pytest.mark.asyncio
async def test_an_empty_usage_block_is_not_reported_as_usage(usage):
    pipe = Pipe()
    seen: list[dict[str, Any]] = []

    async def _emitter(event: dict[str, Any]) -> None:
        seen.append(event)

    try:
        adapter = VideoGenerationAdapter(pipe=pipe, logger=_logger())
        await adapter._emit_completion(_emitter, "hello", usage=usage)
    finally:
        await pipe.close()

    completions = [e for e in seen if e.get("type") == "chat:completion"]
    assert completions and "usage" not in completions[-1]["data"], (
        "an absent cost is not a cost of zero; forwarding an empty block makes the UI draw a "
        f"usage line for a generation that reported none. got {completions!r}"
    )


def _image_only_catalog(model_ids: list[str]) -> dict[str, Any]:
    return {
        "data": [
            {
                "id": model_id,
                "name": model_id.split("/")[-1].replace("-", " ").title(),
                "architecture": {"output_modalities": ["image"]},
            }
            for model_id in model_ids
        ]
    }


async def _stream_pipe_answer(pipe: Pipe, model_id: str, prompt: str) -> list[Any]:
    answer = await pipe.pipe(
        body={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        },
        __user__={"id": "user_1"},
        __request__=None,
        __event_emitter__=None,
        __event_call__=None,
        __metadata__={"model": {"id": model_id}},
        __tools__=None,
        __task__=None,
        __task_body__=None,
    )
    stream = cast(Any, answer)
    assert hasattr(stream, "__aiter__"), f"a streamed request must return a generator: {answer!r}"
    return [chunk async for chunk in stream]


@pytest.mark.parametrize("model_id", ["openai/gpt-image-2", "qwen/qwen-image-3"])
@pytest.mark.asyncio
async def test_the_help_panel_reaches_open_webui_on_the_channel_it_accumulates(model_id):
    pipe = _image_pipe()
    try:
        with aioresponses() as mocked:
            mocked.get(f"{BASE}/models", payload=_image_only_catalog([model_id]), repeat=True)
            mocked.get(
                f"{BASE}/images/models/{model_id}/endpoints",
                payload={"endpoints": [{}]},
                repeat=True,
            )
            chunks = await _stream_pipe_answer(pipe, model_id, "help")
    finally:
        await pipe.close()
        OpenRouterModelRegistry.set_image_endpoints({})

    shown = _shown_completion_text(chunks)
    assert model_id.split("/")[-1] in shown.lower().replace(" ", "-"), (
        f"the panel must name the model it describes for this to distinguish anything: {shown!r}"
    )
    assert _delta_text(chunks) == shown, (
        "the help panel is composed, never streamed; delivering it only on the completion "
        "frame leaves Open WebUI with nothing to store and the turn is saved blank"
    )


@pytest.mark.parametrize("file_id", ["file-end-to-end-one", "file-end-to-end-two"])
@pytest.mark.asyncio
async def test_a_full_streamed_request_persists_the_image_it_rendered(file_id):
    pipe = _image_pipe()
    try:
        uploads = _hand_back_file_id(pipe, file_id)
        with aioresponses() as mocked:
            mocked.get(
                f"{BASE}/models", payload=_image_only_catalog([IMAGE_MODEL]), repeat=True
            )
            mocked.get(
                f"{BASE}/images/models/{IMAGE_MODEL}/endpoints",
                payload={"endpoints": [{}]},
                repeat=True,
            )
            mocked.post(
                f"{BASE}/images",
                payload={
                    "created": 1,
                    "data": [
                        {
                            "b64_json": base64.b64encode(_png()).decode(),
                            "media_type": "image/png",
                        }
                    ],
                },
                repeat=True,
            )
            chunks = await _stream_pipe_answer(pipe, IMAGE_MODEL, "a red mug on a windowsill")
    finally:
        await pipe.close()
        OpenRouterModelRegistry.set_image_endpoints({})

    shown = _shown_completion_text(chunks)
    assert [call["file_data"] for call in uploads] == [_png()], (
        f"the answer must name a file the generation really stored: {uploads!r}"
    )
    assert file_id in shown, (
        f"the generation must have reached the answer for this to mean anything: {shown!r}"
    )
    assert _delta_text(chunks) == shown, (
        "this is the shipped defect: the image renders from the completion frame, Open WebUI "
        "accumulates nothing, and the terminal frame then blanks the message it just drew"
    )


@pytest.mark.parametrize(
    ("status", "heading"),
    [(429, "Rate Limit Exceeded"), (402, "Insufficient Credits")],
)
@pytest.mark.asyncio
async def test_a_rejected_image_generation_hands_back_the_card_it_showed(status, heading):
    """The card the user reads must also be the value the adapter RETURNS.

    With streaming off, Open WebUI writes the assistant message from the pipe's return value
    alone -- a ``chat:message`` event is drawn in the browser and never stored. Returning "" there
    fails Open WebUI's own ``choices and content`` guard, which skips the message write, the
    outlet filters and the background tasks in one go: the user reads a good error, reloads, and
    the turn is empty with no title.

    Two HTTP statuses that select two different templates, and every card carries a freshly
    generated error id, so a production function that returned one fixed string could satisfy
    neither row. Asserting equality with the SHOWN text -- not just "something non-empty" --
    is what rules out re-rendering the card at the call site, which would mint a second error id
    and leave the user quoting an id that appears nowhere in the log.
    """
    pipe = _image_pipe()
    shown: list[str] = []

    async def _emitter(event: dict[str, Any]) -> None:
        if event.get("type") == "chat:message":
            shown.append(event["data"]["content"])

    try:
        adapter = ImageGenerationAdapter(pipe=cast(Any, pipe), logger=_logger())
        adapter._endpoint_cache[IMAGE_MODEL] = (time.monotonic(), [{}])

        import aiohttp

        with aioresponses() as mocked:
            mocked.post(
                f"{BASE}/images",
                status=status,
                payload={"error": {"message": "the provider said no", "code": status}},
            )
            async with aiohttp.ClientSession() as session:
                returned = await adapter.generate(
                    body={},
                    responses_body=SimpleNamespace(
                        input=[
                            {
                                "role": "user",
                                "content": [{"type": "input_text", "text": "a leaf"}],
                            }
                        ],
                        provider=None,
                    ),
                    valves=pipe.valves,
                    session=session,
                    event_emitter=_emitter,
                    metadata={"chat_id": "chat-1", "message_id": "msg-1"},
                    user={"id": "user-1"},
                    request=object(),
                    user_obj=object(),
                    normalized_model_id=IMAGE_MODEL.replace("/", "."),
                    api_model_id=IMAGE_MODEL,
                )
    finally:
        await pipe.close()

    assert len(shown) == 1, f"the rejection must produce exactly one card: {shown!r}"
    assert heading in shown[0], (
        f"the {status} card must come from the template that status selects: {shown[0]!r}"
    )
    assert returned == shown[0], (
        "with streaming off this return value is the whole assistant message; anything else "
        f"leaves the turn blank on reload. returned={returned!r} shown={shown[0]!r}"
    )


class _ShortCircuitPlugin(PluginBase):
    plugin_id = "test-short-circuit"
    plugin_name = "Short circuit"
    hooks = {"on_request": 0}

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.seen: list[dict[str, Any]] = []

    async def on_request(
        self,
        body: dict[str, Any],
        user: dict[str, Any],
        metadata: dict[str, Any],
        event_emitter: Any,
        task: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.seen.append({"body": body, "user": user, "metadata": metadata, "task": task})
        return {
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": self._answer}}
            ]
        }


@pytest.mark.parametrize(
    "answer",
    ["the plugin answered instead of the model", "a second, different plugin answer"],
)
@pytest.mark.asyncio
async def test_a_plugin_answer_reaches_open_webui_on_the_channel_it_accumulates(answer):
    """A plugin that short-circuits the request must deliver on the accumulated channel.

    Only the plugin is stubbed -- the registry's real dispatch, the real short-circuit branch and
    the real middleware stream translator all run -- so what is asserted is the bytes Open WebUI
    would accumulate. Two distinct answers, so a hardcoded string in production fails one row.
    """
    pipe = Pipe()
    pipe.valves.ENABLE_PLUGIN_SYSTEM = True
    plugin = _ShortCircuitPlugin(answer)
    try:
        registry = pipe._ensure_plugin_registry()
        registry._hook_subscribers.setdefault("on_request", []).append((plugin, 0))
        chunks = await _stream_pipe_answer(pipe, IMAGE_MODEL, "anything at all")
    finally:
        await pipe.close()

    assert len(plugin.seen) == 1, (
        f"the short-circuit must have been the thing that answered: {plugin.seen!r}"
    )
    assert _delta_text(chunks) == answer, (
        "a plugin answer emitted on any other channel is rendered and then wiped; Open WebUI "
        f"stores the delta chunks and nothing else. got {chunks!r}"
    )
