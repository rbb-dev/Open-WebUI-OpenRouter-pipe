from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, cast

from unittest.mock import patch

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter
from open_webui_openrouter_pipe.integrations.provider_options import (
    IMAGE_PROVIDER_KEYS,
    carrier_slug,
    merge_provider_options,
)
from open_webui_openrouter_pipe.integrations.image_client import OpenRouterImageClient
from open_webui_openrouter_pipe.storage.multimodal import _guess_image_mime_type
from open_webui_openrouter_pipe.integrations.image_types import (
    GeneratedImage,
    ImageGenerationError,
    ImageGenerationResult,
)
from open_webui_openrouter_pipe.models.registry import uses_dedicated_image_api
from open_webui_openrouter_pipe.streaming.event_emitter import EventEmitterHandler

BASE = "https://openrouter.ai/api/v1"

def _adapter(pipe: Any) -> ImageGenerationAdapter:
    return ImageGenerationAdapter(pipe=cast(Any, pipe), logger=cast(Any, _Logger()))


def _png(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


class _Logger:
    def __getattr__(self, _name: str):
        def _noop(*_args: Any, **_kwargs: Any) -> None:
            return None

        return _noop

    def isEnabledFor(self, _level: int) -> bool:
        return False


async def _client(session) -> OpenRouterImageClient:
    return OpenRouterImageClient(
        session, base_url=BASE, api_key="test-key", logger=_Logger()
    )


class _Body:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.input = items


def _event_data(event: Any) -> dict[str, Any]:
    data = event.get("data") if isinstance(event, dict) else None
    return data if isinstance(data, dict) else {}


class _Emitter:
    def __init__(self):
        self.statuses = []

    async def __call__(self, event):
        self.statuses.append(event)


class _StubGateway:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def resolve_storage_context(self, request, user_obj):
        return request, user_obj

    async def upload_to_owui_storage(self, **kwargs):
        self.calls.append(kwargs)
        return f"file-{len(self.calls)}"


class _StubResponsesBody:
    def __init__(self, items, provider: dict[str, Any] | None = None):
        self.input = items
        self.provider = provider


def _user_turn_with_images(count, provider: dict[str, Any] | None = None):
    """Attachments in the shape the request transformer actually leaves them in.

    ``_to_input_image`` inlines every attachment -- data URL, remote download or Open
    WebUI file id -- into a ``data:`` URI before the adapter ever sees it, and raises or
    skips when it cannot. A relative ``/api/v1/files/...`` path in ``responses_body.input``
    is a shape production does not produce, and building references from one made these
    tests assert that the adapter forwards an internal path to a third party.
    """
    content: list[dict[str, Any]] = [{"type": "input_text", "text": "make it bluer"}]
    for index in range(count):
        content.append(
            {"type": "input_image", "image_url": f"data:image/png;base64,att{index}", "detail": "auto"}
        )
    return _StubResponsesBody([{"role": "user", "content": content}], provider=provider)


class _Posted:
    def __init__(
        self, payload, content, calls, statuses, events, generations=None, headers=None,
        requests=None,
    ):
        self.generations = generations or []
        self.headers = headers or {}
        self.requests = requests or {}
        self.payload = payload
        self.content = content
        self.calls = calls
        self.statuses = statuses
        self.events = events


async def _posted(
    adapter: ImageGenerationAdapter,
    *,
    body: dict[str, Any],
    responses_body: Any,
    valves: Any,
    event_emitter: Any,
    normalized_model_id: str,
    api_model_id: str,
    metadata: dict[str, Any] | None = None,
    reply: dict[str, Any] | None = None,
    user: Any = None,
    user_obj: Any = None,
    show_usage: bool | None = None,
) -> _Posted:
    import aiohttp

    if show_usage is not None:
        valves.SHOW_FINAL_USAGE_STATUS = show_usage

    with aioresponses() as mocked:
        mocked.get(
            f"{BASE}/images/models/{api_model_id}/endpoints",
            payload={"endpoints": [{}]},
        )
        mocked.post(
            f"{BASE}/images",
            payload=reply
            or {
                "created": 1,
                "data": [{"b64_json": _b64(_png(8, 8)), "media_type": "image/png"}],
            },
        )
        async with aiohttp.ClientSession() as session:
            content = await adapter.generate(
                body=body,
                responses_body=responses_body,
                valves=valves,
                session=session,
                event_emitter=event_emitter,
                metadata={"chat_id": "chat-1", "message_id": "msg-1"}
                if metadata is None
                else metadata,
                user=user,
                request=object(),
                user_obj=user_obj if user_obj is not None else object(),
                normalized_model_id=normalized_model_id,
                api_model_id=api_model_id,
            )

        posts = [
            call
            for key, calls in mocked.requests.items()
            if key[1].path == "/api/v1/images"
            for call in calls
        ]
        assert posts, f"nothing was POSTed to /api/v1/images; saw {[k[1].path for k in mocked.requests]}"
        gateway = cast(Any, adapter._pipe)._file_gateway
        statuses = getattr(event_emitter, "statuses", [])
        return _Posted(
            posts[0].kwargs["json"],
            content,
            list(getattr(gateway, "calls", [])),
            [str(_event_data(item).get("description", "")) for item in statuses],
            list(statuses),
            list(getattr(cast(Any, adapter._pipe), "generations", [])),
            dict(posts[0].kwargs.get("headers") or {}),
            dict(mocked.requests),
        )


async def _posted_payload(adapter, **kwargs) -> dict[str, Any]:
    return (await _posted(adapter, **kwargs)).payload


class _StubValves:
    def __init__(self, key, base64_max_size_mb: int = 50, show_usage: bool = True):
        self.key = key
        self.BASE_URL = BASE
        self.HTTP_REFERER_OVERRIDE = ""
        self.MODEL_CATALOG_REFRESH_SECONDS = 3600
        self.BASE64_MAX_SIZE_MB = base64_max_size_mb
        self.SHOW_FINAL_USAGE_STATUS = show_usage
        self.FINAL_USAGE_STATUS_STYLE = "text"
        self.USAGE_STATUS_ICON_SET = ""
        self.COSTS_REDIS_DUMP = False


class _ReportRecorder:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def _report_openrouter_error(self, exc, **kwargs):
        # The real formatter hands back the card it rendered, and the adapter is required to
        # return that value; a double that returned None would hide a call site that dropped it.
        self.calls.append({"exc": exc, **kwargs})
        return f"### card for {getattr(exc, 'status', None)}"


class _KeyPipe:
    def __init__(self, key, record_errors: bool = False):
        self._key = key
        self._file_gateway = _StubGateway()
        self.valves = _StubValves(key)
        self._event_emitter_handler = EventEmitterHandler(
            logging.getLogger("test.events"), self.valves, cast(Any, self)
        )
        self.id = "orpipe"
        self.reports = _ReportRecorder() if record_errors else None
        self.generations: list[dict[str, Any]] = []
        self._generation_complete_dispatched: set[str] = set()

    async def _dispatch_plugin_event(self, method, *args, **kwargs):
        if method == "dispatch_on_generation_complete":
            self.generations.append({"usage": args[0], "status": args[1], **kwargs})

    async def _dispatch_generation_complete(self, usage, status, **kwargs):
        self._generation_complete_dispatched.add(str(kwargs.get("request_id") or ""))
        await self._dispatch_plugin_event(
            "dispatch_on_generation_complete", usage, status, **kwargs
        )

    def _ensure_error_formatter(self):
        from open_webui_openrouter_pipe.core.error_formatter import ErrorFormatter
        import logging as _logging

        if self.reports is not None:
            return self.reports
        return ErrorFormatter(
            cast(Any, self), cast(Any, self._event_emitter_handler), _logging.getLogger("test.fmt")
        )

    @staticmethod
    def _resolve_openrouter_api_key(valves) -> tuple[str | None, str | None]:
        return valves.key, None


def _labels(content: str) -> list[str]:
    return [
        line.split("](")[0][2:]
        for line in content.split("\n\n")
        if line.startswith("![")
    ]


def _seed_contract(
    adapter: Any,
    model_id: str,
    records: list[dict[str, Any]],
    *,
    at: float,
) -> None:
    """Seed the adapter's contract cache under the key its own pipe's valves resolve to.

    The cache is keyed on `(fingerprint(api_key), api_model_id)`, and every test here
    builds its adapter with the same key it passes to the call under test, so the
    adapter's own pipe is the source of truth for the seed. A bare model id would be a
    silent cache miss and the test would go on to assert against an empty cache.
    """
    adapter._endpoint_cache[adapter._endpoint_cache_key(adapter._pipe.valves, model_id)] = (
        at,
        records,
    )
