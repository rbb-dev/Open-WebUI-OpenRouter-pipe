from __future__ import annotations

from tests.posted_body import posted_json
import logging
import types
from typing import Any

import pytest

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.core.config import EncryptedStr
from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry as Registry

BASE_URL = "https://openrouter.ai/api/v1"
_BASE = "deepseek/deepseek-v4-flash"
_FREE = _BASE + ":free"
_PRESET_BASE = "moonshotai/kimi-k2.5"
_PRICING = {"prompt": "0.1", "completion": "0.4"}

_RESPONSE_OK = {
    "id": "resp_1",
    "object": "response",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hi"}],
        }
    ],
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}


def _row(model_id: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "name": model_id,
        "pricing": dict(_PRICING),
        "supported_parameters": ["temperature", "tools"],
    }


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status = 200
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"http {self.status}")

    async def json(self) -> dict[str, Any]:
        return self._payload

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *_exc: Any) -> bool:
        return False


class _FakeSession:
    def __init__(self, routes: list[tuple[str, dict[str, Any]]]) -> None:
        self._routes = routes
        self.posts: list[dict[str, Any]] = []
        self.closed = False
        self.connector = types.SimpleNamespace(_limit=100)

    def _payload_for(self, url: str) -> dict[str, Any]:
        for suffix, payload in self._routes:
            if str(url).endswith(suffix):
                return payload
        raise AssertionError(f"no fake route for {url}")

    def get(self, url: str, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(self._payload_for(url))

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.posts.append({"url": str(url), "body": posted_json(kwargs)})
        return _FakeResponse(_RESPONSE_OK)

    async def close(self) -> None:
        self.closed = True


def _pipe_with(routes: list[tuple[str, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _FakeSession]:
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-key")
    pipe.valves.BASE_URL = BASE_URL
    pipe.valves.ZDR_ENFORCE = False
    pipe.valves.ALLOW_USER_ZDR_OVERRIDE = False
    session = _FakeSession(routes)
    monkeypatch.setattr(pipe, "_create_http_session", lambda *_args, **_kwargs: session)
    return pipe, session


def _catalog_routes(rows: list[dict[str, Any]], zdr_ids: list[str]) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("models?limit=1", {"data": rows}),
        ("models?output_modalities=image", {"data": []}),
        ("endpoints/zdr", {"data": [{"model_id": item} for item in zdr_ids]}),
        ("videos/models", {"data": []}),
        ("models", {"data": rows}),
    ]


async def _load_catalog(pipe: Any, session: _FakeSession) -> None:
    await Registry.ensure_loaded(
        session,  # type: ignore[arg-type]
        base_url=BASE_URL,
        api_key="test-key",
        cache_seconds=3600,
        logger=pipe.logger,
        valves=pipe.valves,
    )


async def _request(pipe: Any, model: str) -> None:
    async def _emit(_event: dict[str, Any]) -> None:
        return None

    await pipe.pipe(
        body={"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": False},
        __user__={"id": "u1"},
        __request__=None,
        __event_emitter__=_emit,
        __event_call__=None,
        __metadata__={"model": {"id": model}},
        __tools__=None,
        __task__=None,
        __task_body__=None,
    )


@pytest.mark.asyncio
async def test_a_zdr_only_pipe_never_refuses_a_model_the_picker_offers(monkeypatch):
    rows = [_row(_BASE), _row(_FREE)]
    pipe, session = _pipe_with(_catalog_routes(rows, [_BASE]), monkeypatch)
    pipe.valves.MODEL_ID = "auto"
    pipe.valves.VARIANT_MODELS = _FREE
    pipe.valves.ZDR_MODELS_ONLY = True
    await _load_catalog(pipe, session)

    published = pipe._select_model_rows(Registry.list_models())
    assert published, "the picker published nothing"

    for row in published:
        session.posts.clear()
        await _request(pipe, row["id"])
        assert session.posts, (
            f"the picker publishes {row['id']} but the request for it never reached "
            "OpenRouter; the picker offered a model the gate refuses"
        )


@pytest.mark.asyncio
async def test_a_variant_named_in_the_model_allowlist_is_published_and_served(monkeypatch):
    pipe, session = _pipe_with(_catalog_routes([_row(_BASE)], []), monkeypatch)
    pipe.valves.MODEL_ID = f"{_BASE}:nitro"
    await _load_catalog(pipe, session)

    published_ids = [row["id"] for row in pipe._select_model_rows(Registry.list_models())]
    assert "deepseek.deepseek-v4-flash:nitro" in published_ids, (
        "the allowlist names the :nitro variant and the gate admits it, but the "
        "picker never published it"
    )

    await _request(pipe, "deepseek.deepseek-v4-flash:nitro")
    assert session.posts, "the variant request never reached OpenRouter"
    assert session.posts[0]["body"]["model"] == "deepseek/deepseek-v4-flash:nitro"


@pytest.mark.asyncio
async def test_the_reported_preset_config_serves_the_preset_from_the_picker(monkeypatch):
    pipe, session = _pipe_with(_catalog_routes([_row(_PRESET_BASE)], []), monkeypatch)
    pipe.valves.MODEL_ID = "auto"
    pipe.valves.VARIANT_MODELS = f"{_PRESET_BASE}@preset/dev-fallback-chain"
    await _load_catalog(pipe, session)

    published_ids = [row["id"] for row in pipe._select_model_rows(Registry.list_models())]
    assert "moonshotai.kimi-k2.5:preset/dev-fallback-chain" in published_ids

    await _request(pipe, "moonshotai.kimi-k2.5:preset/dev-fallback-chain")
    assert session.posts, "the preset request never reached OpenRouter"
    assert "preset/dev-fallback-chain" in str(session.posts[0]["body"]["model"])


@pytest.mark.asyncio
async def test_a_catalog_row_named_in_the_model_allowlist_without_its_base_logs_no_missing_base(monkeypatch, caplog):
    free = "deepseek/deepseek-r1:free"
    pipe, session = _pipe_with(_catalog_routes([_row("deepseek/deepseek-r1"), _row(free)], []), monkeypatch)
    pipe.valves.MODEL_ID = free
    await _load_catalog(pipe, session)

    with caplog.at_level(logging.DEBUG, logger=pipe.logger.name):
        published_ids = [row["id"] for row in pipe._select_model_rows(Registry.list_models())]

    assert published_ids == ["deepseek.deepseek-r1:free"]
    assert not [record for record in caplog.records if "Variant model base not found" in record.getMessage()]
