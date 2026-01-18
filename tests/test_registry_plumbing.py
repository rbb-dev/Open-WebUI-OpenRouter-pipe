import asyncio
import json
import logging
import time
from typing import Any, cast

import pytest

import open_webui_openrouter_pipe as ow


class DummyResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200, reason: str = "OK"):
        self._payload = payload
        self.status = status
        self.reason = reason

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload

    async def read(self):
        return json.dumps(self._payload).encode()

    async def text(self):
        return json.dumps(self._payload)

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status} {self.reason}")

    @property
    def headers(self):
        return {}


class DummySession:
    def __init__(self, response: DummyResponse):
        self._response = response

    def get(self, *_, **__):
        return self._response


@pytest.fixture(autouse=True)
def reset_registry():
    reg = ow.OpenRouterModelRegistry
    reg._models = []
    reg._specs = {}
    reg._id_map = {}
    reg._last_fetch = 0
    reg._lock = asyncio.Lock()
    reg._next_refresh_after = 0
    reg._consecutive_failures = 0
    reg._last_error = None
    reg._last_error_time = 0.0
    ow.ModelFamily.set_dynamic_specs(None)
    yield


@pytest.mark.asyncio
async def test_registry_refresh_populates_models_and_specs():
    payload = {
        "data": [
            {
                "id": "author/model",
                "name": "Demo",
                "supported_parameters": ["tools", "reasoning", "include_reasoning"],
                "architecture": {
                    "input_modalities": ["image", "file"],
                    "output_modalities": ["text", "image"],
                },
                    "pricing": {"web_search": "0.5"},
                "top_provider": {"max_completion_tokens": 1024},
                "context_length": 8192,
            }
        ]
    }
    session = cast(Any, DummySession(DummyResponse(payload)))
    await ow.OpenRouterModelRegistry.ensure_loaded(
        session,
        base_url="https://api",
        api_key="secret",
        cache_seconds=60,
        logger=logging.getLogger("test"),
    )
    models = ow.OpenRouterModelRegistry.list_models()
    assert models[0]["id"] == "author.model"
    assert ow.OpenRouterModelRegistry.api_model_id("author.model") == "author/model"
    specs = ow.OpenRouterModelRegistry._specs
    assert "author.model" in specs
    assert "function_calling" in specs["author.model"]["features"]


@pytest.mark.asyncio
async def test_registry_cache_prevents_refresh():
    """Test that registry cache prevents unnecessary HTTP refresh.

    Real infrastructure exercised:
    - Real cache expiry logic
    - Real timestamp checking
    - Real ensure_loaded flow
    - Would make HTTP call if cache expired (but doesn't because cache is fresh)
    """
    from aioresponses import aioresponses
    import aiohttp

    reg = ow.OpenRouterModelRegistry
    reg._specs = {"demo": {"features": []}}
    reg._models = [{"id": "demo", "norm_id": "demo", "name": "Demo"}]
    now = time.time()
    reg._last_fetch = now
    reg._next_refresh_after = now + 999  # Cache valid for 999 more seconds
    reg._lock = asyncio.Lock()

    # Mock HTTP endpoint - if called, test should fail because cache should prevent it
    with aioresponses() as mock_http:
        # Set up mock that would be called if cache was ignored
        mock_http.get(
            "https://api/models",
            payload={"data": [{"id": "should/not/be/called"}]},
        )

        async with aiohttp.ClientSession() as session:
            await reg.ensure_loaded(
                session,
                base_url="https://api",
                api_key="key",
                cache_seconds=60,
                logger=logging.getLogger("test")
            )

        # Verify cache was used (models didn't change)
        assert reg._models == [{"id": "demo", "norm_id": "demo", "name": "Demo"}]
        # Verify HTTP was not called (aioresponses would show it in history if it was)
        assert len(mock_http.requests) == 0, "HTTP should not be called when cache is fresh"


@pytest.mark.asyncio
async def test_registry_refresh_failure_uses_cache():
    """Test that registry falls back to cache when HTTP refresh fails.

    Real infrastructure exercised:
    - Real HTTP request attempt
    - Real error handling in _refresh
    - Real cache fallback logic
    - Real ensure_loaded error recovery
    """
    from aioresponses import aioresponses
    import aiohttp

    reg = ow.OpenRouterModelRegistry
    reg._specs = {"demo": {"features": []}}
    reg._models = [{"id": "demo", "norm_id": "demo", "name": "Demo"}]
    reg._lock = asyncio.Lock()
    reg._last_fetch = 0  # Force refresh attempt
    reg._next_refresh_after = 0

    # Mock HTTP endpoint to fail with network error
    with aioresponses() as mock_http:
        mock_http.get(
            "https://api/models",
            exception=RuntimeError("Network failure during refresh"),
        )

        async with aiohttp.ClientSession() as session:
            # Should not raise - should use cached models instead
            await reg.ensure_loaded(
                session,
                base_url="https://api",
                api_key="key",
                cache_seconds=1,
                logger=logging.getLogger("test")
            )

        # Verify cached models are still available
        assert reg._models == [{"id": "demo", "norm_id": "demo", "name": "Demo"}]
        # Verify HTTP was attempted
        assert len(mock_http.requests) == 1, "HTTP refresh should have been attempted"


@pytest.mark.asyncio
async def test_registry_raises_when_key_missing():
    session = cast(Any, DummySession(DummyResponse({"data": []})))
    with pytest.raises(ValueError):
        await ow.OpenRouterModelRegistry.ensure_loaded(session, base_url="https://api", api_key="", cache_seconds=60, logger=logging.getLogger("test"))


@pytest.mark.asyncio
async def test_registry_refresh_error_no_cache():
    """Test that registry refresh failure without cache raises error.

    Real infrastructure exercised:
    - Real HTTP request attempt
    - Real _refresh method execution
    - Real error propagation when no cache available
    """
    from aioresponses import aioresponses
    import aiohttp

    reg = ow.OpenRouterModelRegistry
    # Ensure no cache is available
    reg._models = []
    reg._specs = {}
    reg._last_fetch = 0
    reg._next_refresh_after = 0

    # Mock HTTP to fail
    with aioresponses() as mock_http:
        mock_http.get(
            "https://api/models",
            exception=RuntimeError("boom"),
        )

        async with aiohttp.ClientSession() as session:
            with pytest.raises(RuntimeError, match="boom"):
                await reg.ensure_loaded(
                    session,
                    base_url="https://api",
                    api_key="key",
                    cache_seconds=1,
                    logger=logging.getLogger("test")
                )


def test_registry_record_refresh_bookkeeping():
    reg = ow.OpenRouterModelRegistry
    reg._record_refresh_success(cache_seconds=30)
    assert reg._consecutive_failures == 0
    assert reg._next_refresh_after >= reg._last_fetch

    reg._record_refresh_failure(RuntimeError("boom"), cache_seconds=30)
    assert reg._consecutive_failures == 1
    assert reg._last_error == "boom"


def test_registry_feature_and_capability_derivation():
    features = ow.OpenRouterModelRegistry._derive_features(
        supported_parameters={"tools", "include_reasoning"},
        architecture={"input_modalities": ["image"], "output_modalities": ["image"]},
        pricing={"web_search": "1"},
    )
    assert {"function_calling", "reasoning_summary", "web_search_tool", "vision", "image_gen_tool"}.issubset(features)

    assert ow.OpenRouterModelRegistry._supports_web_search({"web_search": "0.01"}) is True

    capabilities = ow.OpenRouterModelRegistry._derive_capabilities(
        architecture={"input_modalities": ["FILE", "IMAGE"], "output_modalities": ["IMAGE"]},
        pricing={"web_search": 0},
    )
    assert capabilities["vision"] is True
    assert capabilities["file_upload"] is True


def test_registry_list_models_and_mapping():
    reg = ow.OpenRouterModelRegistry
    reg._models = [{"id": "demo", "norm_id": "demo", "name": "Demo"}]
    reg._specs = {"demo": {"capabilities": {"vision": True}}}
    reg._id_map = {"demo": "provider/demo"}
    models = reg.list_models()
    assert models[0]["capabilities"]["vision"] is True
    assert reg.api_model_id("demo") == "provider/demo"
