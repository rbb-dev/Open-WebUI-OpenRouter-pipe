import asyncio
import inspect
import json
import logging
import time
from pathlib import Path
from typing import Any, cast

import pytest

import open_webui_openrouter_pipe as ow
from open_webui_openrouter_pipe.models.registry import _fingerprint
from tests.doc_truth_anchors import between, doc


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
    reg._failure_counts = {}
    reg._last_errors = {}
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
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false
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
    key = _fingerprint("sk-bookkeeping")
    reg._record_refresh_success(cache_seconds=30, api_key="sk-bookkeeping")
    assert key not in reg._failure_counts
    assert reg._next_refresh_after >= reg._last_fetch

    reg._record_refresh_failure(
        RuntimeError("boom"), cache_seconds=30, api_key="sk-bookkeeping")
    assert reg._failure_counts[key] == 1
    assert reg._last_error == "boom"


def test_registry_feature_and_capability_derivation():
    features = ow.OpenRouterModelRegistry._derive_features(
        supported_parameters={"tools", "include_reasoning"},
        architecture={"input_modalities": ["image"], "output_modalities": ["image"]},
        pricing={"web_search": "1"},
    )
    assert {"function_calling", "reasoning_summary", "vision", "image_gen_tool"}.issubset(features)
    assert "web_search_tool" not in features, "a flag nothing reads is still being published"

    assert ow.OpenRouterModelRegistry._supports_web_search({"web_search": "0.01"}) is True
    assert ow.OpenRouterModelRegistry._supports_web_search({"web_search": {"price": "0.02"}}) is True
    assert ow.OpenRouterModelRegistry._supports_web_search({"web_search": {"price": "0"}}) is False
    assert ow.OpenRouterModelRegistry._supports_web_search({"web_search": {"unit": "per_1m"}}) is False

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


@pytest.mark.asyncio
async def test_registry_tilde_alias_catalog_hit_resolves_verbatim():
    """~ router aliases are real catalog entries: the norm keeps the tilde, the
    api id round-trips verbatim, and capabilities resolve from the alias's own spec."""
    payload = {
        "data": [
            {
                "id": "~anthropic/claude-fable-latest",
                "name": "Anthropic: Claude Fable (latest)",
                "supported_parameters": ["tools", "reasoning", "include_reasoning"],
                "architecture": {
                    "input_modalities": ["image", "file"],
                    "output_modalities": ["text"],
                },
                "pricing": {"prompt": "0.000003"},
                "top_provider": {"max_completion_tokens": 64000},
                "context_length": 200000,
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
    assert (
        ow.OpenRouterModelRegistry.api_model_id("~anthropic.claude-fable-latest")
        == "~anthropic/claude-fable-latest"
    )
    assert "~anthropic.claude-fable-latest" in ow.OpenRouterModelRegistry._id_map
    assert ow.ModelFamily.features("~anthropic.claude-fable-latest"), (
        "alias catalog spec must resolve (capabilities are keyed by the tilde norm)"
    )


def test_registry_tilde_alias_non_catalog_fallback_preserves_tilde():
    """Stale-catalog window: an uncataloged ~ id still reconstructs with its tilde."""
    assert "~anthropic.claude-fable-latest" not in ow.OpenRouterModelRegistry._id_map, (
        "precondition: this test must exercise the non-catalog fallback branch"
    )
    assert (
        ow.OpenRouterModelRegistry.api_model_id("~anthropic.claude-fable-latest")
        == "~anthropic/claude-fable-latest"
    )


def test_is_claude_reasoning_model_accepts_tilde_aliases():
    from open_webui_openrouter_pipe.models.registry import _is_claude_reasoning_model

    assert _is_claude_reasoning_model("anthropic.claude-sonnet-4.5") is True
    assert _is_claude_reasoning_model("anthropic.claude-opus-4.6") is True
    assert _is_claude_reasoning_model("~anthropic.claude-sonnet-latest") is True
    assert _is_claude_reasoning_model("~anthropic.claude-opus-latest") is True
    assert _is_claude_reasoning_model("~anthropic.claude-fable-latest") is False
    assert _is_claude_reasoning_model("anthropic.claude") is False
    assert _is_claude_reasoning_model("") is False


def test_classify_gemini_thinking_family_accepts_tilde_aliases():
    from open_webui_openrouter_pipe.models.registry import _classify_gemini_thinking_family

    assert _classify_gemini_thinking_family("google.gemini-2.5-pro") == "gemini-2.5"
    assert _classify_gemini_thinking_family("google.gemini-2.5") == "gemini-2.5"
    assert _classify_gemini_thinking_family("~google.gemini-2.5-flash") == "gemini-2.5"
    assert _classify_gemini_thinking_family("google.gemini-2.55") is None
    assert _classify_gemini_thinking_family("~google.gemini-pro-latest") is None
    assert _classify_gemini_thinking_family("google.gemini-2.5:free") is None


_DOCS = Path(__file__).resolve().parents[1] / "docs"


@pytest.mark.asyncio
async def test_presets_served_from_the_presets_endpoints_survive_a_failed_read_and_reread_without_models(monkeypatch):
    catalog = {
        "data": [
            {"id": "example/one", "name": "One"},
            {"id": "example/two", "name": "Two"},
        ]
    }
    catalog_listing_alpha = {"data": [*catalog["data"], {"id": "@preset/alpha", "name": "Alpha"}]}
    preset_list = {
        "data": [
            {
                "slug": "alpha",
                "name": "Alpha",
                "status": "active",
                "designated_version_id": "v1",
            },
            {
                "slug": "beta",
                "name": "Beta",
                "status": "active",
                "designated_version_id": "v2",
            },
            {
                "slug": "gamma",
                "name": "Gamma",
                "status": "active",
                "designated_version_id": "v3",
            },
        ],
        "total_count": 3,
    }

    def preset_detail(slug: str, version: str, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "data": {
                "slug": slug,
                "name": slug.capitalize(),
                "status": "active",
                "designated_version_id": version,
                "designated_version": {"id": version, "config": config},
            }
        }

    details = {
        "alpha": preset_detail("alpha", "v1", {"model": "example/one"}),
        "beta": preset_detail("beta", "v2", {"models": ["example/two", "example/one"]}),
        "gamma": preset_detail("gamma", "v3", {"model": "example/ghost"}),
    }
    reg = ow.OpenRouterModelRegistry
    reg._models = []
    reg._specs = {}
    reg._id_map = {}
    reg._last_fetch = 0
    reg._next_refresh_after = 0
    import aiohttp
    from aioresponses import aioresponses

    from open_webui_openrouter_pipe import Pipe

    pipe = Pipe()

    def served() -> set[str]:
        return {m["id"] for m in reg.list_models()}

    def gate_reasons(model_id: str) -> list[str]:
        catalog_norm_ids = {
            m["norm_id"]
            for m in reg.list_models()
            if isinstance(m, dict) and m.get("norm_id")
        }
        return pipe._model_restriction_reasons(
            model_id,
            valves=pipe.valves,
            allowlist_norm_ids=set(),
            catalog_norm_ids=catalog_norm_ids,
        )

    def mock_presets(
        http: Any,
        *,
        list_payload: dict[str, Any] | None,
        error: Exception | None = None,
    ) -> None:
        if error is not None:
            http.get("https://api/presets?offset=0&limit=100", exception=error)
            return
        assert list_payload is not None
        http.get("https://api/presets?offset=0&limit=100", payload=list_payload)
        for slug, payload in details.items():
            http.get(f"https://api/presets/{slug}", payload=payload)

    with aioresponses() as http:
        http.get("https://api/models", payload=catalog_listing_alpha)
        http.get("https://api/endpoints/zdr", payload={"data": []}, repeat=True)
        mock_presets(http, list_payload=preset_list)
        async with aiohttp.ClientSession() as session:
            await reg.ensure_loaded(
                session,
                base_url="https://api",
                api_key="k",
                cache_seconds=3600,
                logger=logging.getLogger("test"),
            )
        served_now = served()
        assert {"example.one", "example.two", "@preset.alpha", "@preset.beta"} <= served_now
        assert "@preset.gamma" not in served_now, (
            "a preset whose base the catalog does not list is left out,"
            " the way OpenRouter leaves it out"
        )
        for piped in ("@preset.alpha", "@preset.beta"):
            reasons = gate_reasons(piped)
            assert "not_in_catalog" not in reasons, (piped, reasons)
        assert len(reg._models) == len({m["norm_id"] for m in reg._models}), (
            "a preset OpenRouter's own row serves was served a second time"
        )

    reg._last_fetch = 0
    reg._next_refresh_after = 0
    with aioresponses() as http:
        http.get("https://api/models", payload=catalog)
        http.get("https://api/endpoints/zdr", payload={"data": []}, repeat=True)
        mock_presets(http, list_payload=None, error=RuntimeError("presets list unavailable"))
        async with aiohttp.ClientSession() as session:
            await reg.ensure_loaded(
                session,
                base_url="https://api",
                api_key="k",
                cache_seconds=3600,
                logger=logging.getLogger("test"),
            )
    assert {"@preset.alpha", "@preset.beta"} <= served(), (
        "a failed presets read dropped the presets the previous refresh served,"
        " OpenRouter's own row (alpha) or the one the pipe built (beta)"
    )
    for piped in ("@preset.alpha", "@preset.beta"):
        reasons = gate_reasons(piped)
        assert "not_in_catalog" not in reasons, (piped, reasons)

    with aioresponses() as http:
        mock_presets(http, list_payload=None, error=RuntimeError("presets list still unavailable"))
        async with aiohttp.ClientSession() as session:
            await reg.ensure_loaded(
                session,
                base_url="https://api",
                api_key="k",
                cache_seconds=3600,
                logger=logging.getLogger("test"),
            )
    assert {"@preset.alpha", "@preset.beta"} <= served()

    later = time.time() + 60
    monkeypatch.setattr(time, "time", lambda: later)
    with aioresponses() as http:
        mock_presets(http, list_payload=preset_list)
        async with aiohttp.ClientSession() as session:
            await reg.ensure_loaded(
                session,
                base_url="https://api",
                api_key="k",
                cache_seconds=3600,
                logger=logging.getLogger("test"),
            )
    assert {"@preset.alpha", "@preset.beta"} <= served()
    touched = [str(key) for key in http.requests]
    assert [key for key in touched if "/models" in key] == [], (
        "the owed retry must re-read the presets without refetching the catalog"
    )
    assert [key for key in touched if "/presets?" in key], (
        "the owed retry never re-read the presets"
    )

    after_window = later + 3601
    monkeypatch.setattr(time, "time", lambda: after_window)
    with aioresponses() as http:
        http.get("https://api/models", payload=catalog_listing_alpha)
        http.get("https://api/endpoints/zdr", payload={"data": []}, repeat=True)
        mock_presets(http, list_payload=None, error=RuntimeError("presets list unavailable again"))
        async with aiohttp.ClientSession() as session:
            await reg.ensure_loaded(
                session,
                base_url="https://api",
                api_key="k",
                cache_seconds=3600,
                logger=logging.getLogger("test"),
            )
    assert {"@preset.alpha", "@preset.beta"} <= served()

    retry_time = after_window + 60
    monkeypatch.setattr(time, "time", lambda: retry_time)
    without_alpha = {
        "data": [item for item in preset_list["data"] if item["slug"] != "alpha"],
        "total_count": 2,
    }
    with aioresponses() as http:
        mock_presets(http, list_payload=without_alpha)
        async with aiohttp.ClientSession() as session:
            await reg.ensure_loaded(
                session,
                base_url="https://api",
                api_key="k",
                cache_seconds=3600,
                logger=logging.getLogger("test"),
            )
    assert "@preset.alpha" not in served(), (
        "a preset the presets list no longer returns stayed in the picker"
        " because OpenRouter's own row had served it"
    )
    assert "@preset.alpha" not in reg._chat_catalog_norms
    assert "@preset.beta" in served()


def _listed_preset(slug: str, version: str, status: str = "active") -> dict[str, Any]:
    return {
        "slug": slug,
        "name": slug.capitalize(),
        "status": status,
        "designated_version_id": version,
    }


def _preset_record(slug: str, version: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "data": {
            "slug": slug,
            "name": slug.capitalize(),
            "status": "active",
            "designated_version_id": version,
            "designated_version": {"id": version, "config": config},
        }
    }


async def _load_catalog(
    *,
    models: dict[str, Any] | None,
    presets: list[dict[str, Any]] | None,
    records: dict[str, dict[str, Any]],
) -> None:
    import aiohttp
    from aioresponses import aioresponses

    with aioresponses() as http:
        if models is not None:
            http.get("https://api/models", payload=models)
            http.get("https://api/endpoints/zdr", payload={"data": []}, repeat=True)
        presets_url = "https://api/presets?offset=0&limit=100"
        if presets is None:
            http.get(presets_url, exception=RuntimeError("presets list unavailable"))
        else:
            http.get(presets_url, payload={"data": presets, "total_count": len(presets)})
            for slug, payload in records.items():
                http.get(f"https://api/presets/{slug}", payload=payload)
        async with aiohttp.ClientSession() as session:
            await ow.OpenRouterModelRegistry.ensure_loaded(
                session,
                base_url="https://api",
                api_key="k",
                cache_seconds=3600,
                logger=logging.getLogger("test"),
            )


def _force_full_refresh() -> None:
    ow.OpenRouterModelRegistry._last_fetch = 0
    ow.OpenRouterModelRegistry._next_refresh_after = 0


@pytest.mark.asyncio
async def test_a_preset_offers_file_upload_exactly_when_its_base_model_does():
    from open_webui_openrouter_pipe.models.blocklists import DIRECT_UPLOAD_BLOCKLIST
    from open_webui_openrouter_pipe.models.registry import sanitize_model_id

    blocked_base = sorted(DIRECT_UPLOAD_BLOCKLIST)[0]
    catalog = {
        "data": [
            {"id": blocked_base, "name": "Blocked base"},
            {"id": "example/open", "name": "Open base"},
            {"id": "@preset/own-blocked", "name": "Own blocked"},
            {"id": "@preset/own-open", "name": "Own open"},
        ]
    }
    presets = [
        _listed_preset("own-blocked", "v1"),
        _listed_preset("own-open", "v2"),
        _listed_preset("built-blocked", "v3"),
        _listed_preset("built-open", "v4"),
    ]
    records = {
        "own-blocked": _preset_record("own-blocked", "v1", {"model": blocked_base}),
        "own-open": _preset_record("own-open", "v2", {"model": "example/open"}),
        "built-blocked": _preset_record(
            "built-blocked", "v3", {"models": [blocked_base, "example/open"]}
        ),
        "built-open": _preset_record("built-open", "v4", {"model": "example/open"}),
    }
    reg = ow.OpenRouterModelRegistry
    expected = {
        sanitize_model_id(blocked_base): (False, False),
        "example.open": (True, True),
        "@preset.own-blocked": (False, False),
        "@preset.own-open": (True, True),
        "@preset.built-blocked": (False, False),
        "@preset.built-open": (True, True),
    }

    def offered() -> dict[str, tuple[bool, bool]]:
        return {
            m["id"]: (
                m["capabilities"]["file_upload"],
                ow.ModelFamily.supports("file_input", m["id"]),
            )
            for m in reg.list_models()
        }

    await _load_catalog(models=catalog, presets=None, records={})
    await _load_catalog(models=None, presets=presets, records=records)
    assert offered() == expected, (
        "after the presets-only retry a preset's file upload did not follow its base model"
    )

    _force_full_refresh()
    await _load_catalog(models=catalog, presets=presets, records=records)
    assert offered() == expected, (
        "after a full refresh a preset's file upload did not follow its base model"
    )


@pytest.mark.asyncio
async def test_a_switched_off_preset_stays_out_of_the_catalog_on_every_path_and_is_logged_once(caplog):
    catalog = {
        "data": [
            {"id": "example/one", "name": "One"},
            {"id": "example/two", "name": "Two"},
            {"id": "@preset/alpha", "name": "Alpha"},
            {"id": "@preset/beta", "name": "Beta"},
        ]
    }
    presets = [_listed_preset("alpha", "v1", "disabled"), _listed_preset("beta", "v2")]
    records = {"beta": _preset_record("beta", "v2", {"model": "example/two"})}
    reg = ow.OpenRouterModelRegistry

    def holders(norm_id: str) -> list[str]:
        return [
            where
            for where, found in (
                ("picker", norm_id in {m["id"] for m in reg.list_models()}),
                ("specs", norm_id in reg._specs),
                ("id map", norm_id in reg._id_map),
                ("chat norms", norm_id in reg._chat_catalog_norms),
            )
            if found
        ]

    everywhere = ["picker", "specs", "id map", "chat norms"]

    await _load_catalog(models=catalog, presets=None, records={})
    assert holders("@preset.alpha") == everywhere

    await _load_catalog(models=None, presets=presets, records=records)
    assert holders("@preset.alpha") == [], "the presets-only retry kept a disabled preset"
    assert holders("@preset.beta") == everywhere

    _force_full_refresh()
    await _load_catalog(models=catalog, presets=presets, records=records)
    assert holders("@preset.alpha") == [], (
        "a full refresh served the disabled preset's row from the model list"
    )
    assert holders("@preset.beta") == everywhere

    _force_full_refresh()
    await _load_catalog(models=catalog, presets=None, records={})
    assert holders("@preset.alpha") == [], (
        "a failed presets read brought a preset known to be disabled back"
    )
    assert holders("@preset.beta") == everywhere

    assert len([r for r in caplog.records if "alpha" in r.getMessage()]) == 1
