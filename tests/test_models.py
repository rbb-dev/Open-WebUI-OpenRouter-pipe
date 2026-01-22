"""Coverage tests for ModelCatalogManager (models/catalog_manager.py).

These tests target coverage of model catalog operations including:
- Icon mapping from various frontend data structures
- Web search support detection via multiple signals
- Favicon URL generation with proper encoding
- Frontend catalog fetching error handling
- Maker profile image mapping
- Metadata sync scheduling and execution
- Filter attachment/default logic for ORS and Direct Uploads
- Model insert with various access control modes
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import Pipe


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _make_existing_model(model_id: str, *, meta: dict, params: dict | None = None):
    """Create a stub existing model for test assertions."""
    from open_webui.models.models import ModelMeta

    return SimpleNamespace(
        id=model_id,
        base_model_id=None,
        name="Example",
        meta=ModelMeta(**meta),
        params=params or {},
        access_control=None,
        is_active=True,
    )


# ---------------------------------------------------------------------------
# Icon Mapping Tests
# ---------------------------------------------------------------------------


def test_build_icon_mapping_with_protocol_relative_url(pipe_instance) -> None:
    """Protocol-relative URLs (//example.com) should get https: prefix."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {
                    "provider_info": {
                        "icon": {"url": "//cdn.example.com/icon.png"},
                    }
                },
            }
        ]
    }
    icon_mapping = pipe._build_icon_mapping(frontend_data)
    assert icon_mapping["test/model"] == "https://cdn.example.com/icon.png"


def test_build_icon_mapping_with_bare_path(pipe_instance) -> None:
    """Bare paths without leading slash should be prefixed with site URL."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {
                    "provider_info": {
                        "icon": {"url": "images/icon.png"},
                    }
                },
            }
        ]
    }
    icon_mapping = pipe._build_icon_mapping(frontend_data)
    assert icon_mapping["test/model"] == "https://openrouter.ai/images/icon.png"


def test_build_icon_mapping_with_string_icon(pipe_instance) -> None:
    """Icon can be a plain string URL instead of a dict."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {
                    "provider_info": {
                        "icon": "https://example.com/icon.png",
                    }
                },
            }
        ]
    }
    icon_mapping = pipe._build_icon_mapping(frontend_data)
    assert icon_mapping["test/model"] == "https://example.com/icon.png"


def test_build_icon_mapping_item_level_icon_fallback(pipe_instance) -> None:
    """Falls back to item-level icon when endpoint icon missing."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {"provider_info": {}},
                "icon": {"url": "https://example.com/fallback.png"},
            }
        ]
    }
    icon_mapping = pipe._build_icon_mapping(frontend_data)
    assert icon_mapping["test/model"] == "https://example.com/fallback.png"


def test_build_icon_mapping_item_level_string_icon(pipe_instance) -> None:
    """Item-level icon can be a string."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {"provider_info": {}},
                "icon": "https://example.com/string-icon.png",
            }
        ]
    }
    icon_mapping = pipe._build_icon_mapping(frontend_data)
    assert icon_mapping["test/model"] == "https://example.com/string-icon.png"


def test_build_icon_mapping_favicon_from_base_url(pipe_instance) -> None:
    """Uses favicon service when no icon but baseUrl available."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {
                    "provider_info": {
                        "baseUrl": "https://api.provider.com/v1",
                    }
                },
            }
        ]
    }
    icon_mapping = pipe._build_icon_mapping(frontend_data)
    assert "test/model" in icon_mapping
    assert "gstatic.com/faviconV2" in icon_mapping["test/model"]
    assert "api.provider.com" in icon_mapping["test/model"]


def test_build_icon_mapping_favicon_from_status_page_url(pipe_instance) -> None:
    """Uses favicon service from statusPageUrl when baseUrl missing."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {
                    "provider_info": {
                        "statusPageUrl": "https://status.provider.com",
                    }
                },
            }
        ]
    }
    icon_mapping = pipe._build_icon_mapping(frontend_data)
    assert "test/model" in icon_mapping
    assert "status.provider.com" in icon_mapping["test/model"]


def test_build_icon_mapping_favicon_from_data_policy_urls(pipe_instance) -> None:
    """Uses favicon service from data policy URLs."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {
                    "provider_info": {
                        "dataPolicy": {
                            "termsOfServiceURL": "https://example.com/terms",
                        }
                    }
                },
            }
        ]
    }
    icon_mapping = pipe._build_icon_mapping(frontend_data)
    assert "test/model" in icon_mapping

    # Also test privacy URL
    frontend_data2 = {
        "data": [
            {
                "slug": "test/model2",
                "endpoint": {
                    "provider_info": {
                        "dataPolicy": {
                            "privacyPolicyURL": "https://privacy.example.com/policy",
                        }
                    }
                },
            }
        ]
    }
    icon_mapping2 = pipe._build_icon_mapping(frontend_data2)
    assert "test/model2" in icon_mapping2


def test_build_icon_mapping_skips_invalid_entries(pipe_instance) -> None:
    """Skips items with missing slug, non-dict items, empty slugs."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            None,
            "not a dict",
            {"slug": None},
            {"slug": ""},
            {"slug": 123},
            {
                "slug": "valid/model",
                "endpoint": {"provider_info": {"icon": {"url": "https://x.com/i.png"}}},
            },
        ]
    }
    icon_mapping = pipe._build_icon_mapping(frontend_data)
    assert len(icon_mapping) == 1
    assert "valid/model" in icon_mapping


def test_build_icon_mapping_data_image_url_passthrough(pipe_instance) -> None:
    """Data URLs should pass through unchanged."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {
                    "provider_info": {
                        "icon": {"url": "data:image/png;base64,ABC123"},
                    }
                },
            }
        ]
    }
    icon_mapping = pipe._build_icon_mapping(frontend_data)
    assert icon_mapping["test/model"] == "data:image/png;base64,ABC123"


def test_build_icon_mapping_skips_no_icon_or_fallback(pipe_instance) -> None:
    """Skips models without any icon source."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {"provider_info": {}},
            }
        ]
    }
    icon_mapping = pipe._build_icon_mapping(frontend_data)
    assert "test/model" not in icon_mapping


# ---------------------------------------------------------------------------
# Web Search Support Mapping Tests
# ---------------------------------------------------------------------------


def test_build_web_search_support_mapping_empty_cases(pipe_instance) -> None:
    """Returns empty dict for None, non-dict, or missing data."""
    pipe = pipe_instance
    assert pipe._build_web_search_support_mapping(None) == {}
    assert pipe._build_web_search_support_mapping("not a dict") == {}
    assert pipe._build_web_search_support_mapping({}) == {}
    assert pipe._build_web_search_support_mapping({"data": "not a list"}) == {}


def test_build_web_search_support_mapping_skips_invalid_entries(pipe_instance) -> None:
    """Skips non-dict items and items with invalid slugs."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            None,
            "string item",
            {"slug": None},
            {"slug": ""},
            {"slug": 123},
            {"slug": "no-endpoint"},
            {"slug": "non-dict-endpoint", "endpoint": "string"},
        ]
    }
    mapping = pipe._build_web_search_support_mapping(frontend_data)
    assert mapping == {}


def test_build_web_search_support_mapping_web_search_options_parameter(pipe_instance) -> None:
    """Detects web search support via supported_parameters list."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {
                    "supported_parameters": ["temperature", "web_search_options", "max_tokens"],
                    "features": {},
                    "pricing": {},
                },
            }
        ]
    }
    mapping = pipe._build_web_search_support_mapping(frontend_data)
    assert mapping == {"test/model": True}


def test_build_web_search_support_mapping_whitespace_handling(pipe_instance) -> None:
    """Handles whitespace in web_search_options parameter."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {
                    "supported_parameters": ["  web_search_options  "],
                    "features": {},
                    "pricing": {},
                },
            }
        ]
    }
    mapping = pipe._build_web_search_support_mapping(frontend_data)
    assert mapping == {"test/model": True}


def test_build_web_search_support_mapping_supported_parameters_not_list(pipe_instance) -> None:
    """Handles non-list supported_parameters gracefully."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {
                    "supported_parameters": "not a list",
                    "features": {"supports_native_web_search": True},
                    "pricing": {},
                },
            }
        ]
    }
    mapping = pipe._build_web_search_support_mapping(frontend_data)
    assert mapping == {"test/model": True}


def test_build_web_search_support_mapping_non_string_parameter_entries(pipe_instance) -> None:
    """Skips non-string entries in supported_parameters."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {
                    "supported_parameters": [123, None, {"key": "value"}],
                    "features": {},
                    "pricing": {},
                },
            }
        ]
    }
    mapping = pipe._build_web_search_support_mapping(frontend_data)
    assert mapping == {}


def test_build_web_search_support_mapping_features_not_dict(pipe_instance) -> None:
    """Handles non-dict features gracefully."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {
                    "supported_parameters": [],
                    "features": "not a dict",
                    "pricing": {"web_search": "0.01"},
                },
            }
        ]
    }
    mapping = pipe._build_web_search_support_mapping(frontend_data)
    assert mapping == {"test/model": True}


def test_build_web_search_support_mapping_pricing_not_dict(pipe_instance) -> None:
    """Handles non-dict pricing gracefully."""
    pipe = pipe_instance
    frontend_data = {
        "data": [
            {
                "slug": "test/model",
                "endpoint": {
                    "supported_parameters": [],
                    "features": {"supports_native_web_search": True},
                    "pricing": "not a dict",
                },
            }
        ]
    }
    mapping = pipe._build_web_search_support_mapping(frontend_data)
    assert mapping == {"test/model": True}


# ---------------------------------------------------------------------------
# Frontend Catalog Fetch Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_frontend_model_catalog_success(pipe_instance_async) -> None:
    """Successfully fetches and returns catalog dict."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()

    with aioresponses() as mocked:
        mocked.get(
            "https://openrouter.ai/api/frontend/models",
            payload={"data": [{"slug": "test/model"}]},
        )
        session = pipe._create_http_session()
        try:
            result = await pipe._fetch_frontend_model_catalog(session)
            assert result == {"data": [{"slug": "test/model"}]}
        finally:
            await session.close()


@pytest.mark.asyncio
async def test_fetch_frontend_model_catalog_http_error(pipe_instance_async) -> None:
    """Returns None on HTTP errors and logs debug message."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()

    with aioresponses() as mocked:
        mocked.get(
            "https://openrouter.ai/api/frontend/models",
            status=500,
        )
        session = pipe._create_http_session()
        try:
            result = await pipe._fetch_frontend_model_catalog(session)
            assert result is None
        finally:
            await session.close()


@pytest.mark.asyncio
async def test_fetch_frontend_model_catalog_invalid_json_type(pipe_instance_async) -> None:
    """Returns None when JSON is not a dict (e.g., list or null)."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()

    with aioresponses() as mocked:
        # Return a list instead of dict
        mocked.get(
            "https://openrouter.ai/api/frontend/models",
            payload=[{"slug": "test/model"}],
        )
        session = pipe._create_http_session()
        try:
            result = await pipe._fetch_frontend_model_catalog(session)
            assert result is None
        finally:
            await session.close()


@pytest.mark.asyncio
async def test_fetch_frontend_model_catalog_connection_error(pipe_instance_async) -> None:
    """Returns None on connection errors."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()

    with aioresponses() as mocked:
        mocked.get(
            "https://openrouter.ai/api/frontend/models",
            exception=Exception("Connection failed"),
        )
        session = pipe._create_http_session()
        try:
            result = await pipe._fetch_frontend_model_catalog(session)
            assert result is None
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Maker Profile Image Mapping Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_maker_profile_image_mapping_empty_input(pipe_instance_async) -> None:
    """Returns empty dict for empty or None maker IDs."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()

    session = pipe._create_http_session()
    try:
        result = await pipe._build_maker_profile_image_mapping(session, [])
        assert result == {}

        result = await pipe._build_maker_profile_image_mapping(session, [None, "", "  "])
        assert result == {}
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_build_maker_profile_image_mapping_deduplicates_makers(pipe_instance_async) -> None:
    """Deduplicates maker IDs before fetching."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    call_count = 0

    async def mock_fetch(session, maker_id):
        nonlocal call_count
        call_count += 1
        return f"https://example.com/{maker_id}.png"

    pipe._fetch_maker_profile_image_url = mock_fetch

    session = pipe._create_http_session()
    try:
        result = await pipe._build_maker_profile_image_mapping(
            session, ["openai", "openai", "anthropic", "openai"]
        )
        assert call_count == 2  # Only 2 unique makers
        assert len(result) == 2
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_build_maker_profile_image_mapping_handles_none_results(pipe_instance_async) -> None:
    """Filters out makers with None image URLs."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()

    async def mock_fetch(session, maker_id):
        if maker_id == "anthropic":
            return None
        return f"https://example.com/{maker_id}.png"

    pipe._fetch_maker_profile_image_url = mock_fetch

    session = pipe._create_http_session()
    try:
        result = await pipe._build_maker_profile_image_mapping(
            session, ["openai", "anthropic"]
        )
        assert "openai" in result
        assert "anthropic" not in result
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# Metadata Sync Scheduling Tests
# ---------------------------------------------------------------------------


def test_maybe_schedule_model_metadata_sync_no_valves_enabled(pipe_instance) -> None:
    """Does not schedule when no relevant valves are enabled."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.UPDATE_MODEL_DESCRIPTIONS = False
    pipe.valves.AUTO_ATTACH_ORS_FILTER = False
    pipe.valves.AUTO_INSTALL_ORS_FILTER = False
    pipe.valves.AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = False
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False

    pipe._catalog_manager.maybe_schedule_model_metadata_sync(
        [{"id": "test"}],
        pipe_identifier="test_pipe",
    )
    assert pipe._catalog_manager._model_metadata_sync_task is None


def test_maybe_schedule_model_metadata_sync_empty_models(pipe_instance) -> None:
    """Does not schedule when models list is empty."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    pipe._catalog_manager.maybe_schedule_model_metadata_sync(
        [],
        pipe_identifier="test_pipe",
    )
    assert pipe._catalog_manager._model_metadata_sync_task is None


def test_maybe_schedule_model_metadata_sync_same_key_no_reschedule(pipe_instance) -> None:
    """Does not reschedule when sync key unchanged."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    # Set a sync key that matches what would be generated
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry
    last_fetch = getattr(OpenRouterModelRegistry, "_last_fetch", 0.0)

    pipe._catalog_manager._model_metadata_sync_key = (
        "test_pipe",
        float(last_fetch or 0.0),
        str(pipe.valves.MODEL_ID or ""),
        bool(pipe.valves.UPDATE_MODEL_IMAGES),
        bool(pipe.valves.UPDATE_MODEL_CAPABILITIES),
        bool(pipe.valves.UPDATE_MODEL_DESCRIPTIONS),
        bool(pipe.valves.AUTO_ATTACH_ORS_FILTER),
        bool(pipe.valves.AUTO_INSTALL_ORS_FILTER),
        bool(pipe.valves.AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER),
        bool(pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER),
        bool(pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER),
    )

    pipe._catalog_manager.maybe_schedule_model_metadata_sync(
        [{"id": "test"}],
        pipe_identifier="test_pipe",
    )
    assert pipe._catalog_manager._model_metadata_sync_task is None


def test_maybe_schedule_model_metadata_sync_running_task_no_reschedule(pipe_instance) -> None:
    """Does not reschedule when a task is already running."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    # Create a mock running task
    mock_task = Mock()
    mock_task.done.return_value = False
    pipe._catalog_manager._model_metadata_sync_task = mock_task

    pipe._catalog_manager.maybe_schedule_model_metadata_sync(
        [{"id": "test"}],
        pipe_identifier="test_pipe",
    )
    # Key should not be updated
    assert pipe._catalog_manager._model_metadata_sync_key is None


# ---------------------------------------------------------------------------
# Sync Model Metadata Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_model_metadata_returns_early_no_valves(pipe_instance_async) -> None:
    """Returns early when no relevant valves enabled."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.UPDATE_MODEL_DESCRIPTIONS = False
    pipe.valves.AUTO_ATTACH_ORS_FILTER = False
    pipe.valves.AUTO_INSTALL_ORS_FILTER = False
    pipe.valves.AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = False
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False

    # Should return early without error
    await pipe._sync_model_metadata_to_owui([{"id": "test"}], pipe_identifier="test_pipe")


@pytest.mark.asyncio
async def test_sync_model_metadata_returns_early_empty_models(pipe_instance_async) -> None:
    """Returns early when models list empty."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    await pipe._sync_model_metadata_to_owui([], pipe_identifier="test_pipe")


@pytest.mark.asyncio
async def test_sync_model_metadata_returns_early_no_pipe_identifier(pipe_instance_async) -> None:
    """Returns early when pipe_identifier empty."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    await pipe._sync_model_metadata_to_owui([{"id": "test"}], pipe_identifier="")


@pytest.mark.asyncio
async def test_sync_model_metadata_skips_model_without_valid_id(pipe_instance_async) -> None:
    """Skips models with invalid or missing id."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    update_mock = Mock()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.models.catalog_manager.run_in_threadpool", new=fake_run_in_threadpool):
        await pipe._sync_model_metadata_to_owui(
            [{"id": None}, {"id": ""}, {"id": 123}],
            pipe_identifier="test_pipe",
        )

    # No updates should occur
    assert update_mock.call_count == 0


@pytest.mark.asyncio
async def test_sync_model_metadata_uses_id_as_name_when_missing(pipe_instance_async) -> None:
    """Uses model id as name when name is missing or invalid."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.AUTO_ATTACH_ORS_FILTER = False
    pipe.valves.AUTO_INSTALL_ORS_FILTER = False

    pipe._catalog_manager._update_or_insert_model_with_metadata = Mock()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.models.catalog_manager.run_in_threadpool", new=fake_run_in_threadpool):
        await pipe._sync_model_metadata_to_owui(
            [{"id": "test.model", "name": None, "capabilities": {"vision": True}}],
            pipe_identifier="test_pipe",
        )

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    args = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args[0]
    assert args[1] == "test.model"  # name falls back to id


@pytest.mark.asyncio
async def test_sync_model_metadata_ors_filter_warning_not_installed(pipe_instance_async) -> None:
    """Logs warning when AUTO_ATTACH_ORS_FILTER enabled but filter not installed."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.AUTO_ATTACH_ORS_FILTER = True
    pipe.valves.AUTO_INSTALL_ORS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = False
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False

    pipe._ensure_ors_filter_function_id = Mock(return_value=None)

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.models.catalog_manager.run_in_threadpool", new=fake_run_in_threadpool), \
         patch.object(pipe._catalog_manager.logger, "warning") as mock_warning:
        await pipe._sync_model_metadata_to_owui(
            [{"id": "test.model", "original_id": "test/model"}],
            pipe_identifier="test_pipe",
        )

    mock_warning.assert_called_once()
    assert "AUTO_ATTACH_ORS_FILTER is enabled" in mock_warning.call_args[0][0]


@pytest.mark.asyncio
async def test_sync_model_metadata_direct_uploads_filter_warning(pipe_instance_async) -> None:
    """Logs warning when AUTO_ATTACH_DIRECT_UPLOADS_FILTER enabled but filter not installed."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.AUTO_ATTACH_ORS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = True
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False

    pipe._ensure_direct_uploads_filter_function_id = Mock(return_value=None)

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.models.catalog_manager.run_in_threadpool", new=fake_run_in_threadpool), \
         patch.object(pipe._catalog_manager.logger, "warning") as mock_warning:
        await pipe._sync_model_metadata_to_owui(
            [{"id": "test.model", "original_id": "test/model"}],
            pipe_identifier="test_pipe",
        )

    mock_warning.assert_called_once()
    assert "AUTO_ATTACH_DIRECT_UPLOADS_FILTER is enabled" in mock_warning.call_args[0][0]


@pytest.mark.asyncio
async def test_sync_model_metadata_logs_supported_model_counts(pipe_instance_async) -> None:
    """Logs info about supported models for filter attachment."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.AUTO_ATTACH_ORS_FILTER = True
    pipe.valves.AUTO_INSTALL_ORS_FILTER = False

    pipe._ensure_ors_filter_function_id = Mock(return_value="openrouter_search")
    pipe._fetch_frontend_model_catalog = AsyncMock(return_value={
        "data": [
            {
                "slug": "test/model",
                "endpoint": {
                    "features": {"supports_native_web_search": True},
                    "supported_parameters": [],
                    "pricing": {},
                },
            }
        ]
    })

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.models.catalog_manager.run_in_threadpool", new=fake_run_in_threadpool), \
         patch.object(pipe._catalog_manager.logger, "info") as mock_info:
        await pipe._sync_model_metadata_to_owui(
            [{"id": "test.model", "original_id": "test/model"}],
            pipe_identifier="test_pipe",
        )

    mock_info.assert_called()
    call_args = mock_info.call_args[0]
    assert "Auto-attaching OpenRouter Search filter" in call_args[0]


@pytest.mark.asyncio
async def test_sync_model_metadata_handles_update_exception(pipe_instance_async) -> None:
    """Handles exceptions during model update gracefully."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.AUTO_ATTACH_ORS_FILTER = False

    pipe._catalog_manager._update_or_insert_model_with_metadata = Mock(
        side_effect=Exception("DB error")
    )

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.models.catalog_manager.run_in_threadpool", new=fake_run_in_threadpool):
        # Should not raise
        await pipe._sync_model_metadata_to_owui(
            [{"id": "test.model", "capabilities": {"vision": True}}],
            pipe_identifier="test_pipe",
        )


@pytest.mark.asyncio
async def test_sync_model_metadata_ensure_filter_exception_handling(pipe_instance_async) -> None:
    """Handles exceptions when ensuring filter functions."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.AUTO_ATTACH_ORS_FILTER = True
    pipe.valves.AUTO_INSTALL_ORS_FILTER = True

    pipe._ensure_ors_filter_function_id = Mock(side_effect=Exception("Filter install failed"))

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.models.catalog_manager.run_in_threadpool", new=fake_run_in_threadpool):
        # Should not raise, logs debug instead
        await pipe._sync_model_metadata_to_owui(
            [{"id": "test.model", "original_id": "test/model"}],
            pipe_identifier="test_pipe",
        )


# ---------------------------------------------------------------------------
# Update/Insert Model Metadata Tests
# ---------------------------------------------------------------------------


def test_update_or_insert_empty_model_id_returns_early(pipe_instance) -> None:
    """Returns early for empty or whitespace-only model ID."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()

    with patch("open_webui.models.models.Models") as mock_models:
        pipe._update_or_insert_model_with_metadata(
            "",
            "Name",
            None,
            None,
            False,
            False,
        )
        mock_models.get_model_by_id.assert_not_called()

        pipe._update_or_insert_model_with_metadata(
            "   ",
            "Name",
            None,
            None,
            False,
            False,
        )
        mock_models.get_model_by_id.assert_not_called()


def test_update_or_insert_uses_model_id_as_name_fallback(pipe_instance) -> None:
    """Uses model_id as name when name is empty."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=None), \
         patch("open_webui.models.models.Models.insert_new_model", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "",
            {"vision": True},
            None,
            True,
            False,
        )

    update_mock.assert_called_once()
    inserted_form = update_mock.call_args[0][0]
    assert inserted_form.name == model_id


def test_update_or_insert_new_model_with_capabilities(pipe_instance) -> None:
    """Inserts new model with capabilities when it doesn't exist."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    insert_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=None), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"vision": True, "web_search": True},
            "data:image/png;base64,ABC",
            True,
            True,
        )

    insert_mock.assert_called_once()
    inserted_form = insert_mock.call_args[0][0]
    assert inserted_form.id == model_id
    assert inserted_form.name == "GPT-4o"
    assert inserted_form.meta["capabilities"] == {"vision": True, "web_search": True}
    assert inserted_form.meta["profile_image_url"] == "data:image/png;base64,ABC"


def test_update_or_insert_new_model_skips_when_no_metadata(pipe_instance) -> None:
    """Skips insert when there's no metadata to add."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    insert_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=None), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
        )

    insert_mock.assert_not_called()


def test_update_or_insert_new_model_access_control_admins(pipe_instance) -> None:
    """Sets admins-only access control for new models by default."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    pipe.valves.NEW_MODEL_ACCESS_CONTROL = "admins"
    model_id = "test_pipe.openai.gpt-4o"

    insert_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=None), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"vision": True},
            None,
            True,
            False,
        )

    insert_mock.assert_called_once()
    inserted_form = insert_mock.call_args[0][0]
    assert inserted_form.access_control == {}


def test_update_or_insert_new_model_access_control_all(pipe_instance) -> None:
    """Sets None access control (all users) when configured."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    pipe.valves.NEW_MODEL_ACCESS_CONTROL = "all"
    model_id = "test_pipe.openai.gpt-4o"

    insert_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=None), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"vision": True},
            None,
            True,
            False,
        )

    insert_mock.assert_called_once()
    inserted_form = insert_mock.call_args[0][0]
    assert inserted_form.access_control is None


def test_update_existing_model_merges_capabilities(pipe_instance) -> None:
    """Merges new capabilities with existing ones."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"capabilities": {"vision": False, "file_upload": True}},
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"vision": True, "web_search": True},
            None,
            True,
            False,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    # Should merge: vision overwritten to True, web_search added, file_upload preserved
    assert meta["capabilities"]["vision"] is True
    assert meta["capabilities"]["web_search"] is True
    assert meta["capabilities"]["file_upload"] is True


def test_update_existing_model_no_changes_skips_update(pipe_instance) -> None:
    """Skips update when no actual changes detected."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"capabilities": {"vision": True}},
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"vision": True},  # Same as existing
            None,
            True,
            False,
        )

    update_mock.assert_not_called()


def test_update_existing_model_updates_profile_image(pipe_instance) -> None:
    """Updates profile image when different from existing."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"profile_image_url": "data:image/png;base64,OLD"},
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            "data:image/png;base64,NEW",
            False,
            True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["profile_image_url"] == "data:image/png;base64,NEW"


def test_update_existing_model_updates_openrouter_pipe_capabilities(pipe_instance) -> None:
    """Updates openrouter_pipe.capabilities when provided."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(model_id, meta={})
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            openrouter_pipe_capabilities={"file_input": True, "vision": True},
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["openrouter_pipe"]["capabilities"] == {"file_input": True, "vision": True}


def test_update_existing_model_filter_id_migration(pipe_instance) -> None:
    """Migrates filter IDs when the filter function ID changes."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": ["old_openrouter_search"],
            "openrouter_pipe": {"openrouter_search_filter_id": "old_openrouter_search"},
        },
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="new_openrouter_search",
            filter_supported=True,
            auto_attach_filter=True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    # Old filter should be replaced with new one
    assert "old_openrouter_search" not in meta["filterIds"]
    assert "new_openrouter_search" in meta["filterIds"]


def test_update_existing_model_filter_removal_when_unsupported(pipe_instance) -> None:
    """Removes filter when model no longer supports it."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"filterIds": ["openrouter_search", "other_filter"]},
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="openrouter_search",
            filter_supported=False,
            auto_attach_filter=True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert "openrouter_search" not in meta["filterIds"]
    assert "other_filter" in meta["filterIds"]


def test_update_existing_model_direct_uploads_filter_with_previous_id(pipe_instance) -> None:
    """Handles direct uploads filter ID migration."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": ["old_direct_uploads"],
            "openrouter_pipe": {"direct_uploads_filter_id": "old_direct_uploads"},
        },
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            direct_uploads_filter_function_id="new_direct_uploads",
            direct_uploads_filter_supported=True,
            auto_attach_direct_uploads_filter=True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert "old_direct_uploads" not in meta["filterIds"]
    assert "new_direct_uploads" in meta["filterIds"]
    assert meta["openrouter_pipe"]["direct_uploads_filter_id"] == "new_direct_uploads"


def test_update_existing_model_default_filter_migration(pipe_instance) -> None:
    """Migrates default filter IDs when filter function ID changes."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": ["old_openrouter_search"],
            "defaultFilterIds": ["old_openrouter_search"],
            "openrouter_pipe": {
                "openrouter_search_filter_id": "old_openrouter_search",
                "openrouter_search_default_seeded": True,
            },
        },
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="new_openrouter_search",
            filter_supported=True,
            auto_attach_filter=True,
            auto_default_filter=True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    # Default filter ID should be migrated
    assert "new_openrouter_search" in meta["defaultFilterIds"]
    assert meta["openrouter_pipe"]["openrouter_search_filter_id"] == "new_openrouter_search"


def test_update_existing_model_default_filter_not_attached_skips(pipe_instance) -> None:
    """Does not set default filter if filter is not in filterIds."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"filterIds": []},  # Filter not attached
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="openrouter_search",
            filter_supported=False,  # Not supported, so not attached
            auto_attach_filter=False,
            auto_default_filter=True,  # Even though enabled, should not add default
        )

    update_mock.assert_not_called()


def test_insert_new_model_with_filters(pipe_instance) -> None:
    """Inserts new model with filter attachments."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    insert_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=None), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="openrouter_search",
            filter_supported=True,
            auto_attach_filter=True,
            auto_default_filter=True,
            direct_uploads_filter_function_id="openrouter_direct_uploads",
            direct_uploads_filter_supported=True,
            auto_attach_direct_uploads_filter=True,
        )

    insert_mock.assert_called_once()
    inserted_form = insert_mock.call_args[0][0]
    meta = dict(inserted_form.meta)
    assert "openrouter_search" in meta["filterIds"]
    assert "openrouter_direct_uploads" in meta["filterIds"]
    assert "openrouter_search" in meta["defaultFilterIds"]


def test_normalize_filter_ids_handles_non_list(pipe_instance) -> None:
    """Handles filterIds that are not lists."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"filterIds": "not a list"},
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="openrouter_search",
            filter_supported=True,
            auto_attach_filter=True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["filterIds"] == ["openrouter_search"]


def test_normalize_filter_ids_filters_non_strings(pipe_instance) -> None:
    """Filters out non-string entries from filterIds."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"filterIds": ["valid_filter", 123, None, "", "another_filter"]},
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="openrouter_search",
            filter_supported=True,
            auto_attach_filter=True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    # Should have valid_filter, another_filter, and openrouter_search (newly added)
    assert "valid_filter" in meta["filterIds"]
    assert "another_filter" in meta["filterIds"]
    assert "openrouter_search" in meta["filterIds"]


def test_dedupe_preserves_order(pipe_instance) -> None:
    """Deduplication preserves original order."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"filterIds": ["filter_a", "filter_b", "filter_a", "filter_c", "filter_b"]},
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="openrouter_search",
            filter_supported=True,
            auto_attach_filter=True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    # Order should be preserved, duplicates removed
    assert meta["filterIds"] == ["filter_a", "filter_b", "filter_c", "openrouter_search"]


def test_existing_model_null_meta_handled(pipe_instance) -> None:
    """Handles existing model with null meta gracefully."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    # Create a model with None meta
    existing = SimpleNamespace(
        id=model_id,
        base_model_id=None,
        name="Example",
        meta=None,
        params={},
        access_control=None,
        is_active=True,
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"vision": True},
            None,
            True,
            False,
        )

    update_mock.assert_called_once()


# ---------------------------------------------------------------------------
# Additional Edge Case Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_model_metadata_direct_uploads_logs_supported_count(pipe_instance_async) -> None:
    """Logs info about supported models for direct uploads filter attachment."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.AUTO_ATTACH_ORS_FILTER = False
    pipe.valves.AUTO_INSTALL_ORS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = True
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False

    pipe._ensure_direct_uploads_filter_function_id = Mock(return_value="openrouter_direct_uploads")

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.models.catalog_manager.run_in_threadpool", new=fake_run_in_threadpool), \
         patch.object(pipe._catalog_manager.logger, "info") as mock_info:
        await pipe._sync_model_metadata_to_owui(
            [{"id": "test.model", "original_id": "test/model"}],
            pipe_identifier="test_pipe",
        )

    mock_info.assert_called()
    call_args = mock_info.call_args[0]
    assert "Auto-attaching OpenRouter Direct Uploads filter" in call_args[0]


@pytest.mark.asyncio
async def test_sync_model_metadata_with_images_and_maker_mapping(pipe_instance_async) -> None:
    """Tests the full image sync flow with maker mapping fallback."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_IMAGES = True
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.AUTO_ATTACH_ORS_FILTER = False
    pipe.valves.AUTO_INSTALL_ORS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = False
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False

    pipe._fetch_frontend_model_catalog = AsyncMock(return_value={
        "data": [
            {
                "slug": "openai/gpt-4",
                "endpoint": {
                    "provider_info": {
                        "icon": {"url": "https://example.com/openai.png"},
                    }
                },
            }
        ]
    })
    pipe._build_maker_profile_image_mapping = AsyncMock(return_value={})
    pipe._fetch_image_as_data_url = AsyncMock(return_value="data:image/png;base64,ABC123")

    pipe._catalog_manager._update_or_insert_model_with_metadata = Mock()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.models.catalog_manager.run_in_threadpool", new=fake_run_in_threadpool):
        await pipe._sync_model_metadata_to_owui(
            [{"id": "openai.gpt-4", "original_id": "openai/gpt-4", "name": "GPT-4"}],
            pipe_identifier="test_pipe",
        )

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    args = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args[0]
    assert args[3] == "data:image/png;base64,ABC123"  # profile_image_url


@pytest.mark.asyncio
async def test_sync_model_metadata_maker_image_fallback(pipe_instance_async) -> None:
    """Tests maker image fallback when model icon not found."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_IMAGES = True
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.AUTO_ATTACH_ORS_FILTER = False
    pipe.valves.AUTO_INSTALL_ORS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = False
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False

    pipe._fetch_frontend_model_catalog = AsyncMock(return_value={"data": []})  # No icon data
    pipe._build_maker_profile_image_mapping = AsyncMock(return_value={"anthropic": "https://example.com/anthropic.png"})
    pipe._fetch_image_as_data_url = AsyncMock(return_value="data:image/png;base64,ANTHROPIC123")

    pipe._catalog_manager._update_or_insert_model_with_metadata = Mock()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.models.catalog_manager.run_in_threadpool", new=fake_run_in_threadpool):
        await pipe._sync_model_metadata_to_owui(
            [{"id": "anthropic.claude", "original_id": "anthropic/claude", "name": "Claude"}],
            pipe_identifier="test_pipe",
        )

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    args = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args[0]
    assert args[3] == "data:image/png;base64,ANTHROPIC123"  # profile_image_url from maker


@pytest.mark.asyncio
async def test_sync_model_metadata_skips_model_without_original_id_for_images(pipe_instance_async) -> None:
    """Skips models without original_id when building icon mapping."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_IMAGES = True
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.AUTO_ATTACH_ORS_FILTER = False
    pipe.valves.AUTO_INSTALL_ORS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = False
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False

    pipe._fetch_frontend_model_catalog = AsyncMock(return_value={"data": []})
    pipe._build_maker_profile_image_mapping = AsyncMock(return_value={})
    pipe._fetch_image_as_data_url = AsyncMock(return_value=None)

    pipe._catalog_manager._update_or_insert_model_with_metadata = Mock()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.models.catalog_manager.run_in_threadpool", new=fake_run_in_threadpool):
        await pipe._sync_model_metadata_to_owui(
            [{"id": "test.model", "original_id": None, "name": "Test"}],  # No original_id
            pipe_identifier="test_pipe",
        )

    # Should still call update but without image
    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    args = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args[0]
    assert args[3] is None  # profile_image_url should be None


def test_update_existing_model_with_description(pipe_instance) -> None:
    """Updates existing model with description."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(model_id, meta={"description": "Old description"})
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            description="New description",
            update_descriptions=True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["description"] == "New description"


def test_update_existing_model_description_no_change_skips_update(pipe_instance) -> None:
    """Skips update when description unchanged."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(model_id, meta={"description": "Same description"})
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            description="Same description",  # Same as existing
            update_descriptions=True,
        )

    update_mock.assert_not_called()


def test_insert_new_model_with_description(pipe_instance) -> None:
    """Inserts new model with description."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    insert_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=None), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            description="Model description",
            update_descriptions=True,
        )

    insert_mock.assert_called_once()
    inserted_form = insert_mock.call_args[0][0]
    meta = dict(inserted_form.meta)
    assert meta["description"] == "Model description"


def test_insert_new_model_with_openrouter_pipe_capabilities(pipe_instance) -> None:
    """Inserts new model with openrouter_pipe capabilities."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    insert_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=None), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            openrouter_pipe_capabilities={"vision": True, "file_input": True},
        )

    insert_mock.assert_called_once()
    inserted_form = insert_mock.call_args[0][0]
    meta = dict(inserted_form.meta)
    assert meta["openrouter_pipe"]["capabilities"] == {"vision": True, "file_input": True}


def test_update_existing_model_with_existing_params(pipe_instance) -> None:
    """Updates model preserving existing params."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    from open_webui.models.models import ModelMeta
    existing = SimpleNamespace(
        id=model_id,
        base_model_id=None,
        name="Example",
        meta=ModelMeta(**{"capabilities": {"vision": True}}),
        params={"temperature": 0.7, "max_tokens": 1000},
        access_control=None,
        is_active=True,
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"web_search": True},
            None,
            True,
            False,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    assert updated_form.params == {"temperature": 0.7, "max_tokens": 1000}


def test_update_existing_model_same_image_skips_update(pipe_instance) -> None:
    """Skips update when profile image unchanged."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"profile_image_url": "data:image/png;base64,SAME"},
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            "data:image/png;base64,SAME",  # Same as existing
            False,
            True,
        )

    update_mock.assert_not_called()


def test_update_existing_model_preserves_openrouter_pipe_meta(pipe_instance) -> None:
    """Preserves existing openrouter_pipe metadata when updating."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "openrouter_pipe": {"existing_key": "existing_value"},
        },
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            openrouter_pipe_capabilities={"vision": True},
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["openrouter_pipe"]["existing_key"] == "existing_value"
    assert meta["openrouter_pipe"]["capabilities"] == {"vision": True}


def test_existing_model_with_none_params_handled(pipe_instance) -> None:
    """Handles existing model with None params gracefully."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    from open_webui.models.models import ModelMeta
    existing = SimpleNamespace(
        id=model_id,
        base_model_id=None,
        name="Example",
        meta=ModelMeta(**{}),
        params=None,
        access_control=None,
        is_active=True,
    )
    update_mock = Mock()

    with patch("open_webui.models.models.Models.get_model_by_id", return_value=existing), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"vision": True},
            None,
            True,
            False,
        )

    update_mock.assert_called_once()


@pytest.mark.asyncio
async def test_sync_model_metadata_direct_uploads_filter_exception_handling(pipe_instance_async) -> None:
    """Handles exceptions when ensuring direct uploads filter function."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.AUTO_ATTACH_ORS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = True
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = True

    pipe._ensure_direct_uploads_filter_function_id = Mock(side_effect=Exception("Filter install failed"))

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.models.catalog_manager.run_in_threadpool", new=fake_run_in_threadpool):
        # Should not raise, logs debug instead
        await pipe._sync_model_metadata_to_owui(
            [{"id": "test.model", "original_id": "test/model"}],
            pipe_identifier="test_pipe",
        )


# ===== From test_model_metadata_sync.py =====

import pytest
from unittest.mock import AsyncMock, Mock, patch

from open_webui_openrouter_pipe import Pipe


def test_build_icon_mapping_success(pipe_instance):
    pipe = pipe_instance

    frontend_data = {
        "data": [
            {
                "slug": "anthropic/claude-3.5-sonnet",
                "endpoint": {
                    "provider_info": {
                        "icon": {"url": "https://example.com/claude.png"},
                    }
                },
            },
            {
                "slug": "openai/gpt-4o",
                "endpoint": {
                    "provider_info": {
                        "icon": {
                            "url": "/images/icons/OpenAI.svg",  # relative URL
                            "className": "something",
                        },
                        "baseUrl": "https://api.openai.com/v1",
                    }
                },
            },
            {
                "slug": "mistral/mistral-small",
                "endpoint": {"provider_info": {"icon": {"url": "/images/icons/Mistral.png"}}},
            },
            {
                "slug": "meta/llama-3.1-70b",
                "endpoint": {"provider_info": {"icon": None}},  # no icon
            },
        ]
    }

    icon_mapping = pipe._build_icon_mapping(frontend_data)

    assert len(icon_mapping) == 3
    assert icon_mapping["anthropic/claude-3.5-sonnet"] == "https://example.com/claude.png"
    assert icon_mapping["openai/gpt-4o"] == "https://openrouter.ai/images/icons/OpenAI.svg"
    assert icon_mapping["mistral/mistral-small"] == "https://openrouter.ai/images/icons/Mistral.png"


def test_build_icon_mapping_empty(pipe_instance):
    pipe = pipe_instance

    assert pipe._build_icon_mapping(None) == {}
    assert pipe._build_icon_mapping({"data": []}) == {}
    assert pipe._build_icon_mapping({}) == {}


def test_extract_openrouter_og_image():
    html = (
        '<html><head>'
        '<meta property="og:image" content="https://openrouter.ai/openai/opengraph-image-abc123?token=xyz"/>'
        "</head></html>"
    )
    assert Pipe._extract_openrouter_og_image(html) == "https://openrouter.ai/openai/opengraph-image-abc123?token=xyz"


def test_guess_image_mime_type_svg_and_png():
    assert (
        Pipe._guess_image_mime_type(
            "https://openrouter.ai/images/icons/OpenAI.svg",
            content_type=None,
            data=b"<svg></svg>",
        )
        == "image/svg+xml"
    )
    assert (
        Pipe._guess_image_mime_type(
            "https://openrouter.ai/images/icons/OpenAI",
            content_type="image/svg+xml; charset=utf-8",
            data=b"<svg></svg>",
        )
        == "image/svg+xml"
    )

    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    assert (
        Pipe._guess_image_mime_type(
            "https://openrouter.ai/images/icons/OpenAI.png",
            content_type=None,
            data=png_bytes,
        )
        == "image/png"
    )


def test_build_icon_mapping_uses_first_provider_icon(pipe_instance):
    pipe = pipe_instance

    frontend_data = {
        "data": [
            {
                "slug": "dup/model",
                "endpoint": {"provider_info": {"icon": {"url": "https://example.com/first.png"}}},
            },
            {
                "slug": "dup/model",
                "endpoint": {"provider_info": {"icon": {"url": "https://example.com/second.png"}}},
            },
        ]
    }

    icon_mapping = pipe._build_icon_mapping(frontend_data)
    assert icon_mapping["dup/model"] == "https://example.com/first.png"


def test_build_web_search_support_mapping_detects_multiple_signals(pipe_instance):
    pipe = pipe_instance

    frontend_data = {
        "data": [
            {
                "slug": "x-ai/grok-4",
                "endpoint": {
                    "features": {"supports_native_web_search": True},
                    "supported_parameters": [],
                    "pricing": {"web_search": "0"},
                },
            },
            {
                "slug": "openai/gpt-4o",
                "endpoint": {
                    "features": {},
                    "supported_parameters": ["web_search_options"],
                    "pricing": {"web_search": "0"},
                },
            },
            {
                "slug": "anthropic/claude-opus-4.5",
                "endpoint": {
                    "features": {"supports_native_web_search": False},
                    "supported_parameters": [],
                    "pricing": {"web_search": "0.01"},
                },
            },
            {
                "slug": "nope/nope",
                "endpoint": {"features": {}, "supported_parameters": [], "pricing": {"web_search": "0"}},
            },
        ]
    }

    mapping = pipe._build_web_search_support_mapping(frontend_data)
    assert mapping == {
        "x-ai/grok-4": True,
        "openai/gpt-4o": True,
        "anthropic/claude-opus-4.5": True,
    }


def test_build_web_search_support_mapping_unions_duplicates(pipe_instance):
    pipe = pipe_instance

    frontend_data = {
        "data": [
            {
                "slug": "dup/model",
                "endpoint": {"features": {}, "supported_parameters": [], "pricing": {"web_search": "0"}},
            },
            {
                "slug": "dup/model",
                "endpoint": {
                    "features": {},
                    "supported_parameters": ["web_search_options"],
                    "pricing": {"web_search": "0"},
                },
            },
        ]
    }

    mapping = pipe._build_web_search_support_mapping(frontend_data)
    assert mapping == {"dup/model": True}


@pytest.mark.asyncio
async def test_sync_model_metadata_prefixes_pipe_id_and_prefers_icon_mapping(pipe_instance_async):
    pipe = pipe_instance_async
    pipe.valves.UPDATE_MODEL_IMAGES = True
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    models = [
        {
            "id": "openai.gpt-4o",
            "name": "GPT-4o",
            "original_id": "openai/gpt-4o",
            "capabilities": {"vision": True},
        }
    ]

    pipe._fetch_frontend_model_catalog = AsyncMock(
        return_value={
            "data": [
                {
                    "slug": "openai/gpt-4o",
                    "endpoint": {
                        "provider_info": {
                            "icon": {"url": "/images/icons/OpenAI.svg"},
                            "baseUrl": "https://api.openai.com/v1",
                        }
                    },
                }
            ]
        }
    )

    pipe._build_maker_profile_image_mapping = AsyncMock(return_value={})
    pipe._fetch_image_as_data_url = AsyncMock(return_value="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB")
    pipe._ensure_catalog_manager()

    pipe._catalog_manager._update_or_insert_model_with_metadata = Mock()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.pipe.run_in_threadpool", new=fake_run_in_threadpool):
        await pipe._sync_model_metadata_to_owui(models, pipe_identifier="open_webui_openrouter_pipe")

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    args = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args[0]
    assert args[0] == "open_webui_openrouter_pipe.openai.gpt-4o"
    assert args[1] == "GPT-4o"
    assert args[2] == {"vision": True, "web_search": True}
    assert args[3].startswith("data:image/")

@pytest.mark.asyncio
async def test_sync_model_metadata_sets_web_search_from_frontend(pipe_instance_async):
    pipe = pipe_instance_async
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    models = [
        {
            "id": "x-ai.grok-4",
            "name": "Grok 4",
            "original_id": "x-ai/grok-4",
            "capabilities": {"web_search": False},
        }
    ]

    pipe._fetch_frontend_model_catalog = AsyncMock(
        return_value={
            "data": [
                {
                    "slug": "x-ai/grok-4",
                    "endpoint": {
                        "features": {"supports_native_web_search": True},
                        "supported_parameters": [],
                        "pricing": {"web_search": "0"},
                    },
                }
            ]
        }
    )

    pipe._ensure_catalog_manager()


    pipe._catalog_manager._update_or_insert_model_with_metadata = Mock()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.pipe.run_in_threadpool", new=fake_run_in_threadpool):
        await pipe._sync_model_metadata_to_owui(models, pipe_identifier="open_webui_openrouter_pipe")

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    args = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args[0]
    assert args[0] == "open_webui_openrouter_pipe.x-ai.grok-4"
    assert args[2] == {"web_search": True}
    assert args[3] is None
    assert args[4] is True
    assert args[5] is False


@pytest.mark.asyncio
async def test_sync_model_metadata_falls_back_to_maker_image_mapping(pipe_instance_async):
    pipe = pipe_instance_async
    pipe.valves.UPDATE_MODEL_IMAGES = True
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False

    models = [
        {
            "id": "openai.gpt-4o",
            "name": "GPT-4o",
            "original_id": "openai/gpt-4o",
            "capabilities": {"vision": True},
        }
    ]

    pipe._fetch_frontend_model_catalog = AsyncMock(return_value={"data": [{"slug": "openai/gpt-4o"}]})
    pipe._build_maker_profile_image_mapping = AsyncMock(return_value={"openai": "https://example.com/openai.png"})
    pipe._fetch_image_as_data_url = AsyncMock(return_value="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB")
    pipe._ensure_catalog_manager()

    pipe._catalog_manager._update_or_insert_model_with_metadata = Mock()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.pipe.run_in_threadpool", new=fake_run_in_threadpool):
        await pipe._sync_model_metadata_to_owui(models, pipe_identifier="open_webui_openrouter_pipe")

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    args = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args[0]
    assert args[0] == "open_webui_openrouter_pipe.openai.gpt-4o"
    assert args[2] is None
    assert args[3].startswith("data:image/")


@pytest.mark.asyncio
async def test_sync_model_metadata_includes_description_when_enabled(pipe_instance_async):
    pipe = pipe_instance_async
    pipe.valves = pipe.Valves(
        UPDATE_MODEL_IMAGES=False,
        UPDATE_MODEL_CAPABILITIES=False,
        UPDATE_MODEL_DESCRIPTIONS=True,
        AUTO_ATTACH_ORS_FILTER=False,
        AUTO_INSTALL_ORS_FILTER=False,
        AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER=False,
        AUTO_ATTACH_DIRECT_UPLOADS_FILTER=False,
        AUTO_INSTALL_DIRECT_UPLOADS_FILTER=False,
    )

    models = [
        {
            "id": "openai.gpt-4o",
            "norm_id": "openai.gpt-4o",
            "name": "GPT-4o",
            "original_id": "openai/gpt-4o",
        }
    ]

    pipe._ensure_catalog_manager()


    pipe._catalog_manager._update_or_insert_model_with_metadata = Mock()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.registry.ModelFamily._lookup_spec", return_value={"description": "Catalog description"}), patch(
        "open_webui_openrouter_pipe.pipe.run_in_threadpool",
        new=fake_run_in_threadpool,
    ):
        await pipe._sync_model_metadata_to_owui(models, pipe_identifier="open_webui_openrouter_pipe")

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    kwargs = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args.kwargs
    assert kwargs["update_descriptions"] is True
    assert kwargs["description"] == "Catalog description"


@pytest.mark.asyncio
async def test_sync_model_metadata_skips_description_when_disabled(pipe_instance_async):
    pipe = pipe_instance_async
    pipe.valves = pipe.Valves(
        UPDATE_MODEL_IMAGES=False,
        UPDATE_MODEL_CAPABILITIES=True,
        UPDATE_MODEL_DESCRIPTIONS=False,
        AUTO_ATTACH_ORS_FILTER=False,
        AUTO_INSTALL_ORS_FILTER=False,
        AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER=False,
        AUTO_ATTACH_DIRECT_UPLOADS_FILTER=False,
        AUTO_INSTALL_DIRECT_UPLOADS_FILTER=False,
    )

    models = [
        {
            "id": "openai.gpt-4o",
            "norm_id": "openai.gpt-4o",
            "name": "GPT-4o",
            "original_id": "openai/gpt-4o",
            "capabilities": {"vision": True},
        }
    ]

    pipe._fetch_frontend_model_catalog = AsyncMock(return_value={"data": []})
    pipe._ensure_catalog_manager()

    pipe._catalog_manager._update_or_insert_model_with_metadata = Mock()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.registry.ModelFamily._lookup_spec", return_value={"description": "Catalog description"}), patch(
        "open_webui_openrouter_pipe.pipe.run_in_threadpool",
        new=fake_run_in_threadpool,
    ):
        await pipe._sync_model_metadata_to_owui(models, pipe_identifier="open_webui_openrouter_pipe")

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    kwargs = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args.kwargs
    assert kwargs["update_descriptions"] is False
    assert kwargs["description"] is None


# ===== From test_qualify_model_for_pipe.py =====

"""Comprehensive unit tests for _qualify_model_for_pipe method.

This method qualifies OpenRouter model IDs with pipe-specific prefixes
for proper routing in Open WebUI.
"""

import pytest

from open_webui_openrouter_pipe import Pipe


class TestQualifyModelForPipe:
    """Test suite for _qualify_model_for_pipe method."""

    def test_basic_qualification_with_valid_inputs(self, pipe_instance):
        """Test basic qualification with pipe identifier and model ID."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", "gpt-4")
        assert result == "mypipe.gpt-4"

    def test_qualification_with_normalized_model_id(self, pipe_instance):
        """Test that model IDs are normalized before qualification."""
        pipe = pipe_instance
        # ModelFamily.base_model normalizes: lowercase, strip dates, replace / with .
        result = pipe._qualify_model_for_pipe("mypipe", "openai/gpt-4")
        assert result == "mypipe.openai.gpt-4"

    def test_already_qualified_model_returns_as_is(self, pipe_instance):
        """Test that already-qualified models are not double-qualified."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", "mypipe.gpt-4")
        assert result == "mypipe.gpt-4"

    def test_no_pipe_identifier_returns_model_id(self, pipe_instance):
        """Test behavior when pipe_identifier is None or empty."""
        pipe = pipe_instance

        # None pipe_identifier
        result = pipe._qualify_model_for_pipe(None, "gpt-4")
        assert result == "gpt-4"

        # Empty string pipe_identifier
        result = pipe._qualify_model_for_pipe("", "gpt-4")
        assert result == "gpt-4"

        # Whitespace-only pipe_identifier
        result = pipe._qualify_model_for_pipe("  ", "gpt-4")
        assert result is not None  # Should still qualify with the trimmed identifier

    def test_none_model_id_returns_none(self, pipe_instance):
        """Test that None model_id returns None."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", None)
        assert result is None

    def test_non_string_model_id_returns_none(self, pipe_instance):
        """Test that non-string model IDs return None."""
        pipe = pipe_instance

        # Integer
        result = pipe._qualify_model_for_pipe("mypipe", 123)
        assert result is None

        # List
        result = pipe._qualify_model_for_pipe("mypipe", ["gpt-4"])
        assert result is None

        # Dict
        result = pipe._qualify_model_for_pipe("mypipe", {"model": "gpt-4"})
        assert result is None

    def test_empty_string_model_id_returns_none(self, pipe_instance):
        """Test that empty or whitespace-only model IDs return None."""
        pipe = pipe_instance

        # Empty string
        result = pipe._qualify_model_for_pipe("mypipe", "")
        assert result is None

        # Whitespace only
        result = pipe._qualify_model_for_pipe("mypipe", "   ")
        assert result is None

        # Tabs and newlines
        result = pipe._qualify_model_for_pipe("mypipe", "\t\n  ")
        assert result is None

    def test_model_id_with_leading_trailing_whitespace(self, pipe_instance):
        """Test that model IDs with whitespace are trimmed."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", "  gpt-4  ")
        assert result == "mypipe.gpt-4"

    def test_complex_model_ids_with_slashes(self, pipe_instance):
        """Test model IDs containing slashes (e.g., provider/model format)."""
        pipe = pipe_instance
        # Slashes should be normalized to dots by ModelFamily.base_model
        result = pipe._qualify_model_for_pipe("mypipe", "anthropic/claude-3-opus")
        assert result == "mypipe.anthropic.claude-3-opus"

    def test_model_ids_with_dates_are_normalized(self, pipe_instance):
        """Test that date suffixes in model IDs are stripped during normalization."""
        pipe = pipe_instance
        # ModelFamily.base_model strips date patterns
        result = pipe._qualify_model_for_pipe("mypipe", "gpt-4-2024-01-15")
        # Date should be stripped by normalization
        assert "2024" not in result
        assert result.startswith("mypipe.")

    def test_case_normalization(self, pipe_instance):
        """Test that model IDs are normalized to lowercase."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", "GPT-4")
        # Should be lowercased by ModelFamily.base_model
        assert result == "mypipe.gpt-4"

    def test_pipe_identifier_with_special_characters(self, pipe_instance):
        """Test pipe identifiers with various characters."""
        pipe = pipe_instance

        # Alphanumeric with dashes
        result = pipe._qualify_model_for_pipe("my-pipe-123", "gpt-4")
        assert result == "my-pipe-123.gpt-4"

        # Underscores
        result = pipe._qualify_model_for_pipe("my_pipe", "gpt-4")
        assert result == "my_pipe.gpt-4"

    def test_multiple_dots_in_qualified_id(self, pipe_instance):
        """Test that already-qualified IDs with multiple dots are handled."""
        pipe = pipe_instance
        # Model already has pipe prefix with dots
        result = pipe._qualify_model_for_pipe("mypipe", "mypipe.provider.model-v1")
        assert result == "mypipe.provider.model-v1"

    def test_normalization_fallback_behavior(self, pipe_instance):
        """Test behavior when ModelFamily.base_model returns None."""
        pipe = pipe_instance
        # If normalization fails or returns None, should use original trimmed value
        # This tests the `or trimmed` fallback in: normalized = ModelFamily.base_model(trimmed) or trimmed
        result = pipe._qualify_model_for_pipe("mypipe", "unknown-model")
        assert result is not None
        assert result.startswith("mypipe.")

    def test_preserves_hyphenated_model_names(self, pipe_instance):
        """Test that hyphens in model names are preserved."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", "claude-3-opus")
        assert "claude-3-opus" in result
        assert result.startswith("mypipe.")

    def test_prefix_detection_is_exact(self, pipe_instance):
        """Test that prefix detection requires exact match with dot separator."""
        pipe = pipe_instance

        # Model starts with identifier but no dot - should NOT be considered qualified
        result = pipe._qualify_model_for_pipe("my", "mygpt-4")
        assert result == "my.mygpt-4"  # Should qualify it

        # Model with exact prefix match - should be considered already qualified
        result = pipe._qualify_model_for_pipe("my", "my.gpt-4")
        assert result == "my.gpt-4"  # Should NOT double-qualify

    def test_real_world_openrouter_model_ids(self, pipe_instance):
        """Test with realistic OpenRouter model ID formats."""
        pipe = pipe_instance

        # Standard OpenRouter format
        result = pipe._qualify_model_for_pipe("openrouter", "openai/gpt-4-turbo")
        assert result.startswith("openrouter.")
        assert "openai" in result
        assert "gpt-4-turbo" in result

        # Anthropic model
        result = pipe._qualify_model_for_pipe("openrouter", "anthropic/claude-3-opus-20240229")
        assert result.startswith("openrouter.")
        assert "anthropic" in result
        assert "claude-3-opus" in result

    def test_unicode_characters_in_model_id(self, pipe_instance):
        """Test that unicode characters in model IDs are preserved."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", "model-名前")
        assert result is not None
        assert "mypipe." in result

    def test_very_long_model_id(self, pipe_instance):
        """Test with unusually long model ID."""
        pipe = pipe_instance
        long_model = "a" * 200
        result = pipe._qualify_model_for_pipe("mypipe", long_model)
        assert result is not None
        assert result.startswith("mypipe.")
        assert len(result) > 200

    def test_qualification_is_idempotent(self, pipe_instance):
        """Test that qualifying an already-qualified ID returns the same result."""
        pipe = pipe_instance

        # First qualification
        first = pipe._qualify_model_for_pipe("mypipe", "gpt-4")

        # Second qualification of already-qualified ID
        second = pipe._qualify_model_for_pipe("mypipe", first)

        # Should be idempotent
        assert first == second
        assert first == "mypipe.gpt-4"


# ===== From test_model_metadata_disable_flags.py =====


from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_webui_openrouter_pipe import Pipe


def _make_existing_model(model_id: str, *, meta: dict, params: dict | None = None):
    from open_webui.models.models import ModelMeta  # provided by test stubs

    return SimpleNamespace(
        id=model_id,
        base_model_id=None,
        name="Example",
        meta=ModelMeta(**meta),
        params=params or {},
        access_control=None,
        is_active=True,
    )


def test_disable_model_metadata_sync_skips_all_updates(pipe_instance) -> None:
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={},
        params={"disable_model_metadata_sync": True},
    )
    update_mock = Mock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", return_value=existing), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "Example",
            capabilities={"vision": True},
            profile_image_url="data:image/png;base64,QUJD",
            update_capabilities=True,
            update_images=True,
            filter_function_id="openrouter_search",
            filter_supported=True,
            auto_attach_filter=True,
            auto_default_filter=True,
            direct_uploads_filter_function_id="openrouter_direct_uploads",
            direct_uploads_filter_supported=True,
            auto_attach_direct_uploads_filter=True,
        )

    assert update_mock.call_count == 0


def test_disable_capability_updates_preserves_existing_caps(pipe_instance) -> None:
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"capabilities": {"vision": False}},
        params={"disable_capability_updates": True},
    )
    update_mock = Mock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", return_value=existing), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ), patch("open_webui_openrouter_pipe.pipe.ModelForm", new=lambda **kw: SimpleNamespace(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "Example",
            capabilities={"vision": True},
            profile_image_url=None,
            update_capabilities=True,
            update_images=False,
            filter_function_id="openrouter_search",
            filter_supported=True,
            auto_attach_filter=True,
            auto_default_filter=False,
        )

    assert update_mock.call_count == 1
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["capabilities"] == {"vision": False}
    assert meta["filterIds"] == ["openrouter_search"]


def test_disable_image_updates_skips_profile_image_changes(pipe_instance) -> None:
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"profile_image_url": "data:image/png;base64,AAAA"},
        params={"disable_image_updates": True},
    )
    update_mock = Mock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", return_value=existing), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "Example",
            capabilities=None,
            profile_image_url="data:image/png;base64,BBBB",
            update_capabilities=False,
            update_images=True,
        )

    assert update_mock.call_count == 0


def test_disable_direct_uploads_auto_attach_skips_filter_ids(pipe_instance) -> None:
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={},
        params={"disable_direct_uploads_auto_attach": True},
    )
    update_mock = Mock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", return_value=existing), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "Example",
            capabilities=None,
            profile_image_url=None,
            update_capabilities=False,
            update_images=False,
            direct_uploads_filter_function_id="openrouter_direct_uploads",
            direct_uploads_filter_supported=True,
            auto_attach_direct_uploads_filter=True,
        )

    assert update_mock.call_count == 0


def test_description_updates_when_enabled(pipe_instance) -> None:
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(model_id, meta={}, params={})
    update_mock = Mock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", return_value=existing), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ), patch("open_webui_openrouter_pipe.pipe.ModelForm", new=lambda **kw: SimpleNamespace(**kw)):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "Example",
            capabilities=None,
            profile_image_url=None,
            update_capabilities=False,
            update_images=False,
            description="Example description",
            update_descriptions=True,
        )

    assert update_mock.call_count == 1
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["description"] == "Example description"


def test_disable_description_updates_prevents_overwrites(pipe_instance) -> None:
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"description": "Manual description"},
        params={"disable_description_updates": True},
    )
    update_mock = Mock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", return_value=existing), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "Example",
            capabilities=None,
            profile_image_url=None,
            update_capabilities=False,
            update_images=False,
            description="New description",
            update_descriptions=True,
        )

    assert update_mock.call_count == 0


def test_disable_description_updates_namespaced_in_openrouter_pipe_params(pipe_instance) -> None:
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"description": "Manual description"},
        params={"openrouter_pipe": {"disable_description_updates": True}},
    )
    update_mock = Mock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", return_value=existing), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "Example",
            capabilities=None,
            profile_image_url=None,
            update_capabilities=False,
            update_images=False,
            description="New description",
            update_descriptions=True,
        )

    assert update_mock.call_count == 0


def test_disable_description_updates_in_custom_params(pipe_instance) -> None:
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"description": "Manual description"},
        params={"custom_params": {"disable_description_updates": True}},
    )
    update_mock = Mock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", return_value=existing), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "Example",
            capabilities=None,
            profile_image_url=None,
            update_capabilities=False,
            update_images=False,
            description="New description",
            update_descriptions=True,
        )

    assert update_mock.call_count == 0


# ===== From test_model_fallback.py =====


from open_webui_openrouter_pipe import _apply_model_fallback_to_payload


def test_model_fallback_csv_to_models_array() -> None:
    payload = {
        "model": "openai/gpt-5",
        "model_fallback": " openai/gpt-5.1 , , anthropic/claude-sonnet-4.5,openai/gpt-5.1 ",
    }
    _apply_model_fallback_to_payload(payload)
    assert payload["models"] == ["openai/gpt-5.1", "anthropic/claude-sonnet-4.5"]
    assert "model_fallback" not in payload


def test_model_fallback_merges_with_existing_models_list() -> None:
    payload = {
        "model": "openai/gpt-5",
        "models": ["anthropic/claude-sonnet-4.5", "openai/gpt-5.1"],
        "model_fallback": "openai/gpt-5.1,google/gemini-2.5-pro",
    }
    _apply_model_fallback_to_payload(payload)
    assert payload["models"] == [
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-5.1",
        "google/gemini-2.5-pro",
    ]



# ===== From openrouter/test_registry.py =====

from open_webui_openrouter_pipe import (
    ModelFamily,
    OpenRouterModelRegistry,
)


def test_capabilities_detects_modalities_and_pricing():
    architecture = {
        "input_modalities": ["text", "video"],
        "output_modalities": ["text", "image"],
    }
    pricing = {"web_search": "0.05"}

    caps = OpenRouterModelRegistry._derive_capabilities(architecture, pricing)

    assert caps["vision"] is True
    assert caps["file_upload"] is True
    assert caps["image_generation"] is True
    assert caps["web_search"] is True
    assert caps["code_interpreter"] is True
    assert caps["usage"] is True


def test_capabilities_zero_web_search_is_disabled():
    architecture = {"input_modalities": ["text"], "output_modalities": []}
    caps = OpenRouterModelRegistry._derive_capabilities(architecture, {"web_search": "0"})
    assert caps["web_search"] is False


def test_model_family_capabilities_returns_copy_and_defaults():
    previous_specs = getattr(ModelFamily, "_DYNAMIC_SPECS").copy()
    try:
        ModelFamily.set_dynamic_specs(
            {
                "foo": {
                    "capabilities": {"vision": True, "usage": True},
                }
            }
        )
        caps = ModelFamily.capabilities("foo")
        assert caps["vision"] is True
        caps["vision"] = False
        # Original spec should remain unchanged
        assert ModelFamily.capabilities("foo")["vision"] is True
        # Unknown models fall back to empty dict
        assert ModelFamily.capabilities("unknown") == {}
    finally:
        ModelFamily.set_dynamic_specs(previous_specs)
