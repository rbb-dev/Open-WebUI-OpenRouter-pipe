"""Coverage tests for ModelCatalogManager (models/catalog_manager.py).

These tests target coverage of model catalog operations including:
- Icon mapping from various frontend data structures
- Web search support detection via multiple signals
- Favicon URL generation with proper encoding
- Frontend catalog fetching error handling
- Maker profile image mapping
- Metadata sync scheduling and execution
- Filter attachment/default logic for Web Tools and Direct Uploads
- Model insert with various access control modes
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

from __future__ import annotations

import asyncio
import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.filters import FilterManager
from open_webui_openrouter_pipe.storage.multimodal import (
    _extract_openrouter_og_image,
    _guess_image_mime_type,
)


# Helper Functions


def _make_existing_model(model_id: str, *, meta: dict, params: dict | None = None):
    """Create a stub existing model for test assertions."""
    from open_webui.models.models import ModelMeta

    return SimpleNamespace(
        id=model_id,
        base_model_id=None,
        name="Example",
        meta=ModelMeta(**meta),
        params={"reasoning_tags": False, **(params or {})},
        access_grants=[],
        is_active=True,
    )


# Icon Mapping Tests


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
    icon_mapping = pipe._ensure_catalog_manager()._build_icon_mapping(frontend_data)
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
    icon_mapping = pipe._ensure_catalog_manager()._build_icon_mapping(frontend_data)
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
    icon_mapping = pipe._ensure_catalog_manager()._build_icon_mapping(frontend_data)
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
    icon_mapping = pipe._ensure_catalog_manager()._build_icon_mapping(frontend_data)
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
    icon_mapping = pipe._ensure_catalog_manager()._build_icon_mapping(frontend_data)
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
    icon_mapping = pipe._ensure_catalog_manager()._build_icon_mapping(frontend_data)
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
    icon_mapping = pipe._ensure_catalog_manager()._build_icon_mapping(frontend_data)
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
    icon_mapping = pipe._ensure_catalog_manager()._build_icon_mapping(frontend_data)
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
    icon_mapping2 = pipe._ensure_catalog_manager()._build_icon_mapping(frontend_data2)
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
    icon_mapping = pipe._ensure_catalog_manager()._build_icon_mapping(frontend_data)
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
    icon_mapping = pipe._ensure_catalog_manager()._build_icon_mapping(frontend_data)
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
    icon_mapping = pipe._ensure_catalog_manager()._build_icon_mapping(frontend_data)
    assert "test/model" not in icon_mapping


# Frontend Catalog Fetch Tests


@pytest.mark.asyncio
async def test_fetch_frontend_model_catalog_success(pipe_instance_async) -> None:
    """Successfully fetches and returns catalog dict."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()

    with aioresponses() as mocked:
        mocked.get(
            "https://openrouter.ai/api/frontend/v1/catalog/models",
            payload={"data": [{"slug": "test/model"}]},
        )
        session = pipe._create_http_session()
        try:
            result = await pipe._ensure_catalog_manager()._fetch_frontend_model_catalog(session)
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
            "https://openrouter.ai/api/frontend/v1/catalog/models",
            status=500,
        )
        session = pipe._create_http_session()
        try:
            result = await pipe._ensure_catalog_manager()._fetch_frontend_model_catalog(session)
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
            "https://openrouter.ai/api/frontend/v1/catalog/models",
            payload=[{"slug": "test/model"}],
        )
        session = pipe._create_http_session()
        try:
            result = await pipe._ensure_catalog_manager()._fetch_frontend_model_catalog(session)
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
            "https://openrouter.ai/api/frontend/v1/catalog/models",
            exception=Exception("Connection failed"),
        )
        session = pipe._create_http_session()
        try:
            result = await pipe._ensure_catalog_manager()._fetch_frontend_model_catalog(session)
            assert result is None
        finally:
            await session.close()


# Maker Profile Image Mapping Tests


@pytest.mark.asyncio
async def test_build_maker_profile_image_mapping_empty_input(pipe_instance_async) -> None:
    """Returns empty dict for empty or None maker IDs."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()

    session = pipe._create_http_session()
    try:
        result = await pipe._ensure_catalog_manager()._build_maker_profile_image_mapping([])
        assert result == {}

        result = await pipe._ensure_catalog_manager()._build_maker_profile_image_mapping([None, "", "  "])
        assert result == {}
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_build_maker_profile_image_mapping_deduplicates_makers(pipe_instance_async) -> None:
    """Deduplicates maker IDs before fetching."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    call_count = 0

    async def mock_fetch(maker_id):
        nonlocal call_count
        call_count += 1
        return f"https://example.com/{maker_id}.png"

    pipe._multimodal_handler._fetch_maker_profile_image_url = mock_fetch

    session = pipe._create_http_session()
    try:
        result = await pipe._ensure_catalog_manager()._build_maker_profile_image_mapping(
            ["openai", "openai", "anthropic", "openai"]
        )
        assert call_count == 2
        assert len(result) == 2
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_build_maker_profile_image_mapping_handles_none_results(pipe_instance_async) -> None:
    """Filters out makers with None image URLs."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()

    async def mock_fetch(maker_id):
        if maker_id == "anthropic":
            return None
        return f"https://example.com/{maker_id}.png"

    pipe._multimodal_handler._fetch_maker_profile_image_url = mock_fetch

    session = pipe._create_http_session()
    try:
        result = await pipe._ensure_catalog_manager()._build_maker_profile_image_mapping(
            ["openai", "anthropic"]
        )
        assert "openai" in result
        assert "anthropic" not in result
    finally:
        await session.close()


# Metadata Sync Scheduling Tests


def test_maybe_schedule_model_metadata_sync_no_valves_enabled(pipe_instance) -> None:
    """Does not schedule when no relevant valves are enabled."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.UPDATE_MODEL_DESCRIPTIONS = False
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_INSTALL_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_DEFAULT_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = False
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False
    pipe.valves.AUTO_INSTALL_IMAGE_GEN_FILTER = False
    pipe.valves.AUTO_ATTACH_IMAGE_GEN_FILTER = False
    pipe.valves.AUTO_INSTALL_VIDEO_FILTERS = False
    pipe.valves.AUTO_ATTACH_VIDEO_FILTERS = False
    pipe.valves.AUTO_INSTALL_IMAGE_FILTERS = False
    pipe.valves.AUTO_ATTACH_IMAGE_FILTERS = False
    pipe.valves.AUTO_INSTALL_FUSION_FILTER = False
    pipe.valves.AUTO_ATTACH_FUSION_FILTER = False

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


def test_maybe_schedule_model_metadata_sync_same_key_no_reschedule(
    pipe_instance, monkeypatch
) -> None:
    """A second call with nothing changed must not schedule a second sync.

    The key comes from the manager's own first run rather than from a term-by-term copy
    of the tuple: the copy goes stale the moment a term is added, and it goes stale
    SILENTLY -- the assertion still passes, because the key now differs for the wrong
    reason and the test stops testing what it names.
    """
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    scheduled = []

    def _fake_create_task(coro, *args, **kwargs):
        scheduled.append(coro)
        coro.close()
        task = Mock()
        task.done.return_value = True
        return task

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.models.catalog_manager.asyncio.create_task",
        _fake_create_task,
    )

    pipe._catalog_manager.maybe_schedule_model_metadata_sync(
        [{"id": "test"}],
        pipe_identifier="test_pipe",
    )
    assert len(scheduled) == 1, "the first call must schedule the sync"

    pipe._catalog_manager.maybe_schedule_model_metadata_sync(
        [{"id": "test"}],
        pipe_identifier="test_pipe",
    )
    assert len(scheduled) == 1, (
        "nothing changed between the two calls, so the second must not reschedule"
    )


def test_maybe_schedule_model_metadata_sync_reschedules_on_fusion_valve_change(
    pipe_instance, monkeypatch
) -> None:
    """A change to any Fusion valve must invalidate the sync key and reschedule.

    Regression: the four Fusion valves were absent from the sync key, so toggling
    AUTO_ATTACH/AUTO_DEFAULT_FUSION_FILTER silently did nothing until some other
    tracked valve changed.
    """
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True
    pipe.valves.AUTO_ATTACH_FUSION_FILTER = True

    created = {"called": False}

    def _fake_create_task(coro, *args, **kwargs):
        created["called"] = True
        coro.close()
        task = Mock()
        task.done.return_value = True
        return task

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.models.catalog_manager.asyncio.create_task",
        _fake_create_task,
    )

    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry
    last_fetch = getattr(OpenRouterModelRegistry, "_last_fetch", 0.0)
    last_video_fetch = OpenRouterModelRegistry.last_video_fetch()
    last_image_fetch = OpenRouterModelRegistry.last_image_fetch()

    pipe._catalog_manager._model_metadata_sync_key = (
        "test_pipe",
        float(last_fetch or 0.0),
        float(last_video_fetch or 0.0),
        float(last_image_fetch or 0.0),
        pipe.valves.MODEL_ID,
        pipe.valves.UPDATE_MODEL_IMAGES,
        pipe.valves.UPDATE_MODEL_CAPABILITIES,
        pipe.valves.UPDATE_MODEL_DESCRIPTIONS,
        pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER,
        pipe.valves.AUTO_INSTALL_WEB_TOOLS_FILTER,
        pipe.valves.AUTO_DEFAULT_WEB_TOOLS_FILTER,
        pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER,
        pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER,
        pipe.valves.AUTO_INSTALL_IMAGE_GEN_FILTER,
        pipe.valves.AUTO_ATTACH_IMAGE_GEN_FILTER,
        pipe.valves.AUTO_INSTALL_VIDEO_FILTERS,
        pipe.valves.AUTO_ATTACH_VIDEO_FILTERS,
        pipe.valves.AUTO_DEFAULT_VIDEO_FILTERS,
        pipe.valves.ENABLE_VIDEO_GENERATION,
        pipe.valves.ENABLE_OPENROUTER_IMAGE_GENERATION,
        pipe.valves.AUTO_INSTALL_IMAGE_FILTERS,
        pipe.valves.AUTO_ATTACH_IMAGE_FILTERS,
        pipe.valves.AUTO_DEFAULT_IMAGE_FILTERS,
        pipe.valves.ENABLE_OPENROUTER_FUSION,
        pipe.valves.AUTO_INSTALL_FUSION_FILTER,
        not pipe.valves.AUTO_ATTACH_FUSION_FILTER,
        pipe.valves.AUTO_DEFAULT_FUSION_FILTER,
        pipe.valves.ENABLE_WEB_SEARCH,
        pipe.valves.ENABLE_WEB_FETCH,
        pipe.valves.ENABLE_DATETIME,
        pipe.valves.ENABLE_IMAGE_GENERATION,
        pipe.valves.ADMIN_PROVIDER_ROUTING_MODELS,
        pipe.valves.USER_PROVIDER_ROUTING_MODELS,
    )

    pipe._catalog_manager.maybe_schedule_model_metadata_sync(
        [{"id": "test"}],
        pipe_identifier="test_pipe",
    )
    assert created["called"] is True


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


# Sync Model Metadata Tests


@pytest.mark.asyncio
async def test_sync_model_metadata_returns_early_no_valves(pipe_instance_async) -> None:
    """Returns early when no relevant valves enabled."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.UPDATE_MODEL_DESCRIPTIONS = False
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_INSTALL_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_DEFAULT_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = False
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False
    pipe.valves.AUTO_INSTALL_IMAGE_GEN_FILTER = False
    pipe.valves.AUTO_INSTALL_FUSION_FILTER = False
    pipe.valves.AUTO_ATTACH_FUSION_FILTER = False

    # Should return early without error
    await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui([{"id": "test"}], pipe_identifier="test_pipe")


@pytest.mark.asyncio
async def test_sync_model_metadata_returns_early_empty_models(pipe_instance_async) -> None:
    """Returns early when models list empty."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui([], pipe_identifier="test_pipe")


@pytest.mark.asyncio
async def test_sync_model_metadata_returns_early_no_pipe_identifier(pipe_instance_async) -> None:
    """Returns early when pipe_identifier empty."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui([{"id": "test"}], pipe_identifier="")


@pytest.mark.asyncio
async def test_sync_model_metadata_skips_model_without_valid_id(pipe_instance_async) -> None:
    """Skips models with invalid or missing id."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    update_mock = Mock()

    await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(
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
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_INSTALL_WEB_TOOLS_FILTER = False

    pipe._catalog_manager._update_or_insert_model_with_metadata = AsyncMock()

    await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(
        [{"id": "test.model", "name": None, "capabilities": {"vision": True}}],
        pipe_identifier="test_pipe",
    )

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    args = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args[0]
    assert args[1] == "test.model"


@pytest.mark.asyncio
async def test_sync_model_metadata_web_tools_filter_warning_not_installed(pipe_instance_async) -> None:
    """Logs warning when AUTO_ATTACH_WEB_TOOLS_FILTER enabled but filter not installed."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = True
    pipe.valves.AUTO_INSTALL_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = False
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False

    pipe._ensure_filter_manager()
    pipe._filter_manager.ensure_openrouter_web_tools_filter_function_id = AsyncMock(return_value=None)

    with patch.object(pipe._catalog_manager.logger, "warning") as mock_warning:
        await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(
            [{"id": "test.model", "original_id": "test/model"}],
            pipe_identifier="test_pipe",
        )

    warning_messages = [call[0][0] for call in mock_warning.call_args_list]
    assert any("AUTO_ATTACH_WEB_TOOLS_FILTER is enabled" in msg for msg in warning_messages)


@pytest.mark.asyncio
async def test_sync_model_metadata_direct_uploads_filter_warning(pipe_instance_async) -> None:
    """Logs warning when AUTO_ATTACH_DIRECT_UPLOADS_FILTER enabled but filter not installed."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = True
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False

    pipe._ensure_filter_manager()
    pipe._filter_manager.ensure_direct_uploads_filter_function_id = AsyncMock(return_value=None)

    with patch.object(pipe._catalog_manager.logger, "warning") as mock_warning:
        await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(
            [{"id": "test.model", "original_id": "test/model"}],
            pipe_identifier="test_pipe",
        )

    warning_messages = [call[0][0] for call in mock_warning.call_args_list]
    assert any("AUTO_ATTACH_DIRECT_UPLOADS_FILTER is enabled" in msg for msg in warning_messages)


@pytest.mark.asyncio
async def test_sync_model_metadata_logs_supported_model_counts(pipe_instance_async) -> None:
    """Logs info about supported models for filter attachment."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = True
    pipe.valves.AUTO_INSTALL_WEB_TOOLS_FILTER = False

    pipe._ensure_filter_manager()
    pipe._filter_manager.ensure_openrouter_web_tools_filter_function_id = AsyncMock(return_value="openrouter_web_tools")
    pipe._ensure_catalog_manager()._fetch_frontend_model_catalog = AsyncMock(return_value={
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

    with patch.object(pipe._catalog_manager.logger, "info") as mock_info:
        await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(
            [{"id": "test.model", "original_id": "test/model"}],
            pipe_identifier="test_pipe",
        )

    mock_info.assert_called()
    call_args = mock_info.call_args[0]
    assert "Auto-attaching OpenRouter Web Tools filter" in call_args[0]


@pytest.mark.asyncio
async def test_sync_model_metadata_handles_update_exception(pipe_instance_async) -> None:
    """Handles exceptions during model update gracefully."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = False

    pipe._catalog_manager._update_or_insert_model_with_metadata = AsyncMock(
        side_effect=Exception("DB error")
    )

    # Should not raise
    await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(
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
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = True
    pipe.valves.AUTO_INSTALL_WEB_TOOLS_FILTER = True

    pipe._ensure_filter_manager()
    pipe._filter_manager.ensure_openrouter_web_tools_filter_function_id = AsyncMock(side_effect=Exception("Filter install failed"))

    await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(
        [{"id": "test.model", "original_id": "test/model"}],
        pipe_identifier="test_pipe",
    )


# Update/Insert Model Metadata Tests


@pytest.mark.asyncio
async def test_update_or_insert_empty_model_id_returns_early(pipe_instance_async) -> None:
    """Returns early for empty or whitespace-only model ID."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()

    with patch("open_webui.models.models.Models") as mock_models:
        mock_models.get_model_by_id = AsyncMock()
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            "",
            "Name",
            None,
            None,
            False,
            False,
        )
        mock_models.get_model_by_id.assert_not_called()

        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            "   ",
            "Name",
            None,
            None,
            False,
            False,
        )
        mock_models.get_model_by_id.assert_not_called()


@pytest.mark.asyncio
async def test_update_or_insert_uses_model_id_as_name_fallback(pipe_instance_async) -> None:
    """Uses model_id as name when name is empty."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=None)), \
         patch("open_webui.models.models.Models.insert_new_model", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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


@pytest.mark.asyncio
async def test_update_or_insert_new_model_with_capabilities(pipe_instance_async) -> None:
    """Inserts new model with capabilities when it doesn't exist."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    insert_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=None)), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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


@pytest.mark.asyncio
async def test_update_or_insert_new_model_skips_when_no_metadata(pipe_instance_async) -> None:
    """Skips insert when there's no metadata to add."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    insert_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=None)), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
        )

    insert_mock.assert_not_called()


@pytest.mark.asyncio
async def test_update_or_insert_new_model_access_control_admins(pipe_instance_async) -> None:
    """Sets admins-only access grants for new models by default."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.NEW_MODEL_ACCESS_CONTROL = "admins"
    model_id = "test_pipe.openai.gpt-4o"

    insert_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=None)), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"vision": True},
            None,
            True,
            False,
        )

    insert_mock.assert_called_once()
    inserted_form = insert_mock.call_args[0][0]
    assert inserted_form.access_grants == []


@pytest.mark.asyncio
async def test_update_or_insert_new_model_access_control_public(pipe_instance_async) -> None:
    """Sets wildcard access grants (all users) when configured to 'public'."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.NEW_MODEL_ACCESS_CONTROL = "public"
    model_id = "test_pipe.openai.gpt-4o"

    insert_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=None)), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"vision": True},
            None,
            True,
            False,
        )

    insert_mock.assert_called_once()
    inserted_form = insert_mock.call_args[0][0]
    assert inserted_form.access_grants == [
        {"principal_type": "user", "principal_id": "*", "permission": "read"}
    ]


@pytest.mark.asyncio
async def test_update_or_insert_new_model_invalid_access_control_defaults_private(pipe_instance_async) -> None:
    """Invalid access control values should default to private (fail-safe)."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.NEW_MODEL_ACCESS_CONTROL = "invalid_value"  # type: ignore[assignment]
    model_id = "test_pipe.openai.gpt-4o"

    insert_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=None)), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"vision": True},
            None,
            True,
            False,
        )

    insert_mock.assert_called_once()
    inserted_form = insert_mock.call_args[0][0]
    assert inserted_form.access_grants == []


@pytest.mark.asyncio
async def test_update_existing_model_merges_capabilities(pipe_instance_async) -> None:
    """Merges new capabilities with existing ones."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"capabilities": {"vision": False, "file_upload": True}},
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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
    assert meta["capabilities"]["vision"] is True
    assert meta["capabilities"]["web_search"] is True
    assert meta["capabilities"]["file_upload"] is True


@pytest.mark.asyncio
async def test_update_existing_model_no_changes_skips_update(pipe_instance_async) -> None:
    """Skips update when no actual changes detected."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"capabilities": {"vision": True}},
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"vision": True},
            None,
            True,
            False,
        )

    update_mock.assert_not_called()


@pytest.mark.asyncio
async def test_a_row_missing_the_scanner_verdict_is_updated_even_with_current_metadata(
    pipe_instance_async,
) -> None:
    """A row without `reasoning_tags` is out of date, exactly like stale capabilities.

    Every other reason the sync writes a row is a metadata condition that has drifted.
    A missing scanner verdict is the same kind of drift: Open WebUI's tag scanner is
    still running on that model and can still truncate an answer at a literal `<think>`.
    The write happens once -- afterwards the row carries the verdict and the sync reads
    it back and returns early.
    """
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(model_id, meta={"capabilities": {"vision": True}})
    existing.params = {"temperature": 0.7}
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id, "GPT-4o", {"vision": True}, None, True, False,
        )

    update_mock.assert_called_once()
    written = update_mock.call_args[0][1].params
    written = written if isinstance(written, dict) else written.model_dump()
    assert written["reasoning_tags"] is False
    assert written["temperature"] == 0.7



@pytest.mark.asyncio
async def test_update_existing_model_updates_profile_image(pipe_instance_async) -> None:
    """Updates profile image when different from existing."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"profile_image_url": "data:image/png;base64,OLD"},
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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


@pytest.mark.asyncio
async def test_update_existing_model_updates_openrouter_pipe_capabilities(pipe_instance_async) -> None:
    """Updates openrouter_pipe.capabilities when provided."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(model_id, meta={})
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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


@pytest.mark.asyncio
async def test_update_existing_model_filter_id_migration(pipe_instance_async) -> None:
    """Migrates filter IDs when the filter function ID changes."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": ["old_openrouter_web_tools"],
            "openrouter_pipe": {"web_tools_filter_id": "old_openrouter_web_tools"},
        },
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="new_openrouter_web_tools",
            filter_supported=True,
            auto_attach_filter=True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert "old_openrouter_web_tools" not in meta["filterIds"]
    assert "new_openrouter_web_tools" in meta["filterIds"]


@pytest.mark.asyncio
async def test_update_existing_model_filter_removal_when_unsupported(pipe_instance_async) -> None:
    """Removes filter when model no longer supports it."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"filterIds": ["openrouter_web_tools", "other_filter"]},
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="openrouter_web_tools",
            filter_supported=False,
            auto_attach_filter=True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert "openrouter_web_tools" not in meta["filterIds"]
    assert "other_filter" in meta["filterIds"]


@pytest.mark.asyncio
async def test_update_existing_model_direct_uploads_filter_with_previous_id(pipe_instance_async) -> None:
    """Handles direct uploads filter ID migration."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": ["old_direct_uploads"],
            "openrouter_pipe": {"direct_uploads_filter_id": "old_direct_uploads"},
        },
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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


@pytest.mark.asyncio
async def test_update_existing_model_default_filter_migration(pipe_instance_async) -> None:
    """Migrates default filter IDs when filter function ID changes."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": ["old_openrouter_web_tools"],
            "defaultFilterIds": ["old_openrouter_web_tools"],
            "openrouter_pipe": {
                "web_tools_filter_id": "old_openrouter_web_tools",
                "web_tools_default_seeded": True,
            },
        },
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="new_openrouter_web_tools",
            filter_supported=True,
            auto_attach_filter=True,
            auto_default_filter=True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert "new_openrouter_web_tools" in meta["defaultFilterIds"]
    assert meta["openrouter_pipe"]["web_tools_filter_id"] == "new_openrouter_web_tools"


@pytest.mark.asyncio
async def test_update_existing_model_default_filter_not_attached_skips(pipe_instance_async) -> None:
    """Does not set default filter if filter is not in filterIds."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"filterIds": []},
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="openrouter_web_tools",
            filter_supported=False,
            auto_attach_filter=False,
            auto_default_filter=True,
        )

    update_mock.assert_not_called()


@pytest.mark.asyncio
async def test_insert_new_model_with_filters(pipe_instance_async) -> None:
    """Inserts new model with filter attachments."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    insert_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=None)), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="openrouter_web_tools",
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
    assert "openrouter_web_tools" in meta["filterIds"]
    assert "openrouter_direct_uploads" in meta["filterIds"]
    assert "openrouter_web_tools" in meta["defaultFilterIds"]


@pytest.mark.asyncio
async def test_video_filter_auto_default_on_insert(pipe_instance_async) -> None:
    """Per-model video filter is auto-attached AND auto-defaulted on insert."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.google.veo-3-fast"

    insert_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=None)), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "Veo 3 Fast",
            None,
            None,
            False,
            False,
            video_gen_filter_function_id="openrouter_video_gen_google_veo_3_fast",
            video_gen_filter_supported=True,
            auto_attach_video_gen_filter=True,
            auto_default_video_gen_filter=True,
        )

    insert_mock.assert_called_once()
    inserted_form = insert_mock.call_args[0][0]
    meta = dict(inserted_form.meta)
    assert "openrouter_video_gen_google_veo_3_fast" in meta["filterIds"]
    assert "openrouter_video_gen_google_veo_3_fast" in meta["defaultFilterIds"]


@pytest.mark.asyncio
async def test_video_filter_default_reasserted_on_update(pipe_instance_async) -> None:
    """Removing the video filter from defaults is undone on the next sync (no seeding)."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.google.veo-3-fast"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": ["openrouter_video_gen_google_veo_3_fast"],
            "defaultFilterIds": [],
        },
    )

    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "Veo 3 Fast",
            None,
            None,
            False,
            False,
            video_gen_filter_function_id="openrouter_video_gen_google_veo_3_fast",
            video_gen_filter_supported=True,
            auto_attach_video_gen_filter=True,
            auto_default_video_gen_filter=True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert "openrouter_video_gen_google_veo_3_fast" in meta["defaultFilterIds"]


@pytest.mark.asyncio
async def test_video_filter_default_skipped_when_valve_off(pipe_instance_async) -> None:
    """auto_default_video_gen_filter=False keeps video filter out of defaultFilterIds."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.google.veo-3-fast"

    insert_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=None)), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "Veo 3 Fast",
            None,
            None,
            False,
            False,
            video_gen_filter_function_id="openrouter_video_gen_google_veo_3_fast",
            video_gen_filter_supported=True,
            auto_attach_video_gen_filter=True,
            auto_default_video_gen_filter=False,
        )

    insert_mock.assert_called_once()
    inserted_form = insert_mock.call_args[0][0]
    meta = dict(inserted_form.meta)
    assert "openrouter_video_gen_google_veo_3_fast" in meta["filterIds"]
    assert "openrouter_video_gen_google_veo_3_fast" not in meta.get("defaultFilterIds", [])


@pytest.mark.asyncio
async def test_normalize_filter_ids_handles_non_list(pipe_instance_async) -> None:
    """Handles filterIds that are not lists."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"filterIds": "not a list"},
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="openrouter_web_tools",
            filter_supported=True,
            auto_attach_filter=True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["filterIds"] == ["openrouter_web_tools"]


@pytest.mark.asyncio
async def test_normalize_filter_ids_filters_non_strings(pipe_instance_async) -> None:
    """Filters out non-string entries from filterIds."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"filterIds": ["valid_filter", 123, None, "", "another_filter"]},
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="openrouter_web_tools",
            filter_supported=True,
            auto_attach_filter=True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert "valid_filter" in meta["filterIds"]
    assert "another_filter" in meta["filterIds"]
    assert "openrouter_web_tools" in meta["filterIds"]


@pytest.mark.asyncio
async def test_dedupe_preserves_order(pipe_instance_async) -> None:
    """Deduplication preserves original order."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"filterIds": ["filter_a", "filter_b", "filter_a", "filter_c", "filter_b"]},
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            filter_function_id="openrouter_web_tools",
            filter_supported=True,
            auto_attach_filter=True,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["filterIds"] == ["filter_a", "filter_b", "filter_c", "openrouter_web_tools"]


@pytest.mark.asyncio
async def test_existing_model_null_meta_handled(pipe_instance_async) -> None:
    """Handles existing model with null meta gracefully."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    # Create a model with None meta
    existing = SimpleNamespace(
        id=model_id,
        base_model_id=None,
        name="Example",
        meta=None,
        params={},
        access_grants=[],
        is_active=True,
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"vision": True},
            None,
            True,
            False,
        )

    update_mock.assert_called_once()


# Additional Edge Case Tests


@pytest.mark.asyncio
async def test_sync_model_metadata_direct_uploads_logs_supported_count(pipe_instance_async) -> None:
    """Logs info about supported models for direct uploads filter attachment."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_INSTALL_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = True
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False

    pipe._ensure_filter_manager()
    pipe._filter_manager.ensure_direct_uploads_filter_function_id = AsyncMock(return_value="openrouter_direct_uploads")

    with patch.object(pipe._catalog_manager.logger, "info") as mock_info:
        await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(
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
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_INSTALL_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = False
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False

    pipe._ensure_catalog_manager()._fetch_frontend_model_catalog = AsyncMock(return_value={
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
    pipe._ensure_catalog_manager()._build_maker_profile_image_mapping = AsyncMock(return_value={})
    pipe._multimodal_handler._fetch_image_as_data_url = AsyncMock(return_value="data:image/png;base64,ABC123")

    pipe._catalog_manager._update_or_insert_model_with_metadata = AsyncMock()

    await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(
        [{"id": "openai.gpt-4", "original_id": "openai/gpt-4", "name": "GPT-4"}],
        pipe_identifier="test_pipe",
    )

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    args = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args[0]
    assert args[3] == "data:image/png;base64,ABC123"


@pytest.mark.asyncio
async def test_sync_model_metadata_maker_image_fallback(pipe_instance_async) -> None:
    """Tests maker image fallback when model icon not found."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_IMAGES = True
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_INSTALL_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = False
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False

    pipe._ensure_catalog_manager()._fetch_frontend_model_catalog = AsyncMock(return_value={"data": []})
    pipe._ensure_catalog_manager()._build_maker_profile_image_mapping = AsyncMock(return_value={"anthropic": "https://example.com/anthropic.png"})
    pipe._multimodal_handler._fetch_image_as_data_url = AsyncMock(return_value="data:image/png;base64,ANTHROPIC123")

    pipe._catalog_manager._update_or_insert_model_with_metadata = AsyncMock()

    await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(
        [{"id": "anthropic.claude", "original_id": "anthropic/claude", "name": "Claude"}],
        pipe_identifier="test_pipe",
    )

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    args = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args[0]
    assert args[3] == "data:image/png;base64,ANTHROPIC123"


@pytest.mark.asyncio
async def test_sync_model_metadata_skips_model_without_original_id_for_images(pipe_instance_async) -> None:
    """Skips models without original_id when building icon mapping."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_IMAGES = True
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_INSTALL_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = False
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = False

    pipe._ensure_catalog_manager()._fetch_frontend_model_catalog = AsyncMock(return_value={"data": []})
    pipe._ensure_catalog_manager()._build_maker_profile_image_mapping = AsyncMock(return_value={})
    pipe._multimodal_handler._fetch_image_as_data_url = AsyncMock(return_value=None)

    pipe._catalog_manager._update_or_insert_model_with_metadata = AsyncMock()

    await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(
        [{"id": "test.model", "original_id": None, "name": "Test"}],
        pipe_identifier="test_pipe",
    )

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    args = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args[0]
    assert args[3] is None


@pytest.mark.asyncio
async def test_update_existing_model_with_description(pipe_instance_async) -> None:
    """Updates existing model with description."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(model_id, meta={"description": "Old description"})
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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


@pytest.mark.asyncio
async def test_update_existing_model_description_no_change_skips_update(pipe_instance_async) -> None:
    """Skips update when description unchanged."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(model_id, meta={"description": "Same description"})
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            description="Same description",
            update_descriptions=True,
        )

    update_mock.assert_not_called()


@pytest.mark.asyncio
async def test_insert_new_model_with_description(pipe_instance_async) -> None:
    """Inserts new model with description."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    insert_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=None)), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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


@pytest.mark.asyncio
async def test_insert_new_model_with_openrouter_pipe_capabilities(pipe_instance_async) -> None:
    """Inserts new model with openrouter_pipe capabilities."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    insert_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=None)), \
         patch("open_webui.models.models.Models.insert_new_model", new=insert_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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


@pytest.mark.asyncio
async def test_update_existing_model_with_existing_params(pipe_instance_async) -> None:
    """Updates model preserving existing params, and stamps the scanner verdict.

    Open WebUI's `<think>`-tag scanner has no true positives on a pipe model -- the pipe
    emits reasoning as native output items -- so every row the sync touches carries
    `reasoning_tags: False` alongside whatever the operator set.
    """
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    from open_webui.models.models import ModelMeta
    existing = SimpleNamespace(
        id=model_id,
        base_model_id=None,
        name="Example",
        meta=ModelMeta(**{"capabilities": {"vision": True}}),
        params={"temperature": 0.7, "max_tokens": 1000},
        access_grants=[],
        is_active=True,
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"web_search": True},
            None,
            True,
            False,
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    assert updated_form.params == {
        "temperature": 0.7,
        "max_tokens": 1000,
        "reasoning_tags": False,
    }


@pytest.mark.asyncio
async def test_update_existing_model_same_image_skips_update(pipe_instance_async) -> None:
    """Skips update when profile image unchanged."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"profile_image_url": "data:image/png;base64,SAME"},
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            "data:image/png;base64,SAME",
            False,
            True,
        )

    update_mock.assert_not_called()


@pytest.mark.asyncio
async def test_update_existing_model_preserves_openrouter_pipe_meta(pipe_instance_async) -> None:
    """Preserves existing openrouter_pipe metadata when updating."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "openrouter_pipe": {"existing_key": "existing_value"},
        },
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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


@pytest.mark.asyncio
async def test_existing_model_with_none_params_handled(pipe_instance_async) -> None:
    """Handles existing model with None params gracefully."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    from open_webui.models.models import ModelMeta
    existing = SimpleNamespace(
        id=model_id,
        base_model_id=None,
        name="Example",
        meta=ModelMeta(**{}),
        params=None,
        access_grants=[],
        is_active=True,
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = False
    pipe.valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER = True
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = True

    pipe._ensure_filter_manager()
    pipe._filter_manager.ensure_direct_uploads_filter_function_id = AsyncMock(side_effect=Exception("Filter install failed"))

    await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(
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
                            "url": "/images/icons/OpenAI.svg",
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
                "endpoint": {"provider_info": {"icon": None}},
            },
        ]
    }

    icon_mapping = pipe._ensure_catalog_manager()._build_icon_mapping(frontend_data)

    assert len(icon_mapping) == 3
    assert icon_mapping["anthropic/claude-3.5-sonnet"] == "https://example.com/claude.png"
    assert icon_mapping["openai/gpt-4o"] == "https://openrouter.ai/images/icons/OpenAI.svg"
    assert icon_mapping["mistral/mistral-small"] == "https://openrouter.ai/images/icons/Mistral.png"


def test_build_icon_mapping_empty(pipe_instance):
    pipe = pipe_instance

    assert pipe._ensure_catalog_manager()._build_icon_mapping(None) == {}
    assert pipe._ensure_catalog_manager()._build_icon_mapping({"data": []}) == {}
    assert pipe._ensure_catalog_manager()._build_icon_mapping({}) == {}


def test_extract_openrouter_og_image():
    html = (
        '<html><head>'
        '<meta property="og:image" content="https://openrouter.ai/openai/opengraph-image-abc123?token=xyz"/>'
        "</head></html>"
    )
    assert _extract_openrouter_og_image(html) == "https://openrouter.ai/openai/opengraph-image-abc123?token=xyz"


def test_guess_image_mime_type_svg_and_png():
    assert (
        _guess_image_mime_type(
            "https://openrouter.ai/images/icons/OpenAI.svg",
            content_type=None,
            data=b"<svg></svg>",
        )
        == "image/svg+xml"
    )
    assert (
        _guess_image_mime_type(
            "https://openrouter.ai/images/icons/OpenAI",
            content_type="image/svg+xml; charset=utf-8",
            data=b"<svg></svg>",
        )
        == "image/svg+xml"
    )

    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    assert (
        _guess_image_mime_type(
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

    icon_mapping = pipe._ensure_catalog_manager()._build_icon_mapping(frontend_data)
    assert icon_mapping["dup/model"] == "https://example.com/first.png"


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

    pipe._ensure_catalog_manager()._fetch_frontend_model_catalog = AsyncMock(
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

    pipe._ensure_catalog_manager()._build_maker_profile_image_mapping = AsyncMock(return_value={})
    pipe._multimodal_handler._fetch_image_as_data_url = AsyncMock(return_value="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB")
    pipe._ensure_catalog_manager()

    pipe._catalog_manager._update_or_insert_model_with_metadata = AsyncMock()

    await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(models, pipe_identifier="open_webui_openrouter_pipe")

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

    pipe._ensure_catalog_manager()._fetch_frontend_model_catalog = AsyncMock(
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


    pipe._catalog_manager._update_or_insert_model_with_metadata = AsyncMock()

    await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(models, pipe_identifier="open_webui_openrouter_pipe")

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

    pipe._ensure_catalog_manager()._fetch_frontend_model_catalog = AsyncMock(return_value={"data": [{"slug": "openai/gpt-4o"}]})
    pipe._ensure_catalog_manager()._build_maker_profile_image_mapping = AsyncMock(return_value={"openai": "https://example.com/openai.png"})
    pipe._multimodal_handler._fetch_image_as_data_url = AsyncMock(return_value="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB")
    pipe._ensure_catalog_manager()

    pipe._catalog_manager._update_or_insert_model_with_metadata = AsyncMock()

    await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(models, pipe_identifier="open_webui_openrouter_pipe")

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
        AUTO_ATTACH_WEB_TOOLS_FILTER=False,
        AUTO_INSTALL_WEB_TOOLS_FILTER=False,
        AUTO_DEFAULT_WEB_TOOLS_FILTER=False,
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
    pipe._catalog_manager._fetch_frontend_model_catalog = AsyncMock(return_value=None)


    pipe._catalog_manager._update_or_insert_model_with_metadata = AsyncMock()

    with patch("open_webui_openrouter_pipe.registry.ModelFamily._lookup_spec", return_value={"description": "Catalog description"}):
        await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(models, pipe_identifier="open_webui_openrouter_pipe")

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
        AUTO_ATTACH_WEB_TOOLS_FILTER=False,
        AUTO_INSTALL_WEB_TOOLS_FILTER=False,
        AUTO_DEFAULT_WEB_TOOLS_FILTER=False,
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

    pipe._ensure_catalog_manager()._fetch_frontend_model_catalog = AsyncMock(return_value={"data": []})
    pipe._ensure_catalog_manager()

    pipe._catalog_manager._update_or_insert_model_with_metadata = AsyncMock()

    with patch("open_webui_openrouter_pipe.registry.ModelFamily._lookup_spec", return_value={"description": "Catalog description"}):
        await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(models, pipe_identifier="open_webui_openrouter_pipe")

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
        assert result is not None

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
        result = pipe._qualify_model_for_pipe("mypipe", "anthropic/claude-3-opus")
        assert result == "mypipe.anthropic.claude-3-opus"

    def test_model_ids_with_dates_are_normalized(self, pipe_instance):
        """Test that date suffixes in model IDs are stripped during normalization."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", "gpt-4-2024-01-15")
        assert "2024" not in result
        assert result.startswith("mypipe.")

    def test_case_normalization(self, pipe_instance):
        """Test that model IDs are normalized to lowercase."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", "GPT-4")
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
        result = pipe._qualify_model_for_pipe("mypipe", "mypipe.provider.model-v1")
        assert result == "mypipe.provider.model-v1"

    def test_normalization_fallback_behavior(self, pipe_instance):
        """Test behavior when ModelFamily.base_model returns None."""
        pipe = pipe_instance
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

        result = pipe._qualify_model_for_pipe("my", "mygpt-4")
        assert result == "my.mygpt-4"

        result = pipe._qualify_model_for_pipe("my", "my.gpt-4")
        assert result == "my.gpt-4"

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

        second = pipe._qualify_model_for_pipe("mypipe", first)

        # Should be idempotent
        assert first == second
        assert first == "mypipe.gpt-4"


from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_webui_openrouter_pipe import Pipe


def _make_existing_model(model_id: str, *, meta: dict, params: dict | None = None):
    from open_webui.models.models import ModelMeta

    return SimpleNamespace(
        id=model_id,
        base_model_id=None,
        name="Example",
        meta=ModelMeta(**meta),
        params={"reasoning_tags": False, **(params or {})},
        access_grants=[],
        is_active=True,
    )


@pytest.mark.asyncio
async def test_disable_model_metadata_sync_skips_all_updates(pipe_instance_async) -> None:
    pipe = pipe_instance_async
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={},
        params={"disable_model_metadata_sync": True},
    )
    update_mock = AsyncMock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", new=AsyncMock(return_value=existing)), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "Example",
            capabilities={"vision": True},
            profile_image_url="data:image/png;base64,QUJD",
            update_capabilities=True,
            update_images=True,
            filter_function_id="openrouter_web_tools",
            filter_supported=True,
            auto_attach_filter=True,
            auto_default_filter=True,
            direct_uploads_filter_function_id="openrouter_direct_uploads",
            direct_uploads_filter_supported=True,
            auto_attach_direct_uploads_filter=True,
        )

    assert update_mock.call_count == 0


@pytest.mark.asyncio
async def test_disable_capability_updates_preserves_existing_caps(pipe_instance_async) -> None:
    pipe = pipe_instance_async
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"capabilities": {"vision": False}},
        params={"disable_capability_updates": True},
    )
    update_mock = AsyncMock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", new=AsyncMock(return_value=existing)), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ), patch("open_webui_openrouter_pipe.pipe.ModelForm", new=lambda **kw: SimpleNamespace(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "Example",
            capabilities={"vision": True},
            profile_image_url=None,
            update_capabilities=True,
            update_images=False,
            filter_function_id="openrouter_web_tools",
            filter_supported=True,
            auto_attach_filter=True,
            auto_default_filter=False,
        )

    assert update_mock.call_count == 1
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["capabilities"] == {"vision": False}
    assert meta["filterIds"] == ["openrouter_web_tools"]


@pytest.mark.asyncio
async def test_disable_image_updates_skips_profile_image_changes(pipe_instance_async) -> None:
    pipe = pipe_instance_async
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"profile_image_url": "data:image/png;base64,AAAA"},
        params={"disable_image_updates": True},
    )
    update_mock = AsyncMock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", new=AsyncMock(return_value=existing)), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "Example",
            capabilities=None,
            profile_image_url="data:image/png;base64,BBBB",
            update_capabilities=False,
            update_images=True,
        )

    assert update_mock.call_count == 0


@pytest.mark.asyncio
async def test_disable_direct_uploads_auto_attach_skips_filter_ids(pipe_instance_async) -> None:
    pipe = pipe_instance_async
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={},
        params={"disable_direct_uploads_auto_attach": True},
    )
    update_mock = AsyncMock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", new=AsyncMock(return_value=existing)), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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


@pytest.mark.asyncio
async def test_description_updates_when_enabled(pipe_instance_async) -> None:
    pipe = pipe_instance_async
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(model_id, meta={}, params={})
    update_mock = AsyncMock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", new=AsyncMock(return_value=existing)), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ), patch("open_webui_openrouter_pipe.pipe.ModelForm", new=lambda **kw: SimpleNamespace(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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


@pytest.mark.asyncio
async def test_disable_description_updates_prevents_overwrites(pipe_instance_async) -> None:
    pipe = pipe_instance_async
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"description": "Manual description"},
        params={"disable_description_updates": True},
    )
    update_mock = AsyncMock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", new=AsyncMock(return_value=existing)), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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


@pytest.mark.asyncio
async def test_disable_description_updates_namespaced_in_openrouter_pipe_params(pipe_instance_async) -> None:
    pipe = pipe_instance_async
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"description": "Manual description"},
        params={"openrouter_pipe": {"disable_description_updates": True}},
    )
    update_mock = AsyncMock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", new=AsyncMock(return_value=existing)), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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


@pytest.mark.asyncio
async def test_disable_description_updates_in_custom_params(pipe_instance_async) -> None:
    pipe = pipe_instance_async
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"description": "Manual description"},
        params={"custom_params": {"disable_description_updates": True}},
    )
    update_mock = AsyncMock()

    with patch("open_webui_openrouter_pipe.pipe.Models.get_model_by_id", new=AsyncMock(return_value=existing)), patch(
        "open_webui_openrouter_pipe.pipe.Models.update_model_by_id", new=update_mock
    ):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
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
        assert ModelFamily.capabilities("foo")["vision"] is True
        assert ModelFamily.capabilities("unknown") == {}
    finally:
        ModelFamily.set_dynamic_specs(previous_specs)


# Stale Filter ID Pruning Tests


@pytest.mark.asyncio
async def test_prune_stale_openrouter_filter_removes_nonexistent_id(pipe_instance_async) -> None:
    """Stale openrouter_* filter IDs not in the valid set should be pruned."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": [
                "openrouter_web_tools",
                "openrouter_native_attachments",
                "openrouter_direct_uploads",
            ],
        },
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            valid_openrouter_filter_ids=frozenset({
                "openrouter_web_tools",
                "openrouter_direct_uploads",
                "openrouter_provider_openai_gpt_4o",
            }),
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert "openrouter_native_attachments" not in meta["filterIds"]
    assert "openrouter_web_tools" in meta["filterIds"]
    assert "openrouter_direct_uploads" in meta["filterIds"]


@pytest.mark.asyncio
async def test_prune_stale_preserves_non_openrouter_filter_ids(pipe_instance_async) -> None:
    """Filter IDs that don't start with openrouter_ are never pruned, even if unknown."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": [
                "openrouter_web_tools",
                "openrouter_native_attachments",
                "some_other_plugin_filter",
                "openrouter_direct_uploads",
            ],
        },
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            valid_openrouter_filter_ids=frozenset({
                "openrouter_web_tools",
                "openrouter_direct_uploads",
            }),
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert "openrouter_native_attachments" not in meta["filterIds"]
    assert "some_other_plugin_filter" in meta["filterIds"]
    assert "openrouter_web_tools" in meta["filterIds"]
    assert "openrouter_direct_uploads" in meta["filterIds"]


@pytest.mark.asyncio
async def test_prune_stale_empty_valid_set_skips_pruning(pipe_instance_async) -> None:
    """When the valid set is empty (batch query failed), pruning is skipped entirely."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": [
                "openrouter_web_tools",
                "openrouter_native_attachments",
            ],
        },
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            valid_openrouter_filter_ids=frozenset(),
        )

    update_mock.assert_not_called()


@pytest.mark.asyncio
async def test_prune_stale_no_filter_ids_is_noop(pipe_instance_async) -> None:
    """When model has no filterIds, pruning is a no-op."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={"capabilities": {"vision": True}},
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"vision": True},
            None,
            True,
            False,
            valid_openrouter_filter_ids=frozenset({"openrouter_web_tools"}),
        )

    update_mock.assert_not_called()


@pytest.mark.asyncio
async def test_prune_stale_all_valid_no_update(pipe_instance_async) -> None:
    """When all openrouter_* filter IDs are valid, no pruning update is triggered."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": [
                "openrouter_web_tools",
                "openrouter_direct_uploads",
                "openrouter_provider_openai_gpt_4o",
            ],
        },
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            valid_openrouter_filter_ids=frozenset({
                "openrouter_web_tools",
                "openrouter_direct_uploads",
                "openrouter_provider_openai_gpt_4o",
            }),
        )

    update_mock.assert_not_called()


@pytest.mark.asyncio
async def test_prune_stale_preserves_order_of_remaining_ids(pipe_instance_async) -> None:
    """Pruning preserves the original order of the remaining filter IDs."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": [
                "other_plugin",
                "openrouter_stale_one",
                "openrouter_web_tools",
                "openrouter_stale_two",
                "openrouter_direct_uploads",
                "another_plugin",
            ],
        },
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            None,
            None,
            False,
            False,
            valid_openrouter_filter_ids=frozenset({
                "openrouter_web_tools",
                "openrouter_direct_uploads",
            }),
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["filterIds"] == [
        "other_plugin",
        "openrouter_web_tools",
        "openrouter_direct_uploads",
        "another_plugin",
    ]


@pytest.mark.asyncio
async def test_prune_stale_triggers_update_even_when_nothing_else_changed(pipe_instance_async) -> None:
    """Pruning alone should trigger a model update write, even if no other metadata changed."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": ["openrouter_web_tools", "openrouter_ghost"],
            "capabilities": {"vision": True},
        },
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4o",
            {"vision": True},
            None,
            True,
            False,
            valid_openrouter_filter_ids=frozenset({"openrouter_web_tools"}),
        )

    update_mock.assert_called_once()
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["filterIds"] == ["openrouter_web_tools"]
    assert "openrouter_ghost" not in meta["filterIds"]


@pytest.mark.asyncio
async def test_prune_stale_provider_routing_filter_kept_when_valid(pipe_instance_async) -> None:
    """Provider routing filter IDs (openrouter_provider_*) are kept when in the valid set."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()
    model_id = "test_pipe.openai.gpt-4.1"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": [
                "openrouter_web_tools",
                "openrouter_direct_uploads",
                "openrouter_provider_openai_gpt_4_1",
            ],
        },
    )
    update_mock = AsyncMock()

    with patch("open_webui.models.models.Models.get_model_by_id", new=AsyncMock(return_value=existing)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        await pipe._ensure_catalog_manager()._update_or_insert_model_with_metadata(
            model_id,
            "GPT-4.1",
            None,
            None,
            False,
            False,
            valid_openrouter_filter_ids=frozenset({
                "openrouter_web_tools",
                "openrouter_direct_uploads",
                "openrouter_provider_openai_gpt_4_1",
            }),
        )

    update_mock.assert_not_called()


def _make_model_for_bulk_prune(model_id: str, filter_ids: list[str]):
    """Create a stub model for bulk prune tests with filterIds in meta."""
    from open_webui.models.models import ModelMeta, ModelParams

    return SimpleNamespace(
        id=model_id,
        base_model_id=None,
        name="Test Model",
        meta=ModelMeta(filterIds=filter_ids) if filter_ids else None,
        params=ModelParams(),
        access_grants=[],
        is_active=True,
    )


def _make_filter_function(function_id: str):
    """Create a stub filter function record."""
    return SimpleNamespace(id=function_id, type="filter")


def _make_functions_module(valid_filters: list):
    """Create a mock ``open_webui.models.functions`` module for lazy imports.

    The ``prune_stale_openrouter_filter_ids`` method does a lazy
    ``from open_webui.models.functions import Functions`` inside the body.
    Since conftest does not stub that module, we inject one into
    ``sys.modules`` using ``patch.dict``.
    """
    mod = types.ModuleType("open_webui.models.functions")

    class _Functions:
        @staticmethod
        async def get_functions_by_type(filter_type):
            return valid_filters

    mod.Functions = _Functions  # type: ignore[attr-defined]
    return mod


@pytest.mark.asyncio
async def test_bulk_prune_removes_stale_ids_from_multiple_models(pipe_instance_async) -> None:
    """Bulk prune should remove stale openrouter_* IDs across all models."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()

    models = [
        _make_model_for_bulk_prune("model_a", [
            "openrouter_web_tools", "openrouter_native_attachments", "openrouter_direct_uploads",
        ]),
        _make_model_for_bulk_prune("model_b", [
            "openrouter_web_tools", "openrouter_direct_uploads",
        ]),
        _make_model_for_bulk_prune("model_c", [
            "openrouter_native_attachments", "openrouter_provider_qwen_qwen3_coder",
        ]),
    ]

    valid_filters = [
        _make_filter_function("openrouter_web_tools"),
        _make_filter_function("openrouter_direct_uploads"),
    ]

    update_calls = {}

    async def capture_update(model_id, form):
        update_calls[model_id] = form

    functions_mod = _make_functions_module(valid_filters)

    with patch.dict(sys.modules, {"open_webui.models.functions": functions_mod}), \
         patch("open_webui.models.models.Models.get_all_models", new=AsyncMock(return_value=models)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=AsyncMock(side_effect=capture_update)), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        count = await pipe._catalog_manager.prune_stale_openrouter_filter_ids()

    assert count == 2
    assert "model_a" in update_calls
    assert "model_b" not in update_calls
    assert "model_c" in update_calls

    meta_a = dict(update_calls["model_a"].meta)
    assert meta_a["filterIds"] == ["openrouter_web_tools", "openrouter_direct_uploads"]

    meta_c = dict(update_calls["model_c"].meta)
    assert meta_c["filterIds"] == []


@pytest.mark.asyncio
async def test_bulk_prune_preserves_non_openrouter_filter_ids(pipe_instance_async) -> None:
    """Bulk prune should never touch filter IDs that don't start with openrouter_."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()

    models = [
        _make_model_for_bulk_prune("model_x", [
            "other_plugin_filter",
            "openrouter_native_attachments",
            "openrouter_web_tools",
            "yet_another_filter",
        ]),
    ]

    valid_filters = [_make_filter_function("openrouter_web_tools")]
    update_calls = {}

    async def capture_update(model_id, form):
        update_calls[model_id] = form

    functions_mod = _make_functions_module(valid_filters)

    with patch.dict(sys.modules, {"open_webui.models.functions": functions_mod}), \
         patch("open_webui.models.models.Models.get_all_models", new=AsyncMock(return_value=models)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=AsyncMock(side_effect=capture_update)), \
         patch("open_webui.models.models.ModelForm", new=lambda **kw: SimpleNamespace(**kw)), \
         patch("open_webui.models.models.ModelMeta", new=lambda **kw: dict(**kw)), \
         patch("open_webui.models.models.ModelParams", new=lambda **kw: dict(**kw)):
        count = await pipe._catalog_manager.prune_stale_openrouter_filter_ids()

    assert count == 1
    meta = dict(update_calls["model_x"].meta)
    assert meta["filterIds"] == ["other_plugin_filter", "openrouter_web_tools", "yet_another_filter"]


@pytest.mark.asyncio
async def test_bulk_prune_returns_zero_when_nothing_stale(pipe_instance_async) -> None:
    """Returns 0 when all openrouter_* IDs are valid."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()

    models = [
        _make_model_for_bulk_prune("model_clean", [
            "openrouter_web_tools", "openrouter_direct_uploads",
        ]),
    ]

    valid_filters = [
        _make_filter_function("openrouter_web_tools"),
        _make_filter_function("openrouter_direct_uploads"),
    ]

    functions_mod = _make_functions_module(valid_filters)

    update_mock = AsyncMock()

    with patch.dict(sys.modules, {"open_webui.models.functions": functions_mod}), \
         patch("open_webui.models.models.Models.get_all_models", new=AsyncMock(return_value=models)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        count = await pipe._catalog_manager.prune_stale_openrouter_filter_ids()

    assert count == 0
    update_mock.assert_not_called()


@pytest.mark.asyncio
async def test_bulk_prune_returns_zero_when_no_valid_filters(pipe_instance_async) -> None:
    """Returns 0 (skip) when the valid filter set is empty — fail-safe."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()

    models = [
        _make_model_for_bulk_prune("model_stale", [
            "openrouter_native_attachments",
        ]),
    ]

    functions_mod = _make_functions_module([])

    update_mock = AsyncMock()

    with patch.dict(sys.modules, {"open_webui.models.functions": functions_mod}), \
         patch("open_webui.models.models.Models.get_all_models", new=AsyncMock(return_value=models)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        count = await pipe._catalog_manager.prune_stale_openrouter_filter_ids()

    assert count == 0
    update_mock.assert_not_called()


@pytest.mark.asyncio
async def test_bulk_prune_skips_models_without_meta(pipe_instance_async) -> None:
    """Models with no meta should be silently skipped."""
    pipe = pipe_instance_async
    pipe._ensure_catalog_manager()

    model_no_meta = SimpleNamespace(
        id="model_none", base_model_id=None, name="No Meta",
        meta=None, params={}, access_grants=[], is_active=True,
    )
    models = [model_no_meta]

    valid_filters = [_make_filter_function("openrouter_web_tools")]
    functions_mod = _make_functions_module(valid_filters)

    update_mock = AsyncMock()

    with patch.dict(sys.modules, {"open_webui.models.functions": functions_mod}), \
         patch("open_webui.models.models.Models.get_all_models", new=AsyncMock(return_value=models)), \
         patch("open_webui.models.models.Models.update_model_by_id", new=update_mock):
        count = await pipe._catalog_manager.prune_stale_openrouter_filter_ids()

    assert count == 0
    update_mock.assert_not_called()


def test_startup_prune_flag_prevents_repeat_calls(pipe_instance) -> None:
    """The _stale_filter_ids_pruned flag should prevent repeat bulk prune calls."""
    pipe = pipe_instance
    assert pipe._stale_filter_ids_pruned is False
    pipe._stale_filter_ids_pruned = True


def test_register_video_models_uses_atomic_assignment_for_all_state_attrs():
    """`register_video_models` must rebuild `_specs`/`_id_map`/`_models` and assign
    each as a NEW object (not in-place mutation), and `ModelFamily._DYNAMIC_SPECS`
    must be the same object as `cls._specs` post-call so feature lookups are
    consistent with the spec dict."""
    import json
    from pathlib import Path
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry, ModelFamily

    fixture = json.loads((Path(__file__).parent / "fixtures" / "video_models_catalog.json").read_text())
    video_model = fixture["data"][0]

    OpenRouterModelRegistry._specs = {"chat-model-norm": {"features": frozenset(), "capabilities": {}}}
    OpenRouterModelRegistry._id_map = {"chat-model-norm": "openai/some-chat"}
    OpenRouterModelRegistry._models = [{"norm_id": "chat-model-norm", "id": "openai-some-chat", "name": "x"}]

    pre_specs_id = id(OpenRouterModelRegistry._specs)
    pre_id_map_id = id(OpenRouterModelRegistry._id_map)
    pre_models_id = id(OpenRouterModelRegistry._models)

    OpenRouterModelRegistry.register_video_models([video_model])

    post_specs_id = id(OpenRouterModelRegistry._specs)
    post_id_map_id = id(OpenRouterModelRegistry._id_map)
    post_models_id = id(OpenRouterModelRegistry._models)

    assert pre_specs_id != post_specs_id, "_specs must be atomically replaced (new dict)"
    assert pre_id_map_id != post_id_map_id, "_id_map must be atomically replaced (new dict)"
    assert pre_models_id != post_models_id, "_models must be atomically replaced (new list)"
    assert ModelFamily._DYNAMIC_SPECS is OpenRouterModelRegistry._specs, (
        "ModelFamily._DYNAMIC_SPECS must reference the same dict as _specs after register_video_models "
        "(set_dynamic_specs called with the post-merge dict)"
    )
    assert "chat-model-norm" in OpenRouterModelRegistry._specs, "chat model must be preserved across video registration"


def test_register_video_models_picks_up_kling_v3_pro_and_std():
    """The full fixture must register both new Kling v3.0 ids end-to-end:
    norm ids land in `_specs` with the `video_generation` feature, the original
    ids round-trip through `_id_map`, and the supported_parameters frozenset
    includes `cfg_scale` so the renderer's gate trips correctly."""
    import json
    from pathlib import Path
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry, sanitize_model_id, ModelFamily

    fixture = json.loads((Path(__file__).parent / "fixtures" / "video_models_catalog.json").read_text())
    OpenRouterModelRegistry.register_video_models(fixture["data"])

    for original_id in ("kwaivgi/kling-v3.0-pro", "kwaivgi/kling-v3.0-std"):
        norm_id = ModelFamily.base_model(sanitize_model_id(original_id))
        assert norm_id in OpenRouterModelRegistry._specs, f"{original_id} (norm={norm_id}) missing from _specs"
        spec = OpenRouterModelRegistry._specs[norm_id]
        assert "video_generation" in spec["features"], f"{original_id} missing video_generation feature"
        assert OpenRouterModelRegistry._id_map[norm_id] == original_id, (
            f"_id_map[{norm_id}] should round-trip to {original_id}, got {OpenRouterModelRegistry._id_map[norm_id]!r}"
        )
        assert "cfg_scale" in spec["supported_parameters"], (
            f"{original_id} supported_parameters must include cfg_scale (renderer gate keys off this set)"
        )
        assert "negative_prompt" in spec["supported_parameters"], f"{original_id} missing negative_prompt"


@pytest.mark.asyncio
async def test_chat_refresh_preserves_video_models():
    """Chat catalog `_refresh` must NOT wipe video models from `_specs`/`_id_map`/`_models`.
    Today's pipe.py runs ensure_loaded then ensure_video_catalog_loaded; the TTL gate
    on the latter means video re-registration is skipped after the first hour. If
    `_refresh` doesn't preserve video entries across the chat dict-replacement,
    video models silently disappear from the dropdown until the video TTL expires."""
    import json
    import logging
    from pathlib import Path
    import aiohttp
    from open_webui_openrouter_pipe import EncryptedStr, Pipe
    from open_webui_openrouter_pipe.models.registry import (
        ModelFamily,
        OpenRouterModelRegistry,
        sanitize_model_id,
    )

    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "video_models_catalog.json").read_text()
    )
    video_model = fixture["data"][0]
    video_norm_id = ModelFamily.base_model(
        sanitize_model_id(video_model["id"])
    )

    OpenRouterModelRegistry.register_video_models([video_model])
    assert video_norm_id in OpenRouterModelRegistry._specs

    pipe = Pipe()
    try:
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"

        with aioresponses() as mock_http:
            mock_http.get(
                "https://openrouter.ai/api/v1/models",
                payload={"data": [{"id": "openai/gpt-4o", "name": "GPT-4o"}]},
                repeat=True,
            )
            mock_http.get(
                "https://openrouter.ai/api/v1/endpoints/zdr",
                payload={"data": []},
                repeat=True,
            )

            async with aiohttp.ClientSession() as session:
                await OpenRouterModelRegistry.ensure_loaded(
                    session,
                    base_url=pipe.valves.BASE_URL,
                    api_key="test-api-key",
                    cache_seconds=3600,
                    logger=logging.getLogger("test"),
                )

        assert video_norm_id in OpenRouterModelRegistry._specs, (
            "Chat refresh wiped video model from _specs"
        )
        assert video_norm_id in OpenRouterModelRegistry._id_map, (
            "Chat refresh wiped video model from _id_map"
        )
        assert any(
            m.get("norm_id") == video_norm_id for m in OpenRouterModelRegistry._models
        ), "Chat refresh wiped video model from _models"
        assert ModelFamily.supports("video_generation", video_norm_id), (
            "ModelFamily lost video_generation feature for preserved video model "
            "(set_dynamic_specs may have been called with pre-merge specs)"
        )
    finally:
        await pipe.close()


@pytest.mark.asyncio
async def test_chat_refresh_chat_wins_on_norm_id_collision_with_video():
    """If a chat model's norm_id ever collides with a registered video norm_id, the
    chat refresh must keep the chat spec (the catalog's authoritative answer)."""
    import json
    import logging
    from pathlib import Path
    import aiohttp
    from open_webui_openrouter_pipe import EncryptedStr, Pipe
    from open_webui_openrouter_pipe.models.registry import (
        ModelFamily,
        OpenRouterModelRegistry,
        sanitize_model_id,
    )

    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "video_models_catalog.json").read_text()
    )
    video_model = fixture["data"][0]
    video_id = video_model["id"]
    video_norm_id = ModelFamily.base_model(sanitize_model_id(video_id))

    OpenRouterModelRegistry.register_video_models([video_model])

    pipe = Pipe()
    try:
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"

        with aioresponses() as mock_http:
            mock_http.get(
                "https://openrouter.ai/api/v1/models",
                payload={"data": [{"id": video_id, "name": "ChatModelMasqueradingAsVideo"}]},
                repeat=True,
            )
            mock_http.get(
                "https://openrouter.ai/api/v1/endpoints/zdr",
                payload={"data": []},
                repeat=True,
            )

            async with aiohttp.ClientSession() as session:
                await OpenRouterModelRegistry.ensure_loaded(
                    session,
                    base_url=pipe.valves.BASE_URL,
                    api_key="test-api-key",
                    cache_seconds=3600,
                    logger=logging.getLogger("test"),
                )

        spec = OpenRouterModelRegistry._specs.get(video_norm_id) or {}
        features = set(spec.get("features") or set())
        assert "video_generation" not in features, (
            "Chat-wins-on-collision violated: chat refresh preserved the video spec "
            "instead of the new chat-side entry."
        )
    finally:
        await pipe.close()


def test_catalog_manager_uses_direct_attr_for_last_fetch(pipe_instance):
    """catalog_manager.maybe_schedule_model_metadata_sync reads _last_fetch via
    direct attr access (not getattr-with-default), so any future rename fails loudly."""
    import inspect
    from open_webui_openrouter_pipe.models import catalog_manager

    source = inspect.getsource(catalog_manager.ModelCatalogManager.maybe_schedule_model_metadata_sync)
    assert 'getattr(OpenRouterModelRegistry, "_last_fetch"' not in source, (
        "catalog_manager must NOT use getattr-with-default for _last_fetch; "
        "use direct attribute access so renames fail loudly."
    )
    assert "OpenRouterModelRegistry._last_fetch" in source, (
        "catalog_manager must read _last_fetch via direct attribute access."
    )


_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{slug}/endpoints"


def _endpoints_payload(model_id: str, endpoints: list[dict], name: str = "") -> dict:
    return {"data": {"id": model_id, "name": name or model_id, "endpoints": endpoints}}


class TestProviderOverlayFetch:
    """_fetch_model_endpoints behavior."""

    @pytest.mark.asyncio
    async def test_fetch_model_endpoints_success(self, pipe_instance_async) -> None:
        pipe = pipe_instance_async
        manager = pipe._ensure_catalog_manager()
        payload = _endpoints_payload("a/b", [{"provider_name": "A", "tag": "a"}])
        with aioresponses() as mocked:
            mocked.get(_ENDPOINTS_URL.format(slug="a/b"), payload=payload)
            session = pipe._create_http_session()
            try:
                result = await manager._fetch_model_endpoints(session, "a/b")
            finally:
                await session.close()
        assert result == payload

    @pytest.mark.asyncio
    async def test_fetch_model_endpoints_http_error_returns_none(self, pipe_instance_async) -> None:
        pipe = pipe_instance_async
        manager = pipe._ensure_catalog_manager()
        with aioresponses() as mocked:
            mocked.get(_ENDPOINTS_URL.format(slug="a/b"), status=500)
            session = pipe._create_http_session()
            try:
                result = await manager._fetch_model_endpoints(session, "a/b")
            finally:
                await session.close()
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_model_endpoints_non_dict_returns_none(self, pipe_instance_async) -> None:
        pipe = pipe_instance_async
        manager = pipe._ensure_catalog_manager()
        with aioresponses() as mocked:
            mocked.get(_ENDPOINTS_URL.format(slug="a/b"), payload=["not", "a", "dict"])
            session = pipe._create_http_session()
            try:
                result = await manager._fetch_model_endpoints(session, "a/b")
            finally:
                await session.close()
        assert result is None


class TestProviderOverlayBuilder:
    """_build_routed_provider_overlay parsing, caps, and safety."""

    @pytest.mark.asyncio
    async def test_overlay_tag_edge_cases(self, pipe_instance_async) -> None:
        pipe = pipe_instance_async
        manager = pipe._ensure_catalog_manager()
        payload = _endpoints_payload(
            "a/b",
            [
                {"provider_name": "Alpha", "tag": "alpha/fp8", "quantization": "fp8"},
                {"provider_name": "Beta", "tag": "beta", "quantization": "unknown"},
                {"provider_name": "NoTag"},
                {"provider_name": "EmptyTag", "tag": ""},
                {"provider_name": "IntTag", "tag": 123},
                "not-a-dict",
                {"provider_name": "SlashOnly", "tag": "/fp8"},
            ],
        )
        with aioresponses() as mocked:
            mocked.get(_ENDPOINTS_URL.format(slug="a/b"), payload=payload)
            session = pipe._create_http_session()
            try:
                overlay = await manager._build_routed_provider_overlay(session, ["a/b"])
            finally:
                await session.close()
        assert overlay["a/b"]["providers"] == ["alpha", "beta"]
        assert overlay["a/b"]["provider_names"] == {"alpha": "Alpha", "beta": "Beta"}
        assert overlay["a/b"]["quantizations"] == ["fp8", "unknown"]

    @pytest.mark.asyncio
    async def test_overlay_skips_corrupt_shapes(self, pipe_instance_async) -> None:
        pipe = pipe_instance_async
        manager = pipe._ensure_catalog_manager()
        with aioresponses() as mocked:
            mocked.get(
                _ENDPOINTS_URL.format(slug="bad/list"),
                payload={"data": {"id": "bad/list", "endpoints": "not-a-list"}},
            )
            mocked.get(
                _ENDPOINTS_URL.format(slug="bad/data"),
                payload={"data": ["not", "a", "dict"]},
            )
            session = pipe._create_http_session()
            try:
                overlay = await manager._build_routed_provider_overlay(
                    session, ["bad/list", "bad/data"]
                )
            finally:
                await session.close()
        assert overlay == {}

    @pytest.mark.asyncio
    async def test_overlay_caps_routed_model_count(self, pipe_instance_async) -> None:
        pipe = pipe_instance_async
        manager = pipe._ensure_catalog_manager()
        slugs = [f"author/model-{i:03d}" for i in range(60)]
        fetched: list[str] = []

        async def _recorder(session, slug):
            fetched.append(slug)
            return None

        manager._fetch_model_endpoints = _recorder
        session = pipe._create_http_session()
        try:
            await manager._build_routed_provider_overlay(session, slugs)
        finally:
            await session.close()
        assert len(fetched) == 50
        assert sorted(fetched) == sorted(slugs)[:50]

    @pytest.mark.asyncio
    async def test_overlay_clamps_provider_count(self, pipe_instance_async) -> None:
        pipe = pipe_instance_async
        manager = pipe._ensure_catalog_manager()
        endpoints = [
            {"provider_name": f"P{i}", "tag": f"prov-{i:04d}", "quantization": "fp8"}
            for i in range(120)
        ]
        with aioresponses() as mocked:
            mocked.get(
                _ENDPOINTS_URL.format(slug="a/b"),
                payload=_endpoints_payload("a/b", endpoints),
            )
            session = pipe._create_http_session()
            try:
                overlay = await manager._build_routed_provider_overlay(session, ["a/b"])
            finally:
                await session.close()
        assert len(overlay["a/b"]["providers"]) == 100
        assert overlay["a/b"]["providers"] == sorted(overlay["a/b"]["providers"])
        assert set(overlay["a/b"]["provider_names"]) <= set(overlay["a/b"]["providers"])


class TestProviderOverlayMerge:
    """_merge_provider_overlay semantics and _build_provider_map_with_overlay wiring."""

    def _manager(self, pipe):
        return pipe._ensure_catalog_manager()

    def test_merge_replaces_providers_and_quantizations(self, pipe_instance) -> None:
        manager = self._manager(pipe_instance)
        frontend_map = {
            "a/b": {
                "providers": ["streamlake"],
                "quantizations": ["fp8"],
                "short_name": "AB",
                "provider_names": {"streamlake": "StreamLake"},
            }
        }
        overlay = {
            "a/b": {
                "providers": ["baidu", "streamlake"],
                "quantizations": ["fp4", "fp8"],
                "short_name": "A: B",
                "provider_names": {"baidu": "Baidu", "streamlake": "StreamLake"},
            }
        }
        merged = manager._merge_provider_overlay(frontend_map, overlay, ["a/b"])
        assert merged["a/b"]["providers"] == ["baidu", "streamlake"]
        assert merged["a/b"]["quantizations"] == ["fp4", "fp8"]
        assert merged["a/b"]["short_name"] == "AB"

    def test_merge_provider_names_frontend_wins(self, pipe_instance) -> None:
        manager = self._manager(pipe_instance)
        frontend_map = {
            "a/b": {
                "providers": ["streamlake"],
                "quantizations": [],
                "short_name": "AB",
                "provider_names": {"streamlake": "StreamLake Classic"},
            }
        }
        overlay = {
            "a/b": {
                "providers": ["baidu", "streamlake"],
                "quantizations": [],
                "short_name": "",
                "provider_names": {"baidu": "Baidu", "streamlake": "StreamLake"},
            }
        }
        merged = manager._merge_provider_overlay(frontend_map, overlay, ["a/b"])
        assert merged["a/b"]["provider_names"] == {
            "baidu": "Baidu",
            "streamlake": "StreamLake Classic",
        }
        assert list(merged["a/b"]["provider_names"]) == ["baidu", "streamlake"]

    def test_merge_adds_models_missing_from_frontend(self, pipe_instance) -> None:
        manager = self._manager(pipe_instance)
        overlay = {
            "null/model": {
                "providers": ["alpha", "beta"],
                "quantizations": ["fp8"],
                "short_name": "Null Model",
                "provider_names": {"alpha": "Alpha", "beta": "Beta"},
            }
        }
        merged = manager._merge_provider_overlay({}, overlay, ["null/model"])
        assert merged["null/model"]["providers"] == ["alpha", "beta"]
        assert merged["null/model"]["short_name"] == "Null Model"

    def test_merge_fetch_failure_uses_last_known_good(self, pipe_instance) -> None:
        manager = self._manager(pipe_instance)
        manager._cached_provider_map = {
            "a/b": {
                "providers": ["alpha", "beta", "gamma"],
                "quantizations": ["fp8"],
                "short_name": "AB",
                "provider_names": {"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma"},
            }
        }
        frontend_map = {
            "a/b": {
                "providers": ["alpha"],
                "quantizations": [],
                "short_name": "AB",
                "provider_names": {"alpha": "Alpha"},
            }
        }
        merged = manager._merge_provider_overlay(frontend_map, {}, ["a/b"])
        assert merged["a/b"]["providers"] == ["alpha", "beta", "gamma"]

    def test_merge_fetch_failure_keeps_frontend_when_no_cache(self, pipe_instance) -> None:
        manager = self._manager(pipe_instance)
        manager._cached_provider_map = {}
        frontend_map = {
            "a/b": {
                "providers": ["alpha"],
                "quantizations": [],
                "short_name": "AB",
                "provider_names": {"alpha": "Alpha"},
            }
        }
        merged = manager._merge_provider_overlay(frontend_map, {}, ["a/b"])
        assert merged["a/b"]["providers"] == ["alpha"]

    def test_merge_absent_everywhere_stays_absent(self, pipe_instance) -> None:
        manager = self._manager(pipe_instance)
        manager._cached_provider_map = {}
        merged = manager._merge_provider_overlay({}, {}, ["ghost/model"])
        assert "ghost/model" not in merged

    def test_merge_failure_warning_dedupes_across_cycles(self, pipe_instance) -> None:
        manager = self._manager(pipe_instance)
        manager._cached_provider_map = {}
        manager.logger = Mock()
        manager._merge_provider_overlay({}, {}, ["ghost/model"])
        first_warnings = manager.logger.warning.call_count
        manager._merge_provider_overlay({}, {}, ["ghost/model"])
        assert first_warnings >= 1
        assert manager.logger.warning.call_count == first_warnings
        manager._merge_provider_overlay({}, {}, ["ghost/model", "other/model"])
        assert manager.logger.warning.call_count > first_warnings

    @pytest.mark.asyncio
    async def test_overlay_survives_frontend_outage(self, pipe_instance_async) -> None:
        pipe = pipe_instance_async
        manager = pipe._ensure_catalog_manager()
        payload = _endpoints_payload(
            "a/b",
            [
                {"provider_name": "Alpha", "tag": "alpha", "quantization": "fp8"},
                {"provider_name": "Beta", "tag": "beta/fp4", "quantization": "fp4"},
            ],
            name="A: B",
        )
        with aioresponses() as mocked:
            mocked.get(_ENDPOINTS_URL.format(slug="a/b"), payload=payload)
            session = pipe._create_http_session()
            try:
                provider_map = await manager._build_provider_map_with_overlay(
                    session, None, "a/b", ""
                )
            finally:
                await session.close()
        assert provider_map["a/b"]["providers"] == ["alpha", "beta"]
        assert provider_map["a/b"]["short_name"] == "A: B"

    @pytest.mark.asyncio
    async def test_overlay_entries_stable_across_endpoint_order(self, pipe_instance_async) -> None:
        pipe = pipe_instance_async
        manager = pipe._ensure_catalog_manager()
        endpoints = [
            {"provider_name": "Alpha", "tag": "alpha/fp8", "quantization": "fp8"},
            {"provider_name": "Beta", "tag": "beta", "quantization": "unknown"},
            {"provider_name": "Gamma", "tag": "gamma/fp4", "quantization": "fp4"},
        ]

        async def _build(order: list[dict]) -> dict:
            with aioresponses() as mocked:
                mocked.get(
                    _ENDPOINTS_URL.format(slug="a/b"),
                    payload=_endpoints_payload("a/b", order),
                )
                session = pipe._create_http_session()
                try:
                    return await manager._build_routed_provider_overlay(session, ["a/b"])
                finally:
                    await session.close()

        forward = await _build(endpoints)
        backward = await _build(list(reversed(endpoints)))
        assert forward == backward
        assert FilterManager.compute_provider_routing_hash(
            "a/b", "", forward
        ) == FilterManager.compute_provider_routing_hash("a/b", "", backward)


class TestProviderOverlayQuantizationClamp:
    @pytest.mark.asyncio
    async def test_overlay_clamps_quantization_count(self, pipe_instance_async) -> None:
        pipe = pipe_instance_async
        manager = pipe._ensure_catalog_manager()
        endpoints = [
            {"provider_name": "Alpha", "tag": "alpha", "quantization": f"q{i:04d}"}
            for i in range(120)
        ]
        with aioresponses() as mocked:
            mocked.get(
                _ENDPOINTS_URL.format(slug="a/b"),
                payload=_endpoints_payload("a/b", endpoints),
            )
            session = pipe._create_http_session()
            try:
                overlay = await manager._build_routed_provider_overlay(session, ["a/b"])
            finally:
                await session.close()
        assert len(overlay["a/b"]["quantizations"]) == 100
        assert overlay["a/b"]["quantizations"] == sorted(overlay["a/b"]["quantizations"])


class TestProviderRoutingDefaultFilterIds:
    """Provider routing filters must default on in new chats (like web tools)."""

    def _helper(self):
        from open_webui_openrouter_pipe.models.catalog_manager import (
            _apply_provider_routing_default_filter_ids,
        )

        return _apply_provider_routing_default_filter_ids

    def test_default_added_when_enabled(self) -> None:
        meta = {"filterIds": ["openrouter_provider_a_b"]}
        changed = self._helper()(
            meta,
            provider_routing_filter_id="openrouter_provider_a_b",
            auto_default_provider_routing_filter=True,
        )
        assert changed is True
        assert meta["defaultFilterIds"] == ["openrouter_provider_a_b"]

    def test_not_added_when_disabled(self) -> None:
        meta = {"filterIds": ["openrouter_provider_a_b"]}
        changed = self._helper()(
            meta,
            provider_routing_filter_id="openrouter_provider_a_b",
            auto_default_provider_routing_filter=False,
        )
        assert changed is False
        assert "defaultFilterIds" not in meta

    def test_not_added_when_not_attached(self) -> None:
        meta = {"filterIds": ["openrouter_web_tools"]}
        changed = self._helper()(
            meta,
            provider_routing_filter_id="openrouter_provider_a_b",
            auto_default_provider_routing_filter=True,
        )
        assert changed is False

    def test_idempotent_and_preserves_existing_defaults(self) -> None:
        meta = {
            "filterIds": ["openrouter_web_tools", "openrouter_provider_a_b"],
            "defaultFilterIds": ["openrouter_web_tools"],
        }
        helper = self._helper()
        assert helper(
            meta,
            provider_routing_filter_id="openrouter_provider_a_b",
            auto_default_provider_routing_filter=True,
        ) is True
        assert meta["defaultFilterIds"] == ["openrouter_web_tools", "openrouter_provider_a_b"]
        assert helper(
            meta,
            provider_routing_filter_id="openrouter_provider_a_b",
            auto_default_provider_routing_filter=True,
        ) is False
        assert meta["defaultFilterIds"] == ["openrouter_web_tools", "openrouter_provider_a_b"]

    def test_no_filter_id_is_noop(self) -> None:
        meta = {"filterIds": []}
        changed = self._helper()(
            meta,
            provider_routing_filter_id=None,
            auto_default_provider_routing_filter=True,
        )
        assert changed is False


class TestDescriptionsFetchGate:
    """Descriptions-only configs must still fetch the frontend catalog."""

    class _FetchReached(Exception):
        pass

    def _configure(self, pipe, *, descriptions: bool) -> list:
        for valve_name in (
            "UPDATE_MODEL_IMAGES",
            "UPDATE_MODEL_CAPABILITIES",
            "UPDATE_MODEL_DESCRIPTIONS",
            "AUTO_ATTACH_WEB_TOOLS_FILTER",
            "AUTO_INSTALL_WEB_TOOLS_FILTER",
            "AUTO_DEFAULT_WEB_TOOLS_FILTER",
            "AUTO_ATTACH_DIRECT_UPLOADS_FILTER",
            "AUTO_INSTALL_DIRECT_UPLOADS_FILTER",
            "AUTO_INSTALL_IMAGE_GEN_FILTER",
            "AUTO_INSTALL_VIDEO_FILTERS",
            "AUTO_ATTACH_VIDEO_FILTERS",
            "AUTO_INSTALL_IMAGE_FILTERS",
            "AUTO_ATTACH_IMAGE_FILTERS",
            "AUTO_INSTALL_FUSION_FILTER",
            "AUTO_ATTACH_FUSION_FILTER",
        ):
            setattr(pipe.valves, valve_name, False)
        pipe.valves.UPDATE_MODEL_DESCRIPTIONS = descriptions
        pipe.valves.ADMIN_PROVIDER_ROUTING_MODELS = ""
        pipe.valves.USER_PROVIDER_ROUTING_MODELS = ""

        manager = pipe._ensure_catalog_manager()
        reached: list = []

        async def _recorder(session):
            reached.append(True)
            raise TestDescriptionsFetchGate._FetchReached()

        manager._fetch_frontend_model_catalog = _recorder
        return reached

    @pytest.mark.asyncio
    async def test_descriptions_only_config_fetches_frontend_catalog(self, pipe_instance_async) -> None:
        pipe = pipe_instance_async
        reached = self._configure(pipe, descriptions=True)
        try:
            await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(
                [{"id": "a/b", "name": "AB"}], pipe_identifier="openrouter"
            )
        except TestDescriptionsFetchGate._FetchReached:
            pass
        assert reached == [True]

    @pytest.mark.asyncio
    async def test_everything_off_skips_sync_entirely(self, pipe_instance_async) -> None:
        pipe = pipe_instance_async
        reached = self._configure(pipe, descriptions=False)
        await pipe._ensure_catalog_manager()._sync_model_metadata_to_owui(
            [{"id": "a/b", "name": "AB"}], pipe_identifier="openrouter"
        )
        assert reached == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("any_tool_enabled", "should_install"),
    [(True, True), (False, False)],
)
async def test_the_web_tools_filter_is_not_installed_when_every_tool_is_off(
    pipe_instance_async, monkeypatch, any_tool_enabled, should_install
) -> None:
    """AUTO_INSTALL_WEB_TOOLS_FILTER alone is not the gate.

    Installing writes a filter function into Open WebUI and shows it in the operator's
    Integrations menu. Doing that when every underlying tool is disabled puts a control
    in front of users that cannot do anything, and refreshes it on every `pipes()` call.
    Dropping the second conjunct went entirely unnoticed by the suite.

    The catalog endpoint is mocked because it was not: `pipes()` fetches
    /api/v1/models, and with a real API key set and no mock this reached
    openrouter.ai over the network on every run. That made the negative arm pass for the
    wrong reason -- a failed fetch installs nothing, so "every tool disabled" and "the
    request did not complete" produced the same answer -- and made both arms fail on any
    DNS or connectivity hiccup with a message about web tools.
    """
    pipe = pipe_instance_async
    pipe.valves.API_KEY = "sk-test-key"
    pipe.valves.ENABLE_VIDEO_GENERATION = False
    pipe.valves.ENABLE_OPENROUTER_IMAGE_GENERATION = False
    pipe.valves.AUTO_INSTALL_WEB_TOOLS_FILTER = True
    for name in (
        "ENABLE_WEB_SEARCH",
        "ENABLE_WEB_FETCH",
        "ENABLE_DATETIME",
        "ENABLE_ADVISOR",
        "ENABLE_SUBAGENT",
        "ENABLE_SEARCH_MODELS",
    ):
        setattr(pipe.valves, name, False)
    if any_tool_enabled:
        pipe.valves.ENABLE_WEB_FETCH = True

    manager = pipe._ensure_filter_manager()
    installer = AsyncMock(return_value="openrouter_web_tools")
    monkeypatch.setattr(
        manager,
        "ensure_openrouter_web_tools_filter_function_id",
        installer,
        raising=False,
    )

    with aioresponses() as mock_http:
        mock_http.get(
            "https://openrouter.ai/api/v1/models",
            payload={"data": [{"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini"}]},
            repeat=True,
        )
        models = await pipe.pipes()

    assert models and not any("error" in str(m.get("id", "")).lower() for m in models), (
        f"pipes() returned {models!r}; the catalog did not load, so this test would "
        "report 'not installed' whatever the valve gate did"
    )
    assert installer.called is should_install, (
        "the web tools filter was "
        f"{'installed' if installer.called else 'not installed'} with "
        f"{'one tool enabled' if any_tool_enabled else 'every tool disabled'}; "
        "installing with all tools off publishes a filter that can do nothing"
    )


class _HostileModel(dict):
    """A model entry whose own read raises, so `_apply` fails BEFORE the sync call.

    There are two failure recorders: one inside `_apply` for a failed sync, and one at
    the gather for an exception that escaped `_apply` altogether. They feed the same
    aggregate warning, so covering only one leaves the other deletable in silence.
    """

    def get(self, key, default=None):
        # "name" rather than "id": the caller reads the id before dispatching, so raising
        # there escapes the gather instead of being captured by it.
        if key == "name":
            raise RuntimeError("model entry is unreadable")
        return super().get(key, default)


@pytest.mark.asyncio
@pytest.mark.parametrize("escapes_apply", [False, True], ids=["sync-call-fails", "apply-raises"])
async def test_a_partial_metadata_sync_failure_is_reported_with_its_counts(
    pipe_instance_async, caplog, escapes_apply
) -> None:
    """An operator must learn HOW MANY models failed, not just that something did.

    `_sync_model_metadata_to_owui` swallows every per-model failure into a DEBUG line and
    reports one aggregate warning. Nothing tested it: the per-model failure recorder was
    executed by zero of 5885 tests, and replacing the whole warning with `if False:` left
    the suite green -- restoring exactly the silence the diagnostic was added to end.
    Capabilities, descriptions and filter attachments would silently stop updating.

    Two models with ONE failing, so the counts are distinguishable: with a single model
    `1/1` cannot tell the real counters from two hardcoded `1`s.

    Asserted on `record.args`, not on the rendered string -- the numbers are separate `%`
    arguments, so a string assertion couples the test to the formatting and still passes
    if both counts are hardcoded.
    """
    import logging as _logging

    from tests.log_capture import emitted

    pipe = pipe_instance_async
    manager = pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = False

    async def _one_fails(*args, **kwargs):
        model_id = kwargs.get("openrouter_id") or (args[0] if args else "")
        if "bad" in str(model_id):
            raise RuntimeError("model row is read-only")
        return None

    manager._update_or_insert_model_with_metadata = AsyncMock(side_effect=_one_fails)

    bad = (
        _HostileModel({"id": "vendor/bad", "name": "Bad"})
        if escapes_apply
        else {"id": "vendor/bad", "name": "Bad"}
    )
    with caplog.at_level(_logging.DEBUG, logger=manager.logger.name):
        await manager._sync_model_metadata_to_owui(
            [{"id": "vendor/good", "name": "Good"}, bad],
            pipe_identifier="openrouter",
        )

    aggregates = [
        r
        for r in emitted(caplog, min_level=_logging.WARNING)
        if r.args
        and isinstance(r.args, tuple)
        and len(r.args) >= 2
        and isinstance(r.args[0], int)
        and isinstance(r.args[1], int)
    ]
    assert len(aggregates) == 1, (
        f"expected exactly one aggregate sync-failure warning, got {len(aggregates)}: "
        f"{[r.getMessage() for r in emitted(caplog, min_level=_logging.WARNING)]}"
    )
    per_model = [
        r
        for r in emitted(caplog, level=_logging.DEBUG)
        if "metadata" in r.getMessage() and "failed" in r.getMessage()
    ]
    assert len(per_model) == 1, (
        "the aggregate names how many failed; the per-model DEBUG line is the only place "
        f"the operator learns WHY. Got {[r.getMessage() for r in per_model]}"
    )
    assert per_model[0].exc_info is not None, (
        "the per-model line carries no traceback, so raising the log level buys nothing"
    )
    failed, total = aggregates[0].args[0], aggregates[0].args[1]
    assert (failed, total) == (1, 2), (
        f"the warning reported {failed}/{total} model(s) failed; one of two did. An "
        "operator sizes the blast radius from these numbers."
    )


@pytest.mark.parametrize(
    ("features", "expects_default"),
    [
        ({"image_output"}, True),
        ({"video_generation"}, True),
        ({"image_output", "vision"}, True),
        ({"vision"}, False),
        (set(), False),
    ],
)
@pytest.mark.parametrize("valve_on", [True, False])
def test_media_models_arrive_with_builtin_tools_unticked(features, expects_default, valve_on):
    """A model that answers with a picture or a clip is offered no built-in tools.

    The box is unticked rather than the tools withheld at request time, so an operator can
    see the setting instead of wondering why tools went quiet.
    """
    from open_webui_openrouter_pipe.core.config import Valves
    from open_webui_openrouter_pipe.models.catalog_manager import media_capability_defaults

    valves = Valves()
    valves.DISABLE_BUILTIN_TOOLS_ON_MEDIA_MODELS = valve_on
    valves.UPDATE_MODEL_CAPABILITIES = True

    defaults = media_capability_defaults(
        valves,
        {
            "image_output": "image_output" in features,
            "video_generation": "video_generation" in features,
            "vision": "vision" in features,
        },
    )

    assert (defaults.get("builtin_tools") is False) is (expects_default and valve_on), (
        f"features={sorted(features)} valve_on={valve_on} produced {defaults!r}"
    )


@pytest.mark.parametrize(
    "media_valve",
    [
        "ENABLE_VIDEO_GENERATION",
        "AUTO_INSTALL_VIDEO_FILTERS",
        "AUTO_INSTALL_IMAGE_FILTERS",
        "AUTO_INSTALL_IMAGE_GEN_FILTER",
    ],
)
@pytest.mark.asyncio
async def test_the_provider_map_is_built_when_no_routing_models_are_configured(
    pipe_instance_async, monkeypatch, media_valve
) -> None:
    """Each media valve alone must reach the fetch, with every legacy trigger switched off.

    Setting UPDATE_MODEL_CAPABILITIES here would make this pass on a legacy valve and prove
    nothing about the media ones -- the provider map would still be empty in exactly the
    configuration the video adapter needs it.
    """
    pipe = pipe_instance_async
    manager = pipe._ensure_catalog_manager()
    pipe.valves.ADMIN_PROVIDER_ROUTING_MODELS = ""
    pipe.valves.USER_PROVIDER_ROUTING_MODELS = ""
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False
    pipe.valves.UPDATE_MODEL_DESCRIPTIONS = False
    pipe.valves.AUTO_ATTACH_WEB_TOOLS_FILTER = False
    pipe.valves.UPDATE_MODEL_IMAGES = False
    for name in (
        "ENABLE_VIDEO_GENERATION",
        "AUTO_INSTALL_VIDEO_FILTERS",
        "AUTO_INSTALL_IMAGE_FILTERS",
        "AUTO_INSTALL_IMAGE_GEN_FILTER",
    ):
        setattr(pipe.valves, name, name == media_valve)
    manager._update_or_insert_model_with_metadata = AsyncMock()
    monkeypatch.setattr(
        manager,
        "_fetch_frontend_model_catalog",
        AsyncMock(
            return_value={
                "data": [
                    {
                        "slug": "google/veo-3.1",
                        "endpoint": {
                            "model_variant_slug": "google/veo-3.1",
                            "provider_info": {"slug": "google-vertex", "displayName": "Vertex"},
                        },
                    }
                ]
            }
        ),
    )

    await manager._sync_model_metadata_to_owui(
        [{"id": "google.veo-3.1", "name": "Veo"}], pipe_identifier="test_pipe"
    )

    assert manager.get_cached_provider_map().get("google/veo-3.1", {}).get("providers") == [
        "google-vertex"
    ], (
        "provider routing is off by default, yet the video adapter reads this map to key "
        f"provider.options; {media_valve} alone must reach the fetch"
    )


@pytest.mark.asyncio
async def test_an_empty_rebuild_keeps_the_previous_provider_map_and_says_so(
    pipe_instance_async, monkeypatch, caplog
) -> None:
    pipe = pipe_instance_async
    manager = pipe._ensure_catalog_manager()
    pipe.valves.ADMIN_PROVIDER_ROUTING_MODELS = ""
    pipe.valves.USER_PROVIDER_ROUTING_MODELS = ""
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True
    pipe.valves.UPDATE_MODEL_IMAGES = False
    manager._update_or_insert_model_with_metadata = AsyncMock()
    manager._cached_provider_map = {"google/veo-3.1": {"providers": ["google-vertex"]}}
    monkeypatch.setattr(
        manager, "_fetch_frontend_model_catalog", AsyncMock(return_value={"data": []})
    )

    with caplog.at_level(logging.WARNING):
        await manager._sync_model_metadata_to_owui(
            [{"id": "google.veo-3.1", "name": "Veo"}], pipe_identifier="test_pipe"
        )

    assert manager.get_cached_provider_map() == {
        "google/veo-3.1": {"providers": ["google-vertex"]}
    }, "a failed rebuild must not clobber a good map"
    assert any("keeping the previous map" in record.getMessage() for record in caplog.records), (
        "silently keeping a stale map is how an operator loses provider options with no signal"
    )



# ============================================================================
# REGFIX: infrastructure fixes 7 and 8
# ============================================================================


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("VIDEO_INTENT_ENABLED", False),
        ("VIDEO_INTENT_CONFIRM_MODE", "never"),
        ("VIDEO_INTENT_MAX_CLARIFICATIONS", 3),
        ("VIDEO_INTENT_FRAME_EXTRACTION_INDEX", "first"),
    ],
)
def test_a_valve_baked_into_a_rendered_filter_reschedules_the_sync(
    pipe_instance, monkeypatch, attribute, value
) -> None:
    """Every valve build_video_filter_spec bakes into a default must invalidate the key.

    Four distinct valves with four distinct new values, so a key that happens to differ
    for an unrelated reason cannot satisfy all four.
    """
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    scheduled = []

    def _fake_create_task(coro, *args, **kwargs):
        scheduled.append(coro)
        coro.close()
        task = Mock()
        task.done.return_value = True
        return task

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.models.catalog_manager.asyncio.create_task",
        _fake_create_task,
    )

    pipe._catalog_manager.maybe_schedule_model_metadata_sync(
        [{"id": "test"}], pipe_identifier="test_pipe"
    )
    assert len(scheduled) == 1

    assert getattr(pipe.valves, attribute) != value, (
        f"{attribute} already equals {value!r}, so this test would prove nothing"
    )
    setattr(pipe.valves, attribute, value)

    pipe._catalog_manager.maybe_schedule_model_metadata_sync(
        [{"id": "test"}], pipe_identifier="test_pipe"
    )
    assert len(scheduled) == 2, (
        f"changing {attribute} did not reschedule, so the filters keep the defaults they "
        "were rendered with and the valve silently does nothing"
    )


@pytest.mark.parametrize("model", ["openai/gpt-5-image", "google/gemini-3-pro-image"])
def test_the_server_tool_filter_s_selected_model_reschedules_the_sync(
    pipe_instance, monkeypatch, model
) -> None:
    """The selected model lives in the installed function's valves, not in ours."""
    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    scheduled = []

    def _fake_create_task(coro, *args, **kwargs):
        scheduled.append(coro)
        coro.close()
        task = Mock()
        task.done.return_value = True
        return task

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.models.catalog_manager.asyncio.create_task",
        _fake_create_task,
    )

    pipe._catalog_manager.maybe_schedule_model_metadata_sync(
        [{"id": "test"}], pipe_identifier="test_pipe", image_gen_filter_model=""
    )
    assert len(scheduled) == 1

    pipe._catalog_manager.maybe_schedule_model_metadata_sync(
        [{"id": "test"}], pipe_identifier="test_pipe", image_gen_filter_model=model
    )
    assert len(scheduled) == 2, (
        f"selecting {model} in the installed filter did not reschedule, so the filter is "
        "never re-rendered for the model it now targets"
    )


@pytest.mark.parametrize("stored", ["openai/gpt-5-image", "google/gemini-3-pro-image"])
@pytest.mark.asyncio
async def test_the_selected_model_is_read_from_the_installed_function(
    monkeypatch, stored
) -> None:
    """Read back from Open WebUI, not from the rendered default."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import open_webui.models.functions as functions_module

    from open_webui_openrouter_pipe.core.config import _OPENROUTER_IMAGE_GEN_FILTER_MARKER
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    row = SimpleNamespace(id="or_image_gen", content=_OPENROUTER_IMAGE_GEN_FILTER_MARKER)

    class _Table:
        @staticmethod
        async def get_functions_by_type(kind, active_only=False):
            return [SimpleNamespace(id="other", content="unrelated"), row]

        @staticmethod
        async def get_function_valves_by_id(function_id):
            assert function_id == "or_image_gen"
            return {"IMAGE_GENERATION_MODEL": stored}

    monkeypatch.setattr(functions_module, "Functions", _Table)
    manager = FilterManager(pipe=MagicMock(), valves=MagicMock(), logger=MagicMock())

    assert await manager.image_gen_filter_selected_model() == stored


def test_the_sync_s_gate_is_a_strict_subset_of_the_scheduler_s():
    """Discovered over every single-valve-on world, not asserted term by term.

    A scheduler that fires where the sync returns early is a wasted task; a sync that
    would run where the scheduler never fires is a feature that silently never happens.
    The relation has to hold in one direction for every valve, so it is checked that way.
    """
    from types import SimpleNamespace

    from open_webui_openrouter_pipe.core.config import Valves
    from open_webui_openrouter_pipe.models.catalog_manager import (
        schedules_owui_model_sync,
        syncs_owui_models,
    )

    names = [
        name
        for name, field in Valves.model_fields.items()
        if field.annotation is bool and (name.startswith(("UPDATE_MODEL_", "AUTO_")))
    ]
    assert len(names) > 15, (
        f"only {len(names)} candidate valves discovered; the sweep has gone blind"
    )

    off = dict.fromkeys(names, False)
    assert not schedules_owui_model_sync(SimpleNamespace(**off), False), (
        "with every valve off and no routing, nothing should be scheduled"
    )
    assert not syncs_owui_models(SimpleNamespace(**off), False)

    for name in names:
        world = SimpleNamespace(**{**off, name: True})
        if syncs_owui_models(world, False):
            assert schedules_owui_model_sync(world, False), (
                f"{name} makes the sync run but does not make the scheduler start it, so "
                "the run never happens"
            )

    routing = SimpleNamespace(**off)
    assert syncs_owui_models(routing, True) and schedules_owui_model_sync(routing, True), (
        "provider routing must reach both gates"
    )


def test_the_scheduler_carries_exactly_one_term_the_sync_does_not():
    """Named, so promoting or dropping that term is a reviewed edit."""
    from types import SimpleNamespace

    from open_webui_openrouter_pipe.core.config import Valves
    from open_webui_openrouter_pipe.models.catalog_manager import (
        schedules_owui_model_sync,
        syncs_owui_models,
    )

    names = [
        name
        for name, field in Valves.model_fields.items()
        if field.annotation is bool and (name.startswith(("UPDATE_MODEL_", "AUTO_")))
    ]
    off = dict.fromkeys(names, False)
    extra = sorted(
        name
        for name in names
        if schedules_owui_model_sync(SimpleNamespace(**{**off, name: True}), False)
        and not syncs_owui_models(SimpleNamespace(**{**off, name: True}), False)
    )

    assert extra == ["AUTO_ATTACH_IMAGE_GEN_FILTER"], (
        f"the scheduler and the sync now differ by {extra}. Adding a term to only one of "
        "them is the drift these predicates were composed to make impossible; if the "
        "difference is intended, it belongs in schedules_owui_model_sync and here"
    )


@pytest.mark.parametrize("answer", [False, True])
def test_the_scheduler_asks_the_predicate_rather_than_its_own_or_chain(
    pipe_instance, monkeypatch, answer
) -> None:
    """The call site, which two green predicate tests do not reach.

    With the inline `or`-chain left in place beside the new function, every predicate test
    passes and the extraction changes nothing. Forcing the predicate to answer the
    OPPOSITE of what the valves say is the only observation that separates the two.
    """
    from open_webui_openrouter_pipe.models import catalog_manager as cm

    pipe = pipe_instance
    pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = answer

    scheduled = []

    def _fake_create_task(coro, *args, **kwargs):
        scheduled.append(coro)
        coro.close()
        task = Mock()
        task.done.return_value = True
        return task

    monkeypatch.setattr(cm.asyncio, "create_task", _fake_create_task)
    monkeypatch.setattr(cm, "schedules_owui_model_sync", lambda *_a, **_k: answer)

    pipe._catalog_manager.maybe_schedule_model_metadata_sync(
        [{"id": "test"}], pipe_identifier="test_pipe"
    )

    assert bool(scheduled) is answer, (
        "maybe_schedule_model_metadata_sync did not consult schedules_owui_model_sync, so "
        "the predicate and the gate it was extracted from can drift apart again"
    )


@pytest.mark.parametrize("answer", [False, True])
@pytest.mark.asyncio
async def test_the_sync_asks_the_predicate_rather_than_its_own_or_chain(
    pipe_instance, monkeypatch, answer
) -> None:
    """The second call site, for the same reason."""
    from open_webui_openrouter_pipe.models import catalog_manager as cm

    pipe = pipe_instance
    manager = pipe._ensure_catalog_manager()
    pipe.valves.UPDATE_MODEL_CAPABILITIES = not answer

    reached = []

    class _ReachedTheBody(Exception):
        pass

    def _session(*_a, **_k):
        reached.append("x")
        raise _ReachedTheBody

    monkeypatch.setattr(cm, "syncs_owui_models", lambda *_a, **_k: answer)
    monkeypatch.setattr(pipe, "_create_http_session", _session)

    try:
        await manager._sync_model_metadata_to_owui(
            [{"id": "test"}], pipe_identifier="test_pipe"
        )
    except _ReachedTheBody:
        pass

    assert bool(reached) is answer, (
        "_sync_model_metadata_to_owui did not consult syncs_owui_models, so its gate and "
        "the predicate it was extracted from can drift apart again"
    )


class TestTagScannerIsDisabledOnEveryModelRow:
    """Open WebUI's `<think>`-tag scanner has no true positives on a pipe model.

    The pipe emits reasoning as native output items, so the scanner can only ever
    false-positive -- and when it does, it truncates the answer at the tag and swallows
    the rest of the turn. `reasoning_tags: False` on the model row turns it off, and
    `middleware.py` reads that with an identity check, so only the literal False works.
    """

    @pytest.mark.parametrize(
        ("existing", "expected"),
        [
            (None, False),
            ({}, False),
            ({"temperature": 0.5}, False),
            ({"reasoning_tags": ["<a>", "</a>"]}, ["<a>", "</a>"]),
            ({"reasoning_tags": False}, False),
        ],
        ids=["absent", "empty", "other-params", "operator-set", "already-disabled"],
    )
    def test_the_row_carries_a_scanner_verdict(self, existing, expected):
        from open_webui.models.models import ModelParams

        from open_webui_openrouter_pipe.models.catalog_manager import (
            _params_without_tag_scanning,
        )

        source = None if existing is None else ModelParams(**existing)
        result = _params_without_tag_scanning(ModelParams, source).model_dump()
        assert result["reasoning_tags"] == expected

    def test_an_unrelated_param_survives_the_seeding(self):
        from open_webui.models.models import ModelParams

        from open_webui_openrouter_pipe.models.catalog_manager import (
            _params_without_tag_scanning,
        )

        source = ModelParams(temperature=0.7, top_p=0.9)
        result = _params_without_tag_scanning(ModelParams, source).model_dump()
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["reasoning_tags"] is False

    def test_the_seeded_value_is_the_false_singleton(self):
        """`DETECT_REASONING_TAGS = reasoning_tags_param is not False` -- an identity check.

        0, "" and None are all falsy and all leave the scanner running. Only the literal
        False disables it, so the assertion has to be `is`, not `==`.
        """
        from open_webui.models.models import ModelParams

        from open_webui_openrouter_pipe.models.catalog_manager import (
            _params_without_tag_scanning,
        )

        value = _params_without_tag_scanning(ModelParams, None).model_dump()["reasoning_tags"]
        assert value is False
        assert (value is not False) is False
