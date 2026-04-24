# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from open_webui_openrouter_pipe import Pipe


def _make_existing_model(model_id: str, meta: dict, params: dict | None = None):
    from open_webui.models.models import ModelMeta  # provided by test stubs

    return SimpleNamespace(
        id=model_id,
        base_model_id=None,
        name="Example",
        meta=ModelMeta(**meta),
        params=params or {},
        access_grants=[],
        is_active=True,
    )


@pytest.mark.asyncio
async def test_auto_default_web_tools_seeds_default_filter_once(pipe_instance):
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(model_id, meta={})
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
            filter_function_id="openrouter_web_tools",
            filter_supported=True,
            auto_attach_filter=True,
            auto_default_filter=True,
        )

    assert update_mock.call_count == 1
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["filterIds"] == ["openrouter_web_tools"]
    assert meta["defaultFilterIds"] == ["openrouter_web_tools"]
    assert meta["openrouter_pipe"]["web_tools_default_seeded"] is True
    assert meta["openrouter_pipe"]["web_tools_filter_id"] == "openrouter_web_tools"


@pytest.mark.asyncio
async def test_auto_default_web_tools_respects_operator_disabling_default(pipe_instance):
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": ["openrouter_web_tools"],
            "defaultFilterIds": [],
            "openrouter_pipe": {
                "web_tools_default_seeded": True,
                "web_tools_filter_id": "openrouter_web_tools",
            },
        },
    )
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
            filter_function_id="openrouter_web_tools",
            filter_supported=True,
            auto_attach_filter=True,
            auto_default_filter=True,
        )

    # No metadata should be re-written: operator choice is respected.
    assert update_mock.call_count == 0


@pytest.mark.asyncio
async def test_auto_attach_removes_filter_from_unsupported_models_but_preserves_default_ids(pipe_instance):
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": ["openrouter_web_tools"],
            "defaultFilterIds": ["openrouter_web_tools"],
            "openrouter_pipe": {
                "web_tools_default_seeded": True,
                "web_tools_filter_id": "openrouter_web_tools",
            },
        },
    )
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
            filter_function_id="openrouter_web_tools",
            filter_supported=False,
            auto_attach_filter=True,
            auto_default_filter=True,
        )

    assert update_mock.call_count == 1
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["filterIds"] == []
    assert meta["defaultFilterIds"] == ["openrouter_web_tools"]


@pytest.mark.asyncio
async def test_disable_web_tools_auto_attach_prevents_filter_and_default_updates(pipe_instance) -> None:
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={},
        params={"disable_web_tools_auto_attach": True},
    )
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
            filter_function_id="openrouter_web_tools",
            filter_supported=True,
            auto_attach_filter=True,
            auto_default_filter=True,
        )

    assert update_mock.call_count == 0


@pytest.mark.asyncio
async def test_disable_web_tools_default_on_skips_default_filter_ids(pipe_instance) -> None:
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={},
        params={"disable_web_tools_default_on": True},
    )
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
            filter_function_id="openrouter_web_tools",
            filter_supported=True,
            auto_attach_filter=True,
            auto_default_filter=True,
        )

    assert update_mock.call_count == 1
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["filterIds"] == ["openrouter_web_tools"]
    assert "defaultFilterIds" not in meta


# ===== From test_disable_native_websearch.py =====


from open_webui_openrouter_pipe import (
    _apply_disable_native_websearch_to_payload,
)


def test_disable_native_websearch_removes_web_plugin() -> None:
    payload = {
        "model": "openai/gpt-5",
        "disable_native_websearch": True,
        "plugins": [
            {"id": "web", "max_results": 3},
            {"id": "other"},
        ],
    }

    _apply_disable_native_websearch_to_payload(payload)

    assert payload["plugins"] == [{"id": "other"}]
    assert "disable_native_websearch" not in payload


def test_disable_native_websearch_removes_plugins_key_when_empty() -> None:
    payload = {
        "model": "openai/gpt-5",
        "disable_native_websearch": "true",
        "plugins": [{"id": "web"}],
    }

    _apply_disable_native_websearch_to_payload(payload)

    assert "plugins" not in payload
    assert "disable_native_websearch" not in payload


def test_disable_native_websearch_false_keeps_web_plugin() -> None:
    payload = {
        "model": "openai/gpt-5",
        "disable_native_websearch": False,
        "plugins": [{"id": "web"}],
    }

    _apply_disable_native_websearch_to_payload(payload)

    assert payload["plugins"] == [{"id": "web"}]
    assert "disable_native_websearch" not in payload


def test_disable_native_websearch_removes_web_search_options_alias_key() -> None:
    payload = {
        "model": "openai/gpt-5",
        "disable_native_web_search": "1",
        "web_search_options": {"search_context_size": "low"},
    }

    _apply_disable_native_websearch_to_payload(payload)

    assert "web_search_options" not in payload
    assert "disable_native_web_search" not in payload

