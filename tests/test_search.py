from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_webui_openrouter_pipe import Pipe


def _make_existing_model(model_id: str, meta: dict, params: dict | None = None):
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


def test_auto_default_openrouter_search_seeds_default_filter_once(pipe_instance):
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(model_id, meta={})
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
            filter_function_id="openrouter_search",
            filter_supported=True,
            auto_attach_filter=True,
            auto_default_filter=True,
        )

    assert update_mock.call_count == 1
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["filterIds"] == ["openrouter_search"]
    assert meta["defaultFilterIds"] == ["openrouter_search"]
    assert meta["openrouter_pipe"]["openrouter_search_default_seeded"] is True
    assert meta["openrouter_pipe"]["openrouter_search_filter_id"] == "openrouter_search"


def test_auto_default_openrouter_search_respects_operator_disabling_default(pipe_instance):
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": ["openrouter_search"],
            "defaultFilterIds": [],
            "openrouter_pipe": {
                "openrouter_search_default_seeded": True,
                "openrouter_search_filter_id": "openrouter_search",
            },
        },
    )
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
            filter_function_id="openrouter_search",
            filter_supported=True,
            auto_attach_filter=True,
            auto_default_filter=True,
        )

    # No metadata should be re-written: operator choice is respected.
    assert update_mock.call_count == 0


def test_auto_attach_removes_filter_from_unsupported_models_but_preserves_default_ids(pipe_instance):
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": ["openrouter_search"],
            "defaultFilterIds": ["openrouter_search"],
            "openrouter_pipe": {
                "openrouter_search_default_seeded": True,
                "openrouter_search_filter_id": "openrouter_search",
            },
        },
    )
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
            filter_function_id="openrouter_search",
            filter_supported=False,
            auto_attach_filter=True,
            auto_default_filter=True,
        )

    assert update_mock.call_count == 1
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["filterIds"] == []
    assert meta["defaultFilterIds"] == ["openrouter_search"]


def test_disable_openrouter_search_auto_attach_prevents_filter_and_default_updates(pipe_instance) -> None:
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={},
        params={"disable_openrouter_search_auto_attach": True},
    )
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
            filter_function_id="openrouter_search",
            filter_supported=True,
            auto_attach_filter=True,
            auto_default_filter=True,
        )

    assert update_mock.call_count == 0


def test_disable_openrouter_search_default_on_skips_default_filter_ids(pipe_instance) -> None:
    pipe = pipe_instance
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={},
        params={"disable_openrouter_search_default_on": True},
    )
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
            filter_function_id="openrouter_search",
            filter_supported=True,
            auto_attach_filter=True,
            auto_default_filter=True,
        )

    assert update_mock.call_count == 1
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["filterIds"] == ["openrouter_search"]
    assert "defaultFilterIds" not in meta


# ===== From test_openrouter_search_toggle_filter.py =====


from typing import Any, cast

from filters.openrouter_search_toggle import Filter, _FEATURE_FLAG


def test_search_toggle_sets_features_and_metadata_marker():
    filt = Filter()

    features: dict = {}
    body = {"features": features}
    metadata = {"features": features}

    result = filt.inlet(body, __metadata__=metadata)

    assert result["features"]["web_search"] is False
    assert metadata["features"] is not features
    assert metadata["features"][_FEATURE_FLAG] is True


def test_search_toggle_skips_when_metadata_invalid():
    filt = Filter()
    body = {}

    result = filt.inlet(body, __metadata__=cast(Any, "invalid"))

    assert result is body
    assert "features" not in result


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

