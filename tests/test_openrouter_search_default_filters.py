from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import Pipe


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


def test_auto_default_openrouter_search_seeds_default_filter_once():
    pipe = Pipe()
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(model_id, meta={})
    update_mock = Mock()

    with patch("open_webui_openrouter_pipe.open_webui_openrouter_pipe.Models.get_model_by_id", return_value=existing), patch(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.Models.update_model_by_id", new=update_mock
    ), patch("open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelForm", new=lambda **kw: SimpleNamespace(**kw)):
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


def test_auto_default_openrouter_search_respects_operator_disabling_default():
    pipe = Pipe()
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

    with patch("open_webui_openrouter_pipe.open_webui_openrouter_pipe.Models.get_model_by_id", return_value=existing), patch(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.Models.update_model_by_id", new=update_mock
    ), patch("open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelForm", new=lambda **kw: SimpleNamespace(**kw)):
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


def test_auto_attach_removes_filter_from_unsupported_models_but_preserves_default_ids():
    pipe = Pipe()
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

    with patch("open_webui_openrouter_pipe.open_webui_openrouter_pipe.Models.get_model_by_id", return_value=existing), patch(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.Models.update_model_by_id", new=update_mock
    ), patch("open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelForm", new=lambda **kw: SimpleNamespace(**kw)):
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


def test_disable_openrouter_search_auto_attach_prevents_filter_and_default_updates() -> None:
    pipe = Pipe()
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={},
        params={"disable_openrouter_search_auto_attach": True},
    )
    update_mock = Mock()

    with patch("open_webui_openrouter_pipe.open_webui_openrouter_pipe.Models.get_model_by_id", return_value=existing), patch(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.Models.update_model_by_id", new=update_mock
    ), patch("open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelForm", new=lambda **kw: SimpleNamespace(**kw)):
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


def test_disable_openrouter_search_default_on_skips_default_filter_ids() -> None:
    pipe = Pipe()
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={},
        params={"disable_openrouter_search_default_on": True},
    )
    update_mock = Mock()

    with patch("open_webui_openrouter_pipe.open_webui_openrouter_pipe.Models.get_model_by_id", return_value=existing), patch(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.Models.update_model_by_id", new=update_mock
    ), patch("open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelForm", new=lambda **kw: SimpleNamespace(**kw)):
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
