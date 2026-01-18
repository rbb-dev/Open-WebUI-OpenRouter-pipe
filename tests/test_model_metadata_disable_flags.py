from __future__ import annotations

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
