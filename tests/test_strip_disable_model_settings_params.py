from __future__ import annotations

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import _strip_disable_model_settings_params


def test_strip_disable_model_settings_params_removes_pipe_control_flags() -> None:
    payload = {
        "model": "openai/gpt-5",
        "disable_model_metadata_sync": True,
        "disable_capability_updates": True,
        "disable_image_updates": True,
        "disable_openrouter_search_auto_attach": True,
        "disable_openrouter_search_default_on": True,
        "disable_direct_uploads_auto_attach": True,
        "disable_description_updates": True,
        "disable_native_websearch": True,
        "disable_native_web_search": True,
    }

    _strip_disable_model_settings_params(payload)

    assert payload == {"model": "openai/gpt-5"}

