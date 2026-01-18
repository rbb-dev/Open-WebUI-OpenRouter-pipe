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
