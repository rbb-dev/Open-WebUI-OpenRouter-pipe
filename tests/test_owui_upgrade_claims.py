import asyncio
import sys
import types
from dataclasses import dataclass
from typing import Any, Optional


def test_claim_pipe_model_metadata_sync_merges_existing_capabilities(monkeypatch):
    """
    Verifies the pipe preserves existing OWUI-only capability keys while updating
    OpenRouter-derived capability fields.
    """
    import open_webui_openrouter_pipe.pipe as pipe_mod

    # Stub the inner import `from open_webui.models.models import ModelMeta, ModelParams`.
    original_models_mod = sys.modules.get("open_webui.models.models")
    stub_models_mod = types.ModuleType("open_webui.models.models")

    class ModelMeta:  # noqa: D401 - minimal stub
        def __init__(self, **kwargs: Any) -> None:
            self._data = dict(kwargs)

        def model_dump(self) -> dict[str, Any]:
            return dict(self._data)

    class ModelParams:  # noqa: D401 - minimal stub
        def __init__(self, **kwargs: Any) -> None:
            self._data = dict(kwargs)

    @dataclass
    class ModelForm:  # noqa: D401 - minimal stub
        id: str
        base_model_id: Optional[str]
        name: str
        meta: Any
        params: Any
        access_control: Any
        is_active: bool

    setattr(stub_models_mod, "ModelMeta", ModelMeta)
    setattr(stub_models_mod, "ModelParams", ModelParams)
    setattr(stub_models_mod, "ModelForm", ModelForm)

    # Ensure the package parent exists.
    sys.modules.setdefault("open_webui.models", types.ModuleType("open_webui.models"))
    sys.modules["open_webui.models.models"] = stub_models_mod

    @dataclass
    class DummyExistingModel:
        id: str
        base_model_id: Optional[str]
        name: str
        meta: Any
        params: Any
        access_control: Any
        is_active: bool

    @dataclass
    class DummyModelForm:
        id: str
        base_model_id: Optional[str]
        name: str
        meta: Any
        params: Any
        access_control: Any
        is_active: bool

    captured: dict[str, Any] = {}

    class DummyModels:
        @staticmethod
        def get_model_by_id(model_id: str) -> Any:
            # Existing model already has OWUI-only capability keys.
            existing_meta = ModelMeta(capabilities={"builtin_tools": False, "file_context": False})
            return DummyExistingModel(
                id=model_id,
                base_model_id=None,
                name="Existing",
                meta=existing_meta,
                params=ModelParams(),
                access_control=None,
                is_active=True,
            )

        @staticmethod
        def update_model_by_id(model_id: str, model_form: Any) -> None:
            captured["model_id"] = model_id
            captured["meta"] = getattr(model_form, "meta", None)

    # Add Models to stub module so catalog_manager can import it
    setattr(stub_models_mod, "Models", DummyModels)

    try:
        monkeypatch.setattr(pipe_mod, "Models", DummyModels)
        monkeypatch.setattr(pipe_mod, "ModelForm", DummyModelForm)

        pipe = pipe_mod.Pipe()
        try:
            pipe._update_or_insert_model_with_metadata(
                "openrouter/test",
                "Test",
                capabilities={"vision": True},  # pipe-provided capabilities update
                profile_image_url=None,
                update_capabilities=True,
                update_images=False,
            )

            # Confirm merge: existing `builtin_tools`/`file_context` were preserved.
            meta = captured["meta"]
            assert meta is not None
            meta_dict = meta.model_dump()
            assert meta_dict.get("capabilities") == {
                "builtin_tools": False,
                "file_context": False,
                "vision": True,
            }
        finally:
            pipe.shutdown()
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                loop.create_task(pipe.close())
            else:
                asyncio.run(pipe.close())
    finally:
        # Restore original module if present.
        if original_models_mod is not None:
            sys.modules["open_webui.models.models"] = original_models_mod
        else:
            sys.modules.pop("open_webui.models.models", None)
