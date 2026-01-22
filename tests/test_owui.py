from __future__ import annotations

import json
from typing import Any

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import OpenRouterModelRegistry, Pipe


@pytest.mark.asyncio
async def test_owui_metadata_tools_registry_is_used_for_native_tools(monkeypatch):
    """OWUI 0.7.x puts builtin tool executors in __metadata__["tools"], not __tools__.

    This test verifies that:
    1. Tools from __metadata__["tools"] are properly extracted
    2. They are passed to the OpenRouter API request
    3. The real infrastructure processes tool calling correctly
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key-12345")

    pipe = Pipe()
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    # Clear registry to ensure we fetch from mocked endpoint
    OpenRouterModelRegistry._models = []
    OpenRouterModelRegistry._specs = {}
    OpenRouterModelRegistry._id_map = {}
    OpenRouterModelRegistry._last_fetch = 0.0

    # Track if tool was called
    tool_called = False
    tool_call_args = None

    async def builtin_search_web(query: str):
        nonlocal tool_called, tool_call_args
        tool_called = True
        tool_call_args = {"query": query}
        return {"ok": True, "query": query, "results": ["result1", "result2"]}

    # Build metadata with tool registry (OWUI 0.7.x style)
    metadata = {
        "chat_id": "c1",
        "message_id": "m1",
        "session_id": "s1",
        "model": {"id": "openai/gpt-4o-mini"},
        # In OWUI 0.7.x, middleware attaches the tool registry to metadata["tools"]
        "tools": {
            "search_web": {
                "tool_id": "builtin:search_web",
                "type": "builtin",
                "spec": {
                    "name": "search_web",
                    "description": "Search the web.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
                "callable": builtin_search_web,
            }
        },
    }

    # Mock HTTP at boundary
    with aioresponses() as mock_http:
        # Mock catalog endpoint
        catalog_response = {
            "data": [
                {
                    "id": "openai/gpt-4o-mini",
                    "name": "GPT-4o Mini",
                    "context_length": 128000,
                    "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
                    "supported_parameters": ["tools", "tool_choice"],
                }
            ]
        }
        mock_http.get(
            "https://openrouter.ai/api/v1/models",
            payload=catalog_response,
            headers={"Content-Type": "application/json"},
        )

        # Mock the API response with a tool call (using /responses endpoint)
        # In Responses API, function calls are separate items with type:"function_call"
        api_response = {
            "id": "gen-123",
            "model": "openai/gpt-4o-mini",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_abc123",
                    "name": "search_web",
                    "arguments": json.dumps({"query": "hello world"}),
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=api_response,
            headers={"Content-Type": "application/json"},
        )

        # Mock the follow-up response after tool execution
        final_response = {
            "id": "gen-124",
            "model": "openai/gpt-4o-mini",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I found some results for you!"}],
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        }
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=final_response,
            headers={"Content-Type": "application/json"},
        )

        try:
            body = {
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "search for hello world"}],
                "stream": False,
                # Native tools are sent in OpenAI-style chat tool schema by OWUI middleware
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "description": "Search the web.",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            },
                        },
                    }
                ],
            }

            result = await pipe.pipe(
                body=body,
                __user__={"id": "u1"},
                __request__=None,
                __event_emitter__=None,
                __event_call__=None,
                __metadata__=metadata,
                __tools__={},  # Empty - tools come from metadata["tools"]
            )

            # Verify tool was actually called through the real infrastructure
            # This is the KEY assertion: tools from metadata["tools"] must be used
            assert tool_called, "Tool should have been called through real execution"
            assert tool_call_args is not None
            assert tool_call_args["query"] == "hello world"

            # Verify the result is valid (non-streaming with event_emitter=None returns string)
            assert result is not None
            assert isinstance(result, str)

        finally:
            await session.close()
            await pipe.close()
            # Clear the registry to avoid polluting other tests
            OpenRouterModelRegistry._models = []
            OpenRouterModelRegistry._specs = {}
            OpenRouterModelRegistry._id_map = {}
            OpenRouterModelRegistry._last_fetch = 0.0

# ===== From test_owui_upgrade_claims.py =====

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

# ===== From test_owui_db_autodiscovery.py =====


import sys
import types
from typing import Any


class _ContextManager:
    def __init__(self, value: Any):
        self._value = value

    def __enter__(self) -> Any:
        return self._value

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_discover_engine_prefers_db_context_over_symbol_names() -> None:
    import open_webui_openrouter_pipe.pipe as pipe_mod

    engine = object()

    class _Session:
        def get_bind(self) -> Any:
            return engine

    owui_db: Any = types.ModuleType("open_webui.internal.db")
    owui_db.get_db_context = lambda *args, **kwargs: _ContextManager(_Session())

    discovered_engine, discovered_schema, details = pipe_mod.Pipe._discover_owui_engine_and_schema(owui_db)
    assert discovered_engine is engine
    assert discovered_schema is None
    assert details["engine_source"] == "get_db_context.get_bind"


def test_discover_schema_prefers_base_metadata_schema() -> None:
    import open_webui_openrouter_pipe.pipe as pipe_mod

    engine = object()

    class _Meta:
        schema = "owui_schema"

    class _Base:
        metadata = _Meta()

    owui_db: Any = types.ModuleType("open_webui.internal.db")
    owui_db.engine = engine
    owui_db.Base = _Base

    discovered_engine, discovered_schema, details = pipe_mod.Pipe._discover_owui_engine_and_schema(owui_db)
    assert discovered_engine is engine
    assert discovered_schema == "owui_schema"
    assert details["schema_source"] == "owui_db.Base.metadata.schema"


def test_discover_schema_falls_back_to_open_webui_env() -> None:
    import open_webui_openrouter_pipe.pipe as pipe_mod

    engine = object()

    owui_db: Any = types.ModuleType("open_webui.internal.db")
    owui_db.engine = engine

    open_webui_pkg = sys.modules.get("open_webui")
    assert open_webui_pkg is not None

    original_path = getattr(open_webui_pkg, "__path__", None)
    setattr(open_webui_pkg, "__path__", [])

    env_mod: Any = types.ModuleType("open_webui.env")
    env_mod.DATABASE_SCHEMA = "fallback_schema"
    sys.modules["open_webui.env"] = env_mod

    try:
        discovered_engine, discovered_schema, details = pipe_mod.Pipe._discover_owui_engine_and_schema(owui_db)
        assert discovered_engine is engine
        assert discovered_schema == "fallback_schema"
        assert details["schema_source"] == "open_webui.env.DATABASE_SCHEMA"
    finally:
        if original_path is None:
            delattr(open_webui_pkg, "__path__")
        else:
            setattr(open_webui_pkg, "__path__", original_path)
        sys.modules.pop("open_webui.env", None)
