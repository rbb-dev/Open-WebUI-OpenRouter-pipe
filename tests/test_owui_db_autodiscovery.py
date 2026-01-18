from __future__ import annotations

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
