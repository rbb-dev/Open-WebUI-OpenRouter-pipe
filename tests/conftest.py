"""Test configuration helpers for unit tests."""

from __future__ import annotations

import sys
import types

import pydantic


def _ensure_pydantic_backports() -> None:
    if not hasattr(pydantic, "model_validator"):
        def _model_validator(*_args, **_kwargs):
            def decorator(func):
                return func
            return decorator

        pydantic.model_validator = _model_validator  # type: ignore[attr-defined]

    if not hasattr(pydantic, "GetCoreSchemaHandler"):
        class _GetCoreSchemaHandler:  # minimal stub for typing
            ...

        pydantic.GetCoreSchemaHandler = _GetCoreSchemaHandler  # type: ignore[attr-defined]


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


def _install_open_webui_stubs() -> None:
    open_webui = _ensure_module("open_webui")
    models_pkg = _ensure_module("open_webui.models")
    models_pkg.__path__ = []  # mark as package
    chats_mod = _ensure_module("open_webui.models.chats")
    models_mod = _ensure_module("open_webui.models.models")

    class _Chats:
        @staticmethod
        def upsert_message_to_chat_by_id_and_message_id(*_args, **_kwargs):
            return None

    class _ModelForm:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Models:
        @staticmethod
        def get_model_by_id(_model_id):
            return None

        @staticmethod
        def update_model_by_id(_model_id, _model_form):
            return None

    chats_mod.Chats = _Chats
    models_mod.ModelForm = _ModelForm
    models_mod.Models = _Models

    models_pkg.chats = chats_mod
    models_pkg.models = models_mod
    open_webui.models = models_pkg



def _install_pydantic_core_stub() -> None:
    core_pkg = _ensure_module("pydantic_core")
    core_schema_mod = _ensure_module("pydantic_core.core_schema")

    def _builder(*args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    for name in (
        "union_schema",
        "is_instance_schema",
        "chain_schema",
        "str_schema",
        "no_info_plain_validator_function",
        "plain_serializer_function_ser_schema",
    ):
        setattr(core_schema_mod, name, _builder)

    core_pkg.core_schema = core_schema_mod


def _install_sqlalchemy_stub() -> None:
    sa_pkg = _ensure_module("sqlalchemy")
    exc_mod = _ensure_module("sqlalchemy.exc")
    engine_mod = _ensure_module("sqlalchemy.engine")
    orm_mod = _ensure_module("sqlalchemy.orm")

    class _SQLAlchemyError(Exception):
        ...

    class _Engine:
        ...

    class _Session:
        ...

    def _placeholder(*_args, **_kwargs):
        return object()

    def _sessionmaker(*_args, **_kwargs):
        return lambda *a, **k: None

    def _declarative_base(*_args, **_kwargs):
        return type("Base", (), {})

    for attr in ("Boolean", "Column", "DateTime", "JSON", "String", "text", "create_engine", "inspect"):
        setattr(sa_pkg, attr, _placeholder)

    exc_mod.SQLAlchemyError = _SQLAlchemyError
    engine_mod.Engine = _Engine
    orm_mod.Session = _Session
    orm_mod.declarative_base = _declarative_base
    orm_mod.sessionmaker = _sessionmaker

    sa_pkg.exc = exc_mod
    sa_pkg.engine = engine_mod
    sa_pkg.orm = orm_mod


def _install_tenacity_stub() -> None:
    tenacity_mod = _ensure_module("tenacity")

    class _DummyAttempt:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class AsyncRetrying:
        def __init__(self, *args, **kwargs):
            self._yielded = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._yielded:
                raise StopAsyncIteration
            self._yielded = True
            return _DummyAttempt()

    def _passthrough(*_args, **_kwargs):
        return lambda *a, **k: None

    tenacity_mod.AsyncRetrying = AsyncRetrying
    tenacity_mod.retry_if_exception_type = _passthrough
    tenacity_mod.stop_after_attempt = _passthrough
    tenacity_mod.wait_exponential = _passthrough


_ensure_pydantic_backports()
_install_pydantic_core_stub()
_install_open_webui_stubs()
_install_sqlalchemy_stub()
_install_tenacity_stub()
