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
    files_mod = _ensure_module("open_webui.models.files")
    users_mod = _ensure_module("open_webui.models.users")

    routers_pkg = _ensure_module("open_webui.routers")
    routers_pkg.__path__ = []  # mark as package
    routers_files_mod = _ensure_module("open_webui.routers.files")

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

    class _Files:
        @staticmethod
        def get_file_by_id(_file_id):
            return None

        @staticmethod
        def insert_new_file(*_args, **_kwargs):
            return None

    class _Users:
        @staticmethod
        def get_user_by_id(_user_id):
            return None

    async def _upload_file_handler(*_args, **_kwargs):
        """Stub for upload_file_handler."""
        return None

    chats_mod.Chats = _Chats
    models_mod.ModelForm = _ModelForm
    models_mod.Models = _Models
    files_mod.Files = _Files
    users_mod.Users = _Users
    routers_files_mod.upload_file_handler = _upload_file_handler

    models_pkg.chats = chats_mod
    models_pkg.models = models_mod
    models_pkg.files = files_mod
    models_pkg.users = users_mod
    routers_pkg.files = routers_files_mod
    open_webui.models = models_pkg
    open_webui.routers = routers_pkg



def _install_pydantic_core_stub() -> None:
    core_pkg = _ensure_module("pydantic_core")
    core_schema_mod = _ensure_module("pydantic_core.core_schema")

    def _builder(*args, **kwargs):
        # Return a valid Pydantic schema dictionary with required 'type' key
        return {"type": "any", "args": args, "kwargs": kwargs}

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


def _install_fastapi_stub() -> None:
    """Stub FastAPI to avoid Pydantic schema validation errors.

    The error occurs in fastapi.openapi.models during import:
    KeyError: 'type' in pydantic/_internal/_schema_gather.py:94

    This is a known issue with Pydantic 2.11.9 + FastAPI 0.118.0 when
    using certain BaseModel configurations in FastAPI's OpenAPI models.
    """
    import sys

    fastapi_pkg = _ensure_module("fastapi")

    # Create minimal Request class
    class _Request:
        def __init__(self, *args, **kwargs):
            self.app = None
            self.url = None
            self.headers = {}
            self.query_params = {}

        class App:
            @staticmethod
            def url_path_for(*args, **kwargs):
                return "/api/v1/files/test"

        def __getattr__(self, name):
            if name == "app":
                return self.App()
            return None

    # Create BackgroundTasks stub
    class _BackgroundTasks:
        def add_task(self, *args, **kwargs):
            pass

    # Create UploadFile stub
    class _UploadFile:
        def __init__(self, file=None, filename="", headers=None):
            self.file = file
            self.filename = filename
            self.headers = headers or {}

    # Create Headers stub
    class _Headers(dict):
        pass

    # Create JSONResponse stub
    class _JSONResponse:
        def __init__(self, content=None, status_code=200, **kwargs):
            self.body = content
            self.status_code = status_code
            self.headers = kwargs.get("headers", {})

    # Create run_in_threadpool stub
    async def _run_in_threadpool(func, *args, **kwargs):
        """Stub for FastAPI's run_in_threadpool."""
        return func(*args, **kwargs)

    fastapi_pkg.Request = _Request
    fastapi_pkg.BackgroundTasks = _BackgroundTasks
    fastapi_pkg.UploadFile = _UploadFile

    # Create fastapi.datastructures module
    datastructures_mod = _ensure_module("fastapi.datastructures")
    datastructures_mod.UploadFile = _UploadFile

    # Create fastapi.responses module
    responses_mod = _ensure_module("fastapi.responses")
    responses_mod.JSONResponse = _JSONResponse

    # Create fastapi.concurrency module
    concurrency_mod = _ensure_module("fastapi.concurrency")
    concurrency_mod.run_in_threadpool = _run_in_threadpool

    # Create starlette stubs
    starlette_pkg = _ensure_module("starlette")
    starlette_datastructures_mod = _ensure_module("starlette.datastructures")
    starlette_datastructures_mod.Headers = _Headers
    starlette_pkg.datastructures = starlette_datastructures_mod

    fastapi_pkg.datastructures = datastructures_mod
    fastapi_pkg.responses = responses_mod
    fastapi_pkg.concurrency = concurrency_mod


_ensure_pydantic_backports()
_install_pydantic_core_stub()
_install_open_webui_stubs()
_install_sqlalchemy_stub()
_install_tenacity_stub()
_install_fastapi_stub()  # Add FastAPI stub to fix import errors
