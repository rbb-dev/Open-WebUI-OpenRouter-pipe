"""Test configuration helpers for unit tests."""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: Environment setup MUST happen before ANY other imports
# ─────────────────────────────────────────────────────────────────────────────
# When OWUI middleware is imported, it triggers heavy initialization:
# - Alembic/Peewee database migrations (~30-60s)
# - Vector DB client instantiation with ChromaDB (~20-40s)
# - ML model loading (sentence-transformers, torch) (~60-90s)
# - 100+ PersistentConfig database lookups (~10-20s)
# Total: ~208 seconds without these flags!
import os
os.environ.setdefault("ENABLE_DB_MIGRATIONS", "false")
os.environ.setdefault("DATA_DIR", "/tmp/owui-test-data")
os.environ.setdefault("WEBUI_AUTH", "false")

import asyncio
import base64
import sys
import types
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pydantic
import pytest
import pytest_asyncio


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
    open_webui = cast(Any, _ensure_module("open_webui"))
    models_pkg = cast(Any, _ensure_module("open_webui.models"))
    models_pkg.__path__ = []  # mark as package
    chats_mod = cast(Any, _ensure_module("open_webui.models.chats"))
    models_mod = cast(Any, _ensure_module("open_webui.models.models"))
    files_mod = cast(Any, _ensure_module("open_webui.models.files"))
    users_mod = cast(Any, _ensure_module("open_webui.models.users"))

    routers_pkg = cast(Any, _ensure_module("open_webui.routers"))
    routers_pkg.__path__ = []  # mark as package
    routers_files_mod = cast(Any, _ensure_module("open_webui.routers.files"))

    class _Chats:
        @staticmethod
        def upsert_message_to_chat_by_id_and_message_id(*_args, **_kwargs):
            return None

    class _ModelForm:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _ModelMeta(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def model_dump(self):
            return dict(self)

    class _ModelParams(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class _Models:
        @staticmethod
        def get_model_by_id(_model_id):
            return None

        @staticmethod
        def update_model_by_id(_model_id, _model_form):
            return None

        @staticmethod
        def insert_new_model(_model_form, user_id=""):
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
    models_mod.ModelMeta = _ModelMeta
    models_mod.ModelParams = _ModelParams
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

    # Create open_webui.storage package
    storage_pkg = cast(Any, _ensure_module("open_webui.storage"))
    storage_pkg.__path__ = []  # mark as package
    storage_main_mod = cast(Any, _ensure_module("open_webui.storage.main"))

    async def _upload_file_stub(*args, **kwargs):
        """Stub for Open WebUI's upload_file handler."""
        return None

    storage_main_mod.upload_file = _upload_file_stub
    storage_pkg.main = storage_main_mod
    open_webui.storage = storage_pkg

    utils_pkg = cast(Any, _ensure_module("open_webui.utils"))
    utils_pkg.__path__ = []  # mark as package
    misc_mod = cast(Any, _ensure_module("open_webui.utils.misc"))

    def _openai_chat_message_template(model: str) -> dict[str, Any]:
        import time
        import uuid

        return {
            "id": f"{model}-{str(uuid.uuid4())}",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "logprobs": None, "finish_reason": None}],
        }

    def _openai_chat_chunk_message_template(
        model: str,
        content: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[dict] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        template = _openai_chat_message_template(model)
        template["object"] = "chat.completion.chunk"
        template["choices"][0]["delta"] = {}
        if content:
            template["choices"][0]["delta"]["content"] = content
        if reasoning_content:
            template["choices"][0]["delta"]["reasoning_content"] = reasoning_content
        if tool_calls:
            template["choices"][0]["delta"]["tool_calls"] = tool_calls
        if not content and not reasoning_content and not tool_calls:
            template["choices"][0]["finish_reason"] = "stop"
        if usage:
            template["usage"] = usage
        return template

    async def _run_in_threadpool(func, *args, **kwargs):
        """Stub for Open WebUI's run_in_threadpool."""
        import inspect
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    misc_mod.run_in_threadpool = _run_in_threadpool
    misc_mod.openai_chat_chunk_message_template = _openai_chat_chunk_message_template
    utils_pkg.misc = misc_mod

    # Stub for open_webui.utils.middleware (used by streaming_core.py)
    middleware_mod = cast(Any, _ensure_module("open_webui.utils.middleware"))

    def _apply_source_context_to_messages(
        request_context: Any,
        messages: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        user_message: str,
    ) -> list[dict[str, Any]]:
        """Stub for OWUI's apply_source_context_to_messages.

        In real OWUI, this injects RAG context from sources into the messages
        using <source> XML tags. We simulate that behavior here.
        """
        if not sources or not messages:
            return messages

        # Build source context with <source> tags (matches OWUI format)
        # A single source may contain multiple documents, each gets its own tag
        source_tags = []
        global_idx = 1
        for src in sources:
            name = src.get("name", src.get("source", {}).get("name", f"Source {global_idx}"))
            raw_content = src.get("content", src.get("document", [""]))
            metadata_list = src.get("metadata", [])

            # Handle multiple documents in a single source entry
            if isinstance(raw_content, list):
                for doc_idx, content in enumerate(raw_content):
                    # Get URL from corresponding metadata if available
                    url = ""
                    if doc_idx < len(metadata_list):
                        url = metadata_list[doc_idx].get("source", "")
                    source_tags.append(
                        f'<source id="{global_idx}" name="{name}" url="{url}">{content[:500]}</source>'
                    )
                    global_idx += 1
            else:
                url = src.get("url", src.get("source", {}).get("url", ""))
                source_tags.append(
                    f'<source id="{global_idx}" name="{name}" url="{url}">{raw_content[:500]}</source>'
                )
                global_idx += 1

        if not source_tags:
            return messages

        # Build context block with citation instructions (OWUI format)
        # OWUI adds instructions like "Cite sources as [1], [2], etc."
        context_block = (
            "Use the following sources to answer. Cite using [id] format (e.g., [1], [2]).\n\n"
            + "\n".join(source_tags)
            + "\n\n"
        )

        # Prepend to the last user message (OWUI behavior)
        modified = []
        user_found = False
        for msg in reversed(messages):
            if not user_found and msg.get("role") == "user":
                content = msg.get("content", "")
                # Handle both string content and list content (Chat Completions multimodal format)
                if isinstance(content, str):
                    msg = {**msg, "content": context_block + content}
                elif isinstance(content, list) and content:
                    # Find first text block and prepend source context
                    new_content = []
                    prepended = False
                    for block in content:
                        if not prepended and isinstance(block, dict) and block.get("type") in ("text", "input_text"):
                            new_block = {**block, "text": context_block + block.get("text", "")}
                            new_content.append(new_block)
                            prepended = True
                        else:
                            new_content.append(block)
                    msg = {**msg, "content": new_content}
                user_found = True
            modified.insert(0, msg)
        return modified

    def _get_citation_source_from_tool_result(
        tool_name: str,
        tool_params: dict[str, Any],
        tool_result: str,
        tool_id: str = "",
    ) -> list[dict[str, Any]]:
        """Stub for OWUI's get_citation_source_from_tool_result.

        In real OWUI, this extracts citation sources from tool results.
        For tests, we return a simple citation based on tool output.
        Note: tool_id parameter matches production OWUI signature.
        """
        import json as _json
        sources = []
        try:
            result_data = _json.loads(tool_result) if isinstance(tool_result, str) else tool_result
            if isinstance(result_data, list):
                for item in result_data[:5]:  # Limit to 5 sources
                    if isinstance(item, dict):
                        sources.append({
                            "name": item.get("title", item.get("name", "Source")),
                            "url": item.get("url", item.get("link", "")),
                            "content": item.get("content", item.get("snippet", "")),
                        })
        except Exception:
            pass
        return sources

    middleware_mod.apply_source_context_to_messages = _apply_source_context_to_messages
    middleware_mod.get_citation_source_from_tool_result = _get_citation_source_from_tool_result
    utils_pkg.middleware = middleware_mod
    open_webui.utils = utils_pkg

    config_mod = cast(Any, _ensure_module("open_webui.config"))

    class _ConfigValue:
        def __init__(self, value):
            self.value = value

    config_mod.RAG_FILE_MAX_SIZE = _ConfigValue(None)
    config_mod.FILE_MAX_SIZE = _ConfigValue(None)
    config_mod.BYPASS_EMBEDDING_AND_RETRIEVAL = _ConfigValue(False)
    open_webui.config = config_mod



def _install_pydantic_core_stub() -> None:
    core_pkg = cast(Any, _ensure_module("pydantic_core"))
    core_schema_mod = cast(Any, _ensure_module("pydantic_core.core_schema"))

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
    import importlib.util

    # Prefer real SQLAlchemy when available so tests can exercise DB-backed paths.
    if importlib.util.find_spec("sqlalchemy") is not None:
        return

    sa_pkg = cast(Any, _ensure_module("sqlalchemy"))
    exc_mod = cast(Any, _ensure_module("sqlalchemy.exc"))
    engine_mod = cast(Any, _ensure_module("sqlalchemy.engine"))
    orm_mod = cast(Any, _ensure_module("sqlalchemy.orm"))

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
    # Re-export Engine at top level (as SQLAlchemy 2.0+ does)
    sa_pkg.Engine = _Engine


def _install_tenacity_stub() -> None:
    import importlib.util

    # Prefer the real tenacity implementation when available so tests exercise
    # production retry semantics.
    if importlib.util.find_spec("tenacity") is not None:
        return

    tenacity_mod = cast(Any, _ensure_module("tenacity"))

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


# ─────────────────────────────────────────────────────────────────────────────
# Shared Fixtures
# ─────────────────────────────────────────────────────────────────────────────

_PIPES_TO_CLOSE: list["Pipe"] = []


def _schedule_pipe_cleanup(pipe: "Pipe") -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(pipe.close())
    else:
        _PIPES_TO_CLOSE.append(pipe)


@pytest_asyncio.fixture(autouse=True)
async def _cleanup_pending_pipes():
    yield
    while _PIPES_TO_CLOSE:
        pipe = _PIPES_TO_CLOSE.pop()
        await pipe.close()


@pytest.fixture
def pipe_instance(request):
    """Return a fresh Pipe instance for tests."""
    pipe = Pipe()

    def _finalize() -> None:
        _schedule_pipe_cleanup(pipe)

    request.addfinalizer(_finalize)
    return pipe


@pytest_asyncio.fixture
async def pipe_instance_async():
    """Return a fresh Pipe instance for async tests with proper cleanup."""
    pipe = Pipe()
    yield pipe
    await pipe.close()


@pytest.fixture
def mock_request():
    """Mock FastAPI request used for storage uploads."""
    request = Mock()
    request.app = Mock()
    request.app.url_path_for = Mock(return_value="/api/v1/files/test123")
    return request


@pytest.fixture
def mock_user():
    """Mock user object used for uploads and storage context."""
    user = Mock()
    user.id = "user123"
    user.email = "test@example.com"
    user.name = "Test User"
    return user


@pytest.fixture
def sample_image_base64() -> str:
    """Return a 1x1 transparent PNG encoded as base64."""
    return (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


@pytest.fixture
def sample_audio_base64() -> str:
    """Return sample base64-encoded audio data."""
    return base64.b64encode(b"FAKE_AUDIO_DATA").decode("utf-8")


def _install_fastapi_stub() -> None:
    """Stub FastAPI to avoid Pydantic schema validation errors.

    The error occurs in fastapi.openapi.models during import:
    KeyError: 'type' in pydantic/_internal/_schema_gather.py:94

    This is a known issue with Pydantic 2.11.9 + FastAPI 0.118.0 when
    using certain BaseModel configurations in FastAPI's OpenAPI models.
    """
    import sys

    fastapi_pkg = cast(Any, _ensure_module("fastapi"))

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
        def __init__(self, file=None, filename="", headers=None, content_type=None):
            self.file = file
            self.filename = filename
            self.headers = headers or {}
            # Derive content_type from headers if not explicitly provided (like real FastAPI UploadFile)
            if content_type is not None:
                self.content_type = content_type
            elif self.headers:
                self.content_type = self.headers.get("content-type")

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
    datastructures_mod = cast(Any, _ensure_module("fastapi.datastructures"))
    datastructures_mod.UploadFile = _UploadFile

    # Create fastapi.responses module
    responses_mod = cast(Any, _ensure_module("fastapi.responses"))
    responses_mod.JSONResponse = _JSONResponse

    # Create fastapi.concurrency module
    concurrency_mod = cast(Any, _ensure_module("fastapi.concurrency"))
    concurrency_mod.run_in_threadpool = _run_in_threadpool

    # Create starlette stubs
    starlette_pkg = cast(Any, _ensure_module("starlette"))
    starlette_datastructures_mod = cast(Any, _ensure_module("starlette.datastructures"))
    starlette_datastructures_mod.Headers = _Headers
    starlette_pkg.datastructures = starlette_datastructures_mod

    # Starlette requests stub
    starlette_requests_mod = cast(Any, _ensure_module("starlette.requests"))
    class _StarletteRequest:
        """Stub for starlette Request."""
        ...
    starlette_requests_mod.Request = _StarletteRequest
    starlette_pkg.requests = starlette_requests_mod

    fastapi_pkg.datastructures = datastructures_mod
    fastapi_pkg.responses = responses_mod
    fastapi_pkg.concurrency = concurrency_mod


# ─────────────────────────────────────────────────────────────────────────────
# Install Stubs (MUST RUN BEFORE IMPORTING PIPE)
# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: These MUST be called before "from open_webui_openrouter_pipe import Pipe"
# otherwise pipe.py will execute its try/except blocks and set Models=None, Chats=None, etc.

_ensure_pydantic_backports()
_install_pydantic_core_stub()
_install_open_webui_stubs()
_install_sqlalchemy_stub()
_install_tenacity_stub()
_install_fastapi_stub()


def _maybe_install_bundled_pipe() -> None:
    """Optionally preload a generated monolith bundle for testing.

    Set `OWUI_PIPE_BUNDLE_PATH` to a bundled .py file (e.g.
    open_webui_openrouter_pipe_bundled.py or open_webui_openrouter_pipe_bundled_compressed.py)
    to run the test suite against the single-file package import-hook implementation.
    """
    bundle_path = os.environ.get("OWUI_PIPE_BUNDLE_PATH")
    if not bundle_path:
        return

    # Pytest loads a bootstrap plugin from `pytest.ini` under this package name
    # (open_webui_openrouter_pipe.pytest_bootstrap) before it imports conftest.
    # When running in bundled mode we want to replace the on-disk package with
    # the monolith implementation, so purge any previously imported package
    # modules before importing the bundle file.
    prefix = "open_webui_openrouter_pipe"
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)

    path = Path(bundle_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"OWUI_PIPE_BUNDLE_PATH does not exist: {path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("owui_pipe_bundle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for bundle: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["owui_pipe_bundle"] = module
    spec.loader.exec_module(module)


# ─────────────────────────────────────────────────────────────────────────────
# Import Pipe (AFTER stubs are installed)
# ─────────────────────────────────────────────────────────────────────────────

_maybe_install_bundled_pipe()

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry, ModelFamily


# ─────────────────────────────────────────────────────────────────────────────
# Global Model Registry Reset (prevents catalog leakage between tests)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_model_registry():
    """Reset OpenRouterModelRegistry class-level state before each test.

    The registry uses class-level attributes for catalog caching. Without this reset,
    tests that run earlier can pollute the catalog state, causing later tests that
    mock HTTP responses to fail because the mock is never hit (cache is still valid).
    """
    reg = OpenRouterModelRegistry
    reg._models = []
    reg._specs = {}
    reg._id_map = {}
    reg._last_fetch = 0.0
    reg._lock = asyncio.Lock()
    reg._next_refresh_after = 0.0
    reg._consecutive_failures = 0
    reg._last_error = None
    reg._last_error_time = 0.0
    ModelFamily.set_dynamic_specs(None)
    yield
