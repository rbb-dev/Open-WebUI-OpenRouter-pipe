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
        async def upsert_message_to_chat_by_id_and_message_id(*_args, **_kwargs):
            return None

        @staticmethod
        async def get_message_by_id_and_message_id(*_args, **_kwargs):
            return None

        @staticmethod
        async def insert_chat_files(chat_id, message_id, file_ids, user_id, db=None):
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
        async def get_model_by_id(_model_id):
            return None

        @staticmethod
        async def get_all_models():
            return []

        @staticmethod
        async def update_model_by_id(_model_id, _model_form):
            return None

        @staticmethod
        async def insert_new_model(_model_form, user_id=""):
            return None

    class _Files:
        @staticmethod
        async def get_file_by_id(_file_id):
            return None

        @staticmethod
        async def insert_new_file(*_args, **_kwargs):
            return None

    class _FileForm:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def model_dump(self):
            return dict(self.__dict__)

    class _Users:
        @staticmethod
        async def get_user_by_id(_user_id):
            return None

        @staticmethod
        async def get_user_by_email(email, db=None):
            return None

        @staticmethod
        async def insert_new_user(id, name, email, profile_image_url='/user.png', role='pending', username=None, oauth=None, db=None):
            return type('UserModel', (), {'id': id, 'name': name, 'email': email, 'role': role, 'profile_image_url': profile_image_url})()

    async def _upload_file_handler(*_args, **_kwargs):
        """Stub for upload_file_handler."""
        return None

    class _FunctionMeta(pydantic.BaseModel):
        description: str = ""
        manifest: dict = {}

    class _FunctionForm(pydantic.BaseModel):
        id: str = ""
        name: str = ""
        type: str = ""
        content: str = ""
        meta: _FunctionMeta = _FunctionMeta()

    class _Functions:
        @staticmethod
        async def get_functions_by_type(type, active_only=False, db=None):
            return []

        @staticmethod
        async def get_function_by_id(id, db=None):
            return None

        @staticmethod
        async def get_function_valves_by_id(id, db=None):
            return {}

        @staticmethod
        async def insert_new_function(user_id, type, form_data, db=None):
            return None

        @staticmethod
        async def update_function_by_id(id, updated, db=None):
            return None

    functions_mod = cast(Any, _ensure_module("open_webui.models.functions"))
    functions_mod.Functions = _Functions
    functions_mod.FunctionForm = _FunctionForm
    functions_mod.FunctionMeta = _FunctionMeta

    chats_mod.Chats = _Chats
    models_mod.ModelForm = _ModelForm
    models_mod.ModelMeta = _ModelMeta
    models_mod.ModelParams = _ModelParams
    models_mod.Models = _Models
    files_mod.Files = _Files
    files_mod.FileForm = _FileForm
    users_mod.Users = _Users
    routers_files_mod.upload_file_handler = _upload_file_handler

    storage_pkg = cast(Any, _ensure_module("open_webui.storage"))
    storage_pkg.__path__ = []
    storage_provider_mod = cast(Any, _ensure_module("open_webui.storage.provider"))

    class _Storage:
        @staticmethod
        def upload_file(file, filename, tags=None):
            contents = file.read()
            if not contents:
                raise ValueError("empty file")
            return contents, f"/tmp/{filename}"

        @staticmethod
        def delete_file(_file_path):
            return None

        @staticmethod
        def get_file(file_path):
            return file_path

    storage_provider_mod.Storage = _Storage
    storage_pkg.provider = storage_provider_mod

    models_pkg.chats = chats_mod
    models_pkg.models = models_mod
    models_pkg.files = files_mod
    models_pkg.users = users_mod
    models_pkg.functions = functions_mod
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

    # Loader seams used by the pipe_dashboard update service. Faithful minimal
    # copies of OWUI 0.10.2 behavior; a real open_webui import always wins
    # (attributes are only added when missing).
    plugin_mod = cast(Any, _ensure_module("open_webui.utils.plugin"))
    if not hasattr(plugin_mod, "extract_frontmatter"):
        def _extract_frontmatter(content: str) -> dict:
            import re as _re

            frontmatter: dict[str, str] = {}
            pattern = _re.compile(r"^\s*([a-z_]+):\s*(.*)\s*$", _re.IGNORECASE)
            try:
                lines = content.splitlines()
                if len(lines) < 1 or lines[0].strip() != '"""':
                    return {}
                for line in lines[1:]:
                    if '"""' in line:
                        break
                    match = pattern.match(line)
                    if match:
                        key, value = match.groups()
                        frontmatter[key.strip()] = value.strip()
            except Exception:
                return {}
            return frontmatter

        plugin_mod.extract_frontmatter = _extract_frontmatter
    if not hasattr(plugin_mod, "replace_imports"):
        def _replace_imports(content: str) -> str:
            for old, new in {
                "from utils": "from open_webui.utils",
                "from apps": "from open_webui.apps",
                "from main": "from open_webui.main",
                "from config": "from open_webui.config",
            }.items():
                content = content.replace(old, new)
            return content

        plugin_mod.replace_imports = _replace_imports
    if not hasattr(plugin_mod, "load_function_module_by_id"):
        async def _load_function_module_by_id(function_id: str, content: str | None = None):
            raise NotImplementedError(
                "patch open_webui.utils.plugin.load_function_module_by_id in tests"
            )

        plugin_mod.load_function_module_by_id = _load_function_module_by_id
    if not hasattr(plugin_mod, "get_functions_cache"):
        def _plugin_state_cache(request: Any, name: str) -> dict:
            state = request.app.state
            if not hasattr(state, name):
                setattr(state, name, {})
            return getattr(state, name)

        plugin_mod.get_functions_cache = lambda request: _plugin_state_cache(request, "FUNCTIONS")
        plugin_mod.get_function_contents_cache = lambda request: _plugin_state_cache(
            request, "FUNCTION_CONTENTS"
        )
    utils_pkg.plugin = plugin_mod

    env_mod = cast(Any, _ensure_module("open_webui.env"))
    if not hasattr(env_mod, "VERSION"):
        env_mod.VERSION = "0.10.2"
    if not hasattr(env_mod, "SRC_LOG_LEVELS"):
        env_mod.SRC_LOG_LEVELS = {}
    open_webui.env = env_mod

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
        _reasoning_unused: str | None = None,
        tool_calls: list[dict] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Mirrors the pipe's ignore-3rd contract: the former reasoning slot is
        # accepted positionally for OWUI signature parity but never placed in the
        # chunk delta (reasoning flows via native output items, not chunk deltas).
        template = _openai_chat_message_template(model)
        template["object"] = "chat.completion.chunk"
        template["choices"][0]["delta"] = {}
        if content:
            template["choices"][0]["delta"]["content"] = content
        if tool_calls:
            template["choices"][0]["delta"]["tool_calls"] = tool_calls
        if not content and not tool_calls:
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

    def _sanitize_text_for_db(text: str) -> str:
        if not isinstance(text, str):
            return text
        text = text.replace("\x00", "").replace("\u0000", "")
        try:
            text = text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        return text

    def _sanitize_data_for_db(obj):
        if isinstance(obj, str):
            return _sanitize_text_for_db(obj)
        if isinstance(obj, dict):
            return {k: _sanitize_data_for_db(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_data_for_db(v) for v in obj]
        return obj

    misc_mod.run_in_threadpool = _run_in_threadpool
    misc_mod.sanitize_text_for_db = _sanitize_text_for_db
    misc_mod.sanitize_data_for_db = _sanitize_data_for_db
    misc_mod.openai_chat_chunk_message_template = _openai_chat_chunk_message_template
    utils_pkg.misc = misc_mod

    # Stub for open_webui.utils.middleware (used by streaming_core.py)
    middleware_mod = cast(Any, _ensure_module("open_webui.utils.middleware"))

    async def _apply_source_context_to_messages(
        request_context: Any,
        messages: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        user_message: str,
    ) -> list[dict[str, Any]]:
        """Stub for OWUI's apply_source_context_to_messages.

        In real OWUI (≥0.9.x), this function is async. The stub mirrors that
        signature so production code's `await` works against both real OWUI
        and the test stub.

        Injects RAG context from sources into the messages using <source> XML
        tags. Simulates the real implementation's behavior.
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

    async def _process_tool_result(request=None, tool_function_name='', tool_result='', tool_type='', direct_tool=False, metadata=None, user=None):
        return (str(tool_result), [], [])

    middleware_mod.apply_source_context_to_messages = _apply_source_context_to_messages
    middleware_mod.get_citation_source_from_tool_result = _get_citation_source_from_tool_result
    middleware_mod.process_tool_result = _process_tool_result
    utils_pkg.middleware = middleware_mod

    # Stub for open_webui.utils.access_control.files.has_access_to_file
    # (lazily imported by storage/owui_files.authorize_file_read). The default
    # raises so tests must opt in explicitly via monkeypatch; this also exercises
    # the gateway's fail-closed except path when left unpatched. Additive only.
    access_control_pkg = cast(Any, _ensure_module("open_webui.utils.access_control"))
    access_control_pkg.__path__ = []  # mark as package
    access_control_files_mod = cast(Any, _ensure_module("open_webui.utils.access_control.files"))

    async def _has_access_to_file(_file_id, _access_type, _user, db=None):
        """Default stub: raise so callers must monkeypatch per-test."""
        raise NotImplementedError(
            "has_access_to_file is not stubbed; monkeypatch it in the test"
        )

    access_control_files_mod.has_access_to_file = _has_access_to_file
    access_control_pkg.files = access_control_files_mod
    utils_pkg.access_control = access_control_pkg

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
    tenacity_mod.retry_if_not_exception_type = _passthrough
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
    reg._zdr_model_ids = None
    reg._last_fetch = 0.0
    reg._last_video_fetch = 0.0
    reg._last_video_attempt = 0.0
    reg._last_image_fetch = 0.0
    reg._last_image_attempt = 0.0
    reg._lock = asyncio.Lock()
    reg._next_refresh_after = 0.0
    reg._consecutive_failures = 0
    reg._last_error = None
    reg._last_error_time = 0.0
    ModelFamily.set_dynamic_specs(None)
    yield


@pytest.fixture(autouse=True)
def _isolate_webui_secret_key(monkeypatch):
    """Keep WEBUI_SECRET_KEY unset by default so the SEND_CACHE_SESSION_ID cache pin is
    deterministic regardless of ambient env or test order; tests that exercise the pin set
    it explicitly via monkeypatch.setenv.
    """
    monkeypatch.delenv("WEBUI_SECRET_KEY", raising=False)
