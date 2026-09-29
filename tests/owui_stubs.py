"""Third-party stand-ins the pipe imports at module scope.

Separate from conftest because a subprocess probe needs the stubs WITHOUT the fixture
library: conftest also imports Pipe, which drags redis and opentelemetry in behind it
and costs ~4.3s per interpreter -- more than the real Open WebUI import it replaces.
conftest imports this module, so there is one installation of these stubs, not two.
"""

from __future__ import annotations

import json
import os
os.environ.setdefault("ENABLE_DB_MIGRATIONS", "false")
os.environ.setdefault("DATA_DIR", "/tmp/owui-test-data")
os.environ.setdefault("WEBUI_AUTH", "false")

import inspect
import sys
import types
import uuid
from collections.abc import Awaitable, Callable
from enum import Enum
from functools import partial, update_wrapper
from types import SimpleNamespace
from typing import Any, cast, get_args, get_type_hints

import pydantic


def _ensure_pydantic_backports() -> None:
    if not hasattr(pydantic, "model_validator"):
        def _model_validator(*_args, **_kwargs):
            def decorator(func):
                return func
            return decorator

        pydantic.model_validator = _model_validator  # type: ignore[attr-defined]

    if not hasattr(pydantic, "GetCoreSchemaHandler"):
        class _GetCoreSchemaHandler:
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
    models_pkg.__path__ = []
    chats_mod = cast(Any, _ensure_module("open_webui.models.chats"))
    messages_mod = cast(Any, _ensure_module("open_webui.models.messages"))
    messages_mod.Messages = types.SimpleNamespace(get_message_by_id=None)
    models_mod = cast(Any, _ensure_module("open_webui.models.models"))
    files_mod = cast(Any, _ensure_module("open_webui.models.files"))
    users_mod = cast(Any, _ensure_module("open_webui.models.users"))

    routers_pkg = cast(Any, _ensure_module("open_webui.routers"))
    routers_pkg.__path__ = []
    routers_files_mod = cast(Any, _ensure_module("open_webui.routers.files"))

    class _Chats:
        @staticmethod
        async def upsert_message_to_chat_by_id_and_message_id(*_args, **_kwargs):
            return None

        @staticmethod
        async def get_message_by_id_and_message_id(*_args, **_kwargs):
            return None

        _chat_files: dict[tuple, list] = {}

        @staticmethod
        async def insert_chat_files(chat_id, message_id, file_ids, user_id, db=None):
            if not file_ids:
                return None
            key = (chat_id, message_id)
            rows = _Chats._chat_files.setdefault(key, [])

            existing = {r.file_id for r in rows}
            if any(f in existing for f in file_ids if f):
                return None

            created = [
                SimpleNamespace(
                    id=f"row-{uuid.uuid4().hex}",
                    user_id=user_id,
                    chat_id=chat_id,
                    message_id=message_id,
                    file_id=f,
                )
                for f in file_ids
                if f
            ]
            if not created:
                return None
            rows.extend(created)
            return created

        @staticmethod
        async def get_chat_files_by_chat_id_and_message_id(chat_id, message_id, db=None):
            return list(_Chats._chat_files.get((chat_id, message_id), []))

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

        def model_dump(self):
            return dict(self)

    class _Models:
        @staticmethod
        async def get_model_by_id(_model_id):
            return None

        @staticmethod
        async def get_models_by_ids(_ids, db=None):
            return []

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
        # Open WebUI's real FunctionMeta is extra='allow'; pydantic's default is
        # 'ignore', which silently dropped every key the pipe sets beyond these two.
        model_config = pydantic.ConfigDict(extra="allow")
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
        async def get_user_valves_by_id_and_user_id(id, user_id, db=None):
            return {}

        @staticmethod
        async def insert_new_function(user_id, type, form_data, db=None):
            return None

        @staticmethod
        async def update_function_by_id(id, updated, db=None):
            return None

        @staticmethod
        async def delete_function_by_id(id, db=None):
            return True

    functions_mod = cast(Any, _ensure_module("open_webui.models.functions"))
    functions_mod.Functions = _Functions
    functions_mod.FunctionForm = _FunctionForm
    functions_mod.FunctionMeta = _FunctionMeta

    models_config_mod = cast(Any, _ensure_module("open_webui.models.config"))

    class _Config:
        # The Task Model settings an admin picks in OWUI land here as
        # `task.model.default` / `task.model.external` (routers/tasks.py:41-44).
        # Module-global so a test can set rows, and a conftest sweep clears them
        # per test; the real one reads a DB row per call and caches nothing.
        _rows: dict[str, Any] = {}

        @staticmethod
        async def get_many(*keys: str) -> dict[str, Any]:
            return {k: _Config._rows[k] for k in keys if k in _Config._rows}

        @staticmethod
        async def get(key: str) -> Any:
            return _Config._rows.get(key)

    models_config_mod.Config = _Config
    models_pkg.config = models_config_mod

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
    storage_pkg.__path__ = []
    storage_main_mod = cast(Any, _ensure_module("open_webui.storage.main"))

    async def _upload_file_stub(*args, **kwargs):
        """Stub for Open WebUI's upload_file handler."""
        return None

    storage_main_mod.upload_file = _upload_file_stub
    storage_pkg.main = storage_main_mod
    open_webui.storage = storage_pkg

    utils_pkg = cast(Any, _ensure_module("open_webui.utils"))
    utils_pkg.__path__ = []

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

        source_tags = []
        global_idx = 1
        for src in sources:
            name = src.get("name", src.get("source", {}).get("name", f"Source {global_idx}"))
            raw_content = src.get("content", src.get("document", [""]))
            metadata_list = src.get("metadata", [])

            if isinstance(raw_content, list):
                for doc_idx, content in enumerate(raw_content):
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

        context_block = (
            "Use the following sources to answer. Cite using [id] format (e.g., [1], [2]).\n\n"
            + "\n".join(source_tags)
            + "\n\n"
        )

        modified = []
        user_found = False
        for msg in reversed(messages):
            if not user_found and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    msg = {**msg, "content": context_block + content}
                elif isinstance(content, list) and content:
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
                for item in result_data[:5]:
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
        """Open WebUI 0.11.4 `utils/middleware.py::process_tool_result` (`:1052-1265`), tail only.

        Transcribed from the installed function, not from its shape: the model-visible text is a
        dict or list as pretty JSON, a list folded under `{"results": ...}`, a tuple unpacked to
        its first element, and a bare image data URI moved into the files slot with a summary
        text. The rendered text is what `_is_tool_result_error` reads, so a stand-in that
        returns `str(tool_result)` classifies every tool failure as a success.

        Three branches are deliberately left out, and the divergence they would create is
        pinned by `tests/test_a_stand_in_for_an_open_webui_function_is_that_function.py`:

        * `HTMLResponse` and external-tool header handling (`:1064-1165`). The installed
          function decodes an inline body into embeds and answers with a `ui_component` result
          dict. For a **bare** `HTMLResponse` this returns `(str(tool_result), [], [])`, which is
          what arm D pins. The installed function also accepts the 2-tuple
          `(HTMLResponse, result_context)` that `:1064-1067` unpacks; the tail's tuple branch is
          **not** transcribed, so the text is `str()` of the whole tuple and the embeds stay
          empty. Deliberate: transcribing the branch would need the `HTMLResponse` import here.
        * the MCP list branch (`:1176-1245`), which needs `get_file_url_from_base64` and the
          Open WebUI storage behind it.
        * `extract_base64_images` (`:1013-1026`), which walks dicts and lists and would drag
          that same storage import into every tool test. The flat `data:image/` branch the
          pipe's picture path needs is two lines with no Open WebUI imports.
        """
        tool_result_embeds = []
        tool_result_files = []
        if direct_tool and isinstance(tool_result, list) and len(tool_result) == 2:
            tool_result = tool_result[0]
        if isinstance(tool_result, str) and tool_result.startswith("data:image/"):
            tool_result_files.append({"type": "image", "url": tool_result})
            tool_result = f"{tool_function_name}: Image file read successfully."
        if isinstance(tool_result, list):
            tool_result = {"results": tool_result}
        if isinstance(tool_result, dict) or isinstance(tool_result, list):
            tool_result = json.dumps(tool_result, indent=2, ensure_ascii=False)
        if tool_result is not None and not isinstance(tool_result, str):
            if isinstance(tool_result, tuple):
                head = tool_result[0] if len(tool_result) > 0 else None
                if isinstance(head, (str, int, float, bool, type(None), dict, list)):
                    tool_result = json.dumps(head, indent=2, ensure_ascii=False) if len(tool_result) > 0 else ""
                else:
                    tool_result = str(tool_result)
            else:
                tool_result = str(tool_result)
        return tool_result, tool_result_files, tool_result_embeds

    middleware_mod.apply_source_context_to_messages = _apply_source_context_to_messages
    middleware_mod.get_citation_source_from_tool_result = _get_citation_source_from_tool_result
    def _build_terminal_file_tool_result(tool_function_name='', tool_function_params=None, tool_result=None,
                                         tool=None, metadata=None):
        return None

    async def _terminal_event_handler(tool_function_name='', tool_function_params=None, tool_result=None,
                                      event_emitter=None):
        return None

    def _is_tool_result_error(value: Any) -> bool:
        """Open WebUI 0.11.4 `utils/middleware.py::_is_tool_result_error`, verbatim but for `json` in place of
        its `JSONCodec` wrapper."""
        if isinstance(value, str):
            text = value.strip().lower()
            if (
                text.startswith('error:')
                or text.startswith('exception:')
                or text.startswith('traceback')
                or text.startswith('http error!')
            ):
                return True

        parsed = value
        while isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except (json.JSONDecodeError, TypeError, ValueError):
                break

        if not isinstance(parsed, dict):
            return False

        error = parsed.get('error')
        if isinstance(error, str):
            has_error = bool(error.strip())
        else:
            has_error = isinstance(error, (dict, list)) and bool(error)
        if has_error:
            return True

        status = parsed.get('status')
        if isinstance(status, str) and status.strip().lower() in {'error', 'failed'}:
            return True

        if parsed.get('success') is False or parsed.get('ok') is False:
            message = parsed.get('message')
            return has_error or (
                bool(message.strip()) if isinstance(message, str) else isinstance(message, (dict, list)) and bool(message)
            )

        return False

    middleware_mod._is_tool_result_error = _is_tool_result_error
    middleware_mod.process_tool_result = _process_tool_result
    middleware_mod.build_terminal_file_tool_result = _build_terminal_file_tool_result
    middleware_mod.terminal_event_handler = _terminal_event_handler
    utils_pkg.middleware = middleware_mod

    # Open WebUI 0.11.4 `utils/ask_user.py`, copied verbatim: the pipe asks it how long an
    # ask_user prompt stays open and whether an ask_user call may run in its turn, so a
    # simplified stand-in would test a rule Open WebUI does not have.
    ask_user_mod = cast(Any, _ensure_module("open_webui.utils.ask_user"))

    def _get_ask_user_tool_calls(tool_calls: list[dict]) -> tuple[list[dict], str | None]:
        ask_user_calls = [
            tool_call for tool_call in tool_calls if tool_call.get('function', {}).get('name') == 'ask_user'
        ]
        if not ask_user_calls:
            return [], None
        if len(tool_calls) != 1:
            return (
                ask_user_calls,
                'Error: ask_user must be the only tool call, so it did not run. Call ask_user on its own.',
            )
        if len(ask_user_calls) != 1:
            return ask_user_calls, 'Error: only one ask_user call is allowed per turn.'
        return ask_user_calls, None

    def _normalize_ask_user_request(arguments: dict) -> dict:
        questions = arguments.get('questions')
        if not isinstance(questions, list) or not 1 <= len(questions) <= 3:
            raise ValueError('ask_user requires 1-3 questions.')

        normalized_questions = []
        seen_ids = set()
        allow_other = bool(arguments.get('allow_other', True))
        for index, question in enumerate(questions):
            if not isinstance(question, dict):
                raise ValueError('Each question must be an object.')

            question_id = str(question.get('id') or '').strip()[:64]
            if not question_id:
                raise ValueError('Each question requires a non-empty id.')
            if question_id in seen_ids:
                raise ValueError(f'Duplicate question id: {question_id}')
            seen_ids.add(question_id)

            options = question.get('options')
            if not isinstance(options, list) or not 2 <= len(options) <= 3:
                raise ValueError('Each question requires 2-3 options.')

            normalized_options = []
            for option in options:
                if not isinstance(option, dict):
                    raise ValueError('Each option must be an object.')
                label = str(option.get('label') or '').strip()[:80]
                description = str(option.get('description') or '').strip()[:240]
                if not label or not description:
                    raise ValueError('Each option requires a label and description.')
                normalized_options.append({'label': label, 'description': description})

            question_text = str(question.get('question') or '').strip()[:500]
            if not question_text:
                raise ValueError('Each question requires question text.')

            normalized_questions.append(
                {
                    'id': question_id,
                    'header': str(question.get('header') or '').strip()[:48] or f'Question {index + 1}',
                    'question': question_text,
                    'options': normalized_options,
                    'allow_other': bool(question.get('allow_other', allow_other)),
                }
            )

        timeout_ms = arguments.get('timeout_ms', 120_000)
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or not 60_000 <= timeout_ms <= 240_000:
            timeout_ms = 120_000

        return {
            'questions': normalized_questions,
            'allow_other': allow_other,
            'timeout_ms': timeout_ms,
        }

    ask_user_mod.get_ask_user_tool_calls = _get_ask_user_tool_calls
    ask_user_mod.normalize_ask_user_request = _normalize_ask_user_request

    # Open WebUI 0.11.4 `utils/chat_id.py`, copied verbatim: the pipe asks it whether a chat is saved, because only
    # a saved chat is reloaded from the database with its structured output, so only there does Open WebUI replay
    # a recorded tool round. A stand-in that called every id saved would hide the temporary-chat case.
    chat_id_mod = cast(Any, _ensure_module("open_webui.utils.chat_id"))
    chat_id_mod.TEMPORARY_CHAT_ID_PREFIX = 'temporary:'
    chat_id_mod.LEGACY_TEMPORARY_CHAT_ID_PREFIX = 'local:'
    chat_id_mod.CHANNEL_CHAT_ID_PREFIX = 'channel:'
    chat_id_mod.TEMPORARY_CHAT_ID_PREFIXES = (
        chat_id_mod.TEMPORARY_CHAT_ID_PREFIX,
        chat_id_mod.LEGACY_TEMPORARY_CHAT_ID_PREFIX,
    )
    chat_id_mod.NON_SAVED_CHAT_ID_PREFIXES = (*chat_id_mod.TEMPORARY_CHAT_ID_PREFIXES, chat_id_mod.CHANNEL_CHAT_ID_PREFIX)

    def _is_saved_chat_id(chat_id: str | None) -> bool:
        return bool(chat_id) and not chat_id.startswith(chat_id_mod.NON_SAVED_CHAT_ID_PREFIXES)

    def _is_temporary_chat_id(chat_id: str | None) -> bool:
        return bool(chat_id) and chat_id.startswith(chat_id_mod.TEMPORARY_CHAT_ID_PREFIXES)

    chat_id_mod.is_saved_chat_id = _is_saved_chat_id
    chat_id_mod.is_temporary_chat_id = _is_temporary_chat_id
    utils_pkg.ask_user = ask_user_mod

    # Open WebUI 0.11.4 `utils/tools.py`, copied verbatim: the pipe re-binds a tool's `__messages__` and `__files__`
    # through Open WebUI's own re-binder, so a stand-in would test a re-bind Open WebUI does not do.
    tools_mod = cast(Any, _ensure_module("open_webui.utils.tools"))

    async def _get_async_tool_function_and_apply_extra_params(
        function: Callable, extra_params: dict, function_introspection=None
    ) -> Callable[..., Awaitable]:
        if function_introspection is None:
            sig = inspect.signature(function)
            try:
                type_hints = get_type_hints(function)
            except Exception:
                type_hints = {}
        else:
            sig, type_hints = function_introspection

        def coerce_kwargs(kwargs):
            for name, value in kwargs.items():
                if name not in sig.parameters or value is None:
                    continue

                annotation = type_hints.get(name, sig.parameters[name].annotation)
                args = set(get_args(annotation))
                if isinstance(value, str) and (annotation is int or args == {int, type(None)}):
                    kwargs[name] = int(value)
                elif (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and (annotation is str or args == {str, type(None)})
                ):
                    kwargs[name] = str(value)
            return kwargs

        extra_params = {k: v for k, v in extra_params.items() if k in sig.parameters}
        partial_func = partial(function, **extra_params)

        parameters = []
        for name, parameter in sig.parameters.items():
            if name in extra_params:
                continue
            parameters.append(parameter)

        new_sig = inspect.Signature(parameters=parameters, return_annotation=sig.return_annotation)

        if inspect.iscoroutinefunction(function):

            async def new_function(*args, **kwargs):
                return await partial_func(*args, **coerce_kwargs(kwargs))

        else:

            async def new_function(*args, **kwargs):
                return partial_func(*args, **coerce_kwargs(kwargs))

        update_wrapper(new_function, function)
        new_function.__signature__ = new_sig  # type: ignore[attr-defined]

        new_function.__function__ = function  # type: ignore
        new_function.__extra_params__ = extra_params  # type: ignore

        return new_function

    async def _get_updated_tool_function(function: Callable, extra_params: dict):
        # Get the original function and merge updated params
        __function__ = getattr(function, '__function__', None)
        __extra_params__ = getattr(function, '__extra_params__', None)

        if __function__ is not None and __extra_params__ is not None:
            return await _get_async_tool_function_and_apply_extra_params(
                __function__,
                {**__extra_params__, **extra_params},
            )

        return function

    tools_mod.get_async_tool_function_and_apply_extra_params = _get_async_tool_function_and_apply_extra_params
    tools_mod.get_updated_tool_function = _get_updated_tool_function
    utils_pkg.tools = tools_mod

    access_control_pkg = cast(Any, _ensure_module("open_webui.utils.access_control"))
    access_control_pkg.__path__ = []
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

    # No FILE_MAX_SIZE: no Open WebUI has ever bound that name at module level, and
    # inventing it here is the only reason the pipe's dead branch stayed alive.
    config_mod.RAG_FILE_MAX_SIZE = _ConfigValue(None)
    config_mod.BYPASS_EMBEDDING_AND_RETRIEVAL = _ConfigValue(False)
    open_webui.config = config_mod

    constants_mod = cast(Any, _ensure_module("open_webui.constants"))

    class _Tasks(str, Enum):
        def __str__(self) -> str:
            return super().__str__()

        TITLE_GENERATION = "title_generation"
        FOLLOW_UP_GENERATION = "follow_up_generation"
        TAGS_GENERATION = "tags_generation"
        EMOJI_GENERATION = "emoji_generation"
        QUERY_GENERATION = "query_generation"
        IMAGE_PROMPT_GENERATION = "image_prompt_generation"
        AUTOCOMPLETE_GENERATION = "autocomplete_generation"
        FUNCTION_CALLING = "function_calling"
        MOA_RESPONSE_GENERATION = "moa_response_generation"

    # Mirror of backend/open_webui/constants.py:135-148, without its DEFAULT member (a
    # callable, not a task kind). Re-check it when the host moves.
    constants_mod.TASKS = _Tasks
    open_webui.constants = constants_mod


def _install_pydantic_core_stub() -> None:
    import importlib.util

    if importlib.util.find_spec("pydantic_core") is not None:
        return

    core_pkg = cast(Any, _ensure_module("pydantic_core"))
    core_schema_mod = cast(Any, _ensure_module("pydantic_core.core_schema"))

    def _builder(*args, **kwargs):
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
    sa_pkg.Engine = _Engine


def _install_tenacity_stub() -> None:
    import importlib.util

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


_ensure_pydantic_backports()
_install_pydantic_core_stub()
_install_open_webui_stubs()
_install_sqlalchemy_stub()
_install_tenacity_stub()
