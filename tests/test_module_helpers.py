import asyncio
import copy
import datetime
import json
import sys
import types
import io
import logging
from typing import Any, Dict, TYPE_CHECKING, cast

import httpx
import pytest

import open_webui_openrouter_pipe as ow
from open_webui_openrouter_pipe.core import config as ow_config

if TYPE_CHECKING:
    from aiohttp import ClientResponse


def _install_logger(monkeypatch):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("ow.test.logger")
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    # Patch at config and registry level (all import LOGGER from config)
    monkeypatch.setattr(ow, "LOGGER", logger, raising=False)
    monkeypatch.setattr(ow_config, "LOGGER", logger, raising=False)
    # Also patch registry since test_debug_print_error_response uses it
    try:
        from open_webui_openrouter_pipe.models import registry as ow_registry
        monkeypatch.setattr(ow_registry, "LOGGER", logger, raising=False)
    except:
        pass
    return stream


def test_detect_runtime_pipe_id_from_module_prefix(monkeypatch):
    monkeypatch.setitem(ow_config.__dict__, "__name__", "function_demo.plugin")
    assert ow._detect_runtime_pipe_id("fallback") == "demo.plugin"


def test_detect_runtime_pipe_id_default(monkeypatch):
    monkeypatch.setitem(ow.__dict__, "__name__", "open_webui_openrouter_pipe.pipe")
    assert ow._detect_runtime_pipe_id("fallback") == "fallback"


def test_render_error_template_handles_conditionals():
    template = """{{#if show}}Line {value}\n{{/if}}{{#if skip}}{missing}{{/if}}"""
    rendered = ow._render_error_template(template, {"show": True, "value": "X", "skip": False, "missing": ""})
    assert rendered == "Line X"


def test_pretty_json_and_template_value_present():
    data = {"a": 1}
    text = ow._pretty_json(data)
    assert "\n" in text and "\"a\"" in text
    assert ow._pretty_json(" hi ") == "hi"
    assert ow._pretty_json(b"bytes") == "bytes"
    assert ow._template_value_present(0) is True
    assert ow._template_value_present("") is False


def _make_error(**overrides: Any) -> ow.OpenRouterAPIError:
    base: Dict[str, Any] = {
        "status": 400,
        "reason": "Bad",
        "provider": "Provider",
        "metadata": {"retry_after_seconds": 2},
        "upstream_message": "upstream",
        "requested_model": "demo",
    }
    base.update(overrides)
    return ow.OpenRouterAPIError(**base)


def test_build_error_template_values_includes_context():
    error = _make_error(openrouter_message="fail", metadata={"retry_after_seconds": 3})
    values = ow._build_error_template_values(
        error,
        heading="Heading",
        diagnostics=["diag"],
        metrics={"context_limit": 8192, "max_output_tokens": 256},
        model_identifier="provider.demo",
        normalized_model_id="demo",
        api_model_id="demo-api",
        context={"error_id": "err-1", "timestamp": "now"},
    )
    assert values["heading"] == "Heading"
    assert values["retry_after_seconds"] == 3
    assert "diag" in values["diagnostics"]


def test_get_open_webui_config_module(monkeypatch):
    sentinel = object()
    # _OPEN_WEBUI_CONFIG_MODULE moved to core.utils, patch it there
    from open_webui_openrouter_pipe.core import utils as core_utils
    monkeypatch.setattr(core_utils, "_OPEN_WEBUI_CONFIG_MODULE", sentinel, raising=False)
    assert ow._get_open_webui_config_module() is sentinel


def test_unwrap_and_coerce_helpers():
    class Wrapper:
        value = 5

    assert ow._unwrap_config_value(Wrapper()) == 5
    assert ow._coerce_positive_int("10") == 10
    assert ow._coerce_positive_int("-1") is None
    assert ow._coerce_bool("true") is True
    assert ow._coerce_bool("off") is False


def test_retry_after_seconds_parses_date():
    dt = (datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=5)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    seconds = ow._retry_after_seconds(dt)
    assert seconds is not None and seconds >= 4
    assert ow._retry_after_seconds("invalid") is None


def test_classify_retryable_http_error():
    response = httpx.Response(502, request=httpx.Request("GET", "http://x"))
    exc = httpx.HTTPStatusError("boom", request=response.request, response=response)
    retryable, retry_after = ow._classify_retryable_http_error(exc)
    assert retryable is True
    assert retry_after is None


def test_read_rag_file_constraints(monkeypatch):
    cfg = types.SimpleNamespace(
        BYPASS_EMBEDDING_AND_RETRIEVAL=types.SimpleNamespace(value=False),
        RAG_FILE_MAX_SIZE=types.SimpleNamespace(value=70),
    )
    # Patch in core.errors where _read_rag_file_constraints imports it from core.utils
    from open_webui_openrouter_pipe.core import errors as ow_errors
    monkeypatch.setattr(ow_errors, "_get_open_webui_config_module", lambda: cfg)
    rag_enabled, limit = ow._read_rag_file_constraints()
    assert rag_enabled is True
    assert limit == 70


def test_sanitize_model_id():
    assert ow.sanitize_model_id("author/model/v1") == "author.model.v1"
    assert ow.sanitize_model_id("simple") == "simple"


def test_debug_print_request_redacts(monkeypatch):
    stream = _install_logger(monkeypatch)
    headers = {"Authorization": "abcdefghijk"}
    ow._debug_print_request(headers, {"a": 1})
    assert "abcdefghij..." in stream.getvalue()


@pytest.mark.asyncio
async def test_debug_print_error_response(monkeypatch):
    stream = _install_logger(monkeypatch)
    class FakeResponse:
        status = 400
        reason = "Bad"
        url = "http://api"

        async def text(self):
            return "body"

    resp = cast("ClientResponse", FakeResponse())
    body = await ow._debug_print_error_response(resp)
    assert body == "body"
    assert "OpenRouter error response" in stream.getvalue()


def test_safe_json_and_normalizers():
    assert ow._safe_json_loads('{"a":1}') == {"a": 1}
    assert ow._safe_json_loads("invalid") is None
    assert ow._normalize_optional_str("  hi ") == "hi"
    assert ow._normalize_optional_str(0) == "0"
    assert ow._normalize_string_list([" a ", None, 5]) == ["a", "5"]


def test_extract_openrouter_error_details_and_builder():
    payload = json.dumps({
        "error": {
            "message": "oops",
            "code": 400,
            "metadata": {"raw": json.dumps({"provider": "demo"}), "rate_limit_type": "burst"},
        }
    })
    details = ow._extract_openrouter_error_details(payload)
    assert details["provider_raw"] == {"provider": "demo"}
    error = ow._build_openrouter_api_error(400, "Bad", payload, requested_model="demo")
    assert isinstance(error, ow.OpenRouterAPIError)
    assert error.requested_model == "demo"


def test_resolve_and_format_openrouter_error_markdown(monkeypatch):
    ow.ModelFamily.set_dynamic_specs({
        "demo": {
            "context_length": 123,
            "max_completion_tokens": 10,
            "full_model": {"name": "Demo"},
        }
    })
    error = _make_error()
    display, diagnostics, metrics = ow._resolve_error_model_context(error, normalized_model_id="demo", api_model_id="demo")
    assert metrics["context_limit"] == 123
    assert display == "Demo"
    markdown = ow._format_openrouter_error_markdown(error, normalized_model_id="demo", api_model_id="demo", template="{heading}")
    assert "OpenRouter" in markdown or "Provider" in markdown
    ow.ModelFamily.set_dynamic_specs(None)


def test_filter_openrouter_request_drops_invalid_keys():
    payload = {
        "model": "demo",
        "input": "hi",
        "max_output_tokens": None,
        "extra": 1,
        "reasoning": {"effort": "high", "other": True},
    }
    filtered = ow._filter_openrouter_request(payload)
    assert "extra" not in filtered
    assert filtered["reasoning"] == {"effort": "high"}


def test_internal_file_helpers():
    uid = ow._extract_internal_file_id("https://host/files/ABC/")
    assert uid == "ABC"
    assert ow._is_internal_file_url("https://x/api/v1/files/123") is True


@pytest.mark.asyncio
async def test_wrap_event_emitter_controls_events(pipe_instance_async):
    calls = []

    async def emitter(event):
        calls.append(event)

    wrapped = ow._wrap_event_emitter(emitter, suppress_chat_messages=True)
    assert wrapped is not None
    await wrapped({"type": "chat:message"})
    await wrapped({"type": "status"})
    assert calls == [{"type": "status"}]


def test_merge_usage_stats_and_wrap_code_block():
    total = {"a": 1, "nested": {"x": 1}}
    merged = ow.merge_usage_stats(total, {"a": 2, "nested": {"x": 1, "y": 2}})
    assert merged["a"] == 3
    assert merged["nested"]["y"] == 2
    block = ow.wrap_code_block("print('x')", "python")
    assert block.startswith("```python")


def test_normalize_persisted_item_variants(monkeypatch):
    fn_call = ow._normalize_persisted_item({"type": "function_call", "name": "tool", "arguments": {"a": 1}})
    assert fn_call is not None
    assert fn_call["name"] == "tool"
    fn_output = ow._normalize_persisted_item({"type": "function_call_output", "call_id": "123", "output": 5})
    assert fn_output is not None
    assert fn_output["output"] == "5"
    reasoning = ow._normalize_persisted_item({"type": "reasoning", "content": "text"})
    assert reasoning is not None
    assert reasoning["content"]


def test_classify_function_call_artifacts():
    artifacts = {
        "1": {"type": "function_call", "call_id": "call"},
        "2": {"type": "function_call_output", "call_id": "call"},
    }
    valid, orphan_calls, orphan_outputs = ow._classify_function_call_artifacts(artifacts)
    assert "call" in valid
    assert not orphan_calls and not orphan_outputs


def test_crockford_and_markers():
    # ULID functions moved to storage.persistence
    from open_webui_openrouter_pipe.storage.persistence import _encode_crockford
    encoded = _encode_crockford(31, 2)
    assert encoded == "0Z"
    item_id = ow.generate_item_id()
    assert len(item_id) == ow.ULID_LENGTH
    marker = ow._serialize_marker(item_id)
    assert ow._extract_marker_ulid(marker) == item_id
    text = f"{marker}\nfoo"
    assert ow.contains_marker(text)
    spans = ow._iter_marker_spans(text)
    assert spans[0]["marker"] == item_id
    segments = ow.split_text_by_markers(text)
    marker_segments = [seg for seg in segments if seg.get("type") == "marker"]
    assert marker_segments and marker_segments[0]["marker"] == item_id


def test_sanitize_table_fragment():
    assert ow._sanitize_table_fragment("Model-Name!@#") == "model_name"


def test_build_tools_and_dedupe():
    """Test tool building and deduplication logic.

    Real infrastructure exercised:
    - Real ModelFamily.supports checking from registry
    - Real tool building and merging
    - Real deduplication logic
    """
    # Set up model with function_calling support
    ow.ModelFamily.set_dynamic_specs({
        "demo": {
            "features": ["function_calling"],
            "context_length": 8192,
            "full_model": {"name": "Demo"},
        }
    })

    try:
        body = ow.ResponsesBody(model="demo", input="hi")
        valves = ow.Pipe.Valves()
        registry = {
            "tool": {"spec": {"name": "search", "parameters": {"type": "object", "properties": {}}}}
        }
        tools = ow.build_tools(body, valves, __tools__=registry, extra_tools=[{"type": "function", "name": "search"}])
        assert len(tools) == 1
    finally:
        # Clean up
        ow.ModelFamily.set_dynamic_specs(None)


def test_strictify_schema_helpers():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "number"},
        },
    }
    strict = ow._strictify_schema(schema)
    assert strict["required"] == ["a", "b"]
    assert strict["properties"]["a"]["type"] == ["string", "null"]
    deduped = ow._dedupe_tools([
        {"type": "function", "name": "a", "data": 1},
        {"type": "function", "name": "a", "data": 2},
    ])
    assert deduped == [{"type": "function", "name": "a", "data": 2}]


def test_decode_payload_bytes_rejects_headerless_ciphertext(pipe_instance):
    pipe = pipe_instance
    legacy_bytes = b'{"type":"reasoning"}'
    with pytest.raises(ValueError, match="Invalid artifact payload flag"):
        pipe._decode_payload_bytes(legacy_bytes)


def _build_encryption_ready_pipe(pipe: ow.Pipe) -> ow.Pipe:
    pipe._encryption_key = "a" * 32  # type: ignore[attr-defined]
    pipe._encrypt_all = True  # type: ignore[attr-defined]
    pipe._fernet = None  # type: ignore[attr-defined]
    return pipe


def test_prepare_rows_for_storage_encrypts_payloads(pipe_instance):
    pipe = _build_encryption_ready_pipe(pipe_instance)
    rows = [
        {
            "chat_id": "chat",
            "message_id": "msg",
            "item_type": "reasoning",
            "payload": {"type": "reasoning", "content": "secret"},
        }
    ]
    pipe._prepare_rows_for_storage(rows)
    stored = rows[0]
    payload = stored["payload"]
    assert stored["is_encrypted"] is True
    assert isinstance(payload, dict)
    assert payload.get("enc_v") == ow._ENCRYPTED_PAYLOAD_VERSION
    decrypted = pipe._decrypt_payload(payload["ciphertext"])
    assert decrypted["content"] == "secret"


def test_prepare_rows_for_storage_idempotent(pipe_instance):
    pipe = _build_encryption_ready_pipe(pipe_instance)
    rows = [
        {
            "chat_id": "chat",
            "message_id": "msg",
            "item_type": "reasoning",
            "payload": {"type": "reasoning", "content": "secret"},
        }
    ]
    pipe._prepare_rows_for_storage(rows)
    first_payload = copy.deepcopy(rows[0]["payload"])
    pipe._prepare_rows_for_storage(rows)
    assert rows[0]["payload"] == first_payload
    assert rows[0]["is_encrypted"] is True


@pytest.mark.asyncio
async def test_redis_fetch_rows_decrypts_cached_payloads(pipe_instance_async):
    pipe = _build_encryption_ready_pipe(pipe_instance_async)
    row = {
        "id": "01TEST",
        "chat_id": "chat",
        "message_id": "msg",
        "item_type": "reasoning",
        "payload": {"type": "reasoning", "content": "secret"},
    }
    pipe._prepare_rows_for_storage([row])
    cached_json = json.dumps(row, ensure_ascii=False)

    class FakeRedis:
        async def mget(self, keys):
            return [cached_json]

    pipe._redis_client = FakeRedis()  # type: ignore[attr-defined]
    pipe._redis_enabled = True  # type: ignore[attr-defined]
    fetched = await pipe._redis_fetch_rows("chat", ["01TEST"])
    assert fetched["01TEST"]["content"] == "secret"


@pytest.mark.asyncio
async def test_flush_redis_queue_warns_when_lock_release_returns_zero(caplog, pipe_instance_async):
    pipe = pipe_instance_async
    pipe._redis_enabled = True  # type: ignore[attr-defined]

    class FakeRedis:
        async def set(self, key, value, *, nx=False, ex=None):
            return True

        async def lpop(self, key):
            return None

        async def eval(self, script, numkeys, key, token):
            return 0

    pipe._redis_client = FakeRedis()  # type: ignore[attr-defined]

    caplog.set_level(logging.WARNING)
    await pipe._flush_redis_queue()
    assert any(
        "Redis flush lock was not released" in record.getMessage()
        for record in caplog.records
    )
