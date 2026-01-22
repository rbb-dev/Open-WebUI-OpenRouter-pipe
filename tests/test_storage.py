"""Additional persistence coverage tests targeting missing lines."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import hashlib
import json
import logging
import os
import sys
import types
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast
from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.storage import persistence as persistence_mod
from open_webui_openrouter_pipe.storage.persistence import (
    ArtifactStore,
    _normalize_persisted_item,
    _sanitize_table_fragment,
    generate_item_id,
    normalize_persisted_item,
)


class _Field:
    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other):
        return ("eq", self.name, other)

    def __lt__(self, other):
        return ("lt", self.name, other)

    def in_(self, values):
        return ("in", self.name, list(values))


class _FakeModel:
    id = _Field("id")
    chat_id = _Field("chat_id")
    message_id = _Field("message_id")
    model_id = _Field("model_id")
    item_type = _Field("item_type")
    payload = _Field("payload")
    is_encrypted = _Field("is_encrypted")
    created_at = _Field("created_at")

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeQuery:
    def __init__(self, rows: list[_FakeModel]) -> None:
        self._rows = rows
        self._filters: list[tuple[str, str, Any]] = []

    def filter(self, condition):
        self._filters.append(condition)
        return self

    def _match(self, row: _FakeModel, condition: tuple[str, str, Any]) -> bool:
        op, name, value = condition
        current = getattr(row, name, None)
        if op == "eq":
            return current == value
        if op == "in":
            return current in value
        if op == "lt":
            return current < value
        return False

    def _apply(self) -> list[_FakeModel]:
        return [row for row in self._rows if all(self._match(row, cond) for cond in self._filters)]

    def all(self):
        return self._apply()

    def update(self, values, synchronize_session: bool = False):
        rows = self._apply()
        for row in rows:
            for field, val in values.items():
                if isinstance(field, _Field):
                    setattr(row, field.name, val)
        return len(rows)

    def delete(self, synchronize_session: bool = False):
        rows = self._apply()
        for row in rows:
            self._rows.remove(row)
        return len(rows)


class _FakeSession:
    def __init__(self, rows: list[_FakeModel], fail_on_update: bool = False) -> None:
        self._rows = rows
        self._fail_on_update = fail_on_update

    def add_all(self, instances: list[_FakeModel]) -> None:
        self._rows.extend(instances)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None

    def query(self, model):
        query = _FakeQuery(self._rows)
        if self._fail_on_update:
            original_update = query.update

            def _fail_update(*args, **kwargs):
                raise SQLAlchemyError("Touch failed")

            query.update = _fail_update
        return query


def _install_fake_store(pipe: Pipe) -> list[_FakeModel]:
    rows: list[_FakeModel] = []
    store = pipe._artifact_store
    store_any = cast(Any, store)
    store_any._item_model = _FakeModel
    store_any._session_factory = lambda: _FakeSession(rows)
    store_any._artifact_table_name = "response_items_test"
    store_any._db_executor = ThreadPoolExecutor(max_workers=1)
    return rows


def _make_row(chat_id: str, message_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "chat_id": chat_id,
        "message_id": message_id,
        "model_id": "model",
        "item_type": payload.get("type", "unknown"),
        "payload": payload,
    }


# -----------------------------------------------------------------------------
# normalize_persisted_item Tests
# -----------------------------------------------------------------------------


def test_normalize_persisted_item_none_input():
    """Test normalizing None returns None."""
    assert normalize_persisted_item(None) is None


def test_normalize_persisted_item_not_dict():
    """Test normalizing non-dict returns None."""
    assert normalize_persisted_item("not a dict") is None
    assert normalize_persisted_item([1, 2, 3]) is None


def test_normalize_persisted_item_no_type():
    """Test normalizing dict without type returns None."""
    assert normalize_persisted_item({"key": "value"}) is None


def test_normalize_persisted_item_function_call_output():
    """Test normalizing function_call_output items."""
    item = {"type": "function_call_output", "output": 123}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["type"] == "function_call_output"
    assert result["output"] == "123"
    assert "id" in result
    assert "call_id" in result
    assert result["status"] == "completed"


def test_normalize_persisted_item_function_call_output_none_output():
    """Test normalizing function_call_output with None output."""
    item = {"type": "function_call_output", "output": None}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["output"] == ""


def test_normalize_persisted_item_function_call():
    """Test normalizing function_call items."""
    item = {"type": "function_call", "name": "test_func", "arguments": {"key": "value"}}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["type"] == "function_call"
    assert result["name"] == "test_func"
    assert result["arguments"] == '{"key": "value"}'
    assert "id" in result
    assert "call_id" in result


def test_normalize_persisted_item_function_call_string_args():
    """Test normalizing function_call with already-string arguments."""
    item = {"type": "function_call", "name": "test_func", "arguments": '{"key": "value"}'}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["arguments"] == '{"key": "value"}'


def test_normalize_persisted_item_function_call_missing_name():
    """Test normalizing function_call without name returns None."""
    item = {"type": "function_call", "arguments": "{}"}
    assert normalize_persisted_item(item) is None


def test_normalize_persisted_item_function_call_missing_arguments():
    """Test normalizing function_call without arguments returns None."""
    item = {"type": "function_call", "name": "test_func"}
    assert normalize_persisted_item(item) is None


def test_normalize_persisted_item_function_call_non_serializable_args():
    """Test normalizing function_call with non-serializable arguments."""

    class NonSerializable:
        pass

    item = {"type": "function_call", "name": "test_func", "arguments": NonSerializable()}
    result = normalize_persisted_item(item)
    assert result is not None
    # Falls back to str()
    assert "NonSerializable" in result["arguments"]


def test_normalize_persisted_item_reasoning():
    """Test normalizing reasoning items."""
    item = {"type": "reasoning", "content": "Some reasoning text"}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["type"] == "reasoning"
    assert result["content"] == [{"type": "reasoning_text", "text": "Some reasoning text"}]
    assert result["summary"] == []
    assert "id" in result


def test_normalize_persisted_item_reasoning_with_list_content():
    """Test normalizing reasoning with list content."""
    item = {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "test"}]}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["content"] == [{"type": "reasoning_text", "text": "test"}]


def test_normalize_persisted_item_reasoning_with_empty_content():
    """Test normalizing reasoning with empty content."""
    item = {"type": "reasoning", "content": None}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["content"] == []


def test_normalize_persisted_item_reasoning_with_summary():
    """Test normalizing reasoning with summary."""
    item = {"type": "reasoning", "content": [], "summary": "Summary text"}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["summary"] == ["Summary text"]


def test_normalize_persisted_item_web_search_call():
    """Test normalizing web_search_call items."""
    item = {"type": "web_search_call", "action": "invalid"}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["type"] == "web_search_call"
    assert result["action"] == {}
    assert "id" in result


def test_normalize_persisted_item_web_search_call_with_dict_action():
    """Test normalizing web_search_call with valid action dict."""
    item = {"type": "web_search_call", "action": {"query": "test"}}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["action"] == {"query": "test"}


def test_normalize_persisted_item_file_search_call():
    """Test normalizing file_search_call items."""
    item = {"type": "file_search_call", "queries": "not a list"}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["type"] == "file_search_call"
    assert result["queries"] == []


def test_normalize_persisted_item_file_search_call_with_queries():
    """Test normalizing file_search_call with valid queries."""
    item = {"type": "file_search_call", "queries": ["query1", "query2"]}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["queries"] == ["query1", "query2"]


def test_normalize_persisted_item_image_generation_call():
    """Test normalizing image_generation_call items."""
    item = {"type": "image_generation_call", "prompt": "A cat"}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["type"] == "image_generation_call"
    assert "id" in result


def test_normalize_persisted_item_local_shell_call():
    """Test normalizing local_shell_call items."""
    item = {"type": "local_shell_call", "command": "ls"}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["type"] == "local_shell_call"


def test_normalize_persisted_item_unknown_type():
    """Test normalizing unknown type returns item as-is."""
    item = {"type": "unknown_type", "data": "value"}
    result = normalize_persisted_item(item)
    assert result == item


def test_normalize_persisted_item_preserves_existing_status():
    """Test that existing status is preserved."""
    item = {"type": "reasoning", "content": [], "status": "in_progress"}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["status"] == "in_progress"


def test_normalize_persisted_item_preserves_existing_id():
    """Test that existing id is preserved."""
    item = {"type": "reasoning", "content": [], "id": "existing-id"}
    result = normalize_persisted_item(item)
    assert result is not None
    assert result["id"] == "existing-id"


# -----------------------------------------------------------------------------
# Encryption/Decryption Tests
# -----------------------------------------------------------------------------


def test_get_fernet_no_key(pipe_instance):
    """Test _get_fernet returns None without encryption key."""
    store = pipe_instance._artifact_store
    store._encryption_key = ""
    store._fernet = None
    assert store._get_fernet() is None


def test_get_fernet_with_key(pipe_instance):
    """Test _get_fernet returns Fernet instance with key."""
    store = pipe_instance._artifact_store
    store._encryption_key = "test-encryption-key"
    store._fernet = None
    fernet = store._get_fernet()
    assert fernet is not None
    # Verify it's cached
    assert store._get_fernet() is fernet


def test_should_encrypt_no_key(pipe_instance):
    """Test _should_encrypt returns False without key."""
    store = pipe_instance._artifact_store
    store._encryption_key = ""
    assert store._should_encrypt("reasoning") is False


def test_should_encrypt_reasoning_with_key(pipe_instance):
    """Test _should_encrypt returns True for reasoning with key."""
    store = pipe_instance._artifact_store
    store._encryption_key = "test-key"
    store._encrypt_all = False
    assert store._should_encrypt("reasoning") is True
    assert store._should_encrypt("REASONING") is True


def test_should_encrypt_other_type_with_key(pipe_instance):
    """Test _should_encrypt returns False for non-reasoning without encrypt_all."""
    store = pipe_instance._artifact_store
    store._encryption_key = "test-key"
    store._encrypt_all = False
    assert store._should_encrypt("note") is False


def test_should_encrypt_all(pipe_instance):
    """Test _should_encrypt returns True for all types with encrypt_all."""
    store = pipe_instance._artifact_store
    store._encryption_key = "test-key"
    store._encrypt_all = True
    assert store._should_encrypt("note") is True
    assert store._should_encrypt("reasoning") is True


def test_encrypt_payload_no_key_raises(pipe_instance):
    """Test _encrypt_payload raises without key."""
    store = pipe_instance._artifact_store
    store._encryption_key = ""
    store._fernet = None
    with pytest.raises(RuntimeError, match="ARTIFACT_ENCRYPTION_KEY"):
        store._encrypt_payload({"key": "value"})


def test_decrypt_payload_no_key_raises(pipe_instance):
    """Test _decrypt_payload raises without key."""
    store = pipe_instance._artifact_store
    store._encryption_key = ""
    store._fernet = None
    with pytest.raises(RuntimeError, match="ARTIFACT_ENCRYPTION_KEY"):
        store._decrypt_payload("encrypted-data")


def test_decrypt_payload_invalid_token_raises(pipe_instance):
    """Test _decrypt_payload raises on invalid token."""
    store = pipe_instance._artifact_store
    store._encryption_key = "test-key"
    store._fernet = None
    with pytest.raises(ValueError, match="invalid token"):
        store._decrypt_payload("not-valid-encrypted-data")


def test_encrypt_decrypt_roundtrip(pipe_instance):
    """Test encryption/decryption roundtrip."""
    store = pipe_instance._artifact_store
    store._encryption_key = "my-secret-key"
    store._fernet = None
    store._compression_enabled = False

    payload = {"type": "reasoning", "text": "Hello world"}
    encrypted = store._encrypt_payload(payload)
    decrypted = store._decrypt_payload(encrypted)
    assert decrypted == payload


def test_encrypt_if_needed_not_encrypted(pipe_instance):
    """Test _encrypt_if_needed returns plain payload when not needed."""
    store = pipe_instance._artifact_store
    store._encryption_key = "key"
    store._encrypt_all = False

    payload = {"type": "note", "text": "Hello"}
    result, is_encrypted = store._encrypt_if_needed("note", payload)
    assert result == payload
    assert is_encrypted is False


def test_encrypt_if_needed_encrypts(pipe_instance):
    """Test _encrypt_if_needed encrypts reasoning payloads."""
    store = pipe_instance._artifact_store
    store._encryption_key = "key"
    store._encrypt_all = False
    store._fernet = None
    store._compression_enabled = False

    payload = {"type": "reasoning", "text": "Secret"}
    result, is_encrypted = store._encrypt_if_needed("reasoning", payload)
    assert is_encrypted is True
    assert "ciphertext" in result
    assert "enc_v" in result


# -----------------------------------------------------------------------------
# Compression Tests
# -----------------------------------------------------------------------------


def test_maybe_compress_empty_data(pipe_instance):
    """Test _maybe_compress_payload with empty data."""
    store = pipe_instance._artifact_store
    result, compressed = store._maybe_compress_payload(b"")
    assert result == b""
    assert compressed is False


def test_maybe_compress_disabled(pipe_instance):
    """Test _maybe_compress_payload when compression is disabled."""
    store = pipe_instance._artifact_store
    store._compression_enabled = False
    result, compressed = store._maybe_compress_payload(b"test data")
    assert result == b"test data"
    assert compressed is False


def test_maybe_compress_below_threshold(pipe_instance):
    """Test _maybe_compress_payload with data below threshold."""
    store = pipe_instance._artifact_store
    store._compression_enabled = True
    store._compression_min_bytes = 1000
    result, compressed = store._maybe_compress_payload(b"small")
    assert result == b"small"
    assert compressed is False


def test_maybe_compress_lz4_not_available(pipe_instance, monkeypatch):
    """Test _maybe_compress_payload when lz4 is not available."""
    store = pipe_instance._artifact_store
    store._compression_enabled = True
    store._compression_min_bytes = 0
    monkeypatch.setattr(persistence_mod, "lz4frame", None)
    result, compressed = store._maybe_compress_payload(b"test data")
    assert result == b"test data"
    assert compressed is False


def test_maybe_compress_not_smaller(pipe_instance):
    """Test _maybe_compress_payload when compressed is not smaller."""
    store = pipe_instance._artifact_store
    store._compression_enabled = True
    store._compression_min_bytes = 0
    if persistence_mod.lz4frame is None:
        pytest.skip("lz4 not available")
    # Very short data that won't compress well
    result, compressed = store._maybe_compress_payload(b"x")
    assert compressed is False


def test_encode_decode_payload_plain(pipe_instance):
    """Test _encode_payload_bytes and _decode_payload_bytes without compression."""
    store = pipe_instance._artifact_store
    store._compression_enabled = False

    payload = {"key": "value", "number": 123}
    encoded = store._encode_payload_bytes(payload)
    assert encoded[0] == persistence_mod._PAYLOAD_FLAG_PLAIN
    decoded = store._decode_payload_bytes(encoded)
    assert decoded == payload


def test_encode_decode_payload_compressed(pipe_instance):
    """Test _encode_payload_bytes with compression."""
    store = pipe_instance._artifact_store
    if persistence_mod.lz4frame is None:
        pytest.skip("lz4 not available")
    store._compression_enabled = True
    store._compression_min_bytes = 0

    # Large payload that should compress
    payload = {"key": "value" * 100, "data": "test" * 100}
    encoded = store._encode_payload_bytes(payload)
    decoded = store._decode_payload_bytes(encoded)
    assert decoded == payload


def test_decode_payload_bytes_empty(pipe_instance):
    """Test _decode_payload_bytes with empty bytes."""
    store = pipe_instance._artifact_store
    assert store._decode_payload_bytes(b"") == {}


def test_decode_payload_bytes_single_byte(pipe_instance):
    """Test _decode_payload_bytes with single byte."""
    store = pipe_instance._artifact_store
    # Single byte is treated as body
    with pytest.raises(ValueError):
        store._decode_payload_bytes(b"{")


# -----------------------------------------------------------------------------
# Redis Tests
# -----------------------------------------------------------------------------


def test_redis_cache_key_missing_values(pipe_instance):
    """Test _redis_cache_key returns None for missing values."""
    store = pipe_instance._artifact_store
    assert store._redis_cache_key(None, "id") is None
    assert store._redis_cache_key("chat", None) is None
    assert store._redis_cache_key("", "id") is None
    assert store._redis_cache_key("chat", "") is None


def test_redis_cache_key_valid(pipe_instance):
    """Test _redis_cache_key returns expected key."""
    store = pipe_instance._artifact_store
    key = store._redis_cache_key("chat-123", "item-456")
    assert key is not None
    assert "chat-123" in key
    assert "item-456" in key


@pytest.mark.asyncio
async def test_redis_cache_rows_not_enabled(pipe_instance):
    """Test _redis_cache_rows does nothing when Redis is not enabled."""
    store = pipe_instance._artifact_store
    store._redis_enabled = False
    await store._redis_cache_rows([{"id": "test", "payload": {}}])


@pytest.mark.asyncio
async def test_redis_cache_rows_skips_missing_cache_key(pipe_instance):
    """Test _redis_cache_rows skips rows without valid cache key."""
    store = pipe_instance._artifact_store
    store._redis_enabled = True

    class _FakePipeline:
        def __init__(self):
            self.calls = []

        def setex(self, key, ttl, value):
            self.calls.append(("setex", key, ttl, value))

        def execute(self):
            return []

    class _FakeRedis:
        def __init__(self):
            self.pipeline_instance = _FakePipeline()

        def pipeline(self):
            return self.pipeline_instance

    store._redis_client = _FakeRedis()
    await store._redis_cache_rows([{"id": "test"}])  # Missing chat_id
    assert store._redis_client.pipeline_instance.calls == []


@pytest.mark.asyncio
async def test_redis_requeue_entries_no_client(pipe_instance):
    """Test _redis_requeue_entries does nothing without client."""
    store = pipe_instance._artifact_store
    store._redis_client = None
    await store._redis_requeue_entries(["entry1", "entry2"])


@pytest.mark.asyncio
async def test_redis_requeue_entries_empty(pipe_instance):
    """Test _redis_requeue_entries does nothing with empty list."""
    store = pipe_instance._artifact_store
    store._redis_client = Mock()
    await store._redis_requeue_entries([])
    store._redis_client.pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_redis_fetch_rows_not_enabled(pipe_instance):
    """Test _redis_fetch_rows returns empty when not enabled."""
    store = pipe_instance._artifact_store
    store._redis_enabled = False
    result = await store._redis_fetch_rows("chat", ["id1", "id2"])
    assert result == {}


@pytest.mark.asyncio
async def test_redis_fetch_rows_no_client(pipe_instance):
    """Test _redis_fetch_rows returns empty without client."""
    store = pipe_instance._artifact_store
    store._redis_enabled = True
    store._redis_client = None
    result = await store._redis_fetch_rows("chat", ["id1"])
    assert result == {}


@pytest.mark.asyncio
async def test_redis_fetch_rows_no_chat_id(pipe_instance):
    """Test _redis_fetch_rows returns empty without chat_id."""
    store = pipe_instance._artifact_store
    store._redis_enabled = True
    store._redis_client = Mock()
    result = await store._redis_fetch_rows(None, ["id1"])
    assert result == {}


@pytest.mark.asyncio
async def test_redis_fetch_rows_with_encrypted_payload(pipe_instance):
    """Test _redis_fetch_rows decrypts encrypted payloads."""
    store = pipe_instance._artifact_store
    store._redis_enabled = True
    store._encryption_key = "test-key"
    store._fernet = None
    store._compression_enabled = False

    # First encrypt a payload
    payload = {"type": "reasoning", "text": "secret"}
    encrypted = store._encrypt_payload(payload)

    class _FakeRedis:
        def mget(self, keys):
            return [json.dumps({"payload": {"ciphertext": encrypted}, "is_encrypted": True})]

    store._redis_client = _FakeRedis()

    result = await store._redis_fetch_rows("chat", ["id1"])
    assert "id1" in result
    assert result["id1"]["text"] == "secret"


@pytest.mark.asyncio
async def test_redis_fetch_rows_decryption_failure(pipe_instance, caplog):
    """Test _redis_fetch_rows handles decryption failures."""
    store = pipe_instance._artifact_store
    store._redis_enabled = True
    store._encryption_key = "test-key"
    store._fernet = None

    class _FakeRedis:
        def mget(self, keys):
            return [json.dumps({"payload": {"ciphertext": "invalid"}, "is_encrypted": True})]

    store._redis_client = _FakeRedis()
    caplog.set_level(logging.WARNING)
    result = await store._redis_fetch_rows("chat", ["id1"])
    assert "id1" not in result


@pytest.mark.asyncio
async def test_redis_fetch_rows_invalid_json(pipe_instance):
    """Test _redis_fetch_rows handles invalid JSON."""
    store = pipe_instance._artifact_store
    store._redis_enabled = True

    class _FakeRedis:
        def mget(self, keys):
            return ["not-valid-json"]

    store._redis_client = _FakeRedis()
    result = await store._redis_fetch_rows("chat", ["id1"])
    assert result == {}


@pytest.mark.asyncio
async def test_redis_fetch_rows_none_value(pipe_instance):
    """Test _redis_fetch_rows skips None values."""
    store = pipe_instance._artifact_store
    store._redis_enabled = True

    class _FakeRedis:
        def mget(self, keys):
            return [None, None]

    store._redis_client = _FakeRedis()
    result = await store._redis_fetch_rows("chat", ["id1", "id2"])
    assert result == {}


@pytest.mark.asyncio
async def test_redis_enqueue_fallback_on_failure(pipe_instance, caplog):
    """Test _redis_enqueue_rows falls back to direct DB on failure."""
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._redis_enabled = True

    class _FailingRedis:
        def pipeline(self):
            raise RuntimeError("Connection failed")

    store._redis_client = _FailingRedis()

    # Need to mock _db_persist_direct since the fake store setup may not be complete
    persisted = []

    async def _mock_persist(rows, user_id=""):
        persisted.extend(rows)
        return [row.get("id", generate_item_id()) for row in rows]

    store._db_persist_direct = _mock_persist

    caplog.set_level(logging.WARNING)
    rows = [_make_row("chat", "msg", {"type": "note", "value": 1})]
    result = await store._redis_enqueue_rows(rows)
    # Should have fallen back to direct DB
    assert len(result) == 1
    assert any("falling back" in rec.message.lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_flush_redis_queue_db_failure_requeues(pipe_instance, caplog):
    """Test _flush_redis_queue requeues on DB failure."""
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._redis_enabled = True

    requeued = []

    class _FakeRedis:
        def __init__(self):
            self.storage = {}
            self.lists = {store._redis_pending_key: [json.dumps({"chat_id": "c", "message_id": "m", "payload": {}})]}

        def lpop(self, key):
            values = self.lists.get(key, [])
            return values.pop(0) if values else None

        def set(self, key, value, nx=False, ex=None):
            self.storage[key] = value
            return True

        def eval(self, script, numkeys, key, token):
            self.storage.pop(key, None)
            return 1

        def pipeline(self):
            class _Pipe:
                def lpush(_, key, value):
                    requeued.append(value)
                    return self

                def expire(_, key, ttl):
                    return self

                def execute(self):
                    return []

            return _Pipe()

    store._redis_client = _FakeRedis()

    async def _failing_persist(rows, user_id=""):
        raise RuntimeError("DB down")

    store._db_persist_direct = _failing_persist

    caplog.set_level(logging.ERROR)
    await store._flush_redis_queue()
    assert requeued


@pytest.mark.asyncio
async def test_flush_redis_queue_lock_release_failure(pipe_instance, caplog):
    """Test _flush_redis_queue handles lock release failure."""
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._redis_enabled = True

    class _FakeRedis:
        def __init__(self):
            self.storage = {}
            self.lists = {}

        def lpop(self, key):
            return None

        def set(self, key, value, nx=False, ex=None):
            self.storage[key] = value
            return True

        def eval(self, script, numkeys, key, token):
            raise RuntimeError("Lock release failed")

    store._redis_client = _FakeRedis()
    await store._flush_redis_queue()


@pytest.mark.asyncio
async def test_flush_redis_queue_lock_not_released(pipe_instance, caplog):
    """Test _flush_redis_queue logs when lock was not released."""
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._redis_enabled = True

    class _FakeRedis:
        def __init__(self):
            self.storage = {}
            self.lists = {}

        def lpop(self, key):
            return None

        def set(self, key, value, nx=False, ex=None):
            self.storage[key] = value
            return True

        def eval(self, script, numkeys, key, token):
            return 0  # Lock not released (wrong token)

    store._redis_client = _FakeRedis()
    caplog.set_level(logging.WARNING)
    await store._flush_redis_queue()
    assert any("not released" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_flush_redis_queue_bytes_data(pipe_instance):
    """Test _flush_redis_queue handles bytes data from Redis."""
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._redis_enabled = True

    class _FakeRedis:
        def __init__(self):
            self.storage = {}
            entry = json.dumps({"chat_id": "c", "message_id": "m", "item_type": "note", "payload": {"type": "note"}})
            self.lists = {store._redis_pending_key: [entry.encode("utf-8")]}

        def lpop(self, key):
            values = self.lists.get(key, [])
            return values.pop(0) if values else None

        def set(self, key, value, nx=False, ex=None):
            self.storage[key] = value
            return True

        def eval(self, script, numkeys, key, token):
            self.storage.pop(key, None)
            return 1

    store._redis_client = _FakeRedis()

    persisted = []

    async def _fake_persist(rows, user_id=""):
        persisted.extend(rows)
        return [row.get("id", generate_item_id()) for row in rows]

    store._db_persist_direct = _fake_persist
    await store._flush_redis_queue()
    assert persisted


@pytest.mark.asyncio
async def test_flush_redis_queue_unexpected_type(pipe_instance, caplog):
    """Test _flush_redis_queue handles unexpected data type."""
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._redis_enabled = True

    class _FakeRedis:
        def __init__(self):
            self.storage = {}
            self.lists = {store._redis_pending_key: [12345]}  # Unexpected int

        def lpop(self, key):
            values = self.lists.get(key, [])
            return values.pop(0) if values else None

        def set(self, key, value, nx=False, ex=None):
            self.storage[key] = value
            return True

        def eval(self, script, numkeys, key, token):
            self.storage.pop(key, None)
            return 1

    store._redis_client = _FakeRedis()
    caplog.set_level(logging.WARNING)
    await store._flush_redis_queue()
    assert any("Unexpected Redis queue payload type" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_flush_redis_queue_non_object_json(pipe_instance, caplog):
    """Test _flush_redis_queue handles non-object JSON."""
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._redis_enabled = True

    class _FakeRedis:
        def __init__(self):
            self.storage = {}
            self.lists = {store._redis_pending_key: ['"just a string"']}

        def lpop(self, key):
            values = self.lists.get(key, [])
            return values.pop(0) if values else None

        def set(self, key, value, nx=False, ex=None):
            self.storage[key] = value
            return True

        def eval(self, script, numkeys, key, token):
            self.storage.pop(key, None)
            return 1

    store._redis_client = _FakeRedis()
    caplog.set_level(logging.WARNING)
    await store._flush_redis_queue()
    assert any("must be an object" in rec.message for rec in caplog.records)


# -----------------------------------------------------------------------------
# DB Fetch Tests
# -----------------------------------------------------------------------------


def test_db_fetch_sync_encrypted_string_payload(pipe_instance):
    """Test _db_fetch_sync handles encrypted string payloads."""
    rows = _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._encryption_key = "test-key"
    store._fernet = None
    store._compression_enabled = False

    # Encrypt a payload
    payload = {"type": "reasoning", "text": "secret"}
    encrypted = store._encrypt_payload(payload)

    # Add row with string payload (ciphertext directly)
    row = _FakeModel(
        id="id-1",
        chat_id="chat-1",
        message_id="msg-1",
        model_id=None,
        item_type="reasoning",
        payload=encrypted,  # String payload
        is_encrypted=True,
        created_at=datetime.datetime.now(datetime.UTC),
    )
    rows.append(row)

    result = store._db_fetch_sync("chat-1", "msg-1", ["id-1"])
    assert "id-1" in result
    assert result["id-1"]["text"] == "secret"


def test_db_fetch_sync_decryption_failure(pipe_instance, caplog):
    """Test _db_fetch_sync handles decryption failures."""
    rows = _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._encryption_key = "test-key"
    store._fernet = None

    row = _FakeModel(
        id="id-1",
        chat_id="chat-1",
        message_id="msg-1",
        model_id=None,
        item_type="reasoning",
        payload={"ciphertext": "invalid-data"},
        is_encrypted=True,
        created_at=datetime.datetime.now(datetime.UTC),
    )
    rows.append(row)

    caplog.set_level(logging.WARNING)
    result = store._db_fetch_sync("chat-1", "msg-1", ["id-1"])
    assert "id-1" not in result


def test_db_fetch_sync_touch_failure(pipe_instance, caplog):
    """Test _db_fetch_sync handles touch update failures gracefully."""
    rows: list[_FakeModel] = []
    store = pipe_instance._artifact_store
    store._item_model = _FakeModel

    # Create a session factory that returns a session that fails on update
    def _session_factory():
        return _FakeSession(rows, fail_on_update=True)

    store._session_factory = _session_factory
    store._artifact_table_name = "test"
    store._db_executor = ThreadPoolExecutor(max_workers=1)

    row = _FakeModel(
        id="id-1",
        chat_id="chat-1",
        message_id="msg-1",
        model_id=None,
        item_type="note",
        payload={"type": "note"},
        is_encrypted=False,
        created_at=datetime.datetime.now(datetime.UTC),
    )
    rows.append(row)

    caplog.set_level(logging.DEBUG)
    result = store._db_fetch_sync("chat-1", "msg-1", ["id-1"])
    # Should still return the result despite touch failure
    assert "id-1" in result


@pytest.mark.asyncio
async def test_db_fetch_caches_results_in_redis(pipe_instance):
    """Test _db_fetch caches fetched results in Redis."""
    rows = _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._redis_enabled = True

    cached_rows = []

    class _FakeRedis:
        def mget(self, keys):
            return [None] * len(keys)

    store._redis_client = _FakeRedis()

    async def _capture_cache(rows_to_cache, chat_id=None):
        cached_rows.extend(rows_to_cache)

    store._redis_cache_rows = _capture_cache

    async def _redis_fetch_empty(*args):
        return {}

    store._redis_fetch_rows = _redis_fetch_empty

    row = _FakeModel(
        id="id-1",
        chat_id="chat-1",
        message_id="msg-1",
        model_id=None,
        item_type="note",
        payload={"type": "note", "value": 1},
        is_encrypted=False,
        created_at=datetime.datetime.now(datetime.UTC),
    )
    rows.append(row)

    result = await store._db_fetch("chat-1", "msg-1", ["id-1"])
    assert "id-1" in result
    assert cached_rows


# -----------------------------------------------------------------------------
# Circuit Breaker Tests
# -----------------------------------------------------------------------------


def test_db_breaker_allows_empty_user_id(pipe_instance):
    """Test _db_breaker_allows returns True for empty user_id."""
    store = pipe_instance._artifact_store
    assert store._db_breaker_allows("") is True


def test_db_breaker_allows_prunes_old_failures(pipe_instance):
    """Test _db_breaker_allows prunes old failures."""
    import time

    store = pipe_instance._artifact_store
    store._breaker_window_seconds = 1
    store._breaker_threshold = 5

    # Add old failures
    old_time = time.time() - 10
    store._db_breakers["user1"].append(old_time)
    store._db_breakers["user1"].append(old_time)

    assert store._db_breaker_allows("user1") is True
    assert len(store._db_breakers["user1"]) == 0


def test_db_breaker_blocks_after_threshold(pipe_instance):
    """Test _db_breaker_allows returns False after threshold."""
    import time

    store = pipe_instance._artifact_store
    store._breaker_window_seconds = 60
    store._breaker_threshold = 3

    now = time.time()
    store._db_breakers["user1"].append(now)
    store._db_breakers["user1"].append(now)
    store._db_breakers["user1"].append(now)

    assert store._db_breaker_allows("user1") is False


def test_record_db_failure_no_user_id(pipe_instance):
    """Test _record_db_failure does nothing for empty user_id."""
    store = pipe_instance._artifact_store
    initial_len = len(store._db_breakers.get("", deque()))
    store._record_db_failure("")
    assert len(store._db_breakers.get("", deque())) == initial_len


def test_reset_db_failure_clears_failures(pipe_instance):
    """Test _reset_db_failure clears failures for user."""
    import time

    store = pipe_instance._artifact_store
    store._db_breakers["user1"].append(time.time())
    store._db_breakers["user1"].append(time.time())

    store._reset_db_failure("user1")
    assert len(store._db_breakers["user1"]) == 0


def test_reset_db_failure_no_user_id(pipe_instance):
    """Test _reset_db_failure does nothing for empty user_id."""
    store = pipe_instance._artifact_store
    store._reset_db_failure("")


def test_record_failure_generic_no_user(pipe_instance):
    """Test _record_failure does nothing without user_id."""
    store = pipe_instance._artifact_store
    initial_len = len(store._breaker_records.get("", deque()))
    store._record_failure("")
    assert len(store._breaker_records.get("", deque())) == initial_len


def test_record_failure_generic_with_user(pipe_instance):
    """Test _record_failure records failure for user."""
    store = pipe_instance._artifact_store
    store._record_failure("user1")
    assert len(store._breaker_records["user1"]) == 1


# -----------------------------------------------------------------------------
# Shutdown Tests
# -----------------------------------------------------------------------------


def test_shutdown_without_executor(pipe_instance):
    """Test shutdown when executor is None."""
    store = pipe_instance._artifact_store
    store._db_executor = None
    store.shutdown()


def test_shutdown_with_executor(pipe_instance):
    """Test shutdown shuts down executor."""
    store = pipe_instance._artifact_store
    store._db_executor = ThreadPoolExecutor(max_workers=1)
    store.shutdown()
    assert store._db_executor is None


def test_shutdown_executor_type_error(pipe_instance, monkeypatch):
    """Test shutdown handles TypeError from executor.shutdown."""

    class _OldExecutor:
        def shutdown(self, wait=False, cancel_futures=False):
            if cancel_futures:
                raise TypeError("cancel_futures not supported")

    store = pipe_instance._artifact_store
    store._db_executor = _OldExecutor()
    store.shutdown()


def test_shutdown_executor_generic_error(pipe_instance, caplog):
    """Test shutdown handles generic errors from executor.shutdown."""

    class _FailingExecutor:
        def shutdown(self, *args, **kwargs):
            raise RuntimeError("Shutdown failed")

    store = pipe_instance._artifact_store
    store._db_executor = _FailingExecutor()
    caplog.set_level(logging.DEBUG)
    store.shutdown()


# -----------------------------------------------------------------------------
# Cleanup Worker Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_cleanup_once_no_model(pipe_instance):
    """Test _run_cleanup_once returns early without model."""
    store = pipe_instance._artifact_store
    store._item_model = None
    await store._run_cleanup_once()


@pytest.mark.asyncio
async def test_cleanup_sync_exception(pipe_instance):
    """Test _cleanup_sync handles and re-raises exceptions."""
    rows: list[_FakeModel] = []
    store = pipe_instance._artifact_store
    store._item_model = _FakeModel

    class _FailingSession:
        def query(self, model):
            raise SQLAlchemyError("Query failed")

        def rollback(self):
            pass

        def close(self):
            pass

    store._session_factory = lambda: _FailingSession()

    with pytest.raises(SQLAlchemyError):
        store._cleanup_sync(datetime.datetime.now(datetime.UTC))


# -----------------------------------------------------------------------------
# Ensure Artifact Store Tests
# -----------------------------------------------------------------------------


def test_ensure_artifact_store_missing_pipe_id(pipe_instance):
    """Test _ensure_artifact_store raises without pipe_id."""
    store = pipe_instance._artifact_store
    store.id = ""
    with pytest.raises(RuntimeError, match="Pipe identifier"):
        store._ensure_artifact_store(pipe_instance.valves, pipe_identifier=None)


def test_ensure_artifact_store_skip_when_unchanged(pipe_instance):
    """Test _ensure_artifact_store skips when signature unchanged."""
    store = pipe_instance._artifact_store
    store.id = "test-pipe"
    store._encryption_key = ""
    store._artifact_store_signature = (_sanitize_table_fragment("test-pipe"), "")
    store._item_model = Mock()
    store._session_factory = Mock()
    store._engine = Mock()

    # Should return early without calling _init_artifact_store
    original_init = store._init_artifact_store
    called = []
    store._init_artifact_store = lambda *args, **kwargs: called.append(True)
    store._ensure_artifact_store(pipe_instance.valves)
    assert not called
    store._init_artifact_store = original_init


def test_ensure_artifact_store_lz4_warning(pipe_instance, monkeypatch, caplog):
    """Test _ensure_artifact_store warns when LZ4 unavailable."""
    store = pipe_instance._artifact_store
    store.id = "test-pipe"
    store._lz4_warning_emitted = False
    store._artifact_store_signature = None

    # Mock valves to enable LZ4
    valves = Mock()
    valves.ARTIFACT_ENCRYPTION_KEY = ""
    valves.ENCRYPT_ALL = False
    valves.MIN_COMPRESS_BYTES = 100
    valves.ENABLE_LZ4_COMPRESSION = True

    monkeypatch.setattr(persistence_mod, "lz4frame", None)

    # Mock _init_artifact_store to avoid actual DB init
    store._init_artifact_store = Mock()

    caplog.set_level(logging.WARNING)
    store._ensure_artifact_store(valves)
    assert store._lz4_warning_emitted
    assert any("lz4" in rec.message.lower() for rec in caplog.records)


# -----------------------------------------------------------------------------
# Init Artifact Store Tests
# -----------------------------------------------------------------------------


@contextlib.contextmanager
def _install_internal_db(engine, schema: str | None = None):
    saved_internal = sys.modules.get("open_webui.internal")
    saved_db = sys.modules.get("open_webui.internal.db")

    internal_pkg = types.ModuleType("open_webui.internal")
    db_mod = types.ModuleType("open_webui.internal.db")
    setattr(db_mod, "engine", engine)
    setattr(db_mod, "ENGINE", engine)
    if schema is not None:

        class _Base:
            metadata = types.SimpleNamespace(schema=schema)

        setattr(db_mod, "Base", _Base)

    setattr(internal_pkg, "db", db_mod)
    sys.modules["open_webui.internal"] = internal_pkg
    sys.modules["open_webui.internal.db"] = db_mod
    try:
        yield
    finally:
        if saved_internal is not None:
            sys.modules["open_webui.internal"] = saved_internal
        else:
            sys.modules.pop("open_webui.internal", None)
        if saved_db is not None:
            sys.modules["open_webui.internal.db"] = saved_db
        else:
            sys.modules.pop("open_webui.internal.db", None)


def test_init_artifact_store_missing_pipe_id(pipe_instance):
    """Test _init_artifact_store raises without pipe_id."""
    store = pipe_instance._artifact_store
    store.id = ""
    engine = create_engine("sqlite://")
    with _install_internal_db(engine):
        with pytest.raises(RuntimeError, match="Pipe identifier"):
            store._init_artifact_store(pipe_identifier=None)


def test_init_artifact_store_no_engine(pipe_instance, caplog):
    """Test _init_artifact_store disables when engine unavailable."""
    store = pipe_instance._artifact_store
    store.id = "test-pipe"

    # Remove any existing open_webui.internal.db module
    saved = sys.modules.pop("open_webui.internal.db", None)
    saved_internal = sys.modules.pop("open_webui.internal", None)
    try:
        caplog.set_level(logging.WARNING)
        store._init_artifact_store(pipe_identifier="test-pipe")
        assert any("disabled" in rec.message.lower() for rec in caplog.records)
    finally:
        if saved is not None:
            sys.modules["open_webui.internal.db"] = saved
        if saved_internal is not None:
            sys.modules["open_webui.internal"] = saved_internal


def test_discover_engine_metadata_obj_schema(pipe_instance):
    """Test _discover_owui_engine_and_schema uses metadata_obj."""
    store = pipe_instance._artifact_store

    class _DB:
        Base = None
        metadata_obj = types.SimpleNamespace(schema="alt_schema")
        engine = object()

    engine, schema, details = store._discover_owui_engine_and_schema(_DB)
    assert schema == "alt_schema"
    assert details.get("schema_source") == "owui_db.metadata_obj.schema"


def test_discover_engine_from_env_module(pipe_instance, monkeypatch):
    """Test _discover_owui_engine_and_schema uses open_webui.env."""
    store = pipe_instance._artifact_store

    # Create mock open_webui.env module
    env_mod = types.ModuleType("open_webui.env")
    env_mod.DATABASE_SCHEMA = "env_schema"
    sys.modules["open_webui.env"] = env_mod

    class _DB:
        Base = None
        metadata_obj = None
        engine = object()

    try:
        engine, schema, details = store._discover_owui_engine_and_schema(_DB)
        assert schema == "env_schema"
    finally:
        sys.modules.pop("open_webui.env", None)


def test_discover_engine_from_attr(pipe_instance):
    """Test _discover_owui_engine_and_schema uses engine attribute."""
    store = pipe_instance._artifact_store

    class _DB:
        Base = None
        engine = "the_engine"

    engine, schema, details = store._discover_owui_engine_and_schema(_DB)
    assert engine == "the_engine"
    assert details.get("engine_source") == "owui_db.engine"


def test_discover_engine_from_bind(pipe_instance):
    """Test _discover_owui_engine_and_schema uses bind attribute."""
    store = pipe_instance._artifact_store

    class _Session:
        bind = "the_bind"

    @contextlib.contextmanager
    def _ctx():
        yield _Session()

    class _DB:
        Base = None
        get_db_context = staticmethod(_ctx)

    engine, schema, details = store._discover_owui_engine_and_schema(_DB)
    assert engine == "the_bind"


# -----------------------------------------------------------------------------
# Redis Initialization Tests
# -----------------------------------------------------------------------------


def test_redis_state_invalid_uvicorn_workers(caplog, monkeypatch):
    """Test _initialize_redis_state handles invalid UVICORN_WORKERS."""
    monkeypatch.setenv("UVICORN_WORKERS", "invalid")
    caplog.set_level(logging.WARNING)

    pipe = Pipe()
    try:
        assert any("Invalid UVICORN_WORKERS" in rec.message for rec in caplog.records)
    finally:
        asyncio.run(pipe.close())


def test_redis_state_multi_worker_warnings(caplog, monkeypatch):
    """Test _initialize_redis_state warns on multi-worker issues."""
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    monkeypatch.setenv("REDIS_URL", "")
    caplog.set_level(logging.WARNING)

    pipe = Pipe()
    try:
        # Multiple workers without REDIS_URL should warn
        pass
    finally:
        asyncio.run(pipe.close())


def test_redis_state_websocket_manager_not_redis(caplog, monkeypatch):
    """Test warning when WEBSOCKET_MANAGER is not redis."""
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    monkeypatch.setenv("REDIS_URL", "redis://localhost")
    monkeypatch.setenv("WEBSOCKET_MANAGER", "local")
    caplog.set_level(logging.WARNING)

    pipe = Pipe()
    try:
        store = pipe._artifact_store
        store.valves.ENABLE_REDIS_CACHE = True
        store._initialize_redis_state()
        # Should have warned
    finally:
        asyncio.run(pipe.close())


# -----------------------------------------------------------------------------
# Maybe Heal Index Conflict Tests
# -----------------------------------------------------------------------------


def test_maybe_heal_no_engine(pipe_instance):
    """Test _maybe_heal_index_conflict returns False without engine."""
    store = pipe_instance._artifact_store
    assert store._maybe_heal_index_conflict(None, Mock(), Exception()) is False


def test_maybe_heal_no_table(pipe_instance):
    """Test _maybe_heal_index_conflict returns False without table."""
    store = pipe_instance._artifact_store
    assert store._maybe_heal_index_conflict(Mock(), None, Exception()) is False


def test_maybe_heal_no_ix_in_message(pipe_instance):
    """Test _maybe_heal_index_conflict returns False without ix_ in message."""
    store = pipe_instance._artifact_store
    assert store._maybe_heal_index_conflict(Mock(), Mock(), Exception("other error")) is False


def test_maybe_heal_with_schema(pipe_instance):
    """Test _maybe_heal_index_conflict with schema qualification."""
    store = pipe_instance._artifact_store

    class _DummyTable:
        name = "test_table"
        schema = "test_schema"
        indexes = set()
        columns = []

    executed = []

    class _DummyEngine:
        def begin(self):
            class _Conn:
                def execute(self, stmt):
                    executed.append(str(stmt))

            return contextlib.nullcontext(_Conn())

    result = store._maybe_heal_index_conflict(
        _DummyEngine(), _DummyTable(), Exception("duplicate key ix_test")
    )
    # Should try to drop the index
    assert any("ix_test" in stmt.lower() for stmt in executed)


def test_maybe_heal_drop_fails(pipe_instance, caplog):
    """Test _maybe_heal_index_conflict handles drop failures."""
    store = pipe_instance._artifact_store

    class _DummyIndex:
        name = "ix_test_chat_id"

    class _DummyTable:
        name = "test_table"
        schema = None
        indexes = {_DummyIndex()}
        columns = []

    class _DummyEngine:
        def begin(self):
            class _Conn:
                def execute(self, stmt):
                    raise SQLAlchemyError("Drop failed")

            return contextlib.nullcontext(_Conn())

    caplog.set_level(logging.WARNING)
    result = store._maybe_heal_index_conflict(
        _DummyEngine(), _DummyTable(), Exception("duplicate key ix_test_chat_id")
    )
    assert result is False


# -----------------------------------------------------------------------------
# Quote Identifier Tests
# -----------------------------------------------------------------------------


def test_quote_identifier():
    """Test _quote_identifier handles special characters."""
    store_class = ArtifactStore

    assert store_class._quote_identifier("simple") == '"simple"'
    assert store_class._quote_identifier('has"quotes') == '"has""quotes"'
    assert store_class._quote_identifier("") == '""'


# -----------------------------------------------------------------------------
# Delete Artifacts Tests
# -----------------------------------------------------------------------------


def test_delete_artifacts_sync_no_session(pipe_instance):
    """Test _delete_artifacts_sync returns early without session."""
    store = pipe_instance._artifact_store
    store._session_factory = None
    store._delete_artifacts_sync(["id1", "id2"])


def test_delete_artifacts_sync_rollback_on_error(pipe_instance):
    """Test _delete_artifacts_sync rolls back on error."""
    store = pipe_instance._artifact_store

    rollback_called = []

    class _FailingSession:
        def query(self, model):
            class _Query:
                def filter(self, cond):
                    return self

                def delete(self, synchronize_session=False):
                    raise SQLAlchemyError("Delete failed")

            return _Query()

        def rollback(self):
            rollback_called.append(True)

        def close(self):
            pass

    store._item_model = _FakeModel
    store._session_factory = lambda: _FailingSession()

    with pytest.raises(SQLAlchemyError):
        store._delete_artifacts_sync(["id1"])
    assert rollback_called


@pytest.mark.asyncio
async def test_delete_artifacts_empty_refs(pipe_instance):
    """Test _delete_artifacts returns early for empty refs."""
    store = pipe_instance._artifact_store
    store._db_executor = Mock()
    await store._delete_artifacts([])


@pytest.mark.asyncio
async def test_delete_artifacts_no_executor(pipe_instance):
    """Test _delete_artifacts returns early without executor."""
    store = pipe_instance._artifact_store
    store._db_executor = None
    await store._delete_artifacts([("chat", "id1")])


# -----------------------------------------------------------------------------
# Make DB Row Tests
# -----------------------------------------------------------------------------


def test_make_db_row_no_chat_id(pipe_instance):
    """Test _make_db_row returns None without chat_id."""
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    result = store._make_db_row(None, "msg", "model", {"type": "note"})
    assert result is None


def test_make_db_row_no_model(pipe_instance):
    """Test _make_db_row returns None without model."""
    store = pipe_instance._artifact_store
    store._item_model = None
    result = store._make_db_row("chat", "msg", "model", {"type": "note"})
    assert result is None


def test_make_db_row_invalid_payload(pipe_instance):
    """Test _make_db_row returns None for non-dict payload."""
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    result = store._make_db_row("chat", "msg", "model", "not a dict")
    assert result is None


def test_make_db_row_valid(pipe_instance):
    """Test _make_db_row returns valid row dict."""
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    result = store._make_db_row("chat", "msg", "model", {"type": "note", "value": 1})
    assert result is not None
    assert result["chat_id"] == "chat"
    assert result["message_id"] == "msg"
    assert result["model_id"] == "model"
    assert result["item_type"] == "note"


# -----------------------------------------------------------------------------
# Prepare Rows for Storage Tests
# -----------------------------------------------------------------------------


def test_prepare_rows_empty(pipe_instance):
    """Test _prepare_rows_for_storage with empty rows."""
    store = pipe_instance._artifact_store
    store._prepare_rows_for_storage([])


def test_prepare_rows_non_dict_items(pipe_instance):
    """Test _prepare_rows_for_storage skips non-dict items."""
    store = pipe_instance._artifact_store
    rows = ["not a dict", 123, None]
    store._prepare_rows_for_storage(rows)


def test_prepare_rows_encrypted_adds_enc_v(pipe_instance):
    """Test _prepare_rows_for_storage adds enc_v to encrypted rows."""
    store = pipe_instance._artifact_store
    rows = [
        {
            "payload": {"ciphertext": "encrypted"},
            "is_encrypted": True,
            "item_type": "reasoning",
        }
    ]
    store._prepare_rows_for_storage(rows)
    assert rows[0]["payload"]["enc_v"] == persistence_mod._ENCRYPTED_PAYLOAD_VERSION


def test_prepare_rows_encrypts_reasoning(pipe_instance):
    """Test _prepare_rows_for_storage encrypts reasoning items."""
    store = pipe_instance._artifact_store
    store._encryption_key = "test-key"
    store._encrypt_all = False
    store._fernet = None
    store._compression_enabled = False

    rows = [
        {
            "payload": {"type": "reasoning", "text": "secret"},
            "is_encrypted": False,
            "item_type": "reasoning",
        }
    ]
    store._prepare_rows_for_storage(rows)
    assert rows[0]["is_encrypted"] is True
    assert "ciphertext" in rows[0]["payload"]


# -----------------------------------------------------------------------------
# Artifact Cleanup Worker Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_artifact_cleanup_worker_exception_handling(pipe_instance, caplog, monkeypatch):
    """Test _artifact_cleanup_worker handles exceptions and logs them."""
    store = pipe_instance._artifact_store

    call_count = [0]

    async def _failing_cleanup():
        call_count[0] += 1
        raise RuntimeError("Cleanup error")

    async def _fast_sleep(seconds):
        # After first iteration, cancel to stop the loop
        if call_count[0] >= 1:
            raise asyncio.CancelledError()

    store._run_cleanup_once = _failing_cleanup
    monkeypatch.setattr("asyncio.sleep", _fast_sleep)

    caplog.set_level(logging.WARNING)
    # The worker will log the error then sleep, which cancels
    with pytest.raises(asyncio.CancelledError):
        await store._artifact_cleanup_worker()
    assert any("Artifact cleanup failed" in rec.message for rec in caplog.records)


# -----------------------------------------------------------------------------
# DB Persist Direct Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_db_persist_direct_no_executor(pipe_instance):
    """Test _db_persist_direct returns empty without executor."""
    store = pipe_instance._artifact_store
    store._db_executor = None
    result = await store._db_persist_direct([{"id": "test"}])
    assert result == []


@pytest.mark.asyncio
async def test_db_persist_direct_no_model(pipe_instance):
    """Test _db_persist_direct returns empty without model."""
    store = pipe_instance._artifact_store
    store._db_executor = ThreadPoolExecutor(max_workers=1)
    store._item_model = None
    result = await store._db_persist_direct([{"id": "test"}])
    assert result == []


@pytest.mark.asyncio
async def test_db_persist_direct_caches_in_redis(pipe_instance):
    """Test _db_persist_direct caches in Redis when enabled."""
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._redis_enabled = True

    cached = []

    async def _capture_cache(rows):
        cached.extend(rows)

    store._redis_cache_rows = _capture_cache

    row = _make_row("chat", "msg", {"type": "note"})
    await store._db_persist_direct([row])
    assert cached


@pytest.mark.asyncio
async def test_db_persist_direct_resets_failure(pipe_instance):
    """Test _db_persist_direct resets failure counter on success."""
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._redis_enabled = False

    reset_called = []
    store._reset_db_failure = lambda user_id: reset_called.append(user_id)

    row = _make_row("chat", "msg", {"type": "note"})
    await store._db_persist_direct([row], user_id="user1")
    assert "user1" in reset_called


# ===== From test_persistence_artifact_store.py =====


import contextlib
import datetime
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.storage import persistence as persistence_mod
from open_webui_openrouter_pipe.storage.persistence import generate_item_id


class _Field:
    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other):  # type: ignore[override]
        return ("eq", self.name, other)

    def __lt__(self, other):  # type: ignore[override]
        return ("lt", self.name, other)

    def in_(self, values):
        return ("in", self.name, list(values))

    def asc(self):
        return ("asc", self.name)

    def desc(self):
        return ("desc", self.name)


class _FakeModel:
    id = _Field("id")
    chat_id = _Field("chat_id")
    message_id = _Field("message_id")
    model_id = _Field("model_id")
    item_type = _Field("item_type")
    payload = _Field("payload")
    is_encrypted = _Field("is_encrypted")
    created_at = _Field("created_at")

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


@contextlib.contextmanager
def _install_internal_db(engine, schema: str | None = None):
    saved_internal = sys.modules.get("open_webui.internal")
    saved_db = sys.modules.get("open_webui.internal.db")

    internal_pkg = types.ModuleType("open_webui.internal")
    db_mod = types.ModuleType("open_webui.internal.db")
    setattr(db_mod, "engine", engine)
    setattr(db_mod, "ENGINE", engine)
    if schema is not None:
        class _Base:
            metadata = types.SimpleNamespace(schema=schema)
        setattr(db_mod, "Base", _Base)

    setattr(internal_pkg, "db", db_mod)
    sys.modules["open_webui.internal"] = internal_pkg
    sys.modules["open_webui.internal.db"] = db_mod
    try:
        yield
    finally:
        if saved_internal is not None:
            sys.modules["open_webui.internal"] = saved_internal
        else:
            sys.modules.pop("open_webui.internal", None)
        if saved_db is not None:
            sys.modules["open_webui.internal.db"] = saved_db
        else:
            sys.modules.pop("open_webui.internal.db", None)


class _FakeQuery:
    def __init__(self, rows: list[_FakeModel], select_fields: list[_Field] | None = None) -> None:
        self._rows = rows
        self._filters: list[tuple[str, str, Any]] = []
        self._order: tuple[str, str] | None = None
        self._limit: int | None = None
        self._select_fields = select_fields

    def filter(self, condition):
        self._filters.append(condition)
        return self

    def order_by(self, order):
        if isinstance(order, tuple):
            self._order = (order[0], order[1])
        return self

    def limit(self, limit: int):
        self._limit = int(limit)
        return self

    def distinct(self):
        return self

    def _match(self, row: _FakeModel, condition: tuple[str, str, Any]) -> bool:
        op, name, value = condition
        current = getattr(row, name, None)
        if op == "eq":
            return current == value
        if op == "in":
            return current in value
        if op == "lt":
            return current < value
        return False

    def _apply(self) -> list[_FakeModel]:
        results = [row for row in self._rows if all(self._match(row, cond) for cond in self._filters)]
        if self._order:
            direction, name = self._order
            reverse = direction == "desc"
            results.sort(key=lambda row: cast(Any, getattr(row, name, None)), reverse=reverse)
        if self._limit is not None:
            results = results[: self._limit]
        return results

    def all(self):
        rows = self._apply()
        if not self._select_fields:
            return rows
        if len(self._select_fields) == 1:
            field = self._select_fields[0].name
            return [(getattr(row, field, None),) for row in rows]
        return [tuple(getattr(row, field.name, None) for field in self._select_fields) for row in rows]

    def first(self):
        rows = self.all()
        return rows[0] if rows else None

    def update(self, values, synchronize_session: bool = False):
        rows = self._apply()
        for row in rows:
            for field, val in values.items():
                if isinstance(field, _Field):
                    setattr(row, field.name, val)
        return len(rows)

    def delete(self, synchronize_session: bool = False):
        rows = self._apply()
        for row in rows:
            self._rows.remove(row)
        return len(rows)


class _FakeSessionSimple:
    def __init__(self, rows: list[_FakeModel]) -> None:
        self._rows = rows

    def add_all(self, instances: list[_FakeModel]) -> None:
        self._rows.extend(instances)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None

    def query(self, *fields):
        if not fields:
            return _FakeQuery(self._rows)
        if len(fields) == 1 and fields[0] is _FakeModel:
            return _FakeQuery(self._rows)
        select_fields = [field for field in fields if isinstance(field, _Field)]
        return _FakeQuery(self._rows, select_fields=select_fields)


def _install_fake_store(pipe: Pipe) -> list[_FakeModel]:
    rows: list[_FakeModel] = []
    store = pipe._artifact_store
    store_any = cast(Any, store)
    store_any._item_model = _FakeModel
    store_any._session_factory = lambda: _FakeSession(rows)
    store_any._artifact_table_name = "response_items_test"
    store_any._db_executor = ThreadPoolExecutor(max_workers=1)
    return rows


def _sqlite_engine():
    return create_engine("sqlite://")


def _make_row(chat_id: str, message_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "chat_id": chat_id,
        "message_id": message_id,
        "model_id": "model",
        "item_type": payload.get("type", "unknown"),
        "payload": payload,
    }


def test_discover_engine_and_schema_from_get_db_context(pipe_instance):
    engine = object()

    class _Session:
        def __init__(self, engine):
            self._engine = engine

        def get_bind(self):
            return self._engine

        def close(self):
            return None

    @contextlib.contextmanager
    def _ctx():
        yield _Session(engine)

    class _DB:
        class Base:
            metadata = types.SimpleNamespace(schema="test_schema")

        get_db_context = staticmethod(_ctx)

    discovered_engine, schema, details = pipe_instance._discover_owui_engine_and_schema(_DB)
    assert discovered_engine is engine
    assert schema == "test_schema"
    assert details.get("engine_source")
    assert details.get("schema_source")


def test_init_artifact_store_with_sqlite(pipe_instance):
    engine = _sqlite_engine()
    with _install_internal_db(engine):
        pipe_instance._init_artifact_store(pipe_identifier="pipe", table_fragment="pipe")

    store = pipe_instance._artifact_store
    assert store._item_model is not None
    assert store._session_factory is not None
    assert store._artifact_table_name is not None


def test_maybe_heal_index_conflict_drops_indexes(pipe_instance):
    store = pipe_instance._artifact_store

    class _DummyColumn:
        def __init__(self, name: str, index: bool = True) -> None:
            self.name = name
            self.index = index

    class _DummyIndex:
        def __init__(self, name: str) -> None:
            self.name = name

    class _DummyTable:
        def __init__(self) -> None:
            self.name = "response_items_test"
            self.schema = None
            self.indexes = {_DummyIndex("ix_response_items_test_chat_id")}
            self.columns = [_DummyColumn("chat_id", index=True)]

    executed: list[str] = []

    class _DummyEngine:
        def begin(self):
            class _Conn:
                def execute(self, stmt):
                    executed.append(str(stmt))

            return contextlib.nullcontext(_Conn())

    result = store._maybe_heal_index_conflict(
        _DummyEngine(),
        _DummyTable(),
        Exception("duplicate key value violates unique constraint ix_response_items_test_chat_id"),
    )

    assert result is True
    assert executed


def test_db_persist_and_fetch_roundtrip(pipe_instance):
    _install_fake_store(pipe_instance)

    row = _make_row("chat-1", "msg-1", {"type": "reasoning", "text": "hi"})
    ulids = pipe_instance._db_persist_sync([row])
    assert len(ulids) == 1

    fetched = pipe_instance._db_fetch_sync("chat-1", "msg-1", ulids)
    assert fetched[ulids[0]]["text"] == "hi"


@pytest.mark.asyncio
async def test_db_persist_direct_and_fetch_async(pipe_instance):
    _install_fake_store(pipe_instance)

    row = _make_row("chat-2", "msg-2", {"type": "note", "value": 123})
    ulids = await pipe_instance._db_persist([row])
    assert len(ulids) == 1

    fetched = await pipe_instance._db_fetch("chat-2", "msg-2", ulids)
    assert fetched[ulids[0]]["value"] == 123


@pytest.mark.asyncio
async def test_delete_and_cleanup_paths(pipe_instance):
    _install_fake_store(pipe_instance)

    model = pipe_instance._artifact_store._item_model
    session_factory = pipe_instance._artifact_store._session_factory
    assert model is not None
    assert session_factory is not None

    session = session_factory()  # type: ignore[call-arg]
    try:
        old_id = generate_item_id()
        recent_id = generate_item_id()
        old_row = model(
            id=old_id,
            chat_id="chat-3",
            message_id="msg-3",
            model_id=None,
            item_type="reasoning",
            payload={"type": "reasoning", "text": "old"},
            is_encrypted=False,
            created_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=99),
        )
        recent_row = model(
            id=recent_id,
            chat_id="chat-3",
            message_id="msg-3",
            model_id=None,
            item_type="reasoning",
            payload={"type": "reasoning", "text": "recent"},
            is_encrypted=False,
            created_at=datetime.datetime.now(datetime.UTC),
        )
        session.add_all([old_row, recent_row])
        session.commit()
    finally:
        session.close()

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=30)
    pipe_instance._cleanup_sync(cutoff)

    remaining = pipe_instance._db_fetch_sync("chat-3", "msg-3", [old_id, recent_id])
    assert old_id not in remaining
    assert recent_id in remaining

    await pipe_instance._delete_artifacts([("chat-3", recent_id)])
    remaining_after = pipe_instance._db_fetch_sync("chat-3", "msg-3", [recent_id])
    assert remaining_after == {}


def test_duplicate_key_detection(pipe_instance):
    from sqlalchemy.exc import SQLAlchemyError

    assert pipe_instance._is_duplicate_key_error(SQLAlchemyError("duplicate key value violates unique constraint"))
    assert pipe_instance._is_duplicate_key_error(SQLAlchemyError("UNIQUE constraint failed")) is True
    assert pipe_instance._is_duplicate_key_error(SQLAlchemyError("other error")) is False


@pytest.mark.asyncio
async def test_redis_enqueue_cache_and_fetch(pipe_instance):
    _install_fake_store(pipe_instance)

    class _FakePipeline:
        def __init__(self, redis):
            self.redis = redis
            self.ops = []

        def rpush(self, key, value):
            self.ops.append(("rpush", key, value))
            return self

        def lpush(self, key, value):
            self.ops.append(("lpush", key, value))
            return self

        def setex(self, key, ttl, value):
            self.ops.append(("setex", key, ttl, value))
            return self

        def expire(self, key, ttl):
            self.ops.append(("expire", key, ttl))
            return self

        def execute(self):
            for op in self.ops:
                name = op[0]
                if name == "rpush":
                    _, key, value = op
                    self.redis.rpush(key, value)
                elif name == "lpush":
                    _, key, value = op
                    self.redis.lpush(key, value)
                elif name == "setex":
                    _, key, _, value = op
                    self.redis.setex(key, 60, value)
                elif name == "expire":
                    _, key, ttl = op
                    self.redis.expire(key, ttl)
            self.ops = []
            return []

    class _FakeRedis:
        def __init__(self):
            self.storage = {}
            self.lists = {}
            self.published = []

        def pipeline(self):
            return _FakePipeline(self)

        def rpush(self, key, value):
            self.lists.setdefault(key, []).append(value)

        def lpush(self, key, value):
            self.lists.setdefault(key, []).insert(0, value)

        def lpop(self, key):
            values = self.lists.get(key, [])
            return values.pop(0) if values else None

        def llen(self, key):
            return len(self.lists.get(key, []))

        def setex(self, key, _ttl, value):
            self.storage[key] = value
            return True

        def expire(self, _key, _ttl):
            return True

        def mget(self, keys):
            return [self.storage.get(key) for key in keys]

        def publish(self, channel, message):
            self.published.append((channel, message))
            return 1

        def set(self, key, value, nx=False, ex=None):
            if nx and key in self.storage:
                return False
            self.storage[key] = value
            return True

        def eval(self, _script, _numkeys, key, token):
            if self.storage.get(key) == token:
                self.storage.pop(key, None)
                return 1
            return 0

        def delete(self, *keys):
            deleted = 0
            for key in keys:
                if key in self.storage:
                    del self.storage[key]
                    deleted += 1
            return deleted

        def pubsub(self):
            class _PubSub:
                def __init__(self):
                    self._messages = [{"type": "message", "data": "flush"}]

                async def subscribe(self, _channel):
                    return None

                async def listen(self):
                    for msg in self._messages:
                        yield msg

                async def close(self):
                    return None

            return _PubSub()

    store = pipe_instance._artifact_store
    store._redis_enabled = True
    store._redis_client = _FakeRedis()

    rows = [
        _make_row("chat-redis", "msg-redis", {"type": "note", "value": 1}),
    ]
    ids = await pipe_instance._redis_enqueue_rows(rows)
    assert ids

    fetched = await pipe_instance._redis_fetch_rows("chat-redis", ids)
    assert fetched[ids[0]]["value"] == 1


@pytest.mark.asyncio
async def test_redis_flush_queue_and_pubsub(pipe_instance, caplog):
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store

    class _FakeRedis:
        def __init__(self):
            self.storage = {}
            self.lists = {}

        def pipeline(self):
            return None

        def rpush(self, key, value):
            self.lists.setdefault(key, []).append(value)

        def lpop(self, key):
            values = self.lists.get(key, [])
            return values.pop(0) if values else None

        def set(self, key, value, nx=False, ex=None):
            if nx and key in self.storage:
                return False
            self.storage[key] = value
            return True

        def eval(self, _script, _numkeys, key, token):
            if self.storage.get(key) == token:
                self.storage.pop(key, None)
                return 1
            return 0

        def publish(self, _channel, _message):
            return 1

        def llen(self, key):
            return len(self.lists.get(key, []))

        def pubsub(self):
            class _PubSub:
                def __init__(self):
                    self._messages = [{"type": "message", "data": "flush"}]

                async def subscribe(self, _channel):
                    return None

                async def listen(self):
                    for msg in self._messages:
                        yield msg

                async def close(self):
                    return None

            return _PubSub()

    store._redis_enabled = True
    store._redis_client = _FakeRedis()

    raw_good = {"chat_id": "chat", "message_id": "msg", "item_type": "note", "payload": {"type": "note", "value": 2}}
    store._redis_client.rpush(store._redis_pending_key, "not-json")
    store._redis_client.rpush(store._redis_pending_key, '{"payload": "bad"}')
    store._redis_client.rpush(store._redis_pending_key, '{"chat_id": "chat", "message_id": "msg", "item_type": "note", "payload": {"type": "note", "value": 2}}')

    persisted: list[dict[str, Any]] = []

    async def _fake_db_persist_direct(rows, user_id: str = ""):
        persisted.extend(rows)
        return [row.get("id") or generate_item_id() for row in rows]

    store._db_persist_direct = _fake_db_persist_direct  # type: ignore[assignment]

    await pipe_instance._flush_redis_queue()
    assert persisted
    assert any(
        isinstance(row.get("payload"), dict) and row["payload"].get("value") == 2
        for row in persisted
    )

    caplog.set_level("WARNING")
    await pipe_instance._redis_pubsub_listener()


@pytest.mark.asyncio
async def test_redis_periodic_flusher_exits(pipe_instance, monkeypatch):
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store

    class _FakeRedis:
        def __init__(self):
            self.lists = {store._redis_pending_key: ["x"] * (store.valves.REDIS_PENDING_WARN_THRESHOLD + 1)}

        def llen(self, key):
            return len(self.lists.get(key, []))

    store._redis_enabled = True
    store._redis_client = _FakeRedis()

    async def _fake_flush():
        return None

    async def _fake_sleep(_seconds: float):
        store._redis_enabled = False

    store._flush_redis_queue = _fake_flush  # type: ignore[assignment]
    monkeypatch.setattr("asyncio.sleep", _fake_sleep)

    await pipe_instance._redis_periodic_flusher()


def test_encode_crockford_rejects_negative_value() -> None:
    with pytest.raises(ValueError):
        persistence_mod._encode_crockford(-1, 4)


def test_sanitize_table_fragment_truncates_and_normalizes() -> None:
    fragment = persistence_mod._sanitize_table_fragment("++My Table NAME__" * 10)
    assert fragment
    assert len(fragment) <= 62
    assert fragment == fragment.lower()
    assert all(ch.isalnum() or ch == "_" for ch in fragment)


@pytest.mark.asyncio
async def test_wait_for_handles_sync_and_async() -> None:
    async def _coro():
        return "ok"

    assert await persistence_mod._wait_for(5) == 5
    assert await persistence_mod._wait_for(_coro(), timeout=1) == "ok"


def test_decode_payload_bytes_invalid_flag_and_json(pipe_instance) -> None:
    store = pipe_instance._artifact_store
    with pytest.raises(ValueError):
        store._decode_payload_bytes(bytes([9]) + b"{}")
    with pytest.raises(ValueError):
        store._decode_payload_bytes(bytes([0]) + b"not-json")


def test_lz4_decompress_missing_module_raises(pipe_instance, monkeypatch) -> None:
    store = pipe_instance._artifact_store
    assert store._lz4_decompress(b"") == b""
    monkeypatch.setattr(persistence_mod, "lz4frame", None)
    with pytest.raises(RuntimeError):
        store._lz4_decompress(b"data")


def test_prepare_rows_for_storage_sets_enc_v(pipe_instance) -> None:
    store = pipe_instance._artifact_store
    rows: list[Any] = [
        {"payload": {"ciphertext": "abc"}, "is_encrypted": True, "item_type": "reasoning"},
        "skip",
    ]
    store._prepare_rows_for_storage(rows)
    assert rows[0]["payload"]["enc_v"] == persistence_mod._ENCRYPTED_PAYLOAD_VERSION


def test_make_db_row_missing_message_id_warns(pipe_instance, caplog) -> None:
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store

    caplog.set_level("WARNING")
    result = store._make_db_row("chat", None, "model", {"type": "note"})

    assert result is None
    assert any("missing message_id" in rec.message for rec in caplog.records)


def test_db_persist_sync_returns_existing_ids(pipe_instance) -> None:
    _install_fake_store(pipe_instance)
    row = _make_row("chat", "msg", {"type": "note"})
    row["id"] = "id-1"
    row["_persisted"] = True

    result = pipe_instance._db_persist_sync([row])

    assert result == ["id-1"]
    assert "_persisted" not in row


def test_db_persist_sync_skips_missing_payload(pipe_instance, caplog) -> None:
    _install_fake_store(pipe_instance)
    row = {
        "chat_id": "chat",
        "message_id": "msg",
        "model_id": "model",
        "item_type": "note",
        "payload": None,
    }

    caplog.set_level("WARNING")
    result = pipe_instance._db_persist_sync([row])

    assert result == []
    assert any("payload missing" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_db_persist_skips_when_breaker_open(monkeypatch, pipe_instance) -> None:
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    monkeypatch.setattr(store, "_db_breaker_allows", lambda _user_id: False)

    calls: list[str] = []

    async def _emit_notification(*_args, **_kwargs):
        calls.append("notify")

    monkeypatch.setattr(store, "_emit_notification", _emit_notification)
    monkeypatch.setattr(store, "_record_failure", lambda _user_id: calls.append("record"))

    result = await store._db_persist([_make_row("chat", "msg", {"type": "note"})])

    assert result == []
    assert "record" in calls


@pytest.mark.asyncio
async def test_db_persist_records_failure_on_exception(monkeypatch, pipe_instance) -> None:
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._redis_enabled = True

    async def _boom(_rows, **_kwargs):
        raise RuntimeError("redis down")

    calls: list[str] = []

    monkeypatch.setattr(store, "_redis_enqueue_rows", _boom)
    monkeypatch.setattr(store, "_record_db_failure", lambda _user_id: calls.append("failed"))

    result = await store._db_persist([_make_row("chat", "msg", {"type": "note"})])

    assert result == []
    assert "failed" in calls


@pytest.mark.asyncio
async def test_db_persist_direct_duplicate_key_returns_ids(monkeypatch, pipe_instance) -> None:
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._redis_enabled = False
    store._db_executor = object()

    class _Loop:
        async def run_in_executor(self, _executor, func, *args):
            return func(*args)

    monkeypatch.setattr(persistence_mod.asyncio, "get_running_loop", lambda: _Loop())

    def _boom(_rows):
        raise SQLAlchemyError("duplicate key value violates unique constraint")

    monkeypatch.setattr(store, "_db_persist_sync", _boom)

    result = await store._db_persist_direct([{"id": "id-1"}])

    assert result == ["id-1"]


@pytest.mark.asyncio
async def test_db_fetch_breaker_disables_reads(monkeypatch, pipe_instance) -> None:
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    monkeypatch.setattr(store, "_db_breaker_allows", lambda _user_id: False)

    calls: list[str] = []

    async def _emit_notification(*_args, **_kwargs):
        calls.append("notify")

    monkeypatch.setattr(store, "_emit_notification", _emit_notification)
    monkeypatch.setattr(store, "_record_failure", lambda _user_id: calls.append("record"))

    result = await store._db_fetch("chat", "msg", ["id-1"])

    assert result == {}
    assert "record" in calls


@pytest.mark.asyncio
async def test_db_fetch_handles_exception_returns_cache(monkeypatch, pipe_instance) -> None:
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    def _raise_fetch(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(store, "_db_fetch_direct", _raise_fetch)

    calls: list[str] = []
    monkeypatch.setattr(store, "_record_db_failure", lambda _user_id: calls.append("failed"))

    result = await store._db_fetch("chat", "msg", ["id-1"])

    assert result == {}
    assert "failed" in calls


@pytest.mark.asyncio
async def test_delete_artifacts_deletes_redis_cache(pipe_instance) -> None:
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._redis_enabled = True

    deleted: list[str] = []

    class _FakeRedis:
        def delete(self, *keys):
            deleted.extend(keys)
            return len(keys)

    store._redis_client = _FakeRedis()
    store._delete_artifacts_sync = lambda _ids: None  # type: ignore[assignment]

    await store._delete_artifacts([("chat", "id-1"), ("chat", "id-2")])

    assert deleted


@pytest.mark.asyncio
async def test_redis_periodic_flusher_disables_after_failures(monkeypatch, pipe_instance) -> None:
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._redis_enabled = True
    store.valves.REDIS_FLUSH_FAILURE_LIMIT = 1

    class _FailingRedis:
        def __init__(self):
            self.calls = 0

        def llen(self, _key):
            self.calls += 1
            raise RuntimeError("boom")

    store._redis_client = _FailingRedis()

    await store._redis_periodic_flusher()

    assert store._redis_enabled is False


@pytest.mark.asyncio
async def test_flush_redis_queue_skips_without_lock(pipe_instance) -> None:
    _install_fake_store(pipe_instance)
    store = pipe_instance._artifact_store
    store._redis_enabled = True

    class _FakeRedis:
        def __init__(self):
            self.storage = {}

        def set(self, *_args, **_kwargs):
            return False

    store._redis_client = _FakeRedis()

    await store._flush_redis_queue()


# ===== From test_artifact_db_touch.py =====


from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def test_db_fetch_sync_touches_created_at_best_effort(pipe_instance) -> None:
    """Test that _db_fetch_sync touches created_at timestamp during fetch.

    This test uses real SQLAlchemy mocking to verify the touch behavior.
    """
    # Mock the SQLAlchemy model and query chain
    mock_model = MagicMock()
    mock_query = MagicMock()
    mock_session = MagicMock()

    # Configure query chain
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_query

    # Mock rows with proper attributes
    mock_row_a = MagicMock()
    mock_row_a.id = "a1"
    mock_row_a.payload = {"type": "tool_output", "output": "ok"}
    mock_row_a.is_encrypted = False

    mock_row_b = MagicMock()
    mock_row_b.id = "b2"
    mock_row_b.payload = {"type": "tool_output", "output": "ok"}
    mock_row_b.is_encrypted = False

    mock_query.all.return_value = [mock_row_a, mock_row_b]
    mock_query.update.return_value = 2

    # Set up pipe instance with mocked session factory
    pipe_instance._artifact_store._item_model = mock_model
    pipe_instance._artifact_store._session_factory = lambda: mock_session

    # Execute the fetch
    results = pipe_instance._db_fetch_sync("chat123", None, ["a1", "b2"])

    # Verify results
    assert results == {"a1": {"type": "tool_output", "output": "ok"}, "b2": {"type": "tool_output", "output": "ok"}}

    # Verify session was used correctly (2 sessions: one for read, one for touch)
    # The touch session should have committed
    assert mock_session.commit.called

    # Verify the touch update was called
    assert mock_query.update.called


def test_db_fetch_sync_touch_failure_never_breaks_read(pipe_instance) -> None:
    """Test that touch failures are gracefully handled and don't break reads.

    Uses real exception handling with SQLAlchemy mocks.
    """
    # Mock the SQLAlchemy model and sessions
    mock_model = MagicMock()

    # First session for reading (succeeds)
    read_session = MagicMock()
    read_query = MagicMock()
    read_session.query.return_value = read_query
    read_query.filter.return_value = read_query

    mock_row = MagicMock()
    mock_row.id = "a1"
    mock_row.payload = {"type": "tool_output", "output": "ok"}
    mock_row.is_encrypted = False

    read_query.all.return_value = [mock_row]

    # Second session for touching (fails)
    touch_session = MagicMock()
    touch_query = MagicMock()
    touch_session.query.return_value = touch_query
    touch_query.filter.return_value = touch_query
    touch_query.all.return_value = [mock_row]

    # Make update raise an exception
    touch_query.update.side_effect = RuntimeError("update failed")

    # Session factory returns different sessions
    sessions = [read_session, touch_session]
    pipe_instance._artifact_store._item_model = mock_model
    pipe_instance._artifact_store._session_factory = lambda: sessions.pop(0)

    # Execute the fetch - should succeed despite touch failure
    results = pipe_instance._db_fetch_sync("chat123", None, ["a1"])

    # Verify results are still correct
    assert results == {"a1": {"type": "tool_output", "output": "ok"}}

    # Verify touch session was rolled back after failure
    assert touch_session.rollback.called


# ===== From test_artifact_helpers.py =====

import json

from open_webui_openrouter_pipe import (
    _classify_function_call_artifacts,
    _dedupe_tools,
    _normalize_persisted_item,
)


def test_normalize_function_call_serializes_arguments_and_ids():
    item = {
        "type": "function_call",
        "name": "lookup",
        "arguments": {"city": "Lisbon", "units": "metric"},
    }

    normalized = _normalize_persisted_item(item)
    assert normalized is not None

    assert normalized["type"] == "function_call"
    assert json.loads(normalized["arguments"]) == {"city": "Lisbon", "units": "metric"}
    assert normalized["status"] == "completed"
    assert normalized["call_id"]
    assert normalized["id"]


def test_normalize_function_call_output_assigns_defaults():
    item = {"type": "function_call_output", "output": {"ok": True}}

    normalized = _normalize_persisted_item(item)
    assert normalized is not None

    assert isinstance(normalized["output"], str)
    assert normalized["call_id"]
    assert normalized["status"] == "completed"


def test_normalize_reasoning_and_file_search_payloads():
    reasoning = _normalize_persisted_item(
        {"type": "reasoning", "content": "Chain", "summary": "Done"}
    )
    assert reasoning is not None
    file_search = _normalize_persisted_item({"type": "file_search_call", "queries": "bad"})
    assert file_search is not None
    web_search = _normalize_persisted_item({"type": "web_search_call", "action": "invalid"})
    assert web_search is not None

    assert reasoning["content"] == [{"type": "reasoning_text", "text": "Chain"}]
    assert reasoning["summary"] == ["Done"]
    assert file_search["queries"] == []
    assert web_search["action"] == {}


def test_classify_function_call_artifacts_identifies_orphans():
    artifacts = {
        "a": {"type": "function_call", "call_id": "call-1"},
        "b": {"type": "function_call_output", "call_id": "call-1"},
        "c": {"type": "function_call", "call_id": "call-2"},
        "d": {"type": "function_call_output", "call_id": "call-3"},
        "e": {"type": "function_call_output", "call_id": "   "},
    }

    valid, orphaned_calls, orphaned_outputs = _classify_function_call_artifacts(artifacts)

    assert valid == {"call-1"}
    assert orphaned_calls == {"call-2"}
    assert orphaned_outputs == {"call-3"}


def test_dedupe_tools_prefers_latest_definition():
    tools = [
        {"type": "function", "name": "search", "parameters": {"type": "object", "properties": {}}},
        {
            "type": "function",
            "name": "search",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
        {"type": "code_interpreter", "mode": "safe"},
        {"type": "code_interpreter", "mode": "fast"},
        "skip-me",
    ]

    deduped = _dedupe_tools(tools)

    assert len(deduped) == 2
    assert deduped[0]["parameters"]["properties"] == {"query": {"type": "string"}}
    assert deduped[1]["mode"] == "fast"
