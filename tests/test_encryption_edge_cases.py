"""Encryption and compression edge case tests."""

from __future__ import annotations

import json

import pytest

from open_webui_openrouter_pipe import (
    Pipe,
    _PAYLOAD_FLAG_LZ4,
    _PAYLOAD_FLAG_PLAIN,
)


def test_maybe_compress_payload_respects_min_bytes_threshold(pipe_instance) -> None:
    """When MIN_COMPRESS_BYTES is set, small payloads are not compressed."""
    pipe = pipe_instance
    pipe._compression_enabled = True
    pipe._compression_min_bytes = 10_000

    serialized = json.dumps({"k": "v"}, separators=(",", ":")).encode("utf-8")
    data, compressed = pipe._maybe_compress_payload(serialized)
    assert data == serialized
    assert compressed is False


def test_encode_payload_bytes_sets_plain_flag_when_not_compressed(pipe_instance) -> None:
    """_encode_payload_bytes uses the plain flag when compression is skipped."""
    pipe = pipe_instance
    pipe._compression_enabled = True
    pipe._compression_min_bytes = 10_000

    payload = {"k": "v"}
    encoded = pipe._encode_payload_bytes(payload)
    assert encoded[:1] == bytes([_PAYLOAD_FLAG_PLAIN])
    assert pipe._decode_payload_bytes(encoded) == payload


def test_encode_payload_bytes_uses_lz4_flag_when_compression_is_effective(pipe_instance) -> None:
    """When LZ4 yields a smaller buffer, _encode_payload_bytes marks it as compressed."""
    pipe = pipe_instance
    pipe._compression_enabled = True
    pipe._compression_min_bytes = 0

    payload = {"data": "x" * 10000}
    encoded = pipe._encode_payload_bytes(payload)
    assert pipe._decode_payload_bytes(encoded) == payload
    assert encoded[:1] in {bytes([_PAYLOAD_FLAG_PLAIN]), bytes([_PAYLOAD_FLAG_LZ4])}


def test_encrypt_payload_requires_encryption_key(pipe_instance) -> None:
    """_encrypt_payload raises when ARTIFACT_ENCRYPTION_KEY is not configured."""
    pipe = pipe_instance
    pipe._encryption_key = ""
    pipe._fernet = None
    with pytest.raises(RuntimeError):
        pipe._encrypt_payload({"k": "v"})


def test_encrypt_decrypt_payload_roundtrip_with_key(pipe_instance) -> None:
    """_encrypt_payload and _decrypt_payload roundtrip when a key is configured."""
    pipe = pipe_instance
    pipe._encryption_key = "unit-test-artifact-key"
    pipe._fernet = None

    payload = {"content": "test data", "reasoning": "thinking..."}
    ciphertext = pipe._encrypt_payload(payload)
    assert isinstance(ciphertext, str) and ciphertext
    assert pipe._decrypt_payload(ciphertext) == payload


def test_encrypt_if_needed_encrypts_reasoning_only_when_encrypt_all_false(pipe_instance) -> None:
    """When ENCRYPT_ALL is False, only reasoning items are encrypted."""
    pipe = pipe_instance
    pipe._encryption_key = "unit-test-artifact-key"
    pipe._fernet = None
    pipe._encrypt_all = False

    payload = {"k": "v"}
    stored, is_encrypted = pipe._encrypt_if_needed("tool", payload)
    assert is_encrypted is False
    assert stored == payload

    stored, is_encrypted = pipe._encrypt_if_needed("reasoning", payload)
    assert is_encrypted is True
    assert isinstance(stored, dict)
    assert "ciphertext" in stored
    decrypted = pipe._decrypt_payload(stored["ciphertext"])
    assert decrypted == payload
