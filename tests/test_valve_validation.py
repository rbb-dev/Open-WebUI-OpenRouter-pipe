"""Tests for valve validation, defaults, and edge cases."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from open_webui_openrouter_pipe import (
    EncryptedStr,
    Pipe,
)


class TestEncryptedStr:
    """Tests for EncryptedStr encryption/decryption."""

    def test_encrypted_str_roundtrip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EncryptedStr encrypts/decrypts when WEBUI_SECRET_KEY is set."""
        monkeypatch.setenv("WEBUI_SECRET_KEY", "unit-test-webui-secret")
        original_value = "secret_api_key_12345"

        # Encrypt the value
        encrypted = EncryptedStr.encrypt(original_value)

        # Verify it's encrypted (not plain text)
        assert encrypted != original_value
        assert encrypted.startswith("encrypted:")

        # Decrypt the value
        decrypted = EncryptedStr.decrypt(encrypted)

        # Verify roundtrip
        assert decrypted == original_value

    def test_encrypted_str_without_webui_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No WEBUI_SECRET_KEY → values are stored/returned plain."""
        monkeypatch.delenv("WEBUI_SECRET_KEY", raising=False)
        value = "test_value_without_secret"
        encrypted = EncryptedStr.encrypt(value)
        assert encrypted == value
        decrypted = EncryptedStr.decrypt(encrypted)
        assert decrypted == value


class TestValveNumericBounds:
    """Tests for valve numeric constraints."""

    def test_valve_numeric_bounds_enforced(self) -> None:
        """Pydantic ge/le constraints reject invalid valve values."""
        with pytest.raises(ValidationError):
            Pipe.Valves(HTTP_CONNECT_TIMEOUT_SECONDS=0)
        with pytest.raises(ValidationError):
            Pipe.Valves(MAX_CONCURRENT_REQUESTS=0)
        with pytest.raises(ValidationError):
            Pipe.Valves(SSE_WORKERS_PER_REQUEST=0)


class TestValveEnvironmentDefaults:
    """Tests for valve defaults from environment variables."""

    def test_default_log_level_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LOG_LEVEL defaults from GLOBAL_LOG_LEVEL env."""
        monkeypatch.setenv("GLOBAL_LOG_LEVEL", "DEBUG")
        valves = Pipe.Valves()
        assert valves.LOG_LEVEL == "DEBUG"

    def test_invalid_log_level_env_falls_back_to_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid GLOBAL_LOG_LEVEL values are normalized to INFO."""
        monkeypatch.setenv("GLOBAL_LOG_LEVEL", "nope")
        valves = Pipe.Valves()
        assert valves.LOG_LEVEL == "INFO"

    def test_api_key_default_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """API_KEY defaults from OPENROUTER_API_KEY env."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_api_key_from_env")
        monkeypatch.delenv("WEBUI_SECRET_KEY", raising=False)
        valves = Pipe.Valves()
        assert isinstance(valves.API_KEY, EncryptedStr)
        assert EncryptedStr.decrypt(cast(str, valves.API_KEY)) == "test_api_key_from_env"


class TestValveValidation:
    """Tests for valve validation logic."""

    def test_valve_boolean_defaults(self) -> None:
        """Boolean fields have correct defaults."""
        valves = Pipe.Valves()
        assert isinstance(valves.AUTO_FALLBACK_CHAT_COMPLETIONS, bool)
        assert isinstance(valves.PERSIST_TOOL_RESULTS, bool)
        assert isinstance(valves.ENCRYPT_ALL, bool)
        assert isinstance(valves.ENABLE_LZ4_COMPRESSION, bool)

    def test_valve_pattern_defaults(self) -> None:
        """Pattern valves are strings (glob lists) and default to empty."""
        valves = Pipe.Valves()
        assert isinstance(valves.FORCE_CHAT_COMPLETIONS_MODELS, str)
        assert isinstance(valves.FORCE_RESPONSES_MODELS, str)

    def test_valve_encryption_key_default(self) -> None:
        """ARTIFACT_ENCRYPTION_KEY defaults to an empty EncryptedStr placeholder."""
        valves = Pipe.Valves()
        assert isinstance(valves.ARTIFACT_ENCRYPTION_KEY, EncryptedStr)
        assert EncryptedStr.decrypt(cast(str, valves.ARTIFACT_ENCRYPTION_KEY)) == ""

    def test_valve_compression_settings(self) -> None:
        """Compression settings are validated."""
        valves = Pipe.Valves()
        assert isinstance(valves.ENABLE_LZ4_COMPRESSION, bool)
        assert valves.MIN_COMPRESS_BYTES >= 0

    def test_valve_breaker_settings(self) -> None:
        """Circuit breaker settings are validated."""
        valves = Pipe.Valves()
        assert valves.BREAKER_MAX_FAILURES > 0
        assert valves.BREAKER_WINDOW_SECONDS > 0
        assert valves.BREAKER_HISTORY_SIZE > 0

    def test_valve_concurrency_settings(self) -> None:
        """Concurrency settings are validated."""
        valves = Pipe.Valves()
        assert valves.MAX_CONCURRENT_REQUESTS > 0
        assert valves.MAX_PARALLEL_TOOLS_GLOBAL >= 0
        assert valves.MAX_PARALLEL_TOOLS_PER_REQUEST >= 0
