"""Tests for direct upload blocklist functionality.

These tests verify the blocklist module and its integration with the registry.
All tests use the REAL blocklist and registry code - no mocking.
"""

from __future__ import annotations

import pytest

# Import the REAL blocklist module - no mocks
from open_webui_openrouter_pipe.models.blocklists import (
    DIRECT_UPLOAD_BLOCKLIST,
    is_direct_upload_blocklisted,
)


class TestBlocklistFunction:
    """Tests for is_direct_upload_blocklisted() function."""

    def test_blocklisted_model_returns_true(self):
        """Models in the blocklist should return True."""
        # Test a few known blocklisted models
        assert is_direct_upload_blocklisted("openai/gpt-4-turbo") is True
        assert is_direct_upload_blocklisted("openai/gpt-4-1106-preview") is True
        assert is_direct_upload_blocklisted("meta-llama/llama-guard-3-8b") is True
        assert is_direct_upload_blocklisted("liquid/lfm2-8b-a1b") is True

    def test_non_blocklisted_model_returns_false(self):
        """Models not in the blocklist should return False."""
        assert is_direct_upload_blocklisted("openai/gpt-4o") is False
        assert is_direct_upload_blocklisted("anthropic/claude-3-opus") is False
        assert is_direct_upload_blocklisted("google/gemini-pro") is False
        assert is_direct_upload_blocklisted("mistralai/mistral-large") is False

    def test_case_insensitivity(self):
        """Blocklist check should be case-insensitive."""
        # Mixed case should still match
        assert is_direct_upload_blocklisted("OpenAI/GPT-4-Turbo") is True
        assert is_direct_upload_blocklisted("OPENAI/GPT-4-TURBO") is True
        assert is_direct_upload_blocklisted("Meta-Llama/Llama-Guard-3-8B") is True

    def test_whitespace_handling(self):
        """Leading/trailing whitespace should be ignored."""
        assert is_direct_upload_blocklisted("  openai/gpt-4-turbo  ") is True
        assert is_direct_upload_blocklisted("\topenai/gpt-4-turbo\n") is True

    def test_empty_string_returns_false(self):
        """Empty string should return False, not crash."""
        assert is_direct_upload_blocklisted("") is False

    def test_none_like_values(self):
        """Edge cases should not crash."""
        # Empty/whitespace only
        assert is_direct_upload_blocklisted("   ") is False

    def test_blocklist_has_expected_count(self):
        """Blocklist should contain the expected number of models."""
        assert len(DIRECT_UPLOAD_BLOCKLIST) == 24

    def test_all_blocklist_entries_are_strings(self):
        """All blocklist entries should be non-empty strings."""
        for model_id in DIRECT_UPLOAD_BLOCKLIST:
            assert isinstance(model_id, str)
            assert len(model_id) > 0
            assert "/" in model_id  # Should be in provider/model format


class TestBlocklistCategories:
    """Verify each category of blocklisted models is present."""

    def test_provider_rejected_models_present(self):
        """HTTP 400 models should be in blocklist."""
        provider_rejected = [
            "liquid/lfm2-8b-a1b",
            "liquid/lfm-2.2-6b",
            "relace/relace-apply-3",
            "openai/gpt-4o-audio-preview",
            "arcee-ai/spotlight",
            "arcee-ai/coder-large",
        ]
        for model_id in provider_rejected:
            assert model_id in DIRECT_UPLOAD_BLOCKLIST, f"{model_id} should be blocklisted"

    def test_guard_models_present(self):
        """Guard/classifier models should be in blocklist."""
        guard_models = [
            "meta-llama/llama-guard-2-8b",
            "meta-llama/llama-guard-3-8b",
            "meta-llama/llama-guard-4-12b",
        ]
        for model_id in guard_models:
            assert model_id in DIRECT_UPLOAD_BLOCKLIST, f"{model_id} should be blocklisted"

    def test_explicit_no_file_support_present(self):
        """Models that explicitly say they can't process files should be blocklisted."""
        no_file_models = [
            "openai/gpt-4-turbo",
            "openai/gpt-4-1106-preview",
            "qwen/qwen2.5-coder-7b-instruct",
        ]
        for model_id in no_file_models:
            assert model_id in DIRECT_UPLOAD_BLOCKLIST, f"{model_id} should be blocklisted"
