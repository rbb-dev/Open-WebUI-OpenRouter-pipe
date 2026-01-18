"""Comprehensive unit tests for _qualify_model_for_pipe method.

This method qualifies OpenRouter model IDs with pipe-specific prefixes
for proper routing in Open WebUI.
"""

import pytest

from open_webui_openrouter_pipe import Pipe


class TestQualifyModelForPipe:
    """Test suite for _qualify_model_for_pipe method."""

    def test_basic_qualification_with_valid_inputs(self, pipe_instance):
        """Test basic qualification with pipe identifier and model ID."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", "gpt-4")
        assert result == "mypipe.gpt-4"

    def test_qualification_with_normalized_model_id(self, pipe_instance):
        """Test that model IDs are normalized before qualification."""
        pipe = pipe_instance
        # ModelFamily.base_model normalizes: lowercase, strip dates, replace / with .
        result = pipe._qualify_model_for_pipe("mypipe", "openai/gpt-4")
        assert result == "mypipe.openai.gpt-4"

    def test_already_qualified_model_returns_as_is(self, pipe_instance):
        """Test that already-qualified models are not double-qualified."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", "mypipe.gpt-4")
        assert result == "mypipe.gpt-4"

    def test_no_pipe_identifier_returns_model_id(self, pipe_instance):
        """Test behavior when pipe_identifier is None or empty."""
        pipe = pipe_instance

        # None pipe_identifier
        result = pipe._qualify_model_for_pipe(None, "gpt-4")
        assert result == "gpt-4"

        # Empty string pipe_identifier
        result = pipe._qualify_model_for_pipe("", "gpt-4")
        assert result == "gpt-4"

        # Whitespace-only pipe_identifier
        result = pipe._qualify_model_for_pipe("  ", "gpt-4")
        assert result is not None  # Should still qualify with the trimmed identifier

    def test_none_model_id_returns_none(self, pipe_instance):
        """Test that None model_id returns None."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", None)
        assert result is None

    def test_non_string_model_id_returns_none(self, pipe_instance):
        """Test that non-string model IDs return None."""
        pipe = pipe_instance

        # Integer
        result = pipe._qualify_model_for_pipe("mypipe", 123)
        assert result is None

        # List
        result = pipe._qualify_model_for_pipe("mypipe", ["gpt-4"])
        assert result is None

        # Dict
        result = pipe._qualify_model_for_pipe("mypipe", {"model": "gpt-4"})
        assert result is None

    def test_empty_string_model_id_returns_none(self, pipe_instance):
        """Test that empty or whitespace-only model IDs return None."""
        pipe = pipe_instance

        # Empty string
        result = pipe._qualify_model_for_pipe("mypipe", "")
        assert result is None

        # Whitespace only
        result = pipe._qualify_model_for_pipe("mypipe", "   ")
        assert result is None

        # Tabs and newlines
        result = pipe._qualify_model_for_pipe("mypipe", "\t\n  ")
        assert result is None

    def test_model_id_with_leading_trailing_whitespace(self, pipe_instance):
        """Test that model IDs with whitespace are trimmed."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", "  gpt-4  ")
        assert result == "mypipe.gpt-4"

    def test_complex_model_ids_with_slashes(self, pipe_instance):
        """Test model IDs containing slashes (e.g., provider/model format)."""
        pipe = pipe_instance
        # Slashes should be normalized to dots by ModelFamily.base_model
        result = pipe._qualify_model_for_pipe("mypipe", "anthropic/claude-3-opus")
        assert result == "mypipe.anthropic.claude-3-opus"

    def test_model_ids_with_dates_are_normalized(self, pipe_instance):
        """Test that date suffixes in model IDs are stripped during normalization."""
        pipe = pipe_instance
        # ModelFamily.base_model strips date patterns
        result = pipe._qualify_model_for_pipe("mypipe", "gpt-4-2024-01-15")
        # Date should be stripped by normalization
        assert "2024" not in result
        assert result.startswith("mypipe.")

    def test_case_normalization(self, pipe_instance):
        """Test that model IDs are normalized to lowercase."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", "GPT-4")
        # Should be lowercased by ModelFamily.base_model
        assert result == "mypipe.gpt-4"

    def test_pipe_identifier_with_special_characters(self, pipe_instance):
        """Test pipe identifiers with various characters."""
        pipe = pipe_instance

        # Alphanumeric with dashes
        result = pipe._qualify_model_for_pipe("my-pipe-123", "gpt-4")
        assert result == "my-pipe-123.gpt-4"

        # Underscores
        result = pipe._qualify_model_for_pipe("my_pipe", "gpt-4")
        assert result == "my_pipe.gpt-4"

    def test_multiple_dots_in_qualified_id(self, pipe_instance):
        """Test that already-qualified IDs with multiple dots are handled."""
        pipe = pipe_instance
        # Model already has pipe prefix with dots
        result = pipe._qualify_model_for_pipe("mypipe", "mypipe.provider.model-v1")
        assert result == "mypipe.provider.model-v1"

    def test_normalization_fallback_behavior(self, pipe_instance):
        """Test behavior when ModelFamily.base_model returns None."""
        pipe = pipe_instance
        # If normalization fails or returns None, should use original trimmed value
        # This tests the `or trimmed` fallback in: normalized = ModelFamily.base_model(trimmed) or trimmed
        result = pipe._qualify_model_for_pipe("mypipe", "unknown-model")
        assert result is not None
        assert result.startswith("mypipe.")

    def test_preserves_hyphenated_model_names(self, pipe_instance):
        """Test that hyphens in model names are preserved."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", "claude-3-opus")
        assert "claude-3-opus" in result
        assert result.startswith("mypipe.")

    def test_prefix_detection_is_exact(self, pipe_instance):
        """Test that prefix detection requires exact match with dot separator."""
        pipe = pipe_instance

        # Model starts with identifier but no dot - should NOT be considered qualified
        result = pipe._qualify_model_for_pipe("my", "mygpt-4")
        assert result == "my.mygpt-4"  # Should qualify it

        # Model with exact prefix match - should be considered already qualified
        result = pipe._qualify_model_for_pipe("my", "my.gpt-4")
        assert result == "my.gpt-4"  # Should NOT double-qualify

    def test_real_world_openrouter_model_ids(self, pipe_instance):
        """Test with realistic OpenRouter model ID formats."""
        pipe = pipe_instance

        # Standard OpenRouter format
        result = pipe._qualify_model_for_pipe("openrouter", "openai/gpt-4-turbo")
        assert result.startswith("openrouter.")
        assert "openai" in result
        assert "gpt-4-turbo" in result

        # Anthropic model
        result = pipe._qualify_model_for_pipe("openrouter", "anthropic/claude-3-opus-20240229")
        assert result.startswith("openrouter.")
        assert "anthropic" in result
        assert "claude-3-opus" in result

    def test_unicode_characters_in_model_id(self, pipe_instance):
        """Test that unicode characters in model IDs are preserved."""
        pipe = pipe_instance
        result = pipe._qualify_model_for_pipe("mypipe", "model-名前")
        assert result is not None
        assert "mypipe." in result

    def test_very_long_model_id(self, pipe_instance):
        """Test with unusually long model ID."""
        pipe = pipe_instance
        long_model = "a" * 200
        result = pipe._qualify_model_for_pipe("mypipe", long_model)
        assert result is not None
        assert result.startswith("mypipe.")
        assert len(result) > 200

    def test_qualification_is_idempotent(self, pipe_instance):
        """Test that qualifying an already-qualified ID returns the same result."""
        pipe = pipe_instance

        # First qualification
        first = pipe._qualify_model_for_pipe("mypipe", "gpt-4")

        # Second qualification of already-qualified ID
        second = pipe._qualify_model_for_pipe("mypipe", first)

        # Should be idempotent
        assert first == second
        assert first == "mypipe.gpt-4"
