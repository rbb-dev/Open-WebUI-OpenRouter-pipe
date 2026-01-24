"""Tests for OpenRouter model variants functionality."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from open_webui_openrouter_pipe.pipe import Pipe, EncryptedStr
from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry


class TestVariantModelsExpansion:
    """Test the _expand_variant_models() method."""

    def test_expand_single_variant(self):
        """Test expanding a single variant model."""
        pipe = Pipe()
        pipe.valves.VARIANT_MODELS = "openai/gpt-4o:exacto"

        base_models = [
            {
                "id": "openai.gpt-4o",
                "name": "GPT-4o",
                "original_id": "openai/gpt-4o",
                "norm_id": "openai.gpt-4o",
            }
        ]

        result = pipe._expand_variant_models(base_models, pipe.valves)

        assert len(result) == 2  # Base + variant
        assert result[0]["name"] == "GPT-4o"  # Base unchanged
        assert result[1]["name"] == "GPT-4o Exacto"  # Variant added (no angle brackets)
        assert result[1]["id"] == "openai.gpt-4o:exacto"
        assert result[1]["original_id"] == "openai/gpt-4o"  # Kept as base for icon lookup
        assert result[1]["norm_id"] == "openai.gpt-4o"  # Points to base

    def test_expand_multiple_variants(self):
        """Test expanding multiple variant models."""
        pipe = Pipe()
        pipe.valves.VARIANT_MODELS = "openai/gpt-4o:exacto,anthropic/claude-opus:extended"

        base_models = [
            {
                "id": "openai.gpt-4o",
                "name": "GPT-4o",
                "original_id": "openai/gpt-4o",
                "norm_id": "openai.gpt-4o",
            },
            {
                "id": "anthropic.claude-opus",
                "name": "Claude Opus",
                "original_id": "anthropic/claude-opus",
                "norm_id": "anthropic.claude-opus",
            },
        ]

        result = pipe._expand_variant_models(base_models, pipe.valves)

        assert len(result) == 4  # 2 base + 2 variants
        variant_names = [m["name"] for m in result[2:]]
        assert "GPT-4o Exacto" in variant_names
        assert "Claude Opus Extended" in variant_names

    def test_expand_variant_missing_base(self):
        """Test variant with non-existent base model (should skip)."""
        pipe = Pipe()
        pipe.valves.VARIANT_MODELS = "nonexistent/model:free"

        base_models = []
        result = pipe._expand_variant_models(base_models, pipe.valves)

        assert len(result) == 0  # No variants added

    def test_expand_variant_empty_valve(self):
        """Test with empty VARIANT_MODELS valve."""
        pipe = Pipe()
        pipe.valves.VARIANT_MODELS = ""

        base_models = [{"id": "openai.gpt-4o", "name": "GPT-4o", "original_id": "openai/gpt-4o"}]
        result = pipe._expand_variant_models(base_models, pipe.valves)

        assert len(result) == 1  # Unchanged
        assert result[0]["name"] == "GPT-4o"

    def test_expand_variant_malformed_csv(self):
        """Test with malformed CSV entries (should skip invalid)."""
        pipe = Pipe()
        pipe.valves.VARIANT_MODELS = "openai/gpt-4o:exacto,invalid-no-colon,  ,anthropic/claude-opus:extended"

        base_models = [
            {"id": "openai.gpt-4o", "name": "GPT-4o", "original_id": "openai/gpt-4o"},
            {"id": "anthropic.claude-opus", "name": "Claude Opus", "original_id": "anthropic/claude-opus"},
        ]

        result = pipe._expand_variant_models(base_models, pipe.valves)

        # Should have 2 base + 2 valid variants (invalid entries skipped)
        assert len(result) == 4
        variant_names = [m["name"] for m in result[2:]]
        assert "GPT-4o Exacto" in variant_names
        assert "Claude Opus Extended" in variant_names

    def test_expand_variant_capitalization(self):
        """Test that variant tags are capitalized in display name."""
        pipe = Pipe()
        pipe.valves.VARIANT_MODELS = "openai/gpt-4o:FREE,anthropic/claude-opus:EXTENDED"

        base_models = [
            {"id": "openai.gpt-4o", "name": "GPT-4o", "original_id": "openai/gpt-4o"},
            {"id": "anthropic.claude-opus", "name": "Claude Opus", "original_id": "anthropic/claude-opus"},
        ]

        result = pipe._expand_variant_models(base_models, pipe.valves)

        # Tags should be lowercase in ID but capitalized in display name
        variant1 = [m for m in result if "Exacto" in m.get("name", "") or "Free" in m.get("name", "")][0]
        variant2 = [m for m in result if "Extended" in m.get("name", "")][0]

        assert variant1["id"] == "openai.gpt-4o:free"  # Lowercase in ID
        assert variant1["name"] == "GPT-4o Free"  # Capitalized in name

        assert variant2["id"] == "anthropic.claude-opus:extended"
        assert variant2["name"] == "Claude Opus Extended"

    def test_expand_variant_preserves_metadata(self):
        """Test that variant models preserve base model metadata."""
        pipe = Pipe()
        pipe.valves.VARIANT_MODELS = "openai/gpt-4o:exacto"

        base_models = [
            {
                "id": "openai.gpt-4o",
                "name": "GPT-4o",
                "original_id": "openai/gpt-4o",
                "norm_id": "openai.gpt-4o",
                "description": "Advanced AI model",
                "pricing": {"prompt": 0.001, "completion": 0.002},
                "context_length": 128000,
            }
        ]

        result = pipe._expand_variant_models(base_models, pipe.valves)

        variant = result[1]
        # Metadata should be preserved (shallow copy)
        assert variant["description"] == "Advanced AI model"
        assert variant["pricing"] == {"prompt": 0.001, "completion": 0.002}
        assert variant["context_length"] == 128000


class TestAPIModelIDPreservation:
    """Test that api_model_id() preserves variant suffixes."""

    def test_api_model_id_without_variant(self):
        """Test normal model ID mapping without variant."""
        # Setup mock registry
        OpenRouterModelRegistry._id_map = {"openai.gpt-4o": "openai/gpt-4o"}

        result = OpenRouterModelRegistry.api_model_id("openai.gpt-4o")
        assert result == "openai/gpt-4o"

    def test_api_model_id_with_variant(self):
        """Test model ID mapping preserves variant suffix."""
        # Setup mock registry
        OpenRouterModelRegistry._id_map = {"openai.gpt-4o": "openai/gpt-4o"}

        result = OpenRouterModelRegistry.api_model_id("openai.gpt-4o:exacto")
        assert result == "openai/gpt-4o:exacto"

    def test_api_model_id_multiple_variants(self):
        """Test various variant suffixes are preserved."""
        OpenRouterModelRegistry._id_map = {
            "openai.gpt-4o": "openai/gpt-4o",
            "anthropic.claude-opus": "anthropic/claude-opus",
            "deepseek.deepseek-r1": "deepseek/deepseek-r1",
        }

        test_cases = [
            ("openai.gpt-4o:free", "openai/gpt-4o:free"),
            ("openai.gpt-4o:nitro", "openai/gpt-4o:nitro"),
            ("anthropic.claude-opus:extended", "anthropic/claude-opus:extended"),
            ("deepseek.deepseek-r1:thinking", "deepseek/deepseek-r1:thinking"),
        ]

        for input_id, expected_output in test_cases:
            result = OpenRouterModelRegistry.api_model_id(input_id)
            assert result == expected_output, f"Failed for {input_id}"

    def test_api_model_id_nonexistent_base(self):
        """Test that non-existent base model returns None even with variant."""
        OpenRouterModelRegistry._id_map = {}

        result = OpenRouterModelRegistry.api_model_id("nonexistent.model:exacto")
        assert result is None


class TestVariantModelsIntegration:
    """Integration tests verifying end-to-end expansion behavior.

    Note: Full pipes() endpoint integration tests are omitted because they require
    complex catalog manager mocking that is brittle and doesn't add significant value
    beyond the comprehensive unit tests above. The unit tests already verify:
    - Expansion logic correctness
    - Metadata inheritance
    - API ID preservation
    - Edge case handling (missing base, empty valve, malformed CSV)

    The expansion is integrated into pipes() via a simple method call (pipe.py:1679),
    which is tested implicitly during manual testing and real usage.
    """

    def test_expansion_workflow(self):
        """Test the expansion workflow with realistic data."""
        pipe = Pipe()
        pipe.valves.VARIANT_MODELS = "deepseek/deepseek-r1:thinking,openai/gpt-4o:exacto"

        # Simulate base models from catalog
        base_models = [
            {
                "id": "deepseek.deepseek-r1",
                "name": "DeepSeek R1",
                "original_id": "deepseek/deepseek-r1",
                "norm_id": "deepseek.deepseek-r1",
                "description": "Advanced reasoning model",
                "pricing": {"prompt": "0.0001", "completion": "0.0002"},
                "context_length": 128000,
            },
            {
                "id": "openai.gpt-4o",
                "name": "GPT-4o",
                "original_id": "openai/gpt-4o",
                "norm_id": "openai.gpt-4o",
                "description": "OpenAI GPT-4o",
                "pricing": {"prompt": "0.005", "completion": "0.015"},
                "context_length": 128000,
            },
        ]

        # Expand variants
        expanded = pipe._expand_variant_models(base_models, pipe.valves)

        # Should have 4 models: 2 base + 2 variants
        assert len(expanded) == 4

        # Verify base models unchanged
        assert expanded[0]["id"] == "deepseek.deepseek-r1"
        assert expanded[1]["id"] == "openai.gpt-4o"

        # Verify variants added
        thinking_variant = next((m for m in expanded if ":thinking" in m["id"]), None)
        exacto_variant = next((m for m in expanded if ":exacto" in m["id"]), None)

        assert thinking_variant is not None
        assert thinking_variant["name"] == "DeepSeek R1 Thinking"
        assert thinking_variant["original_id"] == "deepseek/deepseek-r1"  # Kept as base for icon lookup

        assert exacto_variant is not None
        assert exacto_variant["name"] == "GPT-4o Exacto"
        assert exacto_variant["original_id"] == "openai/gpt-4o"  # Kept as base for icon lookup

    def test_provider_prefix_preservation(self):
        """Test that provider prefixes are preserved in variant names.

        Real OpenRouter model names include provider prefixes like "OpenAI: GPT-5.2".
        Variant models should preserve the exact same format as their base model,
        just appending the variant tag at the end.
        """
        pipe = Pipe()
        pipe.valves.VARIANT_MODELS = "openai/gpt-5.2:online,anthropic/claude-opus:extended"

        # Simulate real OpenRouter model names WITH provider prefixes
        base_models = [
            {
                "id": "openai.gpt-5.2",
                "name": "OpenAI: GPT-5.2",  # Real format from OpenRouter
                "original_id": "openai/gpt-5.2",
                "norm_id": "openai.gpt-5.2",
            },
            {
                "id": "anthropic.claude-opus",
                "name": "Anthropic: Claude Opus",  # Real format from OpenRouter
                "original_id": "anthropic/claude-opus",
                "norm_id": "anthropic.claude-opus",
            },
        ]

        # Expand variants
        expanded = pipe._expand_variant_models(base_models, pipe.valves)

        # Should have 4 models: 2 base + 2 variants
        assert len(expanded) == 4

        # Find variant models
        online_variant = next((m for m in expanded if ":online" in m["id"]), None)
        extended_variant = next((m for m in expanded if ":extended" in m["id"]), None)

        # Verify variant names preserve provider prefix from base model
        assert online_variant is not None
        assert online_variant["name"] == "OpenAI: GPT-5.2 Online"  # Keeps "OpenAI: " prefix

        assert extended_variant is not None
        assert extended_variant["name"] == "Anthropic: Claude Opus Extended"  # Keeps "Anthropic: " prefix


class TestPresetModelsExpansion:
    """Test the _expand_variant_models() method with OpenRouter presets."""

    def test_expand_single_preset(self):
        """Test expanding a single preset model using @ syntax."""
        pipe = Pipe()
        pipe.valves.VARIANT_MODELS = "openai/gpt-4o@preset/email-copywriter"

        base_models = [
            {
                "id": "openai.gpt-4o",
                "name": "GPT-4o",
                "original_id": "openai/gpt-4o",
                "norm_id": "openai.gpt-4o",
            }
        ]

        result = pipe._expand_variant_models(base_models, pipe.valves)

        assert len(result) == 2  # Base + preset
        assert result[0]["name"] == "GPT-4o"  # Base unchanged
        assert result[1]["name"] == "GPT-4o Preset: email-copywriter"  # Preset format
        assert result[1]["id"] == "openai.gpt-4o:preset/email-copywriter"  # Internal uses :
        assert result[1]["original_id"] == "openai/gpt-4o"  # Kept as base for icon lookup

    def test_expand_preset_with_provider_prefix(self):
        """Test preset display name preserves provider prefix."""
        pipe = Pipe()
        pipe.valves.VARIANT_MODELS = "openai/gpt-4o@preset/code-reviewer"

        base_models = [
            {
                "id": "openai.gpt-4o",
                "name": "OpenAI: GPT-4o",  # Real OpenRouter format
                "original_id": "openai/gpt-4o",
                "norm_id": "openai.gpt-4o",
            }
        ]

        result = pipe._expand_variant_models(base_models, pipe.valves)

        assert len(result) == 2
        preset_model = result[1]
        assert preset_model["name"] == "OpenAI: GPT-4o Preset: code-reviewer"

    def test_expand_preset_case_preservation(self):
        """Test that preset slugs preserve case (case-sensitive)."""
        pipe = Pipe()
        pipe.valves.VARIANT_MODELS = "openai/gpt-4o@preset/Email-Copywriter"

        base_models = [
            {
                "id": "openai.gpt-4o",
                "name": "GPT-4o",
                "original_id": "openai/gpt-4o",
                "norm_id": "openai.gpt-4o",
            }
        ]

        result = pipe._expand_variant_models(base_models, pipe.valves)

        preset_model = result[1]
        # Preset slug case is preserved (unlike variants which are lowercased)
        assert preset_model["id"] == "openai.gpt-4o:preset/Email-Copywriter"
        assert preset_model["name"] == "GPT-4o Preset: Email-Copywriter"

    def test_expand_mixed_variants_and_presets(self):
        """Test mixing variants and presets in VARIANT_MODELS."""
        pipe = Pipe()
        pipe.valves.VARIANT_MODELS = "openai/gpt-4o:nitro,openai/gpt-4o@preset/email-copywriter,anthropic/claude-opus:exacto"

        base_models = [
            {
                "id": "openai.gpt-4o",
                "name": "GPT-4o",
                "original_id": "openai/gpt-4o",
                "norm_id": "openai.gpt-4o",
            },
            {
                "id": "anthropic.claude-opus",
                "name": "Claude Opus",
                "original_id": "anthropic/claude-opus",
                "norm_id": "anthropic.claude-opus",
            },
        ]

        result = pipe._expand_variant_models(base_models, pipe.valves)

        # Should have 5 models: 2 base + 2 variants + 1 preset
        assert len(result) == 5

        # Find by ID suffix
        nitro = next((m for m in result if ":nitro" in m["id"]), None)
        preset = next((m for m in result if ":preset/" in m["id"]), None)
        exacto = next((m for m in result if ":exacto" in m["id"]), None)

        assert nitro is not None
        assert nitro["name"] == "GPT-4o Nitro"  # Variant format

        assert preset is not None
        assert preset["name"] == "GPT-4o Preset: email-copywriter"  # Preset format

        assert exacto is not None
        assert exacto["name"] == "Claude Opus Exacto"  # Variant format

    def test_expand_preset_missing_base(self):
        """Test preset with non-existent base model (should skip)."""
        pipe = Pipe()
        pipe.valves.VARIANT_MODELS = "nonexistent/model@preset/my-preset"

        base_models = []
        result = pipe._expand_variant_models(base_models, pipe.valves)

        assert len(result) == 0  # No presets added


class TestAPIModelIDPresetConversion:
    """Test that api_model_id() converts :preset/ to @preset/ for API calls."""

    def test_api_model_id_with_preset(self):
        """Test preset model ID uses @ separator for API."""
        OpenRouterModelRegistry._id_map = {"openai.gpt-4o": "openai/gpt-4o"}

        result = OpenRouterModelRegistry.api_model_id("openai.gpt-4o:preset/email-copywriter")
        assert result == "openai/gpt-4o@preset/email-copywriter"  # @ for presets

    def test_api_model_id_variant_unchanged(self):
        """Test variant model ID still uses : separator for API."""
        OpenRouterModelRegistry._id_map = {"openai.gpt-4o": "openai/gpt-4o"}

        result = OpenRouterModelRegistry.api_model_id("openai.gpt-4o:exacto")
        assert result == "openai/gpt-4o:exacto"  # : for variants

    def test_api_model_id_preset_case_preservation(self):
        """Test preset slug case is preserved in API model ID."""
        OpenRouterModelRegistry._id_map = {"openai.gpt-4o": "openai/gpt-4o"}

        result = OpenRouterModelRegistry.api_model_id("openai.gpt-4o:preset/Email-Copywriter")
        assert result == "openai/gpt-4o@preset/Email-Copywriter"  # Case preserved

    def test_api_model_id_multiple_presets(self):
        """Test various preset conversions."""
        OpenRouterModelRegistry._id_map = {
            "openai.gpt-4o": "openai/gpt-4o",
            "anthropic.claude-opus": "anthropic/claude-opus",
        }

        test_cases = [
            ("openai.gpt-4o:preset/email-copywriter", "openai/gpt-4o@preset/email-copywriter"),
            ("openai.gpt-4o:preset/code-reviewer", "openai/gpt-4o@preset/code-reviewer"),
            ("anthropic.claude-opus:preset/inbound-classifier", "anthropic/claude-opus@preset/inbound-classifier"),
        ]

        for input_id, expected_output in test_cases:
            result = OpenRouterModelRegistry.api_model_id(input_id)
            assert result == expected_output, f"Failed for {input_id}"

    def test_api_model_id_nonexistent_preset_base(self):
        """Test that non-existent base model returns None even with preset."""
        OpenRouterModelRegistry._id_map = {}

        result = OpenRouterModelRegistry.api_model_id("nonexistent.model:preset/my-preset")
        assert result is None
