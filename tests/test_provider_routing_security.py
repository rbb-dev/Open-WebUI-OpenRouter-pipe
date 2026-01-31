"""Security and integration tests for the provider routing filter generation feature.

These tests verify:
1. String escaping for Python literal interpolation
2. Provider name/slug validation and sanitization
3. Filter source code validation
4. Defense-in-depth against code injection
5. Generated inlet() method correctly reads from __user__["valves"]
6. Generated filter code is valid and executable
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false

from __future__ import annotations

import ast
import json
import re

import pytest

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.pipe import (
    _PROVIDER_SLUG_PATTERN,
    _QUANTIZATION_PATTERN,
)


class TestSafeLiteralString:
    """Tests for Pipe._safe_literal_string()."""

    def test_simple_string(self):
        """Simple strings should be escaped with quotes."""
        result = Pipe._safe_literal_string("hello")
        assert result in ("'hello'", '"hello"')

    def test_string_with_single_quote(self):
        """Strings with single quotes should use double quotes."""
        result = Pipe._safe_literal_string("it's")
        assert result == '"it\'s"'

    def test_string_with_double_quote(self):
        """Strings with double quotes should be properly escaped."""
        result = Pipe._safe_literal_string('say "hello"')
        # Either escaped or use single quotes
        assert '"' not in result[1:-1] or '\\"' in result or result.startswith("'")

    def test_injection_attempt(self):
        """Injection attempts should be safely escaped."""
        malicious = "'; import os; os.system('rm -rf /'); x='"
        result = Pipe._safe_literal_string(malicious)
        # Result should be a valid Python literal
        parsed = ast.literal_eval(result)
        assert parsed == malicious  # The dangerous string is just data

    def test_empty_string(self):
        """Empty string should return empty quoted string."""
        result = Pipe._safe_literal_string("")
        assert result in ("''", '""')

    def test_none_input(self):
        """None input should return empty quoted string."""
        result = Pipe._safe_literal_string(None)
        assert result in ("''", '""')

    def test_non_string_input(self):
        """Non-string input should be converted to string."""
        result = Pipe._safe_literal_string(123)
        assert result in ("'123'", '"123"')

    def test_unicode_characters(self):
        """Unicode characters should be handled safely."""
        result = Pipe._safe_literal_string("café ☕")
        parsed = ast.literal_eval(result)
        assert parsed == "café ☕"

    def test_backslash_escaping(self):
        """Backslashes should be properly escaped."""
        result = Pipe._safe_literal_string("path\\to\\file")
        parsed = ast.literal_eval(result)
        assert parsed == "path\\to\\file"


class TestValidateProviderName:
    """Tests for Pipe._validate_provider_name()."""

    def test_simple_name(self):
        """Simple provider names should pass through."""
        assert Pipe._validate_provider_name("Amazon Bedrock") == "Amazon Bedrock"

    def test_removes_special_characters(self):
        """Special characters should be removed."""
        result = Pipe._validate_provider_name("Evil<script>Provider")
        assert "<" not in result
        assert ">" not in result
        assert "script" in result.lower()

    def test_removes_quotes(self):
        """Quote characters should be removed."""
        result = Pipe._validate_provider_name('Test"Provider\'Name')
        assert '"' not in result
        assert "'" not in result

    def test_collapses_whitespace(self):
        """Multiple spaces/underscores should be collapsed."""
        result = Pipe._validate_provider_name("Too    Many   Spaces")
        assert "    " not in result

    def test_empty_input(self):
        """Empty input should return 'Unknown'."""
        assert Pipe._validate_provider_name("") == "Unknown"
        assert Pipe._validate_provider_name(None) == "Unknown"
        assert Pipe._validate_provider_name("   ") == "Unknown"

    def test_truncation_with_hash(self):
        """Long names should be truncated with hash suffix."""
        long_name = "A" * 100
        result = Pipe._validate_provider_name(long_name, slug="test-slug")
        assert len(result) <= 64
        assert "_" in result  # Hash suffix added

    def test_truncation_preserves_uniqueness(self):
        """Different long names should have different hashes."""
        name1 = "A" * 100
        name2 = "B" * 100
        result1 = Pipe._validate_provider_name(name1, slug="slug1")
        result2 = Pipe._validate_provider_name(name2, slug="slug2")
        assert result1 != result2

    def test_injection_in_name(self):
        """Injection attempts should be sanitized."""
        malicious = '"; import os; #'
        result = Pipe._validate_provider_name(malicious)
        assert "import" in result  # Word preserved but quotes removed
        assert '"' not in result
        assert ";" not in result


class TestValidateFilterSource:
    """Tests for Pipe._validate_filter_source()."""

    def test_valid_python(self):
        """Valid Python code should pass."""
        source = "x = 1\ny = 2\nprint(x + y)"
        is_valid, error = Pipe._validate_filter_source(source)
        assert is_valid is True
        assert error is None

    def test_syntax_error(self):
        """Invalid Python syntax should fail."""
        source = "def broken(:\n    pass"
        is_valid, error = Pipe._validate_filter_source(source)
        assert is_valid is False
        assert error is not None
        assert "Line" in error or "syntax" in error.lower()

    def test_empty_source(self):
        """Empty source should fail."""
        is_valid, error = Pipe._validate_filter_source("")
        assert is_valid is False
        assert error is not None and ("Empty" in error or "invalid" in error.lower())

    def test_none_source(self):
        """None source should fail."""
        is_valid, error = Pipe._validate_filter_source(None)
        assert is_valid is False

    def test_valid_class_definition(self):
        """Valid class definition should pass."""
        source = '''
class Filter:
    def __init__(self):
        self.x = 1

    def inlet(self, body):
        return body
'''
        is_valid, error = Pipe._validate_filter_source(source)
        assert is_valid is True


class TestProviderSlugPattern:
    """Tests for the _PROVIDER_SLUG_PATTERN regex."""

    @pytest.mark.parametrize("slug", [
        "openai",
        "amazon-bedrock",
        "google-vertex",
        "together",
        "deepinfra",
        "anthropic",
        "mistral",
        "openai/gpt-4o",  # With segment
        "google-vertex/us",
    ])
    def test_valid_slugs(self, slug):
        """Valid provider slugs should match."""
        assert _PROVIDER_SLUG_PATTERN.match(slug) is not None

    @pytest.mark.parametrize("slug", [
        "OPENAI",  # Uppercase
        "Amazon Bedrock",  # Spaces
        "amazon_bedrock",  # Underscore
        'evil"inject',  # Quote
        "test;drop",  # Semicolon
        "",  # Empty
        "test\nline",  # Newline
    ])
    def test_invalid_slugs(self, slug):
        """Invalid provider slugs should not match."""
        assert _PROVIDER_SLUG_PATTERN.match(slug) is None


class TestQuantizationPattern:
    """Tests for the _QUANTIZATION_PATTERN regex."""

    @pytest.mark.parametrize("quant", [
        "int4",
        "int8",
        "fp8",
        "fp16",
        "bf16",
        "unknown",
        "INT4",  # Uppercase allowed
        "fp-8",  # Hyphen allowed
    ])
    def test_valid_quantizations(self, quant):
        """Valid quantization values should match."""
        assert _QUANTIZATION_PATTERN.match(quant) is not None

    @pytest.mark.parametrize("quant", [
        "int 4",  # Space
        'fp8"',  # Quote
        "bf16;",  # Semicolon
        "",  # Empty
    ])
    def test_invalid_quantizations(self, quant):
        """Invalid quantization values should not match."""
        assert _QUANTIZATION_PATTERN.match(quant) is None


class TestJsonDumpsEscaping:
    """Tests verifying json.dumps provides correct escaping for double-quoted contexts."""

    def test_double_quote_escaped(self):
        """Double quotes in input should be escaped."""
        malicious = 'test"injection'
        escaped = json.dumps(malicious)[1:-1]
        assert '\\"' in escaped or '"' not in escaped

    def test_backslash_escaped(self):
        """Backslashes should be escaped."""
        input_str = "test\\path"
        escaped = json.dumps(input_str)[1:-1]
        assert "\\\\" in escaped

    def test_newline_escaped(self):
        """Newlines should be escaped."""
        input_str = "line1\nline2"
        escaped = json.dumps(input_str)[1:-1]
        assert "\\n" in escaped
        assert "\n" not in escaped

    def test_injection_prevented(self):
        """Code injection via quotes should be prevented."""
        malicious = '"; import os; os.system("id"); x="'
        escaped = json.dumps(malicious)[1:-1]

        # Construct the template output
        template = f'MODEL_SLUG = "{escaped}"'

        # Parse it - should be valid Python with the malicious code as a string literal
        tree = ast.parse(template)

        # Should be a single assignment, not multiple statements
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.Assign)

    def test_unicode_preserved(self):
        """Unicode characters should be preserved or safely escaped."""
        input_str = "café"
        escaped = json.dumps(input_str)[1:-1]
        # Either preserved or escaped, but recoverable
        recovered = json.loads(f'"{escaped}"')
        assert recovered == input_str


class TestRenderProviderRoutingFilterSourceSecurity:
    """Integration tests for security in _render_provider_routing_filter_source."""

    def test_malicious_model_slug(self):
        """Malicious model slugs should be safely escaped."""
        # This would break out of a string if not escaped
        malicious_slug = 'openai/gpt-4"; import os; os.system("id"); x="'

        source = Pipe._render_provider_routing_filter_source(
            model_slug=malicious_slug,
            providers=["openai"],
            quantizations=["fp16"],
            visibility="user",
        )

        # The generated source should be syntactically valid
        is_valid, error = Pipe._validate_filter_source(source)
        assert is_valid is True, f"Generated source is invalid: {error}"

        # Parse and check for injection
        tree = ast.parse(source)

        # Count import statements - should only be standard library imports
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        import_names = []
        for imp in imports:
            if isinstance(imp, ast.Import):
                import_names.extend(alias.name for alias in imp.names)
            else:
                import_names.append(imp.module or "")

        # Should not have 'os' import from injection
        assert "os" not in import_names

    def test_invalid_providers_filtered(self):
        """Invalid provider slugs should be filtered out."""
        providers = ["openai", 'INVALID"INJECT', "together", "has space"]

        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=providers,
            quantizations=["fp16"],
            visibility="user",
        )

        # Source should be valid
        is_valid, _ = Pipe._validate_filter_source(source)
        assert is_valid is True

        # Should only contain valid providers
        assert "openai" in source
        assert "together" in source
        assert "INVALID" not in source
        assert "has space" not in source

    def test_all_invalid_providers(self):
        """When all providers are invalid, should have empty _PROVIDER_MAP."""
        providers = ['INVALID"', "HAS SPACE", "BAD;CHAR"]

        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=providers,
            quantizations=[],
            visibility="user",
        )

        is_valid, _ = Pipe._validate_filter_source(source)
        assert is_valid is True
        # When all providers are invalid, _PROVIDER_MAP should be empty
        assert "_PROVIDER_MAP: dict[str, str] = {}" in source

    def test_empty_model_slug_raises(self):
        """Empty model slug should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            Pipe._render_provider_routing_filter_source(
                model_slug="",
                providers=["openai"],
                quantizations=[],
                visibility="user",
            )


class TestGeneratedInletMethod:
    """Integration tests for the generated inlet() method behavior.

    Note: These tests check the generated source code structure without executing it,
    since the generated code imports open_webui modules not available in test env.
    """

    def test_inlet_reads_from_user_valves(self):
        """Generated inlet should read from __user__['valves'], not self.user_valves."""
        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=["openai", "anthropic"],
            quantizations=["fp16"],
            visibility="user",
        )

        # The generated code should reference __user__.get("valves")
        assert '__user__.get("valves")' in source or "__user__.get('valves')" in source

        # The inlet method body (after "def inlet") should use user_valves from __user__
        # NOT self.user_valves which only holds defaults
        inlet_section = source.split("def inlet(")[1] if "def inlet(" in source else ""
        # Should have the extraction pattern
        assert "user_valves = __user__" in inlet_section

    def test_generated_filter_has_correct_structure(self):
        """Generated filter code should have correct class structure."""
        source = Pipe._render_provider_routing_filter_source(
            model_slug="openai/gpt-4o",
            providers=["openai", "azure", "together"],
            quantizations=["fp16", "bf16"],
            visibility="both",
        )

        # Check structure via string matching (avoids import issues)
        assert "class Filter:" in source
        assert "class Valves(BaseModel):" in source
        assert "class UserValves(BaseModel):" in source
        assert "toggle = True" in source  # 'both' visibility should be toggleable
        assert "def inlet(" in source
        assert "def __init__(" in source

    def test_generated_filter_includes_max_price_extensions(self):
        """Generated filter should include extended max_price valves and injection."""
        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=["openai"],
            quantizations=["fp16"],
            visibility="both",
        )

        assert "MAX_PRICE_IMAGE" in source
        assert "MAX_PRICE_AUDIO" in source
        assert "MAX_PRICE_REQUEST" in source
        assert 'max_price_image = get_float("MAX_PRICE_IMAGE")' in source
        assert 'max_price_audio = get_float("MAX_PRICE_AUDIO")' in source
        assert 'max_price_request = get_float("MAX_PRICE_REQUEST")' in source
        assert 'provider["max_price"]["image"] = max_price_image' in source
        assert 'provider["max_price"]["audio"] = max_price_audio' in source
        assert 'provider["max_price"]["request"] = max_price_request' in source

    def test_admin_only_filter_structure(self):
        """Admin-only filters should have toggle=False and no UserValves."""
        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=["openai"],
            quantizations=[],
            visibility="admin",
        )

        assert "toggle = False" in source
        assert "class Valves(BaseModel):" in source
        assert "class UserValves" not in source

    def test_user_only_filter_structure(self):
        """User-only filters should have toggle=True and no Valves."""
        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=["openai"],
            quantizations=[],
            visibility="user",
        )

        assert "toggle = True" in source
        assert "class UserValves(BaseModel):" in source
        # Check that Valves class is NOT present (but UserValves is)
        lines = source.split("\n")
        has_admin_valves = any("class Valves(BaseModel):" in line for line in lines)
        assert not has_admin_valves

    def test_both_visibility_has_both_valves(self):
        """Both visibility should have both Valves and UserValves."""
        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=["openai"],
            quantizations=[],
            visibility="both",
        )

        assert "class Valves(BaseModel):" in source
        assert "class UserValves(BaseModel):" in source
        assert "toggle = True" in source


class TestGeneratedInletLogic:
    """Tests for the _generate_inlet_logic helper."""

    def test_admin_visibility_logic(self):
        """Admin visibility should only reference self.valves."""
        logic = Pipe._generate_inlet_logic("admin")
        assert "self.valves" in logic
        assert "user_valves" not in logic

    def test_user_visibility_logic(self):
        """User visibility should only reference user_valves from __user__."""
        logic = Pipe._generate_inlet_logic("user")
        assert "user_valves" in logic
        # Should extract from __user__, not self.user_valves
        assert '__user__.get("valves")' in logic or "__user__.get('valves')" in logic

    def test_both_visibility_logic(self):
        """Both visibility should reference both sources with user override."""
        logic = Pipe._generate_inlet_logic("both")
        assert "self.valves" in logic
        assert "user_valves" in logic
        assert '__user__.get("valves")' in logic or "__user__.get('valves')" in logic


class TestOrderFieldPermutations:
    """Tests for the ORDER field with provider permutations."""

    def test_order_field_in_valves(self):
        """ORDER field should be present in generated Valves class."""
        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=["openai", "azure"],
            provider_names={"openai": "OpenAI", "azure": "Azure"},
            quantizations=[],
            visibility="admin",
        )
        assert "ORDER: Literal[" in source

    def test_order_field_in_user_valves(self):
        """ORDER field should be present in generated UserValves class."""
        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=["openai", "azure"],
            provider_names={"openai": "OpenAI", "azure": "Azure"},
            quantizations=[],
            visibility="user",
        )
        assert "ORDER: Literal[" in source

    def test_order_map_generated(self):
        """_ORDER_MAP should be generated with provider slug mappings."""
        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=["openai", "azure"],
            provider_names={"openai": "OpenAI", "azure": "Azure"},
            quantizations=[],
            visibility="user",
        )
        assert "_ORDER_MAP: dict[str, list[str]]" in source

    def test_two_providers_two_permutations(self):
        """2 providers should generate 2! = 2 full permutations (all providers in each)."""
        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=["openai", "azure"],
            provider_names={"openai": "OpenAI", "azure": "Azure"},
            quantizations=[],
            visibility="user",
        )
        # Both full permutations with ALL providers
        assert "OpenAI > Azure" in source
        assert "Azure > OpenAI" in source
        # Should NOT have single-provider entries (no partial permutations)
        assert "'OpenAI': [" not in source
        assert "'Azure': [" not in source

    def test_three_providers_six_permutations(self):
        """3 providers should generate 3! = 6 full permutations (all providers in each)."""
        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=["openai", "azure", "together"],
            provider_names={"openai": "OpenAI", "azure": "Azure", "together": "Together"},
            quantizations=[],
            visibility="user",
        )
        # All 6 full permutations (3! = 6, each with all 3 providers)
        perms = [
            "OpenAI > Azure > Together",
            "OpenAI > Together > Azure",
            "Azure > OpenAI > Together",
            "Azure > Together > OpenAI",
            "Together > OpenAI > Azure",
            "Together > Azure > OpenAI",
        ]
        for perm in perms:
            assert perm in source, f"Missing permutation: {perm}"

    def test_order_map_contains_full_slug_lists(self):
        """_ORDER_MAP should map display names to full lists of ALL slugs."""
        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=["openai", "azure"],
            provider_names={"openai": "OpenAI", "azure": "Azure"},
            quantizations=[],
            visibility="user",
        )
        # Full permutations map to lists with ALL provider slugs
        assert "'OpenAI > Azure': ['openai', 'azure']" in source
        assert "'Azure > OpenAI': ['azure', 'openai']" in source

    def test_order_handling_in_inlet(self):
        """Inlet logic should handle ORDER field via _ORDER_MAP lookup."""
        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=["openai", "azure"],
            provider_names={"openai": "OpenAI", "azure": "Azure"},
            quantizations=[],
            visibility="user",
        )
        # Should look up ORDER value in _ORDER_MAP
        assert 'order_display = get_literal("ORDER")' in source
        assert "_ORDER_MAP.get(order_display)" in source
        assert 'provider["order"] = order_slugs' in source

    def test_order_is_first_field(self):
        """ORDER should be the first field in Valves (matching OpenRouter API docs)."""
        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=["openai", "azure"],
            provider_names={"openai": "OpenAI", "azure": "Azure"},
            quantizations=[],
            visibility="both",
        )
        # Check that ORDER appears before other fields
        valves_start = source.find("class Valves(BaseModel):")
        assert valves_start != -1
        order_pos = source.find("ORDER:", valves_start)
        allow_fallbacks_pos = source.find("ALLOW_FALLBACKS:", valves_start)
        assert order_pos < allow_fallbacks_pos, "ORDER should come before ALLOW_FALLBACKS"

        # Same for UserValves
        user_valves_start = source.find("class UserValves(BaseModel):")
        assert user_valves_start != -1
        order_pos_user = source.find("ORDER:", user_valves_start)
        allow_fallbacks_pos_user = source.find("ALLOW_FALLBACKS:", user_valves_start)
        assert order_pos_user < allow_fallbacks_pos_user, "ORDER should come before ALLOW_FALLBACKS in UserValves"

    def test_generated_filter_is_valid_python(self):
        """Generated filter with ORDER should be valid Python syntax."""
        source = Pipe._render_provider_routing_filter_source(
            model_slug="test/model",
            providers=["openai", "azure", "together"],
            provider_names={"openai": "OpenAI", "azure": "Azure", "together": "Together"},
            quantizations=["fp16", "bf16"],
            visibility="both",
        )
        is_valid, error = Pipe._validate_filter_source(source)
        assert is_valid is True, f"Generated source is invalid: {error}"
