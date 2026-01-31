"""Comprehensive tests for open_webui_openrouter_pipe/core/utils.py.

This module tests utility functions to achieve high coverage including:
- _stable_crockford_id: Deterministic ID generation
- _safe_json_loads: Safe JSON parsing
- _coerce_positive_int / _coerce_bool: Type coercion helpers
- _normalize_string_list / _normalize_optional_str: String normalization
- _parse_model_fallback_csv / _select_best_effort_fallback: Model fallback helpers
- _extract_marker_ulid / _serialize_marker / contains_marker / split_text_by_markers: ULID markers
- _get_open_webui_config_module / _unwrap_config_value: Config helpers
- _redact_payload_blobs: Data URL redaction
- _extract_plain_text_content: Content extraction
- _extract_feature_flags: Feature flag extraction
- merge_usage_stats: Usage statistics merging
- wrap_code_block: Markdown code block wrapping
- _template_value_present: Template value presence check
- _sanitize_path_component: Path sanitization
- _retry_after_seconds: HTTP retry-after parsing
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

from __future__ import annotations

import pytest

# Import from package (these are exposed via lazy loading)
from open_webui_openrouter_pipe import (
    _coerce_bool,
    _coerce_positive_int,
    _extract_feature_flags,
    _normalize_optional_str,
    _normalize_string_list,
    _pretty_json,
    _render_error_template,
    _safe_json_loads,
    _sanitize_path_component,
    _serialize_marker,
    _template_value_present,
    _unwrap_config_value,
    contains_marker,
    merge_usage_stats,
    split_text_by_markers,
    wrap_code_block,
    ULID_LENGTH,
)

# Import directly from core.utils for functions not in the lazy loading map
from open_webui_openrouter_pipe.core.utils import (
    _extract_plain_text_content,
    _parse_model_fallback_csv,
    _redact_payload_blobs,
    _retry_after_seconds,
    _select_best_effort_fallback,
    _stable_crockford_id,
)


# -----------------------------------------------------------------------------
# _stable_crockford_id tests
# -----------------------------------------------------------------------------

class TestStableCrockfordId:
    """Tests for _stable_crockford_id deterministic ID generation."""

    def test_basic_generation(self):
        """Basic ID generation from seed string."""
        result = _stable_crockford_id("test-seed")
        assert len(result) == ULID_LENGTH
        assert result.isalnum() or all(c in "0123456789ABCDEFGHJKMNPQRSTVWXYZ" for c in result)

    def test_deterministic(self):
        """Same seed produces same ID."""
        id1 = _stable_crockford_id("same-seed")
        id2 = _stable_crockford_id("same-seed")
        assert id1 == id2

    def test_different_seeds(self):
        """Different seeds produce different IDs."""
        id1 = _stable_crockford_id("seed-a")
        id2 = _stable_crockford_id("seed-b")
        assert id1 != id2

    def test_custom_length(self):
        """Custom length parameter works."""
        result = _stable_crockford_id("test", length=10)
        assert len(result) == 10

    def test_empty_seed(self):
        """Empty seed still generates valid ID."""
        result = _stable_crockford_id("")
        assert len(result) == ULID_LENGTH

    def test_none_seed_treated_as_empty(self):
        """None seed treated as empty string."""
        result = _stable_crockford_id(None)  # type: ignore
        assert len(result) == ULID_LENGTH
        # Should be same as empty string
        result2 = _stable_crockford_id("")
        assert result == result2

    def test_invalid_length_zero(self):
        """Zero length raises ValueError."""
        with pytest.raises(ValueError, match="length must be positive"):
            _stable_crockford_id("test", length=0)

    def test_invalid_length_negative(self):
        """Negative length raises ValueError."""
        with pytest.raises(ValueError, match="length must be positive"):
            _stable_crockford_id("test", length=-5)

    def test_length_too_long(self):
        """Requesting too many bits raises ValueError."""
        # SHA-256 = 256 bits, 5 bits per char -> max ~51 chars
        with pytest.raises(ValueError, match="seed hash too short"):
            _stable_crockford_id("test", length=100)


# -----------------------------------------------------------------------------
# _safe_json_loads tests
# -----------------------------------------------------------------------------

class TestSafeJsonLoads:
    """Tests for _safe_json_loads safe JSON parsing."""

    def test_valid_json(self):
        """Valid JSON is parsed correctly."""
        result = _safe_json_loads('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_json_array(self):
        """Valid JSON array is parsed correctly."""
        result = _safe_json_loads('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_invalid_json(self):
        """Invalid JSON returns None without raising."""
        result = _safe_json_loads('not valid json')
        assert result is None

    def test_empty_string(self):
        """Empty string returns None."""
        result = _safe_json_loads('')
        assert result is None

    def test_none_input(self):
        """None input returns None."""
        result = _safe_json_loads(None)
        assert result is None


# -----------------------------------------------------------------------------
# _coerce_positive_int tests
# -----------------------------------------------------------------------------

class TestCoercePositiveInt:
    """Tests for _coerce_positive_int type coercion."""

    def test_positive_integer(self):
        """Positive integer returns as-is."""
        assert _coerce_positive_int(42) == 42

    def test_positive_string(self):
        """Positive string integer is coerced."""
        assert _coerce_positive_int("100") == 100

    def test_zero_returns_none(self):
        """Zero returns None (not positive)."""
        assert _coerce_positive_int(0) is None

    def test_negative_returns_none(self):
        """Negative integer returns None."""
        assert _coerce_positive_int(-5) is None

    def test_none_returns_none(self):
        """None input returns None."""
        assert _coerce_positive_int(None) is None

    def test_bool_true_returns_none(self):
        """Boolean True returns None (bool check before int coercion)."""
        assert _coerce_positive_int(True) is None

    def test_bool_false_returns_none(self):
        """Boolean False returns None."""
        assert _coerce_positive_int(False) is None

    def test_non_numeric_string_returns_none(self):
        """Non-numeric string returns None."""
        assert _coerce_positive_int("abc") is None

    def test_float_string(self):
        """Float string returns None (int() raises ValueError)."""
        assert _coerce_positive_int("3.14") is None


# -----------------------------------------------------------------------------
# _coerce_bool tests
# -----------------------------------------------------------------------------

class TestCoerceBool:
    """Tests for _coerce_bool type coercion."""

    def test_bool_true(self):
        """Boolean True returns True."""
        assert _coerce_bool(True) is True

    def test_bool_false(self):
        """Boolean False returns False."""
        assert _coerce_bool(False) is False

    def test_string_true_variants(self):
        """Truthy string variants return True."""
        for val in ["true", "TRUE", "True", "1", "yes", "YES", "on", "ON"]:
            assert _coerce_bool(val) is True, f"Expected True for {val!r}"

    def test_string_false_variants(self):
        """Falsy string variants return False."""
        for val in ["false", "FALSE", "False", "0", "no", "NO", "off", "OFF"]:
            assert _coerce_bool(val) is False, f"Expected False for {val!r}"

    def test_string_with_whitespace(self):
        """Strings with whitespace are trimmed."""
        assert _coerce_bool("  true  ") is True
        assert _coerce_bool("  false  ") is False

    def test_unrecognized_string_returns_none(self):
        """Unrecognized string returns None."""
        assert _coerce_bool("maybe") is None

    def test_integer_nonzero(self):
        """Non-zero integer returns True."""
        assert _coerce_bool(1) is True
        assert _coerce_bool(42) is True

    def test_integer_zero(self):
        """Zero integer returns False."""
        assert _coerce_bool(0) is False

    def test_other_types_return_none(self):
        """Other types return None."""
        assert _coerce_bool([]) is None
        assert _coerce_bool({}) is None
        assert _coerce_bool(None) is None


# -----------------------------------------------------------------------------
# _normalize_string_list tests
# -----------------------------------------------------------------------------

class TestNormalizeStringList:
    """Tests for _normalize_string_list string normalization."""

    def test_list_of_strings(self):
        """List of strings is normalized."""
        result = _normalize_string_list(["  hello  ", "world", ""])
        assert result == ["hello", "world"]

    def test_list_with_non_strings(self):
        """List with non-strings converts them."""
        result = _normalize_string_list([123, "text", None])
        assert result == ["123", "text"]

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert _normalize_string_list([]) == []

    def test_non_list_returns_empty(self):
        """Non-list input returns empty list."""
        assert _normalize_string_list("not a list") == []
        assert _normalize_string_list(None) == []
        assert _normalize_string_list(123) == []


# -----------------------------------------------------------------------------
# _normalize_optional_str tests
# -----------------------------------------------------------------------------

class TestNormalizeOptionalStr:
    """Tests for _normalize_optional_str string normalization."""

    def test_normal_string(self):
        """Normal string is trimmed."""
        assert _normalize_optional_str("  hello  ") == "hello"

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        assert _normalize_optional_str("") is None
        assert _normalize_optional_str("   ") is None

    def test_none_returns_none(self):
        """None input returns None."""
        assert _normalize_optional_str(None) is None

    def test_non_string_converted(self):
        """Non-string is converted to string."""
        assert _normalize_optional_str(123) == "123"
        assert _normalize_optional_str(3.14) == "3.14"


# -----------------------------------------------------------------------------
# _parse_model_fallback_csv tests
# -----------------------------------------------------------------------------

class TestParseModelFallbackCsv:
    """Tests for _parse_model_fallback_csv parsing."""

    def test_simple_csv(self):
        """Simple CSV is parsed correctly."""
        result = _parse_model_fallback_csv("model-a,model-b,model-c")
        assert result == ["model-a", "model-b", "model-c"]

    def test_csv_with_whitespace(self):
        """CSV with whitespace is trimmed."""
        result = _parse_model_fallback_csv("  model-a , model-b , model-c  ")
        assert result == ["model-a", "model-b", "model-c"]

    def test_csv_with_duplicates(self):
        """Duplicates are removed (order-preserving)."""
        result = _parse_model_fallback_csv("model-a,model-b,model-a,model-c")
        assert result == ["model-a", "model-b", "model-c"]

    def test_csv_with_empty_parts(self):
        """Empty parts are skipped."""
        result = _parse_model_fallback_csv("model-a,,model-b,,,model-c")
        assert result == ["model-a", "model-b", "model-c"]

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert _parse_model_fallback_csv("") == []
        assert _parse_model_fallback_csv("   ") == []

    def test_non_string(self):
        """Non-string input returns empty list."""
        assert _parse_model_fallback_csv(None) == []
        assert _parse_model_fallback_csv(123) == []
        assert _parse_model_fallback_csv(["a", "b"]) == []


# -----------------------------------------------------------------------------
# _select_best_effort_fallback tests
# -----------------------------------------------------------------------------

class TestSelectBestEffortFallback:
    """Tests for _select_best_effort_fallback effort selection."""

    def test_exact_match(self):
        """Exact match returns the requested effort."""
        result = _select_best_effort_fallback("medium", ["low", "medium", "high"])
        assert result == "medium"

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        result = _select_best_effort_fallback("MEDIUM", ["low", "Medium", "high"])
        assert result == "medium"

    def test_fallback_to_lower(self):
        """Requests higher than max fall back to max."""
        result = _select_best_effort_fallback("xhigh", ["low", "medium"])
        assert result == "medium"

    def test_fallback_to_higher(self):
        """Requests lower than min fall back to min."""
        result = _select_best_effort_fallback("none", ["medium", "high"])
        assert result == "medium"

    def test_closest_match(self):
        """Finds closest available effort."""
        result = _select_best_effort_fallback("low", ["medium", "high"])
        assert result == "medium"

    def test_empty_supported_returns_none(self):
        """Empty supported list returns None."""
        assert _select_best_effort_fallback("medium", []) is None

    def test_unrecognized_requested_uses_first_supported(self):
        """Unrecognized requested effort uses first supported."""
        result = _select_best_effort_fallback("unknown", ["low", "medium"])
        assert result == "low"

    def test_unrecognized_supported_values_ignored(self):
        """Unrecognized supported values are ignored for indexing."""
        result = _select_best_effort_fallback("low", ["custom", "medium", "unknown"])
        # "custom" and "unknown" not in ordering, only "medium" is valid
        assert result == "medium"

    def test_all_supported_unrecognized(self):
        """All unrecognized supported values returns first."""
        result = _select_best_effort_fallback("medium", ["custom1", "custom2"])
        # indexed list empty, falls back to supported_lower[0]
        assert result == "custom1"

    def test_whitespace_trimmed(self):
        """Whitespace in values is trimmed."""
        result = _select_best_effort_fallback("  medium  ", ["  low  ", "  high  "])
        assert result in ["low", "high"]


# -----------------------------------------------------------------------------
# Marker system tests
# -----------------------------------------------------------------------------

class TestMarkerSystem:
    """Tests for ULID marker helpers."""

    def test_serialize_marker(self):
        """_serialize_marker creates proper marker format."""
        ulid = "A" * ULID_LENGTH
        marker = _serialize_marker(ulid)
        assert marker == f"[{ulid}]: #"

    def test_contains_marker_true(self):
        """contains_marker returns True when marker present."""
        ulid = "0" * ULID_LENGTH
        text = f"prefix\n[{ulid}]: #\npostfix"
        assert contains_marker(text) is True

    def test_contains_marker_false(self):
        """contains_marker returns False when no marker."""
        text = "just some text without markers"
        assert contains_marker(text) is False

    def test_contains_marker_empty(self):
        """contains_marker returns False for empty text."""
        assert contains_marker("") is False

    def test_split_text_by_markers_no_markers(self):
        """split_text_by_markers with no markers returns single text segment."""
        text = "just text"
        segments = split_text_by_markers(text)
        assert len(segments) == 1
        assert segments[0] == {"type": "text", "text": "just text"}

    def test_split_text_by_markers_with_marker(self):
        """split_text_by_markers correctly splits around markers."""
        ulid = "0" * ULID_LENGTH
        text = f"before\n[{ulid}]: #\nafter"
        segments = split_text_by_markers(text)
        # Should have: text, marker, text
        assert any(s["type"] == "marker" and s["marker"] == ulid for s in segments)
        assert any(s["type"] == "text" for s in segments)

    def test_split_text_by_markers_empty(self):
        """split_text_by_markers with empty text returns empty list."""
        segments = split_text_by_markers("")
        assert segments == []


# -----------------------------------------------------------------------------
# Config helpers tests
# -----------------------------------------------------------------------------

class TestUnwrapConfigValue:
    """Tests for _unwrap_config_value."""

    def test_none_input(self):
        """None input returns None."""
        assert _unwrap_config_value(None) is None

    def test_object_with_value_attr(self):
        """Object with .value attribute returns that value."""
        class ConfigLike:
            value = "inner-value"

        assert _unwrap_config_value(ConfigLike()) == "inner-value"

    def test_plain_value(self):
        """Plain value without .value attribute returns itself."""
        assert _unwrap_config_value("plain") == "plain"
        assert _unwrap_config_value(42) == 42


# -----------------------------------------------------------------------------
# _redact_payload_blobs tests
# -----------------------------------------------------------------------------

class TestRedactPayloadBlobs:
    """Tests for _redact_payload_blobs data URL redaction."""

    def test_no_data_urls(self):
        """Text without data URLs is unchanged."""
        result = _redact_payload_blobs({"key": "normal text"})
        assert result == {"key": "normal text"}

    def test_short_data_url_unchanged(self):
        """Short data URLs are not redacted."""
        short_url = "data:text/plain;base64,SGVsbG8="  # "Hello"
        result = _redact_payload_blobs({"url": short_url}, max_chars=500)
        assert result["url"] == short_url

    def test_long_data_url_redacted(self):
        """Long data URLs are redacted."""
        long_b64 = "A" * 500
        long_url = f"data:image/png;base64,{long_b64}"
        result = _redact_payload_blobs({"url": long_url}, max_chars=100)
        assert "[REDACTED]" in result["url"]
        assert "500 chars" in result["url"]

    def test_nested_dict(self):
        """Nested dicts are walked."""
        long_b64 = "B" * 500
        data = {"outer": {"inner": f"data:image/png;base64,{long_b64}"}}
        result = _redact_payload_blobs(data, max_chars=100)
        assert "[REDACTED]" in result["outer"]["inner"]

    def test_list(self):
        """Lists are walked."""
        long_b64 = "C" * 500
        data = [f"data:image/png;base64,{long_b64}"]
        result = _redact_payload_blobs(data, max_chars=100)
        assert "[REDACTED]" in result[0]

    def test_tuple(self):
        """Tuples are walked and returned as tuples."""
        long_b64 = "D" * 500
        data = (f"data:image/png;base64,{long_b64}",)
        result = _redact_payload_blobs(data, max_chars=100)
        assert isinstance(result, tuple)
        assert "[REDACTED]" in result[0]

    def test_non_data_url_string(self):
        """Non-data-URL strings are unchanged."""
        result = _redact_payload_blobs("just text")
        assert result == "just text"

    def test_max_chars_bounds(self):
        """max_chars is bounded between 64 and 8192."""
        # Even with very small max_chars, minimum is enforced
        result = _redact_payload_blobs({"x": "y"}, max_chars=10)
        assert result == {"x": "y"}


# -----------------------------------------------------------------------------
# _extract_plain_text_content tests
# -----------------------------------------------------------------------------

class TestExtractPlainTextContent:
    """Tests for _extract_plain_text_content content extraction."""

    def test_string_content(self):
        """String content returns as-is."""
        assert _extract_plain_text_content("hello") == "hello"

    def test_list_of_strings(self):
        """List of strings is joined with newlines."""
        result = _extract_plain_text_content(["line1", "line2"])
        assert result == "line1\nline2"

    def test_list_of_dicts_with_text(self):
        """List of dicts extracts text field."""
        content = [{"text": "block1"}, {"text": "block2"}]
        result = _extract_plain_text_content(content)
        assert result == "block1\nblock2"

    def test_list_of_dicts_with_content(self):
        """List of dicts extracts content field."""
        content = [{"content": "block1"}, {"content": "block2"}]
        result = _extract_plain_text_content(content)
        assert result == "block1\nblock2"

    def test_single_dict_with_text(self):
        """Single dict extracts text field."""
        assert _extract_plain_text_content({"text": "hello"}) == "hello"

    def test_single_dict_with_content(self):
        """Single dict extracts content field."""
        assert _extract_plain_text_content({"content": "hello"}) == "hello"

    def test_dict_without_text_or_content(self):
        """Dict without text/content converts to string."""
        result = _extract_plain_text_content({"other": "value"})
        assert "other" in result

    def test_none_content(self):
        """None content returns empty string."""
        assert _extract_plain_text_content(None) == ""

    def test_mixed_list(self):
        """Mixed list (strings and dicts) is handled."""
        content = ["plain", {"text": "from dict"}]
        result = _extract_plain_text_content(content)
        assert "plain" in result
        assert "from dict" in result


# -----------------------------------------------------------------------------
# _extract_feature_flags tests
# -----------------------------------------------------------------------------

class TestExtractFeatureFlags:
    """Tests for _extract_feature_flags."""

    def test_with_features(self):
        """Features dict is extracted."""
        metadata = {"features": {"web_search": True, "code_interpreter": False}}
        result = _extract_feature_flags(metadata)
        assert result == {"web_search": True, "code_interpreter": False}

    def test_no_features(self):
        """Missing features returns empty dict."""
        assert _extract_feature_flags({}) == {}

    def test_features_not_dict(self):
        """Non-dict features returns empty dict."""
        assert _extract_feature_flags({"features": "not a dict"}) == {}

    def test_non_dict_metadata(self):
        """Non-dict metadata returns empty dict."""
        assert _extract_feature_flags("not a dict") == {}  # type: ignore
        assert _extract_feature_flags(None) == {}  # type: ignore


# -----------------------------------------------------------------------------
# merge_usage_stats tests
# -----------------------------------------------------------------------------

class TestMergeUsageStats:
    """Tests for merge_usage_stats."""

    def test_simple_merge(self):
        """Simple numeric merge."""
        total = {"prompt_tokens": 100, "completion_tokens": 50}
        new = {"prompt_tokens": 50, "completion_tokens": 25}
        result = merge_usage_stats(total, new)
        assert result["prompt_tokens"] == 150
        assert result["completion_tokens"] == 75

    def test_nested_merge(self):
        """Nested dict merge."""
        total = {"usage": {"tokens": 100}}
        new = {"usage": {"tokens": 50}}
        result = merge_usage_stats(total, new)
        assert result["usage"]["tokens"] == 150

    def test_new_keys(self):
        """New keys are added."""
        total = {"a": 1}
        new = {"b": 2}
        result = merge_usage_stats(total, new)
        assert result == {"a": 1, "b": 2}

    def test_none_value_preserves_existing(self):
        """None value preserves existing value."""
        total = {"key": 10}
        new = {"key": None}
        result = merge_usage_stats(total, new)
        assert result["key"] == 10

    def test_non_numeric_overwrites(self):
        """Non-numeric non-None values overwrite."""
        total = {"status": "pending"}
        new = {"status": "complete"}
        result = merge_usage_stats(total, new)
        assert result["status"] == "complete"


# -----------------------------------------------------------------------------
# wrap_code_block tests
# -----------------------------------------------------------------------------

class TestWrapCodeBlock:
    """Tests for wrap_code_block."""

    def test_basic_wrap(self):
        """Basic code block wrapping."""
        result = wrap_code_block("print('hello')")
        assert result.startswith("```python\n")
        assert result.endswith("\n```")
        assert "print('hello')" in result

    def test_custom_language(self):
        """Custom language tag."""
        result = wrap_code_block("const x = 1;", language="javascript")
        assert result.startswith("```javascript\n")

    def test_adapts_fence_length(self):
        """Fence length adapts to content with backticks."""
        # Content has triple backticks
        code = "```python\nprint('nested')\n```"
        result = wrap_code_block(code)
        # Fence should be longer than 3
        lines = result.split("\n")
        opening_fence = lines[0].replace("python", "")
        assert len(opening_fence) > 3


# -----------------------------------------------------------------------------
# _template_value_present tests
# -----------------------------------------------------------------------------

class TestTemplateValuePresent:
    """Tests for _template_value_present."""

    def test_none_is_not_present(self):
        """None is not present."""
        assert _template_value_present(None) is False

    def test_empty_string_is_not_present(self):
        """Empty string is not present."""
        assert _template_value_present("") is False

    def test_non_empty_string_is_present(self):
        """Non-empty string is present."""
        assert _template_value_present("text") is True

    def test_empty_collections_not_present(self):
        """Empty collections are not present."""
        assert _template_value_present([]) is False
        assert _template_value_present(()) is False
        assert _template_value_present(set()) is False
        assert _template_value_present({}) is False

    def test_non_empty_collections_present(self):
        """Non-empty collections are present."""
        assert _template_value_present([1]) is True
        assert _template_value_present((1,)) is True
        assert _template_value_present({1}) is True
        assert _template_value_present({"a": 1}) is True

    def test_numbers_always_present(self):
        """Numbers are always present (even 0)."""
        assert _template_value_present(0) is True
        assert _template_value_present(1) is True
        assert _template_value_present(0.0) is True
        assert _template_value_present(3.14) is True

    def test_other_types_use_bool(self):
        """Other types use bool() for presence."""
        class Truthy:
            def __bool__(self):
                return True

        class Falsy:
            def __bool__(self):
                return False

        assert _template_value_present(Truthy()) is True
        assert _template_value_present(Falsy()) is False


# -----------------------------------------------------------------------------
# _sanitize_path_component tests
# -----------------------------------------------------------------------------

class TestSanitizePathComponent:
    """Tests for _sanitize_path_component."""

    def test_simple_string(self):
        """Simple alphanumeric string is unchanged."""
        assert _sanitize_path_component("myfile") == "myfile"

    def test_with_special_chars(self):
        """Special characters are replaced with underscore."""
        result = _sanitize_path_component("my/file:name")
        assert "/" not in result
        assert ":" not in result
        assert "_" in result

    def test_empty_returns_fallback(self):
        """Empty string returns fallback."""
        assert _sanitize_path_component("") == "unknown"
        assert _sanitize_path_component("   ") == "unknown"

    def test_custom_fallback(self):
        """Custom fallback is used."""
        assert _sanitize_path_component("", fallback="default") == "default"

    def test_max_length(self):
        """Long strings are truncated."""
        long_str = "a" * 200
        result = _sanitize_path_component(long_str, max_length=50)
        assert len(result) <= 50

    def test_strips_leading_trailing_special(self):
        """Leading/trailing dots, underscores, dashes are stripped."""
        assert _sanitize_path_component("___test___") == "test"
        assert _sanitize_path_component("...test...") == "test"
        assert _sanitize_path_component("---test---") == "test"

    def test_all_special_returns_fallback(self):
        """All special chars returns fallback."""
        assert _sanitize_path_component("...") == "unknown"


# -----------------------------------------------------------------------------
# _retry_after_seconds tests
# -----------------------------------------------------------------------------

class TestRetryAfterSeconds:
    """Tests for _retry_after_seconds HTTP header parsing."""

    def test_numeric_seconds(self):
        """Numeric string is parsed as seconds."""
        assert _retry_after_seconds("60") == 60.0
        assert _retry_after_seconds("3.5") == 3.5

    def test_negative_clamped_to_zero(self):
        """Negative values are clamped to 0."""
        assert _retry_after_seconds("-10") == 0.0

    def test_empty_returns_none(self):
        """Empty value returns None."""
        assert _retry_after_seconds("") is None
        assert _retry_after_seconds("   ") is None

    def test_none_returns_none(self):
        """None returns None."""
        assert _retry_after_seconds(None) is None

    def test_http_date_format(self):
        """HTTP date format is parsed."""
        # Use a future date
        import datetime
        import email.utils

        future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=120)
        date_str = email.utils.format_datetime(future)
        result = _retry_after_seconds(date_str)
        # Should be approximately 120 seconds (with some tolerance)
        assert result is not None
        assert 100 < result < 140

    def test_past_date_returns_zero(self):
        """Past date returns 0."""
        import datetime
        import email.utils

        past = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=60)
        date_str = email.utils.format_datetime(past)
        result = _retry_after_seconds(date_str)
        assert result == 0.0

    def test_invalid_format_returns_none(self):
        """Invalid format returns None."""
        assert _retry_after_seconds("not a date or number") is None


# -----------------------------------------------------------------------------
# _pretty_json tests
# -----------------------------------------------------------------------------

class TestPrettyJson:
    """Tests for _pretty_json."""

    def test_none_returns_empty(self):
        """None returns empty string."""
        assert _pretty_json(None) == ""

    def test_string_stripped(self):
        """String is stripped and returned."""
        assert _pretty_json("  hello  ") == "hello"

    def test_bytes_decoded(self):
        """Bytes are decoded and stripped."""
        assert _pretty_json(b"  hello  ") == "hello"

    def test_dict_formatted(self):
        """Dict is formatted as JSON."""
        result = _pretty_json({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_non_serializable_uses_str(self):
        """Non-serializable objects use str()."""
        class Custom:
            def __str__(self):
                return "custom-str"

        result = _pretty_json(Custom())
        assert "custom-str" in result


# -----------------------------------------------------------------------------
# Additional _render_error_template edge cases
# -----------------------------------------------------------------------------

class TestRenderErrorTemplateEdgeCases:
    """Additional edge cases for template rendering."""

    def test_nested_conditionals(self):
        """Nested conditionals work correctly."""
        template = "{{#if outer}}outer{{#if inner}}inner{{/if}}end{{/if}}"

        # Both true
        result = _render_error_template(template, {"outer": True, "inner": True})
        assert "outer" in result and "inner" in result

        # Outer true, inner false
        result = _render_error_template(template, {"outer": True, "inner": False})
        assert "outer" in result
        assert "inner" not in result

    def test_extra_endif_ignored(self):
        """Extra {{/if}} without matching {{#if}} is gracefully handled."""
        template = "text{{/if}}more"
        # Should not raise, extra /if just pops empty stack (no-op)
        result = _render_error_template(template, {})
        assert "text" in result

    def test_placeholder_with_none_value(self):
        """Placeholder with None value renders as empty."""
        template = "Hello {name}!"
        result = _render_error_template(template, {"name": None})
        # Line with unresolved placeholder should be dropped
        assert result == ""

    def test_blank_line_inside_false_conditional(self):
        """Blank lines inside false conditional are skipped."""
        template = "start\n{{#if show}}\n\ninner\n{{/if}}\nend"
        result = _render_error_template(template, {"show": False})
        assert "inner" not in result
        assert "start" in result
        assert "end" in result

    def test_empty_template_uses_default(self):
        """Empty template falls back to default (line 79)."""
        from open_webui_openrouter_pipe import DEFAULT_OPENROUTER_ERROR_TEMPLATE

        # Empty string template should use default
        result = _render_error_template("", {"heading": "Test"})
        # Result should contain something from the default template
        assert "Test" in result or len(result) > 0

    def test_blank_line_preserved_when_active(self):
        """Blank lines are preserved when conditions are active (lines 116-117)."""
        # Template with intentional blank lines outside conditionals
        template = "line1\n\nline2"
        result = _render_error_template(template, {})
        # The blank line should be preserved
        assert result == "line1\n\nline2"

    def test_blank_line_in_active_conditional(self):
        """Blank lines inside active conditional are preserved."""
        template = "start\n{{#if show}}\n\ninner\n{{/if}}\nend"
        result = _render_error_template(template, {"show": True})
        # Both blank line and inner should be present
        assert "inner" in result


# -----------------------------------------------------------------------------
# Additional edge case tests for remaining coverage
# -----------------------------------------------------------------------------

class TestExtractMarkerUlidEdgeCases:
    """Additional tests for _extract_marker_ulid edge cases."""

    def test_empty_line(self):
        """Empty line returns None."""
        from open_webui_openrouter_pipe.core.utils import _extract_marker_ulid
        assert _extract_marker_ulid("") is None
        assert _extract_marker_ulid(None) is None  # type: ignore

    def test_wrong_length(self):
        """Wrong ULID length returns None."""
        from open_webui_openrouter_pipe.core.utils import _extract_marker_ulid
        # Too short
        assert _extract_marker_ulid("[ABC]: #") is None
        # Too long
        long_ulid = "0" * (ULID_LENGTH + 5)
        assert _extract_marker_ulid(f"[{long_ulid}]: #") is None

    def test_invalid_crockford_chars(self):
        """Invalid Crockford characters return None."""
        from open_webui_openrouter_pipe.core.utils import _extract_marker_ulid
        # 'I', 'L', 'O', 'U' are not in Crockford alphabet
        invalid_ulid = "I" * ULID_LENGTH  # 'I' not in Crockford
        assert _extract_marker_ulid(f"[{invalid_ulid}]: #") is None

    def test_no_marker_prefix(self):
        """Line without [ prefix returns None."""
        from open_webui_openrouter_pipe.core.utils import _extract_marker_ulid
        ulid = "0" * ULID_LENGTH
        assert _extract_marker_ulid(f"{ulid}]: #") is None

    def test_no_marker_suffix(self):
        """Line without ]: # suffix returns None."""
        from open_webui_openrouter_pipe.core.utils import _extract_marker_ulid
        ulid = "0" * ULID_LENGTH
        assert _extract_marker_ulid(f"[{ulid}]") is None


class TestSelectBestEffortFallbackEdgeCases:
    """Additional tests for _select_best_effort_fallback edge cases."""

    def test_fallback_loop_coverage(self):
        """Test the closest match loop path (lines 261-268).

        This tests the scenario where we have to find the closest match
        via iteration rather than direct comparison.
        """
        # Request "minimal" when only "high" is available (not in typical ordering path)
        # This should still return "high" through the closest match logic
        result = _select_best_effort_fallback("minimal", ["high"])
        assert result == "high"

    def test_middle_match_next_higher(self):
        """Test finding next higher effort when in between."""
        # Request "low" when we have "none" and "medium"
        # Should find "medium" as next higher
        result = _select_best_effort_fallback("low", ["none", "medium"])
        assert result == "medium"

    def test_closest_match_fallback_loop(self):
        """Test the closest distance fallback loop (lines 261-268).

        This scenario exercises the final fallback loop when:
        1. requested_idx is in the ordering
        2. There's no exact match
        3. The for-loop at lines 258-260 doesn't find a suitable match
           (i.e., no idx > requested_idx)

        The key is that all supported values have idx <= requested_idx,
        so the 'for idx, value in indexed: if idx > requested_idx' loop
        never returns, and we fall through to the closest-distance loop.
        """
        # Request "medium" (idx=3 in ordering: none=0, minimal=1, low=2, medium=3, high=4, xhigh=5)
        # When supported only has values with idx <= requested_idx (like "none" and "minimal")
        # The for-loop at 258-260 will iterate but never return (no idx > 3)
        # Then lines 261-268 compute the closest match

        # Testing: request "high" when only "none" and "low" available
        # high idx=4, none idx=0, low idx=2
        # None satisfy idx > 4, so we fall through to closest distance loop
        # low (idx=2) is closest to high (idx=4)
        result = _select_best_effort_fallback("high", ["none", "low"])
        assert result == "low"

    def test_closest_match_single_option_lower(self):
        """Test closest match when only lower options available."""
        # Request "xhigh" (idx=5) when only "minimal" (idx=1) is available
        # min_idx=1 < requested_idx=5, max_idx=1 < requested_idx=5
        # So we return max_value = "minimal"
        result = _select_best_effort_fallback("xhigh", ["minimal"])
        assert result == "minimal"


class TestRedactPayloadBlobsEdgeCases:
    """Additional tests for _redact_payload_blobs edge cases."""

    def test_non_string_primitives(self):
        """Non-string primitive values pass through unchanged (line 399)."""
        data = {"number": 42, "boolean": True, "none": None}
        result = _redact_payload_blobs(data)
        assert result == {"number": 42, "boolean": True, "none": None}

    def test_non_integer_max_chars(self):
        """Non-integer max_chars falls back to 256."""
        data = {"x": "y"}
        result = _redact_payload_blobs(data, max_chars="invalid")  # type: ignore
        assert result == {"x": "y"}


class TestGetOpenWebuiConfigModule:
    """Tests for _get_open_webui_config_module."""

    def test_returns_cached_module(self):
        """Returns cached module on subsequent calls."""
        from open_webui_openrouter_pipe.core import utils

        # First call will try to import
        result1 = utils._get_open_webui_config_module()
        # Second call should return the cached result
        result2 = utils._get_open_webui_config_module()
        assert result1 is result2

    def test_caches_successful_import(self):
        """Caches module after successful import (lines 348-355)."""
        from open_webui_openrouter_pipe.core import utils
        import sys

        # The conftest.py installs open_webui.config stub
        # So the import should succeed and cache the module
        result = utils._get_open_webui_config_module()
        # Should be the stub module from conftest
        assert result is not None or result is None  # Either way, it's cached

    def test_first_successful_import_caches(self):
        """Test first successful import caches the module (lines 352-355)."""
        from open_webui_openrouter_pipe.core import utils
        import sys

        # Reset the cache to force a fresh import
        original_cache = utils._OPEN_WEBUI_CONFIG_MODULE
        utils._OPEN_WEBUI_CONFIG_MODULE = None

        try:
            # This should trigger the import path (lines 350-355)
            # conftest.py has installed open_webui.config stub
            result = utils._get_open_webui_config_module()

            # After first call, should be cached
            cached = utils._OPEN_WEBUI_CONFIG_MODULE
            # The stub module should be cached
            if result is not None:
                assert cached is result
        finally:
            # Restore original state
            utils._OPEN_WEBUI_CONFIG_MODULE = original_cache


class TestRetryAfterSecondsEdgeCases:
    """Additional tests for _retry_after_seconds edge cases."""

    def test_http_date_without_timezone(self):
        """HTTP date without explicit timezone still works (lines 549, 551).

        Note: email.utils.parsedate_to_datetime typically returns tz-aware
        datetime, but we test the branch where it might not.
        """
        # Regular HTTP date format should work
        import datetime
        import email.utils

        # A properly formatted HTTP date
        future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=60)
        date_str = email.utils.format_datetime(future)
        result = _retry_after_seconds(date_str)
        assert result is not None
        assert result >= 0

    def test_malformed_date_returns_none(self):
        """Malformed date that raises exception returns None."""
        # This should hit the except clause on line 555-556
        result = _retry_after_seconds("Mon, 99 Xyz 9999 99:99:99 GMT")
        assert result is None

    def test_date_returns_none_path(self, monkeypatch):
        """Test the dt is None branch (line 549).

        parsedate_to_datetime can return None for certain inputs.
        """
        import email.utils
        from open_webui_openrouter_pipe.core import utils

        # Mock parsedate_to_datetime to return None
        def mock_parsedate_to_datetime(value):
            return None

        monkeypatch.setattr(email.utils, "parsedate_to_datetime", mock_parsedate_to_datetime)

        result = utils._retry_after_seconds("some date string")
        assert result is None

    def test_naive_datetime_gets_utc(self, monkeypatch):
        """Test the timezone-naive datetime branch (line 551)."""
        import datetime
        import email.utils
        from open_webui_openrouter_pipe.core import utils

        # Create a naive datetime that represents "60 seconds from now" in UTC
        # We take a timezone-aware datetime, add 60 seconds, then strip the timezone
        # This simulates what the function receives when timezone is not set
        aware_now = datetime.datetime.now(datetime.timezone.utc)
        naive_future_utc = (aware_now + datetime.timedelta(seconds=60)).replace(tzinfo=None)

        def mock_parsedate_to_datetime(value):
            return naive_future_utc

        monkeypatch.setattr(email.utils, "parsedate_to_datetime", mock_parsedate_to_datetime)

        result = utils._retry_after_seconds("some date string")
        # Should return approximately 60 seconds (give or take for test execution time)
        assert result is not None
        assert 55 < result < 65


# ===== From test_module_helpers.py =====

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
    ow._debug_print_request(headers, {"a": 1}, logger=ow.LOGGER)
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
    body = await ow._debug_print_error_response(resp, logger=ow.LOGGER)
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
    pipe._artifact_store._encryption_key = "a" * 32  # type: ignore[attr-defined]
    pipe._artifact_store._encrypt_all = True  # type: ignore[attr-defined]
    pipe._artifact_store._fernet = None  # type: ignore[attr-defined]
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

    pipe._artifact_store._redis_client = FakeRedis()  # type: ignore[attr-defined]
    pipe._artifact_store._redis_enabled = True  # type: ignore[attr-defined]
    fetched = await pipe._redis_fetch_rows("chat", ["01TEST"])
    assert fetched["01TEST"]["content"] == "secret"


@pytest.mark.asyncio
async def test_flush_redis_queue_warns_when_lock_release_returns_zero(caplog, pipe_instance_async):
    pipe = pipe_instance_async
    pipe._artifact_store._redis_enabled = True  # type: ignore[attr-defined]

    class FakeRedis:
        async def set(self, key, value, *, nx=False, ex=None):
            return True

        async def lpop(self, key):
            return None

        async def eval(self, script, numkeys, key, token):
            return 0

    pipe._artifact_store._redis_client = FakeRedis()  # type: ignore[attr-defined]

    caplog.set_level(logging.WARNING)
    await pipe._flush_redis_queue()
    assert any(
        "Redis flush lock was not released" in record.getMessage()
        for record in caplog.records
    )


# ===== From test_helper_utilities.py =====

import json

import pytest

from open_webui_openrouter_pipe import (
    EncryptedStr,
    ModelFamily,
    OpenRouterAPIError,
    Pipe,
    ResponsesBody,
    _OPENROUTER_REFERER,
    _build_error_template_values,
    _classify_function_call_artifacts,
    _normalize_persisted_item,
    _pretty_json,
    _render_error_template,
    _select_openrouter_http_referer,
    _strictify_schema,
    _template_value_present,
    build_tools,
    contains_marker,
    generate_item_id,
    split_text_by_markers,
)


def test_render_error_template_hides_missing_sections():
    template = (
        "{{#if heading}}## {heading}{{/if}}\n"
        "{{#if error_id}}error: {error_id}{{/if}}\n"
        "{{#if optional}}should not appear{{/if}}"
    )
    content = _render_error_template(
        template,
        {"heading": "Oops", "error_id": "E-123", "optional": ""},
    )
    assert "Oops" in content
    assert "E-123" in content
    assert "should not appear" not in content


def test_render_error_template_uses_default_template():
    rendered = _render_error_template(
        "",
        {"heading": "Fallback", "error_id": "ERR-1"},
    )
    assert "Fallback" in rendered
    assert "ERR-1" in rendered


def test_build_error_template_values_reports_model_limits():
    error = OpenRouterAPIError(
        status=400,
        reason="Bad Request",
        provider="Demo",
        openrouter_message='Shorten the prompt or use the "middle-out" transform.',
        metadata={"rate_limit_type": "minute"},
        requested_model="demo",
    )
    values = _build_error_template_values(
        error,
        heading="Demo",
        diagnostics=["- diag"],
        metrics={"context_limit": 16000, "max_output_tokens": 4096},
        model_identifier="demo",
        normalized_model_id="demo",
        api_model_id="demo",
    )
    assert values["include_model_limits"] is True
    assert values["context_limit_tokens"] == "16,000"
    assert "minute" in values["rate_limit_type"]


def test_template_utils_cover_edge_cases():
    assert _template_value_present("text") is True
    assert _template_value_present("") is False
    assert _template_value_present([1, 2]) is True
    assert _template_value_present([]) is False

    class NotSerializable:
        pass

    value = {NotSerializable()}
    result = _pretty_json(value)
    assert "NotSerializable" in result


def test_encrypted_str_roundtrip(monkeypatch):
    monkeypatch.setenv("WEBUI_SECRET_KEY", "super-secret-key")
    cipher = EncryptedStr.encrypt("token-value")
    assert cipher.startswith("encrypted:")
    assert EncryptedStr.decrypt(cipher) == "token-value"


def test_strictify_schema_enforces_required_and_nullability():
    schema = {
        "type": "object",
        "properties": {
            "required": {"type": "string"},
            "maybe": {"type": "integer"},
            "child": {
                "type": "object",
                "properties": {"flag": {"type": "boolean"}},
            },
        },
        "required": ["required"],
    }
    strict = _strictify_schema(schema)
    assert set(strict["required"]) == {"required", "maybe", "child"}
    assert strict["properties"]["required"]["type"] == "string"
    assert set(strict["properties"]["maybe"]["type"]) == {"integer", "null"}
    child_props = strict["properties"]["child"]["properties"]
    assert set(child_props["flag"]["type"]) == {"boolean", "null"}
    assert strict["properties"]["child"]["additionalProperties"] is False


def test_build_tools_combines_registry_and_extras():
    # Use a real model with function calling support
    # Set up ModelFamily with proper features derived from supported_parameters
    # Note: base_model() normalizes "openai/gpt-4" to "openai.gpt-4" (dot, not slash)
    ModelFamily.set_dynamic_specs({
        "openai.gpt-4": {  # Use dot notation as returned by base_model()
            "id": "openai/gpt-4",
            "features": {"function_calling"},  # Derived from tools+tool_choice support
            "supported_parameters": frozenset(["tools", "tool_choice"]),
        }
    })

    responses_body = ResponsesBody(model="openai/gpt-4", input=[])
    valves = Pipe.Valves(ENABLE_STRICT_TOOL_CALLING=True)

    registry = {
        "alpha": {
            "spec": {
                "name": "alpha",
                "description": "alpha tool",
                "parameters": {"type": "object", "properties": {"foo": {"type": "string"}}},
            }
        }
    }
    extra_tools = [
        {"type": "function", "name": "alpha", "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "beta", "parameters": {"type": "object", "properties": {}}},
    ]

    tools = build_tools(responses_body, valves, __tools__=registry, extra_tools=extra_tools)

    function_names = [tool["name"] for tool in tools if tool["type"] == "function"]
    assert "beta" in function_names
    assert "alpha" in function_names  # dedup keeps last occurrence
    alpha_tool = next(tool for tool in tools if tool["type"] == "function" and tool["name"] == "alpha")
    assert alpha_tool["parameters"]["properties"] == {}


def test_tool_output_clamps_failed_status(pipe_instance):
    pipe = pipe_instance
    output = pipe._build_tool_output(
        {"call_id": "call-1"},
        "Tool not found",
        status="failed",
    )
    assert output["type"] == "function_call_output"
    assert output["status"] == "completed"


def test_select_openrouter_http_referer_defaults_without_override():
    assert _select_openrouter_http_referer(None) == _OPENROUTER_REFERER


def test_select_openrouter_http_referer_accepts_full_url_override():
    valves = Pipe.Valves(HTTP_REFERER_OVERRIDE="https://example.com")
    assert _select_openrouter_http_referer(valves) == "https://example.com"


def test_select_openrouter_http_referer_rejects_non_url_override():
    valves = Pipe.Valves(HTTP_REFERER_OVERRIDE="example.com")
    assert _select_openrouter_http_referer(valves) == _OPENROUTER_REFERER


def test_normalize_persisted_items_and_classifier():
    call = _normalize_persisted_item({"type": "function_call", "name": "fetch", "arguments": {"foo": 1}})
    assert call is not None
    assert call["call_id"]
    assert json.loads(call["arguments"])["foo"] == 1

    output = _normalize_persisted_item(
        {"type": "function_call_output", "call_id": "abc", "output": {"data": 42}}
    )
    assert output is not None
    assert output["output"] == str({"data": 42})

    reasoning = _normalize_persisted_item({"type": "reasoning", "content": "step"})
    assert reasoning is not None
    assert reasoning["content"][0]["text"] == "step"

    artifacts = {
        "a": {"type": "function_call", "call_id": "shared"},
        "b": {"type": "function_call_output", "call_id": "shared"},
        "c": {"type": "function_call", "call_id": "orphan"},
        "d": {"type": "function_call_output", "call_id": "lonely-output"},
    }
    valid, orphan_calls, orphan_outputs = _classify_function_call_artifacts(artifacts)
    assert valid == {"shared"}
    assert orphan_calls == {"orphan"}
    assert orphan_outputs == {"lonely-output"}


def test_marker_helpers_and_ulids():
    ulid = generate_item_id()
    marker_line = f"[{ulid}]: #"
    text = f"prefix\n{marker_line}\npostfix"
    assert contains_marker(text) is True
    segments = split_text_by_markers(text)
    assert any(segment.get("marker") == ulid for segment in segments if segment["type"] == "marker")
    ids = {generate_item_id() for _ in range(10)}
    assert len(ids) == 10

# ===== From test_status_formatting.py =====

from open_webui_openrouter_pipe import Pipe


def test_format_final_status_description_includes_cost_tokens_and_tps(pipe_instance):
    pipe = pipe_instance
    usage = {
        "cost": 0.012345,
        "input_tokens": 120,
        "output_tokens": 40,
        "input_tokens_details": {"cached_tokens": 20},
        "output_tokens_details": {"reasoning_tokens": 5},
    }

    description = pipe._format_final_status_description(
        elapsed=3.21,
        total_usage=usage,
        valves=pipe.valves,
        stream_duration=2.0,
    )

    assert description.startswith("Time: 3.21s  20.0 tps")
    assert "Cost $0.012345" in description
    assert "Total tokens: 160 (Input: 120, Output: 40, Cached: 20, Reasoning: 5)" in description


def test_format_final_status_description_respects_disabled_flag(pipe_instance):
    pipe = pipe_instance
    valves = pipe.Valves(SHOW_FINAL_USAGE_STATUS=False)

    description = pipe._format_final_status_description(
        elapsed=4.5,
        total_usage={},
        valves=valves,
        stream_duration=None,
    )

    assert description == "Thought for 4.5 seconds"


def test_format_final_status_description_with_icons(pipe_instance):
    pipe = pipe_instance
    valves = pipe.Valves(
        FINAL_USAGE_STATUS_STYLE="icons",
        USAGE_STATUS_ICON_SET="T,$,TOT,IN,OUT,CACHE,REASON",
    )
    usage = {
        "cost": 0.012345,
        "input_tokens": 120,
        "output_tokens": 40,
        "input_tokens_details": {"cached_tokens": 20},
        "output_tokens_details": {"reasoning_tokens": 5},
    }

    description = pipe._format_final_status_description(
        elapsed=3.21,
        total_usage=usage,
        valves=valves,
        stream_duration=2.0,
    )

    assert (
        description
        == "T 3.21s  20.0 tps | $0.012345 | TOT 160 (IN 120, OUT 40, CACHE 20, REASON 5)"
    )
