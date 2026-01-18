"""Tests for OpenRouter error template rendering."""

from unittest.mock import Mock

import pytest

from open_webui_openrouter_pipe import (
    OpenRouterAPIError,
    _build_error_template_values,
    _render_error_template,
    DEFAULT_OPENROUTER_ERROR_TEMPLATE,
    DEFAULT_NETWORK_TIMEOUT_TEMPLATE,
    DEFAULT_RATE_LIMIT_TEMPLATE,
    DEFAULT_AUTHENTICATION_ERROR_TEMPLATE,
    DEFAULT_INSUFFICIENT_CREDITS_TEMPLATE,
)


class TestOpenRouterErrorTemplateRendering:
    """Tests for OpenRouter-specific error template rendering."""

    def test_openrouter_error_template_full_context(self):
        """All placeholders filled → complete error message."""
        error = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            provider="anthropic",
            openrouter_message='Model not available or use the "middle-out" option.',
            openrouter_code="model_not_found",
            upstream_message='The model you requested is not available; or use the "middle-out" option.',
            upstream_type="invalid_request_error",
            request_id="req_123",
            raw_body='{"error": {"message": "Model not available"}}',
            metadata={
                "provider_name": "anthropic",
                "model_slug": "claude-3-haiku",
                "required_cost": 0.01,
                "account_balance": 1.50,
                "retry_after_seconds": 30,
                "rate_limit_type": "requests",
                "reasons": ["inappropriate_content"],
                "flagged_input": "bad content",
            },
            moderation_reasons=["inappropriate_content"],
            flagged_input="bad content",
            model_slug="claude-3-haiku",
            requested_model="anthropic/claude-3-haiku",
            metadata_json='{"provider_name": "anthropic"}',
            provider_raw={"error": {"type": "invalid_request_error"}},
            provider_raw_json='{"error": {"type": "invalid_request_error"}}',
        )

        values = _build_error_template_values(
            error,
            heading="🚫 Request Failed",
            diagnostics=["- Model: claude-3-haiku", "- Provider: anthropic"],
            metrics={"context_limit": 200000, "max_output_tokens": 4096},
            model_identifier="claude-3-haiku",
            normalized_model_id="anthropic.claude-3-haiku",
            api_model_id="claude-3-haiku",
            context={
                "error_id": "err_123",
                "timestamp": "2024-01-01T12:00:00Z",
                "session_id": "sess_123",
                "user_id": "user_123",
                "support_email": "support@example.com",
            },
        )

        result = _render_error_template(DEFAULT_OPENROUTER_ERROR_TEMPLATE, values)

        # Verify all major sections are present
        assert "🚫 Request Failed" in result
        assert "err_123" in result
        assert "2024-01-01T12:00:00Z" in result
        assert "sess_123" in result
        assert "user_123" in result
        assert "anthropic" in result
        assert "claude-3-haiku" in result
        assert "Model not available" in result
        assert "invalid_request_error" in result
        assert "req_123" in result
        assert "inappropriate_content" in result
        assert "bad content" in result
        assert "200,000" in result  # formatted context limit
        assert "4,096" in result  # formatted max tokens

    def test_openrouter_error_template_missing_optionals(self):
        """Missing optional fields → lines omitted cleanly."""
        error = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            openrouter_message="Simple error",
        )

        values = _build_error_template_values(
            error,
            heading="Error",
            diagnostics=[],
            metrics={},
            model_identifier=None,
            normalized_model_id=None,
            api_model_id=None,
            context={"error_id": "err_123"},
        )

        result = _render_error_template(DEFAULT_OPENROUTER_ERROR_TEMPLATE, values)

        # Should contain basic error info
        assert "Error" in result
        assert "err_123" in result
        assert "Simple error" in result

        # Should not contain optional sections that are empty
        assert "Provider:" not in result
        assert "Model:" not in result
        assert "Request ID:" not in result
        assert "Moderation reasons:" not in result

    def test_network_timeout_template(self):
        """Timeout error renders with timeout_seconds placeholder."""
        template = DEFAULT_NETWORK_TIMEOUT_TEMPLATE
        values = {
            "error_id": "timeout_123",
            "timeout_seconds": 30,
            "timestamp": "2024-01-01T12:00:00Z",
            "support_email": "support@example.com",
        }

        result = _render_error_template(template, values)

        assert "⏱️ Request Timeout" in result
        assert "timeout_123" in result
        assert "30s" in result
        assert "2024-01-01T12:00:00Z" in result
        assert "support@example.com" in result

    def test_rate_limit_template_with_retry_after(self):
        """429 error includes retry_after_seconds from header."""
        template = DEFAULT_RATE_LIMIT_TEMPLATE
        values = {
            "error_id": "rate_123",
            "openrouter_code": 429,
            "retry_after_seconds": 60,
            "rate_limit_type": "requests",
            "timestamp": "2024-01-01T12:00:00Z",
            "support_email": "support@example.com",
        }

        result = _render_error_template(template, values)

        assert "⏸️ Rate Limit Exceeded" in result
        assert "rate_123" in result
        assert "429" in result
        assert "60s" in result
        assert "requests" in result
        assert "support@example.com" in result

    def test_authentication_error_template(self):
        """401 error renders authentication guidance."""
        template = DEFAULT_AUTHENTICATION_ERROR_TEMPLATE
        values = {
            "error_id": "auth_123",
            "openrouter_code": 401,
            "openrouter_message": "Invalid API key",
            "timestamp": "2024-01-01T12:00:00Z",
            "support_email": "support@example.com",
        }

        result = _render_error_template(template, values)

        assert "🔐 Authentication Failed" in result
        assert "auth_123" in result
        assert "401" in result
        assert "Invalid API key" in result
        assert "API key" in result
        assert "https://openrouter.ai/keys" in result
        assert "support@example.com" in result

    def test_insufficient_credits_template(self):
        """402 error includes balance and cost info."""
        template = DEFAULT_INSUFFICIENT_CREDITS_TEMPLATE
        values = {
            "error_id": "credit_123",
            "openrouter_code": 402,
            "openrouter_message": "Insufficient credits",
            "required_cost": 0.50,
            "account_balance": 0.25,
            "timestamp": "2024-01-01T12:00:00Z",
            "support_email": "support@example.com",
        }

        result = _render_error_template(template, values)

        assert "💳 Insufficient Credits" in result
        assert "credit_123" in result
        assert "402" in result
        assert "Insufficient credits" in result
        assert "$0.5" in result
        assert "$0.25" in result
        assert "https://openrouter.ai/credits" in result
        assert "support@example.com" in result

    def test_conditional_blocks_render_correctly(self):
        """{{#if variable}}...{{/if}} blocks render only when truthy."""
        template = """
        Always shown
        {{#if show_this}}
        This should appear
        {{/if}}
        {{#if hide_this}}
        This should not appear
        {{/if}}
        End
        """.strip()

        # Test with truthy condition
        values_truthy = {"show_this": "yes", "hide_this": ""}
        result_truthy = _render_error_template(template, values_truthy)
        assert "This should appear" in result_truthy
        assert "This should not appear" not in result_truthy

        # Test with falsy condition
        values_falsy = {"show_this": "", "hide_this": "no"}
        result_falsy = _render_error_template(template, values_falsy)
        assert "This should appear" not in result_falsy
        assert "This should not appear" in result_falsy

    def test_custom_template_override(self):
        """User-provided template in valve replaces default."""
        # This would be tested in integration with the pipe's valve system
        # For now, test the template rendering directly
        custom_template = "Custom error: {error_id} - {detail}"
        values = {
            "error_id": "custom_123",
            "detail": "Custom message",
        }

        result = _render_error_template(custom_template, values)

        assert "Custom error: custom_123 - Custom message" == result.strip()


class TestTemplateValueBuilding:
    """Tests for building template values from errors."""

    def test_build_values_with_minimal_error(self):
        """Build values from a minimal error."""
        error = OpenRouterAPIError(
            status=500,
            reason="Internal Server Error",
            openrouter_message="Something went wrong",
        )

        values = _build_error_template_values(
            error,
            heading="Error",
            diagnostics=[],
            metrics={},
            model_identifier=None,
            normalized_model_id=None,
            api_model_id=None,
        )

        assert values["heading"] == "Error"
        assert values["openrouter_message"] == "Something went wrong"
        assert values["detail"] == "Something went wrong"
        assert values["reason"] == "Something went wrong"

    def test_build_values_with_full_error(self):
        """Build values from a fully populated error."""
        error = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            provider="openai",
            openrouter_message="Rate limit exceeded",
            openrouter_code="rate_limit_exceeded",
            upstream_message="Too many requests",
            upstream_type="rate_limit_error",
            request_id="req_12345",
            raw_body='{"error": {"message": "Rate limit exceeded"}}',
            metadata={
                "provider_name": "openai",
                "model_slug": "gpt-4",
                "retry_after_seconds": 60,
                "rate_limit_type": "requests",
            },
            moderation_reasons=["content_policy"],
            flagged_input="flagged text",
            model_slug="gpt-4",
            requested_model="openai/gpt-4",
            metadata_json='{"provider_name": "openai"}',
            provider_raw={"error": {"type": "rate_limit_error"}},
            provider_raw_json='{"error": {"type": "rate_limit_error"}}',
        )

        values = _build_error_template_values(
            error,
            heading="🚫 Rate Limited",
            diagnostics=["- Check usage dashboard"],
            metrics={"context_limit": 128000, "max_output_tokens": 4096},
            model_identifier="gpt-4",
            normalized_model_id="openai.gpt-4",
            api_model_id="gpt-4",
            context={
                "error_id": "err_abc123",
                "timestamp": "2024-01-01T10:30:00Z",
                "session_id": "sess_xyz",
                "user_id": "user_999",
                "support_email": "help@openrouter.ai",
            },
        )

        # Verify all expected values are present
        assert values["heading"] == "🚫 Rate Limited"
        assert values["error_id"] == "err_abc123"
        assert values["timestamp"] == "2024-01-01T10:30:00Z"
        assert values["session_id"] == "sess_xyz"
        assert values["user_id"] == "user_999"
        assert values["support_email"] == "help@openrouter.ai"
        assert values["provider"] == "openai"
        assert values["model_identifier"] == "gpt-4"
        assert values["requested_model"] == "openai/gpt-4"
        assert values["api_model_id"] == "gpt-4"
        assert values["normalized_model_id"] == "openai.gpt-4"
        assert values["openrouter_code"] == "rate_limit_exceeded"
        assert values["upstream_type"] == "rate_limit_error"
        assert values["upstream_message"] == "Too many requests"
        assert values["openrouter_message"] == "Rate limit exceeded"
        assert values["request_id"] == "req_12345"
        assert values["moderation_reasons"] == "- content_policy"
        assert values["flagged_excerpt"] == "flagged text"
        assert values["context_limit_tokens"] == ""
        assert values["max_output_tokens"] == ""
        assert values["include_model_limits"] is False
        assert values["retry_after_seconds"] == 60
        assert values["rate_limit_type"] == "requests"
        assert values["diagnostics"] == "- Check usage dashboard"
