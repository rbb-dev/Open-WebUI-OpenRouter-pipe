"""Tests for the error template system."""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
import httpx
import aiohttp
from typing import Any, List
from aioresponses import aioresponses
from open_webui_openrouter_pipe import EncryptedStr


@pytest_asyncio.fixture
async def mock_pipe():
    """Create a mock Pipe instance with valves configured."""
    from open_webui_openrouter_pipe import Pipe

    pipe = Pipe()
    pipe.valves.SUPPORT_EMAIL = "support@example.com"
    pipe.valves.SUPPORT_URL = "https://support.example.com"
    pipe.logger = MagicMock()

    yield pipe

    # Cleanup: close the pipe to stop worker tasks
    await pipe.close()


class _Emitter:
    def __init__(self) -> None:
        self.events: List[dict[str, Any]] = []

    async def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(event)


@pytest.fixture
def mock_event_emitter():
    return _Emitter()


class TestEmitTemplatedError:
    """Test the _emit_templated_error helper method."""

    @pytest.mark.asyncio
    async def test_basic_template_rendering(self, mock_pipe, mock_event_emitter):
        """Test that basic template rendering works."""
        await mock_pipe._ensure_error_formatter()._emit_templated_error(
            mock_event_emitter,
            template="### {title}\n\n{message}",
            variables={"title": "Test Error", "message": "Test message"},
            log_message="Test log",
        )

        # Should emit chat message and completion
        assert len(mock_event_emitter.events) == 2
        assert mock_event_emitter.events[0]["type"] == "chat:message"
        assert "Test Error" in mock_event_emitter.events[0]["data"]["content"]
        assert "Test message" in mock_event_emitter.events[0]["data"]["content"]

    @pytest.mark.asyncio
    async def test_error_id_generation(self, mock_pipe, mock_event_emitter):
        """Test that error IDs are generated and included."""
        await mock_pipe._ensure_error_formatter()._emit_templated_error(
            mock_event_emitter,
            template="Error ID: {error_id}",
            variables={},
            log_message="Test",
        )

        content = mock_event_emitter.events[0]["data"]["content"]
        assert "Error ID:" in content
        # Error ID should be 16 hex characters
        error_id = content.split("Error ID:")[-1].strip()
        assert len(error_id) == 16

    @pytest.mark.asyncio
    async def test_conditional_rendering(self, mock_pipe, mock_event_emitter):
        """Test that {{#if}} conditionals work."""
        await mock_pipe._ensure_error_formatter()._emit_templated_error(
            mock_event_emitter,
            template=(
                "### Error\n\n"
                "{{#if detail}}\n"
                "Detail: {detail}\n"
                "{{/if}}\n"
                "{{#if missing}}\n"
                "This should not appear\n"
                "{{/if}}\n"
            ),
            variables={"detail": "Important detail"},
            log_message="Test",
        )

        content = mock_event_emitter.events[0]["data"]["content"]
        assert "Detail: Important detail" in content
        assert "This should not appear" not in content

    @pytest.mark.asyncio
    async def test_support_email_injection(self, mock_pipe, mock_event_emitter):
        """Test that support_email from valves is injected."""
        await mock_pipe._ensure_error_formatter()._emit_templated_error(
            mock_event_emitter,
            template="Support: {support_email}",
            variables={},
            log_message="Test",
        )

        content = mock_event_emitter.events[0]["data"]["content"]
        assert "support@example.com" in content

    @pytest.mark.asyncio
    async def test_timestamp_injection(self, mock_pipe, mock_event_emitter):
        """Test that timestamp is injected."""
        await mock_pipe._ensure_error_formatter()._emit_templated_error(
            mock_event_emitter,
            template="Time: {timestamp}",
            variables={},
            log_message="Test",
        )

        content = mock_event_emitter.events[0]["data"]["content"]
        assert "Time: " in content
        # Should be ISO 8601 format with Z suffix
        assert "Z" in content


class TestNetworkTimeoutError:
    """Test network timeout error handling."""

    @pytest.mark.asyncio
    async def test_timeout_exception_caught(self, mock_pipe, mock_event_emitter):
        """Test that TimeoutException is caught and formatted."""
        # Mock HTTP boundary - let real Registry handle the model lookup
        with aioresponses() as mock_http:
            # Mock the /models endpoint that Registry fetches from
            mock_http.get(
                "https://openrouter.ai/api/v1/models",
                payload={
                    "data": [
                        {
                            "id": "test-model",
                            "name": "Test Model",
                            "pricing": {"prompt": "0", "completion": "0"},
                        }
                    ]
                },
            )

            with patch.object(mock_pipe, '_process_transformed_request', side_effect=httpx.TimeoutException("timeout")):
                mock_pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
                mock_pipe.valves.API_KEY = EncryptedStr("test-key")
                mock_pipe.valves.MODEL_CATALOG_REFRESH_SECONDS = 300

                # Use real aiohttp.ClientSession so aioresponses can mock it
                async with aiohttp.ClientSession() as session:
                    result = await mock_pipe._handle_pipe_call(
                        body={"model": "test-model"},
                        __user__={"id": "test-user"},
                        __request__=MagicMock(),
                        __event_emitter__=mock_event_emitter,
                        __event_call__=None,
                        __metadata__={"model": {"id": "test-model"}},
                        __tools__=None,
                        valves=mock_pipe.valves,
                        session=session,
                    )

        assert result == ""
        assert len(mock_event_emitter.events) == 2
        content = mock_event_emitter.events[0]["data"]["content"]
        assert "⏱️" in content or "Timeout" in content
        assert "Error ID:" in content


class TestConnectionError:
    """Test connection error handling."""

    @pytest.mark.asyncio
    async def test_connect_error_caught(self, mock_pipe, mock_event_emitter):
        """Test that ConnectError is caught and formatted."""
        # Mock HTTP boundary - let real Registry handle the model lookup
        with aioresponses() as mock_http:
            # Mock the /models endpoint that Registry fetches from
            mock_http.get(
                "https://openrouter.ai/api/v1/models",
                payload={
                    "data": [
                        {
                            "id": "test-model",
                            "name": "Test Model",
                            "pricing": {"prompt": "0", "completion": "0"},
                        }
                    ]
                },
            )

            with patch.object(mock_pipe, '_process_transformed_request', side_effect=httpx.ConnectError("connection failed")):
                mock_pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
                mock_pipe.valves.API_KEY = EncryptedStr("test-key")
                mock_pipe.valves.MODEL_CATALOG_REFRESH_SECONDS = 300

                # Use real aiohttp.ClientSession so aioresponses can mock it
                async with aiohttp.ClientSession() as session:
                    result = await mock_pipe._handle_pipe_call(
                        body={"model": "test-model"},
                        __user__={"id": "test-user"},
                        __request__=MagicMock(),
                        __event_emitter__=mock_event_emitter,
                        __event_call__=None,
                        __metadata__={"model": {"id": "test-model"}},
                        __tools__=None,
                        valves=mock_pipe.valves,
                        session=session,
                    )

        assert result == ""
        assert len(mock_event_emitter.events) == 2
        content = mock_event_emitter.events[0]["data"]["content"]
        assert "Connection" in content or "🔌" in content
        assert "Error ID:" in content


class TestServiceError:
    """Test 5xx service error handling."""

    @pytest.mark.asyncio
    async def test_500_error_caught(self, mock_pipe, mock_event_emitter):
        """Test that 5xx errors are caught and formatted."""
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.reason_phrase = "Bad Gateway"

        error = httpx.HTTPStatusError("502", request=MagicMock(), response=mock_response)

        # Mock HTTP boundary - let real Registry handle the model lookup
        with aioresponses() as mock_http:
            # Mock the /models endpoint that Registry fetches from
            mock_http.get(
                "https://openrouter.ai/api/v1/models",
                payload={
                    "data": [
                        {
                            "id": "test-model",
                            "name": "Test Model",
                            "pricing": {"prompt": "0", "completion": "0"},
                        }
                    ]
                },
            )

            with patch.object(mock_pipe, '_process_transformed_request', side_effect=error):
                mock_pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
                mock_pipe.valves.API_KEY = EncryptedStr("test-key")
                mock_pipe.valves.MODEL_CATALOG_REFRESH_SECONDS = 300

                # Use real aiohttp.ClientSession so aioresponses can mock it
                async with aiohttp.ClientSession() as session:
                    result = await mock_pipe._handle_pipe_call(
                        body={"model": "test-model"},
                        __user__={"id": "test-user"},
                        __request__=MagicMock(),
                        __event_emitter__=mock_event_emitter,
                        __event_call__=None,
                        __metadata__={"model": {"id": "test-model"}},
                        __tools__=None,
                        valves=mock_pipe.valves,
                        session=session,
                    )

        assert result == ""
        assert len(mock_event_emitter.events) == 2
        content = mock_event_emitter.events[0]["data"]["content"]
        assert "Service Error" in content or "502" in content
        assert "Error ID:" in content


class TestInternalError:
    """Test generic exception handling."""

    @pytest.mark.asyncio
    async def test_generic_exception_caught(self, mock_pipe, mock_event_emitter):
        """Test that any exception is caught and formatted."""
        # Mock HTTP boundary - let real Registry handle the model lookup
        with aioresponses() as mock_http:
            # Mock the /models endpoint that Registry fetches from
            mock_http.get(
                "https://openrouter.ai/api/v1/models",
                payload={
                    "data": [
                        {
                            "id": "test-model",
                            "name": "Test Model",
                            "pricing": {"prompt": "0", "completion": "0"},
                        }
                    ]
                },
            )

            with patch.object(mock_pipe, '_process_transformed_request', side_effect=ValueError("unexpected error")):
                mock_pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
                mock_pipe.valves.API_KEY = EncryptedStr("test-key")
                mock_pipe.valves.MODEL_CATALOG_REFRESH_SECONDS = 300

                # Use real aiohttp.ClientSession so aioresponses can mock it
                async with aiohttp.ClientSession() as session:
                    result = await mock_pipe._handle_pipe_call(
                        body={"model": "test-model"},
                        __user__={"id": "test-user"},
                        __request__=MagicMock(),
                        __event_emitter__=mock_event_emitter,
                        __event_call__=None,
                        __metadata__={"model": {"id": "test-model"}},
                        __tools__=None,
                        valves=mock_pipe.valves,
                        session=session,
                    )

        assert result == ""
        assert len(mock_event_emitter.events) == 2
        content = mock_event_emitter.events[0]["data"]["content"]
        assert "Unexpected" in content or "⚠️" in content
        assert "Error ID:" in content
        assert "ValueError" in content


class TestTemplateCustomization:
    """Test that admins can customize templates via valves."""

    @pytest.mark.asyncio
    async def test_custom_template_used(self, mock_pipe, mock_event_emitter):
        """Test that custom templates from valves are used."""
        mock_pipe.valves.INTERNAL_ERROR_TEMPLATE = "Custom error: {error_type}"

        # Mock HTTP boundary - let real Registry handle the model lookup
        with aioresponses() as mock_http:
            # Mock the /models endpoint that Registry fetches from
            mock_http.get(
                "https://openrouter.ai/api/v1/models",
                payload={
                    "data": [
                        {
                            "id": "test-model",
                            "name": "Test Model",
                            "pricing": {"prompt": "0", "completion": "0"},
                        }
                    ]
                },
            )

            with patch.object(mock_pipe, '_process_transformed_request', side_effect=RuntimeError("test")):
                mock_pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
                mock_pipe.valves.API_KEY = EncryptedStr("test-key")
                mock_pipe.valves.MODEL_CATALOG_REFRESH_SECONDS = 300

                # Use real aiohttp.ClientSession so aioresponses can mock it
                async with aiohttp.ClientSession() as session:
                    result = await mock_pipe._handle_pipe_call(
                        body={"model": "test-model"},
                        __user__={"id": "test-user"},
                        __request__=MagicMock(),
                        __event_emitter__=mock_event_emitter,
                        __event_call__=None,
                        __metadata__={"model": {"id": "test-model"}},
                        __tools__=None,
                        valves=mock_pipe.valves,
                        session=session,
                    )

        content = mock_event_emitter.events[0]["data"]["content"]
        assert "Custom error" in content
        assert "RuntimeError" in content


# ===== From test_error_template_rendering.py =====

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


# ===== From openrouter/test_errors.py =====

import json

from open_webui_openrouter_pipe import (
    ModelFamily,
    OpenRouterAPIError,
    Pipe,
    _build_openrouter_api_error,
    _resolve_error_model_context,
)


def test_openrouter_api_error_includes_provider_and_request_details():
    raw_metadata = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "prompt is too long: 1015918 tokens > 1000000 maximum",
        },
        "request_id": "req_011CVgpGEQMnFmYrjfT7Zh9h",
    }
    body = json.dumps(
        {
            "error": {
                "message": "Provider returned error",
                "code": 400,
                "metadata": {
                    "provider_name": "Anthropic",
                    "raw": json.dumps(raw_metadata),
                },
            },
            "user_id": "org_123",
        }
    )

    err = _build_openrouter_api_error(400, "Bad Request", body)

    assert isinstance(err, OpenRouterAPIError)
    assert err.provider == "Anthropic"
    assert err.upstream_type == "invalid_request_error"
    assert err.request_id == raw_metadata["request_id"]

    md = err.to_markdown()
    assert "### 🚫 Anthropic could not process your request." in md
    assert "### Error:" in md
    assert raw_metadata["request_id"] in md
    assert "prompt is too long" in md


def test_openrouter_api_error_handles_plain_text_payload():
    err = _build_openrouter_api_error(400, "Bad Request", "plain text body")

    md = err.to_markdown()
    assert "could not process your request" in md
    assert "middle-out option" in md


def test_openrouter_api_error_includes_moderation_metadata():
    body = json.dumps(
        {
            "error": {
                "message": "Input was flagged by moderation",
                "code": 400,
                "metadata": {
                    "provider_name": "OpenRouter",
                    "model_slug": "anthropic/claude-3",
                    "reasons": ["violence", "hate"],
                    "flagged_input": "Some violent text…",
                },
            }
        }
    )

    err = _build_openrouter_api_error(400, "Bad Request", body)
    assert err.model_slug == "anthropic/claude-3"
    assert err.moderation_reasons == ["violence", "hate"]
    assert err.flagged_input == "Some violent text…"

    md = err.to_markdown()
    assert "**Moderation reasons:**" in md
    assert "Some violent text" in md


def test_openrouter_error_markdown_accepts_model_label_and_diagnostics():
    err = OpenRouterAPIError(
        status=400,
        reason="Bad Request",
        provider="Anthropic",
        openrouter_message="This endpoint's maximum context length is 400000 tokens. However, you requested about 564659 tokens. Please reduce the length or use the \"middle-out\" transform.",
    )
    md = err.to_markdown(
        model_label="anthropic/claude-3",
        diagnostics=["- **Context window**: 200,000 tokens"],
        metrics={"context_limit": 400000, "max_output_tokens": 128000},
        fallback_model="anthropic/claude-3",
    )
    assert "### 🚫 Anthropic: anthropic/claude-3 could not process your request." in md
    assert "Model limits" in md
    assert "400,000" in md
    assert "128,000" in md


def test_resolve_error_model_context_uses_registry_spec():
    spec = {
        "full_model": {"name": "Claude 3", "context_length": 200000},
        "context_length": 200000,
        "max_completion_tokens": 4000,
    }
    ModelFamily.set_dynamic_specs({"anthropic.claude-3": spec})
    err = OpenRouterAPIError(status=400, reason="Bad Request")
    label, diagnostics, metrics = _resolve_error_model_context(
        err,
        normalized_model_id="anthropic.claude-3",
        api_model_id="anthropic/claude-3",
    )
    assert label == "Claude 3"
    assert any("4,000" in line for line in diagnostics)
    assert metrics["max_output_tokens"] == 4000
    ModelFamily.set_dynamic_specs({})


def test_custom_error_template_skips_lines_for_missing_values():
    err = OpenRouterAPIError(status=400, reason="Bad Request")
    template = "Line A\n- Req: {request_id}\nLine B\n- Provider: {provider}"
    rendered = err.to_markdown(template=template, diagnostics=[], metrics={})
    assert rendered == "Line A\nLine B"


def test_openrouter_error_includes_metadata_and_raw_blocks():
    body = json.dumps(
        {
            "error": {
                "message": "Provider exploded",
                "code": 400,
                "metadata": {
                    "provider_name": "Anthropic",
                    "raw": {"error": {"message": "buffer overflow"}},
                },
            }
        }
    )
    err = _build_openrouter_api_error(400, "Bad Request", body)
    assert "Anthropic" in (err.metadata_json or "")
    assert "buffer overflow" in (err.provider_raw_json or "")


def test_build_openrouter_api_error_merges_extra_metadata():
    body = json.dumps({"error": {"message": "Too many requests", "code": 429}})
    err = _build_openrouter_api_error(
        429,
        "Too Many Requests",
        body,
        extra_metadata={"retry_after": "30", "rate_limit_type": "account"},
    )
    assert err.metadata.get("retry_after") == "30"
    assert err.metadata.get("rate_limit_type") == "account"


def test_streaming_fields_render_in_templates():
    err = OpenRouterAPIError(
        status=400,
        reason="Streaming error",
        provider="OpenRouter",
        native_finish_reason="rate_limit",
        chunk_id="chunk_123",
        chunk_created="2025-12-10T00:00:00Z",
        chunk_provider="OpenRouter",
        chunk_model="anthropic/claude-3",
        metadata={"foo": "bar"},
        metadata_json="{\n  \"foo\": \"bar\"\n}",
    )
    rendered = err.to_markdown(
        template=(
            "{{#if error_id}}Error {error_id}{{/if}}\n"
            "{{#if native_finish_reason}}finish:{native_finish_reason}{{/if}}\n"
            "{{#if metadata_json}}meta:{metadata_json}{{/if}}"
        ),
        metrics={},
        context={"error_id": "ERR123", "timestamp": "2025-12-10T00:00:00Z"},
    )
    assert "Error ERR123" in rendered
    assert "finish:rate_limit" in rendered
    assert "meta" in rendered


def test_pipe_builds_streaming_error_from_event(pipe_instance):
    pipe = pipe_instance
    event = {
        "type": "response.failed",
        "id": "chunk_999",
        "created": 1700000000,
        "model": "openai/gpt-4o",
        "provider": "OpenRouter",
        "error": {"code": "rate_limit", "message": "Slow down"},
        "choices": [{"native_finish_reason": "rate_limit"}],
    }
    err = pipe._ensure_error_formatter()._build_streaming_openrouter_error(event, requested_model="openai/gpt-4o")
    assert err.is_streaming_error is True
    assert err.native_finish_reason == "rate_limit"
    assert err.chunk_id == "chunk_999"
    assert err.provider == "OpenRouter"


def test_select_openrouter_template_by_status(pipe_instance):
    pipe = pipe_instance
    assert pipe._ensure_error_formatter()._select_openrouter_template(401) == pipe.valves.AUTHENTICATION_ERROR_TEMPLATE
    assert pipe._ensure_error_formatter()._select_openrouter_template(402) == pipe.valves.INSUFFICIENT_CREDITS_TEMPLATE
    assert pipe._ensure_error_formatter()._select_openrouter_template(408) == pipe.valves.SERVER_TIMEOUT_TEMPLATE
    assert pipe._ensure_error_formatter()._select_openrouter_template(429) == pipe.valves.RATE_LIMIT_TEMPLATE
    assert pipe._ensure_error_formatter()._select_openrouter_template(400) == pipe.valves.OPENROUTER_ERROR_TEMPLATE


# =============================================================================
# ErrorFormatter Coverage Gap Tests
# =============================================================================


class TestErrorFormatterNoneHandler:
    """Tests for ErrorFormatter methods when event_emitter_handler is None."""

    @pytest.mark.asyncio
    async def test_emit_error_returns_early_with_none_handler(self):
        """_emit_error returns immediately when handler is None (line 68)."""
        from open_webui_openrouter_pipe import Pipe
        from open_webui_openrouter_pipe.core.error_formatter import ErrorFormatter

        pipe = Pipe()
        try:
            # Create ErrorFormatter with None handler to test early return
            formatter = ErrorFormatter(
                pipe=pipe,
                event_emitter_handler=None,
                logger=pipe.logger,
            )
            # Should return without error (no-op)
            await formatter._emit_error(None, "test error")
        finally:
            await pipe.close()

    @pytest.mark.asyncio
    async def test_emit_templated_error_returns_early_with_none_handler(self):
        """_emit_templated_error returns immediately when handler is None (line 89)."""
        from open_webui_openrouter_pipe import Pipe
        from open_webui_openrouter_pipe.core.error_formatter import ErrorFormatter

        pipe = Pipe()
        try:
            formatter = ErrorFormatter(
                pipe=pipe,
                event_emitter_handler=None,
                logger=pipe.logger,
            )
            # Should return without error (no-op)
            await formatter._emit_templated_error(
                None,
                template="test",
                variables={},
                log_message="test",
            )
        finally:
            await pipe.close()

    def test_build_error_context_returns_empty_with_none_handler(self):
        """_build_error_context returns ('', {}) when handler is None (line 102)."""
        from open_webui_openrouter_pipe import Pipe
        from open_webui_openrouter_pipe.core.error_formatter import ErrorFormatter
        import asyncio

        pipe = Pipe()
        try:
            formatter = ErrorFormatter(
                pipe=pipe,
                event_emitter_handler=None,
                logger=pipe.logger,
            )
            error_id, context = formatter._build_error_context()
            assert error_id == ""
            assert context == {}
        finally:
            asyncio.get_event_loop().run_until_complete(pipe.close())


class TestStreamingErrorEdgeCases:
    """Tests for streaming error parsing edge cases."""

    def test_streaming_error_nested_response_error_message(self, pipe_instance):
        """Error message extracted from nested response.error.message (line 147)."""
        pipe = pipe_instance
        # Event where error_block has no message but response.error.message exists
        event = {
            "type": "response.failed",
            "error": {},  # No message here
            "response": {
                "status": "failed",
                "error": {"message": "Nested error message"},
            },
        }
        err = pipe._ensure_error_formatter()._build_streaming_openrouter_error(event, requested_model="test/model")
        assert "Nested error message" in err.reason

    def test_streaming_error_default_message(self, pipe_instance):
        """Default 'Streaming error' when no message anywhere (line 149)."""
        pipe = pipe_instance
        # Event with no message fields at all
        event = {
            "type": "error",
            "error": {"code": "unknown"},  # No message
        }
        err = pipe._ensure_error_formatter()._build_streaming_openrouter_error(event, requested_model="test/model")
        assert err.reason == "Streaming error"

    def test_streaming_error_with_response_id(self, pipe_instance):
        """Response ID is set in metadata (line 170)."""
        pipe = pipe_instance
        event = {
            "type": "response.failed",
            "response": {
                "id": "resp_12345",
                "status": "failed",
                "error": {"message": "Failed"},
            },
        }
        err = pipe._ensure_error_formatter()._build_streaming_openrouter_error(event, requested_model="test/model")
        assert err.metadata.get("request_id") == "resp_12345"

    def test_extract_streaming_error_with_none_event(self, pipe_instance):
        """_extract_streaming_error_event returns None for non-dict (line 206)."""
        pipe = pipe_instance
        assert pipe._ensure_error_formatter()._extract_streaming_error_event(None, "test/model") is None
        assert pipe._ensure_error_formatter()._extract_streaming_error_event("not a dict", "test/model") is None
        assert pipe._ensure_error_formatter()._extract_streaming_error_event(123, "test/model") is None


class TestUsageFormatting:
    """Tests for usage formatting edge cases in _format_final_status_description."""

    @pytest.mark.asyncio
    async def test_to_int_handles_bool_values(self):
        """_to_int converts True/False to 1/0 (line 314)."""
        from open_webui_openrouter_pipe import Pipe

        pipe = Pipe()
        try:
            # Usage with bool values (unusual but possible)
            usage = {"input_tokens": True, "output_tokens": False, "total_tokens": 10}
            result = pipe._ensure_error_formatter()._format_final_status_description(
                elapsed=1.0,
                stream_duration=1.0,
                total_usage=usage,
                valves=pipe.valves,
            )
            # True → 1, False → 0
            assert "Input: 1" in result
            assert "Output: 0" in result
        finally:
            await pipe.close()

    @pytest.mark.asyncio
    async def test_to_int_handles_float_values(self):
        """_to_int converts floats to ints (line 318)."""
        from open_webui_openrouter_pipe import Pipe

        pipe = Pipe()
        try:
            # Usage with float values (possible from some APIs)
            usage = {"input_tokens": 42.7, "output_tokens": 13.2, "total_tokens": 55.9}
            result = pipe._ensure_error_formatter()._format_final_status_description(
                elapsed=1.0,
                stream_duration=1.0,
                total_usage=usage,
                valves=pipe.valves,
            )
            # Floats truncated to ints
            assert "Input: 42" in result
            assert "Output: 13" in result
        finally:
            await pipe.close()

    @pytest.mark.asyncio
    async def test_tokens_without_total_uses_tokens_prefix(self):
        """When no total_tokens, uses 'Tokens:' prefix (line 356)."""
        from open_webui_openrouter_pipe import Pipe

        pipe = Pipe()
        try:
            # Usage with only detail tokens, no total
            usage = {"input_tokens": 100, "output_tokens": 50}
            result = pipe._ensure_error_formatter()._format_final_status_description(
                elapsed=1.0,
                stream_duration=1.0,
                total_usage=usage,
                valves=pipe.valves,
            )
            # Should use "Total tokens:" since we can compute total
            # But if we pass explicit None for total_tokens...
            usage_no_total = {"input_tokens": 100, "output_tokens": 50, "total_tokens": None}
            result2 = pipe._ensure_error_formatter()._format_final_status_description(
                elapsed=1.0,
                stream_duration=None,  # No TPS calculation
                total_usage=usage_no_total,
                valves=pipe.valves,
            )
            # When total_tokens is explicitly None but we have details,
            # the code computes total from candidates
            assert "150" in result2 or "Tokens:" in result2
        finally:
            await pipe.close()


class TestErrorsModuleCoverage:
    """Tests for errors.py coverage gaps."""

    def test_parse_supported_effort_values_no_match(self):
        """_parse_supported_effort_values returns [] when no match (line 342)."""
        from open_webui_openrouter_pipe.core.errors import _parse_supported_effort_values

        # Message without "Supported values are:" pattern
        result = _parse_supported_effort_values("Some random error message")
        assert result == []

        result = _parse_supported_effort_values("")
        assert result == []

        result = _parse_supported_effort_values("Supported but no values listed.")
        assert result == []

    def test_parse_supported_effort_values_with_match(self):
        """_parse_supported_effort_values extracts quoted values."""
        from open_webui_openrouter_pipe.core.errors import _parse_supported_effort_values

        msg = "Invalid effort. Supported values are: 'low', 'medium', 'high'."
        result = _parse_supported_effort_values(msg)
        assert result == ["low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_await_if_needed_with_non_awaitable(self):
        """_await_if_needed returns non-awaitables directly."""
        from open_webui_openrouter_pipe.core.utils import _await_if_needed

        # Non-awaitable values should be returned as-is
        assert await _await_if_needed("string_value") == "string_value"
        assert await _await_if_needed(42) == 42
        assert await _await_if_needed([1, 2, 3]) == [1, 2, 3]
        assert await _await_if_needed(None) is None

    @pytest.mark.asyncio
    async def test_await_if_needed_with_awaitable_no_timeout(self):
        """_await_if_needed awaits coroutines when timeout=None."""
        from open_webui_openrouter_pipe.core.utils import _await_if_needed

        async def async_value():
            return "awaited_result"

        # Awaitable with no timeout
        result = await _await_if_needed(async_value(), timeout=None)
        assert result == "awaited_result"

    @pytest.mark.asyncio
    async def test_await_if_needed_with_awaitable_and_timeout(self):
        """_await_if_needed awaits with timeout when specified."""
        from open_webui_openrouter_pipe.core.utils import _await_if_needed

        async def async_value():
            return "awaited_with_timeout"

        result = await _await_if_needed(async_value(), timeout=5.0)
        assert result == "awaited_with_timeout"
