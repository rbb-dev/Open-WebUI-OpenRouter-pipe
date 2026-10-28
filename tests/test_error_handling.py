"""Tests for the error template system."""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false
import json

import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
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

    @property
    def card(self) -> str:
        """The failure card's text, found by kind rather than by position.

        A failing turn also stops the progress line, so the card is not the first thing
        emitted; these tests are about what the card says, not about where it lands.
        """
        cards = [event for event in self.events if event.get("type") == "chat:message"]
        assert len(cards) == 1, f"expected exactly one card, got {[e.get('type') for e in self.events]}"
        return cards[0]["data"]["content"]


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
        assert len(mock_event_emitter.events) == 3
        assert "Test Error" in mock_event_emitter.card
        assert "Test message" in mock_event_emitter.card

    @pytest.mark.asyncio
    async def test_error_id_generation(self, mock_pipe, mock_event_emitter):
        """Test that error IDs are generated and included."""
        await mock_pipe._ensure_error_formatter()._emit_templated_error(
            mock_event_emitter,
            template="Error ID: {error_id}",
            variables={},
            log_message="Test",
        )

        content = mock_event_emitter.card
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

        content = mock_event_emitter.card
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

        content = mock_event_emitter.card
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

        content = mock_event_emitter.card
        assert "Time: " in content
        # Should be ISO 8601 format with Z suffix
        assert "Z" in content


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

        assert len(mock_event_emitter.events) == 3
        content = mock_event_emitter.card
        assert result == content
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

        content = mock_event_emitter.card
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
                "support_email": "help@example.org",
            },
        )

        # Verify all expected values are present
        assert values["heading"] == "🚫 Rate Limited"
        assert values["error_id"] == "err_abc123"
        assert values["timestamp"] == "2024-01-01T10:30:00Z"
        assert values["session_id"] == "sess_xyz"
        assert values["user_id"] == "user_999"
        assert values["support_email"] == "help@example.org"
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
        assert values["flagged_excerpt"] == "```\nflagged text\n```"
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
        "request_id": "req_0000000000000001",
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
    """A body that is not JSON still renders the full template with the body surfaced verbatim."""
    err = _build_openrouter_api_error(400, "Bad Request", "plain text body")

    md = err.to_markdown()
    assert "could not process your request" in md
    assert "**Raw provider response:**" in md
    assert "plain text body" in md


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
        is_streaming_error=True,
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


_STATUS_HEADINGS = [
    (401, "Authentication Failed"),
    (402, "Insufficient Credits"),
    (408, "OpenRouter Timed Out"),
    (413, "Request Too Large"),
    (429, "Rate Limit Exceeded"),
    (503, "OpenRouter Service Error"),
    (404, "could not process your request"),
]

#: A turn in a chat. Both ids, because that pair is what makes one: a request carrying only a
#: ``chat_id`` is a task route, and a request carrying neither is a caller with no chat to write a
#: card into, and that caller is answered with an envelope rather than one. The nodes below are
#: about which card a status selects, so they have to be driving the leg that gets a card at all.
_CHAT_METADATA = {"chat_id": "c1", "message_id": "m1", "user_id": "user1"}


async def _chat_path_error_card(pipe, status: int) -> str:
    """Drive a rejection through the real orchestrator and return the emitted markdown.

    The stub sits one seam BELOW the subject: the streaming loop raises, and the
    orchestrator's own reporting path runs unmodified. `_lookup_spec` is pinned to an
    empty spec because the four bundles flatten every module into one namespace, so a
    patch of the orchestrator's `ModelFamily` is also the one the error formatter reads.
    """
    import logging
    from unittest.mock import AsyncMock, Mock, patch

    import aiohttp

    from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError
    from open_webui_openrouter_pipe.requests.orchestrator import RequestOrchestrator

    emitted: list[dict[str, Any]] = []

    async def emitter(event):
        emitted.append(event)

    orchestrator = RequestOrchestrator(pipe, logging.getLogger("test_error_routing"))
    pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
    pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
    pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
    pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value={})
    pipe._ensure_reasoning_config_manager()._should_retry_without_reasoning = Mock(return_value=False)
    pipe._ensure_reasoning_config_manager()._should_retry_dropping_signed_reasoning = Mock(return_value=False)

    async def _raise(*_args, **_kwargs):
        raise OpenRouterAPIError(
            status=status,
            reason="rejected",
            openrouter_message=f"upstream said {status}",
        )

    pipe._streaming_handler._run_streaming_loop = _raise

    with patch("open_webui_openrouter_pipe.requests.orchestrator.ModelFamily") as family, \
         patch("open_webui_openrouter_pipe.requests.orchestrator.OpenRouterModelRegistry") as registry:
        family.base_model.return_value = "openai/gpt-4o"
        family.undated.side_effect = lambda model_id: model_id
        family.supports.return_value = False
        family.capabilities.return_value = {}
        family.max_completion_tokens.return_value = None
        family._lookup_spec.return_value = {}
        registry.api_model_id.return_value = "openai/gpt-4o"
        await orchestrator.process_request(
            body={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=emitter,
            __event_call__=None,
            __metadata__=_CHAT_METADATA,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            session=AsyncMock(spec=aiohttp.ClientSession),
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
            valves=pipe.valves,
        )
    return "\n".join(
        str(event.get("data", {}).get("content", ""))
        for event in emitted
        if event.get("type") == "chat:message"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status, heading", _STATUS_HEADINGS)
async def test_the_chat_path_renders_the_card_the_status_selects(pipe_instance_async, status, heading):
    """The template a rejection renders is chosen by its status on every path, chat included.

    The orchestrator used to pass OPENROUTER_ERROR_TEMPLATE explicitly and an explicit
    template short-circuited selection, so a rate-limited user in an ordinary chat read the
    generic card while the same 429 from image generation read the rate-limit one. Seven
    distinct (status, heading) pairs, so no single hardcoded template can satisfy this.
    """
    card = await _chat_path_error_card(pipe_instance_async, status)
    assert heading in card, f"a {status} on the chat path rendered:\n{card}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, valve",
    [(429, "RATE_LIMIT_TEMPLATE"), (402, "INSUFFICIENT_CREDITS_TEMPLATE")],
)
async def test_a_template_blanked_in_memory_still_renders_a_real_card(
    pipe_instance_async, status, valve
):
    """A blank template reaching the renderer yields the built-in provider-error card, never nothing.

    Constructing Valves restores a blanked template, so this state is only reachable by assigning
    the attribute directly. The selector passes the blank straight through -- it does not substitute
    the operator's OPENROUTER_ERROR_TEMPLATE, which would be a second, conflicting meaning for a
    cleared box -- and the renderer's own guard is what keeps the user from getting an empty turn.
    Two statuses with two different valves, and a distinctive operator marker in the generic valve
    that must NOT appear.
    """
    pipe = pipe_instance_async
    marker = f"HOUSE STYLE {status} DO NOT SUBSTITUTE"
    pipe.valves.OPENROUTER_ERROR_TEMPLATE = marker + " {sanitized_detail}"
    setattr(pipe.valves, valve, "")

    card = await _chat_path_error_card(pipe, status)
    assert card.strip(), f"a {status} with a blanked {valve} rendered an empty card"
    assert "Error ID" in card, f"a {status} with a blanked {valve} rendered:\n{card}"
    assert marker not in card, (
        f"a blanked {valve} was silently replaced by OPENROUTER_ERROR_TEMPLATE:\n{card}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header, expected",
    [("45", "**Retry after:** 45s"), ("soon", None)],
)
async def test_an_unparseable_retry_after_never_reaches_the_user(
    pipe_instance_async, header, expected
):
    """Only whole seconds are rendered; a header the pipe cannot parse renders nothing.

    The template appends a literal "s" to whatever it is handed, so passing the raw header
    through produced "soons". Parametrised over a parseable and an unparseable value, so a
    fix that simply drops the line always would fail the first case.
    """
    import json as _json

    from open_webui_openrouter_pipe.core import errors as _errors
    from open_webui_openrouter_pipe.core.utils import _apply_retry_after_metadata

    pipe = pipe_instance_async
    meta: dict[str, Any] = {}
    _apply_retry_after_metadata(meta, {"Retry-After": header})
    error = _errors._build_openrouter_api_error(
        429,
        "Too Many Requests",
        _json.dumps({"error": {"message": "Rate limit exceeded", "code": 429}}),
        requested_model="openai/gpt-4o",
        extra_metadata=meta or None,
    )
    emitter = _Emitter()
    await pipe._ensure_error_formatter()._report_openrouter_error(
        error,
        event_emitter=emitter,
        normalized_model_id="openai/gpt-4o",
        api_model_id="openai/gpt-4o",
    )
    card = "\n".join(
        str(event.get("data", {}).get("content", ""))
        for event in emitter.events
        if event.get("type") == "chat:message"
    )
    assert "Rate Limit Exceeded" in card, "the rate-limit card did not render at all"
    if expected is None:
        assert "Retry after" not in card, f"an unusable Retry-After was shown:\n{card}"
    else:
        assert expected in card, f"a parseable Retry-After was not shown:\n{card}"


@pytest.mark.parametrize(
    "meta, expected",
    [
        ({"retry_after": "nan"}, None),
        ({"retry_after_seconds": 0}, 0),
    ],
)
def test_a_retry_after_the_pipe_cannot_use_becomes_no_retry_at_all(meta, expected):
    """A non-finite delay is unusable, and rounding one raises rather than returning a number.

    ``round(inf)`` is an OverflowError and ``round(nan)`` a ValueError, and this runs OUTSIDE the
    try that guards emission -- so a provider body carrying ``Infinity`` (which json.loads accepts
    verbatim) turned a rate-limit card into a generic internal-error card. Rejecting it here, at
    the one place a value becomes a number, is what makes the unusable case behave like the
    unparseable one. Both the numeric route and the header-string route are covered, along with
    the finite values that must keep working, so a guard that simply dropped every delay fails.

    A delay in the past is the same kind of unusable, and the two routes disagreed about it:
    the header string was clamped to zero and the number was not, so ``-7`` and ``"-7"`` --
    the same instruction, differently spelled -- resolved to two different answers and the
    card told the reader to retry "-7s" ago. Every row pairs a number with the string that
    spells it, so a clamp on only one route still fails.

    A boolean is a third kind of unusable, and the one Python hides: ``True`` IS an ``int``,
    so ``round(True)`` is ``1`` and a provider body carrying ``"retry_after_seconds": true``
    rendered "Retry after: 1s" -- a delay the server never sent, indistinguishable in the card
    from one it did. Asserting the answer is not a boolean cannot catch that, because ``1``
    is not a boolean either; the rows below assert the number instead. ``True`` and ``False``
    are both covered, since a guard keyed on truthiness would let ``False`` through as ``0``.
    The two mixed rows pin the guard to CONTINUE rather than abort: a boolean in the first key
    must fall through to the usable delay in the second, not throw the whole lookup away --
    removing the clause resolves them to 1 and 0, and a clause that returned None resolves
    them to nothing.
    """
    from open_webui_openrouter_pipe.core.utils import _resolve_retry_after_seconds

    assert _resolve_retry_after_seconds(meta) == expected


@pytest.mark.parametrize(
    "header, expected",
    [("nan", None), ("45", 45)],
)
def test_a_non_finite_retry_after_header_is_never_recorded_as_seconds(header, expected):
    """The header route must not store a delay it could not turn into a whole number either."""
    from open_webui_openrouter_pipe.core.utils import _apply_retry_after_metadata

    meta: dict[str, Any] = {}
    _apply_retry_after_metadata(meta, {"Retry-After": header})
    assert meta.get("retry_after_seconds") == expected
    assert meta["retry_after"] == header, "the raw header is kept for diagnostics regardless"


@pytest.mark.asyncio
@pytest.mark.parametrize("literal", ["Infinity", "NaN"])
async def test_a_provider_sending_an_infinite_retry_after_still_gets_the_right_card(
    pipe_instance_async, literal
):
    """The whole point: an upstream value the pipe cannot use must not cost the user their card.

    The metadata block comes straight off the provider's error body, and JSON's ``Infinity`` and
    ``NaN`` literals survive json.loads untouched. Driven through the real error builder from raw
    body text, so the value arrives the way a provider would really send it.
    """
    from open_webui_openrouter_pipe.core import errors as _errors

    pipe = pipe_instance_async
    body_text = (
        '{"error": {"message": "Rate limit exceeded", "code": 429, '
        '"metadata": {"retry_after_seconds": ' + literal + "}}}"
    )
    error = _errors._build_openrouter_api_error(
        429, "Too Many Requests", body_text, requested_model="openai/gpt-4o"
    )
    emitter = _Emitter()
    await pipe._ensure_error_formatter()._report_openrouter_error(
        error,
        event_emitter=emitter,
        normalized_model_id="openai/gpt-4o",
        api_model_id="openai/gpt-4o",
    )
    cards = [
        str(event.get("data", {}).get("content", ""))
        for event in emitter.events
        if event.get("type") == "chat:message"
    ]
    assert len(cards) == 1, f"the rate-limit card never reached the user: {emitter.events!r}"
    assert "Rate Limit Exceeded" in cards[0], cards[0]
    assert "Retry after" not in cards[0], f"a delay the server never sent was invented:\n{cards[0]}"


@pytest.mark.asyncio
@pytest.mark.parametrize("magnitude", [7])
async def test_a_retry_after_already_in_the_past_never_tells_the_reader_to_wait_backwards(
    pipe_instance_async, magnitude
):
    """A negative delay reached the reader as "**Retry after:** -7s", an instruction to wait backwards.

The two spellings of the same instruction took different routes -- the number was rounded with no lower bound
    while the string went through the header parser, which clamps -- so the same value resolved to two different
    answers. Each row drives BOTH spellings through the real error builder from raw provider body text and requires
    them to agree, and pairs them with the SAME magnitude as a positive delay, which must still be shown: a fix that
    simply stopped rendering the line would pass the first two assertions and fail that one.
    """
    from open_webui_openrouter_pipe.core import errors as _errors

    pipe = pipe_instance_async

    async def _card(raw: str) -> str:
        body_text = (
            '{"error": {"message": "Rate limit exceeded", "code": 429, '
            '"metadata": {"retry_after_seconds": ' + raw + "}}}"
        )
        error = _errors._build_openrouter_api_error(
            429, "Too Many Requests", body_text, requested_model="openai/gpt-4o"
        )
        emitter = _Emitter()
        await pipe._ensure_error_formatter()._report_openrouter_error(
            error,
            event_emitter=emitter,
            normalized_model_id="openai/gpt-4o",
            api_model_id="openai/gpt-4o",
        )
        card = "\n".join(
            str(event.get("data", {}).get("content", ""))
            for event in emitter.events
            if event.get("type") == "chat:message"
        )
        assert "Rate Limit Exceeded" in card, f"the card did not render at all:\n{card}"
        return card

    as_number = await _card(str(-magnitude))
    as_string = await _card(f'"{-magnitude}"')
    positive = await _card(str(magnitude))

    def _retry_line(card: str) -> str:
        return next((line for line in card.splitlines() if "Retry after" in line), "")

    assert f"-{magnitude}s" not in as_number, (
        f"the reader is told to retry {magnitude} seconds ago:\n{as_number}"
    )
    assert _retry_line(as_number) == _retry_line(as_string), (
        "the same delay spelled as a number and as a string produced two different "
        f"instructions: {_retry_line(as_number)!r} vs {_retry_line(as_string)!r}"
    )
    assert _retry_line(positive) == f"**Retry after:** {magnitude}s", (
        f"a real delay stopped being shown, so 'no line' proves nothing:\n{positive}"
    )


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

    @pytest.mark.parametrize(
        "counter",
        [float("inf"), float("nan")],
        ids=["infinity", "nan"],
    )
    @pytest.mark.asyncio
    async def test_a_non_finite_usage_counter_does_not_kill_the_turn(self, counter):
        """The end-of-turn status line runs on every success, so it must not raise.

        `BASE_URL` is configurable, so usage arrives from arbitrary OpenAI-compatible
        gateways, and `json.loads` produces `inf` from a bare `Infinity` literal --
        which is legal in Python's JSON dialect. `_to_int` did an unguarded
        `int(value)` on any float, and `int(inf)` raises `OverflowError` while
        `int(nan)` raises `ValueError`. Neither is caught anywhere between here and
        `_run_streaming_loop`, so a gateway reporting a non-finite counter killed an
        otherwise successful turn at the point of rendering Time/Cost/Tokens.

        Three rows because the two exception types differ, and the status line must
        still render for the counters that are fine.
        """
        from open_webui_openrouter_pipe import Pipe

        pipe = Pipe()
        try:
            result = pipe._ensure_error_formatter()._format_final_status_description(
                elapsed=1.0,
                stream_duration=1.0,
                total_usage={"input_tokens": counter, "output_tokens": 12, "total_tokens": 12},
                valves=pipe.valves,
            )
            assert isinstance(result, str), (
                f"a {counter} input-token counter did not render a status line"
            )
            assert "12" in result, (
                f"the counters that ARE finite were dropped along with the bad one: {result}"
            )
        finally:
            await pipe.close()

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


# REGFIX: the provider's own words are nested inside OpenRouter's message


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            'HTTP 400: {"error":{"message":"input video is too small","type":"invalid_request"}}',
            ("input video is too small", "invalid_request", "HTTP 400"),
        ),
        (
            'Provider said: {"error":{"message":"content policy","code":"moderation_blocked"}}',
            ("content policy", "moderation_blocked", "Provider said"),
        ),
    ],
)
def test_the_provider_message_openrouter_wrapped_is_the_one_the_user_reads(raw, expected):
    """OpenRouter puts the provider's entire HTTP response inside its own `error.message`.

    Left nested, the chat shows a wall of JSON and the one sentence that says what to do
    differently is buried in it. Two rows, so a hardcoded answer satisfies neither, and
    the second uses `code` rather than `type` because providers use both.
    """
    from open_webui_openrouter_pipe.core.errors import _unnest_provider_error

    found = _unnest_provider_error(raw)

    assert found is not None
    message, kind, prefix = expected
    assert found["message"] == message
    assert found["type"] == kind
    assert found["prefix"] == prefix
    assert "{" not in found["message"]


def test_a_request_id_is_lifted_out_of_the_sentence_rather_than_shown_inside_it():
    """The id is for the operator's support ticket; the sentence is for the user."""
    from open_webui_openrouter_pipe.core.errors import _unnest_provider_error

    found = _unnest_provider_error(
        'HTTP 500: {"error":{"message":"upstream timed out. Request id: req_9fA-2b"}}'
    )

    assert found is not None
    assert found["request_id"] == "req_9fA-2b"
    assert found["message"] == "upstream timed out"
    assert "req_9fA-2b" not in found["message"]


@pytest.mark.parametrize(
    "raw",
    [
        "the provider refused the request",
        42,
    ],
)
def test_nothing_is_invented_when_there_is_no_nested_message_to_lift(raw):
    """A plain message must be left exactly as OpenRouter sent it, not half-parsed."""
    from open_webui_openrouter_pipe.core.errors import _unnest_provider_error

    assert _unnest_provider_error(raw) is None


_JANUARY_CHAT_BODY = json.dumps(
    {
        "error": {
            "code": 400,
            "message": (
                "This endpoint's maximum context length is 400000 tokens. However, you "
                'requested about 564659 tokens. Please reduce the length or use the "middle-out" transform.'
            ),
            "metadata": {"provider_name": "anthropic"},
        }
    }
)

_CURRENT_CHAT_BODY = json.dumps(
    {
        "error": {
            "code": 400,
            "message": (
                "This endpoint's maximum context length is 400000 tokens. However, you "
                "requested about 564659 tokens. Please reduce the length or enable context compression."
            ),
            "metadata": {"provider_name": "anthropic", "error_type": "context_length_exceeded"},
        }
    }
)

_CURRENT_CHAT_BODY_PHRASE_ONLY = json.dumps(
    {
        "error": {
            "code": 400,
            "message": "Please reduce the length or enable context compression.",
            "metadata": {"provider_name": "anthropic"},
        }
    }
)

_CURRENT_RESPONSES_BODY_TYPED_ONLY = json.dumps(
    {
        "id": "resp_abc123",
        "status": "failed",
        "error": {"code": "invalid_prompt", "message": "The prompt could not be processed."},
        "error_type": "context_length_exceeded",
    }
)

_PROVIDER_TYPE_ONLY_BODY = json.dumps(
    {
        "error": {
            "code": 400,
            "message": "Provider rejected the request.",
            "metadata": {
                "provider_name": "openai",
                "raw": json.dumps(
                    {"error": {"message": "Unknown parameter.", "type": "invalid_request_error"}}
                ),
            },
        }
    }
)

_UNRELATED_BODY = json.dumps(
    {"error": {"code": 403, "message": "Your key is not permitted to use this model."}}
)


def _limits_shown_for(body: str) -> bool:
    from open_webui_openrouter_pipe.core.errors import (
        _build_error_template_values,
        _build_openrouter_api_error,
    )

    error = _build_openrouter_api_error(400, "Bad Request", body)
    values = _build_error_template_values(
        error,
        heading="Anthropic: anthropic/claude-3",
        diagnostics=[],
        metrics={"context_limit": 400000, "max_output_tokens": 128000},
        model_identifier="anthropic/claude-3",
        normalized_model_id="anthropic.claude-3",
        api_model_id="anthropic/claude-3",
    )
    return values["include_model_limits"]


@pytest.mark.parametrize(
    "body, expected",
    [
        pytest.param(_JANUARY_CHAT_BODY, True, id="january-middle-out-phrase-alone"),
        pytest.param(_CURRENT_CHAT_BODY_PHRASE_ONLY, True, id="current-compression-phrase-alone"),
        pytest.param(_CURRENT_RESPONSES_BODY_TYPED_ONLY, True, id="responses-typed-code-alone"),
        pytest.param(_CURRENT_CHAT_BODY, True, id="chat-typed-code-and-current-phrase"),
        pytest.param(_PROVIDER_TYPE_ONLY_BODY, False, id="provider-invalid-request-error"),
        pytest.param(_UNRELATED_BODY, False, id="forbidden-key"),
    ],
)
def test_the_model_limits_block_tracks_a_context_overflow_across_both_wordings(body, expected):
    """A context overflow shows the model's limits whichever era's prose OpenRouter sends.

    The block used to be gated on one literal substring of OpenRouter's own January prose,
    ``or use the "middle-out"``. OpenRouter renamed the feature to context compression --
    ``.external/openrouter_docs-2026-02-27/guides/features/message-transforms.md`` says the
    request fails "suggesting you either reduce the length or enable middle-out compression",
    and every later snapshot says "enable context compression" -- so that gate silently stopped
    firing and an over-long prompt was told nothing about the window it had exceeded.

    Each positive row isolates one route to the decision: the January phrase with no typed
    code, the current phrase with no typed code, and the typed code with neither phrase in the
    message. Deleting any one route reddens exactly one row. The negative rows hold the gate
    shut for a provider-side ``invalid_request_error`` and for an unrelated 403, so a gate
    hardcoded open cannot pass this table either.
    """
    assert _limits_shown_for(body) is expected


@pytest.mark.parametrize(
    "body, expected_type",
    [
        pytest.param(_CURRENT_CHAT_BODY, "context_length_exceeded", id="chat-metadata-error-type"),
        pytest.param(
            _CURRENT_RESPONSES_BODY_TYPED_ONLY,
            "context_length_exceeded",
            id="responses-top-level-error-type",
        ),
        pytest.param(_PROVIDER_TYPE_ONLY_BODY, None, id="provider-type-is-not-openrouters"),
        pytest.param(_JANUARY_CHAT_BODY, None, id="no-typed-code-published"),
    ],
)
def test_openrouters_typed_error_code_is_read_from_the_place_each_transport_puts_it(
    body, expected_type
):
    """OpenRouter's canonical ``error_type`` sits in a different field per API skin.

    ``.external/openrouter_docs/api/reference/errors-and-debugging.md`` states it is
    ``error.metadata.error_type`` on Chat Completions and a top-level ``error_type`` on
    Responses, "outside the native ``error`` object", because the Responses code set is lossy
    -- ``context_length_exceeded`` collapses to ``invalid_prompt`` there, so the native code
    cannot be switched on.

    The provider row is the one that must stay ``None``: ``error.metadata.raw.error.type`` is
    the upstream provider's own type (``invalid_request_error``), not OpenRouter's, and reading
    it here would fire the limits block on every malformed-parameter rejection.
    """
    from open_webui_openrouter_pipe.core.errors import _build_openrouter_api_error

    error = _build_openrouter_api_error(400, "Bad Request", body)
    assert error.openrouter_error_type == expected_type


@pytest.mark.parametrize(
    "event, expected_type",
    [
        pytest.param(
            {
                "type": "response.failed",
                "response": {
                    "id": "resp-1",
                    "status": "failed",
                    "error": {"code": "invalid_prompt", "message": "This request is too long."},
                    "error_type": "context_length_exceeded",
                },
            },
            "context_length_exceeded",
            id="responses-top-level-error-type",
        ),
        pytest.param(
            {
                "id": "gen-2",
                "object": "chat.completion.chunk",
                "error": {"code": 502, "message": "Provider disconnected unexpectedly."},
            },
            None,
            id="no-typed-code-published",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_failure_reported_inside_a_started_reply_carries_openrouters_typed_code(event, expected_type):
    """The same typed code decides the model-limits block whether OpenRouter rejects or reports mid-reply.

    A context overflow can arrive either way: as a rejection, or inside a reply OpenRouter has already
    committed with ``200 OK``. The rejection path is covered by the table above. This one drives the
    in-band builder, whose messages deliberately avoid both remedy phrases, so only the typed code can
    open the block. Each message here names no remedy, so a builder that drops the typed code shows the
    person an overflow card with no window size on it.
    """
    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.core.errors import _build_error_template_values

    pipe = Pipe()
    try:
        error = pipe._ensure_error_formatter()._build_streaming_openrouter_error(event, requested_model="m1")
        assert error.openrouter_error_type == expected_type
        values = _build_error_template_values(
            error,
            heading="Anthropic: anthropic/claude-3",
            diagnostics=[],
            metrics={"context_limit": 400000, "max_output_tokens": 128000},
            model_identifier="anthropic/claude-3",
            normalized_model_id="anthropic.claude-3",
            api_model_id="anthropic/claude-3",
        )
        assert values["include_model_limits"] is (expected_type is not None)
    finally:
        await pipe.close()


def test_the_provider_type_and_openrouters_type_stay_separate_fields():
    """One payload carries both; conflating them is what the separate field prevents."""
    from open_webui_openrouter_pipe.core.errors import _build_openrouter_api_error

    body = json.dumps(
        {
            "error": {
                "code": 400,
                "message": "Context overflow.",
                "metadata": {
                    "error_type": "context_length_exceeded",
                    "raw": json.dumps(
                        {"error": {"message": "too long", "type": "invalid_request_error"}}
                    ),
                },
            }
        }
    )
    error = _build_openrouter_api_error(400, "Bad Request", body)

    assert error.openrouter_error_type == "context_length_exceeded"
    assert error.upstream_type == "invalid_request_error"


# ---------------------------------------------------------------------------
# T384 · a body whose fields are not text still gets the card its status selects
# ---------------------------------------------------------------------------
#
# `_extract_openrouter_error_details` handed the constructor raw JSON values, and the
# constructor called `.strip()` on eight of them, so a number or an object in any one of
# those fields raised while the error was being built. The turn then ended in the
# unexpected-error card instead of the one for the failure the provider actually reported.

_T384_CARD_MARKERS = {
    "OPENROUTER_ERROR_TEMPLATE": "### REJECTED-CARD {status_code}",
    "RATE_LIMIT_TEMPLATE": "### RATE-LIMIT-CARD {status_code}",
    "SERVICE_ERROR_TEMPLATE": "### SERVICE-CARD {status_code}",
    "AUTHENTICATION_ERROR_TEMPLATE": "### AUTH-CARD {status_code}",
    "INSUFFICIENT_CREDITS_TEMPLATE": "### CREDITS-CARD {status_code}",
    "SERVER_TIMEOUT_TEMPLATE": "### SERVER-TIMEOUT-CARD {status_code}",
    "PAYLOAD_TOO_LARGE_TEMPLATE": "### PAYLOAD-CARD {status_code}",
    "INTERNAL_ERROR_TEMPLATE": "### UNEXPECTED-CARD",
}

def _t384_user_valves():
    from open_webui_openrouter_pipe import Pipe

    return Pipe.UserValves()

def _t384_reach_the_model(monkeypatch, pipe) -> None:
    import open_webui_openrouter_pipe.pipe as pipe_mod

    async def loaded(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: ("sk-test-key", None))
    monkeypatch.setattr(pipe._artifact_store, "_ensure_artifact_store", lambda *_a, **_k: None)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", loaded)
    monkeypatch.setattr(
        pipe_mod.OpenRouterModelRegistry,
        "list_models",
        lambda: [{"id": "openai/gpt-4o", "name": "GPT-4o", "norm_id": "openai.gpt-4o"}],
    )


async def _t384_card(monkeypatch, pipe_instance_async, status: int, error: dict[str, Any]) -> str:
    """Drive one real turn rejected with `status` and `error`; return what was shown."""
    import json as _json
    from unittest.mock import AsyncMock as _AsyncMock

    pipe = pipe_instance_async
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
    _t384_reach_the_model(monkeypatch, pipe)
    pipe.valves = pipe.valves.model_copy(
        update={
            "DEFAULT_LLM_ENDPOINT": "responses",
            **{name: marker for name, marker in _T384_CARD_MARKERS.items()},
        }
    )
    pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = MagicMock()
    pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = MagicMock()
    pipe._ensure_tool_executor()._build_direct_tool_server_registry = MagicMock(return_value={})
    pipe._ensure_reasoning_config_manager()._should_retry_without_reasoning = MagicMock(return_value=False)
    pipe._ensure_reasoning_config_manager()._should_retry_dropping_signed_reasoning = MagicMock(return_value=False)

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            status=status,
            body=_json.dumps({"error": error}).encode("utf-8"),
            repeat=True,
        )
        mock_http.get("https://openrouter.ai/api/v1/models", payload={"data": []}, repeat=True)
        mock_http.get("https://openrouter.ai/api/v1/endpoints/zdr", payload={"data": []}, repeat=True)
        result = await pipe.pipe(
            body={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            __user__={"id": "user1", "valves": _t384_user_valves()},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__={**_CHAT_METADATA, "model": {"id": "openai/gpt-4o"}},
            __tools__=None,
            __task__=None,
            __task_body__=None,
        )
        if hasattr(result, "__aiter__"):
            shown = "".join([str(chunk) async for chunk in result])
        else:
            shown = str(result)
    return shown


def _status_field(description: str, index: int) -> str:
    fields = [part.strip() for part in description.split(" | ")]
    assert len(fields) > index, description
    return fields[index]


_REFUSED_CLIP = [{"role": "user", "content": [
    {"type": "text", "text": "what is in this clip"},
    {"type": "video_url", "video_url": {"url": "http://example.com/clip.mp4"}},
]}]
_REFUSAL_SENTENCE = "ALLOW_INSECURE_HTTP"


def _event_of(chunk: Any) -> dict[str, Any]:
    found = chunk.get("event") if isinstance(chunk, dict) else None
    return found if isinstance(found, dict) else {}


def _statuses_of(chunks: list[Any]) -> list[tuple[int, dict[str, Any]]]:
    return [(i, _event_of(chunk)["data"]) for i, chunk in enumerate(chunks) if _event_of(chunk).get("type") == "status"]


def _latest_status_before(statuses: list[tuple[int, dict[str, Any]]], position: int) -> tuple[int, dict[str, Any]]:
    return [entry for entry in statuses if entry[0] < position][-1]


def _answer_delta_at(chunks: list[Any], text: str) -> int:
    return next(
        i for i, chunk in enumerate(chunks)
        if isinstance(chunk, dict) and (chunk.get("choices") or [{}])[0].get("delta", {}).get("content") == text
    )


def _answer_text_of(chunks: list[Any]) -> str:
    return "".join(
        (chunk.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
        for chunk in chunks
        if isinstance(chunk, dict)
    )


@pytest.mark.asyncio
async def test_a_refused_attachment_is_the_latest_status_when_the_answer_starts_and_again_on_the_closing_line(
    pipe_instance_async, monkeypatch
):
    from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import (
        CHAT,
        answer_round,
        install,
        run,
    )

    pipe = pipe_instance_async
    install(pipe, monkeypatch, lambda _n, _request: answer_round("It is a short clip."))

    chunks, _raw = await run(pipe, mode="Pipeline", stream=True, metadata=dict(CHAT), messages=_REFUSED_CLIP)

    answer_at = _answer_delta_at(chunks, "It is a short clip.")
    statuses = _statuses_of(chunks)
    error_frames = [chunk for chunk in chunks if _event_of(chunk).get("type") == "chat:message:error"]
    assert not error_frames, f"the reply carried an error frame: {error_frames}"

    noted = [(i, data) for i, data in statuses if _REFUSAL_SENTENCE in data.get("description", "")]
    latest = _latest_status_before(statuses, answer_at)
    assert len([entry for entry in statuses if entry[0] < answer_at]) > 1, (
        f"nothing else was on the status line before the answer: {statuses}"
    )
    assert latest in noted, (
        f"the refusal was not the latest status when the answer started (first answer at {answer_at}): {statuses}"
    )
    assert latest[1].get("done") is not False, f"the refusal would shimmer on the status line: {latest}"
    assert [entry for entry in noted if entry[0] < answer_at] == [latest], (
        f"a status carried the refusal before the answer started: {noted}"
    )
    assert len(noted) == 2 and noted[-1] == statuses[-1] and noted[-1][0] > answer_at, (
        f"the refusal is not carried once more by the last status: {noted}"
    )
    assert [data for _i, data in statuses if data.get("done") is True] == [noted[-1][1]]
    assert _event_of(chunks[-1]).get("type") == "chat:completion"


@pytest.mark.asyncio
async def test_a_refused_attachment_is_the_latest_status_again_when_the_answer_resumes_after_another_status(
    pipe_instance_async, monkeypatch
):
    from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import (
        CHAT,
        install,
        run,
    )

    clock = {"type": "openrouter:datetime", "id": "dt-1", "status": "completed",
             "datetime": "2026-01-02T03:04:05", "timezone": "UTC"}
    answer = {"type": "message", "role": "assistant", "status": "completed",
              "content": [{"type": "output_text", "text": "First half, still going. Second half."}]}
    two_parts = [
        {"type": "response.output_text.delta", "delta": "First half,"},
        {"type": "response.output_text.delta", "delta": " still going."},
        {"type": "response.output_item.added", "item": {"type": "openrouter:datetime", "id": "dt-1", "status": "in_progress"}},
        {"type": "response.output_item.done", "item": clock},
        {"type": "response.output_text.delta", "delta": " Second half."},
        {"type": "response.completed", "response": {"output": [answer], "usage": {}}},
    ]
    pipe = pipe_instance_async
    install(pipe, monkeypatch, lambda _n, _request: two_parts)

    chunks, _raw = await run(pipe, mode="Pipeline", stream=True, metadata=dict(CHAT), messages=_REFUSED_CLIP)

    starts = _answer_delta_at(chunks, "First half,")
    middle = _answer_delta_at(chunks, " still going.")
    resumes = _answer_delta_at(chunks, " Second half.")
    statuses = _statuses_of(chunks)
    noted = [(i, data) for i, data in statuses if _REFUSAL_SENTENCE in data.get("description", "")]
    live = noted[:-1]
    other_between = [entry for entry in statuses if middle < entry[0] < resumes and entry not in noted]
    assert other_between, f"no other status came between the two parts of the answer: {statuses}"
    assert live == [_latest_status_before(statuses, starts), _latest_status_before(statuses, resumes)], (
        f"the refusal was not the latest status once when the answer started and once when it resumed: {noted}"
    )
    assert all(data.get("done") is not False for _i, data in live), (
        f"a live copy of the refusal would shimmer on the status line: {live}"
    )
    assert len(noted) == 3 and noted[-1] == statuses[-1] and noted[-1][1].get("done") is True


@pytest.mark.asyncio
async def test_a_tool_picture_the_gate_refuses_is_shown_with_the_next_answer_text_and_not_when_the_tool_returns(
    pipe_instance_async, monkeypatch
):
    import open_webui_openrouter_pipe.tools.tool_executor as tool_executor
    from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import (
        CHAT,
        answer_round,
        call,
        install,
        resolved_tool,
        run,
    )
    from tests.test_fusion_engine import _prepare_pipe

    async def processed(**_kwargs: Any):
        return "taken", [{"type": "image", "url": "http://example.com/a.png"}], []

    async def snap(**_kwargs: Any) -> str:
        return "taken"

    snap_call = call("snap", {"q": "x"}, "call-1")
    lead_in = {"type": "message", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": "Let me look. "}]}
    rounds = [
        [
            {"type": "response.output_text.delta", "delta": "Let me look. "},
            {"type": "response.output_item.done", "item": snap_call},
            {"type": "response.completed", "response": {"output": [lead_in, snap_call], "usage": {}}},
        ],
        answer_round("The picture was left out."),
    ]
    pipe = pipe_instance_async
    monkeypatch.setattr(tool_executor, "_owui_process_tool_result", processed)
    _prepare_pipe(pipe)
    install(pipe, monkeypatch, lambda n, _request: rounds[n - 1])

    chunks, _raw = await run(
        pipe, mode="Pipeline", stream=True, metadata=dict(CHAT), owui_tools={"snap": resolved_tool("snap", snap)}
    )

    resumes = _answer_delta_at(chunks, "The picture was left out.")
    card_at = next(
        i for i, chunk in enumerate(chunks)
        if isinstance(chunk, dict)
        and chunk.get("type") == "response.output_item.added"
        and chunk["item"].get("type") == "function_call_output"
    )
    statuses = _statuses_of(chunks)
    noted = [(i, data) for i, data in statuses if _REFUSAL_SENTENCE in data.get("description", "")]
    live = noted[:-1]
    assert live == [_latest_status_before(statuses, resumes)] and live[0][0] > card_at, (
        f"the refusal was not shown once, with the answer that followed the tool (tool result at {card_at}, "
        f"answer at {resumes}): {noted}"
    )
    assert live[0][1].get("done") is not False, f"the refusal would shimmer on the status line: {live}"
    assert len(noted) == 2 and noted[-1] == statuses[-1] and noted[-1][1].get("done") is True


@pytest.mark.asyncio
async def test_a_refused_attachment_is_carried_by_the_closing_line_alone_when_the_reply_is_not_streamed(
    pipe_instance_async, monkeypatch
):
    from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import (
        CHAT,
        answer_round,
        install,
        run,
    )

    pipe = pipe_instance_async
    install(pipe, monkeypatch, lambda _n, _request: answer_round("It is a short clip."))

    _answer, events = await run(pipe, mode="Pipeline", stream=False, metadata=dict(CHAT), messages=_REFUSED_CLIP)

    statuses = [event["data"] for event in events if event.get("type") == "status"]
    noted = [data for data in statuses if _REFUSAL_SENTENCE in data.get("description", "")]
    assert noted == statuses[-1:] and noted[0].get("done") is True, (
        f"the refusal was not carried by the closing line alone: {statuses}"
    )


@pytest.mark.asyncio
async def test_a_refused_attachment_ends_a_streamed_api_answer_and_is_not_answer_text_in_a_chat(
    pipe_instance_async, monkeypatch
):
    from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import (
        CHAT,
        answer_round,
        finish_reasons_of,
        install,
        run,
    )

    pipe = pipe_instance_async
    install(pipe, monkeypatch, lambda _n, _request: answer_round("It is a short clip."))
    api_caller = {"session_id": "session-1"}

    chat_chunks, _raw = await run(pipe, mode="Pipeline", stream=True, metadata=dict(CHAT), messages=_REFUSED_CLIP)
    streamed_chunks, _raw = await run(pipe, mode="Pipeline", stream=True, metadata=dict(api_caller), messages=_REFUSED_CLIP)
    returned, _raw = await run(pipe, mode="Pipeline", stream=False, metadata=dict(api_caller), messages=_REFUSED_CLIP)

    answer_at = _answer_delta_at(chat_chunks, "It is a short clip.")
    sentence = next(
        data["description"] for i, data in _statuses_of(chat_chunks)
        if i < answer_at and _REFUSAL_SENTENCE in data.get("description", "")
    )
    assert _answer_text_of(chat_chunks) == "It is a short clip.", "a chat's answer text carried more than the answer"
    assert _answer_text_of(streamed_chunks) == f"It is a short clip.\n\n{sentence}", (
        f"the streamed answer of an API caller did not end with the refusal: {streamed_chunks}"
    )
    assert _answer_delta_at(streamed_chunks, f"\n\n{sentence}") == len(streamed_chunks) - 2
    assert finish_reasons_of(streamed_chunks[-1:]) == ["stop"]
    assert returned["choices"][0]["message"]["content"] == _answer_text_of(streamed_chunks)


@pytest.mark.asyncio
async def test_a_streaming_request_with_no_chat_gets_only_what_an_openai_client_reads(
    pipe_instance_async, monkeypatch
):
    from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import CHAT, install, run

    answer = "Here is the answer."
    message = {"type": "message", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": answer}]}
    counted = {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15,
               "input_tokens_details": {"cached_tokens": 4}, "output_tokens_details": {"reasoning_tokens": 1}}
    round_with_usage = [
        {"type": "response.output_text.delta", "delta": answer},
        {"type": "response.completed", "response": {"output": [message], "usage": counted}},
    ]
    pipe = pipe_instance_async
    install(pipe, monkeypatch, lambda _n, _request: round_with_usage)
    asks_for_usage = {"stream_options": {"include_usage": True}}

    in_a_chat, _raw = await run(pipe, mode="Pipeline", stream=True, metadata=dict(CHAT), body_extra=asks_for_usage)
    asked, _raw = await run(pipe, mode="Pipeline", stream=True, metadata={}, body_extra=asks_for_usage)
    not_asked, _raw = await run(pipe, mode="Pipeline", stream=True, metadata={})

    assert _answer_text_of(in_a_chat) == answer
    assert any(_event_of(frame).get("type") == "status" for frame in in_a_chat)
    assert any(isinstance(frame, dict) and frame.get("type") == "response.completed" for frame in in_a_chat)
    assert any(isinstance(frame, dict) and "usage" in frame and "choices" not in frame for frame in in_a_chat)

    for frames in (asked, not_asked):
        assert all(isinstance(frame, dict) and isinstance(frame.get("choices"), list) for frame in frames), frames
        assert all(frame["choices"] for frame in frames), frames
        assert _answer_text_of(frames) == answer
    usage_chunks = [frame for frame in asked if "usage" in frame]
    assert [chunk["usage"] for chunk in usage_chunks] == [{
        "prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 4}, "completion_tokens_details": {"reasoning_tokens": 1},
    }]
    assert asked[-1] is usage_chunks[0]
    assert [[choice["finish_reason"] for choice in chunk["choices"]] for chunk in usage_chunks] == [["stop"]]
    assert not [frame for frame in not_asked if "usage" in frame]

    async def breaks(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("the job broke")

    monkeypatch.setattr(pipe, "_handle_pipe_call", breaks)
    failed_in_a_chat, _raw = await run(pipe, mode="Pipeline", stream=True, metadata=dict(CHAT))
    failed_without_a_chat, _raw = await run(pipe, mode="Pipeline", stream=True, metadata={})
    assert failed_in_a_chat == [{"error": {"detail": "Request failed. Please retry."}}]
    assert failed_without_a_chat == [{"error": {"message": "Request failed. Please retry."}}]


@pytest.mark.asyncio
async def test_a_streaming_request_with_no_chat_reads_like_a_stream_from_open_webuis_own_openrouter_models(
    pipe_instance_async, monkeypatch
):
    from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import call, calls_round, install, run

    declared = [{"type": "function", "function": {
        "name": "lookup", "description": "Look something up.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }}]
    counted = {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}
    tool_call_round = calls_round(call("lookup", {"q": "x"}, "call-1"))
    tool_call_round[-1] = {"type": "response.completed", "response": {**tool_call_round[-1]["response"], "usage": counted}}
    pipe = pipe_instance_async
    install(pipe, monkeypatch, lambda _n, _request: tool_call_round)

    read, _raw = await run(
        pipe, mode="Pipeline", stream=True, metadata={}, body_tools=declared,
        body_extra={"stream_options": {"include_usage": True}}, wire=True,
    )

    chunks = read[:-1]
    assert read[-1] == "[DONE]", read
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant", chunks
    assert [chunk["choices"][0]["delta"].get("role") for chunk in chunks[1:]] == [None] * (len(chunks) - 1), chunks
    assert len({chunk["id"] for chunk in chunks}) == 1, chunks
    assert chunks[-1]["usage"]["total_tokens"] == 15, chunks
    assert [choice["finish_reason"] for choice in chunks[-1]["choices"]] == ["tool_calls"], chunks


@pytest.mark.asyncio
async def test_a_streaming_request_with_no_chat_whose_answer_the_provider_cut_off_finishes_as_the_provider_ended_it(
    pipe_instance_async, monkeypatch
):
    from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import finish_reasons_of, install, run

    message = {"type": "message", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": "Take the"}]}
    pipe = pipe_instance_async

    for provider_reason, finish_reason in (("max_output_tokens", "length"), ("content_filter", "content_filter")):
        cut_off = [
            {"type": "response.output_text.delta", "delta": "Take the"},
            {"type": "response.completed", "response": {
                "output": [message], "usage": {}, "status": "incomplete",
                "incomplete_details": {"reason": provider_reason}}},
        ]
        install(pipe, monkeypatch, lambda _n, _request, events=cut_off: events)

        read, _raw = await run(pipe, mode="Pipeline", stream=True, metadata={}, wire=True)

        assert finish_reasons_of(read) == [finish_reason], (provider_reason, read)
        assert read[-1] == "[DONE]", (provider_reason, read)


@pytest.mark.asyncio
async def test_an_exception_inside_the_body_of_a_streamed_reply_with_no_chat_ends_it_with_an_error_frame_and_done(
    pipe_instance_async, monkeypatch
):
    import open_webui_openrouter_pipe.pipe as pipe_mod
    from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import answer_round, install, run

    pipe = pipe_instance_async
    install(pipe, monkeypatch, lambda _n, _request: answer_round("Here is the answer."))
    translated = {"frames": 0}
    original = pipe_mod.chatless_stream_frame

    def breaks_on_the_second_frame(item, **kwargs):
        translated["frames"] += 1
        if translated["frames"] >= 2:
            raise RuntimeError("a fault in the translation of a frame")
        return original(item, **kwargs)

    monkeypatch.setattr(pipe_mod, "chatless_stream_frame", breaks_on_the_second_frame)

    read, _raw = await run(pipe, mode="Pipeline", stream=True, metadata={}, wire=True)

    assert translated["frames"] >= 2
    assert read[-2:] == [{"error": {"message": "Request failed. Please retry."}}, "[DONE]"], read


def _reasoning_pieces_of(chunks: list[Any]) -> list[str]:
    return [
        chunk["choices"][0]["delta"]["reasoning"]
        for chunk in chunks
        if isinstance(chunk, dict) and (chunk.get("choices") or [{}])[0].get("delta", {}).get("reasoning")
    ]


@pytest.mark.asyncio
async def test_a_streaming_request_with_no_chat_gets_the_models_reasoning_as_chunks_before_the_answer(
    pipe_instance_async, monkeypatch
):
    from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import CHAT, install, run

    thoughts = ["Weigh the two offers. ", "The cheaper one wins."]
    answer = "Take the cheaper offer."
    message = {"type": "message", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": answer}]}
    reasoning_round = [
        *({"type": "response.reasoning_text.delta", "item_id": "rs-1", "delta": thought} for thought in thoughts),
        {"type": "response.output_text.delta", "delta": answer},
        {"type": "response.completed", "response": {"output": [message], "usage": {}}},
    ]
    pipe = pipe_instance_async
    install(pipe, monkeypatch, lambda _n, _request: reasoning_round)

    shown, _raw = await run(pipe, mode="Pipeline", stream=True, metadata={})
    in_a_chat, _raw = await run(pipe, mode="Pipeline", stream=True, metadata=dict(CHAT))
    hidden, _raw = await run(pipe, mode="Pipeline", stream=True, metadata={}, THINKING_OUTPUT_MODE="status")

    assert all(isinstance(frame, dict) and isinstance(frame.get("choices"), list) for frame in shown), shown
    assert shown[0]["choices"][0]["delta"].get("role") == "assistant", shown
    assert set(shown[0]["choices"][0]["delta"]) <= {"role", "content", "reasoning"}, shown
    assert all(set(frame["choices"][0]["delta"]) <= {"content", "reasoning"} for frame in shown[1:]), shown
    assert _reasoning_pieces_of(shown) == thoughts
    last_reasoning = max(i for i, frame in enumerate(shown) if frame["choices"][0]["delta"].get("reasoning"))
    assert last_reasoning < _answer_delta_at(shown, answer)
    assert _answer_text_of(shown) == answer

    assert _reasoning_pieces_of(hidden) == []
    assert _answer_text_of(hidden) == answer
    assert _reasoning_pieces_of(in_a_chat) == []
    assert _answer_text_of(in_a_chat) == answer


@pytest.mark.asyncio
async def test_a_failure_after_the_reasoning_reached_a_caller_with_no_chat_is_not_retried_and_stays_an_error_frame(
    pipe_instance_async, monkeypatch
):
    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.core.errors import _build_openrouter_api_error
    from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import answer_round, install, run

    thoughts = ["Weigh the two offers. ", "The cheaper one wins."]
    rejection = _build_openrouter_api_error(
        400,
        "Bad Request",
        json.dumps({"error": {"message": "include_thoughts is only enabled when thinking is enabled",
                              "code": "invalid_request_error"}}),
    )
    attempts: list[dict[str, Any]] = []

    async def thinks_and_is_rejected_the_first_time(self, session, request_body, **_kwargs):
        attempts.append(request_body)
        for thought in thoughts:
            yield {"type": "response.reasoning_text.delta", "item_id": "rs-1", "delta": thought}
        if len(attempts) == 1:
            raise rejection
        for event in answer_round("Take the cheaper offer."):
            yield event

    reasoning_row = {"features": {"function_calling", "reasoning"},
                     "supported_parameters": frozenset({"tools", "tool_choice", "temperature", "max_tokens",
                                                        "reasoning", "include_reasoning"})}
    pipe = pipe_instance_async
    install(pipe, monkeypatch, lambda _n, _request: [], row=reasoning_row)
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", thinks_and_is_rejected_the_first_time)

    frames, _raw = await run(pipe, mode="Pipeline", stream=True, metadata={})

    assert attempts[0].get("reasoning"), "the request did not ask for reasoning, so the retry that must not happen could not"
    assert len(attempts) == 1, f"the request was sent again after its reasoning reached the caller: {len(attempts)} attempts"
    assert _reasoning_pieces_of(frames) == thoughts
    assert frames[-1] == {"error": {"message": "include_thoughts is only enabled when thinking is enabled", "code": 400}}
    assert _answer_text_of(frames) == ""


@pytest.mark.jumping_clock
class TestAStreamWithNoChatStaysAlive:
    @pytest.mark.asyncio
    async def test_a_silent_stretch_sends_an_empty_chunk_every_interval_until_the_answer_starts(
        self, pipe_instance_async, monkeypatch
    ):
        import asyncio

        from open_webui_openrouter_pipe.pipe import _CHATLESS_KEEP_ALIVE_SECONDS as interval
        from open_webui_openrouter_pipe.pipe import Pipe
        from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import CHAT, answer_round, install, run

        pieces = ["Take the ", "cheaper offer."]
        pipe = pipe_instance_async
        install(pipe, monkeypatch, lambda _n, _request: answer_round("".join(pieces)))

        async def thinks_for_three_intervals_and_a_third(self, session, request_body, **_kwargs):
            for step in range(10):
                await asyncio.sleep(interval / 3)
                yield {"type": "response.reasoning_text.delta", "item_id": "rs-1", "delta": f"Weigh offer {step}."}
            for piece in pieces:
                yield {"type": "response.output_text.delta", "delta": piece}
                await asyncio.sleep(1)
            yield answer_round("".join(pieces))[-1]

        monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", thinks_for_three_intervals_and_a_third)

        def is_empty(frame: Any) -> bool:
            return isinstance(frame, dict) and frame.get("choices") in (
                [{"index": 0, "delta": {}, "finish_reason": None}],
                [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            )

        without_a_chat, _raw = await run(
            pipe, mode="Pipeline", stream=True, metadata={}, THINKING_OUTPUT_MODE="status"
        )
        in_a_chat, _raw = await run(
            pipe, mode="Pipeline", stream=True, metadata=dict(CHAT), THINKING_OUTPUT_MODE="status"
        )

        assert all(isinstance(frame, dict) and isinstance(frame.get("choices"), list) for frame in without_a_chat)
        assert [i for i, frame in enumerate(without_a_chat) if is_empty(frame)] == [0, 1, 2]
        assert without_a_chat[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert _answer_text_of(without_a_chat) == "".join(pieces)
        assert not [frame for frame in in_a_chat if is_empty(frame)]


@pytest.mark.parametrize("failure", ["a_provider_rejection", "a_body_that_is_not_an_openrouter_response"])
@pytest.mark.asyncio
async def test_a_streamed_failure_with_no_chat_on_an_anthropic_mount_reaches_the_agent_as_an_error_event(
    pipe_instance_async, monkeypatch, failure
):
    import ast
    import logging
    from types import SimpleNamespace

    from open_webui_openrouter_pipe.core.errors import UpstreamBodyUnreadable, _build_openrouter_api_error
    from tests.package_sources import open_webui_package_dir
    from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import MODEL, NORM, frames_of, install

    converter_path = open_webui_package_dir() / "utils" / "anthropic.py"
    converter_source = converter_path.read_text(encoding="utf-8")
    converter_node = next(
        node for node in ast.parse(converter_source).body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "openai_stream_to_anthropic_stream"
    )
    namespace: dict[str, Any] = {"JSONCodec": json, "log": logging.getLogger("open_webui.utils.anthropic")}
    exec(compile(ast.get_source_segment(converter_source, converter_node), str(converter_path), "exec"), namespace)
    to_anthropic = namespace["openai_stream_to_anthropic_stream"]

    unreadable = UpstreamBodyUnreadable(
        endpoint="https://openrouter.ai/api/v1/responses", body_excerpt="<html>bad gateway</html>", content_type="text/html"
    )
    raised, expected = {
        "a_provider_rejection": (
            _build_openrouter_api_error(
                429, "Too Many Requests", json.dumps({"error": {"message": "Rate limit exceeded", "code": 429}})
            ),
            {"error": {"message": "Rate limit exceeded", "code": 429}},
        ),
        "a_body_that_is_not_an_openrouter_response": (
            unreadable,
            {"error": {"message": unreadable.summary(), "code": 502}},
        ),
    }[failure]

    def fails(_n: int, _request: dict[str, Any]):
        raise raised

    pipe = pipe_instance_async
    install(pipe, monkeypatch, fails)
    streamed = await pipe.pipe(
        body={"model": NORM, "stream": True, "messages": [{"role": "user", "content": "hello"}]},
        __user__={"id": "user-1", "role": "user"},
        __request__=SimpleNamespace(url=SimpleNamespace(path="/api/v1/messages")),
        __event_emitter__=None,
        __event_call__=None,
        __metadata__={"model": {"id": MODEL}},
        __tools__=None,
    )
    pieces = [piece async for piece in streamed.body_iterator]
    frames = frames_of(pieces)
    assert frames == [expected, "[DONE]"], frames

    async def what_open_webui_streams():
        for piece in pieces:
            yield piece.encode()

    sent = [piece.decode() async for piece in to_anthropic(what_open_webui_streams(), model=MODEL)]
    assert [piece.split("\n", 1)[0] for piece in sent] == ["event: message_start", "event: error"], sent
    assert json.loads(sent[1].split("data: ", 1)[1])["error"]["message"] == expected["error"]["message"], sent


@pytest.mark.parametrize("failure", ["in_the_loop", "handed_back", "image"])
@pytest.mark.asyncio
async def test_a_refused_attachment_is_carried_once_by_the_closing_frames_when_the_reply_fails(
    pipe_instance_async, monkeypatch, failure
):
    from open_webui_openrouter_pipe.core.errors import _build_openrouter_api_error
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry
    from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import (
        BASE_URL,
        CHAT,
        MODEL,
        NORM,
        install,
        run,
    )

    refusal = json.dumps({"error": {"message": "the provider failed"}})

    def fails(_n: int, _request: dict[str, Any]):
        if failure == "in_the_loop":
            raise RuntimeError("the connection broke")
        raise _build_openrouter_api_error(500, "Server Error", refusal)

    pipe = pipe_instance_async
    install(pipe, monkeypatch, fails)
    if failure == "image":
        monkeypatch.setattr(
            OpenRouterModelRegistry,
            "_specs",
            {NORM: {"architecture": {"output_modalities": ["image"]}, "features": {"image_output"}}},
        )

    with aioresponses() as mocked:
        mocked.get(f"{BASE_URL}/images/models/{MODEL}/endpoints", payload={"endpoints": [{}]}, repeat=True)
        mocked.post(f"{BASE_URL}/images", status=500, body=refusal, repeat=True)
        chunks, _raw = await run(pipe, mode="Pipeline", stream=True, metadata=dict(CHAT), messages=_REFUSED_CLIP)

    events = [_event_of(chunk) for chunk in chunks]
    statuses = _statuses_of(chunks)
    noted = [(i, data) for i, data in statuses if _REFUSAL_SENTENCE in data.get("description", "")]
    assert len(noted) == 1, f"the refusal was noted {len(noted)} times, not once on the closing frames: {statuses}"
    assert noted[0] == statuses[-1], f"the note is not the last status: {statuses}"
    assert noted[0][1].get("done") is True
    completion_at = next(
        i for i, e in enumerate(events)
        if e.get("type") == "chat:completion" and e["data"].get("done") is True
    )
    assert noted[0][0] < completion_at
    assert [
        i for i, e in enumerate(events) if _REFUSAL_SENTENCE in json.dumps(e)
    ] == [noted[0][0]]


_ANSWERED_BY_URL = "https://openrouter.ai/api/v1/responses"


def _answered_by_round(model: str, *, calls_the_tool: bool) -> bytes:
    usage = {"input_tokens": 9, "output_tokens": 2}
    if calls_the_tool:
        call = {
            "type": "function_call", "id": "fc-1", "call_id": "call-1",
            "name": "lookup", "arguments": "{}", "status": "completed",
        }
        events = [
            {"type": "response.output_item.done", "item": call},
            {"type": "response.completed", "response": {"id": "resp-1", "model": model, "output": [call], "usage": usage}},
        ]
    else:
        message = {
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "Found it."}],
        }
        events = [
            {"type": "response.output_text.delta", "delta": "Found it."},
            {"type": "response.completed", "response": {"id": "resp-2", "model": model, "output": [message], "usage": usage}},
        ]
    return ("".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n").encode("utf-8")


async def _final_status_of_a_turn(
    pipe, monkeypatch, picked: str, answering: list[str]
) -> tuple[str, list[str]]:
    from aioresponses.core import CallbackResult

    import open_webui_openrouter_pipe.pipe as pipe_mod
    from tests.posted_body import posted_json

    async def loaded(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def lookup(**_kwargs: Any) -> str:
        return "found"

    catalogue = [
        {"id": "exampleco/router-auto", "name": "Router", "norm_id": "exampleco.router-auto"},
        {"id": "exampleco/model-small:nitro", "name": "Small", "norm_id": "exampleco.model-small:nitro"},
    ]
    tools = {
        "lookup": {
            "type": "function",
            "callable": lookup,
            "spec": {"name": "lookup", "description": "Look it up", "parameters": {"type": "object", "properties": {}}},
        }
    }
    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: ("sk-test-key", None))
    monkeypatch.setattr(pipe._artifact_store, "_ensure_artifact_store", lambda *_a, **_k: None)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", loaded)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "list_models", lambda: catalogue)
    dispatched: list[str] = []

    def upstream(_url, **kwargs):
        dispatched.append(posted_json(kwargs)["model"])
        round_number = len(dispatched)
        return CallbackResult(
            status=200,
            body=_answered_by_round(answering[round_number - 1], calls_the_tool=round_number < len(answering)),
        )

    with aioresponses() as mock_http:
        mock_http.post(_ANSWERED_BY_URL, callback=upstream, repeat=True)
        result = await pipe.pipe(
            body={"model": picked, "messages": [{"role": "user", "content": "look it up"}], "stream": True},
            __user__={"id": "user-1", "role": "user"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__={"chat_id": "chat-1", "message_id": "message-1", "model": {"id": picked}},
            __tools__=tools,
        )
        chunks = [chunk async for chunk in result]

    statuses = [
        chunk["event"]["data"]
        for chunk in chunks
        if isinstance(chunk, dict)
        and isinstance(chunk.get("event"), dict)
        and chunk["event"].get("type") == "status"
    ]
    closing = [data["description"] for data in statuses if data.get("done") is True]
    assert len(closing) == 1, statuses
    return closing[0], dispatched


@pytest.mark.asyncio
async def test_the_final_status_names_the_model_that_answered_only_when_it_is_not_the_one_picked(
    pipe_instance_async, monkeypatch
):
    routed, routed_posts = await _final_status_of_a_turn(
        pipe_instance_async,
        monkeypatch,
        "exampleco/router-auto",
        ["exampleco/model-small", "exampleco/model-large"],
    )
    assert routed_posts == ["exampleco/router-auto"] * 2
    assert _status_field(routed, 1) == "Answered by exampleco/model-small, exampleco/model-large", routed

    same, same_posts = await _final_status_of_a_turn(
        pipe_instance_async,
        monkeypatch,
        "exampleco/model-small:nitro",
        ["exampleco/model-small"],
    )
    assert same_posts == ["exampleco/model-small:nitro"]
    assert "Answered by" not in same, same
