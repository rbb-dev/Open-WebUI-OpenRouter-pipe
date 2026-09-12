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

        assert len(mock_event_emitter.events) == 2
        content = mock_event_emitter.events[0]["data"]["content"]
        assert result == content
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

        assert len(mock_event_emitter.events) == 2
        content = mock_event_emitter.events[0]["data"]["content"]
        assert result == content
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

        assert len(mock_event_emitter.events) == 2
        content = mock_event_emitter.events[0]["data"]["content"]
        assert result == content
        assert "Service Error" in content or "502" in content
        assert "Error ID:" in content


class TestRateLimitRetryAfter:
    """4xx Retry-After header parsing in the pipe-level error handler."""

    @pytest.mark.asyncio
    async def test_4xx_http_date_retry_after_parsed_to_seconds(self, mock_pipe, mock_event_emitter):
        """A 429 carrying an HTTP-date Retry-After must store NUMERIC
        retry_after_seconds in the error metadata, not the raw date string."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.reason_phrase = "Too Many Requests"
        mock_response.headers = {"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"}
        mock_response.aread = AsyncMock(return_value=b'{"error": {"message": "rate limited"}}')
        error = httpx.HTTPStatusError("429", request=MagicMock(), response=mock_response)

        report_mock = AsyncMock()
        formatter = mock_pipe._ensure_error_formatter()

        with aioresponses() as mock_http:
            mock_http.get(
                "https://openrouter.ai/api/v1/models",
                payload={"data": [{"id": "test-model", "name": "Test Model", "pricing": {"prompt": "0", "completion": "0"}}]},
            )
            with patch.object(mock_pipe, "_process_transformed_request", side_effect=error), \
                 patch.object(formatter, "_report_openrouter_error", report_mock):
                mock_pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
                mock_pipe.valves.API_KEY = EncryptedStr("test-key")
                mock_pipe.valves.MODEL_CATALOG_REFRESH_SECONDS = 300
                async with aiohttp.ClientSession() as session:
                    await mock_pipe._handle_pipe_call(
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

        assert report_mock.await_count == 1, "error was not reported"
        err = report_mock.call_args.args[0]
        ras = err.metadata.get("retry_after_seconds")
        assert isinstance(ras, (int, float)) and not isinstance(ras, bool), f"expected numeric, got {ras!r}"
        assert ras > 0
        assert err.metadata.get("retry_after") == "Wed, 21 Oct 2099 07:28:00 GMT"
        assert ras != "Wed, 21 Oct 2099 07:28:00 GMT"


class TestNonServerHttpErrorDetails:
    """Everything the non-5xx branch derives besides Retry-After.

    ``TestRateLimitRetryAfter`` reaches this same branch and asserts the two Retry-After
    keys, which leaves the rest of it unmeasured: reading the response body at all,
    DECODING those bytes as UTF-8, and carrying the rate-limit scope onto the metadata the
    card renders from. Each is its own line, and each one, removed, still leaves the error
    reported exactly once -- so counting the report proves none of them.
    """

    @staticmethod
    def _error(body_bytes, scope):
        from unittest.mock import AsyncMock

        headers = {"Content-Type": "application/json"}
        if scope is not None:
            headers["X-RateLimit-Scope"] = scope
        response = MagicMock()
        response.status_code = 429
        response.reason_phrase = "Too Many Requests"
        # The real type, which is case-insensitive: a plain dict would make the lookup
        # pass or fail on the spelling the double happened to use rather than on the code.
        response.headers = httpx.Headers(headers)
        response.aread = AsyncMock(return_value=body_bytes)
        return httpx.HTTPStatusError("429", request=MagicMock(), response=response)

    async def _reported(self, pipe, emitter, error):
        """Drive `_handle_pipe_call` and return the error it handed the reporter.

        Both patches sit one level BELOW the branch under test -- the call that raises, and
        the reporter it ends at -- so the except arm, the body read, the decode and the
        header lookup all run for real.
        """
        from unittest.mock import AsyncMock

        report_mock = AsyncMock()
        formatter = pipe._ensure_error_formatter()
        with aioresponses() as mock_http:
            mock_http.get(
                "https://openrouter.ai/api/v1/models",
                payload={"data": [{"id": "test-model", "name": "Test Model", "pricing": {"prompt": "0", "completion": "0"}}]},
            )
            with patch.object(pipe, "_process_transformed_request", side_effect=error), \
                 patch.object(formatter, "_report_openrouter_error", report_mock):
                pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
                pipe.valves.API_KEY = EncryptedStr("test-key")
                pipe.valves.MODEL_CATALOG_REFRESH_SECONDS = 300
                async with aiohttp.ClientSession() as session:
                    await pipe._handle_pipe_call(
                        body={"model": "test-model"},
                        __user__={"id": "test-user"},
                        __request__=MagicMock(),
                        __event_emitter__=emitter,
                        __event_call__=None,
                        __metadata__={"model": {"id": "test-model"}},
                        __tools__=None,
                        valves=pipe.valves,
                        session=session,
                    )
        assert report_mock.await_count == 1, (
            "precondition: the error never reached the reporter, so nothing below is about "
            "what it carried"
        )
        return report_mock.call_args.args[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("message", "code"),
        [("slow down", 429), ("demasiadas peticiones \u2014 espera", 402)],
        ids=["ascii-body", "non-ascii-body"],
    )
    async def test_the_reported_error_carries_what_the_decoded_body_says(
        self, mock_pipe, mock_event_emitter, message, code
    ):
        """The body has to be READ, and read as UTF-8, or the card says nothing useful.

        Two lines are in play. Skipping the read leaves the whole payload unparsed, and the
        card loses the provider's own sentence. Formatting the bytes instead of decoding
        them yields a `b'...'` repr, which is not JSON -- so the payload is unparsed again,
        and the raw body kept for the report is a Python literal rather than what the server
        sent. ``raw_body`` separates the two, because the repr is visible in it while both
        produce the same missing message.

        The expectation is what OpenRouter's own parser makes of the correctly decoded
        bytes, so it cannot drift from the shape the reporter is handed. One ASCII body and
        one carrying characters outside it, with different messages and different codes, so
        no constant satisfies both -- and the non-ASCII row is the one a latin-1 or a
        `str(bytes)` reading mangles rather than merely fails to parse.
        """
        import json

        from open_webui_openrouter_pipe.core.errors import _extract_openrouter_error_details

        body_bytes = json.dumps(
            {"error": {"message": message, "code": code}}, ensure_ascii=False
        ).encode("utf-8")
        expected = _extract_openrouter_error_details(body_bytes.decode("utf-8"))
        assert expected["openrouter_message"] == message and expected["openrouter_code"] == code, (
            "precondition: the body has to parse from the correct decode, or every "
            "assertion below compares one absence to another"
        )

        err = await self._reported(
            mock_pipe, mock_event_emitter, self._error(body_bytes, "account")
        )

        assert err.raw_body == body_bytes.decode("utf-8"), (
            f"the body kept for the report is not what the server sent: {err.raw_body!r}"
        )
        assert err.openrouter_message == expected["openrouter_message"], (
            "the provider's own sentence never reached the card, so the user is told a "
            f"request failed without being told why: {err.openrouter_message!r}"
        )
        assert err.openrouter_code == expected["openrouter_code"], (
            f"the code the body carried was lost: {err.openrouter_code!r}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scope", ["account", "user", None], ids=["account", "user", "header-absent"]
    )
    async def test_the_rate_limit_scope_reaches_the_metadata_only_when_it_was_sent(
        self, mock_pipe, mock_event_emitter, scope
    ):
        """`rate_limit_type` is what the card's `{{#if rate_limit_type}}` block renders from.

        Dropped, the block vanishes and a user throttled on the account's shared quota is
        given the same card as one throttled on their own -- and goes on retrying against a
        limit that is not theirs to clear. Invented, the card names a scope the response
        never carried.

        Two different scopes, so the value cannot be a constant, and the absent header as
        the third row, so writing one unconditionally cannot pass either. Absence is
        asserted as the key not being present rather than as a falsy value, because the
        template branches on presence.
        """
        body_bytes = b'{"error": {"message": "slow down", "code": 429}}'

        err = await self._reported(
            mock_pipe, mock_event_emitter, self._error(body_bytes, scope)
        )

        if scope is None:
            assert "rate_limit_type" not in err.metadata, (
                "a scope the response never sent was written onto the error, and the card "
                f"now names it: {err.metadata!r}"
            )
        else:
            assert err.metadata.get("rate_limit_type") == scope, (
                "the response said which quota was hit and the card cannot say so: "
                f"{err.metadata!r}"
            )


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

        assert len(mock_event_emitter.events) == 2
        content = mock_event_emitter.events[0]["data"]["content"]
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
    pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
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
            __metadata__={},
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=pipe.valves,
            session=AsyncMock(spec=aiohttp.ClientSession),
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
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
    [("45", "**Retry after:** 45s"), ("soon", None), ("", None)],
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
        ({"retry_after_seconds": float("inf")}, None),
        ({"retry_after_seconds": float("-inf")}, None),
        ({"retry_after_seconds": float("nan")}, None),
        ({"retry_after": "1e400"}, None),
        ({"retry_after": "inf"}, None),
        ({"retry_after": "nan"}, None),
        ({"retry_after_seconds": 0}, 0),
        ({"retry_after_seconds": 45}, 45),
        ({"retry_after": "30"}, 30),
        ({"retry_after_seconds": -5}, 0),
        ({"retry_after_seconds": -7.4}, 0),
        ({"retry_after": -12}, 0),
        ({"retry_after_seconds": "-5"}, 0),
        ({"retry_after": "-12"}, 0),
        ({"retry_after_seconds": float("inf"), "retry_after": "30"}, 30),
        ({"retry_after_seconds": 0, "retry_after": "Wed, 21 Oct 2015 07:28:00 GMT"}, 0),
        ({"retry_after_seconds": True}, None),
        ({"retry_after_seconds": False}, None),
        ({"retry_after": True}, None),
        ({"retry_after_seconds": True, "retry_after": "30"}, 30),
        ({"retry_after_seconds": False, "retry_after": "45"}, 45),
        ({}, None),
        ("not a mapping", None),
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
    [("1e400", None), ("inf", None), ("nan", None), ("45", 45), ("soon", None)],
)
def test_a_non_finite_retry_after_header_is_never_recorded_as_seconds(header, expected):
    """The header route must not store a delay it could not turn into a whole number either."""
    from open_webui_openrouter_pipe.core.utils import _apply_retry_after_metadata

    meta: dict[str, Any] = {}
    _apply_retry_after_metadata(meta, {"Retry-After": header})
    assert meta.get("retry_after_seconds") == expected
    assert meta["retry_after"] == header, "the raw header is kept for diagnostics regardless"


@pytest.mark.asyncio
@pytest.mark.parametrize("literal", ["Infinity", "-Infinity", "NaN"])
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
@pytest.mark.parametrize("magnitude", [7, 45])
async def test_a_retry_after_already_in_the_past_never_tells_the_reader_to_wait_backwards(
    pipe_instance_async, magnitude
):
    """A negative delay reached the reader as "**Retry after:** -7s", an instruction to wait
    backwards.

    The two spellings of the same instruction took different routes -- the number was rounded
    with no lower bound while the string went through the header parser, which clamps -- so
    the same value resolved to two different answers. Each row drives BOTH spellings through
    the real error builder from raw provider body text and requires them to agree, and pairs
    them with the SAME magnitude as a positive delay, which must still be shown: a fix that
    simply stopped rendering the line would pass the first two assertions and fail that one.
    Two magnitudes, so nothing hardcoded satisfies both rows.
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
        [float("inf"), float("-inf"), float("nan")],
        ids=["infinity", "negative-infinity", "nan"],
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
        'HTTP 400: {"error":{"message":"   "}}',
        "HTTP 400: {not json at all}",
        'HTTP 400: {"error":{"code":"bad_request"}}',
        None,
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
