"""Tests for the error template system."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx


@pytest.fixture
def mock_pipe():
    """Create a mock Pipe instance with valves configured."""
    from openrouter_responses_pipe.openrouter_responses_pipe import Pipe

    pipe = Pipe()
    pipe.valves.SUPPORT_EMAIL = "support@example.com"
    pipe.valves.SUPPORT_URL = "https://support.example.com"
    pipe.logger = MagicMock()

    return pipe


@pytest.fixture
def mock_event_emitter():
    """Create a mock event emitter that captures emitted events."""
    emitted_events = []

    async def emitter(event):
        emitted_events.append(event)

    emitter.events = emitted_events
    return emitter


class TestEmitTemplatedError:
    """Test the _emit_templated_error helper method."""

    @pytest.mark.asyncio
    async def test_basic_template_rendering(self, mock_pipe, mock_event_emitter):
        """Test that basic template rendering works."""
        await mock_pipe._emit_templated_error(
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
        await mock_pipe._emit_templated_error(
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
        await mock_pipe._emit_templated_error(
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
        await mock_pipe._emit_templated_error(
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
        await mock_pipe._emit_templated_error(
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
        # Mock registry and process to raise TimeoutException
        with patch('openrouter_responses_pipe.openrouter_responses_pipe.OpenRouterModelRegistry') as mock_registry:
            mock_registry.ensure_loaded = AsyncMock()
            mock_registry.list_models.return_value = [{"norm_id": "test-model"}]

            with patch.object(mock_pipe, '_process_transformed_request', side_effect=httpx.TimeoutException("timeout")):
                mock_pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
                mock_pipe.valves.API_KEY = "test-key"
                mock_pipe.valves.MODEL_CATALOG_REFRESH_SECONDS = 300
                mock_session = MagicMock()

                result = await mock_pipe._handle_pipe_call(
                    body={"model": "test-model"},
                    __user__={"id": "test-user"},
                    __request__=MagicMock(),
                    __event_emitter__=mock_event_emitter,
                    __event_call__=None,
                    __metadata__={"model": {"id": "test-model"}},
                    __tools__=None,
                    valves=mock_pipe.valves,
                    session=mock_session,
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
        with patch('openrouter_responses_pipe.openrouter_responses_pipe.OpenRouterModelRegistry') as mock_registry:
            mock_registry.ensure_loaded = AsyncMock()
            mock_registry.list_models.return_value = [{"norm_id": "test-model"}]

            with patch.object(mock_pipe, '_process_transformed_request', side_effect=httpx.ConnectError("connection failed")):
                mock_pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
                mock_pipe.valves.API_KEY = "test-key"
                mock_pipe.valves.MODEL_CATALOG_REFRESH_SECONDS = 300
                mock_session = MagicMock()

                result = await mock_pipe._handle_pipe_call(
                    body={"model": "test-model"},
                    __user__={"id": "test-user"},
                    __request__=MagicMock(),
                    __event_emitter__=mock_event_emitter,
                    __event_call__=None,
                    __metadata__={"model": {"id": "test-model"}},
                    __tools__=None,
                    valves=mock_pipe.valves,
                    session=mock_session,
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

        with patch('openrouter_responses_pipe.openrouter_responses_pipe.OpenRouterModelRegistry') as mock_registry:
            mock_registry.ensure_loaded = AsyncMock()
            mock_registry.list_models.return_value = [{"norm_id": "test-model"}]

            with patch.object(mock_pipe, '_process_transformed_request', side_effect=error):
                mock_pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
                mock_pipe.valves.API_KEY = "test-key"
                mock_pipe.valves.MODEL_CATALOG_REFRESH_SECONDS = 300
                mock_session = MagicMock()

                result = await mock_pipe._handle_pipe_call(
                    body={"model": "test-model"},
                    __user__={"id": "test-user"},
                    __request__=MagicMock(),
                    __event_emitter__=mock_event_emitter,
                    __event_call__=None,
                    __metadata__={"model": {"id": "test-model"}},
                    __tools__=None,
                    valves=mock_pipe.valves,
                    session=mock_session,
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
        with patch('openrouter_responses_pipe.openrouter_responses_pipe.OpenRouterModelRegistry') as mock_registry:
            mock_registry.ensure_loaded = AsyncMock()
            mock_registry.list_models.return_value = [{"norm_id": "test-model"}]

            with patch.object(mock_pipe, '_process_transformed_request', side_effect=ValueError("unexpected error")):
                mock_pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
                mock_pipe.valves.API_KEY = "test-key"
                mock_pipe.valves.MODEL_CATALOG_REFRESH_SECONDS = 300
                mock_session = MagicMock()

                result = await mock_pipe._handle_pipe_call(
                    body={"model": "test-model"},
                    __user__={"id": "test-user"},
                    __request__=MagicMock(),
                    __event_emitter__=mock_event_emitter,
                    __event_call__=None,
                    __metadata__={"model": {"id": "test-model"}},
                    __tools__=None,
                    valves=mock_pipe.valves,
                    session=mock_session,
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

        with patch('openrouter_responses_pipe.openrouter_responses_pipe.OpenRouterModelRegistry') as mock_registry:
            mock_registry.ensure_loaded = AsyncMock()
            mock_registry.list_models.return_value = [{"norm_id": "test-model"}]

            with patch.object(mock_pipe, '_process_transformed_request', side_effect=RuntimeError("test")):
                mock_pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
                mock_pipe.valves.API_KEY = "test-key"
                mock_pipe.valves.MODEL_CATALOG_REFRESH_SECONDS = 300
                mock_session = MagicMock()

                result = await mock_pipe._handle_pipe_call(
                    body={"model": "test-model"},
                    __user__={"id": "test-user"},
                    __request__=MagicMock(),
                    __event_emitter__=mock_event_emitter,
                    __event_call__=None,
                    __metadata__={"model": {"id": "test-model"}},
                    __tools__=None,
                    valves=mock_pipe.valves,
                    session=mock_session,
                )

        content = mock_event_emitter.events[0]["data"]["content"]
        assert "Custom error" in content
        assert "RuntimeError" in content
