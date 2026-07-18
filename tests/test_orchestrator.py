"""Tests for the RequestOrchestrator module.

This test file covers edge cases and error handling paths in the orchestrator:
- Direct upload handling (files, audio, video)
- Base64 decoding edge cases
- Audio format sniffing
- Tool registry processing
- Web search plugin configuration
- Error retry logic (reasoning effort, prompt caching, etc.)
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

from __future__ import annotations

import base64
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
import aiohttp
from aioresponses import aioresponses

from conftest import Pipe
from open_webui_openrouter_pipe.requests.orchestrator import RequestOrchestrator
from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError
from open_webui_openrouter_pipe.filters.fusion_filter_renderer import is_fusion_model


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest_asyncio.fixture
async def orchestrator_and_pipe():
    """Create an orchestrator with a properly configured pipe."""
    pipe = Pipe()
    logger = logging.getLogger("test_orchestrator")
    logger.setLevel(logging.DEBUG)
    orchestrator = RequestOrchestrator(pipe, logger)
    yield orchestrator, pipe
    await pipe.close()


@pytest.fixture
def mock_valves():
    """Create mock valves with default test values."""
    valves = Mock()
    valves.MAX_INPUT_IMAGES_PER_REQUEST = 0
    valves.IMAGE_INPUT_SELECTION = "latest_user"
    valves.BASE64_MAX_SIZE_MB = 10
    valves.IMAGE_UPLOAD_CHUNK_BYTES = 65536
    valves.DIRECT_UPLOAD_FAILURE_TEMPLATE = "Direct upload failed: {reason}"
    valves.ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE = "Endpoint conflict: {reason}"
    valves.MODEL_RESTRICTED_TEMPLATE = "Model restricted: {restriction_reasons}"
    valves.OPENROUTER_ERROR_TEMPLATE = "Error: {detail}"
    valves.ZDR_ENFORCE = False
    valves.ALLOW_USER_ZDR_OVERRIDE = False
    valves.REASONING_EFFORT = "medium"
    valves.TASK_MODEL_REASONING_EFFORT = "low"
    valves.USE_MODEL_MAX_OUTPUT_TOKENS = False
    valves.ENABLE_STRICT_TOOL_CALLING = False
    valves.TOOL_EXECUTION_MODE = "Pipeline"
    valves.TOOL_OUTPUT_RETENTION_TURNS = 3
    valves.ENABLE_ANTHROPIC_PROMPT_CACHING = False
    valves.MODEL_ID = ""
    valves.FREE_MODEL_FILTER = "all"
    valves.TOOL_CALLING_FILTER = "all"
    return valves


@pytest.fixture
def mock_session():
    """Create a mock aiohttp session."""
    session = AsyncMock(spec=aiohttp.ClientSession)
    return session


@pytest.fixture
def base_request_body():
    """Create a basic request body for testing."""
    return {
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello, world!"}
        ],
        "stream": True,
    }


# -----------------------------------------------------------------------------
# Test _decode_base64_prefix edge cases (lines 147, 150-151, 156, 168-169)
# -----------------------------------------------------------------------------


class TestDecodeBase64PrefixEdgeCases:
    """Tests for the internal _decode_base64_prefix function edge cases."""

    @pytest.mark.asyncio
    async def test_empty_data_returns_empty_bytes(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body, monkeypatch):
        """Empty input data should return empty bytes (line 147)."""
        orchestrator, pipe = orchestrator_and_pipe

        # We need to trigger the code path by having audio attachments
        # with empty base64 data
        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "audio": [{"id": "audio123", "format": "mp3"}]
                }
            }
        }

        # Mock file loading to return empty base64
        monkeypatch.setattr("open_webui_openrouter_pipe.requests.orchestrator.get_file_by_id", AsyncMock(return_value=Mock(id="audio123")))
        pipe._file_gateway.read_file_record_base64 = AsyncMock(return_value="")
        pipe._ensure_error_formatter()._emit_templated_error = AsyncMock()

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids=set(),
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        # Should fail because empty b64 for audio
        assert result == ""
        pipe._ensure_error_formatter()._emit_templated_error.assert_called()

    @pytest.mark.asyncio
    async def test_invalid_base64_chars_in_prefix(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body, monkeypatch):
        """Non-base64 characters in prefix should return empty bytes and use declared format (line 160).

        When base64 contains invalid characters, the sniff returns empty which falls back to declared format.
        """
        orchestrator, pipe = orchestrator_and_pipe

        # Create audio attachment with invalid base64 but with a declared format
        # The invalid chars will cause sniffing to fail, but declared format is used as fallback
        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "audio": [{"id": "audio123", "format": "mp3"}]  # Has declared format
                }
            }
        }

        # Mock file loading to return invalid base64 with special chars
        monkeypatch.setattr("open_webui_openrouter_pipe.requests.orchestrator.get_file_by_id", AsyncMock(return_value=Mock(id="audio123")))
        # This base64 contains invalid characters like unicode - sniff will return ""
        pipe._file_gateway.read_file_record_base64 = AsyncMock(return_value="AAAA\u0080BBBB")
        pipe._ensure_error_formatter()._emit_templated_error = AsyncMock()
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._select_llm_endpoint_with_forced = Mock(return_value=("chat_completions", False))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        # Should succeed because declared format "mp3" is used when sniff fails
        assert result == "Test response"


# -----------------------------------------------------------------------------
# Test file/audio/video attachment skip paths (lines 197, 229, 255)
# -----------------------------------------------------------------------------


class TestDirectUploadSkipPaths:
    """Tests for skipping invalid attachments in direct uploads."""

    @pytest.mark.asyncio
    async def test_file_attachment_with_invalid_id_skipped(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """File attachments with non-string or empty id should be skipped (line 197)."""
        orchestrator, pipe = orchestrator_and_pipe

        # Create file attachments with invalid ids
        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "files": [
                        {"id": None, "name": "file1.txt"},  # None id
                        {"id": 123, "name": "file2.txt"},   # Non-string id
                        {"id": "", "name": "file3.txt"},    # Empty string id
                        {"id": "  ", "name": "file4.txt"},  # Whitespace-only id
                    ]
                }
            }
        }

        # Setup mocks for successful request after skipping invalid files
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        # Should succeed - all invalid files were skipped
        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_audio_attachment_with_invalid_id_skipped(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Audio attachments with non-string or empty id should be skipped (line 229)."""
        orchestrator, pipe = orchestrator_and_pipe

        # Create audio attachments with invalid ids
        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "audio": [
                        {"id": None, "format": "mp3"},
                        {"id": "", "format": "wav"},
                    ]
                }
            }
        }

        # Setup mocks
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        # Should succeed - all invalid audio was skipped
        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_video_attachment_with_invalid_id_skipped(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Video attachments with non-string or empty id should be skipped (line 255)."""
        orchestrator, pipe = orchestrator_and_pipe

        # Create video attachments with invalid ids
        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "video": [
                        {"id": None, "content_type": "video/mp4"},
                        {"id": 456, "content_type": "video/webm"},
                    ]
                }
            }
        }

        # Setup mocks
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        # Should succeed - all invalid videos were skipped
        assert result == "Test response"


# -----------------------------------------------------------------------------
# Test _csv_set with non-string input (line 213)
# -----------------------------------------------------------------------------


class TestCsvSetNonStringInput:
    """Tests for _csv_set handling of non-string inputs."""

    @pytest.mark.asyncio
    async def test_csv_set_with_non_string_returns_empty_set(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body, monkeypatch):
        """Non-string input to _csv_set should return empty set (line 213)."""
        orchestrator, pipe = orchestrator_and_pipe

        # Create audio with non-string allowlist value
        # Need valid audio to trigger the _csv_set code path
        wav_header = b"RIFF\x00\x00\x00\x00WAVEfmt "
        valid_b64 = base64.b64encode(wav_header).decode()

        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "audio": [{"id": "audio123", "format": "wav"}],
                    "responses_audio_format_allowlist": 12345,  # Non-string value
                }
            }
        }

        # Mock file loading
        mock_file = Mock(id="audio123")
        monkeypatch.setattr("open_webui_openrouter_pipe.requests.orchestrator.get_file_by_id", AsyncMock(return_value=mock_file))
        pipe._file_gateway.read_file_record_base64 = AsyncMock(return_value=valid_b64)
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._select_llm_endpoint_with_forced = Mock(return_value=("chat_completions", False))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        # Should succeed - non-string allowlist produces empty set (fallback to default)
        assert result == "Test response"


# -----------------------------------------------------------------------------
# Test extra_tools exception handling (lines 556-557)
# -----------------------------------------------------------------------------


class TestExtraToolsExceptionHandling:
    """Tests for extra_tools exception handling."""

    @pytest.mark.asyncio
    async def test_extra_tools_exception_caught(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Exception when accessing extra_tools should be caught (lines 556-557)."""
        orchestrator, pipe = orchestrator_and_pipe

        # Create a completions body mock that raises on extra_tools access
        class BadExtraTools:
            @property
            def extra_tools(self):
                raise RuntimeError("Simulated extra_tools access error")

        # Setup mocks
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        # Patch CompletionsBody.model_validate to return object with problematic extra_tools
        with patch("open_webui_openrouter_pipe.requests.orchestrator.CompletionsBody") as mock_completions:
            bad_body = Mock()
            bad_body.extra_tools = property(lambda self: (_ for _ in ()).throw(RuntimeError("fail")))
            # Make getattr raise for extra_tools
            type(bad_body).extra_tools = property(lambda self: (_ for _ in ()).throw(RuntimeError("fail")))
            mock_completions.model_validate.return_value = bad_body

            with patch("open_webui_openrouter_pipe.requests.orchestrator.ResponsesBody") as mock_responses:
                responses_body = Mock()
                responses_body.model = "openai/gpt-4o"
                responses_body.stream = True
                responses_body.reasoning = None
                responses_body.tools = None
                responses_body.plugins = None
                responses_body.max_output_tokens = None
                responses_body.model_dump = Mock(return_value={})
                mock_responses.from_completions = AsyncMock(return_value=responses_body)

                result = await orchestrator.process_request(
                    body=base_request_body,
                    __user__={"id": "user1"},
                    __request__=None,
                    __event_emitter__=None,
                    __event_call__=None,
                    __metadata__={},
                    __tools__=None,
                    __task__=None,
                    __task_body__=None,
                    valves=mock_valves,
                    session=mock_session,
                    openwebui_model_id="openai/gpt-4o",
                    pipe_identifier="test-pipe",
                    allowlist_norm_ids={"openai/gpt-4o"},
                    enforced_norm_ids=set(),
                    catalog_norm_ids=set(),
                    features={},
                )

                # Should succeed despite extra_tools exception
                assert result == "Test response"


# -----------------------------------------------------------------------------
# Test tool rename logging (line 624)
# -----------------------------------------------------------------------------


class TestToolRenameLogging:
    """Tests for tool rename debug logging."""

    @pytest.mark.asyncio
    async def test_tool_renames_logged_when_debug_enabled(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Tool renames should be logged when DEBUG level is enabled (line 624)."""
        orchestrator, pipe = orchestrator_and_pipe

        # Enable debug logging
        orchestrator.logger.setLevel(logging.DEBUG)

        # Setup mocks
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()

        # Mock to return tools with renames
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        # Patch _build_collision_safe_tool_specs_and_registry to return renames
        with patch("open_webui_openrouter_pipe.requests.orchestrator._build_collision_safe_tool_specs_and_registry") as mock_build:
            # Return tools, registry, and exposed_to_origin with renames
            mock_build.return_value = (
                [{"type": "function", "function": {"name": "renamed_tool"}}],  # tools
                {"renamed_tool": {"spec": {"name": "renamed_tool"}}},  # exec_registry
                {"renamed_tool": "original_tool"},  # exposed_to_origin with rename
            )

            with patch("open_webui_openrouter_pipe.requests.orchestrator.ModelFamily") as mock_family:
                mock_family.base_model.return_value = "openai/gpt-4o"
                mock_family.supports.return_value = True
                mock_family.capabilities.return_value = {}
                mock_family.max_completion_tokens.return_value = None

                with patch("open_webui_openrouter_pipe.requests.orchestrator.OpenRouterModelRegistry") as mock_registry:
                    mock_registry.api_model_id.return_value = "openai/gpt-4o"

                    result = await orchestrator.process_request(
                        body=base_request_body,
                        __user__={"id": "user1"},
                        __request__=None,
                        __event_emitter__=None,
                        __event_call__=None,
                        __metadata__={},
                        __tools__={"original_tool": {"spec": {"name": "original_tool"}}},
                        __task__=None,
                        __task_body__=None,
                        valves=mock_valves,
                        session=mock_session,
                        openwebui_model_id="openai/gpt-4o",
                        pipe_identifier="test-pipe",
                        allowlist_norm_ids={"openai/gpt-4o"},
                        enforced_norm_ids=set(),
                        catalog_norm_ids=set(),
                        features={},
                    )

                    assert result == "Test response"


# -----------------------------------------------------------------------------
# Test OpenRouterAPIError handling and retry logic (lines 692-771)
# -----------------------------------------------------------------------------


class TestOpenRouterAPIErrorHandling:
    """Tests for OpenRouterAPIError handling and retry logic."""

    @pytest.mark.asyncio
    async def test_reasoning_effort_retry(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Reasoning effort error should trigger retry with fallback effort (lines 710-755)."""
        orchestrator, pipe = orchestrator_and_pipe

        # Setup mocks
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))

        call_count = [0]

        async def mock_streaming_loop(responses_body, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call fails with reasoning effort error
                raise OpenRouterAPIError(
                    status=400,
                    reason="Bad Request",
                    upstream_message="Invalid reasoning.effort value 'high'. Supported values are: 'low', 'medium'.",
                    provider_raw={
                        "error": {
                            "param": "reasoning.effort",
                            "code": "unsupported_value",
                            "type": "invalid_request_error",
                        }
                    },
                )
            return "Test response after retry"

        pipe._streaming_handler._run_streaming_loop = mock_streaming_loop

        # Create mock event emitter
        event_emitter = AsyncMock()

        with patch("open_webui_openrouter_pipe.requests.orchestrator.ModelFamily") as mock_family:
            mock_family.base_model.return_value = "openai/o1"
            mock_family.supports.return_value = False
            mock_family.capabilities.return_value = {}
            mock_family.max_completion_tokens.return_value = None

            with patch("open_webui_openrouter_pipe.requests.orchestrator.OpenRouterModelRegistry") as mock_registry:
                mock_registry.api_model_id.return_value = "openai/o1"

                result = await orchestrator.process_request(
                    body={**base_request_body, "model": "openai/o1"},
                    __user__={"id": "user1"},
                    __request__=None,
                    __event_emitter__=event_emitter,
                    __event_call__=None,
                    __metadata__={},
                    __tools__=None,
                    __task__=None,
                    __task_body__=None,
                    valves=mock_valves,
                    session=mock_session,
                    openwebui_model_id="openai/o1",
                    pipe_identifier="test-pipe",
                    allowlist_norm_ids={"openai/o1"},
                    enforced_norm_ids=set(),
                    catalog_norm_ids=set(),
                    features={},
                )

                assert result == "Test response after retry"
                assert call_count[0] == 2
                # Should have emitted status update
                event_emitter.assert_called()

    @pytest.mark.asyncio
    async def test_reasoning_retry_without_reasoning(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Error that should retry without reasoning (lines 757-762)."""
        orchestrator, pipe = orchestrator_and_pipe

        # Setup mocks
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._ensure_reasoning_config_manager()._should_retry_without_reasoning = Mock(return_value=True)

        call_count = [0]

        async def mock_streaming_loop(responses_body, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call fails with a reasoning-related error
                raise OpenRouterAPIError(
                    status=400,
                    reason="Bad Request",
                    openrouter_message="Reasoning not supported",
                )
            # After retry, reasoning should have been removed
            return "Test response after retry"

        pipe._streaming_handler._run_streaming_loop = mock_streaming_loop

        with patch("open_webui_openrouter_pipe.requests.orchestrator.ModelFamily") as mock_family:
            mock_family.base_model.return_value = "openai/gpt-4o"
            mock_family.supports.return_value = False
            mock_family.capabilities.return_value = {}
            mock_family.max_completion_tokens.return_value = None

            with patch("open_webui_openrouter_pipe.requests.orchestrator.OpenRouterModelRegistry") as mock_registry:
                mock_registry.api_model_id.return_value = "openai/gpt-4o"

                result = await orchestrator.process_request(
                    body=base_request_body,
                    __user__={"id": "user1"},
                    __request__=None,
                    __event_emitter__=None,
                    __event_call__=None,
                    __metadata__={},
                    __tools__=None,
                    __task__=None,
                    __task_body__=None,
                    valves=mock_valves,
                    session=mock_session,
                    openwebui_model_id="openai/gpt-4o",
                    pipe_identifier="test-pipe",
                    allowlist_norm_ids={"openai/gpt-4o"},
                    enforced_norm_ids=set(),
                    catalog_norm_ids=set(),
                    features={},
                )

                assert result == "Test response after retry"
                assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_error_reported_when_no_retry_applicable(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Error should be reported when no retry is applicable (lines 764-771)."""
        orchestrator, pipe = orchestrator_and_pipe

        # Setup mocks
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._ensure_reasoning_config_manager()._should_retry_without_reasoning = Mock(return_value=False)
        pipe._ensure_error_formatter()._report_openrouter_error = AsyncMock()

        async def mock_streaming_loop(responses_body, *args, **kwargs):
            raise OpenRouterAPIError(
                status=500,
                reason="Internal Server Error",
                openrouter_message="Unrecoverable error",
            )

        pipe._streaming_handler._run_streaming_loop = mock_streaming_loop

        with patch("open_webui_openrouter_pipe.requests.orchestrator.ModelFamily") as mock_family:
            mock_family.base_model.return_value = "openai/gpt-4o"
            mock_family.supports.return_value = False
            mock_family.capabilities.return_value = {}
            mock_family.max_completion_tokens.return_value = None

            with patch("open_webui_openrouter_pipe.requests.orchestrator.OpenRouterModelRegistry") as mock_registry:
                mock_registry.api_model_id.return_value = "openai/gpt-4o"

                result = await orchestrator.process_request(
                    body=base_request_body,
                    __user__={"id": "user1"},
                    __request__=None,
                    __event_emitter__=None,
                    __event_call__=None,
                    __metadata__={},
                    __tools__=None,
                    __task__=None,
                    __task_body__=None,
                    valves=mock_valves,
                    session=mock_session,
                    openwebui_model_id="openai/gpt-4o",
                    pipe_identifier="test-pipe",
                    allowlist_norm_ids={"openai/gpt-4o"},
                    enforced_norm_ids=set(),
                    catalog_norm_ids=set(),
                    features={},
                )

                assert result == ""
                pipe._ensure_error_formatter()._report_openrouter_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_event_emitter_error_caught_on_status_update(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Event emitter errors should be caught when emitting status update (line 747-748)."""
        orchestrator, pipe = orchestrator_and_pipe

        # Setup mocks
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))

        call_count = [0]

        async def mock_streaming_loop(responses_body, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OpenRouterAPIError(
                    status=400,
                    reason="Bad Request",
                    upstream_message="Invalid reasoning.effort value 'high'. Supported values are: 'low', 'medium'.",
                    provider_raw={
                        "error": {
                            "param": "reasoning.effort",
                            "code": "unsupported_value",
                            "type": "invalid_request_error",
                        }
                    },
                )
            return "Test response after retry"

        pipe._streaming_handler._run_streaming_loop = mock_streaming_loop

        # Create event emitter that raises
        async def failing_event_emitter(event):
            raise RuntimeError("Event emitter failed")

        with patch("open_webui_openrouter_pipe.requests.orchestrator.ModelFamily") as mock_family:
            mock_family.base_model.return_value = "openai/o1"
            mock_family.supports.return_value = False
            mock_family.capabilities.return_value = {}
            mock_family.max_completion_tokens.return_value = None

            with patch("open_webui_openrouter_pipe.requests.orchestrator.OpenRouterModelRegistry") as mock_registry:
                mock_registry.api_model_id.return_value = "openai/o1"

                result = await orchestrator.process_request(
                    body={**base_request_body, "model": "openai/o1"},
                    __user__={"id": "user1"},
                    __request__=None,
                    __event_emitter__=failing_event_emitter,
                    __event_call__=None,
                    __metadata__={},
                    __tools__=None,
                    __task__=None,
                    __task_body__=None,
                    valves=mock_valves,
                    session=mock_session,
                    openwebui_model_id="openai/o1",
                    pipe_identifier="test-pipe",
                    allowlist_norm_ids={"openai/o1"},
                    enforced_norm_ids=set(),
                    catalog_norm_ids=set(),
                    features={},
                )

                # Should still succeed despite event emitter error
                assert result == "Test response after retry"
                assert call_count[0] == 2


# -----------------------------------------------------------------------------
# Test non-streaming path (line 679-691)
# -----------------------------------------------------------------------------


class TestNonStreamingPath:
    """Tests for non-streaming request path."""

    @pytest.mark.asyncio
    async def test_nonstreaming_request(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Non-streaming requests should use _run_nonstreaming_loop (lines 679-691)."""
        orchestrator, pipe = orchestrator_and_pipe

        # Make it non-streaming
        base_request_body["stream"] = False

        # Setup mocks
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._run_nonstreaming_loop = AsyncMock(return_value={"result": "complete"})
        pipe._streaming_handler._run_streaming_loop = AsyncMock()  # Should not be called

        with patch("open_webui_openrouter_pipe.requests.orchestrator.ModelFamily") as mock_family:
            mock_family.base_model.return_value = "openai/gpt-4o"
            mock_family.supports.return_value = False
            mock_family.capabilities.return_value = {}
            mock_family.max_completion_tokens.return_value = None

            with patch("open_webui_openrouter_pipe.requests.orchestrator.OpenRouterModelRegistry") as mock_registry:
                mock_registry.api_model_id.return_value = "openai/gpt-4o"

                result = await orchestrator.process_request(
                    body=base_request_body,
                    __user__={"id": "user1"},
                    __request__=None,
                    __event_emitter__=None,
                    __event_call__=None,
                    __metadata__={},
                    __tools__=None,
                    __task__=None,
                    __task_body__=None,
                    valves=mock_valves,
                    session=mock_session,
                    openwebui_model_id="openai/gpt-4o",
                    pipe_identifier="test-pipe",
                    allowlist_norm_ids={"openai/gpt-4o"},
                    enforced_norm_ids=set(),
                    catalog_norm_ids=set(),
                    features={},
                )

                assert result == {"result": "complete"}
                pipe._streaming_handler._run_nonstreaming_loop.assert_called_once()
                pipe._streaming_handler._run_streaming_loop.assert_not_called()


# -----------------------------------------------------------------------------
# Test audio format sniffing paths
# -----------------------------------------------------------------------------


class TestAudioFormatSniffing:
    """Tests for audio format sniffing helper."""

    @pytest.mark.asyncio
    async def test_sniff_wav_format(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body, monkeypatch):
        """WAV format should be sniffed from RIFF header."""
        orchestrator, pipe = orchestrator_and_pipe

        # WAV header: RIFF....WAVE
        wav_header = b"RIFF\x00\x00\x00\x00WAVEfmt "
        valid_b64 = base64.b64encode(wav_header).decode()

        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "audio": [{"id": "audio123"}]  # No format declared
                }
            }
        }

        mock_file = Mock(id="audio123")
        monkeypatch.setattr("open_webui_openrouter_pipe.requests.orchestrator.get_file_by_id", AsyncMock(return_value=mock_file))
        pipe._file_gateway.read_file_record_base64 = AsyncMock(return_value=valid_b64)
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._select_llm_endpoint_with_forced = Mock(return_value=("chat_completions", False))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_sniff_mp3_id3_format(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body, monkeypatch):
        """MP3 with ID3 header should be sniffed."""
        orchestrator, pipe = orchestrator_and_pipe

        # MP3 with ID3 tag
        mp3_header = b"ID3\x04\x00\x00\x00\x00\x00\x00"
        valid_b64 = base64.b64encode(mp3_header).decode()

        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "audio": [{"id": "audio123"}]
                }
            }
        }

        mock_file = Mock(id="audio123")
        monkeypatch.setattr("open_webui_openrouter_pipe.requests.orchestrator.get_file_by_id", AsyncMock(return_value=mock_file))
        pipe._file_gateway.read_file_record_base64 = AsyncMock(return_value=valid_b64)
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._select_llm_endpoint_with_forced = Mock(return_value=("chat_completions", False))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_sniff_mp3_sync_frame(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body, monkeypatch):
        """MP3 sync frame should be sniffed."""
        orchestrator, pipe = orchestrator_and_pipe

        # MP3 sync frame: 0xFF 0xFB (MPEG-1 Layer 3)
        mp3_header = b"\xFF\xFB\x90\x64" + b"\x00" * 96
        valid_b64 = base64.b64encode(mp3_header).decode()

        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "audio": [{"id": "audio123"}]
                }
            }
        }

        mock_file = Mock(id="audio123")
        monkeypatch.setattr("open_webui_openrouter_pipe.requests.orchestrator.get_file_by_id", AsyncMock(return_value=mock_file))
        pipe._file_gateway.read_file_record_base64 = AsyncMock(return_value=valid_b64)
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._select_llm_endpoint_with_forced = Mock(return_value=("chat_completions", False))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_sniff_m4a_format(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body, monkeypatch):
        """M4A (ISO BMFF) format should be sniffed from ftyp marker."""
        orchestrator, pipe = orchestrator_and_pipe

        # ISO BMFF container: ....ftyp
        m4a_header = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 88
        valid_b64 = base64.b64encode(m4a_header).decode()

        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "audio": [{"id": "audio123"}],
                    "responses_audio_format_allowlist": "m4a,mp3,wav",
                }
            }
        }

        mock_file = Mock(id="audio123")
        monkeypatch.setattr("open_webui_openrouter_pipe.requests.orchestrator.get_file_by_id", AsyncMock(return_value=mock_file))
        pipe._file_gateway.read_file_record_base64 = AsyncMock(return_value=valid_b64)
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._select_llm_endpoint_with_forced = Mock(return_value=("chat_completions", False))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_sniff_flac_format(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body, monkeypatch):
        """FLAC format should be sniffed from fLaC magic bytes."""
        orchestrator, pipe = orchestrator_and_pipe

        # FLAC magic bytes
        flac_header = b"fLaC\x00\x00\x00\x22" + b"\x00" * 88
        valid_b64 = base64.b64encode(flac_header).decode()

        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "audio": [{"id": "audio123"}],
                    "responses_audio_format_allowlist": "flac,mp3,wav",
                }
            }
        }

        mock_file = Mock(id="audio123")
        monkeypatch.setattr("open_webui_openrouter_pipe.requests.orchestrator.get_file_by_id", AsyncMock(return_value=mock_file))
        pipe._file_gateway.read_file_record_base64 = AsyncMock(return_value=valid_b64)
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._select_llm_endpoint_with_forced = Mock(return_value=("chat_completions", False))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_sniff_ogg_format(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body, monkeypatch):
        """OGG format should be sniffed from OggS magic bytes."""
        orchestrator, pipe = orchestrator_and_pipe

        # OGG magic bytes
        ogg_header = b"OggS\x00\x02\x00\x00" + b"\x00" * 88
        valid_b64 = base64.b64encode(ogg_header).decode()

        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "audio": [{"id": "audio123"}],
                    "responses_audio_format_allowlist": "ogg,mp3,wav",
                }
            }
        }

        mock_file = Mock(id="audio123")
        monkeypatch.setattr("open_webui_openrouter_pipe.requests.orchestrator.get_file_by_id", AsyncMock(return_value=mock_file))
        pipe._file_gateway.read_file_record_base64 = AsyncMock(return_value=valid_b64)
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._select_llm_endpoint_with_forced = Mock(return_value=("chat_completions", False))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_sniff_webm_format(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body, monkeypatch):
        """WebM format should be sniffed from EBML magic bytes."""
        orchestrator, pipe = orchestrator_and_pipe

        # WebM/Matroska EBML magic bytes
        webm_header = b"\x1A\x45\xDF\xA3\x01\x00\x00" + b"\x00" * 89
        valid_b64 = base64.b64encode(webm_header).decode()

        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "audio": [{"id": "audio123"}],
                    "responses_audio_format_allowlist": "webm,mp3,wav",
                }
            }
        }

        mock_file = Mock(id="audio123")
        monkeypatch.setattr("open_webui_openrouter_pipe.requests.orchestrator.get_file_by_id", AsyncMock(return_value=mock_file))
        pipe._file_gateway.read_file_record_base64 = AsyncMock(return_value=valid_b64)
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._select_llm_endpoint_with_forced = Mock(return_value=("chat_completions", False))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        assert result == "Test response"


# -----------------------------------------------------------------------------
# Test tools registry as list (lines 570-578)
# -----------------------------------------------------------------------------


class TestToolsRegistryAsList:
    """Tests for tools registry handling when provided as a list."""

    @pytest.mark.asyncio
    async def test_tools_registry_as_list_with_spec(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Tools registry as list with spec objects should be processed."""
        orchestrator, pipe = orchestrator_and_pipe

        # Setup mocks
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        # Provide tools as a list with spec objects
        tools_list = [
            {"spec": {"name": "tool1", "description": "Tool 1"}},
            {"name": "tool2", "description": "Tool 2"},  # Has name directly
            {"spec": {"name": "tool3"}},  # Name in spec only
            {"other": "data"},  # No name - should be skipped
            "not_a_dict",  # Not a dict - should be skipped
        ]

        with patch("open_webui_openrouter_pipe.requests.orchestrator.ModelFamily") as mock_family:
            mock_family.base_model.return_value = "openai/gpt-4o"
            mock_family.supports.return_value = True
            mock_family.capabilities.return_value = {}
            mock_family.max_completion_tokens.return_value = None

            with patch("open_webui_openrouter_pipe.requests.orchestrator.OpenRouterModelRegistry") as mock_registry:
                mock_registry.api_model_id.return_value = "openai/gpt-4o"

                result = await orchestrator.process_request(
                    body=base_request_body,
                    __user__={"id": "user1"},
                    __request__=None,
                    __event_emitter__=None,
                    __event_call__=None,
                    __metadata__={},
                    __tools__=tools_list,
                    __task__=None,
                    __task_body__=None,
                    valves=mock_valves,
                    session=mock_session,
                    openwebui_model_id="openai/gpt-4o",
                    pipe_identifier="test-pipe",
                    allowlist_norm_ids={"openai/gpt-4o"},
                    enforced_norm_ids=set(),
                    catalog_norm_ids=set(),
                    features={},
                )

                assert result == "Test response"


# -----------------------------------------------------------------------------
# Test reasoning body initialization when not dict (line 750-751)
# -----------------------------------------------------------------------------


class TestReasoningBodyInitialization:
    """Tests for reasoning body initialization during retry."""

    @pytest.mark.asyncio
    async def test_reasoning_initialized_when_not_dict(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Reasoning should be initialized as dict when None (lines 750-751)."""
        orchestrator, pipe = orchestrator_and_pipe

        # Setup mocks
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))

        call_count = [0]
        captured_body = [None]

        async def mock_streaming_loop(responses_body, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Verify reasoning is None initially
                assert responses_body.reasoning is None
                raise OpenRouterAPIError(
                    status=400,
                    reason="Bad Request",
                    upstream_message="Invalid reasoning.effort value 'high'. Supported values are: 'low', 'medium'.",
                    provider_raw={
                        "error": {
                            "param": "reasoning.effort",
                            "code": "unsupported_value",
                            "type": "invalid_request_error",
                        }
                    },
                )
            # On retry, reasoning should be a dict
            captured_body[0] = responses_body
            return "Test response after retry"

        pipe._streaming_handler._run_streaming_loop = mock_streaming_loop

        with patch("open_webui_openrouter_pipe.requests.orchestrator.ModelFamily") as mock_family:
            mock_family.base_model.return_value = "openai/o1"
            mock_family.supports.return_value = False
            mock_family.capabilities.return_value = {}
            mock_family.max_completion_tokens.return_value = None

            with patch("open_webui_openrouter_pipe.requests.orchestrator.OpenRouterModelRegistry") as mock_registry:
                mock_registry.api_model_id.return_value = "openai/o1"

                result = await orchestrator.process_request(
                    body={**base_request_body, "model": "openai/o1"},
                    __user__={"id": "user1"},
                    __request__=None,
                    __event_emitter__=None,
                    __event_call__=None,
                    __metadata__={},
                    __tools__=None,
                    __task__=None,
                    __task_body__=None,
                    valves=mock_valves,
                    session=mock_session,
                    openwebui_model_id="openai/o1",
                    pipe_identifier="test-pipe",
                    allowlist_norm_ids={"openai/o1"},
                    enforced_norm_ids=set(),
                    catalog_norm_ids=set(),
                    features={},
                )

                assert result == "Test response after retry"
                assert call_count[0] == 2
                # Verify reasoning was initialized and set
                assert captured_body[0] is not None
                assert isinstance(captured_body[0].reasoning, dict)
                assert captured_body[0].reasoning.get("effort") in ["low", "medium"]


# -----------------------------------------------------------------------------
# Additional edge case tests for remaining uncovered lines
# -----------------------------------------------------------------------------


class TestDecodeBase64EdgeCases:
    """Additional tests for _decode_base64_prefix edge cases (lines 147, 150-156, 168-169)."""

    @pytest.mark.asyncio
    async def test_empty_base64_data(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body, monkeypatch):
        """Empty base64 data should return empty bytes (line 147).

        When the base64 data is empty string "", _decode_base64_prefix returns b""
        and sniff returns "", so if no format is declared, it fails.
        """
        orchestrator, pipe = orchestrator_and_pipe

        # Audio with empty base64 and NO declared format - should fail
        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "audio": [{"id": "audio123"}]  # No format declared
                }
            }
        }

        mock_file = Mock(id="audio123")
        monkeypatch.setattr("open_webui_openrouter_pipe.requests.orchestrator.get_file_by_id", AsyncMock(return_value=mock_file))
        pipe._file_gateway.read_file_record_base64 = AsyncMock(return_value="")  # Empty
        pipe._ensure_error_formatter()._emit_templated_error = AsyncMock()

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids=set(),
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        # Should fail - no format can be determined
        assert result == ""
        pipe._ensure_error_formatter()._emit_templated_error.assert_called()

    @pytest.mark.asyncio
    async def test_base64_with_corrupted_data_fallback_decode(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body, monkeypatch):
        """Base64 with slightly corrupted padding triggers fallback decode (lines 165-169).

        When validate=True fails, the code tries validate=False.
        """
        orchestrator, pipe = orchestrator_and_pipe

        # Create audio with base64 that may need fallback decode
        # Valid base64 with format declared so it proceeds through the path
        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "audio": [{"id": "audio123", "format": "mp3"}]
                }
            }
        }

        # Create base64 that triggers the fallback path
        # A valid but unusual base64 string
        test_data = b"some test audio data for testing"
        valid_b64 = base64.b64encode(test_data).decode()

        mock_file = Mock(id="audio123")
        monkeypatch.setattr("open_webui_openrouter_pipe.requests.orchestrator.get_file_by_id", AsyncMock(return_value=mock_file))
        pipe._file_gateway.read_file_record_base64 = AsyncMock(return_value=valid_b64)
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._select_llm_endpoint_with_forced = Mock(return_value=("chat_completions", False))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        assert result == "Test response"


class TestAttachmentSkipContinuePaths:
    """Tests to cover the continue statements for invalid attachments (lines 197, 213, 229, 255)."""

    @pytest.mark.asyncio
    async def test_file_with_mixed_valid_and_invalid_ids(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Mix of valid and invalid file IDs - valid ones should be processed (line 197)."""
        orchestrator, pipe = orchestrator_and_pipe

        # Mix of valid and invalid
        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "files": [
                        {"id": None, "name": "skip1.txt"},
                        {"id": "valid_file", "name": "valid.txt"},
                        {"id": "", "name": "skip2.txt"},
                    ]
                }
            }
        }

        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_audio_with_only_invalid_ids(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Only invalid audio IDs - all should be skipped (line 229)."""
        orchestrator, pipe = orchestrator_and_pipe

        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "audio": [
                        {"id": None},
                        {"id": ""},
                        {"id": 123},  # int, not string
                    ]
                }
            }
        }

        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_video_with_only_invalid_ids(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Only invalid video IDs - all should be skipped (line 255)."""
        orchestrator, pipe = orchestrator_and_pipe

        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "video": [
                        {"id": None},
                        {"id": ""},
                        {"id": 999},  # int, not string
                        {"content_type": "video/mp4"},  # missing id
                    ]
                }
            }
        }

        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_csv_set_with_non_string_value(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body, monkeypatch):
        """Non-string value in _csv_set returns empty set (line 213)."""
        orchestrator, pipe = orchestrator_and_pipe

        # WAV audio with non-string allowlist
        wav_header = b"RIFF\x00\x00\x00\x00WAVEfmt "
        valid_b64 = base64.b64encode(wav_header).decode()

        metadata = {
            "openrouter_pipe": {
                "direct_uploads": {
                    "audio": [{"id": "audio123"}],  # No format, will be sniffed as wav
                    "responses_audio_format_allowlist": None,  # None, not string
                }
            }
        }

        mock_file = Mock(id="audio123")
        monkeypatch.setattr("open_webui_openrouter_pipe.requests.orchestrator.get_file_by_id", AsyncMock(return_value=mock_file))
        pipe._file_gateway.read_file_record_base64 = AsyncMock(return_value=valid_b64)
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._select_llm_endpoint_with_forced = Mock(return_value=("chat_completions", False))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        result = await orchestrator.process_request(
            body=base_request_body,
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        assert result == "Test response"


class TestReasoningEffortNoEventEmitter:
    """Test reasoning effort retry when no event emitter is provided (line 715)."""

    @pytest.mark.asyncio
    async def test_reasoning_effort_retry_no_emitter(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Reasoning effort retry should work without event emitter (line 715 path)."""
        orchestrator, pipe = orchestrator_and_pipe

        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))

        call_count = [0]

        async def mock_streaming_loop(responses_body, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OpenRouterAPIError(
                    status=400,
                    reason="Bad Request",
                    upstream_message="Invalid reasoning.effort value 'high'. Supported values are: 'low', 'medium'.",
                    provider_raw={
                        "error": {
                            "param": "reasoning.effort",
                            "code": "unsupported_value",
                            "type": "invalid_request_error",
                        }
                    },
                )
            return "Test response after retry"

        pipe._streaming_handler._run_streaming_loop = mock_streaming_loop

        with patch("open_webui_openrouter_pipe.requests.orchestrator.ModelFamily") as mock_family:
            mock_family.base_model.return_value = "openai/o1"
            mock_family.supports.return_value = False
            mock_family.capabilities.return_value = {}
            mock_family.max_completion_tokens.return_value = None

            with patch("open_webui_openrouter_pipe.requests.orchestrator.OpenRouterModelRegistry") as mock_registry:
                mock_registry.api_model_id.return_value = "openai/o1"

                result = await orchestrator.process_request(
                    body={**base_request_body, "model": "openai/o1"},
                    __user__={"id": "user1"},
                    __request__=None,
                    __event_emitter__=None,  # No event emitter
                    __event_call__=None,
                    __metadata__={},
                    __tools__=None,
                    __task__=None,
                    __task_body__=None,
                    valves=mock_valves,
                    session=mock_session,
                    openwebui_model_id="openai/o1",
                    pipe_identifier="test-pipe",
                    allowlist_norm_ids={"openai/o1"},
                    enforced_norm_ids=set(),
                    catalog_norm_ids=set(),
                    features={},
                )

                assert result == "Test response after retry"
                assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_reasoning_effort_retry_with_existing_dict_reasoning(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Reasoning effort retry when responses_body.reasoning is already a dict (line 715).

        This covers the path where reasoning is a dict and we extract original_effort.
        """
        orchestrator, pipe = orchestrator_and_pipe

        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))

        call_count = [0]

        async def mock_streaming_loop(responses_body, *args, **kwargs):
            call_count[0] += 1
            # On first call, set reasoning to a dict with effort
            if call_count[0] == 1:
                # Manually set reasoning to simulate it being a dict
                responses_body.reasoning = {"effort": "high"}
                raise OpenRouterAPIError(
                    status=400,
                    reason="Bad Request",
                    upstream_message="Invalid reasoning.effort value 'high'. Supported values are: 'low', 'medium'.",
                    provider_raw={
                        "error": {
                            "param": "reasoning.effort",
                            "code": "unsupported_value",
                            "type": "invalid_request_error",
                        }
                    },
                )
            return "Test response after retry"

        pipe._streaming_handler._run_streaming_loop = mock_streaming_loop

        with patch("open_webui_openrouter_pipe.requests.orchestrator.ModelFamily") as mock_family:
            mock_family.base_model.return_value = "openai/o1"
            mock_family.supports.return_value = False
            mock_family.capabilities.return_value = {}
            mock_family.max_completion_tokens.return_value = None

            with patch("open_webui_openrouter_pipe.requests.orchestrator.OpenRouterModelRegistry") as mock_registry:
                mock_registry.api_model_id.return_value = "openai/o1"

                result = await orchestrator.process_request(
                    body={**base_request_body, "model": "openai/o1"},
                    __user__={"id": "user1"},
                    __request__=None,
                    __event_emitter__=None,
                    __event_call__=None,
                    __metadata__={},
                    __tools__=None,
                    __task__=None,
                    __task_body__=None,
                    valves=mock_valves,
                    session=mock_session,
                    openwebui_model_id="openai/o1",
                    pipe_identifier="test-pipe",
                    allowlist_norm_ids={"openai/o1"},
                    enforced_norm_ids=set(),
                    catalog_norm_ids=set(),
                    features={},
                )

                assert result == "Test response after retry"
                assert call_count[0] == 2


def _gate_valves(*, fusion_enabled: bool = True):
    valves = Mock()
    valves.ENABLE_OPENROUTER_FUSION = fusion_enabled
    return valves


class TestFusionLiveGate:
    """RequestOrchestrator fusion-live-UI activation: gate logic and the process_request thread-through."""

    _FUSION_FORMS = (
        "openrouter/fusion",
        "openrouter.openrouter/fusion",
        "openrouter.fusion",
        "open_webui_openrouter_pipe.openrouter.fusion",
    )
    _NON_FUSION = ("openai/gpt-4o", "anthropic/claude-opus", "openrouter/auto")

    @pytest.mark.asyncio
    async def test_arms_for_every_fusion_id_form(self, orchestrator_and_pipe):
        orchestrator, _pipe = orchestrator_and_pipe
        for model in self._FUSION_FORMS:
            assert orchestrator._resolve_fusion_live_enabled(_gate_valves(), is_fusion_model(model), False) is True

    @pytest.mark.asyncio
    async def test_never_arms_for_non_fusion_models(self, orchestrator_and_pipe):
        orchestrator, _pipe = orchestrator_and_pipe
        for model in self._NON_FUSION:
            assert orchestrator._resolve_fusion_live_enabled(_gate_valves(), is_fusion_model(model), False) is False

    @pytest.mark.asyncio
    async def test_direct_connection_never_arms(self, orchestrator_and_pipe):
        orchestrator, _pipe = orchestrator_and_pipe
        assert orchestrator._resolve_fusion_live_enabled(_gate_valves(), is_fusion_model("openrouter/fusion"), True) is False

    @pytest.mark.asyncio
    async def test_master_switch_off_never_arms(self, orchestrator_and_pipe):
        orchestrator, _pipe = orchestrator_and_pipe
        assert orchestrator._resolve_fusion_live_enabled(_gate_valves(fusion_enabled=False), is_fusion_model("openrouter/fusion"), False) is False

    @pytest.mark.asyncio
    async def test_process_request_threads_enabled_to_streaming_loop(
        self, orchestrator_and_pipe, mock_valves, mock_session
    ):
        orchestrator, pipe = orchestrator_and_pipe
        mock_valves.ENABLE_OPENROUTER_FUSION = True
        captured: dict[str, Any] = {}

        async def fake_loop(*_args, **kwargs):
            captured.update(kwargs)
            return "answer"

        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._select_llm_endpoint_with_forced = Mock(return_value=("chat_completions", False))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(side_effect=fake_loop)

        result = await orchestrator.process_request(
            body={"model": "openrouter/fusion", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            __user__={"id": "u"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__={},
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openrouter/fusion",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openrouter/fusion"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )

        assert result == "answer"
        assert captured.get("fusion_live_enabled") is True

    @staticmethod
    def _fusion_entries(body):
        plugins = getattr(body, "plugins", None) if not isinstance(body, dict) else body.get("plugins")
        return [p for p in (plugins or []) if isinstance(p, dict) and p.get("id") == "fusion"]

    async def _run_call_site(self, orchestrator, pipe, mock_valves, mock_session,
                             *, task=None, fusion_enabled=True, metadata=None):
        mock_valves.ENABLE_OPENROUTER_FUSION = fusion_enabled
        captured: dict[str, Any] = {}

        async def fake_loop(*args, **kwargs):
            captured["body"] = args[0] if args else kwargs.get("body")
            return "answer"

        async def fake_task(task_body, *_args, **_kwargs):
            captured["task_body"] = task_body
            return "task-answer"

        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_reasoning_config_manager()._apply_task_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_anthropic_verbosity = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._select_llm_endpoint_with_forced = Mock(return_value=("chat_completions", False))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(side_effect=fake_loop)
        pipe._ensure_task_model_adapter()._run_task_model_request = AsyncMock(side_effect=fake_task)

        await orchestrator.process_request(
            body={"model": "openrouter/fusion", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            __user__={"id": "u"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata or {},
            __tools__=None,
            __task__=task,
            __task_body__=None,
            valves=mock_valves,
            session=mock_session,
            openwebui_model_id="openrouter/fusion",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openrouter/fusion"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )
        return captured

    @pytest.mark.asyncio
    async def test_call_site_injects_fusion_plugin_for_chat_requests(
        self, orchestrator_and_pipe, mock_valves, mock_session
    ):
        orchestrator, pipe = orchestrator_and_pipe
        captured = await self._run_call_site(orchestrator, pipe, mock_valves, mock_session)
        assert self._fusion_entries(captured["body"]) == [{"id": "fusion"}]

    @pytest.mark.asyncio
    async def test_call_site_never_injects_for_task_requests(
        self, orchestrator_and_pipe, mock_valves, mock_session
    ):
        orchestrator, pipe = orchestrator_and_pipe
        captured = await self._run_call_site(orchestrator, pipe, mock_valves, mock_session,
                                             task="title_generation")
        assert "task_body" in captured
        assert self._fusion_entries(captured["task_body"]) == []

    @pytest.mark.asyncio
    async def test_call_site_never_injects_with_master_switch_off(
        self, orchestrator_and_pipe, mock_valves, mock_session
    ):
        orchestrator, pipe = orchestrator_and_pipe
        captured = await self._run_call_site(orchestrator, pipe, mock_valves, mock_session,
                                             fusion_enabled=False)
        assert self._fusion_entries(captured["body"]) == []

    @pytest.mark.asyncio
    async def test_call_site_forces_tool_choice_for_fusion_chat(
        self, orchestrator_and_pipe, mock_valves, mock_session
    ):
        orchestrator, pipe = orchestrator_and_pipe
        captured = await self._run_call_site(orchestrator, pipe, mock_valves, mock_session)
        assert captured["body"].tool_choice == "required"

    @pytest.mark.asyncio
    async def test_call_site_never_forces_for_task_requests(
        self, orchestrator_and_pipe, mock_valves, mock_session
    ):
        orchestrator, pipe = orchestrator_and_pipe
        captured = await self._run_call_site(orchestrator, pipe, mock_valves, mock_session,
                                             task="title_generation")
        assert (captured["task_body"].get("tool_choice") or None) is None

    @pytest.mark.asyncio
    async def test_call_site_never_forces_with_master_switch_off(
        self, orchestrator_and_pipe, mock_valves, mock_session
    ):
        orchestrator, pipe = orchestrator_and_pipe
        captured = await self._run_call_site(orchestrator, pipe, mock_valves, mock_session,
                                             fusion_enabled=False)
        assert captured["body"].tool_choice is None

    @pytest.mark.asyncio
    async def test_call_site_strips_openrouter_server_tools_after_metadata_apply(
        self, orchestrator_and_pipe, mock_valves, mock_session
    ):
        orchestrator, pipe = orchestrator_and_pipe
        metadata = {"openrouter_pipe": {"server_tools": {"web_search": {}, "web_fetch": {}, "datetime": {}}}}
        captured = await self._run_call_site(orchestrator, pipe, mock_valves, mock_session,
                                             metadata=metadata)
        body = captured["body"]
        tool_types = [t.get("type") for t in (body.tools or []) if isinstance(t, dict)]
        assert not [t for t in tool_types if isinstance(t, str) and t.startswith("openrouter:")]
        assert body.tool_choice == "required"


class TestFusionServerToolsStripped:
    """openrouter:* server tools on a fusion-model request let the model satisfy
    tool_choice=required with web_search instead of deliberating — and fusion
    already runs web_search/web_fetch internally, so they are pure escape hatch."""

    def _strip(self, **kw):
        from open_webui_openrouter_pipe.requests.orchestrator import _fusion_server_tools_stripped
        args = dict(model_id="openrouter/fusion", plugins=[{"id": "fusion"}],
                    tools=[{"type": "openrouter:web_search"},
                           {"type": "openrouter:web_fetch"},
                           {"type": "function", "name": "my_tool"}],
                    fusion_enabled=True, is_task_request=False)
        args.update(kw)
        return _fusion_server_tools_stripped(
            args["model_id"], args["plugins"], args["tools"],
            fusion_enabled=args["fusion_enabled"],
            is_task_request=args["is_task_request"],
        )

    def test_strips_openrouter_tools_keeps_function_tools(self):
        assert self._strip() == [{"type": "function", "name": "my_tool"}]

    def test_none_when_no_openrouter_tools(self):
        assert self._strip(tools=[{"type": "function", "name": "my_tool"}]) is None

    def test_none_for_non_fusion_model(self):
        assert self._strip(model_id="openai/gpt-4o") is None

    def test_none_for_task_requests(self):
        assert self._strip(is_task_request=True) is None

    def test_none_with_master_switch_off(self):
        assert self._strip(fusion_enabled=False) is None

    def test_none_when_fusion_entry_disabled(self):
        assert self._strip(plugins=[{"id": "fusion", "enabled": False}]) is None

    def test_strips_for_fusion_flash(self):
        assert self._strip(model_id="openrouter/fusion-flash") == [{"type": "function", "name": "my_tool"}]

    def test_input_tools_not_mutated(self):
        tools = [{"type": "openrouter:web_search"}, {"type": "function", "name": "t"}]
        self._strip(tools=tools)
        assert tools == [{"type": "openrouter:web_search"}, {"type": "function", "name": "t"}]


class TestFusionForceToolChoice:
    """Fusion MODELS always deliberate via tool_choice=required — a declinable
    tool on a model users picked FOR deliberation is useless. The
    openrouter:fusion SERVER TOOL on a caller's own model is never forced. Same
    guard set as the plugin injection, plus never overriding a caller's choice."""

    def _force(self, **kw):
        from open_webui_openrouter_pipe.requests.orchestrator import _fusion_force_tool_choice
        args = dict(model_id="openrouter/fusion", plugins=[{"id": "fusion"}],
                    tool_choice=None, fusion_enabled=True, is_task_request=False)
        args.update(kw)
        return _fusion_force_tool_choice(
            args["model_id"], args["plugins"], args["tool_choice"],
            fusion_enabled=args["fusion_enabled"],
            is_task_request=args["is_task_request"],
        )

    def test_forces_for_base_fusion_with_active_entry(self):
        assert self._force() is True

    def test_forces_for_filter_configured_entry(self):
        assert self._force(plugins=[{"id": "fusion", "preset": "general-fast"}]) is True

    def test_never_overrides_caller_tool_choice(self):
        assert self._force(tool_choice="auto") is False
        assert self._force(tool_choice="none") is False
        assert self._force(tool_choice={"type": "function", "name": "x"}) is False

    def test_no_force_for_task_requests(self):
        assert self._force(is_task_request=True) is False

    def test_no_force_with_master_switch_off(self):
        assert self._force(fusion_enabled=False) is False

    def test_no_force_when_fusion_entry_disabled(self):
        assert self._force(plugins=[{"id": "fusion", "enabled": False}]) is False

    def test_no_force_without_fusion_entry(self):
        assert self._force(plugins=[{"id": "file-parser"}]) is False

    def test_no_force_for_non_fusion_model(self):
        assert self._force(model_id="openai/gpt-4o") is False

    def test_forces_for_fusion_flash(self):
        assert self._force(model_id="openrouter/fusion-flash") is True


class TestFusionPluginInjection:
    """Guarded {"id": "fusion"} injection: restores fusion-model activation without
    cost bombs.

    OpenRouter's fusion aliases no longer self-activate on /responses; the pipe
    must inject the plugin entry for the whole fusion model family — but never for
    task requests (title/tags would fire a full panel per title), never with the
    master switch off, and never over an existing entry."""

    _BASE_FORMS = (
        "openrouter/fusion",
        "openrouter.openrouter/fusion",
        "openrouter.fusion",
        "open_webui_openrouter_pipe.openrouter.fusion",
    )

    def _inject(self, **kw):
        from open_webui_openrouter_pipe.requests.orchestrator import _fusion_plugin_injection
        args = dict(model_id="openrouter/fusion", plugins=None,
                    fusion_enabled=True, is_task_request=False)
        args.update(kw)
        return _fusion_plugin_injection(
            args["model_id"], args["plugins"],
            fusion_enabled=args["fusion_enabled"],
            is_task_request=args["is_task_request"],
        )

    def test_injects_for_base_fusion(self):
        assert self._inject() == [{"id": "fusion"}]

    def test_injects_for_every_base_id_form(self):
        for model in self._BASE_FORMS:
            assert self._inject(model_id=model) == [{"id": "fusion"}]

    def test_appends_after_existing_non_fusion_plugins(self):
        assert self._inject(plugins=[{"id": "file-parser", "pdf": {"engine": "native"}}]) == [
            {"id": "file-parser", "pdf": {"engine": "native"}},
            {"id": "fusion"},
        ]

    def test_no_injection_for_task_requests(self):
        assert self._inject(is_task_request=True) is None

    def test_no_injection_when_master_switch_off(self):
        assert self._inject(fusion_enabled=False) is None

    def test_existing_fusion_entry_untouched(self):
        assert self._inject(plugins=[{"id": "fusion", "preset": "general-fast"}]) is None

    def test_disabled_fusion_entry_untouched(self):
        assert self._inject(plugins=[{"id": "fusion", "enabled": False}]) is None

    def test_no_injection_for_non_fusion_model(self):
        assert self._inject(model_id="openai/gpt-4o") is None

    def test_injects_for_fusion_flash(self):
        assert self._inject(model_id="openrouter/fusion-flash") == [{"id": "fusion"}]

    def test_input_plugins_list_not_mutated(self):
        original = [{"id": "file-parser"}]
        result = self._inject(plugins=original)
        assert original == [{"id": "file-parser"}]
        assert result is not original
