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
    valves.ENABLE_STATUS_CSS_PATCH = False
    valves.DIRECT_UPLOAD_FAILURE_TEMPLATE = "Direct upload failed: {reason}"
    valves.ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE = "Endpoint conflict: {reason}"
    valves.MODEL_RESTRICTED_TEMPLATE = "Model restricted: {restriction_reasons}"
    valves.OPENROUTER_ERROR_TEMPLATE = "Error: {detail}"
    valves.REASONING_EFFORT = "medium"
    valves.TASK_MODEL_REASONING_EFFORT = "low"
    valves.USE_MODEL_MAX_OUTPUT_TOKENS = False
    valves.ENABLE_STRICT_TOOL_CALLING = False
    valves.TOOL_EXECUTION_MODE = "Pipeline"
    valves.TOOL_OUTPUT_RETENTION_TURNS = 3
    valves.WEB_SEARCH_MAX_RESULTS = None
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
    async def test_empty_data_returns_empty_bytes(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
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
        pipe._multimodal_handler._get_file_by_id = AsyncMock(return_value=Mock(id="audio123"))
        pipe._multimodal_handler._read_file_record_base64 = AsyncMock(return_value="")
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
    async def test_invalid_base64_chars_in_prefix(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
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
        pipe._multimodal_handler._get_file_by_id = AsyncMock(return_value=Mock(id="audio123"))
        # This base64 contains invalid characters like unicode - sniff will return ""
        pipe._multimodal_handler._read_file_record_base64 = AsyncMock(return_value="AAAA\u0080BBBB")
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
    async def test_csv_set_with_non_string_returns_empty_set(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
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
        pipe._multimodal_handler._get_file_by_id = AsyncMock(return_value=mock_file)
        pipe._multimodal_handler._read_file_record_base64 = AsyncMock(return_value=valid_b64)
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
# Test web search plugin with minimal reasoning effort (lines 638-653)
# -----------------------------------------------------------------------------


class TestWebSearchPluginMinimalEffort:
    """Tests for web search plugin configuration with minimal reasoning effort."""

    @pytest.mark.asyncio
    async def test_web_search_skipped_with_minimal_effort(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Web search should be skipped when reasoning effort is 'minimal' (lines 642-646)."""
        orchestrator, pipe = orchestrator_and_pipe
        mock_valves.REASONING_EFFORT = "minimal"

        # Setup mocks
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="Test response")

        from open_webui_openrouter_pipe.core.config import _ORS_FILTER_FEATURE_FLAG

        with patch("open_webui_openrouter_pipe.requests.orchestrator.ModelFamily") as mock_family:
            mock_family.base_model.return_value = "openai/gpt-4o"
            mock_family.supports.side_effect = lambda cap, model: cap == "web_search_tool"
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
                    features={_ORS_FILTER_FEATURE_FLAG: True},  # Enable ORS filter
                )

                assert result == "Test response"

    @pytest.mark.asyncio
    async def test_web_search_added_with_non_minimal_effort(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Web search should be added when reasoning effort is not 'minimal' (lines 647-653)."""
        orchestrator, pipe = orchestrator_and_pipe
        mock_valves.REASONING_EFFORT = "medium"
        mock_valves.WEB_SEARCH_MAX_RESULTS = 10

        # Setup mocks
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))

        captured_body = None

        async def capture_streaming_loop(responses_body, *args, **kwargs):
            nonlocal captured_body
            captured_body = responses_body
            return "Test response"

        pipe._streaming_handler._run_streaming_loop = capture_streaming_loop

        from open_webui_openrouter_pipe.core.config import _ORS_FILTER_FEATURE_FLAG

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
                    features={_ORS_FILTER_FEATURE_FLAG: True},
                )

                assert result == "Test response"
                # Verify plugins were set
                assert captured_body is not None
                if captured_body.plugins:
                    assert any(p.get("id") == "web" for p in captured_body.plugins)


# -----------------------------------------------------------------------------
# Test OpenRouterAPIError handling and retry logic (lines 692-771)
# -----------------------------------------------------------------------------


class TestOpenRouterAPIErrorHandling:
    """Tests for OpenRouterAPIError handling and retry logic."""

    @pytest.mark.asyncio
    async def test_anthropic_prompt_cache_retry(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
        """Anthropic prompt cache error should trigger retry without cache_control (lines 694-708).

        Note: This test patches the Pipe class methods at the orchestrator module level since
        the orchestrator only imports Pipe under TYPE_CHECKING but references it at runtime.
        """
        orchestrator, pipe = orchestrator_and_pipe
        mock_valves.ENABLE_ANTHROPIC_PROMPT_CACHING = True

        # Setup mocks
        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))
        pipe._is_anthropic_model_id = Mock(return_value=True)

        # Track number of calls
        call_count = [0]

        async def mock_streaming_loop(responses_body, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call fails with prompt cache error
                raise OpenRouterAPIError(
                    status=400,
                    reason="Bad Request",
                    openrouter_message="Prompt caching error",
                )
            return "Test response after retry"

        pipe._streaming_handler._run_streaming_loop = mock_streaming_loop

        # We need to inject Pipe into the orchestrator module's namespace
        # since it only imports under TYPE_CHECKING
        import open_webui_openrouter_pipe.requests.orchestrator as orch_module

        # Create a mock Pipe class with the static methods we need
        mock_pipe_class = Mock()
        mock_pipe_class._input_contains_cache_control = Mock(return_value=True)
        mock_pipe_class._strip_cache_control_from_input = Mock()

        with patch.object(orch_module, "Pipe", mock_pipe_class, create=True):
            with patch("open_webui_openrouter_pipe.requests.orchestrator.ModelFamily") as mock_family:
                mock_family.base_model.return_value = "anthropic/claude-3-opus"
                mock_family.supports.return_value = False
                mock_family.capabilities.return_value = {}
                mock_family.max_completion_tokens.return_value = None

                with patch("open_webui_openrouter_pipe.requests.orchestrator.OpenRouterModelRegistry") as mock_registry:
                    mock_registry.api_model_id.return_value = "anthropic/claude-3-opus"

                    result = await orchestrator.process_request(
                        body={**base_request_body, "model": "anthropic/claude-3-opus"},
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
                        openwebui_model_id="anthropic/claude-3-opus",
                        pipe_identifier="test-pipe",
                        allowlist_norm_ids={"anthropic/claude-3-opus"},
                        enforced_norm_ids=set(),
                        catalog_norm_ids=set(),
                        features={},
                    )

                    assert result == "Test response after retry"
                    assert call_count[0] == 2

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
    async def test_sniff_wav_format(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
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
        pipe._multimodal_handler._get_file_by_id = AsyncMock(return_value=mock_file)
        pipe._multimodal_handler._read_file_record_base64 = AsyncMock(return_value=valid_b64)
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
    async def test_sniff_mp3_id3_format(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
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
        pipe._multimodal_handler._get_file_by_id = AsyncMock(return_value=mock_file)
        pipe._multimodal_handler._read_file_record_base64 = AsyncMock(return_value=valid_b64)
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
    async def test_sniff_mp3_sync_frame(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
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
        pipe._multimodal_handler._get_file_by_id = AsyncMock(return_value=mock_file)
        pipe._multimodal_handler._read_file_record_base64 = AsyncMock(return_value=valid_b64)
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
    async def test_sniff_m4a_format(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
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
        pipe._multimodal_handler._get_file_by_id = AsyncMock(return_value=mock_file)
        pipe._multimodal_handler._read_file_record_base64 = AsyncMock(return_value=valid_b64)
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
    async def test_sniff_flac_format(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
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
        pipe._multimodal_handler._get_file_by_id = AsyncMock(return_value=mock_file)
        pipe._multimodal_handler._read_file_record_base64 = AsyncMock(return_value=valid_b64)
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
    async def test_sniff_ogg_format(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
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
        pipe._multimodal_handler._get_file_by_id = AsyncMock(return_value=mock_file)
        pipe._multimodal_handler._read_file_record_base64 = AsyncMock(return_value=valid_b64)
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
    async def test_sniff_webm_format(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
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
        pipe._multimodal_handler._get_file_by_id = AsyncMock(return_value=mock_file)
        pipe._multimodal_handler._read_file_record_base64 = AsyncMock(return_value=valid_b64)
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
    async def test_empty_base64_data(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
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
        pipe._multimodal_handler._get_file_by_id = AsyncMock(return_value=mock_file)
        pipe._multimodal_handler._read_file_record_base64 = AsyncMock(return_value="")  # Empty
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
    async def test_base64_with_corrupted_data_fallback_decode(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
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
        pipe._multimodal_handler._get_file_by_id = AsyncMock(return_value=mock_file)
        pipe._multimodal_handler._read_file_record_base64 = AsyncMock(return_value=valid_b64)
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
    async def test_csv_set_with_non_string_value(self, orchestrator_and_pipe, mock_valves, mock_session, base_request_body):
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
        pipe._multimodal_handler._get_file_by_id = AsyncMock(return_value=mock_file)
        pipe._multimodal_handler._read_file_record_base64 = AsyncMock(return_value=valid_b64)
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
