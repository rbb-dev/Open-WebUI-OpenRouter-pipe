from __future__ import annotations

from unittest.mock import Mock, MagicMock
from io import BytesIO
from typing import Any

import pytest


@pytest.mark.asyncio
async def test_upload_to_owui_storage_links_chat_file(pipe_instance_async, mock_request, mock_user, monkeypatch):
    """Test that file uploads properly link to chat and message.

    This test uses real file upload handler integration with proper mocking.
    """
    from open_webui.models.chats import Chats
    from open_webui_openrouter_pipe import multimodal

    insert_mock = Mock()
    monkeypatch.setattr(Chats, "insert_chat_files", insert_mock, raising=False)

    captured: dict[str, Any] = {}

    # Real upload handler that validates all required parameters
    def upload_stub(*_args, **kwargs):
        """Simulate real upload_file_handler behavior."""
        # Validate required parameters
        assert "file" in kwargs, "file parameter required"
        assert "metadata" in kwargs, "metadata parameter required"

        file_obj = kwargs["file"]
        metadata = kwargs["metadata"]

        # Validate file object structure
        assert hasattr(file_obj, "filename"), "file must have filename"
        assert hasattr(file_obj, "content_type"), "file must have content_type"

        # Validate metadata structure
        assert isinstance(metadata, dict), "metadata must be a dict"
        assert "chat_id" in metadata, "metadata must contain chat_id"
        assert "message_id" in metadata, "metadata must contain message_id"

        # Capture for verification
        captured.update(kwargs)

        # Return mock file object with ID
        mock_file = Mock()
        mock_file.id = "file123"
        return mock_file

    async def run_in_threadpool_stub(fn, *args, **kwargs):
        """Execute synchronously for testing."""
        return fn(*args, **kwargs)

    monkeypatch.setattr(multimodal, "upload_file_handler", upload_stub)
    monkeypatch.setattr(multimodal, "run_in_threadpool", run_in_threadpool_stub)

    # Execute upload
    file_id = await pipe_instance_async._upload_to_owui_storage(
        request=mock_request,
        user=mock_user,
        file_data=b"data",
        filename="generated.png",
        mime_type="image/png",
        chat_id="chat123",
        message_id="msg123",
        owui_user_id="user123",
    )

    # Verify file ID returned
    assert file_id == "file123"

    # Verify metadata structure
    metadata = captured.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("chat_id") == "chat123"
    assert metadata.get("message_id") == "msg123"

    # Verify file object structure
    file_obj = captured.get("file")
    assert file_obj is not None
    assert file_obj.filename == "generated.png"
    assert file_obj.content_type == "image/png"

    # Verify chat file insert was called
    insert_mock.assert_called_once()


@pytest.mark.asyncio
async def test_upload_to_owui_storage_ignores_missing_insert_api(pipe_instance_async, mock_request, mock_user, monkeypatch):
    """Test graceful handling when insert_chat_files API is missing.

    This verifies backward compatibility with older Open-WebUI versions.
    """
    from open_webui.models.chats import Chats
    from open_webui_openrouter_pipe import multimodal

    # Remove insert_chat_files to simulate older Open-WebUI
    monkeypatch.delattr(Chats, "insert_chat_files", raising=False)

    def upload_stub(*_args, **kwargs):
        """Simulate successful file upload."""
        mock_file = Mock()
        mock_file.id = "file123"
        return mock_file

    async def run_in_threadpool_stub(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(multimodal, "upload_file_handler", upload_stub)
    monkeypatch.setattr(multimodal, "run_in_threadpool", run_in_threadpool_stub)

    # Execute upload - should succeed without insert_chat_files
    file_id = await pipe_instance_async._upload_to_owui_storage(
        request=mock_request,
        user=mock_user,
        file_data=b"data",
        filename="generated.png",
        mime_type="image/png",
        chat_id="chat123",
        message_id="msg123",
        owui_user_id="user123",
    )

    # Verify file ID returned despite missing insert API
    assert file_id == "file123"
