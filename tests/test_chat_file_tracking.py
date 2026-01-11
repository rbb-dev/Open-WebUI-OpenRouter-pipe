from __future__ import annotations

from unittest.mock import Mock

import pytest


@pytest.mark.asyncio
async def test_upload_to_owui_storage_links_chat_file(pipe_instance, mock_request, mock_user, monkeypatch):
    from open_webui.models.chats import Chats
    from open_webui_openrouter_pipe import open_webui_openrouter_pipe as ow_mod

    insert_mock = Mock()
    monkeypatch.setattr(Chats, "insert_chat_files", insert_mock, raising=False)

    captured: dict[str, object] = {}

    def upload_stub(*_args, **kwargs):
        captured.update(kwargs)
        return Mock(id="file123")

    async def run_in_threadpool_stub(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(ow_mod, "upload_file_handler", upload_stub)
    monkeypatch.setattr(ow_mod, "run_in_threadpool", run_in_threadpool_stub)

    file_id = await pipe_instance._upload_to_owui_storage(
        request=mock_request,
        user=mock_user,
        file_data=b"data",
        filename="generated.png",
        mime_type="image/png",
        chat_id="chat123",
        message_id="msg123",
        owui_user_id="user123",
    )

    assert file_id == "file123"
    metadata = captured.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("chat_id") == "chat123"
    assert metadata.get("message_id") == "msg123"
    insert_mock.assert_called_once()


@pytest.mark.asyncio
async def test_upload_to_owui_storage_ignores_missing_insert_api(pipe_instance, mock_request, mock_user, monkeypatch):
    from open_webui.models.chats import Chats
    from open_webui_openrouter_pipe import open_webui_openrouter_pipe as ow_mod

    monkeypatch.delattr(Chats, "insert_chat_files", raising=False)

    def upload_stub(*_args, **_kwargs):
        return Mock(id="file123")

    async def run_in_threadpool_stub(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(ow_mod, "upload_file_handler", upload_stub)
    monkeypatch.setattr(ow_mod, "run_in_threadpool", run_in_threadpool_stub)

    file_id = await pipe_instance._upload_to_owui_storage(
        request=mock_request,
        user=mock_user,
        file_data=b"data",
        filename="generated.png",
        mime_type="image/png",
        chat_id="chat123",
        message_id="msg123",
        owui_user_id="user123",
    )

    assert file_id == "file123"
