"""VideoPersistence's message lookup, which nothing executed.

A mutation that returned None before the getter -- making every prior-message lookup
come back empty -- left the whole suite green. That lookup is how a resumed video job
finds the message it is attached to, so silently returning nothing detaches the job
from its own conversation.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from open_webui_openrouter_pipe.storage.video_persistence import (
    VideoPersistence,
    is_local_chat_id,
)


@pytest.fixture
def persistence():
    return VideoPersistence(logger=logging.getLogger("video-persistence-test"))


@pytest.mark.asyncio
async def test_the_stored_message_is_actually_returned(persistence, monkeypatch):
    """The control the mutation defeated: a real row must come back."""
    import open_webui.models.chats as owui_chats

    async def _get(chat_id, message_id, **_kw):
        return {"id": message_id, "content": "the stored text"}

    monkeypatch.setattr(
        owui_chats.Chats, "get_message_by_id_and_message_id", _get, raising=False
    )

    message = await persistence.load_message(chat_id="chat-1", message_id="msg-1")

    assert message is not None, (
        "the message lookup returned nothing for a message that exists; a resumed video "
        "job cannot find the message it belongs to"
    )
    assert message["content"] == "the stored text"


@pytest.mark.asyncio
async def test_the_content_helper_unwraps_both_row_shapes(persistence, monkeypatch):
    """Open WebUI hands back a mapping in some paths and an object in others."""
    import open_webui.models.chats as owui_chats

    async def _as_object(chat_id, message_id, **_kw):
        return SimpleNamespace(id=message_id, content="from an attribute")

    monkeypatch.setattr(
        owui_chats.Chats, "get_message_by_id_and_message_id", _as_object, raising=False
    )
    assert (
        await persistence.load_message_content(chat_id="chat-1", message_id="msg-1")
        == "from an attribute"
    )

    async def _as_dict(chat_id, message_id, **_kw):
        return {"id": message_id, "content": "from a key"}

    monkeypatch.setattr(
        owui_chats.Chats, "get_message_by_id_and_message_id", _as_dict, raising=False
    )
    assert (
        await persistence.load_message_content(chat_id="chat-1", message_id="msg-1")
        == "from a key"
    )


@pytest.mark.asyncio
async def test_a_row_without_content_yields_an_empty_string_not_none(
    persistence, monkeypatch
):
    """Callers concatenate this; None would raise rather than degrade."""
    import open_webui.models.chats as owui_chats

    async def _get(chat_id, message_id, **_kw):
        return {"id": message_id}

    monkeypatch.setattr(
        owui_chats.Chats, "get_message_by_id_and_message_id", _get, raising=False
    )
    assert await persistence.load_message_content(chat_id="c", message_id="m") == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_id", "message_id"),
    [("", "msg-1"), ("chat-1", ""), ("local:abc", "msg-1")],
)
async def test_ids_that_cannot_name_a_stored_message_are_not_looked_up(
    persistence, monkeypatch, chat_id, message_id
):
    """A Temporary Chat lives only in the browser, so there is nothing to query."""
    import open_webui.models.chats as owui_chats

    calls: list[tuple] = []

    async def _get(cid, mid, **_kw):
        calls.append((cid, mid))
        return {"content": "should not be reached"}

    monkeypatch.setattr(
        owui_chats.Chats, "get_message_by_id_and_message_id", _get, raising=False
    )

    assert await persistence.load_message(chat_id=chat_id, message_id=message_id) is None
    assert not calls, f"a lookup was issued for {chat_id!r}/{message_id!r}"


@pytest.mark.parametrize(
    ("chat_id", "expected"),
    [
        ("local:abc", True),
        ("temporary:abc", True),
        ("channel:abc", True),
        ("  local:abc  ", True),
        ("  temporary:abc  ", True),
        ("chat-1", False),
        ("", False),
        ("   ", False),
        (None, False),
    ],
)
def test_every_prefix_without_a_chat_row_is_recognised(chat_id, expected):
    """Delegates the prefix rule to is_linkable_chat so there is one list, not three.

    This predicate tested `local:` alone, which is upstream's LEGACY spelling -- so it
    missed `temporary:` (the current one) and `channel:` entirely, and a resumed video
    job in either of those tried to load a message from a chat that does not exist.
    """
    assert is_local_chat_id(chat_id) is expected
