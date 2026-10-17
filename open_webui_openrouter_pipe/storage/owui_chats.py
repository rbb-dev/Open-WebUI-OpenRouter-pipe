from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from ..core.warn_latch import warn_level

logger = logging.getLogger(__name__)

_NARROW_MESSAGE_ID = re.compile(r"^[0-9A-Za-z_-]{1,128}$")

_warned_narrow_chat_read: set[str] = set()


def _stored_from_row(chat_row: Any, message_id: Any) -> Any:
    chat = getattr(chat_row, "chat", None)
    if chat_row is None or not isinstance(chat, dict):
        return {}
    return chat.get("history", {}).get("messages", {}).get(message_id, {})


async def _narrow_message(chat_id: str, message_id: str) -> Any:
    from open_webui.internal.db import get_async_db_context
    from open_webui.models.chats import Chat, Chats
    from sqlalchemy import select

    async with get_async_db_context() as session:
        result = await session.execute(
            select(Chat.chat[("history", "messages", message_id)]).filter_by(id=chat_id)
        )
        row = result.first()
    if row is None or row[0] is None:
        return {}
    return Chats._clean_null_bytes(row[0])


async def _narrow_owner(chat_id: str) -> str | None:
    from open_webui.internal.db import get_async_db_context
    from open_webui.models.chats import Chat
    from sqlalchemy import select

    async with get_async_db_context() as session:
        result = await session.execute(select(Chat.user_id).filter_by(id=chat_id))
        row = result.first()
    if row is None or row[0] is None:
        return None
    return str(row[0])


async def read_stored_message(
    chat_id: Any, message_id: Any, whole_chat: Callable[[Any], Awaitable[Any]]
) -> Any:
    if isinstance(message_id, str) and _NARROW_MESSAGE_ID.match(message_id):
        try:
            return await _narrow_message(str(chat_id), message_id)
        except Exception as exc:
            level = warn_level(_warned_narrow_chat_read, f"message:{type(exc).__name__}")
            logger.log(
                level,
                "The narrow read of one stored message from the chat table failed (%s); the "
                "whole chat is read instead, which is slower on a large chat but returns the "
                "same answer",
                type(exc).__name__,
                exc_info=True,
            )
    return _stored_from_row(await whole_chat(chat_id), message_id)


async def read_chat_owner_id(
    chat_id: Any, whole_chat: Callable[[Any], Awaitable[Any]]
) -> str | None:
    try:
        return await _narrow_owner(str(chat_id))
    except Exception as exc:
        level = warn_level(_warned_narrow_chat_read, f"owner:{type(exc).__name__}")
        logger.log(
            level,
            "The narrow read of a chat's owner from the chat table failed (%s); the whole "
            "chat is read instead, which is slower on a large chat but returns the same answer",
            type(exc).__name__,
            exc_info=True,
        )
    chat = await whole_chat(chat_id)
    if chat is None:
        return None
    return str(getattr(chat, "user_id", "") or "")
