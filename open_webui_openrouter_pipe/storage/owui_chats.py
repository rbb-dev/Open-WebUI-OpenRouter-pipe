from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ..core.warn_latch import warn_level

logger = logging.getLogger(__name__)

_warned_narrow_chat_read: set[str] = set()


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


_ROW_ONLY_COLUMNS = ("id", "chat_id", "updated_at")


async def _narrow_message(chat_id: str, message_id: str) -> dict[str, Any] | None:
    try:
        from open_webui.models.chat_messages import ChatMessages

        read_row = ChatMessages.get_message_by_id
    except (ImportError, AttributeError):
        return None
    try:
        row = await read_row(f"{chat_id}-{message_id}")
    except Exception as exc:
        level = warn_level(_warned_narrow_chat_read, f"message:{type(exc).__name__}")
        logger.log(
            level,
            "The single-row read of one chat message failed (%s); the chat's messages are "
            "read instead, which is slower on a large chat but returns the same answer",
            type(exc).__name__,
            exc_info=True,
        )
        return None
    if row is None or getattr(row, "chat_id", None) != chat_id:
        return None
    renames = getattr(ChatMessages, "DB_TO_JSON_KEY_MAP", None) or {}
    return {
        renames.get(column, column): value
        for column, value in row.model_dump().items()
        if value is not None and column not in _ROW_ONLY_COLUMNS
    }


async def read_chat_message(
    chat_id: Any,
    message_id: Any,
    whole_chat_getter: Callable[[str, str], Awaitable[Any]],
) -> Any:
    narrow = await _narrow_message(str(chat_id), str(message_id))
    if narrow is not None:
        return narrow
    return await whole_chat_getter(str(chat_id), str(message_id))


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
