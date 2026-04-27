from __future__ import annotations

from typing import Any, cast

from ..core.utils import _await_if_needed


def is_local_chat_id(chat_id: str | None) -> bool:
    return isinstance(chat_id, str) and chat_id.strip().startswith("local:")


class VideoPersistence:

    def __init__(self, *, logger: Any) -> None:
        self.logger = logger

    async def load_message_content(self, *, chat_id: str, message_id: str) -> str:
        message = await self.load_message(chat_id=chat_id, message_id=message_id)
        if isinstance(message, dict):
            value = message.get("content")
            return value if isinstance(value, str) else ""
        value = getattr(message, "content", "")
        return value if isinstance(value, str) else ""

    async def load_message(self, *, chat_id: str, message_id: str) -> Any | None:
        if not chat_id or not message_id or is_local_chat_id(chat_id):
            return None
        try:
            from open_webui.models.chats import Chats  # type: ignore[import-not-found]
        except Exception:
            return None
        getter = cast(Any, getattr(Chats, "get_message_by_id_and_message_id", None))
        if not callable(getter):
            return None
        try:
            return await _await_if_needed(getter(chat_id, message_id))
        except TypeError:
            try:
                return await _await_if_needed(getter(chat_id=chat_id, message_id=message_id))
            except Exception:
                return None
        except Exception:
            return None
