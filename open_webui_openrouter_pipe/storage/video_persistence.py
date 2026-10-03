from __future__ import annotations

from typing import Any, cast

from ..core.utils import _await_if_needed, _iter_kind_marker_spans
from .owui_files import channel_id_for_chat, is_channel_chat, is_temporary_chat


def is_local_chat_id(chat_id: str | None) -> bool:
    """Kept as a name; the decision belongs to owui_files.is_linkable_chat.

    This tested `local:` alone, so it missed both `channel:` and the `temporary:`
    prefix that replaced `local:` upstream -- a third parallel spelling of a rule that
    now has one owner.
    """
    from .owui_files import is_linkable_chat

    return isinstance(chat_id, str) and bool(chat_id.strip()) and not is_linkable_chat(chat_id)


class VideoPersistence:

    def __init__(self, *, logger: Any) -> None:
        self.logger = logger

    async def load_message_content(self, *, chat_id: str, message_id: str, user: Any = None) -> str:
        message = await self.load_message(chat_id=chat_id, message_id=message_id, user=user)
        if isinstance(message, dict):
            value = message.get("content")
            return value if isinstance(value, str) else ""
        value = getattr(message, "content", "")
        return value if isinstance(value, str) else ""

    async def load_message(self, *, chat_id: str, message_id: str, user: Any = None) -> Any | None:
        if not chat_id or not message_id:
            return None
        if is_temporary_chat(chat_id):
            return None
        if is_channel_chat(chat_id):
            return await self._load_channel_message(chat_id, message_id, user)
        if is_local_chat_id(chat_id):
            return None
        try:
            from open_webui.models.chats import Chats  # type: ignore[import-not-found]
        except Exception:
            self.logger.debug("Open WebUI chats model unavailable", exc_info=True)
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
                self.logger.debug(
                    "Could not load message %s of chat %s", message_id, chat_id, exc_info=True
                )
                return None
        except Exception:
            self.logger.debug(
                "Could not load message %s of chat %s", message_id, chat_id, exc_info=True
            )
            return None

    def _channel_message_matches(self, chat_id: str, message: Any) -> bool:
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not _iter_kind_marker_spans(content, kind="videojob"):
            return False
        channel_id = channel_id_for_chat(chat_id) or ""
        return str(getattr(message, "channel_id", "") or "") == channel_id

    async def _channel_reader_may_read(self, chat_id: str, user: Any) -> bool:
        user_id = getattr(user, "id", None)
        if not isinstance(user_id, str) or not user_id:
            return False
        channel_id = channel_id_for_chat(chat_id)
        if not channel_id:
            return False
        try:
            from open_webui.models.access_grants import AccessGrants
            from open_webui.models.channels import Channels
        except Exception:
            self.logger.debug("Open WebUI channels model unavailable", exc_info=True)
            return False
        try:
            channel = await _await_if_needed(Channels.get_channel_by_id(channel_id))
        except Exception:
            self.logger.debug("channel row unreadable", exc_info=True)
            return False
        if channel is None:
            return False
        if getattr(channel, "type", "") in ("group", "dm"):
            try:
                return bool(
                    await _await_if_needed(Channels.is_user_channel_member(channel.id, user_id))
                )
            except Exception:
                self.logger.debug("channel membership unreadable", exc_info=True)
                return False
        if getattr(user, "role", None) == "admin":
            return True
        try:
            return bool(await _await_if_needed(AccessGrants.has_access(
                user_id=user_id, resource_type="channel", resource_id=channel.id,
                permission="read")))
        except Exception:
            self.logger.debug("channel grant unreadable", exc_info=True)
            return False

    async def _load_channel_message(self, chat_id: str, message_id: str, user: Any = None) -> Any | None:
        if not await self._channel_reader_may_read(chat_id, user):
            return None
        try:
            from open_webui.models.messages import Messages  # type: ignore[import-not-found] # noqa: I001
        except Exception:
            self.logger.debug("Open WebUI messages model unavailable", exc_info=True)
            return None
        getter = cast(Any, getattr(Messages, "get_message_by_id", None))
        if not callable(getter):
            return None
        try:
            message = await _await_if_needed(getter(message_id))
        except Exception:
            self.logger.debug(
                "Could not load channel message %s of %s", message_id, chat_id, exc_info=True
            )
            return None
        if message is None:
            return None
        if not self._channel_message_matches(chat_id, message):
            return None
        return message
