from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import io
import logging
import mimetypes
import re
import shutil
import tempfile
import uuid
from collections import OrderedDict
from collections.abc import Iterable
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

from fastapi import BackgroundTasks, Request, UploadFile
from starlette.datastructures import Headers

from ..core.config import _INTERNAL_FILE_ID_PATTERN, _PIPE_METADATA_KEY, OWUI_CHAT_ID
from ..core.errors import FileUnavailableError, RequiredInternalFileError
from ..core.media_slot import MediaSlot, SendSpool, current_send_spool
from ..core.timing_logger import timed
from ..core.url_scheme import (
    is_absolute_url,
    is_inline_data_url,
    loggable_link,
    media_type_or_empty,
    url_path,
)
from ..core.warn_latch import bounded_warn_level, warn_level

try:
    from open_webui.models.files import Files  # type: ignore[import-not-found]
except ImportError:
    Files = None  # type: ignore
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.models.files failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    Files = None  # type: ignore

try:
    from open_webui.models.users import Users  # type: ignore[import-not-found]
except ImportError:
    Users = None  # type: ignore
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.models.users failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    Users = None  # type: ignore

try:
    from open_webui.routers.files import (
        upload_file_handler,  # type: ignore[import-not-found]
    )
except ImportError:
    upload_file_handler = None  # type: ignore
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.routers.files failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    upload_file_handler = None  # type: ignore

try:
    from open_webui.config import (  # type: ignore[import-not-found]
        STORAGE_PROVIDER as _OWUI_STORAGE_PROVIDER,
    )
    from open_webui.config import (
        UPLOAD_DIR as _OWUI_UPLOAD_DIR,
    )
except Exception:
    logging.getLogger(__name__).warning(
        "Open WebUI config unavailable; upload-dir containment is disabled and file reads will be refused",
        exc_info=True,
    )
    _OWUI_STORAGE_PROVIDER = None  # type: ignore
    _OWUI_UPLOAD_DIR = None  # type: ignore


class InlinedFile(NamedTuple):
    media: MediaSlot
    filename: str


@timed
async def get_file_by_id(file_id: str, logger: logging.Logger) -> Any | None:
    if Files is None:
        logger.debug("Cannot load file %s: Open WebUI integration not available", file_id)
        return None
    try:
        return await Files.get_file_by_id(file_id)
    except Exception:
        logger.exception("Failed to load file %s", file_id)
        return None


_warned_storage_provider: set[str] = set()


def get_owui_storage() -> Any | None:
    try:
        from open_webui.storage.provider import (
            Storage,  # type: ignore[import-not-found]
        )

        return Storage
    except Exception:
        logging.getLogger(__name__).log(
            warn_level(_warned_storage_provider, "storage_provider_unavailable"),
            "Open WebUI storage provider unavailable; file attachments will fail",
            exc_info=True,
        )
        return None


def owui_storage_provider_kind() -> str:
    raw = _OWUI_STORAGE_PROVIDER
    value = str(raw).strip().lower() if raw is not None else "local"
    if value in ("", "local"):
        return "local"
    if value == "s3":
        return "s3"
    if value in ("gcs", "google"):
        return "gcs"
    if value in ("azure", "azblob"):
        return "azure"
    return "unknown"


def declared_file_size(file_obj: Any) -> int | None:
    meta = getattr(file_obj, "meta", None)
    if not isinstance(meta, dict):
        return None
    candidates = [meta.get("size")]
    inner = meta.get("data")
    if isinstance(inner, dict):
        candidates.append(inner.get("size"))
    for value in candidates:
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
    return None


def contained_under_upload_dir(local_path: Any) -> Path | None:
    if _OWUI_UPLOAD_DIR is None:
        return None
    try:
        resolved = Path(str(local_path)).resolve(strict=True)
        upload_root = Path(str(_OWUI_UPLOAD_DIR)).resolve(strict=True)
        if resolved.is_relative_to(upload_root):
            return resolved
    except (OSError, RuntimeError, ValueError):
        return None
    return None


def copy_to_private_temp(
    contained_path: Path, *, suffix: str = "", directory: str | None = None
) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix="orpipe-read-", suffix=suffix, delete=False, dir=directory
    ) as tmp:
        temp_path = Path(tmp.name)
    try:
        shutil.copyfile(contained_path, temp_path)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise RequiredInternalFileError(
            "A referenced file could not be copied out of storage."
        ) from exc
    return temp_path


async def _reclaim_copied_temp(copy_task: asyncio.Future[Any]) -> None:
    while not copy_task.done():
        try:
            await asyncio.shield(copy_task)
        except asyncio.CancelledError:
            continue
        except Exception:  # noqa: BLE001
            break
    if copy_task.done() and not copy_task.cancelled() and copy_task.exception() is None:
        copy_task.result().unlink(missing_ok=True)


async def _linked_ok(result: Any, chat_id: str, message_id: str, file_id: str) -> bool:
    if result:
        return True
    try:
        from open_webui.models.chats import Chats  # type: ignore[import-not-found]

        linked = await Chats.get_chat_files_by_chat_id_and_message_id(chat_id, message_id)
    except Exception:
        logging.getLogger(__name__).debug(
            "Could not confirm chat-link state for file %s", file_id, exc_info=True
        )
        return False
    return any(getattr(item, "file_id", None) == file_id for item in linked or [])


_CHANNEL_CHAT_PREFIX = "channel:"
OPEN_WEBUI_MAX_UPLOAD_SIZE = "open_webui_max_upload_size"
_UNLINKABLE_CHAT_PREFIXES = ("temporary:", "local:", _CHANNEL_CHAT_PREFIX)


class _PublishedChatIdsUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _resolved_published_chat_id_values() -> tuple[tuple[str, ...], tuple[str, ...], str]:
    try:
        from open_webui.utils.chat_id import (  # pyright: ignore[reportMissingImports]
            CHANNEL_CHAT_ID_PREFIX,
            NON_SAVED_CHAT_ID_PREFIXES,
            TEMPORARY_CHAT_ID_PREFIXES,
        )
    except ImportError as exc:
        raise _PublishedChatIdsUnavailable from exc
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "open_webui.utils.chat_id failed to import for a reason other than absence; "
            "the features that depend on it are now disabled",
            exc_info=True,
        )
        raise _PublishedChatIdsUnavailable from exc
    return (NON_SAVED_CHAT_ID_PREFIXES, TEMPORARY_CHAT_ID_PREFIXES, CHANNEL_CHAT_ID_PREFIX)


def _published_chat_id_values() -> tuple[tuple[str, ...], tuple[str, ...], str] | None:
    try:
        return _resolved_published_chat_id_values()
    except _PublishedChatIdsUnavailable:
        return None


@lru_cache(maxsize=1)
def _resolved_unlinkable_chat_prefixes() -> tuple[str, ...]:
    published = _published_chat_id_values()
    if published is None:
        raise _PublishedChatIdsUnavailable
    non_saved = published[0]
    if isinstance(non_saved, str) or not isinstance(non_saved, Iterable):
        return _UNLINKABLE_CHAT_PREFIXES
    upstream = tuple(p for p in non_saved if isinstance(p, str) and p.strip())
    return _UNLINKABLE_CHAT_PREFIXES + tuple(
        p for p in upstream if p not in _UNLINKABLE_CHAT_PREFIXES
    )


def _unlinkable_chat_prefixes() -> tuple[str, ...]:
    try:
        return _resolved_unlinkable_chat_prefixes()
    except _PublishedChatIdsUnavailable:
        return _UNLINKABLE_CHAT_PREFIXES


def is_linkable_chat(chat_id: Any) -> bool:
    if not isinstance(chat_id, str):
        return False
    normalized = chat_id.strip()
    return bool(normalized) and not normalized.startswith(_unlinkable_chat_prefixes())


def is_unheld_chat(chat_id: Any) -> bool:
    return is_temporary_chat(chat_id) or (
        isinstance(chat_id, str)
        and bool(chat_id.strip())
        and not is_linkable_chat(chat_id)
        and not is_channel_chat(chat_id)
    )


@lru_cache(maxsize=1)
def _resolved_temporary_chat_prefixes() -> tuple[str, ...]:
    published = _published_chat_id_values()
    if published is None:
        raise _PublishedChatIdsUnavailable
    _non_saved, temporary, channel = published
    channel = channel.strip() if isinstance(channel, str) and channel.strip() else _CHANNEL_CHAT_PREFIX
    usable = (
        (p for p in temporary if isinstance(p, str) and p.strip())
        if not isinstance(temporary, str) and isinstance(temporary, Iterable)
        else ()
    )
    local = tuple(
        p for p in _UNLINKABLE_CHAT_PREFIXES if p not in (_CHANNEL_CHAT_PREFIX, channel)
    )
    return local + tuple(p for p in usable if p not in local)


def temporary_chat_prefixes() -> tuple[str, ...]:
    try:
        return _resolved_temporary_chat_prefixes()
    except _PublishedChatIdsUnavailable:
        return tuple(p for p in _UNLINKABLE_CHAT_PREFIXES if p != _CHANNEL_CHAT_PREFIX)


@lru_cache(maxsize=1)
def _resolved_channel_chat_prefix() -> str:
    try:
        from open_webui.utils.chat_id import (  # pyright: ignore[reportMissingImports]
            CHANNEL_CHAT_ID_PREFIX,
        )

        published = CHANNEL_CHAT_ID_PREFIX
    except (ImportError, AttributeError) as exc:
        raise _PublishedChatIdsUnavailable from exc
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "open_webui.utils.chat_id failed to import for a reason other than absence; "
            "the features that depend on it are now disabled",
            exc_info=True,
        )
        raise _PublishedChatIdsUnavailable from exc
    if not isinstance(published, str) or not published.strip():
        return _CHANNEL_CHAT_PREFIX
    return published


def _channel_chat_prefix() -> str:
    try:
        return _resolved_channel_chat_prefix()
    except _PublishedChatIdsUnavailable:
        return _CHANNEL_CHAT_PREFIX


def is_channel_chat(chat_id: Any) -> bool:
    if not isinstance(chat_id, str):
        return False
    return chat_id.strip().startswith(_channel_chat_prefix())


def is_temporary_chat(chat_id: Any) -> bool:
    return isinstance(chat_id, str) and chat_id.strip().startswith(temporary_chat_prefixes())


_warned_chat_write_ownership: OrderedDict[str, None] = OrderedDict()


async def chat_write_allowed(chat_id: Any, user_id: Any) -> bool:
    if not is_linkable_chat(chat_id):
        return False
    principal = str(user_id or "")
    if not principal:
        return False
    from open_webui.models.chats import Chats
    from open_webui.models.users import Users

    if await Chats.is_chat_owner(chat_id, principal):
        return True
    user = await Users.get_user_by_id(principal)
    if getattr(user, "role", None) == "admin":
        return True
    get_accessible_chat = getattr(Chats, "get_accessible_chat_by_id", None)
    if get_accessible_chat is None or user is None:
        return False
    chat = await get_accessible_chat(chat_id, user, permission="write")
    return chat is not None


async def caller_may_write_chat(chat_id: Any, user_id: Any) -> bool:
    try:
        return await chat_write_allowed(chat_id, user_id)
    except Exception as exc:
        logging.getLogger(__name__).log(
            bounded_warn_level(_warned_chat_write_ownership, type(exc).__name__, 16),
            "Chat message write refused: Open WebUI's chat ownership check is unavailable "
            "(chat_id=%s user_id=%s). Refusing rather than writing under an unverified id.",
            chat_id,
            str(user_id or ""),
            exc_info=True,
        )
        return False


NO_CHAT_ID_KEY = "__no_chat_id__"


def chat_latch_key(chat_id: Any) -> str:
    if not isinstance(chat_id, str) or not chat_id:
        return NO_CHAT_ID_KEY
    if is_temporary_chat(chat_id):
        return ""
    return chat_id


def loggable_chat_id(chat_id: Any) -> str:
    return "<not retained>" if is_temporary_chat(chat_id) else str(chat_id or "")


def is_temporary_metadata(metadata: Any) -> bool:
    base = metadata if isinstance(metadata, dict) else {}
    if is_temporary_chat(base.get("chat_id")):
        return True
    pipe_meta = base.get(_PIPE_METADATA_KEY)
    return isinstance(pipe_meta, dict) and pipe_meta.get("temporary_chat") is True


def loggable_session_id(session_id: Any) -> str:
    chat_id = OWUI_CHAT_ID.get()
    if not isinstance(chat_id, str) or not chat_id.strip():
        return str(session_id or "")
    return "" if is_temporary_chat(chat_id) else str(session_id or "")


def channel_id_for_chat(chat_id: Any) -> str | None:
    if not isinstance(chat_id, str):
        return None
    normalized = chat_id.strip()
    prefix = _channel_chat_prefix()
    if normalized.startswith(prefix):
        return normalized.removeprefix(prefix) or None
    return None


def _owui_upload_metadata(mime_type: str, chat_id: Any, message_id: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"mime_type": mime_type}
    conversation_key: str | None = None
    if isinstance(chat_id, str):
        normalized_chat_id = chat_id.strip()
        channel_id = channel_id_for_chat(normalized_chat_id)
        if channel_id is not None:
            conversation_key = "channel_id"
            metadata["channel_id"] = channel_id
        elif is_linkable_chat(normalized_chat_id):
            conversation_key = "chat_id"
            metadata["chat_id"] = normalized_chat_id
    if conversation_key is not None and isinstance(message_id, str):
        normalized_message_id = message_id.strip()
        if normalized_message_id:
            metadata["message_id"] = normalized_message_id
    return metadata


def _upload_identity(user: Any, owui_user_id: str | None) -> str | None:
    candidate = getattr(user, "id", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    if isinstance(owui_user_id, str) and owui_user_id.strip():
        return owui_user_id.strip()
    return None


def is_real_owui_file_record(file_obj: Any) -> bool:
    return bool(getattr(file_obj, "id", None))


def _requester_field(user: Any, field: str) -> Any:
    return user.get(field) if isinstance(user, dict) else getattr(user, field, None)


FILE_READ_AUTH_MEMO: ContextVar[dict[tuple[str, Any], bool] | None] = ContextVar(
    "owui_file_read_auth_memo", default=None
)


@timed
async def authorize_file_read(file_obj: Any, user: Any, logger: logging.Logger) -> bool:
    if user is None:
        return False
    owner_id = getattr(file_obj, "user_id", None)
    requester_id = _requester_field(user, "id")
    if owner_id and requester_id and owner_id == requester_id:
        return True
    if _requester_field(user, "role") == "admin":
        return True
    file_id = getattr(file_obj, "id", None)
    if not file_id:
        return False
    memo = FILE_READ_AUTH_MEMO.get()
    memo_key = (str(file_id), requester_id) if requester_id else None
    if memo is not None and memo_key is not None and memo_key in memo:
        return memo[memo_key]
    try:
        from open_webui.utils.access_control.files import (  # type: ignore[import-not-found]
            has_access_to_file,
        )
    except Exception:
        logger.warning("OWUI access-control helper unavailable; denying file read", exc_info=True)
        return False
    try:
        granted = bool(await has_access_to_file(str(file_id), "read", user))
    except Exception as exc:
        logger.warning("has_access_to_file failed for %s: %s", file_id, exc, exc_info=True)
        granted = False
    if memo is not None and memo_key is not None:
        memo[memo_key] = granted
    return granted


PUBLISHING_NEEDS_OWNERSHIP = (
    "A file has to be your own before it can be uploaded to a public file host, even "
    "when Open WebUI lets you open it here."
)


def authorize_file_publication(file_obj: Any, user: Any) -> bool:
    if user is None:
        return False
    role = _requester_field(user, "role")
    if role == "admin":
        return True
    owner_id = getattr(file_obj, "user_id", None)
    requester_id = _requester_field(user, "id")
    return bool(
        isinstance(owner_id, str)
        and owner_id
        and isinstance(requester_id, str)
        and owner_id == requester_id
    )


@timed
async def materialize_owui_file_to_temp(
    file_obj: Any,
    *,
    user: Any,
    logger: logging.Logger,
    max_bytes: int,
    require_auth: bool = True,
    allowed_suffixes: set[str] | None = None,
    suffix: str = "",
    directory: str | None = None,
) -> Path:
    if (
        require_auth
        and is_real_owui_file_record(file_obj)
        and not await authorize_file_read(file_obj, user, logger)
    ):
        raise RequiredInternalFileError(
            "You do not have access to a referenced file.", denied=True
        )

    raw_path = getattr(file_obj, "path", None)
    if not raw_path:
        raise RequiredInternalFileError("A referenced file is missing its storage path.")

    declared = declared_file_size(file_obj)
    if declared is not None and max_bytes > 0 and declared > max_bytes:
        raise RequiredInternalFileError(
            "A referenced file exceeds the configured size limit.", kind="size"
        )
    storage = get_owui_storage()
    if storage is None:
        raise RequiredInternalFileError("Open WebUI storage is not available.")
    try:
        local_path = await asyncio.to_thread(storage.get_file, str(raw_path))
    except Exception as exc:
        from ..media.frame_extraction import _without_paths

        logger.warning(
            "Storage.get_file failed for %s: %s",
            _without_paths(str(raw_path)), _without_paths(str(exc)),
        )
        raise RequiredInternalFileError(
            "A referenced file could not be retrieved from storage."
        ) from exc

    contained = contained_under_upload_dir(local_path)
    if contained is None:
        raise RequiredInternalFileError(
            "A referenced file resolved outside the allowed storage area."
        )
    if allowed_suffixes is not None and contained.suffix.lower() not in allowed_suffixes:
        raise RequiredInternalFileError("A referenced file has an unsupported type.")

    copy_task = asyncio.ensure_future(
        asyncio.to_thread(
            copy_to_private_temp, contained, suffix=suffix or contained.suffix, directory=directory
        )
    )
    try:
        temp_path = await asyncio.shield(copy_task)
    except BaseException:
        await _reclaim_copied_temp(copy_task)
        raise
    try:
        if max_bytes > 0 and temp_path.stat().st_size > max_bytes:
            raise RequiredInternalFileError(
                "A referenced file exceeds the configured size limit.", kind="size"
            )
    except RequiredInternalFileError:
        temp_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        logger.warning("Failed to stat temp copy for %s: %s", raw_path, exc)
        raise RequiredInternalFileError("A referenced file could not be validated.") from exc
    return temp_path


def infer_file_mime_type(file_obj: Any) -> str:
    candidates = [
        getattr(file_obj, "mime_type", None),
        getattr(file_obj, "content_type", None),
    ]
    meta = getattr(file_obj, "meta", None) or {}
    if isinstance(meta, dict):
        candidates.extend(
            [
                meta.get("content_type"),
                meta.get("mimeType"),
                meta.get("mime_type"),
            ]
        )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            normalized = candidate.strip().lower()
            if normalized == "image/jpg":
                return "image/jpeg"
            return normalized
    for name in (
        meta.get("name") if isinstance(meta, dict) else None,
        getattr(file_obj, "filename", None),
    ):
        if isinstance(name, str) and name.strip():
            guessed = (mimetypes.guess_type(name.strip())[0] or "").strip().lower()
            if guessed:
                return "image/jpeg" if guessed == "image/jpg" else guessed
    return "application/octet-stream"


def extract_internal_file_id(url: str) -> str | None:
    if not isinstance(url, str):
        return None
    match = _INTERNAL_FILE_ID_PATTERN.search(url)
    if match:
        return match.group(1)
    return None


def is_internal_file_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    if is_absolute_url(url):
        return False
    return "/api/v1/files/" in url


def owui_file_content_url(file_id: str) -> str:
    return f"{_INTERNAL_FILE_SUBSTRING}{file_id}/content"


_INTERNAL_FILE_PATH_RE = re.compile(r"/api/v1/files/[^/?#\s]*", re.IGNORECASE)
_INTERNAL_FILE_SUBSTRING = "/api/v1/files/"


def names_an_owui_file_path(url: Any) -> bool:
    if not isinstance(url, str) or is_inline_data_url(url):
        return False
    if _INTERNAL_FILE_PATH_RE.search(url_path(url).casefold()) is not None:
        return True
    return not is_absolute_url(url) and _INTERNAL_FILE_SUBSTRING in url


def _picture_reference(block: Any) -> str | None:
    if not isinstance(block, dict):
        return None
    holder = block.get("image_url")
    url = holder.get("url") if isinstance(holder, dict) else holder
    if isinstance(url, str) and url.strip():
        return url.strip()
    return None


def names_a_hostless_owui_file_path(url: Any) -> bool:
    if not isinstance(url, str) or is_inline_data_url(url):
        return False
    if is_absolute_url(url):
        return False
    return _INTERNAL_FILE_PATH_RE.match(url_path(url).casefold()) is not None


_warned_reference_sizes: set[str] = set()


def _own_internal_reference(file_data: Any, file_url: Any) -> str | None:
    if isinstance(file_data, str) and file_data.strip() and names_an_owui_file_path(file_data.strip()):
        return file_data.strip()
    if isinstance(file_url, str) and file_url.strip() and names_an_owui_file_path(file_url.strip()):
        return file_url.strip()
    return None


def _referenced_file_ids(items: Any) -> dict[str, str]:
    found: dict[str, str] = {}
    if not isinstance(items, list):
        return found
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "input_file":
                continue
            raw_id = block.get("file_id")
            if (
                isinstance(raw_id, str)
                and raw_id.strip()
                and not raw_id.strip().startswith("file-")
            ):
                found[raw_id] = raw_id.strip()
                continue
            for key in ("file_data", "file_url"):
                raw = block.get(key)
                if not isinstance(raw, str) or not raw.strip():
                    continue
                candidate = raw.strip()
                if not names_an_owui_file_path(candidate):
                    continue
                extracted = extract_internal_file_id(candidate)
                if extracted:
                    found[raw] = extracted
                break
    return found


def _names_an_internal_reference(input_items: Any) -> bool:
    if _referenced_file_ids(input_items):
        return True
    for item in input_items if isinstance(input_items, list) else ():
        if not isinstance(item, dict):
            continue
        if item.get("type") in ("input_image", "image_url"):
            url = _picture_reference(item)
            if url is not None and names_an_owui_file_path(url):
                return True
            continue
        parts = item.get("output") if item.get("type") == "function_call_output" else item.get("content")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("input_image", "image_url"):
                url = _picture_reference(part)
                if url is not None and names_an_owui_file_path(url):
                    return True
                continue
            if part.get("type") != "input_file":
                continue
            raw_id = part.get("file_id")
            if (
                isinstance(raw_id, str)
                and raw_id.strip()
                and not raw_id.strip().startswith("file-")
            ):
                return True
            for key in ("file_data", "file_url"):
                raw = part.get(key)
                if isinstance(raw, str) and raw.strip() and names_an_owui_file_path(raw.strip()):
                    return True
    return False


async def _file_records_by_id(
    ids: Iterable[str], logger: logging.Logger
) -> dict[str, Any]:
    ordered = sorted({file_id for file_id in ids if file_id})
    if Files is None or not ordered:
        return {}
    records: dict[str, Any] = {}
    bulk = getattr(Files, "get_file_metadatas_by_ids", None)
    rows: Any = None
    if callable(bulk):
        try:
            rows = bulk(list(ordered))
            if inspect.isawaitable(rows):
                rows = await rows
        except Exception:
            logger.log(
                warn_level(_warned_reference_sizes, "bulk-metadata"),
                "Bulk file-metadata lookup failed; falling back to per-id reads",
                exc_info=True,
            )
            rows = None
        for row in rows or []:
            row_id = getattr(row, "id", None)
            if isinstance(row_id, str) and row_id in ordered:
                records[row_id] = row
    for file_id in ordered:
        if file_id in records or rows is not None:
            continue
        record = await get_file_by_id(file_id, logger)
        if record is not None:
            records[file_id] = record
    return records


async def index_referenced_file_payloads(
    items: Any,
    logger: logging.Logger,
    *,
    user: Any,
    known: dict[str, tuple[int, str, str] | None] | None = None,
) -> dict[str, tuple[int, str, str]]:
    try:
        references = _referenced_file_ids(items)
        if not references:
            return {}
        memo = known if isinstance(known, dict) else {}
        unresolved = {
            reference: file_id
            for reference, file_id in references.items()
            if reference not in memo
        }
        records = await _file_records_by_id(unresolved.values(), logger) if unresolved else {}
        index: dict[str, tuple[int, str, str]] = {}
        for reference, file_id in references.items():
            if reference in memo:
                entry = memo[reference]
                if entry is not None:
                    index[reference] = entry
                continue
            record = records.get(file_id)
            entry = None
            if record is not None and not await authorize_file_read(record, user, logger):
                logger.log(
                    warn_level(_warned_reference_sizes, "unauthorized-record"),
                    "Referenced Open WebUI file %s is not readable by the requester; the "
                    "context budget charges it nothing",
                    file_id,
                )
            else:
                size = declared_file_size(record) if record is not None else None
                if size is None:
                    logger.log(
                        warn_level(_warned_reference_sizes, "unsized-record"),
                        "Referenced Open WebUI file %s has no readable declared size; the "
                        "context budget charges it nothing",
                        file_id,
                    )
                else:
                    meta = getattr(record, "meta", None)
                    name = meta.get("name") if isinstance(meta, dict) else None
                    entry = (
                        size,
                        infer_file_mime_type(record),
                        name if isinstance(name, str) else "",
                    )
                    index[reference] = entry
            memo[reference] = entry
        return index
    except Exception:
        logger.log(
            warn_level(_warned_reference_sizes, "index"),
            "Failed to index referenced Open WebUI file payloads; the context budget "
            "charges them nothing",
            exc_info=True,
        )
        return {}


def _slot_from_file(path: str | Path) -> MediaSlot | None:
    slot = MediaSlot(mime="", kind="bare", path=path)
    return slot if slot.raw_len else None


class OwuiFileGateway:
    def __init__(self, logger, valves, valves_owner=None):
        self.logger = logger
        self._valves = valves
        self._valves_owner = valves_owner
        self._storage_user_cache = None
        self._storage_user_cache_email: str | None = None
        self._storage_user_lock = None
        self._storage_user_lock_loop: asyncio.AbstractEventLoop | None = None
        self._storage_role_warning_emitted = False

    @property
    def valves(self):
        live = getattr(self._valves_owner, "valves", None)
        return self._valves if live is None else live

    @timed
    async def read_file_record_media(
        self,
        file_obj: Any,
        *,
        user: Any = None,
        require_auth: bool = True,
    ) -> MediaSlot | None:
        real = is_real_owui_file_record(file_obj)
        if require_auth and real and not await authorize_file_read(file_obj, user, self.logger):
            raise RequiredInternalFileError(
                "You do not have access to a referenced file.", denied=True
            )

        spool = current_send_spool()
        requester = _requester_field(user, "id") if real else None
        if not requester:
            return await self._media_of_record(file_obj, user, real, spool)
        key = (str(file_obj.id), str(requester))
        async with spool.copy_lock(key):
            held = spool.held_copy(key)
            if held is not None:
                with contextlib.suppress(OSError):
                    kept = _slot_from_file(held)
                    if kept is not None:
                        return kept
            media = await self._media_of_record(file_obj, user, real, spool)
            if media is not None and media.path is not None:
                spool.hold_copy(key, media.path)
            return media

    async def _media_of_record(
        self, file_obj: Any, user: Any, real: bool, spool: SendSpool
    ) -> MediaSlot | None:
        async def _slot_from_bytes(raw: bytes) -> MediaSlot | None:
            if not raw:
                return None
            target = spool.new_path()
            await asyncio.to_thread(target.write_bytes, raw)
            return _slot_from_file(target)

        if real:
            if getattr(file_obj, "path", None):
                temp_path = await materialize_owui_file_to_temp(
                    file_obj,
                    user=user,
                    logger=self.logger,
                    max_bytes=0,
                    require_auth=False,
                    directory=spool.directory,
                )
                return _slot_from_file(temp_path)
        else:
            prefer_paths = [
                getattr(file_obj, attr, None)
                for attr in ("path", "file_path", "absolute_path")
            ]
            for candidate in prefer_paths:
                if not isinstance(candidate, str):
                    continue
                path = Path(candidate)
                if not path.exists():
                    continue
                return _slot_from_file(await asyncio.to_thread(spool.adopt, path))

        data_field = getattr(file_obj, "data", None)
        if isinstance(data_field, dict):
            for key in ("b64", "base64", "data"):
                inline_value = data_field.get(key)
                if isinstance(inline_value, str) and inline_value.strip():
                    try:
                        return MediaSlot(mime="", kind="bare", text=inline_value.strip())
                    except ValueError:
                        self.logger.warning(
                            "A referenced file's stored %r text is not base64, so the file was not read",
                            key,
                        )
                        return None
            blob_value = data_field.get("bytes")
            if isinstance(blob_value, (bytes, bytearray)):
                return await _slot_from_bytes(bytes(blob_value))
            content_value = data_field.get("content")
            if (
                isinstance(content_value, str)
                and content_value
                and not getattr(file_obj, "path", None)
            ):
                return await _slot_from_bytes(content_value.encode("utf-8"))

        for attr in ("content", "blob", "data"):
            value = getattr(file_obj, attr, None)
            if isinstance(value, (bytes, bytearray)):
                return await _slot_from_bytes(bytes(value))
        return None

    @timed
    async def inline_owui_file_id(
        self,
        file_id: str,
        *,
        user: Any = None,
    ) -> InlinedFile | None:
        normalized = (file_id or "").strip()
        if not normalized:
            return None
        file_obj = await get_file_by_id(normalized, self.logger)
        if not file_obj:
            return None
        mime_type = infer_file_mime_type(file_obj)
        media = await self.read_file_record_media(file_obj, user=user)
        if media is None:
            return None
        media_type = media_type_or_empty(mime_type)
        if not media_type:
            raise RequiredInternalFileError(
                "A referenced file's recorded type is not a media type.", kind="file"
            )
        meta = getattr(file_obj, "meta", None)
        filename = ""
        if isinstance(meta, dict):
            filename = meta.get("name", "") or ""
        if not filename:
            filename = getattr(file_obj, "filename", "") or ""
        return InlinedFile(media=media.shaped(mime=media_type, kind="data_url"), filename=filename)

    @timed
    async def inline_internal_file_url(
        self,
        url: str,
        *,
        user: Any = None,
    ) -> InlinedFile | None:
        file_id = extract_internal_file_id(url)
        if not file_id:
            return None
        return await self.inline_owui_file_id(file_id, user=user)

    @timed
    async def inline_internal_responses_input_files(
        self,
        request_body: dict[str, Any],
        *,
        user: Any = None,
        refusals: list[tuple[str, str, str]] | None = None,
    ) -> dict[str, Any]:
        input_items = request_body.get("input")
        if not isinstance(input_items, list) or not input_items:
            return request_body

        async def _inline_picture(part: dict[str, Any]) -> bool:
            holder = part.get("image_url")
            image_url = _picture_reference(part)
            if image_url is None or not names_an_owui_file_path(image_url):
                return True
            picture_id = extract_internal_file_id(image_url)
            if picture_id is None:
                cause = "unreadable_internal_path"
                reason = (
                    f"a picture naming {loggable_link(image_url)} cannot be read from "
                    "Open WebUI storage"
                )
                self.logger.warning(
                    "Not forwarding a tool's picture (%s): %s [cause=%s]",
                    loggable_link(image_url), reason, cause,
                )
                if refusals is not None:
                    refusals.append((image_url, reason, cause))
                return False
            inlined = await self.inline_owui_file_id(picture_id, user=user)
            if not inlined:
                raise FileUnavailableError(
                    "A referenced picture is no longer available in Open WebUI storage.",
                    kind="image",
                )
            if isinstance(holder, dict):
                holder["url"] = inlined.media
            else:
                part["image_url"] = inlined.media
            return True

        async def _inline_file_block(block: dict[str, Any]) -> bool:
            block_type = block.get("type")
            if block_type in ("input_image", "image_url"):
                return await _inline_picture(block)
            if block_type != "input_file":
                return True

            file_url = block.get("file_url")
            file_data = block.get("file_data")
            file_id = block.get("file_id")

            internal_file_id: str | None = None
            named_reference: str | None = None

            if isinstance(file_id, str) and file_id.strip() and not file_id.strip().startswith("file-"):
                internal_file_id = file_id.strip()
            else:
                own_reference = _own_internal_reference(file_data, file_url)
                if own_reference is not None:
                    named_reference = own_reference
                    internal_file_id = extract_internal_file_id(own_reference)

            if not internal_file_id:
                if named_reference is not None:
                    raise FileUnavailableError(
                        "A referenced file is no longer available in Open WebUI storage.",
                    )
                return True

            result = await self.inline_owui_file_id(internal_file_id, user=user)
            if not result:
                raise FileUnavailableError(
                    "A referenced file is no longer available in Open WebUI storage.",
                )

            block["file_data"] = result.media
            if result.filename and "filename" not in block:
                block["filename"] = result.filename
            block.pop("file_id", None)
            if (
                isinstance(file_url, str)
                and file_url.strip()
                and names_an_owui_file_path(file_url.strip())
            ):
                block.pop("file_url", None)
            return True

        if not _names_an_internal_reference(input_items):
            return {**request_body, "input": list(input_items)}
        working = copy.deepcopy(input_items)
        kept_items: list[Any] = []
        for item in working:
            if not isinstance(item, dict):
                kept_items.append(item)
                continue
            if item.get("type") in ("input_image", "image_url"):
                if await _inline_file_block(item):
                    kept_items.append(item)
                continue
            parts = item.get("output") if item.get("type") == "function_call_output" else item.get("content")
            if not isinstance(parts, list) or not parts:
                kept_items.append(item)
                continue
            dropped: list[int] = []
            for position, block in enumerate(parts):
                if not isinstance(block, dict):
                    continue
                if not await _inline_file_block(block):
                    dropped.append(position)
            for position in reversed(dropped):
                del parts[position]
            kept_items.append(item)

        return {**request_body, "input": kept_items}

    @timed
    async def upload_to_owui_storage(
        self,
        request: Request,
        user,
        file_data: bytes,
        filename: str,
        mime_type: str,
        chat_id: str | None = None,
        message_id: str | None = None,
        owui_user_id: str | None = None,
        refusals: list[str] | None = None,
    ) -> str | None:
        if upload_file_handler is None:
            self.logger.error("Open WebUI file upload helpers are unavailable; skipping OWUI storage upload.")
            return None
        try:
            upload_metadata = _owui_upload_metadata(mime_type, chat_id, message_id)

            file_item = await upload_file_handler(
                request=request,
                file=UploadFile(
                    file=io.BytesIO(file_data),
                    filename=filename,
                    headers=Headers({"content-type": mime_type}),
                ),
                metadata=upload_metadata,
                process=False,
                process_in_background=False,
                user=user,
                background_tasks=BackgroundTasks(),
            )
            file_id: str | None = None
            if hasattr(file_item, "id"):
                candidate = getattr(file_item, "id", None)
                if isinstance(candidate, str) and candidate.strip():
                    file_id = candidate.strip()
            elif isinstance(file_item, dict):
                raw_id = file_item.get("id")
                if isinstance(raw_id, str) and raw_id.strip():
                    file_id = raw_id.strip()
            if not file_id:
                self.logger.error("Upload handler returned an object without an id; aborting OWUI storage write.")
                return None

            effective_user_id: str | None = _upload_identity(user, owui_user_id)

            try:
                linked = await self.try_link_file_to_chat(
                    chat_id=chat_id,
                    message_id=message_id,
                    file_id=file_id,
                    user_id=effective_user_id,
                )
            except Exception:
                linked = False
                self.logger.debug(
                    "Chat-link raised for file %s; the upload itself succeeded", file_id, exc_info=True
                )
            if not linked and is_linkable_chat(chat_id):
                self.logger.debug(
                    "File %s was not linked under this message in chat %s; Open WebUI "
                    "enforces one chat_file row per (chat_id, file_id) and reports nothing",
                    file_id,
                    chat_id,
                )

            self.logger.info(
                f"Uploaded {filename} ({len(file_data):,} bytes) to OWUI storage: /api/v1/files/{file_id}"
            )
            return file_id
        except Exception as exc:
            if self._refused_for_size(exc, filename, refusals):
                return None
            self.logger.exception("Failed to upload %s to OWUI storage", filename)
            return None

    async def upload_to_owui_storage_from_path(
        self,
        request: Request,
        user,
        source_path: Path,
        filename: str,
        mime_type: str,
        chat_id: str | None = None,
        message_id: str | None = None,
        owui_user_id: str | None = None,
        refusals: list[str] | None = None,
    ) -> str | None:
        if upload_file_handler is None:
            self.logger.error("Open WebUI file upload helpers are unavailable; skipping OWUI storage upload.")
            return None
        if not source_path.is_file():
            self.logger.error("Source path %s is not a file; aborting OWUI streaming upload.", source_path)
            return None
        try:
            upload_metadata = _owui_upload_metadata(mime_type, chat_id, message_id)

            size_bytes = source_path.stat().st_size

            with source_path.open("rb") as fh:
                file_item = await upload_file_handler(
                    request=request,
                    file=UploadFile(
                        file=fh,
                        filename=filename,
                        headers=Headers({"content-type": mime_type}),
                    ),
                    metadata=upload_metadata,
                    process=False,
                    process_in_background=False,
                    user=user,
                    background_tasks=BackgroundTasks(),
                )

            file_id: str | None = None
            if hasattr(file_item, "id"):
                candidate = getattr(file_item, "id", None)
                if isinstance(candidate, str) and candidate.strip():
                    file_id = candidate.strip()
            elif isinstance(file_item, dict):
                raw_id = file_item.get("id")
                if isinstance(raw_id, str) and raw_id.strip():
                    file_id = raw_id.strip()
            if not file_id:
                self.logger.error("Streaming upload handler returned an object without an id; aborting.")
                return None

            effective_user_id: str | None = _upload_identity(user, owui_user_id)

            try:
                linked = await self.try_link_file_to_chat(
                    chat_id=chat_id,
                    message_id=message_id,
                    file_id=file_id,
                    user_id=effective_user_id,
                )
            except Exception:
                linked = False
                self.logger.debug(
                    "Chat-link raised for file %s; the upload itself succeeded", file_id, exc_info=True
                )
            if not linked and is_linkable_chat(chat_id):
                self.logger.debug(
                    "File %s was not linked under this message in chat %s; Open WebUI "
                    "enforces one chat_file row per (chat_id, file_id) and reports nothing",
                    file_id,
                    chat_id,
                )

            self.logger.info(
                f"Streaming-uploaded {filename} ({size_bytes:,} bytes) to OWUI storage: /api/v1/files/{file_id}"
            )
            return file_id
        except Exception as exc:
            if self._refused_for_size(exc, filename, refusals):
                return None
            self.logger.exception("Failed to streaming-upload %s to OWUI storage", source_path)
            return None

    def _refused_for_size(self, exc: Exception, filename: str, refusals: list[str] | None) -> bool:
        if getattr(exc, "status_code", None) != 413:
            return False
        self.logger.warning(
            "Open WebUI's Max Upload Size refused %s, so it was not stored", filename
        )
        if refusals is not None:
            refusals.append(OPEN_WEBUI_MAX_UPLOAD_SIZE)
        return True

    @timed
    async def try_link_file_to_chat(
        self,
        *,
        chat_id: str | None,
        message_id: str | None,
        file_id: str,
        user_id: str | None,
    ) -> bool:
        if not isinstance(chat_id, str) or not is_linkable_chat(chat_id):
            return False
        normalized_chat_id = chat_id.strip()
        if not isinstance(file_id, str) or not file_id.strip():
            return False
        if not isinstance(user_id, str):
            return False
        normalized_user_id = user_id.strip()
        if not normalized_user_id:
            return False

        if not isinstance(message_id, str) or not message_id.strip():
            return False
        normalized_message_id = message_id.strip()

        try:
            from open_webui.models.chats import Chats  # type: ignore[import-not-found]
        except Exception:
            self.logger.debug("Chat-link skipped: open_webui.models.chats unavailable", exc_info=True)
            return False

        if not hasattr(Chats, "insert_chat_files"):
            return False

        target = file_id.strip()
        try:
            return await _linked_ok(
                await Chats.insert_chat_files(
                    chat_id=normalized_chat_id,
                    message_id=normalized_message_id,
                    file_ids=[target],
                    user_id=normalized_user_id,
                ),
                normalized_chat_id,
                normalized_message_id,
                target,
            )
        except TypeError:
            try:
                return await _linked_ok(
                    await Chats.insert_chat_files(
                        normalized_chat_id,
                        normalized_message_id,
                        [target],
                        normalized_user_id,
                    ),
                    normalized_chat_id,
                    normalized_message_id,
                    target,
                )
            except Exception:
                self.logger.debug(
                    "Chat-link failed for file %s (positional call)", file_id, exc_info=True
                )
                return False
        except Exception:
            self.logger.debug("Chat-link failed for file %s", file_id, exc_info=True)
            return False

    async def resolve_storage_context(
        self,
        request: Request | None,
        user_obj: Any | None,
    ) -> tuple[Request | None, Any | None]:
        if request is None:
            if user_obj:
                self.logger.debug("Storage upload skipped: request context missing.")
            return None, None
        if user_obj is not None and not isinstance(user_obj, dict):
            return request, user_obj
        if isinstance(user_obj, dict):
            self.logger.debug("Storage upload: the turn's user is a mapping, not a user object; the storage account answers instead.")
        fallback_user = await self.ensure_storage_user()
        if fallback_user is None:
            return None, None
        self.logger.debug("Using fallback storage user '%s' for upload.", fallback_user.email)
        return request, fallback_user

    def _storage_user_transport_lock(self) -> asyncio.Lock:
        running = asyncio.get_running_loop()
        if self._storage_user_lock is None or self._storage_user_lock_loop is not running:
            self._storage_user_lock = asyncio.Lock()
            self._storage_user_lock_loop = running
        return self._storage_user_lock

    @timed
    async def ensure_storage_user(self) -> Any | None:
        if Users is None:
            self.logger.debug("Cannot create storage user: Open WebUI integration not available")
            return None

        fallback_email = self.valves.FALLBACK_STORAGE_EMAIL or "openrouter-pipe@system.local"
        if (
            self._storage_user_cache is not None
            and self._storage_user_cache_email == fallback_email
        ):
            return self._storage_user_cache

        async with self._storage_user_transport_lock():
            if (
                self._storage_user_cache is not None
                and self._storage_user_cache_email == fallback_email
            ):
                return self._storage_user_cache

            fallback_name = self.valves.FALLBACK_STORAGE_NAME or "OpenRouter Pipe Storage"
            fallback_role = self.valves.FALLBACK_STORAGE_ROLE or "pending"

            if (
                fallback_role.lower() in {"admin", "system", "owner"}
                and not self._storage_role_warning_emitted
            ):
                self.logger.warning(
                    "Fallback storage role '%s' is highly privileged. Configure FALLBACK_STORAGE_ROLE to a least-privilege service role if possible.",
                    fallback_role,
                )
                self._storage_role_warning_emitted = True

            try:
                fallback_user = await Users.get_user_by_email(
                    fallback_email,
                )
            except Exception:  # pragma: no cover
                self.logger.exception("Failed to load fallback storage user")
                return None

            if fallback_user is None:
                user_id = f"openrouter-pipe-{uuid.uuid4().hex}"
                try:
                    fallback_user = await Users.insert_new_user(
                        user_id,
                        fallback_name,
                        fallback_email,
                        "/user.png",
                        fallback_role or "pending",
                    )
                    self.logger.info(
                        "Created fallback storage user '%s' (%s) for multimodal uploads.",
                        fallback_name,
                        fallback_email,
                    )
                except Exception:  # pragma: no cover
                    self.logger.exception("Failed to create fallback storage user")
                    return None

            self._storage_user_cache = fallback_user
            self._storage_user_cache_email = fallback_email
            return fallback_user
