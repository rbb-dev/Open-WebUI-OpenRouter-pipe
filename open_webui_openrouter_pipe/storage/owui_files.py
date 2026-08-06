"""Authorized, backend-agnostic Open WebUI file storage access.

Single chokepoint for reading Open WebUI files. Every read is routed through
OWUI's ``Storage`` provider so it works on any backend (local/s3/gcs/azure),
authorized against the requester (owner / admin / ``has_access_to_file``),
size-bounded, and copied to a request-owned temp before the caller touches
bytes. ``FileModel.path`` is treated as an opaque storage key; the local
provider's ``get_file`` is a no-op (no remote download), while cloud
providers download into ``UPLOAD_DIR`` first.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import logging
import shutil
import tempfile
import uuid
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

from fastapi import BackgroundTasks, Request, UploadFile
from starlette.datastructures import Headers

from ..core.config import _INTERNAL_FILE_ID_PATTERN
from ..core.errors import RequiredInternalFileError
from ..core.timing_logger import timed
from ..core.warn_latch import warn_level

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
    """Result of inlining an OWUI file: data URL + original filename."""

    data_url: str
    filename: str


@timed
async def get_file_by_id(file_id: str, logger: logging.Logger) -> Any | None:
    """Fetch an OWUI file record by id; log and return None on failure."""
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
    """Return OWUI's Storage provider singleton, or None outside OWUI / on init failure."""
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
    """Normalise STORAGE_PROVIDER to local/s3/gcs/azure/unknown (anything non-local is cloud-risk)."""
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
    """Return a positive integer declared size from file.meta, else None."""
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
    """Resolve a path and return it only if it lies under OWUI UPLOAD_DIR, else None (fail-closed)."""
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


def copy_to_private_temp(contained_path: Path, *, suffix: str = "") -> Path:
    """Copy a contained provider-cache file to a request-unique temp path the pipe owns."""
    with tempfile.NamedTemporaryFile(prefix="orpipe-read-", suffix=suffix, delete=False) as tmp:
        temp_path = Path(tmp.name)
    shutil.copyfile(contained_path, temp_path)
    return temp_path


async def _linked_ok(result: Any, chat_id: str, message_id: str, file_id: str) -> bool:
    """Decide whether a file is linked, given insert_chat_files' ambiguous return.

    Open WebUI returns None from four places -- empty input, every requested file
    already linked, no read access, and a caught exception around the DB write -- and
    its success path returns a list built from a non-empty id list, so it never returns
    an empty one. The return value therefore separates nothing: reading None as failure
    warns on every benign re-link, and reading it as success discards every real failure.

    Asking whether the link now exists distinguishes them. Note the already-linked
    branch is unreachable in practice -- Open WebUI compares the requested file ids
    against chat_file *row* ids, which never match -- so a re-link reaches the INSERT
    and returns None from the unique-constraint failure instead. Both land here.

    The lookup is scoped to one message, matching where the caller wants the file to
    appear; the same file linked under a different message in the chat is reported as
    not linked here, which is correct for that caller.
    """
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


_UNLINKABLE_CHAT_PREFIXES = ("temporary:", "local:", "channel:")


@lru_cache(maxsize=1)
def _unlinkable_chat_prefixes() -> tuple[str, ...]:
    """Open WebUI's own list where it publishes one, the literal above otherwise.

    Upstream owns this set (`utils.chat_id.NON_SAVED_CHAT_ID_PREFIXES`), and the copy
    here has already drifted from it twice -- `local:` -> `temporary:`, and the separate
    addition of `channel:`, which reached exactly one of the gates that needed it. The
    next prefix Open WebUI adds would be silently uncovered.

    Guarded because `utils.chat_id` does not exist in every supported Open WebUI: it is
    absent from the 0.10.x line and the manifest requires only 0.9.1, so an unguarded
    import would kill this module at import time on a deployment that works today.

    Cached because the answer cannot change inside a process and the miss is the
    expensive case -- a failed import re-walks sys.path, and this runs on every upload.
    `cache_clear()` is the reset seam.

    A published value is only adopted once it survives being useless. Taking one at face
    value fails open in two ways that both silently delete the gate: a bare string is
    iterable, so `"temporary:"` becomes the characters `t`, `e`, `m` ... and any chat id
    starting with `t` reads as unlinkable; and `""` makes `startswith` true for
    everything, so nothing is linkable at all.
    """
    try:
        from open_webui.utils.chat_id import (  # pyright: ignore[reportMissingImports]
            NON_SAVED_CHAT_ID_PREFIXES,
        )

        published = NON_SAVED_CHAT_ID_PREFIXES
    except ImportError:
        return _UNLINKABLE_CHAT_PREFIXES
    except Exception:
        logging.getLogger(__name__).warning(
            "open_webui.utils.chat_id failed to import for a reason other than absence; "
            "the features that depend on it are now disabled",
            exc_info=True,
        )
        return _UNLINKABLE_CHAT_PREFIXES
    if isinstance(published, str) or not isinstance(published, Iterable):
        return _UNLINKABLE_CHAT_PREFIXES
    upstream = tuple(p for p in published if isinstance(p, str) and p.strip())
    # UNION, not replace. The local tuple is this pipe's own record of which chat ids
    # have no `chat` row -- not an approximation of upstream's list. Upstream maintains
    # theirs for their own reasons, and `local:` is named LEGACY_TEMPORARY_CHAT_ID_PREFIX
    # there, so the day they retire it a replace would silently retire it here too and
    # every upload in a temporary chat would start reaching the chat_files INSERT.
    return _UNLINKABLE_CHAT_PREFIXES + tuple(
        p for p in upstream if p not in _UNLINKABLE_CHAT_PREFIXES
    )


def is_linkable_chat(chat_id: Any) -> bool:
    """True when a chat id can actually carry a chat_files link.

    Every prefix above names a conversation with no ``chat`` row. Temporary Chats exist
    only in the browser; a channel invocation runs the whole chat pipeline with
    ``chat_id = f"channel:{channel.id}"``. Anything that treats one of them as linkable
    reaches an INSERT whose foreign key cannot resolve -- logging a spurious "was not
    linked" on every upload where the FK is enforced, and accruing an orphan
    ``chat_file`` row per upload where it is not.

    This is the single gate: every other chat-id check in the package delegates here so
    a fourth prefix is one edit. They used to test ``local:`` individually, which is how
    ``channel:`` reached exactly one of them.
    """
    if not isinstance(chat_id, str):
        return False
    normalized = chat_id.strip()
    return bool(normalized) and not normalized.startswith(_unlinkable_chat_prefixes())


def is_real_owui_file_record(file_obj: Any) -> bool:
    """True when the record carries a real OWUI file id (so it requires authorisation)."""
    return bool(getattr(file_obj, "id", None))


@timed
async def authorize_file_read(file_obj: Any, user: Any, logger: logging.Logger) -> bool:
    """Authorise reading a real OWUI file: owner, admin, or has_access_to_file. Fail closed."""
    if user is None:
        return False
    owner_id = getattr(file_obj, "user_id", None)
    requester_id = getattr(user, "id", None)
    if owner_id and requester_id and owner_id == requester_id:
        return True
    if getattr(user, "role", None) == "admin":
        return True
    file_id = getattr(file_obj, "id", None)
    if not file_id:
        return False
    try:
        from open_webui.utils.access_control.files import (  # type: ignore[import-not-found]
            has_access_to_file,
        )
    except Exception:
        logger.warning("OWUI access-control helper unavailable; denying file read", exc_info=True)
        return False
    try:
        return bool(await has_access_to_file(str(file_id), "read", user))
    except Exception as exc:
        logger.warning("has_access_to_file failed for %s: %s", file_id, exc, exc_info=True)
        return False


@timed
async def materialize_owui_file_to_temp(
    file_obj: Any,
    *,
    user: Any,
    logger: logging.Logger,
    max_bytes: int,
    allow_unknown_size: bool,
    require_auth: bool = True,
    allowed_suffixes: set[str] | None = None,
    suffix: str = "",
) -> Path:
    """Authorise, size-gate, and copy an OWUI file to a private temp the caller owns.

    Routes the read through OWUI's Storage provider (any backend). Raises
    RequiredInternalFileError on auth denial, missing path, declared/actual size
    overrun, unknown-size cloud reads (unless allowed), unsupported extension, or
    a path that escapes UPLOAD_DIR. The caller must unlink the returned path.
    """
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
        raise RequiredInternalFileError("A referenced file exceeds the configured size limit.")
    if declared is None and owui_storage_provider_kind() != "local" and not allow_unknown_size:
        raise RequiredInternalFileError(
            "A referenced file has an unknown size and cannot be safely fetched from cloud storage."
        )

    storage = get_owui_storage()
    if storage is None:
        raise RequiredInternalFileError("Open WebUI storage is not available.")
    try:
        local_path = await asyncio.to_thread(storage.get_file, str(raw_path))
    except Exception as exc:
        logger.warning("Storage.get_file failed for %s: %s", raw_path, exc)
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

    temp_path = copy_to_private_temp(contained, suffix=suffix or contained.suffix)
    try:
        if max_bytes > 0 and temp_path.stat().st_size > max_bytes:
            raise RequiredInternalFileError("A referenced file exceeds the configured size limit.")
    except RequiredInternalFileError:
        temp_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        logger.warning("Failed to stat temp copy for %s: %s", raw_path, exc)
        raise RequiredInternalFileError("A referenced file could not be validated.") from exc
    return temp_path


def infer_file_mime_type(file_obj: Any) -> str:
    """Return the best-known MIME type for a stored Open WebUI file.

    Args:
        file_obj: File object from Open WebUI storage

    Returns:
        MIME type string (defaults to application/octet-stream)
    """
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
    return "application/octet-stream"


@timed
async def encode_file_path_base64(path: Path, chunk_size: int, max_bytes: int) -> str:
    """Read ``path`` in chunks and return a base64 string.

    Args:
        path: Path to file
        chunk_size: Chunk size for reading
        max_bytes: Maximum file size in bytes

    Returns:
        Base64-encoded file content

    Raises:
        ValueError: If file exceeds size limit
    """
    chunk_size = max(64 * 1024, min(chunk_size, max_bytes))

    def _encode_stream() -> str:
        total = 0
        buffer = io.StringIO()
        leftover = b""
        with path.open("rb") as source:
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("File exceeds BASE64_MAX_SIZE_MB limit")
                chunk = leftover + chunk
                whole_bytes = (len(chunk) // 3) * 3
                if whole_bytes:
                    buffer.write(base64.b64encode(chunk[:whole_bytes]).decode("ascii"))
                leftover = chunk[whole_bytes:]
        if leftover:
            buffer.write(base64.b64encode(leftover).decode("ascii"))
        return buffer.getvalue()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _encode_stream)


def extract_internal_file_id(url: str) -> str | None:
    """Return the Open WebUI file identifier embedded in a storage URL.

    Args:
        url: URL to extract file ID from

    Returns:
        File ID or None if not found
    """
    if not isinstance(url, str):
        return None
    match = _INTERNAL_FILE_ID_PATTERN.search(url)
    if match:
        return match.group(1)
    return None


def is_internal_file_url(url: str) -> bool:
    """True only for relative Open WebUI file paths (/api/v1/files/...); absolute URLs are external."""
    if not isinstance(url, str):
        return False
    if url.startswith(("http://", "https://")):
        return False
    return "/api/v1/files/" in url


class OwuiFileGateway:
    """Stateful gateway for authorized, backend-agnostic OWUI file storage I/O."""

    def __init__(self, logger, valves):
        self.logger = logger
        self.valves = valves
        self._storage_user_cache = None
        self._storage_user_lock = None
        self._storage_role_warning_emitted = False
        self._user_insert_param_names = None

    def validate_base64_size(self, b64_data: str) -> bool:
        """Validate base64 data size is within configured limits.

        Estimates the decoded size of base64 data and compares it against the
        configured BASE64_MAX_SIZE_MB valve to prevent memory issues from huge payloads.

        Args:
            b64_data: Base64-encoded string to validate

        Returns:
            True if within limits, False if too large
        """
        if not b64_data:
            return True

        estimated_size_bytes = (len(b64_data) * 3) / 4
        max_size_bytes = self.valves.BASE64_MAX_SIZE_MB * 1024 * 1024

        if estimated_size_bytes > max_size_bytes:
            estimated_size_mb = estimated_size_bytes / (1024 * 1024)
            self.logger.warning(
                f"Base64 data size (~{estimated_size_mb:.1f}MB) exceeds configured limit "
                f"({self.valves.BASE64_MAX_SIZE_MB}MB), rejecting to prevent memory issues"
            )
            return False

        return True

    @timed
    async def read_file_record_base64(
        self,
        file_obj: Any,
        chunk_size: int,
        max_bytes: int,
        *,
        user: Any = None,
        require_auth: bool = True,
    ) -> str | None:
        """Return base64 for an OWUI file: authorize, then read inline data or via Storage.

        Real OWUI records are authorized before any byte source is touched, and their
        path read is routed through OWUI Storage (works on any backend) into a private
        temp. Synthetic no-id records keep the legacy local-path/inline behavior.

        Raises:
            ValueError: stored/encoded content exceeds the configured size limit.
            RequiredInternalFileError: requester is unauthorized, or a real record
                cannot be safely materialized (size/containment/storage failure).
        """
        if max_bytes <= 0:
            raise ValueError("BASE64_MAX_SIZE_MB must be greater than zero")

        real = is_real_owui_file_record(file_obj)
        if require_auth and real and not await authorize_file_read(file_obj, user, self.logger):
            raise RequiredInternalFileError(
                "You do not have access to a referenced file.", denied=True
            )

        def _from_bytes(raw: bytes) -> str:
            if len(raw) > max_bytes:
                raise ValueError("File exceeds BASE64_MAX_SIZE_MB limit")
            return base64.b64encode(raw).decode("ascii")

        data_field = getattr(file_obj, "data", None)
        if isinstance(data_field, dict):
            for key in ("b64", "base64", "data"):
                inline_value = data_field.get(key)
                if isinstance(inline_value, str) and inline_value.strip():
                    if not self.validate_base64_size(inline_value):
                        raise ValueError("Stored base64 payload exceeds configured limit")
                    return inline_value.strip()
            blob_value = data_field.get("bytes")
            if isinstance(blob_value, (bytes, bytearray)):
                return _from_bytes(bytes(blob_value))

        if real:
            if getattr(file_obj, "path", None):
                temp_path = await materialize_owui_file_to_temp(
                    file_obj,
                    user=user,
                    logger=self.logger,
                    max_bytes=max_bytes,
                    allow_unknown_size=bool(
                        getattr(self.valves, "ALLOW_UNKNOWN_SIZE_CLOUD_READS", False)
                    ),
                    require_auth=False,
                )
                try:
                    return await encode_file_path_base64(temp_path, chunk_size, max_bytes)
                finally:
                    temp_path.unlink(missing_ok=True)
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
                return await encode_file_path_base64(path, chunk_size, max_bytes)

        raw_bytes = None
        for attr in ("content", "blob", "data"):
            value = getattr(file_obj, attr, None)
            if isinstance(value, (bytes, bytearray)):
                raw_bytes = bytes(value)
                break
        if raw_bytes is not None:
            return _from_bytes(raw_bytes)
        return None

    @timed
    async def inline_owui_file_id(
        self,
        file_id: str,
        *,
        chunk_size: int,
        max_bytes: int,
        user: Any = None,
    ) -> InlinedFile | None:
        """Convert an Open WebUI file id into a data URL with metadata.

        Args:
            file_id: Open WebUI file identifier
            chunk_size: Chunk size for reading file
            max_bytes: Maximum file size in bytes
            user: Requester for authorization; None denies real OWUI records.

        Returns:
            InlinedFile(data_url, filename) or None if conversion fails

        Raises:
            RequiredInternalFileError: requester is not authorized for the file.
        """
        normalized = (file_id or "").strip()
        if not normalized:
            return None
        file_obj = await get_file_by_id(normalized, self.logger)
        if not file_obj:
            return None
        mime_type = infer_file_mime_type(file_obj)
        try:
            b64 = await self.read_file_record_base64(
                file_obj, chunk_size, max_bytes, user=user
            )
        except ValueError as exc:
            self.logger.warning("Failed to inline file %s: %s", normalized, exc)
            return None
        if not b64:
            return None
        data_url = f"data:{mime_type};base64,{b64}"
        meta = getattr(file_obj, "meta", None)
        filename = ""
        if isinstance(meta, dict):
            filename = meta.get("name", "") or ""
        if not filename:
            filename = getattr(file_obj, "filename", "") or ""
        return InlinedFile(data_url=data_url, filename=filename)

    @timed
    async def inline_internal_file_url(
        self,
        url: str,
        *,
        chunk_size: int,
        max_bytes: int,
        user: Any = None,
    ) -> InlinedFile | None:
        """Convert an Open WebUI file URL into a data URL with metadata.

        Args:
            url: Open WebUI file URL
            chunk_size: Chunk size for reading file
            max_bytes: Maximum file size in bytes
            user: Requester for authorization; None denies real OWUI records.

        Returns:
            InlinedFile(data_url, filename) or None if conversion fails
        """
        file_id = extract_internal_file_id(url)
        if not file_id:
            return None
        return await self.inline_owui_file_id(
            file_id, chunk_size=chunk_size, max_bytes=max_bytes, user=user
        )

    @timed
    async def inline_internal_responses_input_files_inplace(
        self,
        request_body: dict[str, Any],
        *,
        chunk_size: int,
        max_bytes: int,
        user: Any = None,
    ) -> None:
        """Inline any Open WebUI internal file URLs referenced by /responses input_file blocks.

        OpenRouter providers cannot fetch Open WebUI internal URLs. This converts internal
        `file_url` (or internal `file_data` URL) values into `file_data` data URLs.

        Args:
            request_body: Request body dict containing input items
            chunk_size: Chunk size for reading files
            max_bytes: Maximum file size in bytes
            user: Requester for authorization; None denies real OWUI records.

        Raises:
            ValueError: If inlining fails for a required file.
            RequiredInternalFileError: requester is not authorized for a referenced file.
        """
        input_items = request_body.get("input")
        if not isinstance(input_items, list) or not input_items:
            return

        for item in input_items:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list) or not content:
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "input_file":
                    continue

                file_url = block.get("file_url")
                file_data = block.get("file_data")
                file_id = block.get("file_id")

                internal_file_id: str | None = None

                if isinstance(file_id, str) and file_id.strip():
                    candidate_id = file_id.strip()
                    if candidate_id.startswith("file-"):
                        continue
                    internal_file_id = candidate_id
                elif isinstance(file_data, str) and file_data.strip() and is_internal_file_url(file_data.strip()):
                    internal_file_id = extract_internal_file_id(file_data.strip())
                elif isinstance(file_url, str) and file_url.strip() and is_internal_file_url(file_url.strip()):
                    internal_file_id = extract_internal_file_id(file_url.strip())

                if not internal_file_id:
                    continue

                result = await self.inline_owui_file_id(
                    internal_file_id,
                    chunk_size=chunk_size,
                    max_bytes=max_bytes,
                    user=user,
                )
                if not result:
                    raise ValueError(
                        f"Failed to inline Open WebUI file id for /responses: {internal_file_id}"
                    )

                block["file_data"] = result.data_url
                if result.filename and "filename" not in block:
                    block["filename"] = result.filename
                block.pop("file_id", None)
                if isinstance(file_url, str) and file_url.strip() and is_internal_file_url(file_url.strip()):
                    block.pop("file_url", None)

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
    ) -> str | None:
        """Upload file or image to Open WebUI storage and return the OWUI file id.

        Args:
            request: FastAPI Request object for URL generation
            user: UserModel object representing the file owner
            file_data: Raw bytes of the file content
            filename: Desired filename (will be prefixed with UUID)
            mime_type: MIME type of the file (e.g., 'image/jpeg', 'application/pdf')
            chat_id: Optional chat ID to associate file with
            message_id: Optional message ID to associate file with
            owui_user_id: Optional user ID override

        Returns:
            Open WebUI file id (UUID string), or None if upload fails.
        """
        if upload_file_handler is None:
            self.logger.error("Open WebUI file upload helpers are unavailable; skipping OWUI storage upload.")
            return None
        try:
            upload_metadata: dict[str, Any] = {"mime_type": mime_type}
            if isinstance(chat_id, str):
                normalized_chat_id = chat_id.strip()
                if is_linkable_chat(normalized_chat_id):
                    upload_metadata["chat_id"] = normalized_chat_id
            if isinstance(message_id, str):
                normalized_message_id = message_id.strip()
                if normalized_message_id:
                    upload_metadata["message_id"] = normalized_message_id

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

            effective_user_id: str | None = None
            if isinstance(owui_user_id, str) and owui_user_id.strip():
                effective_user_id = owui_user_id.strip()
            else:
                candidate = getattr(user, "id", None)
                if isinstance(candidate, str) and candidate.strip():
                    effective_user_id = candidate.strip()

            try:
                linked = await self.try_link_file_to_chat(
                    chat_id=chat_id,
                    message_id=message_id,
                    file_id=file_id,
                    user_id=effective_user_id,
                )
            except Exception:
                linked = False
                self.logger.warning(
                    "Chat-link raised for file %s; the upload itself succeeded", file_id, exc_info=True
                )
            if not linked and is_linkable_chat(chat_id):
                self.logger.warning(
                    "File %s was not linked to chat %s; shared-chat viewers may not load it",
                    file_id,
                    chat_id,
                )

            self.logger.info(
                f"Uploaded {filename} ({len(file_data):,} bytes) to OWUI storage: /api/v1/files/{file_id}"
            )
            return file_id
        except Exception:
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
    ) -> str | None:
        """Stream a file from a local path into Open WebUI storage and return the file id."""
        if upload_file_handler is None:
            self.logger.error("Open WebUI file upload helpers are unavailable; skipping OWUI storage upload.")
            return None
        if not source_path.is_file():
            self.logger.error("Source path %s is not a file; aborting OWUI streaming upload.", source_path)
            return None
        try:
            upload_metadata: dict[str, Any] = {"mime_type": mime_type}
            if isinstance(chat_id, str):
                normalized_chat_id = chat_id.strip()
                if is_linkable_chat(normalized_chat_id):
                    upload_metadata["chat_id"] = normalized_chat_id
            if isinstance(message_id, str):
                normalized_message_id = message_id.strip()
                if normalized_message_id:
                    upload_metadata["message_id"] = normalized_message_id

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

            effective_user_id: str | None = None
            if isinstance(owui_user_id, str) and owui_user_id.strip():
                effective_user_id = owui_user_id.strip()
            else:
                candidate = getattr(user, "id", None)
                if isinstance(candidate, str) and candidate.strip():
                    effective_user_id = candidate.strip()

            try:
                linked = await self.try_link_file_to_chat(
                    chat_id=chat_id,
                    message_id=message_id,
                    file_id=file_id,
                    user_id=effective_user_id,
                )
            except Exception:
                linked = False
                self.logger.warning(
                    "Chat-link raised for file %s; the upload itself succeeded", file_id, exc_info=True
                )
            if not linked and is_linkable_chat(chat_id):
                self.logger.warning(
                    "File %s was not linked to chat %s; shared-chat viewers may not load it",
                    file_id,
                    chat_id,
                )

            self.logger.info(
                f"Streaming-uploaded {filename} ({size_bytes:,} bytes) to OWUI storage: /api/v1/files/{file_id}"
            )
            return file_id
        except Exception:
            self.logger.exception("Failed to streaming-upload %s to OWUI storage", source_path)
            return None

    @timed
    async def try_link_file_to_chat(
        self,
        *,
        chat_id: str | None,
        message_id: str | None,
        file_id: str,
        user_id: str | None,
    ) -> bool:
        """Link uploaded file to chat and message in Open WebUI database.

        Args:
            chat_id: Chat identifier
            message_id: Message identifier
            file_id: File identifier
            user_id: User identifier

        Returns:
            True if linking succeeded, False otherwise
        """
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

        normalized_message_id: str | None = None
        if isinstance(message_id, str):
            candidate = message_id.strip()
            if candidate:
                normalized_message_id = candidate

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
                    message_id=normalized_message_id or "",
                    file_ids=[target],
                    user_id=normalized_user_id,
                ),
                normalized_chat_id,
                normalized_message_id or "",
                target,
            )
        except TypeError:
            try:
                return await _linked_ok(
                    await Chats.insert_chat_files(
                        normalized_chat_id,
                        normalized_message_id or "",
                        [target],
                        normalized_user_id,
                    ),
                    normalized_chat_id,
                    normalized_message_id or "",
                    target,
                )
            except Exception:
                self.logger.warning(
                    "Chat-link failed for file %s (positional call)", file_id, exc_info=True
                )
                return False
        except Exception:
            self.logger.warning("Chat-link failed for file %s", file_id, exc_info=True)
            return False

    async def resolve_storage_context(
        self,
        request: Request | None,
        user_obj: Any | None,
    ) -> tuple[Request | None, Any | None]:
        """Return a `(request, user)` tuple suitable for OWUI uploads.

        Args:
            request: FastAPI Request object
            user_obj: User object

        Returns:
            Tuple of (request, user) or (None, None) if context unavailable
        """
        if request is None:
            if user_obj:
                self.logger.debug("Storage upload skipped: request context missing.")
            return None, None
        if user_obj is not None:
            return request, user_obj

        fallback_user = await self.ensure_storage_user()
        if fallback_user is None:
            return None, None
        self.logger.debug("Using fallback storage user '%s' for upload.", fallback_user.email)
        return request, fallback_user

    @timed
    async def ensure_storage_user(self) -> Any | None:
        """Ensure the fallback storage user exists (lazy creation).

        Returns:
            User object or None if creation fails
        """
        if Users is None:
            self.logger.debug("Cannot create storage user: Open WebUI integration not available")
            return None

        if self._storage_user_cache is not None:
            return self._storage_user_cache

        if self._storage_user_lock is None:
            self._storage_user_lock = asyncio.Lock()

        async with self._storage_user_lock:
            if self._storage_user_cache is not None:
                return self._storage_user_cache

            fallback_email = self.valves.FALLBACK_STORAGE_EMAIL or "openrouter-pipe@system.local"
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
            except Exception:  # pragma: no cover - defensive guard
                self.logger.exception("Failed to load fallback storage user")
                return None

            if fallback_user is None:
                user_id = f"openrouter-pipe-{uuid.uuid4().hex}"
                try:
                    oauth_marker = f"openrouter-pipe-storage:{uuid.uuid4().hex}"
                    insert_fn = Users.insert_new_user
                    insert_kwargs: dict[str, Any] = {}
                    try:
                        if self._user_insert_param_names is None:
                            sig = inspect.signature(insert_fn)
                            self._user_insert_param_names = tuple(sig.parameters.keys())
                    except (TypeError, ValueError):
                        self._user_insert_param_names = ()

                    param_names = self._user_insert_param_names or ()
                    if "oauth" in param_names:
                        insert_kwargs["oauth"] = {"sub": oauth_marker}
                    elif "oauth_sub" in param_names:
                        insert_kwargs["oauth_sub"] = oauth_marker

                    fallback_user = await insert_fn(
                        user_id,
                        fallback_name,
                        fallback_email,
                        "/user.png",
                        fallback_role or "pending",
                        **insert_kwargs,
                    )
                    self.logger.info(
                        "Created fallback storage user '%s' (%s) for multimodal uploads.",
                        fallback_name,
                        fallback_email,
                    )
                except Exception:  # pragma: no cover - defensive guard
                    self.logger.exception("Failed to create fallback storage user")
                    return None

            self._storage_user_cache = fallback_user
            return fallback_user
