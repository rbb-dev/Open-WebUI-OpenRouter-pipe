from __future__ import annotations

import asyncio
import base64
import binascii
import os
import re
import shutil
import tempfile
import uuid
import weakref
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal, Self

from .url_scheme import MEDIA_TYPE_PATTERN

SlotKind = Literal["data_url", "bare"]

_SLOT_WHITESPACE = re.compile(r"\s")

_SLOT_BASE64_TEXT = re.compile(r"[A-Za-z0-9+/=_-]*")


class MediaSlot:
    __slots__ = ("b64_len", "kind", "mime", "path", "raw_len", "text")

    kind: SlotKind
    mime: str
    path: str | None
    text: str | None
    raw_len: int | None
    b64_len: int

    def __init__(
        self,
        *,
        mime: str,
        kind: SlotKind = "data_url",
        path: str | os.PathLike[str] | None = None,
        text: str | None = None,
    ) -> None:
        if (path is None) == (text is None):
            raise ValueError("a media slot holds exactly one of a file path and base64 text")
        if kind not in ("data_url", "bare"):
            raise ValueError(f"unknown media slot kind {kind!r}")
        if kind == "data_url" and not MEDIA_TYPE_PATTERN.match(mime):
            raise ValueError("a data URL slot needs a lower-case media type")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "mime", mime)
        if path is not None:
            fspath = os.fspath(path)
            size = os.stat(fspath).st_size
            object.__setattr__(self, "path", fspath)
            object.__setattr__(self, "text", None)
            object.__setattr__(self, "raw_len", size)
            object.__setattr__(self, "b64_len", 4 * ((size + 2) // 3))
        else:
            assert text is not None
            if _SLOT_WHITESPACE.search(text) is not None:
                text = _SLOT_WHITESPACE.sub("", text)
            if _SLOT_BASE64_TEXT.fullmatch(text) is None:
                raise ValueError("a media slot's text is not base64")
            object.__setattr__(self, "path", None)
            object.__setattr__(self, "text", text)
            object.__setattr__(self, "raw_len", None)
            object.__setattr__(self, "b64_len", len(text))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a media slot is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a media slot is immutable")

    @property
    def prefix(self) -> str:
        return f"data:{self.mime};base64," if self.kind == "data_url" else ""

    @property
    def json_len(self) -> int:
        return 2 + len(self.prefix) + self.b64_len

    @property
    def approx_bytes(self) -> int:
        return (self.b64_len * 3) // 4

    def head(self, count: int) -> bytes:
        count = max(0, int(count))
        if self.path is not None:
            with open(self.path, "rb") as handle:
                return handle.read(count)
        assert self.text is not None
        chars = -(-count // 3) * 4
        piece = self.text[:chars]
        piece += "=" * (-len(piece) % 4)
        try:
            return base64.b64decode(piece)[:count]
        except (binascii.Error, ValueError):
            return b""

    def shaped(self, *, mime: str | None = None, kind: SlotKind | None = None) -> MediaSlot:
        new_mime = self.mime if mime is None else mime
        new_kind = self.kind if kind is None else kind
        if new_mime == self.mime and new_kind == self.kind:
            return self
        if self.path is not None:
            return MediaSlot(mime=new_mime, kind=new_kind, path=self.path)
        return MediaSlot(mime=new_mime, kind=new_kind, text=self.text)

    def _identity(self) -> tuple[str, str, str | None, str | None]:
        return (self.kind, self.mime, self.path, self.text)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MediaSlot) and self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())

    def __repr__(self) -> str:
        source = "file" if self.path is not None else "text"
        return f"<MediaSlot {self.kind} {self.mime or 'untyped'} {source} {self.b64_len} base64 chars>"

    def __copy__(self) -> MediaSlot:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> MediaSlot:
        return self

    def __reduce__(self) -> Any:
        raise TypeError("a media slot points into a request's own temporary files and is never serialised")


class SendSpool:
    def __init__(self) -> None:
        self.directory = tempfile.mkdtemp(prefix="openrouter-send-")
        self._finalizer = weakref.finalize(self, shutil.rmtree, self.directory, True)
        self._copies: dict[tuple[str, str], str] = {}
        self._copy_locks: dict[tuple[str, str], asyncio.Lock] = {}

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def new_path(self) -> Path:
        fd, name = tempfile.mkstemp(dir=self.directory)
        os.close(fd)
        return Path(name)

    def copy_lock(self, key: tuple[str, str]) -> asyncio.Lock:
        lock = self._copy_locks.get(key)
        if lock is None:
            lock = self._copy_locks[key] = asyncio.Lock()
        return lock

    def held_copy(self, key: tuple[str, str]) -> str | None:
        path = self._copies.get(key)
        if path is None:
            return None
        if os.path.isfile(path):
            return path
        del self._copies[key]
        return None

    def hold_copy(self, key: tuple[str, str], path: str) -> None:
        self._copies[key] = path

    def adopt(self, source: str | os.PathLike[str], suffix: str = "") -> Path:
        target = Path(self.directory) / f"{uuid.uuid4().hex}{suffix}"
        try:
            os.link(source, target)
        except OSError:
            try:
                shutil.copyfile(source, target)
            except OSError:
                target.unlink(missing_ok=True)
                raise
        return target

    def close(self) -> None:
        self._copies.clear()
        self._copy_locks.clear()
        self._finalizer()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


SEND_SPOOL: ContextVar[SendSpool | None] = ContextVar("openrouter_send_spool", default=None)


def current_send_spool() -> SendSpool:
    spool = SEND_SPOOL.get()
    if spool is None or spool.closed:
        spool = SendSpool()
        SEND_SPOOL.set(spool)
    return spool
