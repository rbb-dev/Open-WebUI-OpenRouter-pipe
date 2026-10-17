from __future__ import annotations

import base64
import binascii
import codecs
import contextlib
import hashlib
import hmac
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives import hmac as crypto_hmac
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from sqlalchemy import Text, func
from sqlalchemy import cast as sa_cast

from .persistence import (
    _JSON_LEAD_BYTES,
    _PAYLOAD_FLAG_LZ4,
    _PAYLOAD_FLAG_PLAIN,
    _db_session,
    lz4frame,
)

logger = logging.getLogger(__name__)

SQLITE_READ_BYTES = 1024 * 1024
SLICE_CHARS = 8 * 1024 * 1024
INFLATE_STEP_BYTES = 256 * 1024
PULL_CHARS = 256 * 1024
FIELD_MAX_CHARS = 8 * 1024 * 1024
RECORD_MAX_CHARS = 8 * 1024 * 1024
FILE_CHUNK_BYTES = 256 * 1024
FERNET_VERSION = 0x80
FERNET_HEAD_BYTES = 25
FERNET_MAC_BYTES = 32

_BLOBOPEN_WARNED = False

_RAW_DECODE = json.JSONDecoder().raw_decode
_EVENTS = re.compile(r'"events"\s*:\s*\[')
_CIPHERTEXT = re.compile(rb'"ciphertext"\s*:\s*"')


class StoredPayloadUnreadable(ValueError):
    pass


class StoredPayloadUnavailable(ValueError):
    pass


@contextlib.contextmanager
def _sqlite_driver(store: Any) -> Iterator[Any]:
    global _BLOBOPEN_WARNED
    if store._engine.dialect.name != "sqlite":
        yield None
        return
    raw = store._engine.raw_connection()
    try:
        driver = getattr(raw, "driver_connection", None) or raw.connection
        if not hasattr(driver, "blobopen"):
            if not _BLOBOPEN_WARNED:
                _BLOBOPEN_WARNED = True
                logger.warning(
                    "The SQLite driver in use has no incremental blob I/O (blobopen), so staged "
                    "session log rows are read through repeated SQL substring queries instead; "
                    "passes over very large rows hold more memory."
                )
            yield None
            return
        yield driver
    finally:
        raw.close()


def _blob(store: Any, driver: Any, chat_id: str, message_id: str, row_id: str) -> Any:
    table = store._item_model.__table__.name
    found = driver.execute(
        f"SELECT rowid FROM {store._quote_identifier(table)} WHERE id = ? AND chat_id = ? AND message_id = ?",
        (row_id, chat_id, message_id),
    ).fetchone()
    if found is None:
        raise StoredPayloadUnavailable(f"staged row {row_id} is gone")
    return contextlib.closing(driver.blobopen(table, "payload", found[0], readonly=True))


def stored_sizes(store: Any, chat_id: str, message_id: str, row_ids: list[str]) -> dict[str, int]:
    with _sqlite_driver(store) as driver:
        if driver is not None:
            sizes: dict[str, int] = {}
            for row_id in row_ids:
                with _blob(store, driver, chat_id, message_id, row_id) as blob:
                    sizes[row_id] = len(blob)
            return sizes
    model = store._item_model
    with _db_session(store._session_factory) as session:
        rows = (
            session.query(model.id, func.length(sa_cast(model.payload, Text)))
            .filter(model.chat_id == chat_id)
            .filter(model.message_id == message_id)
            .filter(model.id.in_(row_ids))
            .all()
        )
    return {row[0]: int(row[1]) for row in rows if row[1] is not None}


def _stored_chunks(store: Any, chat_id: str, message_id: str, row_id: str) -> Generator[str | bytes, None, None]:
    with _sqlite_driver(store) as driver:
        if driver is not None:
            with _blob(store, driver, chat_id, message_id, row_id) as blob:
                while True:
                    piece = blob.read(SQLITE_READ_BYTES)
                    if not piece:
                        return
                    yield piece
    model = store._item_model
    start = 1
    while True:
        with _db_session(store._session_factory) as session:
            piece = (
                session.query(func.substr(sa_cast(model.payload, Text), start, SLICE_CHARS))
                .filter(model.id == row_id)
                .filter(model.chat_id == chat_id)
                .filter(model.message_id == message_id)
                .scalar()
            )
        if piece is None:
            raise StoredPayloadUnavailable(f"staged row {row_id} is gone")
        if not piece:
            return
        yield piece
        if len(piece) < SLICE_CHARS:
            return
        start += SLICE_CHARS


def _blocks(chunks: Iterator[str | bytes], encoding: str) -> Iterator[bytes]:
    for chunk in chunks:
        if isinstance(chunk, str):
            for start in range(0, len(chunk), SQLITE_READ_BYTES):
                yield chunk[start : start + SQLITE_READ_BYTES].encode(encoding)
        else:
            yield chunk


def _b64(data: bytes) -> bytes:
    try:
        return base64.b64decode(data, altchars=b"-_", validate=True)
    except binascii.Error as exc:
        raise StoredPayloadUnreadable("the stored token is not base64") from exc


def _token_bytes(chunks: Iterator[str | bytes]) -> Iterator[bytes]:
    try:
        pieces = _blocks(chunks, "ascii")
        head = b""
        for piece in pieces:
            head += piece
            if len(head) >= 4096:
                break
        found = _CIPHERTEXT.search(head)
        if found is None:
            if not head.startswith(b'"'):
                raise StoredPayloadUnreadable("the stored row holds no ciphertext")
            start = 1
        else:
            start = found.end()
        carry = b""
        block: bytes | None = head[start:]
        closed = False
        while block is not None:
            end = block.find(b'"')
            data = carry + (block if end < 0 else block[:end])
            usable = len(data) - len(data) % 4
            if usable:
                yield _b64(data[:usable])
            carry = data[usable:]
            if end >= 0:
                closed = True
                break
            block = next(pieces, None)
        if not closed:
            raise StoredPayloadUnreadable("the stored token is not terminated")
        if carry:
            yield _b64(carry)
    except UnicodeEncodeError as exc:
        raise StoredPayloadUnreadable("the stored token holds characters base64 cannot") from exc


def _spooled_token(chunks: Iterator[str | bytes], spool_dir: Path, tick: Callable[[], None]) -> Any:
    with contextlib.ExitStack() as opened:
        sink = opened.enter_context(tempfile.TemporaryFile(dir=spool_dir))
        for piece in _token_bytes(chunks):
            sink.write(piece)
            tick()
        sink.seek(0)
        opened.pop_all()
        return sink


def _verify_token_file(spool: Any, key: bytes) -> None:
    size = os.fstat(spool.fileno()).st_size
    if size < FERNET_HEAD_BYTES + FERNET_MAC_BYTES:
        raise StoredPayloadUnreadable("the stored token is too short")
    if (size - FERNET_HEAD_BYTES - FERNET_MAC_BYTES) % 16:
        raise StoredPayloadUnreadable("the stored token length does not fit the cipher block")
    mac = crypto_hmac.HMAC(key[:16], hashes.SHA256())
    remaining = size
    spool.seek(0)
    first = spool.read(1)
    if not first or first[0] != FERNET_VERSION:
        raise StoredPayloadUnreadable("the stored token has an unknown version")
    mac.update(first)
    remaining -= 1
    while remaining > FERNET_MAC_BYTES:
        piece = spool.read(min(FILE_CHUNK_BYTES, remaining - FERNET_MAC_BYTES))
        if not piece:
            break
        mac.update(piece)
        remaining -= len(piece)
    tail = spool.read(remaining)
    if len(tail) != FERNET_MAC_BYTES:
        raise StoredPayloadUnreadable("the stored token ends early")
    if not hmac.compare_digest(mac.finalize(), tail):
        raise StoredPayloadUnreadable("the stored token fails its signature check")


def _decrypted_file(spool: Any, key: bytes) -> Iterator[bytes]:
    unpadder = padding.PKCS7(128).unpadder()
    body = os.fstat(spool.fileno()).st_size - FERNET_HEAD_BYTES - FERNET_MAC_BYTES
    spool.seek(0)
    head = spool.read(FERNET_HEAD_BYTES)
    if len(head) < FERNET_HEAD_BYTES or body < 0:
        raise StoredPayloadUnreadable("the stored token ends early")
    decryptor: Any = Cipher(algorithms.AES(key[16:]), modes.CBC(head[9:FERNET_HEAD_BYTES])).decryptor()
    while body > 0:
        piece = spool.read(min(FILE_CHUNK_BYTES, body))
        if not piece:
            break
        body -= len(piece)
        out = unpadder.update(decryptor.update(piece))
        if out:
            yield out
    try:
        tail = unpadder.update(decryptor.finalize()) + unpadder.finalize()
    except ValueError as exc:
        raise StoredPayloadUnreadable("the stored token has invalid padding") from exc
    if tail:
        yield tail


def _inflated(plain: Iterator[bytes]) -> Iterator[bytes]:
    first = next((chunk for chunk in plain if chunk), b"")
    if not first:
        return
    flag = first[0]
    if flag == _PAYLOAD_FLAG_LZ4:
        if lz4frame is None:
            raise StoredPayloadUnreadable("the row is compressed and the lz4 package is unavailable")
        decoder = lz4frame.LZ4FrameDecompressor()

        def feed(data: bytes) -> Iterator[bytes]:
            try:
                out = decoder.decompress(data, max_length=INFLATE_STEP_BYTES)
                while True:
                    if out:
                        yield out
                    if decoder.needs_input or decoder.eof:
                        return
                    out = decoder.decompress(b"", max_length=INFLATE_STEP_BYTES)
            except (RuntimeError, ValueError, EOFError) as exc:
                raise StoredPayloadUnreadable("the compressed row is damaged") from exc

        yield from feed(first[1:])
        for chunk in plain:
            yield from feed(chunk)
        if not decoder.eof:
            raise StoredPayloadUnreadable("the compressed row ends early")
    elif flag == _PAYLOAD_FLAG_PLAIN:
        if first[1:]:
            yield first[1:]
        yield from plain
    elif flag in _JSON_LEAD_BYTES:
        yield first
        yield from plain
    else:
        raise StoredPayloadUnreadable(f"invalid stored payload flag: {flag}")


def _utf8(blocks: Iterator[bytes]) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        for block in blocks:
            text = decoder.decode(block)
            if text:
                yield text
        tail = decoder.decode(b"", True)
    except UnicodeDecodeError as exc:
        raise StoredPayloadUnreadable("the stored text is not UTF-8") from exc
    if tail:
        yield tail


class _Window:
    def __init__(self, pieces: Iterator[str]) -> None:
        self.pieces = pieces
        self.buf = ""
        self.pos = 0

    def pull(self, want: int = PULL_CHARS) -> bool:
        parts = [self.buf[self.pos:]]
        got = 0
        for piece in self.pieces:
            parts.append(piece)
            got += len(piece)
            if got >= want:
                break
        if not got:
            return False
        self.buf = "".join(parts)
        self.pos = 0
        return True

    def peek(self) -> str:
        while True:
            while self.pos < len(self.buf) and self.buf[self.pos] in " \t\r\n":
                self.pos += 1
            if self.pos < len(self.buf):
                return self.buf[self.pos]
            if not self.pull():
                return ""

    def value(self, limit: int) -> Any:
        grow = PULL_CHARS
        while True:
            try:
                obj, end = _RAW_DECODE(self.buf, self.pos)
            except json.JSONDecodeError as exc:
                held = len(self.buf) - self.pos
                if held > limit or not self.pull(grow):
                    raise StoredPayloadUnreadable("a stored record is not valid JSON or is too large") from exc
                grow *= 2
                continue
            except RecursionError as exc:
                raise StoredPayloadUnreadable("a stored record is nested too deeply") from exc
            if isinstance(obj, int | float) and not isinstance(obj, bool):
                if end >= len(self.buf):
                    if self.pull(grow):
                        continue
                elif self.buf[end] in ".eE+-":
                    self.pull(grow)
                    continue
            self.pos = end
            return obj


def _members(text: str) -> dict[str, Any]:
    try:
        found = json.loads("{" + text.strip().strip(",{}") + "}")
    except ValueError as exc:
        raise StoredPayloadUnreadable("the stored header is not valid JSON") from exc
    return found if isinstance(found, dict) else {}


def scan_payload(pieces: Iterator[str]) -> Iterator[tuple[str, Any]]:
    win = _Window(pieces)
    while True:
        found = _EVENTS.search(win.buf, win.pos)
        if found is not None:
            break
        if len(win.buf) - win.pos > FIELD_MAX_CHARS:
            raise StoredPayloadUnreadable("the stored payload holds no records")
        if not win.pull():
            yield "fields", _members(win.buf[win.pos:])
            return
    yield "fields", _members(win.buf[win.pos:found.start()])
    win.pos = found.end()
    while True:
        lead = win.peek()
        if lead == "]":
            win.pos += 1
            break
        if lead == ",":
            win.pos += 1
            continue
        if not lead:
            raise StoredPayloadUnreadable("the stored record list is not closed")
        record = win.value(RECORD_MAX_CHARS)
        if isinstance(record, dict):
            yield "event", record
    rest = [win.buf[win.pos:]]
    size = len(rest[0])
    for piece in win.pieces:
        size += len(piece)
        if size > FIELD_MAX_CHARS:
            raise StoredPayloadUnreadable("the stored payload has data after its records")
        rest.append(piece)
    yield "fields", _members("".join(rest))


def stream_row(
    store: Any,
    chat_id: str,
    message_id: str,
    row_id: str,
    *,
    encrypted: bool,
    spool_dir: Path,
    tick: Callable[[], None],
) -> Generator[tuple[str, Any], None, None]:
    chunks = _stored_chunks(store, chat_id, message_id, row_id)
    try:
        if encrypted:
            if not getattr(store, "_encryption_key", ""):
                raise StoredPayloadUnavailable("no encryption key is configured")
            key = hashlib.sha256(store._encryption_key.encode("utf-8")).digest()
            spool = _spooled_token(chunks, spool_dir, tick)
            try:
                chunks.close()
                _verify_token_file(spool, key)
                for item in scan_payload(_utf8(_inflated(_decrypted_file(spool, key)))):
                    tick()
                    yield item
            finally:
                with contextlib.suppress(OSError):
                    spool.close()
            return
        for item in scan_payload(_utf8(_blocks(chunks, "utf-8"))):
            tick()
            yield item
    finally:
        with contextlib.suppress(Exception):
            chunks.close()
