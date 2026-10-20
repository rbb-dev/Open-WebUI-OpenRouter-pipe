from __future__ import annotations

import base64
import binascii
import itertools
import json
import os
import re
import shutil
import tempfile
from typing import IO, Any, Self

RECEIVE_ITEM_MAX_BYTES = 2 * 1024 * 1024 * 1024

EVENT_SCAN_BYTES = 1024 * 1024

PART_INLINE_DECODED_BYTES = 64 * 1024

_PART_KEEP_SLACK_CHARS = 8 * 1024

_PART_KEY_MAX_CHARS = 128

_SCAN_DECODE_QUANTUM_BYTES = 1024 * 1024

_MEDIA_VALUE_KEYS = frozenset(
    (b"b64_json", b"b64", b"base64", b"image_base64", b"imageB64", b"data")
)

_DATA_URL_PREFIX = b"data:image/"

_DATA_URL_B64_MARK = b";base64,"

_TOKEN_LEAD = "rrspool-"

_B64_WS_BYTES = b" \t\r\n"

_CONDITIONAL_MEDIA_KEYS = frozenset((b"data", b"result", b"imageUrl"))


def _media_signature(raw: bytes) -> str | None:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "audio/wav"
    if raw[4:8] == b"ftyp":
        return "video/mp4"
    if raw[:4] == b"\x1aE\xdf\xa3":
        return "video/webm"
    if raw[:3] == b"ID3" or (raw[0] == 0xFF and raw[1] in (0xFB, 0xF3, 0xF2)):
        return "audio/mpeg"
    if raw[:4] == b"OggS":
        return "audio/ogg"
    if raw[:4] == b"fLaC":
        return "audio/flac"
    return None


_SCAN_B64_ALPHABET = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
)

_STRUCT_TOKEN = re.compile(rb'[\[\]{},:"]')

_STRING_CHAR = re.compile(rb'["\\]')

_ESCAPE_MAP = {ord('"'): '"', ord("\\"): "\\", ord("/"): "/", ord("b"): "\b",
               ord("f"): "\f", ord("n"): "\n", ord("r"): "\r", ord("t"): "\t"}

_OVER_GUARD_REFUSAL = (
    "the pipe's per-item receive ceiling of {mib} MiB was passed "
    "({bytes} decoded bytes), so this item was not kept; everything else in "
    "the reply was received and its cost is recorded"
)


def over_guard_refusal(decoded_bytes: int) -> str:
    return _OVER_GUARD_REFUSAL.format(
        mib=RECEIVE_ITEM_MAX_BYTES // (1024 * 1024), bytes=decoded_bytes
    )


class PartRef(str):
    __slots__ = ("broken", "decoded_bytes", "gone", "mime", "over_guard", "path")

    def __new__(
        cls,
        token: str,
        path: str,
        decoded_bytes: int,
        mime: str | None,
        over_guard: bool,
        broken: bool = False,
    ) -> Self:
        ref = super().__new__(cls, token)
        ref.path = path
        ref.decoded_bytes = decoded_bytes
        ref.mime = mime
        ref.over_guard = over_guard
        ref.broken = broken
        ref.gone = False
        return ref

    @property
    def token(self) -> str:
        return str(self)

    def exists(self) -> bool:
        if self.gone:
            return False
        return os.path.exists(self.path)

    def open(self) -> IO[bytes]:
        return open(self.path, "rb")

    def read_head(self, size: int) -> bytes:
        with self.open() as handle:
            return handle.read(size)

    def read_all(self) -> bytes:
        with self.open() as handle:
            return handle.read()

    def mark_gone(self) -> None:
        self.gone = True

    def refusal(self) -> str:
        return over_guard_refusal(self.decoded_bytes)


class _ItemSink:
    __slots__ = ("handle", "over_guard", "path", "written")

    def __init__(self, directory: str) -> None:
        fd, name = tempfile.mkstemp(prefix="part-", dir=directory)
        os.close(fd)
        self.path = name
        self.handle: IO[bytes] | None = open(name, "wb")  # noqa: SIM115
        self.written = 0
        self.over_guard = False

    def feed(self, chunk: bytes) -> None:
        if self.handle is None:
            return
        room = RECEIVE_ITEM_MAX_BYTES - self.written
        if room <= 0:
            self.over_guard = True
            return
        if len(chunk) > room:
            if room:
                self.handle.write(chunk[:room])
                self.written += room
            self.over_guard = True
            return
        self.handle.write(chunk)
        self.written += len(chunk)

    def finish(self) -> None:
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()
            self.handle = None

    def discard(self) -> None:
        self.finish()
        try:
            os.unlink(self.path)
        except OSError:
            pass


class DocumentScanner:
    def __init__(self, spool: ReceiveSpool) -> None:
        self._spool = spool
        self._slim = bytearray()
        self._parts: dict[str, PartRef] = {}
        self._serial = itertools.count(1)
        self._stack: list[bytes] = []
        self._after_colon = False
        self._last_key = b""
        self._key_buf = bytearray()
        self._in_str = False
        self._escape = False
        self._key_mode = False
        self._mode = "pass"
        self._probe = bytearray()
        self._sink: _ItemSink | None = None
        self._mime: str | None = None
        self._keep: list[bytes] = []
        self._keep_len = 0
        self._keep_full = False
        self._token_str = ""
        self._pending_escape = b""
        self._u_left = 0
        self._u_hex = bytearray()
        self._carry = b""
        self._decoded = 0
        self._decode_broken = False

    def _emit(self, piece: bytes) -> None:
        if piece:
            self._slim += piece

    def _token(self) -> str:
        return f"{_TOKEN_LEAD}{next(self._serial)}"

    def _struct(self, chunk: bytes, at: int, char: bytes) -> int:
        self._emit(char)
        if char in b"{[":
            self._stack.append(char)
            self._after_colon = False
        elif char in b"}]":
            if self._stack:
                self._stack.pop()
            self._after_colon = False
        elif char == b",":
            self._after_colon = False
        else:
            self._after_colon = True
        return at + 1

    def _open_string(self, chunk: bytes, at: int) -> None:
        self._in_str = True
        self._escape = False
        self._key_mode = not self._after_colon and bool(self._stack) and self._stack[-1] == b"{"
        self._keep = []
        self._keep_len = 0
        self._keep_full = False
        self._carry = b""
        self._decoded = 0
        self._decode_broken = False
        self._probe = bytearray()
        self._sink = None
        self._mime = None
        self._token_str = ""
        if self._key_mode:
            self._key_buf = bytearray()
            self._mode = "key"
            self._emit(b'"')
            return
        self._mode = "probe"
        if self._after_colon and self._last_key in _MEDIA_VALUE_KEYS and self._last_key != b"data":
            self._promote(None, b"", None)

    def _promote(self, mime: str | None, seed: bytes, feed: bytes | None) -> None:
        self._mode = "copy"
        self._mime = mime
        self._token_str = self._token()
        self._sink = _ItemSink(self._spool.directory)
        if seed:
            self._keep.append(seed)
            self._keep_len = len(seed)
        body = seed if feed is None else feed
        if body:
            self._feed_quad(body, False)

    def _to_pass(self) -> None:
        self._mode = "pass"
        self._emit(b'"')
        if self._probe:
            self._emit(bytes(self._probe))
        self._probe = bytearray()

    def _probe_head_is_media(self) -> bool:
        head = bytes(self._probe[:96].translate(None, _B64_WS_BYTES))
        cut = len(head) - (len(head) % 4)
        if cut < 8:
            return False
        try:
            raw = base64.b64decode(head[:cut], validate=True)
        except (binascii.Error, ValueError):
            return False
        return _media_signature(raw) is not None

    def _probe_check(self) -> None:
        probe = self._probe
        size = len(probe)
        if probe.startswith(_DATA_URL_PREFIX):
            if size >= 11:
                mark = probe.find(_DATA_URL_B64_MARK)
                if mark >= 0:
                    self._promote(
                        probe[5:mark].decode("ascii", "replace"), bytes(probe), bytes(probe[mark + 8 :])
                    )
                    return
            if size >= 96:
                self._to_pass()
            return
        if (
            size >= 96
            and self._after_colon
            and self._last_key in _CONDITIONAL_MEDIA_KEYS
        ):
            if self._probe_head_is_media():
                self._promote(None, bytes(probe), None)
            else:
                self._to_pass()
            return
        if size >= 11:
            self._to_pass()

    def _feed_quad(self, piece: bytes, final: bool) -> None:
        if self._decode_broken:
            return
        piece = piece.translate(None, _B64_WS_BYTES)
        buf = self._carry + piece if self._carry else piece
        if final:
            if b"=" not in buf:
                pad = (-len(buf)) % 4
                if pad:
                    buf = buf + b"=" * pad
            aligned = len(buf)
        else:
            aligned = len(buf) & ~3
        if not aligned:
            self._carry = buf
            return
        self._carry = buf[aligned:] if not final else b""
        at = 0
        while at < aligned:
            take = min(aligned - at, _SCAN_DECODE_QUANTUM_BYTES)
            try:
                decoded = base64.b64decode(buf[at : at + take], validate=True)
            except (binascii.Error, ValueError):
                self._decode_broken = True
                return
            if decoded and self._sink is not None:
                self._sink.feed(decoded)
                self._decoded += len(decoded)
            at += take

    _KEEP_CAP = PART_INLINE_DECODED_BYTES * 4 // 3 + _PART_KEEP_SLACK_CHARS

    def _consume_char(self, char: bytes) -> None:
        if self._mode == "key":
            self._emit(char)
            if len(self._key_buf) < _PART_KEY_MAX_CHARS:
                self._key_buf += char
        elif self._mode == "pass":
            self._emit(char)
        elif self._mode == "probe":
            self._probe += char
            self._probe_check()
        else:
            self._keep_piece(char)

    def _keep_raw(self, piece: bytes) -> None:
        if self._keep_full:
            return
        if self._keep_len + len(piece) > self._KEEP_CAP:
            self._keep_full = True
            return
        self._keep.append(piece)
        self._keep_len += len(piece)

    def _keep_piece(self, raw: bytes) -> None:
        self._keep_raw(raw)
        self._feed_quad(raw, False)

    def _escape_step(self, chunk: bytes, at: int) -> int:
        size = len(chunk)
        if self._escape:
            if at >= size:
                return at
            char = chunk[at : at + 1]
            at += 1
            self._escape = False
            lead = self._pending_escape
            if self._mode != "copy":
                self._consume_char(lead)
                self._consume_char(char)
                return at
            self._keep_raw(lead)
            self._keep_raw(char)
            if char == b"u":
                self._u_left = 4
                self._u_hex = bytearray()
            else:
                mapped = _ESCAPE_MAP.get(char[0]) if len(char) == 1 else None
                if mapped is not None:
                    self._feed_quad(mapped.encode("ascii"), False)
            return at
        while self._u_left and at < size:
            take = min(self._u_left, size - at)
            hexpart = chunk[at : at + take]
            at += take
            self._u_left -= take
            self._keep_raw(hexpart)
            self._u_hex += hexpart
            if not self._u_left:
                try:
                    point = int(bytes(self._u_hex).decode("ascii"), 16)
                except ValueError:
                    point = -1
                self._u_hex = bytearray()
                if 0 <= point < 128:
                    self._feed_quad(chr(point).encode("ascii"), False)
        return at

    def _in_string(self, chunk: bytes, at: int) -> int:
        while True:
            found = _STRING_CHAR.search(chunk, at)
            if found is None:
                self._run_to(chunk, at, len(chunk))
                return len(chunk)
            marker = chunk[found.start() : found.start() + 1]
            self._run_to(chunk, at, found.start())
            if marker == b'"':
                if self._mode == "key":
                    self._emit(marker)
                    self._last_key = (
                        bytes(self._key_buf)
                        if len(self._key_buf) < _PART_KEY_MAX_CHARS
                        else b""
                    )
                else:
                    self._close_value()
                self._in_str = False
                return found.start() + 1
            self._escape = True
            self._pending_escape = marker
            return found.start() + 1

    def _run_to(self, chunk: bytes, at: int, end: int) -> None:
        if end <= at:
            return
        piece = chunk[at:end]
        if self._mode == "key":
            self._emit(piece)
            room = _PART_KEY_MAX_CHARS - len(self._key_buf)
            if room > 0:
                self._key_buf += piece[:room]
        elif self._mode == "pass":
            self._emit(piece)
        elif self._mode == "probe":
            self._probe += piece
            self._probe_check()
        else:
            self._keep_piece(piece)

    def _close_value(self) -> None:
        mode = self._mode
        if mode == "pass":
            self._emit(b'"')
            return
        if mode == "probe":
            self._emit(b'"')
            if self._probe:
                self._emit(bytes(self._probe))
            self._emit(b'"')
            self._probe = bytearray()
            return
        sink = self._sink
        if sink is None:
            self._mode = "pass"
            self._emit(b'"')
            return
        self._feed_quad(b"", True)
        sink.finish()
        if self._decode_broken:
            sink.discard()
            token = self._token_str
            self._emit(('"' + token + '"').encode("ascii"))
            self._parts[token] = PartRef(token, "", self._decoded, self._mime, False, True)
            return
        if (
            not self._keep_full
            and not sink.over_guard
            and self._decoded < PART_INLINE_DECODED_BYTES
        ):
            self._emit(b'"')
            for piece in self._keep:
                self._emit(piece)
            self._emit(b'"')
            sink.discard()
            return
        token = self._token_str
        self._emit(('"' + token + '"').encode("ascii"))
        self._parts[token] = PartRef(
            token, sink.path, self._decoded, self._mime, sink.over_guard
        )

    def feed(self, chunk: bytes) -> None:
        at = 0
        size = len(chunk)
        while at < size:
            if self._escape or self._u_left:
                at = self._escape_step(chunk, at)
                continue
            if self._in_str:
                at = self._in_string(chunk, at)
                continue
            found = _STRUCT_TOKEN.search(chunk, at)
            if found is None:
                self._emit(chunk[at:])
                return
            start = found.start()
            if start > at:
                self._emit(chunk[at:start])
            char = chunk[start : start + 1]
            if char == b'"':
                self._open_string(chunk, start)
                at = start + 1
            else:
                at = self._struct(chunk, start, char)

    def finish(self) -> tuple[bytes, dict[str, PartRef]]:
        if self._in_str:
            if self._mode == "copy" and self._sink is not None:
                try:
                    self._close_value()
                except ValueError:
                    self._sink.discard()
                    self._sink = None
            elif self._mode == "probe":
                self._emit(b'"')
                if self._probe:
                    self._emit(bytes(self._probe))
                self._emit(b'"')
                self._probe = bytearray()
            elif self._mode == "pass":
                self._emit(b'"')
            self._in_str = False
        return bytes(self._slim), self._parts


class ReceiveSpool:
    def __init__(self) -> None:
        self.directory = tempfile.mkdtemp(prefix="openrouter-receive-")
        self._closed = False

    def scanner(self) -> DocumentScanner:
        return DocumentScanner(self)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        shutil.rmtree(self.directory, ignore_errors=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def attach_parts(obj: Any, parts: dict[str, PartRef]) -> Any:
    if not parts:
        return obj
    return _attach(obj, parts)


def _attach(node: Any, parts: dict[str, PartRef]) -> Any:
    if isinstance(node, dict):
        return {k: _attach(v, parts) for k, v in node.items()}
    if isinstance(node, list):
        return [_attach(v, parts) for v in node]
    if type(node) is str:
        found = parts.get(node)
        if found is not None:
            return found
    return node


def scan_event(data_blob: bytes, spool: ReceiveSpool | None) -> Any | None:
    if spool is None:
        return None
    if len(data_blob) <= EVENT_SCAN_BYTES and b"b64" not in data_blob:
        return None
    try:
        scanner = spool.scanner()
        scanner.feed(data_blob)
        slim, parts = scanner.finish()
        return attach_parts(json.loads(slim), parts)
    except (ValueError, RecursionError, UnicodeDecodeError):
        return None
