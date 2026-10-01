from __future__ import annotations

import re
from typing import Any
from urllib.parse import (
    _WHATWG_C0_CONTROL_OR_SPACE,  # pyright: ignore[reportAttributeAccessIssue]
    urlsplit,
)

HTTP_SCHEMES = frozenset({"http", "https"})

_SCHEME_WINDOW = 64
_MEMOISED_MAX_CHARS = 1024
_STRIPPED_SCHEME_BYTES = str.maketrans("", "", "\t\r\n")

MEDIA_TYPE_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9!#$&^_.+-]{0,126}/[a-z0-9][a-z0-9!#$&^_.+-]{0,126}$"
)


def media_type_or_empty(value: Any) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    candidate = text.split(";", 1)[0].strip().lower()
    return candidate if MEDIA_TYPE_PATTERN.match(candidate) else ""


def _scheme_prefix(url: str) -> str | None:
    window = url[:_SCHEME_WINDOW].lstrip(_WHATWG_C0_CONTROL_OR_SPACE)
    if window[:5].lower() != "data:" and (
        "\t" in window or "\r" in window or "\n" in window
    ):
        window = window.translate(_STRIPPED_SCHEME_BYTES)
    return "data" if window[:5].lower() == "data:" else None


_split_uncached = getattr(urlsplit, "__wrapped__", urlsplit)


def _split(value: str) -> Any:
    if _scheme_prefix(value) == "data" or len(value) > _MEMOISED_MAX_CHARS:
        return _split_uncached(value)
    return urlsplit(value)


def url_scheme(url: Any) -> str:
    if not isinstance(url, str):
        return ""
    prefix = _scheme_prefix(url)
    if prefix is not None:
        return prefix
    try:
        return _split(url).scheme
    except ValueError:
        pass
    try:
        return _split(f"{url.partition(':')[0]}:").scheme
    except ValueError:
        return ""


def is_absolute_url(url: Any) -> bool:
    if not isinstance(url, str):
        return False
    if _scheme_prefix(url) is not None:
        return True
    try:
        parts = _split(url)
    except ValueError:
        try:
            parts = _split(f"{url.partition(':')[0]}:")
        except ValueError:
            return False
    return bool(parts.scheme or parts.netloc)


def first_n_non_whitespace(text: str, n: int) -> str:
    step = 4096
    got = ""
    for i in range(0, max(len(text), 1), step):
        got += "".join(text[i : i + step].split())
        if len(got) >= n:
            return got[:n]
    return got[:n]


def is_cleartext_http_url(url: Any) -> bool:
    return url_scheme(url) == "http"


def url_site(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    try:
        parts = _split(value)
        port = parts.port
    except ValueError:
        return ""
    if not parts.scheme or not parts.netloc:
        return ""
    host = parts.hostname or parts.netloc.rpartition("@")[2]
    return f"{parts.scheme.lower()}://{host}" + (f":{port}" if port else "")


def url_path(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    try:
        return _split(value).path
    except ValueError:
        return value


def is_http_or_https_url(url: Any) -> bool:
    return url_scheme(url) in HTTP_SCHEMES


def is_inline_data_url(url: Any) -> bool:
    return url_scheme(url) == "data"


def _data_url_header(value: Any) -> tuple[bool, str] | None:
    if not isinstance(value, str) or url_scheme(value) != "data":
        return None
    colon = value.find(":")
    comma = value.find(",", colon + 1)
    if comma == -1:
        return (False, "")
    return (True, value[colon + 1 : comma])


def loggable_link(url: Any) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    candidate = url.strip()
    if url_scheme(candidate) == "data":
        parsed = _data_url_header(candidate)
        if parsed is None or not parsed[0]:
            return ""
        return f"data:{parsed[1].partition(';')[0].strip()[:64]}"
    try:
        parts = _split(candidate)
    except ValueError:
        return ""
    if not parts.netloc:
        return ""
    authority = parts.netloc.rsplit("@", 1)[-1]
    return f"{parts.scheme}://{authority}" if parts.scheme else f"//{authority}"


def link_media_type(value: Any) -> str:
    _r = _data_url_header(value)
    if _r is None or not _r[0]:
        return ""
    return _r[1].partition(";")[0].strip()[:64]


def _base64_marker_end(header: str) -> int | None:
    lowered = header.lower()
    at = lowered.find(";base64")
    while at != -1:
        lowered_end = at + len(";base64")
        end = len(header) - (len(lowered) - lowered_end)
        if end == len(header) or header[end] == ";":
            return end
        at = lowered.find(";base64", at + 1)
    return None


def split_base64_data_url(value: Any) -> tuple[str, str] | None:
    if url_scheme(value) != "data" or not isinstance(value, str):
        return None
    comma = value.find(",")
    if comma == -1:
        return None
    if _base64_marker_end(value[:comma]) is None:
        return None
    return value[:comma], value[comma + 1:]


def base64_data_url_payload_len(value: Any) -> int | None:
    if not isinstance(value, str) or url_scheme(value) != "data":
        return None
    comma = value.find(",")
    if comma == -1:
        return None
    if _base64_marker_end(value[:comma]) is None:
        return None
    return len(value) - comma - 1


_BASE64_FOLDED_CHARS = (
    "".join(chr(code) for code in range(0x21))
    + "\x7f\x85          "
    + "       　"
)


def base64_data_url_payload_chars(value: Any) -> int | None:
    if not isinstance(value, str) or url_scheme(value) != "data":
        return None
    comma = value.find(",")
    if comma == -1:
        return None
    if _base64_marker_end(value[:comma]) is None:
        return None
    start = comma + 1
    folded = sum(value.count(char, start) for char in _BASE64_FOLDED_CHARS)
    return len(value) - start - folded


def base64_data_url_media_type(value: Any) -> str:
    if not isinstance(value, str) or url_scheme(value) != "data":
        return ""
    colon = value.find(":")
    comma = value.find(",", colon + 1)
    if comma == -1 or _base64_marker_end(value[colon + 1 : comma]) is None:
        return ""
    return media_type_or_empty(value[colon + 1 : comma])
