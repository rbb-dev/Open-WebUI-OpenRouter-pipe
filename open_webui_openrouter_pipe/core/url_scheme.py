from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

HTTP_SCHEMES = frozenset({"http", "https"})

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
    if url[:5].lower() == "data:":
        return "data"
    return None


def url_scheme(url: Any) -> str:
    if not isinstance(url, str):
        return ""
    prefix = _scheme_prefix(url)
    if prefix is not None:
        return prefix
    try:
        return urlsplit(url).scheme
    except ValueError:
        pass
    try:
        return urlsplit(f"{url.partition(':')[0]}:").scheme
    except ValueError:
        return ""


def is_absolute_url(url: Any) -> bool:
    if not isinstance(url, str):
        return False
    if _scheme_prefix(url) is not None:
        return True
    try:
        parts = urlsplit(url)
    except ValueError:
        try:
            parts = urlsplit(f"{url.partition(':')[0]}:")
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
        parts = urlsplit(value)
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
        return urlsplit(value).path
    except ValueError:
        return value


def is_http_or_https_url(url: Any) -> bool:
    return url_scheme(url) in HTTP_SCHEMES


def is_inline_data_url(url: Any) -> bool:
    return url_scheme(url) == "data"


def _data_url_header(value: Any) -> tuple[bool, str] | None:
    if not isinstance(value, str) or url_scheme(value) != "data":
        return None
    header, sep, _ = value[value.find(":") + 1:].partition(",")
    return (bool(sep), header)


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
        parts = urlsplit(candidate)
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


def split_base64_data_url(value: Any) -> tuple[str, str] | None:
    if url_scheme(value) != "data" or not isinstance(value, str):
        return None
    header, sep, payload = value.partition(",")
    if not sep:
        return None
    lowered = header.lower()
    at = lowered.find(";base64")
    while at != -1:
        lowered_end = at + len(";base64")
        end = len(header) - (len(lowered) - lowered_end)
        if end == len(header) or header[end] == ";":
            return header, payload
        at = lowered.find(";base64", at + 1)
    return None
