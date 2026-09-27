from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

HTTP_SCHEMES = frozenset({"http", "https"})


def url_scheme(url: Any) -> str:
    if not isinstance(url, str):
        return ""
    for candidate in (url, f"{url.partition(':')[0]}:"):
        try:
            return urlsplit(candidate).scheme
        except ValueError:
            continue
    return ""


def is_absolute_url(url: Any) -> bool:
    if not isinstance(url, str):
        return False
    for candidate in (url, f"{url.partition(':')[0]}:"):
        try:
            parts = urlsplit(candidate)
        except ValueError:
            continue
        return bool(parts.scheme or parts.netloc)
    return False


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


def is_http_or_https_url(url: Any) -> bool:
    return url_scheme(url) in HTTP_SCHEMES


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
