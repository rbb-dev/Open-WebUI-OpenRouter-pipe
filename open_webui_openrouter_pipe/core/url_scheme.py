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


def is_http_or_https_url(url: Any) -> bool:
    return url_scheme(url) in HTTP_SCHEMES
