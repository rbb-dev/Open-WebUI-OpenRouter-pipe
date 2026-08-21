from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, NamedTuple
from urllib.parse import urlsplit, urlunsplit

import aiohttp

RELAY_HOSTS: tuple[str, ...] = ("litterbox", "catbox")

RELAY_RETENTIONS: tuple[str, ...] = ("1h", "12h", "24h", "72h")


class _Endpoint(NamedTuple):
    url: str
    field: str
    origins: tuple[str, ...]


_ENDPOINTS: dict[str, _Endpoint] = {
    "litterbox": _Endpoint(
        "https://litterbox.catbox.moe/resources/internals/api.php",
        "fileToUpload",
        ("catbox.moe",),
    ),
    "catbox": _Endpoint(
        "https://catbox.moe/user/api.php",
        "fileToUpload",
        ("catbox.moe",),
    ),
}

_KEEPS_FOREVER: frozenset[str] = frozenset({"catbox"})

_MAX_UPLOAD_SECONDS = 300

MAX_RELAY_SECONDS_PER_REQUEST = 300

_UPLOAD_ATTEMPTS = 3

_RETRY_PAUSE_SECONDS = 2.0

_MEDIA_TYPE = re.compile(
    r"^[a-z0-9][a-z0-9!#$&^_.+-]{0,126}/[a-z0-9][a-z0-9!#$&^_.+-]{0,126}$"
)

_UNQUOTABLE = re.compile(r"""[^A-Za-z0-9 .,;'"?!=%/-]""")

_HOST_REPLY_LIMIT = 120


class MediaRelayError(Exception):
    def __init__(self, message: str, *, may_have_stored_it: bool = True) -> None:
        super().__init__(message)
        self.may_have_stored_it = may_have_stored_it


_RAN_OUT_OF_TIME = (
    "the time this request may spend uploading to a file host ran out before the "
    "transfer finished"
)


def host_keeps_forever(host: str) -> bool:
    return host in _KEEPS_FOREVER


def usable_media_type(value: Any) -> str:
    candidate = value.split(";", 1)[0].strip().lower() if isinstance(value, str) else ""
    return candidate if _MEDIA_TYPE.match(candidate) else ""


def megabytes(count: int) -> str:
    return f"{count / (1024 * 1024):.1f} MB"


def _as_the_host_put_it(text: str) -> str:
    return f"`{_UNQUOTABLE.sub(' ', ' '.join(text.split()))[:_HOST_REPLY_LIMIT]}`"


def _served_by(link: str, origins: tuple[str, ...]) -> bool:
    try:
        parts = urlsplit(link)
    except ValueError:
        return False
    if urlunsplit(parts) != link:
        return False
    if parts.scheme != "https":
        return False
    hostname = (parts.hostname or "").strip().rstrip(".").lower()
    if not hostname:
        return False
    return any(
        hostname == origin or hostname.endswith(f".{origin}") for origin in origins
    )


def _validate_endpoints(endpoints: dict[str, _Endpoint]) -> None:
    for host, endpoint in endpoints.items():
        if not endpoint.origins:
            raise ValueError(
                f"{host!r} names no origin, so whatever address it answered with would "
                "be handed to OpenRouter as the user's file"
            )
        if not _served_by(endpoint.url, endpoint.origins):
            raise ValueError(
                f"{host!r} uploads to {endpoint.url!r}, which its own origins "
                f"{endpoint.origins!r} do not cover, so every answer it gave would read "
                "as a link from somewhere else"
            )


_validate_endpoints(_ENDPOINTS)


def _form(
    fields: dict[str, str], filename: str, blob: bytes, mime: str, field: str
) -> aiohttp.FormData:
    form = aiohttp.FormData()
    for name, value in fields.items():
        form.add_field(name, value)
    form.add_field(field, blob, filename=filename, content_type=mime)
    return form


async def relay_to_public_url(
    session: aiohttp.ClientSession,
    blob: bytes,
    *,
    filename: str,
    mime: str,
    host: str,
    retention: str,
    max_bytes: int,
    seconds_left: float = MAX_RELAY_SECONDS_PER_REQUEST,
) -> str:
    endpoint = _ENDPOINTS.get(host)
    if endpoint is None:
        raise MediaRelayError(
            f"{host!r} is not a file host this pipe knows how to use",
            may_have_stored_it=False,
        )
    content_type = usable_media_type(mime)
    if not content_type:
        raise MediaRelayError(
            f"{mime!r} is not a media type, so there is no honest way to declare the "
            "file to a host",
            may_have_stored_it=False,
        )
    if max_bytes > 0 and len(blob) > max_bytes:
        raise MediaRelayError(
            f"the file is {megabytes(len(blob))} and the limit for sending media to a "
            f"file host is {megabytes(max_bytes)}",
            may_have_stored_it=False,
        )
    if not blob:
        raise MediaRelayError("the file is empty", may_have_stored_it=False)

    fields = {"reqtype": "fileupload"}
    if host == "litterbox":
        fields["time"] = retention if retention in RELAY_RETENTIONS else RELAY_RETENTIONS[0]

    deadline = time.monotonic() + max(0.0, seconds_left)
    last = ""
    stored = False
    for attempt in range(_UPLOAD_ATTEMPTS):
        if attempt:
            await asyncio.sleep(min(_RETRY_PAUSE_SECONDS * attempt, _time_left(deadline)))
        window = _time_left(deadline)
        if window <= 0:
            last = last or f"{host} was not reached: {_RAN_OUT_OF_TIME}"
            break
        try:
            async with session.post(
                endpoint.url,
                data=_form(
                    fields, filename or "upload.bin", blob, content_type, endpoint.field
                ),
                timeout=aiohttp.ClientTimeout(total=min(_MAX_UPLOAD_SECONDS, window)),
                allow_redirects=False,
            ) as response:
                text = (await response.text()).strip()
                if response.status == 200:
                    link = _extract_url(text, endpoint.origins)
                    if link:
                        return link
                    stored = True
                    last = (
                        f"{host} answered without a link it serves: "
                        f"{_as_the_host_put_it(text)}"
                    )
                    break
                stored = response.status >= 500
                last = f"{host} answered {response.status}"
                break
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientConnectorError, aiohttp.ConnectionTimeoutError) as exc:
            last = f"{host} could not be reached: {_as_the_host_put_it(str(exc))}"
        except (aiohttp.ClientError, TimeoutError, OSError, UnicodeDecodeError) as exc:
            stored = True
            last = f"{host} did not answer: {_as_the_host_put_it(str(exc))}"
            break
    raise MediaRelayError(
        last or f"{host} did not accept the file", may_have_stored_it=stored
    )


def _time_left(deadline: float) -> float:
    return deadline - time.monotonic()


def _extract_url(text: str, origins: tuple[str, ...]) -> str:
    candidate = ""
    if text.startswith(("http://", "https://")):
        parts = text.split()
        candidate = parts[0] if parts else ""
    elif text.startswith("{"):
        try:
            payload: Any = json.loads(text)
        except ValueError:
            return ""
        data = payload.get("data") if isinstance(payload, dict) else None
        found = data.get("url") if isinstance(data, dict) else None
        candidate = found if isinstance(found, str) else ""
    return candidate if _served_by(candidate, origins) else ""
