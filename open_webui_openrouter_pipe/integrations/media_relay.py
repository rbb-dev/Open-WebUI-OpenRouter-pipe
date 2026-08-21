from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp

RELAY_HOSTS: tuple[str, ...] = ("litterbox", "catbox")

RELAY_RETENTIONS: tuple[str, ...] = ("1h", "12h", "24h", "72h")

_ENDPOINTS: dict[str, tuple[str, str]] = {
    "litterbox": ("https://litterbox.catbox.moe/resources/internals/api.php", "fileToUpload"),
    "catbox": ("https://catbox.moe/user/api.php", "fileToUpload"),
}

_KEEPS_FOREVER: frozenset[str] = frozenset({"catbox"})

_MAX_UPLOAD_SECONDS = 300

_UPLOAD_ATTEMPTS = 3

_RETRY_PAUSE_SECONDS = 2.0


class MediaRelayError(Exception):
    pass


def host_keeps_forever(host: str) -> bool:
    return host in _KEEPS_FOREVER


def _megabytes(count: int) -> str:
    return f"{count / (1024 * 1024):.1f} MB"


def _form(fields: dict[str, str], filename: str, blob: bytes, mime: str, field: str) -> aiohttp.FormData:
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
) -> str:
    if host not in _ENDPOINTS:
        raise MediaRelayError(f"{host!r} is not a file host this pipe knows how to use")
    if max_bytes > 0 and len(blob) > max_bytes:
        raise MediaRelayError(
            f"the file is {_megabytes(len(blob))} and the limit for sending media to a "
            f"file host is {_megabytes(max_bytes)}"
        )
    if not blob:
        raise MediaRelayError("the file is empty")

    url, field = _ENDPOINTS[host]
    fields = {"reqtype": "fileupload"}
    if host == "litterbox":
        fields["time"] = retention if retention in RELAY_RETENTIONS else RELAY_RETENTIONS[0]

    last = ""
    for attempt in range(_UPLOAD_ATTEMPTS):
        if attempt:
            await asyncio.sleep(_RETRY_PAUSE_SECONDS * attempt)
        try:
            async with session.post(
                url,
                data=_form(fields, filename or "upload.bin", blob, mime, field),
                timeout=aiohttp.ClientTimeout(total=_MAX_UPLOAD_SECONDS),
            ) as response:
                text = (await response.text()).strip()
                if response.status == 200:
                    link = _extract_url(text)
                    if link:
                        return link
                    last = f"{host} answered without a link: {text[:120]}"
                    break
                last = f"{host} answered {response.status}"
                if response.status < 500 and response.status != 429:
                    break
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, TimeoutError, OSError, UnicodeDecodeError) as exc:
            last = f"{host} could not be reached: {exc}"
    raise MediaRelayError(last or f"{host} did not accept the file")


def _extract_url(text: str) -> str:
    if text.startswith(("http://", "https://")):
        parts = text.split()
        return parts[0] if parts else ""
    if text.startswith("{"):
        try:
            payload: Any = json.loads(text)
        except ValueError:
            return ""
        found = (payload.get("data") or {}).get("url") if isinstance(payload, dict) else None
        return found if isinstance(found, str) and found.startswith("https://") else ""
    return ""
