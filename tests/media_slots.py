from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from open_webui_openrouter_pipe.core.media_slot import MediaSlot
from open_webui_openrouter_pipe.storage.owui_files import InlinedFile


def slot_base64(slot: MediaSlot) -> str:
    if slot.text is not None:
        return slot.text
    assert slot.path is not None
    return base64.b64encode(Path(slot.path).read_bytes()).decode("ascii")


def wire_form(value: Any) -> Any:
    if isinstance(value, MediaSlot):
        return value.prefix + slot_base64(value)
    return value


def wire_default(value: Any) -> str:
    if isinstance(value, MediaSlot):
        return wire_form(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def inlined_file(data_url: str, filename: str) -> InlinedFile:
    head, _, body = data_url.partition(";base64,")
    mime = head.removeprefix("data:") or "application/octet-stream"
    return InlinedFile(media=MediaSlot(mime=mime, text=body), filename=filename)


def bare_slot(text: str | None) -> MediaSlot | None:
    if not text:
        return None
    return MediaSlot(mime="", kind="bare", text=text)
