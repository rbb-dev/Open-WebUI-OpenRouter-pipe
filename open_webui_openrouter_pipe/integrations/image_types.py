from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..storage.multimodal import image_extension_for_mime


class ImageGenerationError(RuntimeError):
    """A generation the pipe could not turn into an image.

    ``usage`` carries the counters from a 200 response the pipe then failed to decode, so a
    billed request is still reported and costed rather than recorded as free.
    """

    def __init__(self, *args: object, usage: dict[str, object] | None = None) -> None:
        super().__init__(*args)
        self.usage = usage or {}


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    data: bytes
    mime_type: str

    @property
    def extension(self) -> str:
        return image_extension_for_mime(self.mime_type)


_TEXT_LIMIT = 120


def clamp_text(text: Any, limit: int = _TEXT_LIMIT) -> str:
    """Bound a span of text the pipe did not author before it reaches a log or the browser.

    Applies to both sides of the wire: a request key the client chose and a rejection
    reason built from an upstream reply are equally able to size a log record.
    """
    rendered = text if isinstance(text, str) else str(text)
    return rendered if len(rendered) <= limit else f"{rendered[:limit]}…"


def summarise_names(names: list[str], limit: int = 4, width: int = _TEXT_LIMIT) -> str:
    """Render a list the pipe did not author, bounded in both element size and count.

    Clamping each element still lets the count carry the payload, and capping the count
    still lets one element carry it. Both bounds have to hold at the point of rendering.
    ``width`` exists for callers whose elements were already clamped at construction: a
    second, tighter clamp there would truncate a legitimate message rather than bound it.
    """
    shown = [clamp_text(name, width) for name in names[:limit]]
    tail = f" and {len(names) - len(shown)} more" if len(names) > len(shown) else ""
    return f"{'; '.join(shown)}{tail}"


@dataclass(frozen=True, slots=True)
class ImageGenerationResult:
    images: list[GeneratedImage] = field(default_factory=list)
    usage: dict[str, object] = field(default_factory=dict)
    rejected: list[str] = field(default_factory=list)
