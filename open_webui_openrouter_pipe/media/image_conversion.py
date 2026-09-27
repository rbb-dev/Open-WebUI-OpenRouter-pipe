"""Image format / mode helpers shared across video + image features.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image as _Image  # type: ignore[import-untyped]


def normalise_mime(value: Any) -> str:
    """Lowercase, strip parameters: 'IMAGE/JPEG; charset=utf-8' -> 'image/jpeg'."""
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text.split(";", 1)[0].strip().lower()


def _carries_transparency(img: _Image.Image) -> bool:
    return "A" in img.getbands() or "transparency" in img.info


def composite_on_white(img: _Image.Image) -> _Image.Image:
    from PIL import Image

    if img.mode == "RGB" and not _carries_transparency(img):
        return img
    if _carries_transparency(img):
        converted = img.convert("RGBA")
        background = Image.new("RGB", converted.size, (255, 255, 255))
        background.paste(converted, mask=converted.split()[3])
        return background
    return img.convert("RGB")
