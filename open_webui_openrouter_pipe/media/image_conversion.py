"""Image format / mode helpers shared across video + image features.

PIL pattern reuses `storage/multimodal.py:1616-1631` decode→re-encode flow.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

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


def composite_on_white(img: "_Image.Image") -> "_Image.Image":
    """Flatten alpha onto a white background; return RGB image.

    Always converts palette ("P") and alpha modes through RGBA before
    flattening, so palette PNGs/GIFs without explicit `info["transparency"]`
    don't silently produce wrong colors (F14).
    """
    from PIL import Image

    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA", "P", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        return background
    return img.convert("RGB")
