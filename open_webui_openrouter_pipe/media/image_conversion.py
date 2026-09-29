"""Image format / mode helpers shared across video + image features.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.url_scheme import media_type_or_empty

if TYPE_CHECKING:
    from PIL import Image as _Image  # type: ignore[import-untyped]


def normalise_mime(value: Any) -> str:
    """Lowercase, strip parameters: 'IMAGE/JPEG; charset=utf-8' -> 'image/jpeg'."""
    return media_type_or_empty(value)


def _carries_transparency(img: _Image.Image) -> bool:
    return "A" in img.getbands() or "transparency" in img.info


DEEP_SINGLE_BAND_MODES = ("I", "I;16", "I;16B", "I;16L")

_DEEP_RANGE_MAX = 65535
_DEEP_RANGE_SCALE = 255.0 / _DEEP_RANGE_MAX


def _rescale_deep_single_band(img: _Image.Image) -> _Image.Image:
    return img.convert("I").point(lambda v: v * _DEEP_RANGE_SCALE).convert("L")


def composite_on_white(img: _Image.Image) -> _Image.Image:
    from PIL import Image

    if img.mode == "RGB" and not _carries_transparency(img):
        return img
    if _carries_transparency(img):
        converted = img.convert("RGBA")
        background = Image.new("RGB", converted.size, (255, 255, 255))
        background.paste(converted, mask=converted.split()[3])
        return background
    if img.mode in DEEP_SINGLE_BAND_MODES:
        return _rescale_deep_single_band(img).convert("RGB")
    return img.convert("RGB")
