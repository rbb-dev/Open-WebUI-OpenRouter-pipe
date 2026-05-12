"""Thumbnail generation for the Intent Disclosure Block.

256x256 JPEG with letterbox-style aspect preservation. Compose RGBA on white.
Pure CPU-bound; caller wraps in to_thread when called from async context.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

from .image_conversion import composite_on_white

Image.MAX_IMAGE_PIXELS = 25_000_000

_MAX_INPUT_BYTES = 50 * 1024 * 1024


@dataclass
class Thumbnail:
    image_bytes: bytes
    mime_type: str
    width: int
    height: int


def make_thumbnail(
    image_bytes: bytes,
    *,
    target_size: int = 256,
    quality: int = 85,
) -> Thumbnail:
    """Decode arbitrary image bytes via PIL, composite RGBA on white,
    resize to target_size x target_size with letterbox aspect preservation,
    re-encode as JPEG at the requested quality.
    Rejects oversized inputs (>50 MB) and decompression bombs.
    """
    if not image_bytes:
        raise ValueError("image_bytes is empty")
    if len(image_bytes) > _MAX_INPUT_BYTES:
        raise ValueError(f"image_bytes too large: {len(image_bytes)} > {_MAX_INPUT_BYTES}")
    if not (1 <= quality <= 95):
        raise ValueError(f"quality must be in 1..95, got {quality}")
    if target_size <= 0 or target_size > 4096:
        raise ValueError(f"target_size must be in 1..4096, got {target_size}")

    src = Image.open(io.BytesIO(image_bytes))
    src.load()
    src = composite_on_white(src)

    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    src.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    offset = (
        (target_size - src.width) // 2,
        (target_size - src.height) // 2,
    )
    canvas.paste(src, offset)

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=quality, optimize=True)
    return Thumbnail(
        image_bytes=buf.getvalue(),
        mime_type="image/jpeg",
        width=target_size,
        height=target_size,
    )
