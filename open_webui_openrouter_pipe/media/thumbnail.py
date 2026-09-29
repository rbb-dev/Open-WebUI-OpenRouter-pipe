"""Thumbnail generation for the Intent Disclosure Block.

256x256 JPEG with letterbox-style aspect preservation. Compose RGBA on white.
Pure CPU-bound; caller wraps in to_thread when called from async context.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

from .image_conversion import composite_on_white

_MAX_INPUT_BYTES = 50 * 1024 * 1024
_MAX_INPUT_PIXELS = 50_000_000


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

    try:
        src = Image.open(io.BytesIO(image_bytes))
    except Image.DecompressionBombError as exc:
        raise ValueError(
            f"image is too large: exceeds {_MAX_INPUT_PIXELS} pixel cap"
        ) from exc
    if src.width * src.height > _MAX_INPUT_PIXELS:
        src.close()
        raise ValueError(
            f"image is too large: {src.width}x{src.height} exceeds "
            f"{_MAX_INPUT_PIXELS} pixel cap"
        )
    src.load()
    src = composite_on_white(src)

    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    scale = target_size / max(src.width, src.height)
    fitted = src.resize(
        (max(1, round(src.width * scale)), max(1, round(src.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas.paste(fitted, ((target_size - fitted.width) // 2, (target_size - fitted.height) // 2))

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=quality, optimize=True)
    return Thumbnail(
        image_bytes=buf.getvalue(),
        mime_type="image/jpeg",
        width=target_size,
        height=target_size,
    )
