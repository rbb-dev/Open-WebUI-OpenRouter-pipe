from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from ..requests.fusion_engine import item_text, latest_user_text
from ..storage.multimodal import image_extension_for_mime


def capability_declared_off(value: Any) -> bool:
    return value is not None and value is not True

RENDERABLE_FIELD_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,48}")
"""A published name may become a form field only if it can be a Python identifier.

One pattern for both renderers: a name one accepts and the other rejects would mean a
setting reachable on video and not on image, for no reason a user could discover.
"""

PASSTHROUGH_DESCRIPTION = (
    "A setting this model's provider accepts. Type a plain value, or JSON if it takes a "
    "list or an object. Empty leaves it unset."
)
"""One sentence for one encoding rule, read by both renderers."""

PROVIDER_OPTIONS_DESCRIPTION = (
    "Extra settings for the company that runs this model, as a JSON object keyed by its "
    "OpenRouter name. Use it for anything this panel does not already offer. Empty sends "
    "nothing."
)

TOP_LEVEL_PARAMS: tuple[str, ...] = (
    "aspect_ratio",
    "resolution",
    "size",
    "n",
    "seed",
    "quality",
    "background",
    "output_format",
    "output_compression",
)
"""Parameter names the Image API takes at the top level of the request.

Read by both the adapter, which routes a key here or into the provider block, and the
filter renderer, which decides whether to render a control for it. Two copies of this
tuple had already drifted -- one carried ``size`` and the other did not, so a model
publishing ``size`` was accepted by the adapter and offered no control.
"""

SYSTEM_PROMPT_ROLES = frozenset({"system", "developer"})

SCHEMA_ONLY_PARAMS: tuple[str, ...] = ("size",)

CONTRACT_GATED_PARAMS: tuple[str, ...] = tuple(
    name for name in TOP_LEVEL_PARAMS if name not in SCHEMA_ONLY_PARAMS
)

SCHEMA_ENUMS: dict[str, tuple[str, ...]] = {
    "aspect_ratio": (
        "1:1", "1:2", "1:4", "1:8", "2:1", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5",
        "5:4", "8:1", "9:16", "16:9", "9:19.5", "19.5:9", "9:20", "20:9", "9:21",
        "21:9", "auto",
    ),
    "background": ("auto", "transparent", "opaque"),
    "output_format": ("png", "jpeg", "webp", "svg"),
    "quality": ("auto", "low", "medium", "high"),
    "resolution": ("512", "1K", "2K", "4K"),
}


SCHEMA_RANGES: dict[str, tuple[int, int]] = {
    "output_compression": (0, 100),
}


PASSTHROUGH_ENUMS: dict[str, tuple[tuple[str, ...], str]] = {
    "moderation": (
        ("auto", "low"),
        "How strictly the company running this model screens what it will draw.",
    ),
}


def system_prompt_text(input_items: Any) -> str:
    if not isinstance(input_items, list):
        return ""
    system = [
        text.strip()
        for item in input_items
        if isinstance(item, dict)
        and item.get("role") in SYSTEM_PROMPT_ROLES
        and isinstance(text := item_text(item), str)
        and text.strip()
    ]
    return "\n\n".join(system)


def prompt_with_system(input_items: Any) -> str:
    user = latest_user_text(input_items)
    if not user.strip():
        return user
    system = system_prompt_text(input_items)
    return f"{system}\n\n{user}" if system else user


def json_encodable(value: Any) -> bool:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, float) and not math.isfinite(item):
            return False
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return True


def pixel_size(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    parts = value.strip().casefold().split("x")
    if len(parts) != 2 or not all(p.isascii() and p.isdigit() for p in parts):
        return None
    width, height = int(parts[0]), int(parts[1])
    return (width, height) if width > 0 and height > 0 else None


def reduced_ratio(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    parts = value.strip().split(":")
    if len(parts) != 2 or not all(p.isascii() and p.isdigit() for p in parts):
        return None
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        return None
    common = math.gcd(width, height)
    return width // common, height // common


def supersede_size_conflicts(params: dict[str, Any]) -> list[tuple[str, Any, str]]:
    size = params.get("size")
    if size is None:
        return []
    pixels = pixel_size(size)
    if pixels is None:
        resolution = params.get("resolution")
        if resolution is None or str(resolution).strip().casefold() == str(size).strip().casefold():
            return []
        return [("resolution", params.pop("resolution"), "sets the same thing")]
    dropped: list[tuple[str, Any, str]] = []
    if "resolution" in params:
        dropped.append(
            ("resolution", params.pop("resolution"), "already fixes the output dimensions")
        )
    ratio = reduced_ratio(params.get("aspect_ratio"))
    if ratio is not None and ratio != reduced_ratio(f"{pixels[0]}:{pixels[1]}"):
        dropped.append(("aspect_ratio", params.pop("aspect_ratio"), "is not that shape"))
    return dropped


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


@dataclass(frozen=True, slots=True)
class ImageGenerationResult:
    images: list[GeneratedImage] = field(default_factory=list)
    usage: dict[str, object] = field(default_factory=dict)
    rejected: list[str] = field(default_factory=list)
