"""Image generation filter source code renderer.

Five filter variants — `generic`, `gemini`, `sourceful`, `recraft`, `recraft_v3` —
written to `body.image_config` as top-level request fields per OpenRouter's
[image-generation.md](.external/openrouter_docs/guides/overview/multimodal/image-generation.md).
The pipe's orchestrator injects `body.modalities` separately based on the
registered model's `architecture.output_modalities` so filters do not need
runtime registry access.

Filter assignment rules (driven by `filter_manager.ensure_openrouter_image_filter_function_ids`):
- **All** models with `image` in `output_modalities` get the **generic** filter
- Models matching `^google/gemini-.*flash-image.*-preview$` ALSO get **gemini** filter (extended aspect ratios + 0.5K)
- Models matching `^sourceful/riverflow-v\\d+(\\.\\d+)?-(pro|fast)$` ALSO get **sourceful** filter (font_inputs + super_resolution_references)
- Models matching `^recraft/recraft-` ALSO get **recraft** filter (strength + rgb_colors + background_rgb_color)
- Models matching `^recraft/recraft-v3$` ALSO get **recraft_v3** filter (style + text_layout — V3 only per OpenRouter docs)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..core.config import _OPENROUTER_IMAGE_FILTER_MARKER

_IMAGE_FILTER_ID_RE = re.compile(r"[^a-zA-Z0-9_]+")


@dataclass(frozen=True, slots=True)
class ImageFilterSpec:
    """Metadata for an image-generation filter variant."""

    variant: str  # "generic" | "gemini" | "sourceful" | "recraft" | "recraft_v3"
    function_id: str
    display_name: str
    marker: str


def sanitize_image_filter_id(variant: str) -> str:
    raw = (variant or "generic").strip().lower()
    cleaned = _IMAGE_FILTER_ID_RE.sub("_", raw).strip("_")
    if not cleaned:
        cleaned = "generic"
    if len(cleaned) > 30:
        suffix = hashlib.sha1(variant.encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned[:21].rstrip('_')}_{suffix}"
    return f"openrouter_image_filter_{cleaned}"


def build_generic_image_filter_spec() -> ImageFilterSpec:
    return ImageFilterSpec(
        variant="generic",
        function_id=sanitize_image_filter_id("generic"),
        display_name="OR Image Filter",
        marker=f"{_OPENROUTER_IMAGE_FILTER_MARKER}:generic",
    )


def build_gemini_image_filter_spec() -> ImageFilterSpec:
    return ImageFilterSpec(
        variant="gemini",
        function_id=sanitize_image_filter_id("gemini"),
        display_name="Gemini Options",
        marker=f"{_OPENROUTER_IMAGE_FILTER_MARKER}:gemini",
    )


def build_sourceful_image_filter_spec() -> ImageFilterSpec:
    return ImageFilterSpec(
        variant="sourceful",
        function_id=sanitize_image_filter_id("sourceful"),
        display_name="Sourceful Options",
        marker=f"{_OPENROUTER_IMAGE_FILTER_MARKER}:sourceful",
    )


def build_recraft_common_image_filter_spec() -> ImageFilterSpec:
    return ImageFilterSpec(
        variant="recraft",
        function_id=sanitize_image_filter_id("recraft"),
        display_name="Recraft Options",
        marker=f"{_OPENROUTER_IMAGE_FILTER_MARKER}:recraft",
    )


def build_recraft_v3_image_filter_spec() -> ImageFilterSpec:
    return ImageFilterSpec(
        variant="recraft_v3",
        function_id=sanitize_image_filter_id("recraft_v3"),
        display_name="Recraft V3 Extras",
        marker=f"{_OPENROUTER_IMAGE_FILTER_MARKER}:recraft_v3",
    )


def render_generic_image_filter_source() -> str:
    """Render the generic image filter — aspect_ratio (10 standard) + image_size (1K/2K/4K).

    Attached to ALL models with `image` in `output_modalities`. Inlet logic:
    1. Read user-valves
    2. Build image_config overrides dict
    3. Shallow-merge into body.image_config (overrides per-key; preserves any
       user-supplied keys not also set by this filter's UserValves)
    """
    spec = build_generic_image_filter_spec()
    return f'''"""OpenRouter image generation companion filter — generic."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # pragma: no cover - OWUI runtime only
    SRC_LOG_LEVELS = {{}}

OWUI_OPENROUTER_PIPE_MARKER = "{spec.marker}"
IMAGE_FILTER_VARIANT = "{spec.variant}"


class Filter:
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )

    class UserValves(BaseModel):
        IMAGE_ASPECT_RATIO: Literal[
            "", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
        ] = Field(
            default="",
            title="Image aspect ratio",
            description="Aspect ratio for generated images. Empty = model default.",
        )
        IMAGE_SIZE: Literal["", "1K", "2K", "4K"] = Field(
            default="",
            title="Image size",
            description="Image resolution tier. Empty = model default (1K).",
        )

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.image.filter.{spec.variant}")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    def inlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        if not isinstance(body, dict):
            return body
        user_valves = None
        if isinstance(__user__, dict):
            uv_raw = __user__.get("valves")
            if uv_raw is not None and not isinstance(uv_raw, self.UserValves):
                try:
                    user_valves = self.UserValves.model_validate(
                        uv_raw if isinstance(uv_raw, dict) else uv_raw.model_dump()
                    )
                except Exception:
                    user_valves = self.UserValves()
            elif isinstance(uv_raw, self.UserValves):
                user_valves = uv_raw
        if user_valves is None:
            user_valves = self.UserValves()

        overrides: dict = {{}}
        aspect = (user_valves.IMAGE_ASPECT_RATIO or "").strip()
        if aspect:
            overrides["aspect_ratio"] = aspect
        size = (user_valves.IMAGE_SIZE or "").strip()
        if size:
            overrides["image_size"] = size

        if overrides:
            existing = body.get("image_config")
            if not isinstance(existing, dict):
                existing = {{}}
            else:
                existing = dict(existing)
            existing.update(overrides)
            body["image_config"] = existing
        return body
'''


def render_gemini_image_filter_source() -> str:
    """Render the Gemini-specific image filter — extended aspect ratios + 0.5K size.

    Attached only to models matching `^google/gemini-.*flash-image.*-preview$`.
    Shallow-merges into body.image_config alongside the generic filter (per-key
    overwrite; if both filters write the same key, the second one wins).
    """
    spec = build_gemini_image_filter_spec()
    return f'''"""OpenRouter image generation companion filter — Gemini extensions."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # pragma: no cover
    SRC_LOG_LEVELS = {{}}

OWUI_OPENROUTER_PIPE_MARKER = "{spec.marker}"
IMAGE_FILTER_VARIANT = "{spec.variant}"

_GEMINI_MODEL_PATTERN = re.compile(r"^google/gemini-.*flash-image.*-preview$")


class Filter:
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(default=0)

    class UserValves(BaseModel):
        IMAGE_ASPECT_RATIO_EXTENDED: Literal["", "1:4", "4:1", "1:8", "8:1"] = Field(
            default="",
            title="Image aspect ratio (Gemini extended)",
            description="Gemini-only extended aspect ratios. Overrides the generic aspect_ratio when set. Empty = use generic filter's value.",
        )
        IMAGE_SIZE_GEMINI: Literal["", "0.5K"] = Field(
            default="",
            title="Image size (Gemini-only 0.5K)",
            description="Gemini Flash Image only — 0.5K low-res tier. Empty = use generic filter's value.",
        )

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.image.filter.{spec.variant}")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    def inlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        if not isinstance(body, dict):
            return body
        # Model gate: don't emit Gemini-specific knobs unless the model is a
        # Gemini Flash Image Preview variant. Defends against operator misconfiguration
        # (filter manually attached to non-Gemini model would otherwise send invalid params).
        model_id = body.get("model") or ""
        if not isinstance(model_id, str) or not _GEMINI_MODEL_PATTERN.match(model_id):
            return body
        user_valves = None
        if isinstance(__user__, dict):
            uv_raw = __user__.get("valves")
            if uv_raw is not None and not isinstance(uv_raw, self.UserValves):
                try:
                    user_valves = self.UserValves.model_validate(
                        uv_raw if isinstance(uv_raw, dict) else uv_raw.model_dump()
                    )
                except Exception:
                    user_valves = self.UserValves()
            elif isinstance(uv_raw, self.UserValves):
                user_valves = uv_raw
        if user_valves is None:
            user_valves = self.UserValves()

        overrides: dict = {{}}
        ext_aspect = (user_valves.IMAGE_ASPECT_RATIO_EXTENDED or "").strip()
        if ext_aspect:
            overrides["aspect_ratio"] = ext_aspect
        gemini_size = (user_valves.IMAGE_SIZE_GEMINI or "").strip()
        if gemini_size:
            overrides["image_size"] = gemini_size

        if overrides:
            existing = body.get("image_config")
            if not isinstance(existing, dict):
                existing = {{}}
            else:
                existing = dict(existing)
            existing.update(overrides)
            body["image_config"] = existing
        return body
'''


def render_sourceful_image_filter_source() -> str:
    """Render the Sourceful-specific image filter — font_inputs + super_resolution_references.

    Attached only to models matching `^sourceful/riverflow-v\\d+(\\.\\d+)?-(pro|fast)$`.
    Pre-validates cardinality caps (max 2 font_inputs, max 4 super_resolution_references)
    and rejects invalid input BEFORE submission so users get clear errors instead of
    cryptic provider 400s.
    """
    spec = build_sourceful_image_filter_spec()
    return f'''"""OpenRouter image generation companion filter — Sourceful extensions."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # pragma: no cover
    SRC_LOG_LEVELS = {{}}

OWUI_OPENROUTER_PIPE_MARKER = "{spec.marker}"
IMAGE_FILTER_VARIANT = "{spec.variant}"

_SOURCEFUL_MODEL_PATTERN = re.compile(r"^sourceful/riverflow-v\\d+(\\.\\d+)?-(pro|fast)$")
_MAX_FONT_INPUTS = 2
_MAX_SUPER_RESOLUTION_REFERENCES = 4


class ImageGenerationError(Exception):
    """Raised at inlet when Sourceful-specific limits or input validation fail
    (font_inputs cardinality > 2, super_resolution_references > 4, malformed
    JSON, missing required fields)."""


class Filter:
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(default=0)

    class UserValves(BaseModel):
        IMAGE_FONT_INPUTS_JSON: str = Field(
            default="",
            title="Font inputs (JSON array)",
            description=(
                'Sourceful-only font rendering. JSON array of objects: '
                '[{{"font_url": "https://...", "text": "..."}}]. Max 2 entries, +$0.03 each. '
                'Empty = none.'
            ),
        )
        IMAGE_SUPER_RESOLUTION_REFERENCES_JSON: str = Field(
            default="",
            title="Super-resolution references (JSON array)",
            description=(
                'Sourceful-only image-to-image super-resolution. JSON array of URL strings. '
                'Max 4 entries, +$0.20 each. Image-to-image only (requires input images in messages). '
                'Empty = none.'
            ),
        )

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.image.filter.{spec.variant}")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    def _parse_json_list(self, raw: str, field: str) -> list:
        cleaned = (raw or "").strip()
        if not cleaned:
            return []
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ImageGenerationError(
                f"{{field}} is not valid JSON: {{exc}}"
            )
        if not isinstance(parsed, list):
            raise ImageGenerationError(
                f"{{field}} must be a JSON array, got {{type(parsed).__name__}}"
            )
        return parsed

    def inlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        if not isinstance(body, dict):
            return body
        # Model gate: only emit Sourceful-specific knobs for Riverflow Pro/Fast
        # variants. Defends against operator misconfiguration (filter manually
        # attached to non-Sourceful model would otherwise emit invalid params).
        model_id = body.get("model") or ""
        if not isinstance(model_id, str) or not _SOURCEFUL_MODEL_PATTERN.match(model_id):
            return body
        user_valves = None
        if isinstance(__user__, dict):
            uv_raw = __user__.get("valves")
            if uv_raw is not None and not isinstance(uv_raw, self.UserValves):
                try:
                    user_valves = self.UserValves.model_validate(
                        uv_raw if isinstance(uv_raw, dict) else uv_raw.model_dump()
                    )
                except Exception:
                    user_valves = self.UserValves()
            elif isinstance(uv_raw, self.UserValves):
                user_valves = uv_raw
        if user_valves is None:
            user_valves = self.UserValves()

        overrides: dict = {{}}

        font_inputs = self._parse_json_list(
            user_valves.IMAGE_FONT_INPUTS_JSON, "IMAGE_FONT_INPUTS_JSON"
        )
        if font_inputs:
            if len(font_inputs) > _MAX_FONT_INPUTS:
                raise ImageGenerationError(
                    f"font_inputs has {{len(font_inputs)}} entries; max is {{_MAX_FONT_INPUTS}}."
                )
            for idx, entry in enumerate(font_inputs):
                if not isinstance(entry, dict):
                    raise ImageGenerationError(
                        f"font_inputs[{{idx}}] must be an object with 'font_url' and 'text', "
                        f"got {{type(entry).__name__}}."
                    )
                if not entry.get("font_url") or not entry.get("text"):
                    raise ImageGenerationError(
                        f"font_inputs[{{idx}}] requires non-empty 'font_url' and 'text'."
                    )
            overrides["font_inputs"] = font_inputs

        super_refs = self._parse_json_list(
            user_valves.IMAGE_SUPER_RESOLUTION_REFERENCES_JSON,
            "IMAGE_SUPER_RESOLUTION_REFERENCES_JSON",
        )
        if super_refs:
            if len(super_refs) > _MAX_SUPER_RESOLUTION_REFERENCES:
                raise ImageGenerationError(
                    f"super_resolution_references has {{len(super_refs)}} entries; max is {{_MAX_SUPER_RESOLUTION_REFERENCES}}."
                )
            for idx, entry in enumerate(super_refs):
                if not isinstance(entry, str) or not entry.strip():
                    raise ImageGenerationError(
                        f"super_resolution_references[{{idx}}] must be a non-empty URL string."
                    )
            overrides["super_resolution_references"] = super_refs

        if overrides:
            existing = body.get("image_config")
            if not isinstance(existing, dict):
                existing = {{}}
            else:
                existing = dict(existing)
            existing.update(overrides)
            body["image_config"] = existing
        return body
'''


def render_recraft_common_image_filter_source() -> str:
    """Render the Recraft common image filter — strength + rgb_colors + background_rgb_color.

    Attached to all models matching `^recraft/recraft-` (V3, V4, V4 Pro). Validates
    JSON shapes and RGB component ranges (0-255 ints) BEFORE submission so users
    get clear errors instead of cryptic provider 400s.
    """
    spec = build_recraft_common_image_filter_spec()
    return f'''"""OpenRouter image generation companion filter — Recraft common."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # pragma: no cover
    SRC_LOG_LEVELS = {{}}

OWUI_OPENROUTER_PIPE_MARKER = "{spec.marker}"
IMAGE_FILTER_VARIANT = "{spec.variant}"

_RECRAFT_MODEL_PATTERN = re.compile(r"^recraft/recraft-")


class ImageGenerationError(Exception):
    """Raised at inlet when Recraft-specific input validation fails (malformed
    JSON, out-of-range RGB components, wrong array shape)."""


class Filter:
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(default=0)

    class UserValves(BaseModel):
        IMAGE_STRENGTH: float = Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            title="Strength (image-to-image)",
            description=(
                "Recraft only. 0.0-1.0 — controls how much the output deviates "
                "from input image during image-to-image. 0.0 = use model default "
                "(0.5). Lower = closer to input; higher = more creative."
            ),
        )
        IMAGE_RGB_COLORS_JSON: str = Field(
            default="",
            title="RGB color palette (JSON array)",
            description=(
                'Recraft only. JSON array of [r,g,b] arrays (each 0-255). '
                'Example: [[255,0,0],[0,128,0]]. Empty = no palette hint.'
            ),
        )
        IMAGE_BACKGROUND_RGB_JSON: str = Field(
            default="",
            title="Background RGB color (JSON array)",
            description=(
                'Recraft only. Single [r,g,b] array (each 0-255). '
                'Example: [0,0,255]. Empty = no override.'
            ),
        )

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.image.filter.{spec.variant}")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    def _parse_json(self, raw: str, field: str) -> Any:
        cleaned = (raw or "").strip()
        if not cleaned:
            return None
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ImageGenerationError(f"{{field}} is not valid JSON: {{exc}}")

    def _validate_rgb_triple(self, value: Any, field: str) -> list:
        if not isinstance(value, (list, tuple)):
            raise ImageGenerationError(
                f"{{field}} must be a 3-element [r,g,b] array, got {{type(value).__name__}}."
            )
        if len(value) != 3:
            raise ImageGenerationError(
                f"{{field}} must be a 3-element [r,g,b] array, got {{len(value)}} elements."
            )
        for idx, c in enumerate(value):
            if isinstance(c, bool) or not isinstance(c, int):
                raise ImageGenerationError(
                    f"{{field}}[{{idx}}] must be an integer 0-255, got {{type(c).__name__}}."
                )
            if c < 0 or c > 255:
                raise ImageGenerationError(
                    f"{{field}}[{{idx}}] must be 0-255, got {{c}}."
                )
        return [int(c) for c in value]

    def inlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        if not isinstance(body, dict):
            return body
        # Model gate: only emit Recraft-specific knobs for recraft/recraft-* models.
        # Defends against operator misconfiguration (filter manually attached to
        # non-Recraft model would otherwise emit invalid params).
        model_id = body.get("model") or ""
        if not isinstance(model_id, str) or not _RECRAFT_MODEL_PATTERN.match(model_id):
            return body
        user_valves = None
        if isinstance(__user__, dict):
            uv_raw = __user__.get("valves")
            if uv_raw is not None and not isinstance(uv_raw, self.UserValves):
                try:
                    user_valves = self.UserValves.model_validate(
                        uv_raw if isinstance(uv_raw, dict) else uv_raw.model_dump()
                    )
                except Exception:
                    user_valves = self.UserValves()
            elif isinstance(uv_raw, self.UserValves):
                user_valves = uv_raw
        if user_valves is None:
            user_valves = self.UserValves()

        overrides: dict = {{}}

        # 0.0 is the skip sentinel — users wanting actual 0.0 strength can use
        # 0.001 (visually identical effect; same convention as VIDEO_SEED).
        if user_valves.IMAGE_STRENGTH > 0.0:
            overrides["strength"] = float(user_valves.IMAGE_STRENGTH)

        rgb_raw = self._parse_json(user_valves.IMAGE_RGB_COLORS_JSON, "IMAGE_RGB_COLORS_JSON")
        if rgb_raw is not None:
            if not isinstance(rgb_raw, list):
                raise ImageGenerationError(
                    f"IMAGE_RGB_COLORS_JSON must be a JSON array of [r,g,b] arrays, got {{type(rgb_raw).__name__}}."
                )
            validated_rgbs = []
            for idx, entry in enumerate(rgb_raw):
                validated_rgbs.append(self._validate_rgb_triple(entry, f"IMAGE_RGB_COLORS_JSON[{{idx}}]"))
            if validated_rgbs:
                overrides["rgb_colors"] = validated_rgbs

        bg_raw = self._parse_json(user_valves.IMAGE_BACKGROUND_RGB_JSON, "IMAGE_BACKGROUND_RGB_JSON")
        if bg_raw is not None:
            overrides["background_rgb_color"] = self._validate_rgb_triple(bg_raw, "IMAGE_BACKGROUND_RGB_JSON")

        if overrides:
            existing = body.get("image_config")
            if not isinstance(existing, dict):
                existing = {{}}
            else:
                existing = dict(existing)
            existing.update(overrides)
            body["image_config"] = existing
        return body
'''


def render_recraft_v3_image_filter_source() -> str:
    """Render the Recraft V3-only extras filter — style + text_layout.

    Attached only to `recraft/recraft-v3` exactly. V4 and V4 Pro do NOT support
    these parameters per OpenRouter docs; the filter no-ops on those models even
    if manually attached (defensive).
    """
    spec = build_recraft_v3_image_filter_spec()
    return f'''"""OpenRouter image generation companion filter — Recraft V3 extras."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # pragma: no cover
    SRC_LOG_LEVELS = {{}}

OWUI_OPENROUTER_PIPE_MARKER = "{spec.marker}"
IMAGE_FILTER_VARIANT = "{spec.variant}"

_RECRAFT_V3_MODEL_PATTERN = re.compile(r"^recraft/recraft-v3$")


class ImageGenerationError(Exception):
    """Raised at inlet when Recraft V3 input validation fails (malformed JSON,
    bbox out of 0-1 range, wrong array shape)."""


class Filter:
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(default=0)

    class UserValves(BaseModel):
        IMAGE_RECRAFT_STYLE: str = Field(
            default="",
            title="Recraft style",
            description=(
                'Recraft V3 only. Artistic style preset name, e.g. "Photorealism". '
                'See https://www.recraft.ai/docs/api-reference/styles for the full '
                'list. Vector styles are NOT supported. Empty = no style override.'
            ),
        )
        IMAGE_TEXT_LAYOUT_JSON: str = Field(
            default="",
            title="Text layout (JSON array)",
            description=(
                'Recraft V3 only. JSON array of objects with `text` (str) and '
                '`bbox` (4 [x,y] corners in 0-1 normalized coords, order TL, TR, '
                'BR, BL). Example: '
                '[{{"text":"Hello","bbox":[[0.3,0.45],[0.6,0.45],[0.6,0.55],[0.3,0.55]]}}]. '
                'Empty = no text overlay.'
            ),
        )

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.image.filter.{spec.variant}")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    def _parse_json_list(self, raw: str, field: str) -> list:
        cleaned = (raw or "").strip()
        if not cleaned:
            return []
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ImageGenerationError(f"{{field}} is not valid JSON: {{exc}}")
        if not isinstance(parsed, list):
            raise ImageGenerationError(
                f"{{field}} must be a JSON array, got {{type(parsed).__name__}}."
            )
        return parsed

    def _validate_text_layout(self, entries: list) -> list:
        validated = []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ImageGenerationError(
                    f"IMAGE_TEXT_LAYOUT_JSON[{{idx}}] must be an object with 'text' and 'bbox', got {{type(entry).__name__}}."
                )
            text = entry.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ImageGenerationError(
                    f"IMAGE_TEXT_LAYOUT_JSON[{{idx}}].text must be a non-empty string."
                )
            bbox = entry.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ImageGenerationError(
                    f"IMAGE_TEXT_LAYOUT_JSON[{{idx}}].bbox must be a 4-element array of [x,y] points (TL,TR,BR,BL)."
                )
            for pt_idx, pt in enumerate(bbox):
                if not isinstance(pt, list) or len(pt) != 2:
                    raise ImageGenerationError(
                        f"IMAGE_TEXT_LAYOUT_JSON[{{idx}}].bbox[{{pt_idx}}] must be a 2-element [x,y] array."
                    )
                for axis_idx, c in enumerate(pt):
                    if isinstance(c, bool) or not isinstance(c, (int, float)):
                        raise ImageGenerationError(
                            f"IMAGE_TEXT_LAYOUT_JSON[{{idx}}].bbox[{{pt_idx}}][{{axis_idx}}] must be a number 0.0-1.0."
                        )
                    if c < 0.0 or c > 1.0:
                        raise ImageGenerationError(
                            f"IMAGE_TEXT_LAYOUT_JSON[{{idx}}].bbox[{{pt_idx}}][{{axis_idx}}] must be 0.0-1.0, got {{c}}."
                        )
            validated.append({{"text": text, "bbox": bbox}})
        return validated

    def inlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        if not isinstance(body, dict):
            return body
        # Model gate: only emit V3-specific knobs for recraft/recraft-v3 EXACTLY.
        # V4/V4 Pro silently no-op (per OpenRouter docs they don't support style
        # or text_layout, so even if filter is manually attached to them we drop
        # the params instead of sending invalid input).
        model_id = body.get("model") or ""
        if not isinstance(model_id, str) or not _RECRAFT_V3_MODEL_PATTERN.match(model_id):
            return body
        user_valves = None
        if isinstance(__user__, dict):
            uv_raw = __user__.get("valves")
            if uv_raw is not None and not isinstance(uv_raw, self.UserValves):
                try:
                    user_valves = self.UserValves.model_validate(
                        uv_raw if isinstance(uv_raw, dict) else uv_raw.model_dump()
                    )
                except Exception:
                    user_valves = self.UserValves()
            elif isinstance(uv_raw, self.UserValves):
                user_valves = uv_raw
        if user_valves is None:
            user_valves = self.UserValves()

        overrides: dict = {{}}

        style = (user_valves.IMAGE_RECRAFT_STYLE or "").strip()
        if style:
            overrides["style"] = style

        text_layout_raw = self._parse_json_list(
            user_valves.IMAGE_TEXT_LAYOUT_JSON, "IMAGE_TEXT_LAYOUT_JSON"
        )
        if text_layout_raw:
            overrides["text_layout"] = self._validate_text_layout(text_layout_raw)

        if overrides:
            existing = body.get("image_config")
            if not isinstance(existing, dict):
                existing = {{}}
            else:
                existing = dict(existing)
            existing.update(overrides)
            body["image_config"] = existing
        return body
'''
