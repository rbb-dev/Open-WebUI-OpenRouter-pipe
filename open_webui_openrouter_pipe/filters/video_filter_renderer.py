from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from ..core.config import _OPENROUTER_VIDEO_GEN_FILTER_MARKER, _PIPE_METADATA_KEY
from ..core.utils import OWUI_FUNCTION_ID_ILLEGAL_RE as _FILTER_ID_RE
from ..core.utils import _clean_str
from ..integrations.image_types import (
    PASSTHROUGH_DESCRIPTION,
    RENDERABLE_FIELD_NAME_RE,
    capability_declared_off,
    scrub_surrogates,
)

logger = logging.getLogger(__name__)

_LITERAL_VALUE_RE = re.compile(r"^[a-zA-Z0-9:._ -]{1,64}$")

_CONTROL_ENUM = "enum"
_CONTROL_TOGGLE = "toggle"
_CONTROL_NUMBER = "number"
_CONTROL_TEXT = "text"

_CLOSED_DOMAIN_CONTROLS: frozenset[str] = frozenset(
    {_CONTROL_ENUM, _CONTROL_TOGGLE, _CONTROL_NUMBER}
)

_UNCONFIRMED_DOMAIN = "unconfirmed: no vendor page publishes a domain for this parameter"

_TOGGLE_VALUES: tuple[str, ...] = ("model_default", "on", "off")

_VEO_GEMINI_API = "https://ai.google.dev/gemini-api/docs/veo"
_VEO_VERTEX_TEXT_TO_VIDEO = (
    "https://docs.cloud.google.com/vertex-ai/generative-ai/docs/video/generate-videos-from-text"
)
_VEO_VERTEX_PROMPT_REWRITER = (
    "https://docs.cloud.google.com/vertex-ai/generative-ai/docs/video/turn-the-prompt-rewriter-off"
)
_MINIMAX_TEXT_TO_VIDEO = "https://platform.minimax.io/docs/api-reference/video-generation-t2v"
_WAN_DASHSCOPE_TEXT_TO_VIDEO = (
    "https://www.alibabacloud.com/help/en/model-studio/text-to-video-api-reference"
)
_WAN_FAL_2_6_IMAGE_TO_VIDEO = "https://fal.ai/models/wan/v2.6/image-to-video/api"
_SEEDANCE_ARK_TASKS = "https://www.volcengine.com/docs/82379/1520757"
_KLING_LEGACY_IMAGE_TO_VIDEO = (
    "https://app.klingai.com/cn/dev/document-api/api/video/3-0-omni/image-to-video/legacy"
)


@dataclass(frozen=True, slots=True)
class _PassthroughControl:

    param: str
    field: str
    title: str
    description: str
    kind: str
    choices: tuple[tuple[str, str], ...] = ()
    minimum: float = 0.0
    maximum: float = 0.0
    source: str = ""


_PASSTHROUGH_CONTROLS: tuple[_PassthroughControl, ...] = (
    _PassthroughControl(
        param="personGeneration",
        field="VIDEO_PERSON_GENERATION",
        title="Person generation",
        description=(
            "Veo policy for human subjects: allow_all, allow_adult, dont_allow (Gemini API "
            "spelling), disallow (Vertex AI spelling), or model default."
        ),
        kind=_CONTROL_ENUM,
        choices=(
            ("allow_all", _VEO_GEMINI_API),
            ("allow_adult", _VEO_GEMINI_API),
            ("dont_allow", _VEO_GEMINI_API),
            ("disallow", _VEO_VERTEX_TEXT_TO_VIDEO),
        ),
    ),
    _PassthroughControl(
        param="conditioningScale",
        field="VIDEO_CONDITIONING_SCALE",
        title="Conditioning scale",
        description="Strength of frame/reference conditioning (0 leaves the model default).",
        kind=_CONTROL_NUMBER,
        minimum=0.0,
        maximum=1.0,
        source=_UNCONFIRMED_DOMAIN,
    ),
    _PassthroughControl(
        param="cfg_scale",
        field="VIDEO_CFG_SCALE",
        title="CFG scale",
        description=(
            "Classifier-free guidance strength (0 leaves the provider default; higher values "
            "bias toward stricter prompt adherence)."
        ),
        kind=_CONTROL_NUMBER,
        minimum=0.0,
        maximum=1.0,
        source=_KLING_LEGACY_IMAGE_TO_VIDEO,
    ),
    _PassthroughControl(
        param="enhancePrompt",
        field="VIDEO_ENHANCE_PROMPT",
        title="Enhance prompt",
        description="Veo prompt-rewriter for richer scenes (on/off/model default).",
        kind=_CONTROL_TOGGLE,
        source=_VEO_VERTEX_PROMPT_REWRITER,
    ),
    _PassthroughControl(
        param="prompt_optimizer",
        field="VIDEO_PROMPT_OPTIMIZER",
        title="Prompt optimizer",
        description="Provider-side prompt rewriter (on/off/model default).",
        kind=_CONTROL_TOGGLE,
        source=_MINIMAX_TEXT_TO_VIDEO,
    ),
    _PassthroughControl(
        param="fast_pretreatment",
        field="VIDEO_FAST_PRETREATMENT",
        title="Fast pretreatment",
        description="Hailuo fast input preprocessing (on/off/model default).",
        kind=_CONTROL_TOGGLE,
        source=_MINIMAX_TEXT_TO_VIDEO,
    ),
    _PassthroughControl(
        param="prompt_extend",
        field="VIDEO_PROMPT_EXTEND",
        title="Prompt extend",
        description="Wan prompt-extension toggle (on/off/model default).",
        kind=_CONTROL_TOGGLE,
        source=_WAN_DASHSCOPE_TEXT_TO_VIDEO,
    ),
    _PassthroughControl(
        param="ratio",
        field="VIDEO_RATIO",
        title="Ratio",
        description="Wan-specific ratio passthrough; leave blank to use model default.",
        kind=_CONTROL_TEXT,
    ),
    _PassthroughControl(
        param="enable_prompt_expansion",
        field="VIDEO_ENABLE_PROMPT_EXPANSION",
        title="Enable prompt expansion",
        description="Wan 2.6 prompt expansion toggle (on/off/model default).",
        kind=_CONTROL_TOGGLE,
        source=_WAN_FAL_2_6_IMAGE_TO_VIDEO,
    ),
    _PassthroughControl(
        param="shot_type",
        field="VIDEO_SHOT_TYPE",
        title="Shot type",
        description="Wan camera/composition shot type passthrough; blank = model default.",
        kind=_CONTROL_TEXT,
    ),
    _PassthroughControl(
        param="watermark",
        field="VIDEO_WATERMARK",
        title="Watermark",
        description="Seedance watermark toggle (on/off/model default).",
        kind=_CONTROL_TOGGLE,
        source=_SEEDANCE_ARK_TASKS,
    ),
    _PassthroughControl(
        param="req_key",
        field="VIDEO_REQ_KEY",
        title="Request key",
        description="Seedance provider req_key passthrough; blank = model default.",
        kind=_CONTROL_TEXT,
    ),
    _PassthroughControl(
        param="quality",
        field="VIDEO_QUALITY",
        title="Quality",
        description=(
            "Sora quality passthrough; OpenAI's video API publishes no quality values, so "
            "send only one your provider accepts. Blank = model default."
        ),
        kind=_CONTROL_TEXT,
    ),
    _PassthroughControl(
        param="style",
        field="VIDEO_STYLE",
        title="Style",
        description="Sora style passthrough; blank = model default.",
        kind=_CONTROL_TEXT,
    ),
)


def _is_citation(source: str) -> bool:
    return source == _UNCONFIRMED_DOMAIN or source.startswith("https://")


def _validate_passthrough_controls(controls: tuple[_PassthroughControl, ...]) -> None:
    seen: set[str] = set()
    for control in controls:
        if control.param in seen:
            raise ValueError(f"{control.param!r} is declared twice in the passthrough table")
        seen.add(control.param)
        if control.kind not in _CLOSED_DOMAIN_CONTROLS:
            if control.choices or control.source:
                raise ValueError(
                    f"{control.param!r} renders as free text, which asserts nothing about its "
                    "values, so it must carry no citation"
                )
            continue
        if control.kind == _CONTROL_ENUM:
            if not control.choices:
                raise ValueError(f"{control.param!r} declares an enum with no values")
            cited = tuple(source for _, source in control.choices)
        else:
            if control.choices:
                raise ValueError(f"{control.param!r} may not declare enum values")
            cited = (control.source,)
        for source in cited:
            if not _is_citation(source):
                raise ValueError(
                    f"{control.param!r} renders a closed value domain, so it must name the "
                    f"document that domain was read from; got {source!r}"
                )


_validate_passthrough_controls(_PASSTHROUGH_CONTROLS)

_UNCONFIRMED_PASSTHROUGH_DOMAINS: frozenset[str] = frozenset(
    control.param
    for control in _PASSTHROUGH_CONTROLS
    if _UNCONFIRMED_DOMAIN in (control.source, *(source for _, source in control.choices))
)

_HANDLED_PASSTHROUGH_PARAMS: frozenset[str] = frozenset({
    "negative_prompt",
    "negativePrompt",
    "audio",
    "video",
    "videos",
    "images",
    "last_image",
    "aspectRatio",
    "size",
}) | frozenset(control.param for control in _PASSTHROUGH_CONTROLS)


@dataclass(frozen=True, slots=True)
class VideoFilterSpec:

    model_id: str
    display_name: str
    function_id: str
    marker: str
    allowed_params: tuple[str, ...]
    aspect_ratios: tuple[str, ...]
    durations: tuple[int, ...]
    resolutions: tuple[str, ...]
    frame_types: tuple[str, ...]
    size_options: tuple[str, ...]
    seed_capable: bool = False
    audio_capable: bool = False
    intent_classifier_admin_enabled: bool = True
    intent_enabled_default: bool = True
    intent_max_clarifications_default: int = 1
    intent_frame_default: str = "last"
    intent_confirm_mode_default: str = "on_reference"

    @property
    def supports_frames(self) -> bool:
        return bool(self.frame_types)

    @property
    def supports_first_last(self) -> bool:
        return "first_frame" in self.frame_types and "last_frame" in self.frame_types

    @property
    def supports_seed(self) -> bool:
        return self.seed_capable

    @property
    def supports_negative_prompt(self) -> bool:
        return any(param in self.allowed_params for param in ("negative_prompt", "negativePrompt"))

    @property
    def supports_generate_audio_toggle(self) -> bool:
        return self.audio_capable

    @property
    def supports_audio_reference(self) -> bool:
        return "audio" in self.allowed_params

    @property
    def accepts_video_single(self) -> bool:
        return "video" in self.allowed_params

    @property
    def accepts_videos_array(self) -> bool:
        return "videos" in self.allowed_params

    @property
    def accepts_video_attachment(self) -> bool:
        return self.accepts_video_single or self.accepts_videos_array

    @property
    def supports_wan_reference_arrays(self) -> bool:
        return any(param in self.allowed_params for param in ("images", "videos", "video", "last_image"))


def sanitize_video_filter_id(model_id: str) -> str:
    raw = model_id.strip()
    cleaned = _FILTER_ID_RE.sub("_", raw).lower()
    if not cleaned:
        cleaned = "model"
    if len(cleaned) > 54:
        suffix = hashlib.sha1(scrub_surrogates(model_id).encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned[:45].rstrip('_')}_{suffix}"
    return f"openrouter_video_{cleaned}"


def _strip_vendor_prefix(name: str) -> str:
    if ":" in name:
        return name.split(":", 1)[1].strip()
    return name.strip()


_VALID_FRAME_DEFAULTS = ("first", "last")
_VALID_CONFIRM_MODES = ("always", "on_reference", "low_confidence", "never")


def build_video_filter_spec(
    model_id: str,
    video_model: dict[str, Any] | None,
    admin_valves: Any = None,
) -> VideoFilterSpec:
    model = video_model if isinstance(video_model, dict) else {}
    canonical_id = _clean_str(model.get("id")) or _clean_str(model_id)
    raw_name = _clean_str(model.get("name")) or canonical_id
    display_name = _strip_vendor_prefix(raw_name)
    function_id = sanitize_video_filter_id(canonical_id)
    allowed_params = _string_tuple(model.get("allowed_passthrough_parameters"))
    aspect_ratios = _safe_literal_tuple(model.get("supported_aspect_ratios"))
    durations = _int_tuple(model.get("supported_durations"))
    resolutions = _safe_literal_tuple(model.get("supported_resolutions"))
    frame_types = _safe_literal_tuple(model.get("supported_frame_images"))
    size_options = _safe_literal_tuple(model.get("supported_sizes") or model.get("supported_size_options"))

    intent_admin_enabled = True
    intent_enabled_default = True
    intent_max_clar = 1
    intent_frame = "last"
    intent_confirm = "on_reference"
    if admin_valves is not None:
        intent_admin_enabled = bool(getattr(admin_valves, "VIDEO_INTENT_ENABLED", True))
        intent_enabled_default = intent_admin_enabled
        raw_max_clar = getattr(admin_valves, "VIDEO_INTENT_MAX_CLARIFICATIONS", 1)
        if isinstance(raw_max_clar, int) and 0 <= raw_max_clar <= 3:
            intent_max_clar = raw_max_clar
        raw_frame = getattr(admin_valves, "VIDEO_INTENT_FRAME_EXTRACTION_INDEX", "last")
        if isinstance(raw_frame, str) and raw_frame in _VALID_FRAME_DEFAULTS:
            intent_frame = raw_frame
        raw_confirm = getattr(admin_valves, "VIDEO_INTENT_CONFIRM_MODE", "on_reference")
        if isinstance(raw_confirm, str) and raw_confirm in _VALID_CONFIRM_MODES:
            intent_confirm = raw_confirm

    return VideoFilterSpec(
        model_id=canonical_id,
        display_name=display_name,
        function_id=function_id,
        marker=f"{_OPENROUTER_VIDEO_GEN_FILTER_MARKER}:{function_id}",
        allowed_params=allowed_params,
        aspect_ratios=aspect_ratios,
        durations=durations,
        resolutions=resolutions,
        frame_types=frame_types,
        size_options=size_options,
        seed_capable="seed" in model and not capability_declared_off(model.get("seed")),
        audio_capable="generate_audio" in model
        and not capability_declared_off(model.get("generate_audio")),
        intent_classifier_admin_enabled=intent_admin_enabled,
        intent_enabled_default=intent_enabled_default,
        intent_max_clarifications_default=intent_max_clar,
        intent_frame_default=intent_frame,
        intent_confirm_mode_default=intent_confirm,
    )


def render_video_filter_source(
    *,
    model_id: str,
    video_model: dict[str, Any] | None,
    pipe_metadata_key: str = _PIPE_METADATA_KEY,
    admin_valves: Any = None,
) -> str:
    from open_webui_openrouter_pipe import __version__

    spec = build_video_filter_spec(model_id, video_model, admin_valves=admin_valves)
    # Only the names that genuinely cannot be offered: one that is not a legal Python
    # identifier has no field to carry it. Everything else the model publishes is
    # rendered, typed where a purpose-built control exists and free text otherwise.
    unreachable = sorted(
        set(spec.allowed_params)
        - _HANDLED_PASSTHROUGH_PARAMS
        - set(_unhandled_params(spec))
        - _purpose_built_published_names(spec)
    )
    if unreachable:
        logger.warning(
            "Model %r publishes parameter(s) %s whose names cannot become form fields, so "
            "they are not offered. Every other parameter it publishes is.",
            spec.model_id,
            unreachable,
        )
    user_valves_fields = _render_user_valves_fields(spec)
    inlet_param_lines = _render_param_lines(spec)
    frame_block = _render_frame_block(spec)
    intent_inlet_block = (
        _render_intent_inlet_block()
        if spec.intent_classifier_admin_enabled
        else ""
    )
    source = f'''"""OpenRouter video generation companion filter."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:
    SRC_LOG_LEVELS = {{}}


OPENROUTER_PIPE_MARKER = {spec.marker!r}
OPENROUTER_PIPE_VERSION = {__version__!r}
VIDEO_MODEL_ID = {spec.model_id!r}
PIPE_METADATA_KEY = {pipe_metadata_key!r}


class VideoFilterInputError(ValueError):
    """A value the user typed that this filter will not put on the wire."""


def _json_number(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{{text}} is out of range for JSON")
    return value


def _json_constant(literal: str) -> float:
    raise ValueError(f"{{literal}} is not valid JSON")


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )

    class UserValves(BaseModel):
{user_valves_fields}

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.video.gen")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return None
        return None

    @staticmethod
    def _deep_merge_pipe_provider(existing: Any, video_options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(existing) if isinstance(existing, dict) else {{}}
        current_options = merged.get("options")
        merged_options = dict(current_options) if isinstance(current_options, dict) else {{}}
        for slug, payload in video_options.items():
            if isinstance(slug, str) and slug.strip() and isinstance(payload, dict):
                merged_options[slug.strip()] = payload
        if merged_options:
            merged["options"] = merged_options
        return merged

    @staticmethod
    def _file_id(item: dict[str, Any]) -> str:
        raw = item.get("id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        nested = item.get("file")
        if isinstance(nested, dict):
            raw = nested.get("id")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return ""

    @staticmethod
    def _content_type(item: dict[str, Any]) -> str:
        for key in ("content_type", "contentType", "mime_type", "mimeType", "type"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        nested = item.get("file")
        if isinstance(nested, dict):
            for key in ("content_type", "contentType", "mime_type", "mimeType"):
                value = nested.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip().lower()
        return ""

    @staticmethod
    def _owui_skips_file_context(__model__: Any) -> bool:
        """Whether Open WebUI will leave a returned file list alone.

        `file_context` defaults to True, and on True Open WebUI answers an
        attachment-bearing request with a `generate_queries` round-trip and RAG injection
        -- into a video prompt. Handing the files back is only safe once the model's own
        capability says otherwise, so an installation whose capability write never ran
        keeps today's behaviour instead.
        """
        if not isinstance(__model__, dict):
            return False
        info = __model__.get("info")
        meta = info.get("meta") if isinstance(info, dict) else None
        caps = meta.get("capabilities") if isinstance(meta, dict) else None
        return isinstance(caps, dict) and caps.get("file_context") is False

    def _build_attachment(self, item: dict[str, Any]) -> dict[str, Any]:
        return {{
            "id": self._file_id(item),
            "name": item.get("name") or "",
            "size": self._to_int(item.get("size")) or 0,
            "content_type": self._content_type(item),
        }}

    @staticmethod
    def _decode(raw: str, field: str) -> Any:
        """Parse a passthrough value without inventing a type the user did not write.

        The same rule the image filter applies, for the same reason: a JSON container and
        a bare number are parsed, everything else stays the string it is, and ``NaN``,
        ``Infinity`` and ``1e400`` are refused because no JSON encoder can put the float
        they produce on the wire.
        """
        container = raw[:1] in ("[", "{{")
        try:
            parsed = json.loads(raw, parse_float=_json_number, parse_constant=_json_constant)
        except ValueError as exc:
            if container:
                raise VideoFilterInputError(f"{{field}} is not valid JSON: {{exc}}") from exc
            return raw
        if container:
            return parsed
        if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
            return raw
        return parsed

    @staticmethod
    def _json_object(value: Any) -> dict[str, Any]:
        if not isinstance(value, str) or not value.strip():
            return {{}}
        try:
            parsed = json.loads(value, parse_float=_json_number, parse_constant=_json_constant)
        except ValueError as exc:
            raise VideoFilterInputError(
                f"Provider options JSON is not valid JSON: {{exc}}"
            ) from exc
        if not isinstance(parsed, dict):
            raise VideoFilterInputError(
                "Video provider options JSON must be an object keyed by provider slug."
            )
        return parsed

    @staticmethod
    def _json_array(value: Any, label: str) -> list[Any]:
        if not isinstance(value, str) or not value.strip():
            return []
        try:
            parsed = json.loads(value, parse_float=_json_number, parse_constant=_json_constant)
        except ValueError as exc:
            raise VideoFilterInputError(f"{{label}} is not valid JSON: {{exc}}") from exc
        if not isinstance(parsed, list):
            raise VideoFilterInputError(f"{{label}} must be a JSON array.")
        return parsed

    def inlet(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
        __model__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(body, dict):
            return body
        if __metadata__ is not None and not isinstance(__metadata__, dict):
            return body
        if __user__ is not None and not isinstance(__user__, dict):
            __user__ = None

        user_valves = None
        if isinstance(__user__, dict):
            user_valves = __user__.get("valves")
        if not isinstance(user_valves, BaseModel):
            user_valves = self.UserValves()

        params: dict[str, Any] = {{}}
{inlet_param_lines}

        provider_options: dict[str, Any] = {{}}
        raw_provider_json = getattr(user_valves, "VIDEO_PROVIDER_OPTIONS_JSON", "")
        parsed_provider_options = self._json_object(raw_provider_json)
        for slug, payload in parsed_provider_options.items():
            if isinstance(slug, str) and slug.strip() and isinstance(payload, dict):
                provider_options[slug.strip()] = payload

        frame_images: list[dict[str, Any]] = []
{frame_block}

        if isinstance(__metadata__, dict):
            prev_pipe_meta = __metadata__.get(PIPE_METADATA_KEY)
            pipe_meta = dict(prev_pipe_meta) if isinstance(prev_pipe_meta, dict) else {{}}
            __metadata__[PIPE_METADATA_KEY] = pipe_meta

            video_meta = dict(pipe_meta.get("video_generation")) if isinstance(pipe_meta.get("video_generation"), dict) else {{}}
            video_meta["model_id"] = VIDEO_MODEL_ID
            video_meta["params"] = params
            if frame_images:
                video_meta["frame_images"] = frame_images
            else:
                video_meta.pop("frame_images", None)
            if video_attachments:
                video_meta["video_attachments"] = video_attachments
            else:
                video_meta.pop("video_attachments", None)
            if audio_attachments:
                video_meta["audio_attachments"] = audio_attachments
            else:
                video_meta.pop("audio_attachments", None)
            if input_references:
                video_meta["input_references"] = input_references
            else:
                video_meta.pop("input_references", None)
            pipe_meta["video_generation"] = video_meta

{intent_inlet_block}
            if provider_options:
                pipe_meta["provider"] = self._deep_merge_pipe_provider(pipe_meta.get("provider"), provider_options)

        return body
'''
    return source


def _unhandled_params(spec: VideoFilterSpec) -> tuple[str, ...]:
    """Published parameters with no purpose-built control.

    OpenRouter names these and publishes nothing about their values -- no choices, no
    bounds -- so there is nothing to type them from. They are rendered as free text
    rather than dropped, because a parameter the model accepts and the filter refuses to
    offer is a capability the user simply cannot reach.

    The exclusion is on the rendered field name, not the published spelling. A published
    ``duration`` and the purpose-built duration control both want ``VIDEO_DURATION``, and
    the later definition wins -- so a free-text box would silently replace the dropdown
    built from the model's own published values.
    """
    taken = set(_purpose_built_field_names(spec))
    accepted: list[str] = []
    for name in spec.allowed_params:
        if name in _HANDLED_PASSTHROUGH_PARAMS or not RENDERABLE_FIELD_NAME_RE.fullmatch(name):
            continue
        field = f"VIDEO_{name.upper()}"
        if field in taken:
            continue
        taken.add(field)
        accepted.append(name)
    return tuple(accepted)


def _render_purpose_built_fields(spec: VideoFilterSpec) -> list[str]:
    fields = [
        _field_block(
            'VIDEO_PROVIDER_OPTIONS_JSON: str = Field(\n'
            '            default="",\n'
            '            title="Provider options JSON",\n'
            '            description="Provider-specific video options keyed by provider slug.",\n'
            "        )"
        )
    ]
    if spec.durations:
        literals = ", ".join(["0", *(str(item) for item in spec.durations)])
        fields.append(
            _field_block(
                f"VIDEO_DURATION: Literal[{literals}] = Field(\n"
                "            default=0,\n"
                '            title="Duration",\n'
                '            description="Video duration. 0 uses the model default.",\n'
                "        )"
            )
        )
    if spec.aspect_ratios:
        literals = _literal_union(("", *spec.aspect_ratios))
        fields.append(
            _field_block(
                f"VIDEO_ASPECT_RATIO: Literal[{literals}] = Field(\n"
                '            default="",\n'
                '            title="Aspect ratio",\n'
                '            description="Supported aspect ratio for this model.",\n'
                "        )"
            )
        )
    if spec.resolutions:
        literals = _literal_union(("", *spec.resolutions))
        fields.append(
            _field_block(
                f"VIDEO_RESOLUTION: Literal[{literals}] = Field(\n"
                '            default="",\n'
                '            title="Resolution",\n'
                '            description="Supported output resolution for this model.",\n'
                "        )"
            )
        )
    if spec.size_options:
        literals = _literal_union(("", *spec.size_options))
        fields.append(
            _field_block(
                f"VIDEO_SIZE: Literal[{literals}] = Field(\n"
                '            default="",\n'
                '            title="Size",\n'
                '            description="Supported provider size value for this model.",\n'
                "        )"
            )
        )
    if spec.supports_generate_audio_toggle:
        fields.append(
            _field_block(
                'VIDEO_GENERATE_AUDIO: Literal["model_default", "on", "off"] = Field(\n'
                '            default="model_default",\n'
                '            title="Audio",\n'
                '            description="Request generated audio where this provider exposes a toggle.",\n'
                "        )"
            )
        )
    if spec.supports_audio_reference:
        fields.append(
            _field_block(
                'VIDEO_AUDIO_URL: str = Field(\n'
                '            default="",\n'
                '            title="Audio reference URL",\n'
                '            description="Provider-supported audio reference URL for this model.",\n'
                "        )"
            )
        )
    if spec.supports_seed:
        fields.append(
            _field_block(
                "VIDEO_SEED: int = Field(\n"
                "            default=0,\n"
                "            ge=0,\n"
                '            title="Seed",\n'
                '            description="Deterministic seed. 0 uses the model default.",\n'
                "        )"
            )
        )
    if spec.supports_negative_prompt:
        fields.append(
            _field_block(
                'VIDEO_NEGATIVE_PROMPT: str = Field(\n'
                '            default="",\n'
                '            title="Negative prompt",\n'
                '            description="Things the model should avoid where supported.",\n'
                "        )"
            )
        )
    if spec.supports_frames:
        modes = ["auto", "none", "first_only"]
        if spec.supports_first_last:
            modes.append("first_last")
        literals = _literal_union(tuple(modes))
        fields.append(
            _field_block(
                f"VIDEO_FRAME_MODE: Literal[{literals}] = Field(\n"
                '            default="auto",\n'
                '            title="Frames",\n'
                '            description="How attached images are used as video frame references.",\n'
                "        )"
            )
        )
    if "video" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_REFERENCE_VIDEO_URL: str = Field(\n'
                '            default="",\n'
                '            title="Reference video URL",\n'
                '            description="Provider-supported single reference video URL.",\n'
                "        )"
            )
        )
    if "videos" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_REFERENCE_VIDEOS_JSON: str = Field(\n'
                '            default="",\n'
                '            title="Reference videos JSON",\n'
                '            description="JSON array of provider-supported reference video URLs or objects.",\n'
                "        )"
            )
        )
    if "images" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_REFERENCE_IMAGES_JSON: str = Field(\n'
                '            default="",\n'
                '            title="Reference images JSON",\n'
                '            description="JSON array of provider-supported reference image URLs or objects.",\n'
                "        )"
            )
        )
    if "last_image" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_LAST_IMAGE_URL: str = Field(\n'
                '            default="",\n'
                '            title="Last image URL",\n'
                '            description="Provider-supported final image URL.",\n'
                "        )"
            )
        )
    for control in _PASSTHROUGH_CONTROLS:
        if control.param in spec.allowed_params:
            fields.append(_control_field_block(control))

    if spec.intent_classifier_admin_enabled:
        fields.append(
            _field_block(
                'VIDEO_INTENT_ENABLED: bool = Field(\n'
                f'            default={spec.intent_enabled_default!r},\n'
                '            title="Reuse previous videos",\n'
                '            description=(\n'
                '                "When on, follow-up requests like \\"make it black\\" or '
                '\\"continue this scene\\" keep working with the previous video — you get the '
                'same scene with the change applied. When off, each video is generated only '
                'from your latest message and ignores everything that came before, so \\"make '
                'it black\\" would just create a new, unrelated video."\n'
                '            ),\n'
                '        )'
            )
        )
        fields.append(
            _field_block(
                'VIDEO_INTENT_MAX_CLARIFICATIONS: int = Field(\n'
                f'            default={spec.intent_max_clarifications_default},\n'
                '            ge=0,\n'
                '            le=3,\n'
                '            title="Clarifying question limit",\n'
                '            description=(\n'
                '                "When your request is unclear (for example, you have two '
                'previous videos and say \\"make the last one red\\"), the chat can ask a '
                'short clarifying question to pick the right one. This is how many such '
                'questions are allowed in a row before the chat just goes with its best '
                'guess. Set to 0 to skip questions entirely."\n'
                '            ),\n'
                '        )'
            )
        )
        fields.append(
            _field_block(
                'VIDEO_INTENT_FRAME_EXTRACTION_INDEX: Literal["first", "last"] = Field(\n'
                f'            default={spec.intent_frame_default!r},\n'
                '            title="Which frame to use from previous video",\n'
                '            description=(\n'
                '                "When the previous video is reused as a starting point, this '
                'is the frame taken from it. last = the final frame, used to continue the '
                'action from where it ended (the usual pick). first = the opening frame, '
                'used to restart the scene from how it began."\n'
                '            ),\n'
                '        )'
            )
        )
        fields.append(
            _field_block(
                'VIDEO_INTENT_CONFIRM_MODE: Literal["always", "on_reference", "low_confidence", "never"] = Field(\n'
                f'            default={spec.intent_confirm_mode_default!r},\n'
                '            title="Show what was reused",\n'
                '            description=(\n'
                '                "When a previous video or image is reused, the chat can show '
                'a small thumbnail confirming which one — so you can stop and retry if the '
                'wrong thing was picked. always = show for every video. on_reference '
                '= only show when something was actually reused. low_confidence = only '
                'show when the chat was unsure of its choice. never = hide entirely."\n'
                '            ),\n'
                '        )'
            )
        )
    return fields


_VIDEO_FIELD_DEF_RE = re.compile(r"^\s*(VIDEO_[A-Z0-9_]+)\s*:", re.MULTILINE)


def _purpose_built_published_names(spec: VideoFilterSpec) -> frozenset[str]:
    """Published names a purpose-built control already carries.

    Derived from the rendered field names, so a name is never reported as unreachable
    while a typed control for it is on screen.
    """
    emitted = _purpose_built_field_names(spec)
    return frozenset(
        name for name in spec.allowed_params if f"VIDEO_{name.upper()}" in emitted
    )


def _purpose_built_field_names(spec: VideoFilterSpec) -> frozenset[str]:
    """The field names the purpose-built controls actually emit for this model.

    Read back out of the rendered text rather than listed by hand, so a control added
    later cannot be shadowed by a free-text field of the same name without anyone
    noticing.
    """
    return frozenset(
        _VIDEO_FIELD_DEF_RE.findall("\n".join(_render_purpose_built_fields(spec)))
    )


def _render_user_valves_fields(spec: VideoFilterSpec) -> str:
    fields = _render_purpose_built_fields(spec)
    for name in _unhandled_params(spec):
        fields.append(
            _field_block(
                f"VIDEO_{name.upper()}: str = Field(\n"
                '            default="",\n'
                f"            title={name!r},\n"
                f"            description={PASSTHROUGH_DESCRIPTION!r},\n"
                "        )"
            )
        )
    return "\n".join(fields)


def _render_intent_inlet_block() -> str:
    """Inlet code that pushes user-set intent valves into request metadata.

    The pipe consumer (`integrations/video.py`, `integrations/video_intent.py`)
    reads ``__metadata__["openrouter_pipe"]["video_intent"]`` first and falls
    back to the admin Valves when no key is set, so emitting None for an unset
    user valve preserves the admin default.

    Always emitted alongside the four UserValves fields above (i.e. only when
    admin VIDEO_INTENT_ENABLED=True). When disabled, the consumer never sees
    a `video_intent` key and falls back to the admin valve which itself is
    False, short-circuiting the classifier.
    """
    return (
        "            intent_settings: dict[str, Any] = {}\n"
        "            for _user_field, _meta_key in (\n"
        '                ("VIDEO_INTENT_ENABLED", "enabled"),\n'
        '                ("VIDEO_INTENT_MAX_CLARIFICATIONS", "max_clarifications"),\n'
        '                ("VIDEO_INTENT_FRAME_EXTRACTION_INDEX", "frame_extraction_index"),\n'
        '                ("VIDEO_INTENT_CONFIRM_MODE", "confirm_mode"),\n'
        "            ):\n"
        '                _value = getattr(user_valves, _user_field, None)\n'
        '                if _value is not None:\n'
        '                    intent_settings[_meta_key] = _value\n'
        '            if intent_settings:\n'
        '                pipe_meta["video_intent"] = intent_settings\n'
    )


def _render_param_lines(spec: VideoFilterSpec) -> str:
    lines: list[str] = []
    if spec.durations:
        lines.extend(
            [
                '        duration = self._to_int(getattr(user_valves, "VIDEO_DURATION", 0))',
                '        if duration and duration > 0:',
                '            params["duration"] = duration',
            ]
        )
    if spec.aspect_ratios:
        lines.extend(
            [
                '        aspect_ratio = getattr(user_valves, "VIDEO_ASPECT_RATIO", "")',
                "        if isinstance(aspect_ratio, str) and aspect_ratio.strip():",
                '            params["aspect_ratio"] = aspect_ratio.strip()',
            ]
        )
    if spec.resolutions:
        lines.extend(
            [
                '        resolution = getattr(user_valves, "VIDEO_RESOLUTION", "")',
                "        if isinstance(resolution, str) and resolution.strip():",
                '            params["resolution"] = resolution.strip()',
            ]
        )
    if spec.size_options:
        lines.extend(
            [
                '        size = getattr(user_valves, "VIDEO_SIZE", "")',
                "        if isinstance(size, str) and size.strip():",
                '            params["size"] = size.strip()',
            ]
        )
    if spec.supports_generate_audio_toggle:
        lines.extend(
            [
                '        audio_toggle = getattr(user_valves, "VIDEO_GENERATE_AUDIO", "model_default")',
                '        if audio_toggle == "on":',
                '            params["generate_audio"] = True',
                '        elif audio_toggle == "off":',
                '            params["generate_audio"] = False',
            ]
        )
    if spec.supports_audio_reference:
        lines.extend(
            [
                '        audio_url = getattr(user_valves, "VIDEO_AUDIO_URL", "")',
                "        if isinstance(audio_url, str) and audio_url.strip():",
                '            params["audio"] = audio_url.strip()',
            ]
        )
    if spec.supports_seed:
        lines.extend(
            [
                '        seed = self._to_int(getattr(user_valves, "VIDEO_SEED", 0))',
                '        if seed and seed > 0:',
                '            params["seed"] = seed',
            ]
        )
    if spec.supports_negative_prompt:
        lines.extend(
            [
                '        negative_prompt = getattr(user_valves, "VIDEO_NEGATIVE_PROMPT", "")',
                "        if isinstance(negative_prompt, str) and negative_prompt.strip():",
                '            params["negative_prompt"] = negative_prompt.strip()',
            ]
        )
    if "video" in spec.allowed_params:
        lines.extend(
            [
                '        video_url = getattr(user_valves, "VIDEO_REFERENCE_VIDEO_URL", "")',
                "        if isinstance(video_url, str) and video_url.strip():",
                '            params["video"] = video_url.strip()',
            ]
        )
    if "videos" in spec.allowed_params:
        lines.extend(
            [
                '        reference_videos = self._json_array(getattr(user_valves, "VIDEO_REFERENCE_VIDEOS_JSON", ""), "Reference videos JSON")',
                "        if reference_videos:",
                '            params["videos"] = reference_videos',
            ]
        )
    if "images" in spec.allowed_params:
        lines.extend(
            [
                '        reference_images = self._json_array(getattr(user_valves, "VIDEO_REFERENCE_IMAGES_JSON", ""), "Reference images JSON")',
                "        if reference_images:",
                '            params["images"] = reference_images',
            ]
        )
    if "last_image" in spec.allowed_params:
        lines.extend(
            [
                '        last_image_url = getattr(user_valves, "VIDEO_LAST_IMAGE_URL", "")',
                "        if isinstance(last_image_url, str) and last_image_url.strip():",
                '            params["last_image"] = last_image_url.strip()',
            ]
        )
    for control in _PASSTHROUGH_CONTROLS:
        if control.param in spec.allowed_params:
            lines.extend(_control_param_lines(control))
    for name in _unhandled_params(spec):
        lines.extend([
            f'        raw_value = getattr(user_valves, "VIDEO_{name.upper()}", "")',
            "        if isinstance(raw_value, str) and raw_value.strip():",
            f"            params[{name!r}] = self._decode(raw_value.strip(), {name!r})",
        ])
    return "\n".join(lines) if lines else "        pass"


def _render_frame_block(spec: VideoFilterSpec) -> str:
    has_frames = spec.supports_frames
    has_first_last = spec.supports_first_last
    accepts_video_single = spec.accepts_video_single
    accepts_videos_array = spec.accepts_videos_array
    accepts_audio_attachment = spec.supports_audio_reference

    frame_mode_line = ""
    if has_frames:
        frame_mode_line = '        frame_mode = getattr(user_valves, "VIDEO_FRAME_MODE", "auto")\n'

    image_select_block = ""
    if has_frames:
        supported_literal = "{" + ", ".join(repr(f) for f in spec.frame_types) + "}"
        if has_first_last:
            image_select_block = f'''            supported_frames = {supported_literal}
            selected: list[tuple[dict[str, Any], str]] = []
            if frame_mode != "none" and image_items and "first_frame" in supported_frames:
                if frame_mode == "first_only":
                    selected.append((image_items[0], "first_frame"))
                else:
                    selected.append((image_items[0], "first_frame"))
                    if len(image_items) > 1 and "last_frame" in supported_frames:
                        selected.append((image_items[-1], "last_frame"))
            for item, frame_type in selected:
                claimed_ids.add(self._file_id(item))
                frame_images.append(
                    {{
                        "id": self._file_id(item),
                        "name": item.get("name") or "",
                        "size": self._to_int(item.get("size")) or 0,
                        "content_type": self._content_type(item),
                        "frame_type": frame_type,
                    }}
                )'''
        else:
            image_select_block = '''            if frame_mode != "none" and image_items:
                item = image_items[0]
                claimed_ids.add(self._file_id(item))
                frame_images.append(
                    {
                        "id": self._file_id(item),
                        "name": item.get("name") or "",
                        "size": self._to_int(item.get("size")) or 0,
                        "content_type": self._content_type(item),
                        "frame_type": "first_frame",
                    }
                )'''

    video_select_block = ""
    if accepts_video_single or accepts_videos_array:
        emit_video_single = "True" if accepts_video_single else "False"
        emit_videos_array = "True" if accepts_videos_array else "False"
        video_select_block = f'''            if video_items:
                accepts_single = {emit_video_single}
                accepts_array = {emit_videos_array}
                if accepts_array and len(video_items) > 1:
                    taken = list(video_items)
                elif accepts_single or accepts_array:
                    taken = [video_items[0]]
                else:
                    taken = []
                for item in taken:
                    claimed_ids.add(self._file_id(item))
                    video_attachments.append(self._build_attachment(item))'''

    audio_select_block = ""
    if accepts_audio_attachment:
        audio_select_block = '''            if audio_items:
                claimed_ids.add(self._file_id(audio_items[0]))
                audio_attachments.append(self._build_attachment(audio_items[0]))'''

    reference_select_block = '''            for item in image_items + video_items + audio_items:
                if self._file_id(item) in claimed_ids:
                    continue
                claimed_ids.add(self._file_id(item))
                input_references.append(self._build_attachment(item))'''

    select_blocks = "\n".join(
        block
        for block in (
            image_select_block,
            video_select_block,
            audio_select_block,
            reference_select_block,
        )
        if block
    )

    return f'''        files = body.get("files")
        if not (isinstance(files, list) and files) and isinstance(__metadata__, dict):
            user_message = __metadata__.get("user_message")
            if isinstance(user_message, dict):
                files = user_message.get("files")
        retained: list[Any] = []
        claimed: list[Any] = []
        unclaimed: list[Any] = []
        video_attachments: list[dict[str, Any]] = []
        audio_attachments: list[dict[str, Any]] = []
        input_references: list[dict[str, Any]] = []
        claimed_ids: set[str] = set()
{frame_mode_line}        if isinstance(files, list) and files:
            image_items: list[dict[str, Any]] = []
            video_items: list[dict[str, Any]] = []
            audio_items: list[dict[str, Any]] = []
            for item in files:
                if not isinstance(item, dict):
                    retained.append(item)
                    continue
                content_type = self._content_type(item)
                file_id = self._file_id(item)
                if not file_id:
                    retained.append(item)
                    continue
                if content_type.startswith("image/"):
                    image_items.append(item)
                elif content_type.startswith("video/"):
                    video_items.append(item)
                elif content_type.startswith("audio/"):
                    audio_items.append(item)
                else:
                    retained.append(item)
{select_blocks}
            for item in image_items + video_items + audio_items:
                if self._file_id(item) in claimed_ids:
                    claimed.append(item)
                else:
                    unclaimed.append(item)

        if claimed_ids or retained or unclaimed:
            kept = retained + claimed if self._owui_skips_file_context(__model__) else retained
            body["files"] = kept
            if isinstance(__metadata__, dict):
                __metadata__["files"] = kept'''


def _field_block(text: str) -> str:
    return "\n".join(f"        {line}" if line else "" for line in text.splitlines())


def _control_field_block(control: _PassthroughControl) -> str:
    if control.kind == _CONTROL_ENUM:
        values = ("", *(value for value, _ in control.choices))
        annotation = f"Literal[{_quoted_union(values)}]"
        head = '            default="",\n'
    elif control.kind == _CONTROL_TOGGLE:
        annotation = f"Literal[{_quoted_union(_TOGGLE_VALUES)}]"
        head = '            default="model_default",\n'
    elif control.kind == _CONTROL_NUMBER:
        annotation = "float"
        head = (
            "            default=0.0,\n"
            f"            ge={control.minimum},\n"
            f"            le={control.maximum},\n"
        )
    else:
        annotation = "str"
        head = '            default="",\n'
    return _field_block(
        f"{control.field}: {annotation} = Field(\n"
        f"{head}"
        f"            title={json.dumps(control.title)},\n"
        f"            description={json.dumps(control.description)},\n"
        "        )"
    )


def _control_param_lines(control: _PassthroughControl) -> list[str]:
    local = control.field.removeprefix("VIDEO_").lower()
    key = json.dumps(control.param)
    if control.kind == _CONTROL_TOGGLE:
        return [
            f'        {local}_toggle = getattr(user_valves, "{control.field}", "model_default")',
            f'        if {local}_toggle == "on":',
            f"            params[{key}] = True",
            f'        elif {local}_toggle == "off":',
            f"            params[{key}] = False",
        ]
    if control.kind == _CONTROL_NUMBER:
        return [
            f'        {local}_raw = getattr(user_valves, "{control.field}", 0.0)',
            "        try:",
            f"            {local} = float({local}_raw)",
            "        except (TypeError, ValueError):",
            f"            {local} = 0.0",
            f"        if {local} > 0.0:",
            f"            params[{key}] = {local}",
        ]
    return [
        f'        {local} = getattr(user_valves, "{control.field}", "")',
        f"        if isinstance({local}, str) and {local}.strip():",
        f"            params[{key}] = {local}.strip()",
    ]


def _literal_union(values: tuple[str, ...]) -> str:
    return ", ".join(repr(value) for value in values)


def _quoted_union(values: tuple[str, ...]) -> str:
    return ", ".join(json.dumps(value) for value in values)


def _safe_literal_tuple(value: Any) -> tuple[str, ...]:
    values = _string_tuple(value)
    return tuple(item for item in values if _LITERAL_VALUE_RE.fullmatch(item))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    seen: set[str] = set()
    out: list[str] = []
    for item in value:
        text = _clean_str(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def _int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    seen: set[int] = set()
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            number = item
        elif isinstance(item, float) and item.is_integer():
            number = int(item)
        elif isinstance(item, str):
            try:
                number = int(item.strip())
            except ValueError:
                continue
        else:
            continue
        if number <= 0 or number in seen:
            continue
        seen.add(number)
        out.append(number)
    return tuple(out)


def spec_to_json(spec: VideoFilterSpec) -> str:
    return json.dumps(
        {
            "model_id": spec.model_id,
            "function_id": spec.function_id,
            "allowed_params": list(spec.allowed_params),
            "aspect_ratios": list(spec.aspect_ratios),
            "durations": list(spec.durations),
            "resolutions": list(spec.resolutions),
            "frame_types": list(spec.frame_types),
        },
        sort_keys=True,
    )
