from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from ..core.config import _OPENROUTER_VIDEO_GEN_FILTER_MARKER, _PIPE_METADATA_KEY
from ..core.utils import _clean_str

_FILTER_ID_RE = re.compile(r"[^a-zA-Z0-9_]+")
_LITERAL_VALUE_RE = re.compile(r"^[a-zA-Z0-9:._ -]{1,64}$")


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
    raw = model_id.strip().replace("/", "_").replace(".", "_").replace("-", "_")
    cleaned = _FILTER_ID_RE.sub("_", raw).strip("_").lower()
    if not cleaned:
        cleaned = "model"
    if len(cleaned) > 54:
        suffix = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned[:45].rstrip('_')}_{suffix}"
    return f"openrouter_video_{cleaned}"


def _strip_vendor_prefix(name: str) -> str:
    if ":" in name:
        return name.split(":", 1)[1].strip()
    return name.strip()


def build_video_filter_spec(model_id: str, video_model: dict[str, Any] | None) -> VideoFilterSpec:
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
        seed_capable=model.get("seed") is True,
        audio_capable=model.get("generate_audio") is True,
    )


def render_video_filter_source(
    *,
    model_id: str,
    video_model: dict[str, Any] | None,
    pipe_metadata_key: str = _PIPE_METADATA_KEY,
) -> str:
    spec = build_video_filter_spec(model_id, video_model)
    user_valves_fields = _render_user_valves_fields(spec)
    inlet_param_lines = _render_param_lines(spec)
    frame_block = _render_frame_block(spec)
    source = f'''"""OpenRouter video generation companion filter."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:
    SRC_LOG_LEVELS = {{}}


OPENROUTER_PIPE_MARKER = {spec.marker!r}
VIDEO_MODEL_ID = {spec.model_id!r}
PIPE_METADATA_KEY = {pipe_metadata_key!r}


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

    def _build_attachment(self, item: dict[str, Any]) -> dict[str, Any]:
        return {{
            "id": self._file_id(item),
            "name": item.get("name") or "",
            "size": self._to_int(item.get("size")) or 0,
            "content_type": self._content_type(item),
        }}

    @staticmethod
    def _json_object(value: Any) -> dict[str, Any]:
        if not isinstance(value, str) or not value.strip():
            return {{}}
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise Exception("Video provider options JSON must be an object keyed by provider slug.")
        return parsed

    @staticmethod
    def _json_array(value: Any, label: str) -> list[Any]:
        if not isinstance(value, str) or not value.strip():
            return []
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise Exception(f"{{label}} must be a JSON array.")
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
            pipe_meta["video_generation"] = video_meta

            if provider_options:
                pipe_meta["provider"] = self._deep_merge_pipe_provider(pipe_meta.get("provider"), provider_options)

        return body
'''
    return source


def _render_user_valves_fields(spec: VideoFilterSpec) -> str:
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
    if "personGeneration" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_PERSON_GENERATION: Literal["", "allow_all", "allow_adult", "dont_allow"] = Field(\n'
                '            default="",\n'
                '            title="Person generation",\n'
                '            description="Veo policy for human subjects: allow_all, allow_adult, dont_allow, or model default.",\n'
                "        )"
            )
        )
    if "conditioningScale" in spec.allowed_params:
        fields.append(
            _field_block(
                "VIDEO_CONDITIONING_SCALE: float = Field(\n"
                "            default=0.0,\n"
                "            ge=0.0,\n"
                "            le=1.0,\n"
                '            title="Conditioning scale",\n'
                '            description="Strength of frame/reference conditioning (0 leaves the model default).",\n'
                "        )"
            )
        )
    if "enhancePrompt" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_ENHANCE_PROMPT: Literal["model_default", "on", "off"] = Field(\n'
                '            default="model_default",\n'
                '            title="Enhance prompt",\n'
                '            description="Veo prompt-rewriter for richer scenes (on/off/model default).",\n'
                "        )"
            )
        )
    if "prompt_optimizer" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_PROMPT_OPTIMIZER: Literal["model_default", "on", "off"] = Field(\n'
                '            default="model_default",\n'
                '            title="Prompt optimizer",\n'
                '            description="Provider-side prompt rewriter (on/off/model default).",\n'
                "        )"
            )
        )
    if "fast_pretreatment" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_FAST_PRETREATMENT: Literal["model_default", "on", "off"] = Field(\n'
                '            default="model_default",\n'
                '            title="Fast pretreatment",\n'
                '            description="Hailuo fast input preprocessing (on/off/model default).",\n'
                "        )"
            )
        )
    if "prompt_extend" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_PROMPT_EXTEND: Literal["model_default", "on", "off"] = Field(\n'
                '            default="model_default",\n'
                '            title="Prompt extend",\n'
                '            description="Wan prompt-extension toggle (on/off/model default).",\n'
                "        )"
            )
        )
    if "ratio" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_RATIO: str = Field(\n'
                '            default="",\n'
                '            title="Ratio",\n'
                '            description="Wan-specific ratio passthrough; leave blank to use model default.",\n'
                "        )"
            )
        )
    if "enable_prompt_expansion" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_ENABLE_PROMPT_EXPANSION: Literal["model_default", "on", "off"] = Field(\n'
                '            default="model_default",\n'
                '            title="Enable prompt expansion",\n'
                '            description="Wan 2.6 prompt expansion toggle (on/off/model default).",\n'
                "        )"
            )
        )
    if "shot_type" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_SHOT_TYPE: str = Field(\n'
                '            default="",\n'
                '            title="Shot type",\n'
                '            description="Wan camera/composition shot type passthrough; blank = model default.",\n'
                "        )"
            )
        )
    if "watermark" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_WATERMARK: Literal["model_default", "on", "off"] = Field(\n'
                '            default="model_default",\n'
                '            title="Watermark",\n'
                '            description="Seedance watermark toggle (on/off/model default).",\n'
                "        )"
            )
        )
    if "req_key" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_REQ_KEY: str = Field(\n'
                '            default="",\n'
                '            title="Request key",\n'
                '            description="Seedance provider req_key passthrough; blank = model default.",\n'
                "        )"
            )
        )
    if "quality" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_QUALITY: Literal["", "standard", "hd"] = Field(\n'
                '            default="",\n'
                '            title="Quality",\n'
                '            description="Sora quality preset (standard, hd, or model default).",\n'
                "        )"
            )
        )
    if "style" in spec.allowed_params:
        fields.append(
            _field_block(
                'VIDEO_STYLE: str = Field(\n'
                '            default="",\n'
                '            title="Style",\n'
                '            description="Sora style passthrough; blank = model default.",\n'
                "        )"
            )
        )
    return "\n".join(fields)


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
    if "personGeneration" in spec.allowed_params:
        lines.extend(
            [
                '        person_generation = getattr(user_valves, "VIDEO_PERSON_GENERATION", "")',
                "        if isinstance(person_generation, str) and person_generation.strip():",
                '            params["personGeneration"] = person_generation.strip()',
            ]
        )
    if "conditioningScale" in spec.allowed_params:
        lines.extend(
            [
                '        conditioning_scale_raw = getattr(user_valves, "VIDEO_CONDITIONING_SCALE", 0.0)',
                "        try:",
                "            conditioning_scale = float(conditioning_scale_raw)",
                "        except (TypeError, ValueError):",
                "            conditioning_scale = 0.0",
                "        if conditioning_scale > 0.0:",
                '            params["conditioningScale"] = conditioning_scale',
            ]
        )
    if "enhancePrompt" in spec.allowed_params:
        lines.extend(
            [
                '        enhance_prompt_toggle = getattr(user_valves, "VIDEO_ENHANCE_PROMPT", "model_default")',
                '        if enhance_prompt_toggle == "on":',
                '            params["enhancePrompt"] = True',
                '        elif enhance_prompt_toggle == "off":',
                '            params["enhancePrompt"] = False',
            ]
        )
    if "prompt_optimizer" in spec.allowed_params:
        lines.extend(
            [
                '        prompt_optimizer_toggle = getattr(user_valves, "VIDEO_PROMPT_OPTIMIZER", "model_default")',
                '        if prompt_optimizer_toggle == "on":',
                '            params["prompt_optimizer"] = True',
                '        elif prompt_optimizer_toggle == "off":',
                '            params["prompt_optimizer"] = False',
            ]
        )
    if "fast_pretreatment" in spec.allowed_params:
        lines.extend(
            [
                '        fast_pretreatment_toggle = getattr(user_valves, "VIDEO_FAST_PRETREATMENT", "model_default")',
                '        if fast_pretreatment_toggle == "on":',
                '            params["fast_pretreatment"] = True',
                '        elif fast_pretreatment_toggle == "off":',
                '            params["fast_pretreatment"] = False',
            ]
        )
    if "prompt_extend" in spec.allowed_params:
        lines.extend(
            [
                '        prompt_extend_toggle = getattr(user_valves, "VIDEO_PROMPT_EXTEND", "model_default")',
                '        if prompt_extend_toggle == "on":',
                '            params["prompt_extend"] = True',
                '        elif prompt_extend_toggle == "off":',
                '            params["prompt_extend"] = False',
            ]
        )
    if "ratio" in spec.allowed_params:
        lines.extend(
            [
                '        ratio_value = getattr(user_valves, "VIDEO_RATIO", "")',
                "        if isinstance(ratio_value, str) and ratio_value.strip():",
                '            params["ratio"] = ratio_value.strip()',
            ]
        )
    if "enable_prompt_expansion" in spec.allowed_params:
        lines.extend(
            [
                '        prompt_expansion_toggle = getattr(user_valves, "VIDEO_ENABLE_PROMPT_EXPANSION", "model_default")',
                '        if prompt_expansion_toggle == "on":',
                '            params["enable_prompt_expansion"] = True',
                '        elif prompt_expansion_toggle == "off":',
                '            params["enable_prompt_expansion"] = False',
            ]
        )
    if "shot_type" in spec.allowed_params:
        lines.extend(
            [
                '        shot_type_value = getattr(user_valves, "VIDEO_SHOT_TYPE", "")',
                "        if isinstance(shot_type_value, str) and shot_type_value.strip():",
                '            params["shot_type"] = shot_type_value.strip()',
            ]
        )
    if "watermark" in spec.allowed_params:
        lines.extend(
            [
                '        watermark_toggle = getattr(user_valves, "VIDEO_WATERMARK", "model_default")',
                '        if watermark_toggle == "on":',
                '            params["watermark"] = True',
                '        elif watermark_toggle == "off":',
                '            params["watermark"] = False',
            ]
        )
    if "req_key" in spec.allowed_params:
        lines.extend(
            [
                '        req_key_value = getattr(user_valves, "VIDEO_REQ_KEY", "")',
                "        if isinstance(req_key_value, str) and req_key_value.strip():",
                '            params["req_key"] = req_key_value.strip()',
            ]
        )
    if "quality" in spec.allowed_params:
        lines.extend(
            [
                '        quality_value = getattr(user_valves, "VIDEO_QUALITY", "")',
                "        if isinstance(quality_value, str) and quality_value.strip():",
                '            params["quality"] = quality_value.strip()',
            ]
        )
    if "style" in spec.allowed_params:
        lines.extend(
            [
                '        style_value = getattr(user_valves, "VIDEO_STYLE", "")',
                "        if isinstance(style_value, str) and style_value.strip():",
                '            params["style"] = style_value.strip()',
            ]
        )
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
        supported_literal = repr(set(spec.frame_types))
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
                    for item in video_items:
                        video_attachments.append(self._build_attachment(item))
                elif accepts_single:
                    video_attachments.append(self._build_attachment(video_items[0]))
                elif accepts_array:
                    video_attachments.append(self._build_attachment(video_items[0]))'''

    audio_select_block = ""
    if accepts_audio_attachment:
        audio_select_block = '''            if audio_items:
                audio_attachments.append(self._build_attachment(audio_items[0]))'''

    select_blocks = "\n".join(b for b in (image_select_block, video_select_block, audio_select_block) if b)

    return f'''        files = body.get("files")
        if not (isinstance(files, list) and files) and isinstance(__metadata__, dict):
            user_message = __metadata__.get("user_message")
            if isinstance(user_message, dict):
                files = user_message.get("files")
        retained: list[Any] = []
        video_attachments: list[dict[str, Any]] = []
        audio_attachments: list[dict[str, Any]] = []
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

        body["files"] = retained
        if isinstance(__metadata__, dict):
            __metadata__["files"] = retained'''


def _field_block(text: str) -> str:
    return "\n".join(f"        {line}" if line else "" for line in text.splitlines())


def _literal_union(values: tuple[str, ...]) -> str:
    return ", ".join(repr(value) for value in values)


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
