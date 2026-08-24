"""
title: OR Image Gen
author: Open-WebUI-OpenRouter-pipe
author_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: openrouter_image_gen
description: Configures OpenRouter image generation for the OpenRouter pipe.
version: 0.1.0
license: MIT
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # noqa: BLE001 - open_webui.env does filesystem work on import
    SRC_LOG_LEVELS = {}

OWUI_OPENROUTER_PIPE_MARKER = 'openrouter_pipe:image_gen_filter:v1'


class Filter:
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )
        IMAGE_GENERATION_MODEL: str = Field(
            default='openai/gpt-5-image-mini',
            title="Image generation model",
            description="Which OpenRouter model draws the picture. openai/gpt-5-image-mini is not in the image model list this pipe has loaded, so the settings below are not its own: each one offers what OpenRouter's image API accepts in general, and this model decides what to do with the value. Check the id if that is unexpected.",
        )
        IMAGE_GENERATION_MODERATION: Literal['auto', 'low'] = Field(
            default='auto',
            title="Image moderation",
            description='How strictly the company running this model screens what it will draw.',
        )

    class UserValves(BaseModel):
        @model_validator(mode="before")
        @classmethod
        def _keep_what_still_fits(cls, data: Any) -> Any:
            """Drop stored values the model no longer publishes, keep the rest.

            These fields track a live contract, so a provider joining the model can
            narrow a range or remove a ratio while a value the user chose earlier is
            still stored. Open WebUI builds this class from that stored dict and passes
            no valves at all if construction raises -- so one stale entry silently threw
            away every other choice the user had made.
            """
            if not isinstance(data, dict):
                return data
            kept = {}
            for name, field in cls.model_fields.items():
                if name not in data:
                    continue
                annotated = (
                    Annotated[(field.annotation, *field.metadata)]
                    if field.metadata
                    else field.annotation
                )
                try:
                    TypeAdapter(annotated).validate_python(data[name])
                except ValidationError:
                    continue
                kept[name] = data[name]
            return kept

        IMAGE_QUALITY: str = Field(
                    default="",
                    title='Quality',
                    description="Rendering quality tier. This model publishes no preference of its own. OpenRouter's image API takes one of auto, low, medium, high here and refuses anything else before the company running the model sees it. Empty leaves it unset.",
                )
        IMAGE_SIZE: str = Field(
                    default="",
                    title='Output size',
                    description='Either a size tier (512, 1K, 2K or 4K) or exact pixels written like 1024x1024. This model publishes no tiers of its own, so a tier is checked only against those four names and then goes out for the company running the model to interpret. It still takes its shape from Aspect ratio. Exact pixels settle the picture on their own, so Aspect ratio is not sent alongside them unless it is the shape you typed. A toast says so at the time, which Open WebUI does not keep with the message: it is gone once the page reloads. No model publishes a list of pixel sizes, so exact pixels go out as typed and the company running this one decides what to do with them. Empty leaves it unset.',
                )
        IMAGE_ASPECT_RATIO: str = Field(
                    default="",
                    title='Aspect ratio',
                    description="Frame shape. This model publishes no preference of its own. OpenRouter's image API takes one of 1:1, 1:2, 1:4, 1:8, 2:1, 2:3, 3:2, 3:4, 4:1, 4:3, 4:5, 5:4, 8:1, 9:16, 16:9, 9:19.5, 19.5:9, 9:20, 20:9, 9:21, 21:9, auto here and refuses anything else before the company running the model sees it. Empty leaves it unset.",
                )
        IMAGE_BACKGROUND: str = Field(
                    default="",
                    title='Background',
                    description="Background treatment. This model publishes no preference of its own. OpenRouter's image API takes one of auto, transparent, opaque here and refuses anything else before the company running the model sees it. Empty leaves it unset.",
                )
        IMAGE_OUTPUT_FORMAT: str = Field(
                    default="",
                    title='Output format',
                    description="Container the image comes back in. This model publishes no preference of its own. OpenRouter's image API takes one of png, jpeg, webp, svg here and refuses anything else before the company running the model sees it. Empty leaves it unset.",
                )
        IMAGE_OUTPUT_COMPRESSION: int | None = Field(
                    default=None,
                    ge=0,
                    le=100,
                    title='Output compression',
                    description="Compression level, where the format allows one. This model publishes no limits of its own. OpenRouter's image API takes a whole number from 0 to 100 here, and the company running the model decides what it does with it. Empty leaves it unset.",
                )

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.image.gen")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    def inlet(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
        __model__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(body, dict):
            return body

        user_valves = None
        if isinstance(__user__, dict):
            stored = __user__.get("valves")
            if isinstance(stored, self.UserValves):
                user_valves = stored
            elif stored is not None:
                try:
                    user_valves = self.UserValves.model_validate(
                        stored if isinstance(stored, dict) else stored.model_dump()
                    )
                except Exception:  # noqa: BLE001 - a stored valve must not block the turn
                    user_valves = self.UserValves()
        if user_valves is None:
            user_valves = self.UserValves()

        params: dict[str, Any] = {"model": self.valves.IMAGE_GENERATION_MODEL}
        if self.valves.IMAGE_GENERATION_MODERATION != 'auto':
            params["moderation"] = self.valves.IMAGE_GENERATION_MODERATION
        wanted = (user_valves.IMAGE_QUALITY or "").strip()
        if wanted:
            params['quality'] = wanted
        wanted = (user_valves.IMAGE_SIZE or "").strip()
        if wanted:
            params['size'] = wanted
        wanted = (user_valves.IMAGE_ASPECT_RATIO or "").strip()
        if wanted:
            params['aspect_ratio'] = wanted
        wanted = (user_valves.IMAGE_BACKGROUND or "").strip()
        if wanted:
            params['background'] = wanted
        wanted = (user_valves.IMAGE_OUTPUT_FORMAT or "").strip()
        if wanted:
            params['output_format'] = wanted
        measure = user_valves.IMAGE_OUTPUT_COMPRESSION
        if measure is not None:
            params['output_compression'] = int(measure)

        if isinstance(__metadata__, dict):
            prev_pipe_meta = __metadata__.get('openrouter_pipe')
            pipe_meta = dict(prev_pipe_meta) if isinstance(prev_pipe_meta, dict) else {}
            __metadata__['openrouter_pipe'] = pipe_meta

            prev_tools = pipe_meta.get("server_tools")
            server_tools = dict(prev_tools) if isinstance(prev_tools, dict) else {}
            pipe_meta["server_tools"] = server_tools
            server_tools["image_generation"] = params

        return body
