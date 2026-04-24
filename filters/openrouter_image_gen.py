"""
title: OpenRouter Image Generation
author: Open-WebUI-OpenRouter-pipe
author_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: openrouter_image_gen
description: Configures OpenRouter image generation for the OpenRouter pipe.
version: 0.1.0
license: MIT
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from open_webui.env import SRC_LOG_LEVELS

OWUI_OPENROUTER_PIPE_MARKER = "openrouter_pipe:image_gen_filter:v1"


class Filter:
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )
        IMAGE_GENERATION_MODEL: str = Field(
            default="openai/gpt-image-1",
            title="Image generation model",
            description="OpenRouter model ID for image generation. Controls pricing and capabilities.",
        )
        IMAGE_GENERATION_MODERATION: Literal["auto", "low"] = Field(
            default="auto",
            title="Image moderation",
            description="Content moderation level. 'auto' = standard. 'low' = reduced filtering.",
        )

    class UserValves(BaseModel):
        IMAGE_GENERATION: bool = Field(
            default=False,
            title="Image Generation",
            description="Let the model generate images from text prompts. Incurs additional cost per image.",
        )
        IMAGE_QUALITY: Literal["", "low", "medium", "high"] = Field(
            default="",
            title="Image quality",
            description="Quality level for generated images. Empty = model default.",
        )
        IMAGE_SIZE: Literal["", "1024x1024", "1536x1024", "1024x1536", "512x512"] = Field(
            default="",
            title="Image size",
            description="Image dimensions. Empty = model default.",
        )
        IMAGE_ASPECT_RATIO: Literal["", "1:1", "16:9", "4:3", "3:2"] = Field(
            default="",
            title="Image aspect ratio",
            description="Aspect ratio for generated images. Empty = model default.",
        )
        IMAGE_BACKGROUND: Literal["", "transparent", "opaque"] = Field(
            default="",
            title="Image background",
            description="'transparent' removes background (PNG only). Empty = model default.",
        )
        IMAGE_OUTPUT_FORMAT: Literal["", "png", "jpeg", "webp"] = Field(
            default="",
            title="Image format",
            description="Output format. 'png' supports transparency. Empty = model default.",
        )
        IMAGE_OUTPUT_COMPRESSION: int = Field(
            default=0,
            ge=0,
            le=100,
            title="Image compression",
            description="Compression level for jpeg/webp (0-100). 0 = model default.",
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
            user_valves = __user__.get("valves")
        if not isinstance(user_valves, BaseModel):
            user_valves = self.UserValves()

        if not user_valves.IMAGE_GENERATION:
            return body

        # Build image generation parameters
        params: dict[str, Any] = {"model": self.valves.IMAGE_GENERATION_MODEL}
        if self.valves.IMAGE_GENERATION_MODERATION != "auto":
            params["moderation"] = self.valves.IMAGE_GENERATION_MODERATION
        for attr, key in [
            ("IMAGE_QUALITY", "quality"),
            ("IMAGE_SIZE", "size"),
            ("IMAGE_ASPECT_RATIO", "aspect_ratio"),
            ("IMAGE_BACKGROUND", "background"),
            ("IMAGE_OUTPUT_FORMAT", "output_format"),
        ]:
            val = getattr(user_valves, attr, "")
            if isinstance(val, str) and val.strip():
                params[key] = val.strip()
        compression = getattr(user_valves, "IMAGE_OUTPUT_COMPRESSION", 0)
        if isinstance(compression, int) and compression > 0:
            params["output_compression"] = compression

        # Write to metadata
        if isinstance(__metadata__, dict):
            prev_pipe_meta = __metadata__.get("openrouter_pipe")
            pipe_meta = dict(prev_pipe_meta) if isinstance(prev_pipe_meta, dict) else {}
            __metadata__["openrouter_pipe"] = pipe_meta

            prev_tools = pipe_meta.get("server_tools")
            server_tools = dict(prev_tools) if isinstance(prev_tools, dict) else {}
            pipe_meta["server_tools"] = server_tools
            server_tools["image_generation"] = params

        return body
