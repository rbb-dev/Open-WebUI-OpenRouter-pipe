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
            description='Which OpenRouter model draws the picture. No settings are offered for openai/gpt-5-image-mini: it is not in the image model list this pipe has loaded. Check the id if that is unexpected.',
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

        IMAGE_SIZE: str = Field(
                    default="",
                    title="Output size",
                    description="Exact pixel dimensions, where the model takes them rather than a tier. This model publishes no list of what it accepts here, so the value goes out as typed and the company running it decides. Empty leaves it unset.",
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
        wanted = (user_valves.IMAGE_SIZE or "").strip()
        if wanted:
            params['size'] = wanted

        if isinstance(__metadata__, dict):
            prev_pipe_meta = __metadata__.get('openrouter_pipe')
            pipe_meta = dict(prev_pipe_meta) if isinstance(prev_pipe_meta, dict) else {}
            __metadata__['openrouter_pipe'] = pipe_meta

            prev_tools = pipe_meta.get("server_tools")
            server_tools = dict(prev_tools) if isinstance(prev_tools, dict) else {}
            pipe_meta["server_tools"] = server_tools
            server_tools["image_generation"] = params

        return body
