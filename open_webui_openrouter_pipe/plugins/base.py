from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ..pipe import Pipe
    from ..storage.persistence import ArtifactStore


class PluginContext:
    __slots__ = ("_pipe", "logger")

    def __init__(self, pipe: Pipe, logger: logging.Logger) -> None:
        self._pipe = pipe
        self.logger = logger

    @property
    def pipe(self) -> Pipe:
        return self._pipe

    @property
    def valves(self) -> Any:
        return self._pipe.valves

    @property
    def pipe_id(self) -> str:
        return getattr(self._pipe, "id", "openrouter")

    @property
    def artifact_store(self) -> ArtifactStore:
        return self._pipe._artifact_store

    @property
    def circuit_breaker(self) -> Any:
        return self._pipe._circuit_breaker

    def build_response(self, *, model: str | None, content: str) -> dict[str, Any]:
        model_id = (model or "pipe").strip() if isinstance(model, str) else "pipe"
        return {
            "id": f"{model_id}-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
        }


class PluginBase:
    plugin_id: str = ""
    plugin_name: str = ""
    plugin_version: str = "0.0.0"

    hooks: ClassVar[dict[str, int]] = {}

    plugin_valves: ClassVar[dict[str, tuple]] = {}

    plugin_user_valves: ClassVar[dict[str, tuple]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "hooks" not in cls.__dict__:
            cls.hooks = {}
        if "plugin_valves" not in cls.__dict__:
            cls.plugin_valves = {}
        if "plugin_user_valves" not in cls.__dict__:
            cls.plugin_user_valves = {}

    def on_init(self, ctx: PluginContext, **kwargs: Any) -> None:
        pass

    def on_models(self, models: list[dict[str, Any]], **kwargs: Any) -> Awaitable[None] | None:
        return None

    async def on_request(
        self,
        body: dict[str, Any],
        user: dict[str, Any],
        metadata: dict[str, Any],
        event_emitter: Any,
        task: Any,
        **kwargs: Any,
    ) -> dict[str, Any] | str | None:
        return None

    async def on_request_transform(
        self,
        body: dict[str, Any],
        model: str,
        valves: Any,
        **kwargs: Any,
    ) -> None:
        pass

    async def on_emitter_wrap(
        self,
        stream_emitter: Any,
        **kwargs: Any,
    ) -> Any | None:
        return None

    async def on_tool_result(
        self,
        tool_name: str,
        status: str,
        **kwargs: Any,
    ) -> None:
        pass

    async def on_request_retry(
        self,
        kind: str,
        **kwargs: Any,
    ) -> None:
        pass

    async def on_request_alive(
        self,
        request_id: str = "",
        **kwargs: Any,
    ) -> None:
        pass

    async def on_generation_complete(
        self,
        usage: dict[str, Any] | None,
        status: str,
        **kwargs: Any,
    ) -> None:
        pass

    def on_shutdown(self, **kwargs: Any) -> Any:
        pass
