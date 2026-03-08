"""Plugin base class and context for the pipe plugin system."""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..pipe import Pipe
    from ..storage.persistence import ArtifactStore


class PluginContext:
    """Gateway to all pipe internals. Stored by plugins from on_init().

    Convenience properties are shortcuts; ``pipe`` is the escape hatch to
    reach any subsystem, now or in the future.
    """

    __slots__ = ("_pipe", "logger")

    def __init__(self, pipe: Pipe, logger: logging.Logger) -> None:
        self._pipe = pipe
        self.logger = logger

    # ── Core access ──

    @property
    def pipe(self) -> Pipe:
        """Full Pipe instance — dig into any subsystem."""
        return self._pipe

    @property
    def valves(self) -> Any:
        """Live reference to pipe.valves."""
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

    # ── Response builders (delegate to Pipe methods) ──

    def build_response(self, *, model: str | None, content: str) -> dict[str, Any]:
        """Build chat.completions-style response dict."""
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
    """Base class for all plugins.

    Subclasses declare ``plugin_id``, ``plugin_name``, and ``hooks`` (a dict
    mapping hook names to integer priorities). Higher priority = runs first.
    ``on_init`` and ``on_shutdown`` are always called regardless of ``hooks``.

    All hook methods accept ``**kwargs`` for forward compatibility — new
    keyword arguments may be added to dispatch calls without breaking
    existing plugins.
    """

    plugin_id: str = ""
    plugin_name: str = ""
    plugin_version: str = "0.0.0"

    # Declare subscribed hooks and their priority (MUST be a class-level attribute).
    # Higher number = runs first. Omitted hooks = not dispatched.
    # on_init and on_shutdown are ALWAYS called (lifecycle, not subscription-based).
    hooks: dict[str, int] = {}

    # Plugin-contributed valve fields merged into Pipe.Valves at import time.
    # Keys are field names (e.g., "PIPE_STATS_ENABLE"), values are
    # ``(type, Field(...))`` tuples compatible with ``pydantic.create_model()``.
    plugin_valves: dict[str, tuple] = {}

    # Plugin-contributed per-user valve fields merged into Pipe.UserValves.
    # Same format as ``plugin_valves`` but for user-facing settings.
    plugin_user_valves: dict[str, tuple] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Ensure each subclass has its own dicts to prevent mutation
        # of the base class dict if a subclass forgets to define them.
        if "hooks" not in cls.__dict__:
            cls.hooks = {}
        if "plugin_valves" not in cls.__dict__:
            cls.plugin_valves = {}
        if "plugin_user_valves" not in cls.__dict__:
            cls.plugin_user_valves = {}

    # Synchronous lifecycle hook — always called at startup.
    def on_init(self, ctx: PluginContext, **kwargs: Any) -> None:
        """Always called at startup. Store ctx for later use."""

    # Synchronous void dispatch — plugins mutate the models list in place.
    def on_models(self, models: list[dict[str, Any]], **kwargs: Any) -> None:
        """Mutate the model list in place (append, remove, modify entries)."""

    # Async chain dispatch — each subscriber receives the current accumulated result via kwargs.
    async def on_request(
        self,
        body: dict[str, Any],
        user: dict[str, Any],
        metadata: dict[str, Any],
        event_emitter: Any,
        task: Any,
        **kwargs: Any,
    ) -> dict[str, Any] | str | None:
        """Intercept request before API key check. Chain dispatch.

        Subscribers run in priority order (highest first). Each receives the
        current accumulated result via ``current_result`` kwarg (starts as None).
        Return non-None to set or replace the result. Return None to leave the
        current result unchanged. The final accumulated result is returned.

        Extra kwargs: ``valves`` — merged per-request valves,
        ``current_result`` — accumulated result from prior plugins (or None).
        """
        return None

    # Async void dispatch — plugins mutate body dict in place.
    async def on_request_transform(
        self,
        body: dict[str, Any],
        model: str,
        valves: Any,
        **kwargs: Any,
    ) -> None:
        """Mutate body dict in place before OpenRouter.

        The ``model`` parameter reflects the current ``body["model"]`` value,
        updated after each plugin in the chain.

        Extra kwargs: ``user`` — user dict, ``metadata`` — request metadata.
        """

    # Async chain dispatch — each subscriber wraps or replaces the emitter.
    async def on_emitter_wrap(
        self,
        stream_emitter: Any,
        **kwargs: Any,
    ) -> Any | None:
        """Wrap or replace the stream event emitter for the current request.

        Fires once per streaming request in ``_execute_pipe_job()``, after the
        middleware stream emitter is created but before the streaming loop.
        Chain dispatch — each plugin receives the (possibly already-wrapped)
        emitter.  Return a wrapped callable to replace it, or ``None`` to
        leave unchanged.

        Extra kwargs: ``raw_emitter`` — OWUI-facing emitter (for embeds),
        ``job_metadata`` — dict with user_id/chat_id/message_id/request_id,
        ``valves`` — merged system+user valves for the request.
        """
        return None

    # Async void dispatch — plugins mutate completion_data dict in place.
    async def on_response_transform(
        self,
        completion_data: dict[str, Any],
        model: str,
        metadata: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Mutate completion_data dict in place before OWUI delivery.

        ``completion_data`` is the exact dict sent to OWUI via ``chat:completion``.
        Standard keys: ``done`` (bool), ``content`` (str), ``usage`` (dict|absent).
        Plugins may read, modify, or add any key (e.g., ``sources``, ``title``).
        Mutate in place — do NOT reassign the parameter.

        Extra kwargs: ``user_id`` — user ID string, ``user`` — user object.
        """

    # Synchronous lifecycle hook — always called on pipe shutdown.
    def on_shutdown(self, **kwargs: Any) -> None:
        """Always called on pipe shutdown. Release resources."""
