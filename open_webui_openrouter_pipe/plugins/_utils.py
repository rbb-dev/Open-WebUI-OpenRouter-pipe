"""Shared utilities for the plugin subsystem.

Provides:
- :func:`get_owui_app` / :func:`ensure_route_before_spa` — OWUI FastAPI integration
- :func:`register_sse_endpoint` — generic SSE route registration
- :class:`EphemeralKeyStore` — ephemeral session key store with optional Redis
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from typing import Any, Callable

_plugin_utils_log = logging.getLogger(__name__)


# ── OWUI App Integration ──


def get_owui_app() -> Any | None:
    """Import and return the OWUI FastAPI app, or ``None`` if unavailable."""
    try:
        from open_webui.main import app
        return app
    except ImportError:
        _plugin_utils_log.debug("OWUI app not available")
        return None


def ensure_route_before_spa(app: Any) -> None:
    """Push the SPA catch-all mount to the end of ``app.routes``.

    Starlette evaluates routes in list order.  OWUI's SPAStaticFiles mount
    at ``/`` catches everything, so dynamically-registered API routes must
    appear *before* it.  This helper moves the SPA mount to the end,
    allowing all prior routes to match first.  Idempotent.
    """
    for i, route in enumerate(app.routes):
        if getattr(route, "name", "") == "spa-static-files":
            app.routes.append(app.routes.pop(i))
            break


# ── SSE Endpoint Registration ──

# Module-level registry of registered SSE paths for idempotency.
_sse_registered_paths: set[str] = set()
_sse_registration_lock = threading.Lock()


def register_sse_endpoint(
    path: str,
    handler: Callable[..., Any],
    *,
    logger: logging.Logger | None = None,
) -> bool:
    """Register an SSE GET endpoint on OWUI's FastAPI app.

    Handles idempotency (tracked by *path*), Starlette import, OWUI app
    lookup, handler binding, and SPA mount reordering.  Thread-safe.

    *handler* is an ``async def handler(key: str) -> Response`` function.

    Returns ``True`` if registered (or already exists), ``False`` if
    the OWUI app or Starlette is unavailable.
    """
    with _sse_registration_lock:
        if path in _sse_registered_paths:
            return True

        log = logger or _plugin_utils_log

        try:
            from starlette.responses import StreamingResponse  # noqa: F401
        except ImportError:
            log.debug("Starlette not available — SSE endpoint %s not registered", path)
            return False

        app = get_owui_app()
        if app is None:
            log.debug("OWUI app not available — SSE endpoint %s not registered", path)
            return False

        app.get(path)(handler)
        ensure_route_before_spa(app)

        _sse_registered_paths.add(path)
        log.debug("SSE endpoint registered at %s", path)
        return True


# ── Ephemeral Key Store ──

_EK_DEFAULT_TTL = 300.0  # 5 minutes

_ek_log = logging.getLogger(f"{__name__}.ephemeral_keys")


class EphemeralKeyStore:
    """In-memory key store with optional Redis dual-write for multi-worker.

    Keys are 64-char hex tokens generated via :func:`secrets.token_hex`.
    Each key tracks its last-access timestamp; keys idle longer than
    *ttl* seconds are purged automatically on the next :meth:`validate`
    call.  There is no hard cap on key count — TTL-based cleanup is the
    sole eviction strategy, keeping the store elastic under traffic spikes.

    For multi-worker deployments, call :meth:`configure_redis` to enable
    dual-write.  The async methods (``async_generate``, ``async_validate``,
    ``async_revoke``) handle both local and Redis operations.
    """

    __slots__ = ("_keys", "_lock", "_redis_client", "_redis_prefix", "_ttl")

    def __init__(self, ttl: float = _EK_DEFAULT_TTL) -> None:
        self._keys: dict[str, float] = {}
        self._lock = threading.Lock()
        self._ttl = max(1.0, ttl)
        self._redis_client: Any = None
        self._redis_prefix: str = ""

    # ── Redis Configuration ──

    def configure_redis(self, client: Any, namespace: str) -> None:
        """Enable Redis dual-write for cross-worker key sharing.

        Call this after Redis is initialised.  *namespace* is the Redis
        key prefix (e.g. ``"openrouter"``).  Keys are stored at
        ``{namespace}:ephemeral:{token}``.  Thread-safe.
        """
        with self._lock:
            self._redis_client = client
            self._redis_prefix = f"{namespace}:ephemeral"

    # ── Sync API (process-local only) ──

    def generate(self) -> str:
        """Create a new ephemeral key (local dict only).  Thread-safe."""
        with self._lock:
            self._cleanup()
            key = secrets.token_hex(32)
            self._keys[key] = time.monotonic()
        return key

    def validate(self, key: str) -> bool:
        """Check if *key* exists locally and is not expired.

        On success the key's timestamp is refreshed (keep-alive).
        Thread-safe.
        """
        with self._lock:
            self._cleanup()
            if key not in self._keys:
                return False
            self._keys[key] = time.monotonic()
            return True

    def revoke(self, key: str) -> None:
        """Manually delete a key from local store.  Thread-safe."""
        with self._lock:
            self._keys.pop(key, None)

    @property
    def active_count(self) -> int:
        """Number of currently active (non-expired) local keys."""
        with self._lock:
            self._cleanup()
            return len(self._keys)

    # ── Async API (local + Redis dual-write) ──

    async def async_generate(self) -> str:
        """Generate a key and write to both local dict and Redis.

        Falls back to local-only if Redis is unavailable or errors.
        """
        key = self.generate()
        if self._redis_client is not None:
            try:
                await self._redis_client.set(
                    f"{self._redis_prefix}:{key}",
                    "1",
                    ex=int(self._ttl),
                )
            except Exception:
                _ek_log.debug("Redis write failed for ephemeral key", exc_info=True)
        return key

    async def async_validate(self, key: str) -> bool:
        """Validate locally first, then fall back to Redis.

        On a Redis hit the key is imported into the local dict (so
        subsequent checks are fast-path) and the Redis TTL is refreshed.
        """
        if self.validate(key):
            # Local hit — also refresh Redis TTL if configured
            if self._redis_client is not None:
                try:
                    await self._redis_client.expire(
                        f"{self._redis_prefix}:{key}",
                        int(self._ttl),
                    )
                except Exception:
                    pass  # Redis refresh failure is non-critical
            return True
        # Local miss — check Redis for cross-worker key
        if self._redis_client is not None:
            try:
                exists = await self._redis_client.exists(f"{self._redis_prefix}:{key}")
            except Exception:
                _ek_log.debug("Redis validate failed for ephemeral key", exc_info=True)
                return False
            if exists:
                # Import into local store for future fast-path hits
                with self._lock:
                    self._cleanup()
                    self._keys[key] = time.monotonic()
                # Refresh Redis TTL — failure is non-critical
                try:
                    await self._redis_client.expire(
                        f"{self._redis_prefix}:{key}",
                        int(self._ttl),
                    )
                except Exception:
                    pass
                return True
        return False

    async def async_revoke(self, key: str) -> None:
        """Revoke from both local dict and Redis."""
        self.revoke(key)
        if self._redis_client is not None:
            try:
                await self._redis_client.delete(f"{self._redis_prefix}:{key}")
            except Exception:
                _ek_log.debug("Redis delete failed for ephemeral key", exc_info=True)

    # ── Internal ──

    def _cleanup(self) -> None:
        """Purge keys that have been idle longer than TTL."""
        cutoff = time.monotonic() - self._ttl
        expired = [k for k, ts in self._keys.items() if ts < cutoff]
        for k in expired:
            del self._keys[k]


# ── Virtual Model Helpers ──


def extract_task_name(task: Any) -> str:
    """Extract task name from OWUI background task metadata.

    OWUI sends background tasks (title generation, tags, emoji, follow-ups)
    to all models including virtual ones.  The task parameter can be a string
    or a dict with ``type``, ``task``, or ``name`` keys.
    """
    if isinstance(task, str):
        return task.strip()
    if isinstance(task, dict):
        name = task.get("type") or task.get("task") or task.get("name")
        return name.strip() if isinstance(name, str) else ""
    return ""


def extract_user_message(body: dict[str, Any]) -> str:
    """Extract the last user message text from a chat completions request body.

    Handles both plain string content and multimodal content (list of parts).
    Returns empty string if no user message is found.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        if isinstance(text, str):
                            return text.strip()
    return ""


def configure_keystore_redis(key_store: EphemeralKeyStore, pipe: Any) -> None:
    """Wire Redis into an EphemeralKeyStore if the pipe has Redis enabled.

    Safe to call repeatedly — no-ops if Redis is unavailable or already
    configured.  Use as a lazy retry in hooks that fire after ``on_init``
    (e.g. ``on_models``, ``on_emitter_wrap``) to handle late Redis init.
    """
    if pipe is None:
        return
    if getattr(pipe, "_redis_enabled", False):
        client = getattr(pipe, "_redis_client", None)
        if client is not None:
            ns = getattr(pipe, "_redis_namespace", "openrouter")
            key_store.configure_redis(client, ns)
