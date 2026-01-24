"""Model registry and catalog management.

This module handles all model-related functionality:
- ModelFamily: Base capabilities and alias mapping with effort defaults
- OpenRouterModelRegistry: Fetches and caches OpenRouter model catalog
- Model ID helpers: Sanitization, normalization, pattern matching
- Gemini reasoning helpers: Effort mapping, family classification
- Web search and icon metadata management

The registry is the authoritative source for model metadata used throughout
the pipe for model selection, fallback, and feature detection.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import json
import logging
import re
import time
from contextvars import ContextVar
from typing import Any, Dict, Optional, cast
from urllib.parse import quote

import aiohttp

# Import shared logger from config (for test compatibility)
from ..core.config import LOGGER
from .blocklists import is_direct_upload_blocklisted
from ..core.timing_logger import timed

# Lazy imports to avoid circular dependencies (see requests/__init__.py chain).

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_OPENROUTER_TITLE = "Open WebUI plugin for OpenRouter Responses API"
_OPENROUTER_REFERER = "https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/"

# -----------------------------------------------------------------------------
# Model Helper Functions
# -----------------------------------------------------------------------------

@timed
def sanitize_model_id(model_id: str) -> str:
    """Convert `author/model` ids into dot-friendly ids for Open WebUI."""
    if not model_id:
        return model_id
    if "/" not in model_id:
        return model_id
    head, tail = model_id.split("/", 1)
    return f"{head}.{tail.replace('/', '.')}"


# -----------------------------------------------------------------------------
# ModelFamily Class
# -----------------------------------------------------------------------------

class ModelFamily:
    """
    One place for base capabilities + alias mapping (with effort defaults).
    """

    _DATE_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")
    _PIPE_ID: ContextVar[Optional[str]] = ContextVar(
        "owui_pipe_id_ctx",
        default=None,
    )
    _DYNAMIC_SPECS: Dict[str, Dict[str, Any]] = {}

    # -- tiny, intuitive helpers ----------------------------------------------
    @classmethod
    @timed
    def _norm(cls, model_id: str) -> str:
        """Normalize model ids by stripping pipe prefixes and date suffixes.

        Preserves variant/preset suffixes (after :) unchanged to avoid corrupting
        preset slugs that contain forward slashes (e.g., preset/my-preset).
        """
        m = (model_id or "").strip()

        # Preserve variant/preset suffix if present (e.g., ":exacto" or ":preset/slug")
        suffix = ""
        if ":" in m:
            m, suffix = m.rsplit(":", 1)

        # Only replace / with . in the base portion
        if "/" in m:
            m = m.replace("/", ".")

        pipe_id = cls._PIPE_ID.get()
        if pipe_id:
            pref = f"{pipe_id}."
            if m.startswith(pref):
                m = m[len(pref):]

        base = cls._DATE_RE.sub("", m.lower())

        # Re-attach suffix unchanged (not lowercased, not normalized)
        if suffix:
            return f"{base}:{suffix}"
        return base

    @classmethod
    @timed
    def base_model(cls, model_id: str) -> str:
        """Canonical base model id (prefix/date stripped)."""
        return cls._norm(model_id)

    @classmethod
    @timed
    def features(cls, model_id: str) -> frozenset[str]:
        """Capabilities for the base model behind this id."""
        spec = cls._lookup_spec(model_id)
        return frozenset(spec.get("features", set()))

    @classmethod
    @timed
    def max_completion_tokens(cls, model_id: str) -> Optional[int]:
        """Return max completion tokens reported by the provider, if any."""
        spec = cls._lookup_spec(model_id)
        return spec.get("max_completion_tokens")

    @classmethod
    @timed
    def supports(cls, feature: str, model_id: str) -> bool:
        """Check if a model supports a given feature."""
        return feature in cls.features(model_id)

    @classmethod
    @timed
    def capabilities(cls, model_id: str) -> dict[str, bool]:
        """Return derived capability checkboxes for the given model."""
        spec = cls._lookup_spec(model_id)
        caps = spec.get("capabilities") or {}
        # Return a shallow copy so downstream code can mutate safely.
        return dict(caps)

    @classmethod
    @timed
    def supported_parameters(cls, model_id: str) -> frozenset[str]:
        """Return the raw `supported_parameters` set from the OpenRouter catalog."""
        spec = cls._lookup_spec(model_id)
        params = spec.get("supported_parameters")
        if isinstance(params, frozenset):
            return params
        if isinstance(params, (set, list, tuple)):
            return frozenset(params)
        return frozenset()

    @classmethod
    @timed
    def set_dynamic_specs(cls, specs: Dict[str, Dict[str, Any]] | None) -> None:
        """Update cached OpenRouter specs shared with :class:`ModelFamily`."""
        cls._DYNAMIC_SPECS = specs or {}

    @classmethod
    @timed
    def _lookup_spec(cls, model_id: str) -> Dict[str, Any]:
        """Return the stored spec for ``model_id`` or an empty dict."""
        norm = cls.base_model(model_id)
        return cls._DYNAMIC_SPECS.get(norm) or {}

# -----------------------------------------------------------------------------
# OpenRouterModelRegistry Class
# -----------------------------------------------------------------------------

class OpenRouterModelRegistry:
    """Fetches and caches the OpenRouter model catalog."""

    _models: list[dict[str, Any]] = []
    _specs: Dict[str, Dict[str, Any]] = {}
    _id_map: Dict[str, str] = {}  # normalized sanitized id -> original id
    _last_fetch: float = 0.0
    _lock: asyncio.Lock = asyncio.Lock()
    _next_refresh_after: float = 0.0
    _consecutive_failures: int = 0
    _last_error: str | None = None
    _last_error_time: float = 0.0

    @classmethod
    @timed
    async def ensure_loaded(
        cls,
        session: aiohttp.ClientSession,
        *,
        base_url: str,
        api_key: str,
        cache_seconds: int,
        logger: logging.Logger,
        http_referer: str | None = None,
    ) -> None:
        """Refresh the model catalog if the cache is empty or stale."""
        if not api_key:
            raise ValueError("OpenRouter API key is required.")

        now = time.time()
        next_refresh = cls._next_refresh_after or (cls._last_fetch + cache_seconds)
        if cls._specs and now < next_refresh:
            return

        async with cls._lock:
            now = time.time()
            next_refresh = cls._next_refresh_after or (cls._last_fetch + cache_seconds)
            if cls._specs and now < next_refresh:
                return
            try:
                await cls._refresh(
                    session,
                    base_url=base_url,
                    api_key=api_key,
                    logger=logger,
                    http_referer=http_referer,
                )
            except Exception as exc:
                # Catch all refresh errors (network, JSON, API errors) to use cache if available
                cls._record_refresh_failure(exc, cache_seconds)
                if not cls._models:
                    raise
                logger.warning(
                    "OpenRouter catalog refresh failed (%s). Serving %d cached model(s).",
                    exc,
                    len(cls._models),
                )
                return
            cls._record_refresh_success(cache_seconds)

    @classmethod
    @timed
    async def _refresh(
        cls,
        session: aiohttp.ClientSession,
        *,
        base_url: str,
        api_key: str,
        logger: logging.Logger,
        http_referer: str | None = None,
    ) -> None:
        """Fetch and cache the OpenRouter catalog."""
        # Lazy imports to avoid circular dependencies (via requests/__init__.py)
        from ..requests.debug import _debug_print_request, _debug_print_error_response

        url = base_url.rstrip("/") + "/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Title": _OPENROUTER_TITLE,
            "HTTP-Referer": (http_referer or _OPENROUTER_REFERER),
        }
        _debug_print_request(headers, {"method": "GET", "url": url}, logger=logger)
        try:
            async with session.get(url, headers=headers) as resp:
                if resp.status >= 400:
                    await _debug_print_error_response(resp, logger=logger)
                resp.raise_for_status()
                payload = await resp.json()
        except Exception as exc:
            logger.error("Failed to load OpenRouter model catalog: %s", exc)
            raise

        data = payload.get("data") or []
        raw_specs: Dict[str, Dict[str, Any]] = {}
        models: list[dict[str, Any]] = []
        id_map: Dict[str, str] = {}

        for item in data:
            original_id = item.get("id")
            if not original_id:
                continue

            sanitized = sanitize_model_id(original_id)
            norm_id = ModelFamily.base_model(sanitized)

            # Store the FULL model object - it's only 3.5MB total for all models
            # This gives us access to: description, context_length, created, canonical_slug,
            # hugging_face_id, per_request_limits, default_parameters, and everything else
            raw_specs[norm_id] = dict(item)

            id_map[norm_id] = original_id
            models.append(
                {
                    "id": sanitized,
                    "norm_id": norm_id,
                    "original_id": original_id,
                    "name": item.get("name") or original_id,
                }
            )

        # Finalize specs shared with ModelFamily (features + max completion tokens).
        # Now we keep the FULL model object plus derived features for fast lookups
        specs: Dict[str, Dict[str, Any]] = {}
        for norm_id, full_model in raw_specs.items():
            supported_parameters = set(full_model.get("supported_parameters") or [])
            architecture = full_model.get("architecture") or {}
            pricing = full_model.get("pricing") or {}

            # Derive feature flags from model metadata
            features = cls._derive_features(
                supported_parameters,
                architecture,
                pricing,
            )

            # Apply direct upload blocklist: enable file_input by default,
            # except for models known to have issues with file uploads.
            # This overrides OpenRouter's incomplete input_modalities metadata.
            original_id = full_model.get("id") or norm_id
            if is_direct_upload_blocklisted(original_id):
                features.discard("file_input")
            else:
                # Default to enabled - most models support direct uploads
                features.add("file_input")

            capabilities = cls._derive_capabilities(
                architecture,
                pricing,
            )

            max_completion_tokens: Optional[int] = None
            top_provider = full_model.get("top_provider")
            if isinstance(top_provider, dict):
                max_completion_tokens = top_provider.get("max_completion_tokens")

            # Store full model + derived data for downstream use
            specs[norm_id] = {
                # Derived features for fast capability checks
                "features": features,
                "capabilities": capabilities,
                "max_completion_tokens": max_completion_tokens,
                "supported_parameters": frozenset(supported_parameters),

                # Keep full model object for any future needs
                "full_model": full_model,

                # Quick access to commonly used fields
                "context_length": full_model.get("context_length"),
                "description": full_model.get("description"),
                "pricing": pricing,
                "architecture": architecture,
            }

        models.sort(key=lambda m: m["name"].lower())
        if not models:
            raise RuntimeError("OpenRouter returned an empty model catalog.")
        cls._models = models
        cls._specs = specs
        cls._id_map = id_map

        # Share dynamic specs with ModelFamily for downstream feature checks.
        ModelFamily.set_dynamic_specs(specs)

    @classmethod
    @timed
    def _record_refresh_success(cls, cache_seconds: int) -> None:
        """Reset refresh backoff bookkeeping after a successful catalog fetch."""
        now = time.time()
        cls._last_fetch = now
        cls._next_refresh_after = now + max(5, cache_seconds)
        cls._consecutive_failures = 0
        cls._last_error = None
        cls._last_error_time = 0.0

    @classmethod
    @timed
    def _record_refresh_failure(cls, exc: Exception, cache_seconds: int) -> None:
        """Increase backoff delay and track the most recent catalog error."""
        cls._consecutive_failures += 1
        cls._last_error = str(exc)
        cls._last_error_time = time.time()
        exponent = min(cls._consecutive_failures - 1, 5)
        base_backoff = 5.0
        raw_backoff = base_backoff * (2 ** exponent)
        capped_backoff = min(cache_seconds, raw_backoff)
        backoff_until = cls._last_error_time + max(base_backoff, capped_backoff)
        cls._next_refresh_after = max(cls._next_refresh_after, backoff_until)

    @staticmethod
    @timed
    def _derive_features(
        supported_parameters: set[str],
        architecture: Dict[str, Any],
        pricing: Dict[str, Any],
    ) -> set[str]:
        """Translate OpenRouter metadata into capability flags.

        Features include:
        - function_calling: Model supports tools/function calling
        - reasoning: Model supports extended reasoning
        - reasoning_summary: Model supports reasoning summaries
        - web_search_tool: Model has web search capability
        - image_gen_tool: Model can generate images (output)
        - vision: Model accepts image inputs
        - audio_input: Model accepts audio inputs
        - video_input: Model accepts video inputs
        - file_input: Model accepts file/document inputs
        """
        features: set[str] = set()

        if {"tools", "tool_choice"} & supported_parameters:
            features.add("function_calling")
        if "reasoning" in supported_parameters:
            features.add("reasoning")
        if "include_reasoning" in supported_parameters:
            features.add("reasoning_summary")

        if pricing.get("web_search") is not None:
            features.add("web_search_tool")

        output_modalities = architecture.get("output_modalities") or []
        if "image" in output_modalities:
            features.add("image_gen_tool")

        input_modalities = architecture.get("input_modalities") or []
        if "image" in input_modalities:
            features.add("vision")
        if "audio" in input_modalities:
            features.add("audio_input")
        if "video" in input_modalities:
            features.add("video_input")
        if "file" in input_modalities:
            features.add("file_input")

        return features

    @staticmethod
    @timed
    def _supports_web_search(pricing: Dict[str, Any]) -> bool:
        """Return True when the provider exposes paid web-search support."""
        value = pricing.get("web_search")
        if value is None:
            return False
        if isinstance(value, str):
            value = value.strip() or "0"
        try:
            return float(value) > 0.0
        except (TypeError, ValueError):
            return False

    @staticmethod
    @timed
    def _derive_capabilities(
        architecture: Dict[str, Any],
        pricing: Dict[str, Any],
    ) -> dict[str, bool]:
        """Translate metadata into Open WebUI capability checkboxes."""

        @timed
        def _normalize(values: list[Any]) -> set[str]:
            """Return a normalized lowercase set from the provider metadata."""
            normalized: set[str] = set()
            for item in values:
                if isinstance(item, str):
                    normalized.add(item.strip().lower())
            return normalized

        input_modalities = _normalize(architecture.get("input_modalities") or [])
        output_modalities = _normalize(architecture.get("output_modalities") or [])

        vision_capable = "image" in input_modalities or "video" in input_modalities
        # Open WebUI uses file uploads for attachments and RAG workflows; if we mark this
        # as False, the UI blocks uploads entirely (including images) even for vision-capable
        # models. Keep it enabled for all models exposed by this pipe.
        file_upload_capable = True
        image_generation_capable = "image" in output_modalities
        web_search_capable = OpenRouterModelRegistry._supports_web_search(pricing)

        return {
            "vision": vision_capable,
            "file_upload": file_upload_capable,
            "web_search": web_search_capable,
            "image_generation": image_generation_capable,
            "code_interpreter": True,
            "citations": True,
            "status_updates": True,
            "usage": True,
        }

    @classmethod
    @timed
    def list_models(cls) -> list[dict[str, Any]]:
        """Return a shallow copy of the cached catalog."""
        enriched: list[dict[str, Any]] = []
        for model in cls._models:
            item = dict(model)
            spec = cls._specs.get(model["norm_id"])
            if spec and spec.get("capabilities"):
                item["capabilities"] = dict(spec["capabilities"])
            enriched.append(item)
        return enriched


    @classmethod
    @timed
    def api_model_id(cls, model_id: str) -> Optional[str]:
        """Map sanitized Open WebUI ids back to provider ids, preserving variant/preset suffix.

        Examples:
            - "openai.gpt-4o" -> "openai/gpt-4o"
            - "openai.gpt-4o:exacto" -> "openai/gpt-4o:exacto"
            - "anthropic.claude-opus:extended" -> "anthropic/claude-opus:extended"
            - "openai.gpt-4o:preset/email-copywriter" -> "openai/gpt-4o@preset/email-copywriter"
        """
        # Extract variant/preset suffix if present
        suffix_tag = ""
        if ":" in model_id:
            parts = model_id.rsplit(":", 1)
            model_id_base = parts[0]
            suffix_tag = parts[1]  # e.g., "exacto" or "preset/email-copywriter"
        else:
            model_id_base = model_id

        # Normalize base and lookup
        norm = ModelFamily.base_model(model_id_base)
        provider_id = cls._id_map.get(norm)

        if not provider_id:
            return None

        # Re-attach suffix with appropriate separator
        # Presets use @ separator, variants use : separator
        if suffix_tag:
            if suffix_tag.startswith("preset/"):
                return f"{provider_id}@{suffix_tag}"
            else:
                return f"{provider_id}:{suffix_tag}"
        return provider_id

    @classmethod
    @timed
    def spec(cls, model_id: str) -> Dict[str, Any]:
        """Return the cached spec for ``model_id`` (or an empty dict)."""
        norm = ModelFamily.base_model(model_id)
        return cls._specs.get(norm) or {}

# -----------------------------------------------------------------------------
# Additional Model Helper Functions
# -----------------------------------------------------------------------------

@timed
def normalize_model_id_dotted(model_id: str) -> str:
    """Return a dotted variant of a model id (e.g. 'anthropic/claude' -> 'anthropic.claude')."""
    if not isinstance(model_id, str):
        return ""
    return model_id.strip().replace("/", ".")


@timed
def _matches_any_model_pattern(model_id: str, patterns: list[str]) -> bool:
    """Return True when model_id matches any glob-style pattern."""
    if not isinstance(model_id, str):
        return False
    candidate = model_id.strip()
    if not candidate or not patterns:
        return False
    dotted = normalize_model_id_dotted(candidate)
    for pattern in patterns:
        if not pattern:
            continue
        pattern_stripped = pattern.strip()
        if not pattern_stripped:
            continue
        dotted_pattern = normalize_model_id_dotted(pattern_stripped)
        if (
            fnmatch.fnmatchcase(candidate, pattern_stripped)
            or fnmatch.fnmatchcase(candidate, dotted_pattern)
            or fnmatch.fnmatchcase(dotted, pattern_stripped)
            or fnmatch.fnmatchcase(dotted, dotted_pattern)
        ):
            return True
    return False

# -----------------------------------------------------------------------------
# Gemini Reasoning Helpers
# -----------------------------------------------------------------------------

@timed
def _classify_gemini_thinking_family(normalized_model_id: str) -> Optional[str]:
    """Return the Gemini thinking family for the provided normalized model id."""
    lowered = (normalized_model_id or "").lower()
    if lowered.startswith("google.gemini-3-") or lowered == "google.gemini-3":
        return "gemini-3"
    if lowered.startswith("google.gemini-2.5-") or lowered == "google.gemini-2.5":
        return "gemini-2.5"
    return None


@timed
def _map_effort_to_gemini_budget(effort: str, base_budget: int) -> Optional[int]:
    """Scale the configured Gemini 2.5 thinking budget from the requested effort."""
    if base_budget <= 0:
        return 0 if base_budget == 0 else None
    normalized = (effort or "").strip().lower()
    if normalized == "none":
        return None
    scalars = {
        "minimal": 0.25,
        "low": 0.5,
        "medium": 1.0,
        "high": 2.0,
        "xhigh": 4.0,
    }
    scalar = scalars.get(normalized, 1.0)
    return int(max(1, round(base_budget * scalar)))


# -----------------------------------------------------------------------------
# Model Selection and Effort Mapping
# -----------------------------------------------------------------------------

@timed
def _map_effort_to_gemini_level(effort: str, valve_level: str) -> Optional[str]:
    """Map our reasoning effort preference onto Gemini 3 thinking levels."""
    effective = (valve_level or "auto").strip().lower()
    if effective in {"low", "high"}:
        return effective.upper()
    normalized = (effort or "").strip().lower()
    if normalized in {"none", ""}:
        return None
    if normalized in {"minimal", "low"}:
        return "LOW"
    return "HIGH"


@timed
def _parse_model_patterns(value: Any) -> list[str]:
    """Parse comma-separated fnmatch patterns (order preserved, empty removed)."""
    if not isinstance(value, str):
        return []
    raw = value.strip()
    if not raw:
        return []
    patterns: list[str] = []
    for part in raw.split(","):
        candidate = part.strip()
        if not candidate:
            continue
        patterns.append(candidate)
    return patterns

