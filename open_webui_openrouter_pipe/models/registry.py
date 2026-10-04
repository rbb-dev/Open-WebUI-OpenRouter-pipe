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
import fnmatch
import hashlib
import logging
import re
import threading
import time
import weakref
from collections import OrderedDict
from contextvars import ContextVar
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Any, ClassVar

import aiohttp

from ..core.config import (
    _OPENROUTER_CATEGORIES,
    _OPENROUTER_TITLE,
    openrouter_attribution_headers,
)
from ..core.timing_logger import timed
from .blocklists import _DATE_SUFFIX, is_direct_upload_blocklisted_for

# Model Helper Functions

def sanitize_model_id(model_id: str) -> str:
    """Convert `author/model` ids into dot-friendly ids for Open WebUI."""
    if not model_id:
        return model_id
    if "/" not in model_id:
        return model_id
    head, tail = model_id.split("/", 1)
    return f"{head}.{tail.replace('/', '.')}"


def _build_image_endpoint_aliases(
    published: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    aliases: dict[str, list[dict[str, Any]]] = {}
    for original, records in published.items():
        aliases.setdefault(sanitize_model_id(original), records)
    return aliases


PHASE_SUPPORTED_MODELS: tuple[str, ...] = (
    "openai/gpt-5.3-codex",
    "openai/gpt-5.4",
    "openai/gpt-5.4-pro",
    "openai/gpt-5.5",
    "openai/gpt-5.5-pro",
)


# ModelFamily Class

_NO_PIPE_ID = object()


class ModelFamily:
    """
    One place for base capabilities + alias mapping (with effort defaults).
    """

    _DATE_RE = _DATE_SUFFIX
    _PIPE_ID: ContextVar[str | None] = ContextVar(
        "owui_pipe_id_ctx",
        default=None,
    )
    _DYNAMIC_SPECS: ClassVar[dict[str, dict[str, Any]]] = {}

    @classmethod
    def _normalize_catalog_id(
        cls,
        model_id: str,
        pipe_id: str | None | object = None,
    ) -> str:
        m = (model_id or "").strip()

        preset_tag = ""
        if "@" in m:
            head, _sep, tail = m.rpartition("@")
            if head and tail.startswith("preset/"):
                m, preset_tag = head, tail

        suffix = ""
        if ":" in m:
            m, suffix = m.rsplit(":", 1)

        if "/" in m:
            m = m.replace("/", ".")

        if pipe_id is _NO_PIPE_ID:
            pipe_id = None
        elif pipe_id is None:
            pipe_id = cls._PIPE_ID.get()
        if pipe_id:
            pref = f"{pipe_id}."
            m = m.removeprefix(pref)

        base = m.lower()

        if preset_tag:
            suffix = f"{suffix}:{preset_tag}" if suffix else preset_tag

        if suffix:
            return f"{base}:{suffix}"
        return base

    @classmethod
    def _norm(cls, model_id: str, pipe_id: str | None | object = None) -> str:
        return cls._normalize_catalog_id(model_id, pipe_id)

    @classmethod
    def base_model(cls, model_id: str, pipe_id: str | None | object = None) -> str:
        return cls._norm(model_id, pipe_id)

    @classmethod
    def undated(cls, model_id: str) -> str:
        name, separator, suffix = (model_id or "").rpartition(":")
        if not separator or not suffix:
            return cls._DATE_RE.sub("", cls._norm(model_id or ""))
        return f"{cls._DATE_RE.sub('', cls._norm(name))}:{suffix}"

    @classmethod
    def features(cls, model_id: str) -> frozenset[str]:
        """Capabilities for the base model behind this id."""
        spec = cls._lookup_spec(model_id)
        return frozenset(spec.get("features", set()))

    @classmethod
    def max_completion_tokens(cls, model_id: str) -> int | None:
        """Return max completion tokens reported by the provider, if any."""
        spec = cls._lookup_spec(model_id)
        return spec.get("max_completion_tokens")

    @classmethod
    def supports(cls, feature: str, model_id: str) -> bool:
        """Check if a model supports a given feature."""
        return feature in cls.features(model_id)

    @classmethod
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
    def rules_out_tool_use(cls, model_id: str) -> bool:
        params = cls.supported_parameters(model_id)
        if not params:
            return False
        return not ({"tools", "tool_choice"} & params)

    @classmethod
    def set_dynamic_specs(cls, specs: dict[str, dict[str, Any]] | None) -> None:
        """Update cached OpenRouter specs shared with :class:`ModelFamily`."""
        cls._DYNAMIC_SPECS = specs or {}

    @classmethod
    def _strip_known_suffixes(cls, norm: str) -> list[str]:
        candidates = [norm]
        base, _, _tag = norm.rpartition(":")
        if base:
            candidates.append(base)
            if cls._DATE_RE.search(base):
                candidates.append(cls._DATE_RE.sub("", base))
        return candidates

    @classmethod
    def _resolve_spec_key(cls, norm: str, specs: dict[str, dict[str, Any]]) -> str:
        candidates = cls._strip_known_suffixes(norm)
        for candidate in candidates:
            if candidate in specs:
                return candidate
        undated = cls.undated(norm)
        if undated not in candidates:
            if undated in specs:
                return undated
            candidates.append(undated)
        return candidates[1] if len(candidates) > 1 else norm

    @classmethod
    def _lookup_spec(cls, model_id: str) -> dict[str, Any]:
        """Return the stored spec for ``model_id`` or an empty dict."""
        specs = cls._DYNAMIC_SPECS
        pipe_id = cls._PIPE_ID.get()
        key = _NO_PIPE_ID if pipe_id is None else pipe_id
        for candidate in _lookup_candidates(_norm_for_lookup(model_id, _NO_PIPE_ID), key):
            if candidate in specs:
                return specs[candidate] or {}
        if pipe_id is not None:
            for candidate in _lookup_candidates(_norm_for_lookup(model_id, pipe_id), key):
                if candidate in specs:
                    return specs[candidate] or {}
        return {}

    @classmethod
    def catalog_norm_id(cls, model_id: str) -> str:
        norm = cls.base_model(model_id)
        base, _, tag = norm.rpartition(":")
        if norm in cls._DYNAMIC_SPECS:
            return norm
        if base and not tag.startswith("preset/"):
            return base
        undated = cls.undated(norm)
        if not base and undated != norm and undated in cls._DYNAMIC_SPECS:
            return undated
        return norm

    @classmethod
    def catalog_spec(cls, model_id: str) -> dict[str, Any]:
        return cls._DYNAMIC_SPECS.get(cls.catalog_norm_id(model_id)) or {}

    @classmethod
    def supports_verbosity(cls, model_id: str) -> bool:
        return _supports_verbosity(cls.base_model(model_id, _NO_PIPE_ID))

    @classmethod
    def catalog_supported_parameters(cls, model_id: str) -> frozenset[str]:
        norm = cls.catalog_norm_id(model_id)
        base, _, tag = norm.rpartition(":")
        if base and tag.startswith("preset/") and norm not in cls._DYNAMIC_SPECS:
            return frozenset()
        return cls.supported_parameters(norm)

    @classmethod
    def reasoning_contract(cls, model_id: str) -> dict[str, Any]:
        full = cls.catalog_spec(model_id).get("full_model")
        row = full.get("reasoning") if isinstance(full, dict) else None
        return row if isinstance(row, dict) else {}

    @classmethod
    def display_name(cls, model_id: str) -> str | None:
        """Return the OpenRouter catalog display name for ``model_id`` if cached."""
        norm = cls.base_model(model_id)
        spec = cls._DYNAMIC_SPECS.get(norm) or {}
        full = spec.get("full_model") if isinstance(spec, dict) else None
        name = full.get("name") if isinstance(full, dict) else None
        if not (isinstance(name, str) and name):
            base_spec = cls.catalog_spec(model_id)
            base_full = base_spec.get("full_model") if isinstance(base_spec, dict) else None
            base_name = base_full.get("name") if isinstance(base_full, dict) else None
            if not (isinstance(base_name, str) and base_name):
                return None
            _head, separator, tag = norm.rpartition(":")
            if not separator or not tag or tag.startswith("preset/"):
                return base_name
            if "." not in _head:
                return base_name
            return f"{base_name} {tag.capitalize()}"
        return name


def _catalog_norm(model_id: str) -> str:
    return ModelFamily.base_model(sanitize_model_id(model_id), _NO_PIPE_ID)


@lru_cache(maxsize=4096)
def _norm_for_lookup(model_id: str, pipe_id: str | None | object) -> str:
    return ModelFamily._normalize_catalog_id(model_id, pipe_id)


def _undated_for_lookup(norm: str, pipe_id: str | None | object) -> str:
    name, separator, suffix = (norm or "").rpartition(":")
    if not separator or not suffix:
        return ModelFamily._DATE_RE.sub("", _norm_for_lookup(norm or "", pipe_id))
    return f"{ModelFamily._DATE_RE.sub('', _norm_for_lookup(name, pipe_id))}:{suffix}"


def _suffix_spellings(value: str) -> list[str]:
    out = [value]
    base, _, _tag = value.rpartition(":")
    if base:
        out.append(base)
        if ModelFamily._DATE_RE.search(base):
            out.append(ModelFamily._DATE_RE.sub("", base))
    return out


@lru_cache(maxsize=4096)
def _lookup_candidates(norm: str, pipe_id: str | None | object) -> tuple[str, ...]:
    candidates: list[str] = []
    for spelling in _suffix_spellings(norm):
        for candidate in (spelling, _norm_for_lookup(spelling, pipe_id)):
            if candidate not in candidates:
                candidates.append(candidate)
    for spelling in _suffix_spellings(_undated_for_lookup(norm, pipe_id)):
        if spelling not in candidates:
            candidates.append(spelling)
    return tuple(candidates)


_PHASE_SUPPORTED_MODELS_BASE = frozenset(
    ModelFamily.undated(model_id) for model_id in PHASE_SUPPORTED_MODELS
)


def supports_phase_model(model_id: str) -> bool:
    """Return True when the model is one of the documented phase-capable ids."""
    candidate = (model_id or "").strip()
    if "@" in candidate:
        candidate, _ = candidate.split("@", 1)
    normalized = ModelFamily.undated(candidate)
    if ":" in normalized:
        normalized, _ = normalized.rsplit(":", 1)
    key = normalized.removeprefix("~")
    return bool(key) and key in _PHASE_SUPPORTED_MODELS_BASE

# OpenRouterModelRegistry Class

def uses_dedicated_image_api(spec: Any) -> bool:
    """True when a model emits images and no text, so it answers only on the image API.
    """
    if not isinstance(spec, dict):
        return False
    modalities = (spec.get("architecture") or {}).get("output_modalities") or []
    if not isinstance(modalities, list):
        return False
    return "image" in modalities and "text" not in modalities


_TEXTLESS_OUTPUT_NOUNS = {"image": "pictures", "video": "videos"}


def _output_modalities(spec: Any) -> list[Any]:
    if not isinstance(spec, dict):
        return []
    modalities = (spec.get("architecture") or {}).get("output_modalities") or []
    return modalities if isinstance(modalities, list) else []


def cannot_answer_in_text(spec: Any) -> bool:
    if not isinstance(spec, dict):
        return False
    modalities = _output_modalities(spec)
    if not modalities:
        return False
    return "text" not in modalities


def textless_output_noun(spec: Any) -> str:
    if not isinstance(spec, dict):
        return ""
    modalities = _output_modalities(spec)
    nouns = [
        _TEXTLESS_OUTPUT_NOUNS[str(name)]
        for name in modalities
        if str(name) in _TEXTLESS_OUTPUT_NOUNS
    ]
    return ", ".join(nouns) or ", ".join(str(name) for name in modalities)


def is_image_output_architecture(architecture: Any) -> bool:
    if not isinstance(architecture, dict):
        return False
    modalities = architecture.get("output_modalities")
    if not isinstance(modalities, list) or "image" not in modalities:
        return False
    return not (architecture.get("tokenizer") == "Router" and "text" in modalities)


def _fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()[:32]


def _contract_target(valves: Any) -> tuple[str, str]:
    base_url = (getattr(valves, "BASE_URL", "") or "https://openrouter.ai/api/v1").rstrip("/")
    return base_url, _fingerprint(str(getattr(valves, "API_KEY", "") or ""))


_ZDR_CREDENTIAL_HISTORY = 4
_IMAGE_CONTRACT_TARGET_HISTORY = 4
_ZDR_CREDENTIAL_RETENTION_SECONDS = 4 * 60 * 60
_ZDR_LIVE_CREDENTIAL_CEILING = 64
_REGISTRY_CATALOG_TIMEOUT_SECONDS = 15


def _trim_credential_history(
    mapping: dict[Any, Any], live: Any, aliased: Any, limit: int = _ZDR_CREDENTIAL_HISTORY
) -> bool:
    if live in mapping:
        mapping[live] = mapping.pop(live)
    dropped_alias = False
    while len(mapping) > limit:
        if mapping.pop(next(iter(mapping))) is aliased:
            dropped_alias = True
    return dropped_alias


def _record_zdr_use(touched: OrderedDict[str, float], fingerprint: str, now: float) -> None:
    touched[fingerprint] = now
    touched.move_to_end(fingerprint)


def _content_digest(
    norms: frozenset[str] | set[str],
    specs: dict[str, Any],
) -> str:
    parts: list[str] = []
    for norm_id in sorted(norms):
        spec = specs.get(norm_id)
        if not isinstance(spec, dict):
            parts.append(norm_id)
            continue
        features = ",".join(sorted(str(f) for f in (spec.get("features") or ())))
        capabilities = ",".join(
            f"{key}={value!r}"
            for key, value in sorted((spec.get("capabilities") or {}).items())
        )
        parts.append(
            f"{norm_id}\x1f{spec.get('description') or ''}\x1f{features}\x1f{capabilities}"
        )
    return hashlib.sha256("\x1e".join(parts).encode("utf-8", "surrogatepass")).hexdigest()


def _catalog_timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=_REGISTRY_CATALOG_TIMEOUT_SECONDS)


def _chat_merge_base(specs: dict[str, Any]) -> dict[str, Any]:
    return {
        norm: spec
        for norm, spec in specs.items()
        if isinstance(spec, dict)
        and not (
            "video_model" in spec
            and spec.get("context_length") is None
            and spec.get("max_completion_tokens") is None
        )
    }


def _prior_spec(specs: dict[str, Any], norm_id: str) -> dict[str, Any]:
    prior = specs.get(norm_id)
    return prior if isinstance(prior, dict) else {}


def _chat_features_for_video(prior_features: set[str]) -> set[str]:
    merged = set(prior_features)
    merged.discard("image_output")
    merged.discard("image_gen_tool")
    return merged


class OpenRouterModelRegistry:
    """Fetches and caches the OpenRouter model catalog."""

    _models: ClassVar[list[dict[str, Any]]] = []
    _specs: ClassVar[dict[str, dict[str, Any]]] = {}
    _id_map: ClassVar[dict[str, str]] = {}
    _zdr_model_ids: set[str] | None = None
    _zdr_stamped_specs: dict[str, dict[str, Any]] | None = None
    _zdr_stamp_roster: set[str] | None = None
    _zdr_rosters: ClassVar[dict[str, set[str]]] = {}
    _ZDR_KEY: ContextVar[str | None] = ContextVar(
        "owui_zdr_key_ctx",
        default=None,
    )
    _zdr_attempted_key: ClassVar[str | None] = None
    _zdr_settle: ClassVar[dict[str, tuple[int, float]]] = {}
    _zdr_touched: ClassVar[OrderedDict[str, float]] = OrderedDict()
    _last_fetch: float = 0.0
    _lock: asyncio.Lock = asyncio.Lock()
    _lock_guard: ClassVar[threading.Lock] = threading.Lock()
    _locks: ClassVar[weakref.WeakKeyDictionary[Any, asyncio.Lock]] = (
        weakref.WeakKeyDictionary()
    )
    _next_refresh_after: float = 0.0
    _failure_counts: ClassVar[dict[str, int]] = {}
    _media_failure_counts: ClassVar[dict[str, int]] = {}
    _last_errors: ClassVar[dict[str, str]] = {}
    _last_error: str | None = None
    _last_error_time: float = 0.0
    _name_map: ClassVar[dict[str, str] | None] = None
    _enriched_cache: ClassVar[Any] = None
    _chat_content_digest: str = ""
    _video_content_digest: str = ""
    _image_content_digest: str = ""

    @classmethod
    def content_stamp(cls) -> str:
        return (
            f"{cls._chat_content_digest}|{cls._video_content_digest}"
            f"|{cls._image_content_digest}"
        )

    @classmethod
    def _invalidate_name_map(cls) -> None:
        cls._name_map = None

    @classmethod
    def _touch_zdr(cls, fingerprint: str) -> None:
        _record_zdr_use(cls._zdr_touched, fingerprint, time.time())

    @classmethod
    def _trim_zdr_history(cls, mapping: dict[str, Any], live: str, now: float) -> bool:
        _record_zdr_use(cls._zdr_touched, live, now)
        dropped_alias = False
        for fingerprint in [
            entry
            for entry, stamp in cls._zdr_touched.items()
            if now - stamp > _ZDR_CREDENTIAL_RETENTION_SECONDS
        ]:
            cls._zdr_touched.pop(fingerprint, None)
            if cls._zdr_rosters.pop(fingerprint, None) is cls._zdr_model_ids:
                dropped_alias = True
            cls._zdr_settle.pop(fingerprint, None)
        while len(cls._zdr_touched) > _ZDR_LIVE_CREDENTIAL_CEILING:
            oldest = next(iter(cls._zdr_touched))
            if oldest == live:
                break
            cls._zdr_touched.pop(oldest, None)
            cls._zdr_rosters.pop(oldest, None)
            cls._zdr_settle.pop(oldest, None)
        if live in mapping:
            mapping[live] = mapping.pop(live)
        return dropped_alias

    @classmethod
    def _key_changed(cls, api_key: str) -> bool:
        fp = _fingerprint(api_key)
        if fp in cls._zdr_rosters or fp in cls._zdr_settle:
            cls._touch_zdr(fp)
            return False
        return cls._zdr_attempted_key is not None and cls._zdr_attempted_key != fp

    @classmethod
    def _credential_settle(cls, api_key: str) -> tuple[int, float] | None:
        fp = _fingerprint(api_key)
        stored = cls._zdr_settle.get(fp)
        if stored is not None:
            cls._touch_zdr(fp)
        return stored

    @classmethod
    def _settled_until(cls, api_key: str, cache_seconds: int) -> float:
        entry = cls._credential_settle(api_key)
        if entry is not None:
            tries, clock = entry
            return 0.0 if tries < 2 else clock
        if not cls._key_changed(api_key):
            return cls._next_refresh_after or (cls._last_fetch + cache_seconds)
        return 0.0

    @classmethod
    def _throttled(cls, api_key: str, cache_seconds: int, now: float) -> bool:
        if cls._specs:
            return now < cls._settled_until(api_key, cache_seconds)
        entry = cls._credential_settle(api_key)
        window = entry[1] if entry is not None else 0.0
        return bool(window) and now < window

    @classmethod
    def _record_settle(cls, api_key: str, until: float) -> None:
        prior = cls._credential_settle(api_key)
        fp = _fingerprint(api_key)
        cls._zdr_settle[fp] = ((prior[0] if prior else 0) + 1, until)
        cls._trim_zdr_history(cls._zdr_settle, fp, time.time())

    @classmethod
    def _clear_settle(cls, api_key: str) -> None:
        cls._zdr_settle.pop(_fingerprint(api_key), None)

    @classmethod
    def _zdr_roster_for(cls, api_key: str) -> set[str] | None:
        fp = _fingerprint(api_key)
        stored = cls._zdr_rosters.get(fp)
        if stored is not None:
            cls._touch_zdr(fp)
        return stored

    @classmethod
    def arm_zdr_key(cls, api_key: str | None) -> Any:
        if not api_key:
            return cls._ZDR_KEY.set(None)
        return cls._ZDR_KEY.set(_fingerprint(api_key))

    @classmethod
    def _roster_in_force(cls) -> set[str] | None:
        fp = cls._ZDR_KEY.get()
        if fp is None:
            return cls._zdr_model_ids
        stored = cls._zdr_rosters.get(fp)
        if stored is not None:
            cls._touch_zdr(fp)
        return stored

    @classmethod
    def _stamp_zdr_capable(
        cls,
        spec: dict[str, Any],
        norm_id: str,
        roster: set[str] | None,
        specs: dict[str, dict[str, Any]],
    ) -> None:
        if roster is None:
            spec.pop("zdr_capable", None)
            return
        spec["zdr_capable"] = any(
            key in roster for key in cls._zdr_candidate_keys(norm_id, specs)
        )

    @classmethod
    def _adopt_roster_for(cls, api_key: str) -> None:
        roster = cls._zdr_roster_for(api_key)
        if (
            roster is cls._zdr_stamp_roster
            and cls._specs is cls._zdr_stamped_specs
        ):
            return
        cls._zdr_model_ids = roster
        for norm_id, spec in cls._specs.items():
            cls._stamp_zdr_capable(spec, norm_id, roster, cls._specs)
        cls._enriched_cache = None
        cls._zdr_stamp_roster = roster
        cls._zdr_stamped_specs = cls._specs

    @classmethod
    def _catalog_lock(cls) -> asyncio.Lock:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            return cls._lock
        existing = cls._locks.get(running)
        if existing is not None:
            return existing
        with cls._lock_guard:
            existing = cls._locks.get(running)
            if existing is None:
                existing = asyncio.Lock()
                cls._locks[running] = existing
        return existing

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
        valves: Any = None,
    ) -> None:
        """Refresh the model catalog if the cache is empty or stale."""
        if not api_key:
            raise ValueError("OpenRouter API key is required.")

        cls._ZDR_KEY.set(_fingerprint(api_key))
        now = time.time()
        if cls._throttled(api_key, cache_seconds, now):
            if cls._specs:
                cls._adopt_roster_for(api_key)
                return
            raise RuntimeError(
                cls._last_errors.get(_fingerprint(api_key))
                or "OpenRouter model catalog unavailable"
            )

        async with cls._catalog_lock():
            now = time.time()
            if cls._throttled(api_key, cache_seconds, now):
                if cls._specs:
                    cls._adopt_roster_for(api_key)
                    return
                raise RuntimeError(
                    cls._last_errors.get(_fingerprint(api_key))
                    or "OpenRouter model catalog unavailable"
                )
            rotating = cls._key_changed(api_key)
            prior_attempt = cls._zdr_attempted_key
            cls._zdr_attempted_key = _fingerprint(api_key)
            try:
                await cls._refresh(
                    session,
                    base_url=base_url,
                    api_key=api_key,
                    logger=logger,
                    http_referer=http_referer,
                    valves=valves,
                )
            except Exception as exc:
                cls._zdr_attempted_key = prior_attempt
                cls._record_settle(
                    api_key, cls._record_refresh_failure(exc, cache_seconds, api_key))
                if not cls._models:
                    raise
                if rotating:
                    cls._zdr_model_ids = None
                    for norm_id, spec in cls._specs.items():
                        cls._stamp_zdr_capable(spec, norm_id, None, cls._specs)
                    cls._enriched_cache = None
                    cls._zdr_stamp_roster = None
                    cls._zdr_stamped_specs = None
                logger.warning(
                    "OpenRouter catalog refresh failed (%s). Serving %d cached model(s).",
                    exc,
                    len(cls._models),
                    exc_info=True,
                )
                return
            cls._clear_settle(api_key)
            cls._record_refresh_success(cache_seconds, api_key)

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
        valves: Any = None,
    ) -> None:
        """Fetch and cache the OpenRouter catalog."""
        from ..requests.debug import _debug_print_error_response, _debug_print_request

        url = base_url.rstrip("/") + "/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-OpenRouter-Title": _OPENROUTER_TITLE,
            "X-OpenRouter-Categories": _OPENROUTER_CATEGORIES,
            **openrouter_attribution_headers(http_referer),
        }
        _catalog_read_timeout = _catalog_timeout()
        _debug_print_request(headers, {"method": "GET", "url": url}, logger=logger)
        try:
            async with session.get(url, headers=headers, timeout=_catalog_read_timeout) as resp:
                if resp.status >= 400:
                    await _debug_print_error_response(resp, logger=logger)
                resp.raise_for_status()
                payload = await resp.json()
        except Exception:
            logger.debug("Failed to load OpenRouter model catalog", exc_info=True)
            raise

        data = payload.get("data") or []
        zdr_model_ids: set[str] | None = cls._zdr_roster_for(api_key)
        zdr_read_ok = False
        try:
            zdr_model_ids = await cls._fetch_zdr_model_ids(
                session,
                base_url=base_url,
                api_key=api_key,
                logger=logger,
                http_referer=http_referer,
                timeout=_catalog_read_timeout,
            )
            zdr_read_ok = True
        except Exception as exc:
            logger.warning(
                "Failed to load OpenRouter ZDR endpoint list: %s", exc, exc_info=True
            )
        raw_specs: dict[str, dict[str, Any]] = {}
        models: list[dict[str, Any]] = []
        id_map: dict[str, str] = {}

        for item in data:
            original_id = item.get("id")
            if not original_id:
                continue

            sanitized = sanitize_model_id(original_id)
            norm_id = _catalog_norm(original_id)

            raw_specs[norm_id] = dict(item)

            id_map[cls._exact_norm(sanitized)] = original_id
            models.append(
                {
                    "id": sanitized,
                    "norm_id": norm_id,
                    "original_id": original_id,
                    "name": item.get("name") or original_id,
                }
            )

        specs: dict[str, dict[str, Any]] = {}
        for norm_id, full_model in raw_specs.items():
            supported_parameters = set(full_model.get("supported_parameters") or [])
            architecture = full_model.get("architecture") or {}
            pricing = full_model.get("pricing") or {}

            features = cls._derive_features(
                supported_parameters,
                architecture,
                pricing,
            )

            original_id = full_model.get("id") or norm_id
            blocklisted = is_direct_upload_blocklisted_for(original_id, full_model)
            if blocklisted:
                features.discard("file_input")
            else:
                features.add("file_input")

            capabilities = cls._derive_capabilities(
                architecture,
                pricing,
            )
            capabilities["file_upload"] = not blocklisted

            max_completion_tokens: int | None = None
            top_provider = full_model.get("top_provider")
            if isinstance(top_provider, dict):
                max_completion_tokens = top_provider.get("max_completion_tokens")

            specs[norm_id] = {
                "features": features,
                "capabilities": capabilities,
                "max_completion_tokens": max_completion_tokens,
                "supported_parameters": frozenset(supported_parameters),

                "full_model": full_model,

                "context_length": full_model.get("context_length"),
                "description": full_model.get("description"),
                "architecture": architecture,
                **_base_spec_fields(pricing),
            }

        models.sort(key=lambda m: str(m.get("name") or "").lower())
        if not models:
            raise RuntimeError("OpenRouter returned an empty model catalog.")

        chat_catalog_norms = frozenset(specs)
        existing_video_norms = {
            n for n, s in cls._specs.items()
            if "video_generation" in (s.get("features") or set())
        }
        if existing_video_norms:
            for n in existing_video_norms:
                if n not in specs:
                    specs[n] = cls._specs[n]
            existing_norms_in_models = {m.get("norm_id") for m in models}
            preserved_video_models = [
                m for m in cls._models
                if m.get("norm_id") in existing_video_norms
                and m.get("norm_id") not in existing_norms_in_models
            ]
            for row in preserved_video_models:
                id_map.setdefault(cls._exact_norm(str(row["id"])), str(row["original_id"]))
            if preserved_video_models:
                models.extend(preserved_video_models)
                models.sort(key=lambda m: str(m.get("name") or "").lower())

        existing_image_only_norms = {
            n for n, s in cls._specs.items()
            if "image_output" in (s.get("features") or set())
            and "video_generation" not in (s.get("features") or set())
            and "text" not in ((s.get("architecture") or {}).get("output_modalities") or [])
        }
        existing_image_only_norms = existing_image_only_norms - set(cls._chat_catalog_norms)
        if existing_image_only_norms:
            for n in existing_image_only_norms:
                if n not in specs:
                    specs[n] = cls._specs[n]
            existing_norms_after_video = {m.get("norm_id") for m in models}
            preserved_image_models = [
                m for m in cls._models
                if m.get("norm_id") in existing_image_only_norms
                and m.get("norm_id") not in existing_norms_after_video
            ]
            for row in preserved_image_models:
                id_map.setdefault(cls._exact_norm(str(row["id"])), str(row["original_id"]))
            if preserved_image_models:
                models.extend(preserved_image_models)
                models.sort(key=lambda m: str(m.get("name") or "").lower())

        for norm_id in cls._video_catalog_norms & specs.keys():
            prior = cls._specs.get(norm_id)
            video_item = (prior or {}).get("video_model")
            if isinstance(video_item, dict):
                specs[norm_id] = cls._merge_video_spec(specs[norm_id], video_item)

        for norm_id, spec in specs.items():
            cls._stamp_zdr_capable(spec, norm_id, zdr_model_ids, specs)

        cls._models = models
        cls._specs = specs
        cls._id_map = id_map
        cls._invalidate_name_map()
        if zdr_read_ok:
            fp = _fingerprint(api_key)
            cls._zdr_rosters[fp] = set(zdr_model_ids or ())
            if cls._trim_zdr_history(cls._zdr_rosters, fp, time.time()):
                cls._zdr_model_ids = None
        cls._zdr_model_ids = cls._zdr_roster_for(api_key)
        cls._zdr_stamp_roster = cls._zdr_model_ids
        cls._zdr_stamped_specs = cls._specs
        cls._chat_catalog_norms = chat_catalog_norms
        cls._chat_content_digest = _content_digest(chat_catalog_norms, specs)
        ModelFamily.set_dynamic_specs(specs)

    @classmethod
    def _record_refresh_success(cls, cache_seconds: int, api_key: str) -> None:
        """Reset refresh backoff bookkeeping after a successful catalog fetch."""
        now = time.time()
        cls._last_fetch = now
        cls._next_refresh_after = now + max(5, cache_seconds)
        cls._failure_counts.pop(_fingerprint(api_key), None)
        cls._last_errors.pop(_fingerprint(api_key), None)
        cls._last_error = None
        cls._last_error_time = 0.0

    @classmethod
    def _record_refresh_failure(
        cls, exc: Exception, cache_seconds: int, api_key: str
    ) -> float:
        """Increase backoff delay and track the most recent catalog error."""
        key = _fingerprint(api_key)
        cls._failure_counts[key] = cls._failure_counts.get(key, 0) + 1
        failures = cls._failure_counts[key]
        cls._last_errors[key] = str(exc)
        _trim_credential_history(cls._failure_counts, key, None)
        _trim_credential_history(cls._last_errors, key, None)
        cls._last_error = str(exc)
        cls._last_error_time = time.time()
        exponent = min(failures - 1, 5)
        base_backoff = 5.0
        raw_backoff = base_backoff * (2 ** exponent)
        capped_backoff = min(cache_seconds, raw_backoff)
        backoff_until = cls._last_error_time + max(base_backoff, capped_backoff)
        return backoff_until

    @classmethod
    def _media_retry_window(cls, cache_seconds: int, key: str) -> float:
        failures = cls._media_failure_counts.get(key, 0)
        if failures <= 0:
            return float(cache_seconds)
        exponent = min(failures - 1, 5)
        base_backoff = 5.0
        raw_backoff = base_backoff * (2 ** exponent)
        return float(min(cache_seconds, max(base_backoff, raw_backoff)))

    @classmethod
    def record_media_failure(cls, api_key: str) -> None:
        key = _fingerprint(api_key)
        cls._media_failure_counts[key] = cls._media_failure_counts.get(key, 0) + 1
        _trim_credential_history(cls._media_failure_counts, key, None)

    @classmethod
    def record_media_success(cls, api_key: str) -> None:
        key = _fingerprint(api_key)
        cls._media_failure_counts.pop(key, None)
        _trim_credential_history(cls._media_failure_counts, key, None)

    @staticmethod
    def _derive_features(
        supported_parameters: set[str],
        architecture: dict[str, Any],
        pricing: dict[str, Any],
    ) -> set[str]:
        """Translate OpenRouter metadata into capability flags.

        Features include:
        - function_calling: Model supports tools/function calling
        - reasoning: Model supports extended reasoning
        - reasoning_summary: Model supports reasoning summaries
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

        if is_image_output_architecture(architecture):
            features.add("image_gen_tool")
            features.add("image_output")

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
    def _coerce_pricing_number(value: Any) -> Decimal | None:
        """Return a numeric pricing value when possible."""
        if value is None:
            return None
        if isinstance(value, (int, float, Decimal)):
            try:
                return Decimal(str(value))
            except (InvalidOperation, ValueError):
                return None
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            try:
                return Decimal(raw)
            except (InvalidOperation, ValueError):
                return None
        if isinstance(value, dict):
            for key in ("price", "amount", "value", "usd", "cost", "rate"):
                if key in value:
                    return OpenRouterModelRegistry._coerce_pricing_number(value.get(key))
            return None
        if isinstance(value, (list, tuple, set)):
            candidates = [
                OpenRouterModelRegistry._coerce_pricing_number(entry)
                for entry in value
            ]
            numeric = [c for c in candidates if c is not None]
            return max(numeric) if numeric else None
        return None

    @staticmethod
    def _supports_web_search(pricing: dict[str, Any]) -> bool:
        """Return True when the provider exposes paid web-search support."""
        value = pricing.get("web_search")
        amount = OpenRouterModelRegistry._coerce_pricing_number(value)
        if amount is None:
            return False
        return amount > Decimal(0)

    @staticmethod
    def _derive_capabilities(
        architecture: dict[str, Any],
        pricing: dict[str, Any],
    ) -> dict[str, bool]:
        """Translate metadata into Open WebUI capability checkboxes."""

        def _normalize(values: Any) -> set[str]:
            """Return a normalized lowercase set from the provider metadata."""
            if isinstance(values, str):
                values = values.split(",")
            if not isinstance(values, list):
                return set()
            normalized: set[str] = set()
            for item in values:
                if isinstance(item, str):
                    normalized.add(item.strip().lower())
            return normalized

        input_modalities = _normalize(architecture.get("input_modalities") or [])

        vision_capable = "image" in input_modalities
        file_upload_capable = True
        image_generation_capable = is_image_output_architecture(architecture)
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
    def _enriched_models(cls) -> list[dict[str, Any]]:
        roster = cls._roster_in_force()
        cached = cls._enriched_cache
        if (
            cached is not None
            and cached[0] is cls._models
            and cached[1] is cls._specs
            and cached[3] is roster
        ):
            return cached[2]
        enriched: list[dict[str, Any]] = []
        for model in cls._models:
            item = dict(model)
            spec = cls._specs.get(model["norm_id"])
            if spec and spec.get("capabilities"):
                item["capabilities"] = dict(spec["capabilities"])
            if roster is not None:
                item["zdr_capable"] = any(
                    key in roster
                    for key in cls._zdr_candidate_keys(model["norm_id"], cls._specs)
                )
            enriched.append(item)
        cls._enriched_cache = (cls._models, cls._specs, enriched, roster)
        return enriched

    @classmethod
    def list_models(cls) -> list[dict[str, Any]]:
        published: list[dict[str, Any]] = []
        for item in cls._enriched_models():
            copy = dict(item)
            capabilities = copy.get("capabilities")
            if isinstance(capabilities, dict):
                copy["capabilities"] = dict(capabilities)
            published.append(copy)
        return published

    _last_video_fetch: float = 0.0
    _last_video_attempt: float = 0.0
    _last_video_account: str = ""
    _last_video_modality_attempt: float = 0.0
    _video_catalog_norms: frozenset[str] = frozenset()

    @classmethod
    def last_video_fetch(cls) -> float:
        """Return the timestamp of the last successful video catalog registration.

        Used by `maybe_schedule_model_metadata_sync` to detect when video models
        appeared in `cls._models` so the metadata sync re-runs and writes per-model
        rows + auto-attaches the per-model video filters.
        """
        return cls._last_video_fetch

    @classmethod
    def last_video_attempt(cls) -> float:
        """Return the timestamp of the most recent video catalog fetch attempt.

        Bumped on EVERY fetch outcome (success-non-empty, success-empty, error)
        so the loader's TTL gate engages even when the endpoint is down or
        reports zero models. `_last_video_fetch` (data-change) and this
        attempt clock are separate so `catalog_manager.sync_key` doesn't
        invalidate on empty/error fetches.
        """
        return cls._last_video_attempt

    @classmethod
    def record_video_attempt(cls, api_key: str) -> None:
        """Stamp `_last_video_attempt` with the current time."""
        cls._last_video_attempt = time.time()
        cls._last_video_account = _fingerprint(api_key)

    @classmethod
    def video_accounts_match(cls, api_key: str) -> bool:
        return _fingerprint(api_key) == cls._last_video_account

    @classmethod
    def video_catalog_is_published(cls) -> bool:
        return bool(cls._video_catalog_norms)

    @classmethod
    def reset_video_attempt(cls) -> None:
        cls._last_video_attempt = 0.0
        cls._last_video_account = ""
        cls._media_failure_counts.clear()

    @classmethod
    def last_video_modality_attempt(cls) -> float:
        return cls._last_video_modality_attempt

    @classmethod
    def record_video_modality_attempt(cls) -> None:
        cls._last_video_modality_attempt = time.time()

    @classmethod
    def reset_video_modality_attempt(cls) -> None:
        cls._last_video_modality_attempt = 0.0

    @classmethod
    def reset_video_fetch_timestamp(cls) -> None:
        cls._last_video_fetch = 0.0

    @classmethod
    def _merge_video_spec(cls, chat_spec: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
        prior = chat_spec
        original_id = str(item.get("id") or "").strip()
        prior_features = _chat_features_for_video(
            set(prior.get("features") or set()),
        )
        prior_architecture = prior.get("architecture")
        prior_architecture = prior_architecture if isinstance(prior_architecture, dict) else {}

        supported_frames = item.get("supported_frame_images")
        accepts_frame_images = isinstance(supported_frames, list) and bool(supported_frames)
        accepts_uploads = accepts_frame_images and not is_direct_upload_blocklisted_for(original_id, item)
        allowed_params = item.get("allowed_passthrough_parameters")
        allowed_set = {
            param
            for param in allowed_params
            if isinstance(param, str) and param
        } if isinstance(allowed_params, list) else set()
        pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
        features = {"video_generation", "video_output"}
        if accepts_uploads:
            features.update({"vision", "file_input"})
        features |= prior_features

        item_architecture = item.get("architecture")
        item_architecture = item_architecture if isinstance(item_architecture, dict) else {}
        if prior_architecture:
            architecture = dict(prior_architecture)
            for key, value in item_architecture.items():
                architecture.setdefault(key, value)
            chat_out = prior_architecture.get("output_modalities")
            video_out = item_architecture.get("output_modalities")
            if isinstance(chat_out, list) and isinstance(video_out, list):
                architecture["output_modalities"] = list(chat_out) + [
                    m for m in video_out if m not in chat_out
                ]
        else:
            architecture = dict(item_architecture)

        capabilities = dict(prior.get("capabilities") or {})
        capabilities["vision"] = "vision" in features
        capabilities["file_upload"] = "file_input" in features
        for capability, value in {
            "web_search": False,
            "image_generation": is_image_output_architecture(architecture),
            "video_generation": True,
            "code_interpreter": False,
            "citations": False,
            "status_updates": True,
            "usage": True,
        }.items():
            capabilities.setdefault(capability, value)

        full_model = dict(item)
        full_model.update(dict(prior.get("full_model") or {}))

        return {
            "features": features,
            "capabilities": capabilities,
            "max_completion_tokens": prior.get("max_completion_tokens"),
            "supported_parameters": frozenset(allowed_set),
            "full_model": full_model,
            "video_model": dict(item),
            "context_length": prior.get("context_length"),
            "description": prior.get("description") or item.get("description"),
            "architecture": architecture,
            **_base_spec_fields(prior.get("pricing") or pricing, spec={"video_model": item}),
        }

    @classmethod
    def register_video_models(cls, video_models: list[dict[str, Any]]) -> None:
        """Register OpenRouter async video-generation models as selectable models."""
        if not isinstance(video_models, list):
            return

        new_specs = dict(cls._specs)
        new_id_map = dict(cls._id_map)
        chat_specs = _chat_merge_base(cls._specs)

        old_video_norms = (
            set(cls._video_catalog_norms)
            if isinstance(video_models, list) and not video_models
            else {
                norm_id
                for norm_id, spec in new_specs.items()
                if "video_generation" in set(spec.get("features") or set())
            }
        )
        chat_owned = {n for n in old_video_norms if n in cls._chat_catalog_norms}
        retired_video_norms = old_video_norms - chat_owned
        if old_video_norms:
            for norm_id in retired_video_norms:
                new_specs.pop(norm_id, None)
            for model in cls._models:
                if isinstance(model, dict) and model.get("norm_id") in retired_video_norms:
                    new_id_map.pop(cls._exact_norm(str(model.get("id") or "")), None)

        for norm_id in chat_owned:
            merged = new_specs.get(norm_id)
            if not isinstance(merged, dict):
                continue
            full = merged.get("full_model")
            full = full if isinstance(full, dict) else {}
            features = cls._derive_features(
                set(full.get("supported_parameters") or set()),
                full.get("architecture") or {},
                full.get("pricing") or {},
            )
            blocklisted = is_direct_upload_blocklisted_for(str(full.get("id") or ""), full)
            if blocklisted:
                features.discard("file_input")
            else:
                features.add("file_input")
            capabilities = cls._derive_capabilities(
                full.get("architecture") or {}, full.get("pricing") or {}
            )
            capabilities["file_upload"] = not blocklisted
            capabilities["video_generation"] = False
            restored = dict(merged)
            restored["features"] = features
            restored["capabilities"] = capabilities
            restored["supported_parameters"] = frozenset(full.get("supported_parameters") or set())
            restored.pop("video_model", None)
            restored["description"] = full.get("description")
            restored["architecture"] = dict(full.get("architecture") or {})
            restored.update(_base_spec_fields(full.get("pricing") or {}))
            new_specs[norm_id] = restored

        models_by_norm: dict[str, dict[str, Any]] = {}
        for model in cls._models:
            if not isinstance(model, dict):
                continue
            norm = model.get("norm_id")
            if isinstance(norm, str) and norm and norm not in retired_video_norms:
                models_by_norm[cls._exact_norm(str(model.get("id") or ""))] = dict(model)

        owned_video_norms: set[str] = set()
        for item in video_models:
            if not isinstance(item, dict):
                continue
            original_id = item.get("id")
            if not isinstance(original_id, str) or not original_id.strip():
                continue
            original_id = original_id.strip()
            sanitized = sanitize_model_id(original_id)
            norm_id = _catalog_norm(original_id)
            if not norm_id:
                continue

            prior = _prior_spec(chat_specs, norm_id)
            merged = cls._merge_video_spec(prior, item)
            full_model = merged["full_model"]

            new_id_map[cls._exact_norm(sanitized)] = original_id
            row_name = full_model.get("name")
            if not isinstance(row_name, str) or not row_name.strip():
                row_name = None
            models_by_norm[cls._exact_norm(sanitized)] = {
                "id": sanitized,
                "norm_id": norm_id,
                "original_id": original_id,
                "name": row_name or item.get("name") or original_id,
            }
            owned_video_norms.add(norm_id)
            new_specs[norm_id] = merged
            cls._stamp_zdr_capable(
                new_specs[norm_id], norm_id, cls._roster_in_force(), new_specs
            )

        cls._specs = new_specs
        cls._id_map = new_id_map
        cls._models = sorted(models_by_norm.values(), key=lambda m: str(m.get("name") or "").lower())
        cls._invalidate_name_map()
        cls._video_catalog_norms = frozenset(owned_video_norms)
        ModelFamily.set_dynamic_specs(cls._specs)
        if video_models:
            cls._last_video_fetch = time.time()
            cls._video_content_digest = _content_digest(
                cls._video_catalog_norms, cls._specs
            )
        else:
            cls._video_content_digest = ""

    @classmethod
    def _image_endpoint_aliases(cls) -> dict[str, list[dict[str, Any]]]:
        current = cls._image_endpoints
        if cls._image_endpoint_alias_of is not current:
            cls._image_endpoint_alias = _build_image_endpoint_aliases(current)
            cls._image_endpoint_alias_of = current
        return cls._image_endpoint_alias

    _last_image_fetch: float = 0.0
    _last_image_attempt: float = 0.0
    _last_image_account: ClassVar[dict[tuple[str, str] | None, str]] = {}
    _image_catalog_norms: frozenset[str] = frozenset()
    _chat_catalog_norms: frozenset[str] = frozenset()
    _image_endpoints: ClassVar[dict[str, list[dict[str, Any]]]] = {}
    _image_endpoints_by_target: ClassVar[
        dict[tuple[str, str] | None, dict[str, list[dict[str, Any]]]]
    ] = {}
    _image_endpoint_alias: ClassVar[dict[str, list[dict[str, Any]]]] = {}
    _image_endpoint_alias_of: ClassVar[dict[str, list[dict[str, Any]]] | None] = None
    _last_image_contract_attempt: ClassVar[dict[tuple[str, str] | None, float]] = {}
    _last_image_contract_account: ClassVar[dict[tuple[str, str] | None, str]] = {}
    _image_contract_retry_after: float = 0.0
    _image_contract_owed: ClassVar[dict[tuple[str, str] | None, frozenset[str]]] = {}
    _image_contract_target: ClassVar[tuple[str, str] | None] = None

    @classmethod
    def _image_bucket(cls, target: tuple[str, str] | None) -> dict[str, list[dict[str, Any]]]:
        if target is None:
            target = cls._image_contract_target
        bucket = cls._image_endpoints_by_target.get(target)
        if bucket is not None:
            return bucket
        if target == cls._image_contract_target or cls._image_contract_target is None:
            return cls._image_endpoints
        return {}

    @classmethod
    def image_contract_target(cls) -> tuple[str, str] | None:
        return cls._image_contract_target

    @classmethod
    def adopt_image_contract_target(cls, identity: tuple[str, str]) -> bool:
        if cls._image_contract_target == identity:
            return False
        departing_had_contracts = bool(cls._image_endpoints)
        cls._image_contract_target = identity
        bucket = cls._image_endpoints_by_target.get(identity)
        if bucket is None:
            bucket = {}
            cls._image_endpoints_by_target[identity] = bucket
            _trim_credential_history(
                cls._image_endpoints_by_target,
                identity,
                None,
                _IMAGE_CONTRACT_TARGET_HISTORY,
            )
        cls._image_endpoints = bucket
        cls._image_endpoint_alias = {}
        cls._image_endpoint_alias_of = None
        cls._image_contract_owed[identity] = frozenset()
        cls._image_contract_retry_after = 0.0
        return departing_had_contracts and not bucket

    @classmethod
    def last_image_contract_attempt(cls, target: tuple[str, str] | None = None) -> float:
        """When the contract sweep last ran, separately from the model-list fetch.

        A caller that skipped the sweep must not satisfy the freshness check that guards
        it: a chat request refreshes the model list without reading contracts, and would
        otherwise suppress the next catalog refresh's sweep for a whole TTL window --
        leaving every model with no controls.
        """
        if target is None:
            target = cls._image_contract_target
        return cls._last_image_contract_attempt.get(target, 0.0)

    @classmethod
    def record_image_contract_attempt(
        cls, api_key: str, target: tuple[str, str] | None = None
    ) -> None:
        if target is None:
            target = cls._image_contract_target
        cls._last_image_contract_attempt[target] = time.time()
        cls._last_image_contract_account[target] = _fingerprint(api_key)
        cls._last_image_account[target] = cls._last_image_contract_account[target]

    @classmethod
    def clear_image_contract_attempt(cls, target: tuple[str, str] | None = None) -> None:
        if target is None:
            target = cls._image_contract_target
        cls._last_image_contract_attempt.pop(target, None)
        cls._last_image_contract_account.pop(target, None)

    @classmethod
    def image_accounts_match(cls, api_key: str, target: tuple[str, str] | None = None) -> bool:
        if target is None:
            target = cls._image_contract_target
        fingerprint = _fingerprint(api_key)
        return (
            fingerprint
            == cls._last_image_account.get(target, "")
            == cls._last_image_contract_account.get(target, "")
        )

    @classmethod
    def mark_image_contract_retry(cls, cache_seconds: int) -> None:
        cls._image_contract_retry_after = time.time() + cache_seconds

    @classmethod
    def image_contract_retry_pending(cls) -> bool:
        return time.time() < cls._image_contract_retry_after

    @classmethod
    def clear_image_contract_retry(cls) -> None:
        cls._image_contract_retry_after = 0.0

    @classmethod
    def image_contract_owed(cls, target: tuple[str, str] | None = None) -> frozenset[str]:
        if target is None:
            target = cls._image_contract_target
        return cls._image_contract_owed.get(target, frozenset())

    @classmethod
    def set_image_contract_owed(
        cls, model_ids: frozenset[str], target: tuple[str, str] | None = None
    ) -> None:
        if target is None:
            target = cls._image_contract_target
        cls._image_contract_owed[target] = frozenset(model_ids)

    @classmethod
    def clear_image_contract_owed(cls, target: tuple[str, str] | None = None) -> None:
        if target is None:
            target = cls._image_contract_target
        cls._image_contract_owed[target] = frozenset()

    @classmethod
    def set_image_endpoints(
        cls,
        records: dict[str, Any] | None,
        *,
        known_ids: set[str] | None = None,
        target: tuple[str, str] | None = None,
    ) -> None:
        """Hold every published image contract, keyed by the model's OpenRouter id.

        Kept here rather than on the model spec because ``register_image_models`` only
        registers pure-image models -- a text+image model such as Gemini Flash Image
        lives in the chat catalog, so a contract stored on its spec would have nowhere
        to be written and its filter would silently offer nothing.
        """
        if target is None:
            target = cls._image_contract_target
        published: dict[str, list[dict[str, Any]]] = {}
        for key, value in (records or {}).items():
            if not isinstance(key, str) or not key.strip():
                continue
            offered = value if isinstance(value, list) else [value]
            kept = [item for item in offered if isinstance(item, dict)]
            if kept:
                published[key.strip()] = kept

        # A read that timed out is not a contract that changed. Keeping the previous
        # record means one slow response does not strip a model of its controls for a
        # whole refresh cycle -- which is what the adapter already does with its own
        # cache, so the two now agree. ``known_ids`` bounds the carry-over to models the
        # catalog still lists, so a retired model's contract does not live forever.
        for key, previous in cls._image_bucket(target).items():
            if key in published:
                continue
            if known_ids is not None and key not in known_ids:
                continue
            published[key] = previous
        cls._image_endpoints_by_target[target] = published
        _trim_credential_history(
            cls._image_endpoints_by_target, target, None, _IMAGE_CONTRACT_TARGET_HISTORY
        )
        if target == cls._image_contract_target:
            cls._image_endpoints = published
            cls._image_endpoint_alias = {}
            cls._image_endpoint_alias_of = None

    @classmethod
    def listed_model_ids(cls) -> set[str]:
        return {value for value in cls._id_map.values() if isinstance(value, str) and value.strip()}

    @classmethod
    def image_endpoint(
        cls, model_id: str, target: tuple[str, str] | None = None
    ) -> list[dict[str, Any]] | None:
        """Return every published contract for a model, by original or sanitized id."""
        if not isinstance(model_id, str) or not model_id.strip():
            return None
        wanted = model_id.strip()
        bucket = cls._image_bucket(target)
        record = bucket.get(wanted)
        if record is not None:
            return record
        aliases = (
            cls._image_endpoint_aliases()
            if bucket is cls._image_endpoints
            else _build_image_endpoint_aliases(bucket)
        )
        return aliases.get(wanted)

    @classmethod
    def last_image_fetch(cls) -> float:
        """Return the timestamp of the last successful image catalog registration.

        Used by `maybe_schedule_model_metadata_sync` to detect when image-only
        models appeared in `cls._models` so the metadata sync re-runs and
        auto-attaches the per-model image filters.
        """
        return cls._last_image_fetch

    @classmethod
    def last_image_attempt(cls) -> float:
        """Return the timestamp of the most recent image catalog fetch attempt."""
        return cls._last_image_attempt

    @classmethod
    def record_image_attempt(cls, api_key: str, target: tuple[str, str] | None = None) -> None:
        """Stamp `_last_image_attempt` with the current time."""
        if target is None:
            target = cls._image_contract_target
        cls._last_image_attempt = time.time()
        fingerprint = _fingerprint(api_key)
        cls._last_image_account[target] = fingerprint
        cls._last_image_contract_account[target] = fingerprint

    @classmethod
    def reset_image_attempt(cls, target: tuple[str, str] | None = None) -> None:
        if target is None:
            target = cls._image_contract_target
        cls._last_image_attempt = 0.0
        cls._last_image_account.pop(target, None)
        cls._media_failure_counts.clear()

    @classmethod
    def reset_image_fetch_timestamp(cls) -> None:
        """Reset `_last_image_fetch` to 0.

        Called by `image_catalog.ensure_image_catalog_loaded` after a master-disable
        cleanup so subsequent disabled-state calls short-circuit (the cleanup is
        idempotent but the redundant call would still log "cleared" each pass).
        Mirror of how the chat catalog and video catalog avoid repeated cleanups.
        """
        cls._last_image_fetch = 0.0

    @classmethod
    def register_image_models(
        cls,
        image_models: list[dict[str, Any]],
    ) -> None:
        """Register OpenRouter pure-image-only models as selectable models.

        Multimodal text+image models (output_modalities ⊇ {"text"}) already live
        in the chat catalog with `image_gen_tool` set by `_derive_features`;
        they are NOT touched here. Pure-image-only models (output_modalities
        without "text") are absent from `/api/v1/models` and registered here
        with `image_output` + `image_gen_tool` features.

        Stale-norm cleanup mirrors `register_video_models`: any previously-
        registered image-only norm that's no longer in the input list is
        dropped from `_specs`/`_id_map`/`_models`. Chat models and video
        models are preserved.
        """
        if not isinstance(image_models, list):
            return

        new_specs = dict(cls._specs)
        new_id_map = dict(cls._id_map)

        old_image_norms = (
            set(cls._image_catalog_norms)
            if isinstance(image_models, list) and not image_models
            else {
                norm_id
                for norm_id, spec in new_specs.items()
                if (
                    "image_output" in set(spec.get("features") or set())
                    and "video_generation" not in set(spec.get("features") or set())
                    and "text" not in (
                        (spec.get("architecture") or {}).get("output_modalities") or []
                    )
                )
            }
        )
        old_image_norms = old_image_norms - set(cls._chat_catalog_norms)
        if old_image_norms:
            for norm_id in old_image_norms:
                new_specs.pop(norm_id, None)
            for model in cls._models:
                if isinstance(model, dict) and model.get("norm_id") in old_image_norms:
                    new_id_map.pop(cls._exact_norm(str(model.get("id") or "")), None)

        models_by_norm: dict[str, dict[str, Any]] = {}
        for model in cls._models:
            if not isinstance(model, dict):
                continue
            norm = model.get("norm_id")
            if isinstance(norm, str) and norm and norm not in old_image_norms:
                models_by_norm[cls._exact_norm(str(model.get("id") or ""))] = dict(model)

        owned_image_norms: set[str] = set()
        for item in image_models:
            if not isinstance(item, dict):
                continue
            original_id = item.get("id")
            if not isinstance(original_id, str) or not original_id.strip():
                continue
            original_id = original_id.strip()
            sanitized = sanitize_model_id(original_id)
            norm_id = _catalog_norm(original_id)
            if not norm_id:
                continue

            if norm_id in new_specs and cls._exact_norm(sanitized) in new_id_map:
                continue

            arch_raw = item.get("architecture")
            architecture: dict[str, Any] = arch_raw if isinstance(arch_raw, dict) else {}
            input_modalities = architecture.get("input_modalities") or []
            output_modalities = architecture.get("output_modalities") or []
            accepts_image_input = "image" in input_modalities

            if "text" in output_modalities:
                continue

            allowed_params = item.get("supported_parameters")
            allowed_set = {
                param
                for param in allowed_params
                if isinstance(param, str) and param
            } if isinstance(allowed_params, list) else set()
            pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}

            upload_blocked = is_direct_upload_blocklisted_for(original_id, item)

            features: set[str] = {"image_output", "image_gen_tool"}
            if accepts_image_input and not upload_blocked:
                features.update({"vision", "file_input"})

            owned_image_norms.add(norm_id)
            new_id_map[cls._exact_norm(sanitized)] = original_id
            models_by_norm[cls._exact_norm(sanitized)] = {
                "id": sanitized,
                "norm_id": norm_id,
                "original_id": original_id,
                "name": item.get("name") or original_id,
            }
            new_specs[norm_id] = {
                "features": features,
                "capabilities": {
                    "vision": accepts_image_input,
                    "file_upload": accepts_image_input
                    and not upload_blocked,
                    "web_search": False,
                    "image_generation": True,
                    "video_generation": False,
                    "code_interpreter": False,
                    "citations": False,
                    "status_updates": True,
                    "usage": True,
                },
                "max_completion_tokens": None,
                "supported_parameters": frozenset(allowed_set),
                "full_model": dict(item),
                "image_model": dict(item),
                "context_length": item.get("context_length"),
                "description": item.get("description"),
                "architecture": architecture,
                **_base_spec_fields(pricing),
            }
            cls._stamp_zdr_capable(
                new_specs[norm_id], norm_id, cls._roster_in_force(), new_specs
            )

        cls._specs = new_specs
        cls._id_map = new_id_map
        cls._models = sorted(models_by_norm.values(), key=lambda m: str(m.get("name") or "").lower())
        cls._invalidate_name_map()
        cls._image_catalog_norms = frozenset(owned_image_norms)
        ModelFamily.set_dynamic_specs(cls._specs)
        if image_models:
            cls._last_image_fetch = time.time()
            cls._image_content_digest = _content_digest(
                cls._image_catalog_norms, cls._specs
            )
        else:
            cls._image_content_digest = ""

    @classmethod
    def _exact_norm(cls, model_id: str) -> str:
        m = (model_id or "").strip()

        suffix = ""
        if ":" in m:
            m, suffix = m.rsplit(":", 1)

        if "/" in m:
            m = m.replace("/", ".")

        if suffix:
            return f"{m.lower()}:{suffix}"
        return m.lower()

    @classmethod
    def api_model_id(cls, model_id: str) -> str | None:
        """Map sanitized Open WebUI ids back to provider ids, preserving variant/preset suffix.

        Examples:
            - "openai.gpt-4o" -> "openai/gpt-4o"
            - "openai.gpt-4o:exacto" -> "openai/gpt-4o:exacto"
            - "anthropic.claude-opus:extended" -> "anthropic/claude-opus:extended"
            - "openai.gpt-4o:preset/email-copywriter" -> "openai/gpt-4o@preset/email-copywriter"
        """
        suffix_tag = ""
        if ":" in model_id:
            parts = model_id.rsplit(":", 1)
            model_id_base = parts[0]
            suffix_tag = parts[1]
        else:
            model_id_base = model_id

        # Normalize base and lookup
        norm = ModelFamily.base_model(model_id_base)
        provider_id = cls._id_map.get(cls._exact_norm(model_id_base))
        if provider_id is None:
            _pid = ModelFamily._PIPE_ID.get()
            _bare = model_id_base.removeprefix(f"{_pid}.") if _pid else model_id_base
            if _bare != model_id_base:
                provider_id = cls._id_map.get(cls._exact_norm(_bare))
        if provider_id is None:
            provider_id = cls._id_map.get(norm)

        if not provider_id:
            _pid = ModelFamily._PIPE_ID.get()
            if _pid and model_id_base.startswith(f"{_pid}."):
                _rest = model_id_base[len(_pid) + 1:]
                if _rest and "." in _rest and not _rest.startswith(f"{_pid}."):
                    model_id_base = _rest
            if "/" in model_id_base:
                api_base = model_id_base
            elif "." in model_id_base:
                api_base = model_id_base.replace(".", "/", 1)
            else:
                api_base = model_id_base

            if suffix_tag:
                if suffix_tag.startswith("preset/"):
                    return f"{api_base}@{suffix_tag}"
                else:
                    return f"{api_base}:{suffix_tag}"
            return api_base

        if suffix_tag:
            if suffix_tag.startswith("preset/"):
                return f"{provider_id}@{suffix_tag}"
            else:
                return f"{provider_id}:{suffix_tag}"
        return provider_id

    @classmethod
    def spec(cls, model_id: str) -> dict[str, Any]:
        """Return the cached spec for ``model_id`` (or an empty dict)."""
        norm = ModelFamily.base_model(model_id)
        return cls._specs.get(ModelFamily._resolve_spec_key(norm, cls._specs)) or {}

    @classmethod
    def zdr_model_ids(cls) -> set[str] | None:
        """Return cached ZDR-capable model ids (or None if unavailable)."""
        in_force = cls._roster_in_force()
        if in_force is None:
            return None
        return set(in_force)

    @classmethod
    def zdr_list_available(cls) -> bool:
        """Whether the ZDR endpoint list has been loaded at all.

        Separate from `is_zdr_capable`, which answers about one model and returns None
        for the same condition. Callers that only need to know whether filtering can run
        used to build a full `set()` copy of the list and test it for None.
        """
        return cls._roster_in_force() is not None

    @classmethod
    def _zdr_candidate_keys(cls, norm: str, specs: dict[str, dict[str, Any]]) -> list[str]:
        keys: list[str] = []
        for source in (ModelFamily._resolve_spec_key(norm, specs), norm):
            if source in keys:
                continue
            keys.append(source)
            suffix_base, separator, _suffix = norm.rpartition(":")
            if separator and source == norm and suffix_base in specs:
                continue
            full = (specs.get(source) or {}).get("full_model") or {}
            target = full.get("alias_target")
            slug = target.get("slug") if isinstance(target, dict) else None
            if isinstance(slug, str) and slug.strip():
                hop = ModelFamily.base_model(sanitize_model_id(slug.strip()))
                if hop in specs and hop not in keys:
                    keys.append(hop)
        return keys

    @classmethod
    def is_zdr_capable(cls, model_id: str) -> bool | None:
        """Return True/False if ZDR list is available, otherwise None.

        Matched against the suffix-stripped BASE id, because OpenRouter's ZDR endpoint
        list only ever contains base ids: a routing variant such as `:nitro`, `:floor`
        or `:online` selects how the same base model is routed, not a different model,
        and its endpoints are the base model's endpoints.

        The strip lives here rather than at the call sites. It used to be done by the
        ZDR_ENFORCE gate alone -- issue #56 -- so a ZDR-capable model was accepted for
        enforcement and simultaneously rejected by ZDR_MODELS_ONLY, which passed the
        full variant id and got False. One rule in one place is the only arrangement in
        which the three gates cannot disagree.
        """
        norm = ModelFamily.base_model(model_id)
        # The catalog decides, not the string. `:free` and `:thinking` are real,
        # separately-listed models -- 24 of 372 ids in the last dump -- served by
        # different providers under their own retention policies, so answering for them
        # from the paid base would show non-ZDR models under a ZDR-only filter. The base
        # fallback is for the ids the catalog does NOT know: the routing variants the
        # pipe itself synthesises (`:nitro`, `:floor`, `:online`), whose endpoints ARE
        # the base model's.
        in_force = cls._roster_in_force()
        if in_force is None:
            return None
        return any(key in in_force for key in cls._zdr_candidate_keys(norm, cls._specs))

    @classmethod
    @timed
    async def _fetch_zdr_model_ids(
        cls,
        session: aiohttp.ClientSession,
        *,
        base_url: str,
        api_key: str,
        logger: logging.Logger,
        http_referer: str | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> set[str]:
        """Fetch OpenRouter ZDR endpoints and return normalized model ids."""
        from ..requests.debug import _debug_print_error_response, _debug_print_request

        url = base_url.rstrip("/") + "/endpoints/zdr"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-OpenRouter-Title": _OPENROUTER_TITLE,
            "X-OpenRouter-Categories": _OPENROUTER_CATEGORIES,
            **openrouter_attribution_headers(http_referer),
        }
        _debug_print_request(headers, {"method": "GET", "url": url}, logger=logger)
        async with session.get(
            url,
            headers=headers,
            timeout=timeout if timeout is not None else _catalog_timeout(),
        ) as resp:
            if resp.status >= 400:
                await _debug_print_error_response(resp, logger=logger)
            resp.raise_for_status()
            payload = await resp.json()

        data = payload.get("data") or []
        model_ids: set[str] = set()
        for item in data:
            model_id = item.get("model_id") or item.get("id")
            if not model_id:
                continue
            norm = _catalog_norm(model_id)
            if norm:
                model_ids.add(norm)
        return model_ids

# Additional Model Helper Functions

def normalize_model_id_dotted(model_id: str) -> str:
    """Return a dotted variant of a model id (e.g. 'anthropic/claude' -> 'anthropic.claude')."""
    if not isinstance(model_id, str):
        return ""
    return model_id.strip().replace("/", ".")


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


def _is_model_glob(value: str) -> bool:
    return any(ch in value for ch in "*?[")


# Anthropic Reasoning Helpers

_CLAUDE_REASONING_RE = re.compile(r"~?anthropic[./]claude-(opus|sonnet)-")


def _is_claude_reasoning_model(normalized_model_id: str) -> bool:
    """Return True for Claude Opus/Sonnet models that support verbosity mapping."""
    return bool(_CLAUDE_REASONING_RE.match((normalized_model_id or "").lower()))


def _supports_verbosity(normalized_model_id: str) -> bool:
    normalized = normalized_model_id or ""
    if not _is_claude_reasoning_model(normalized):
        return False
    if not ModelFamily.catalog_spec(normalized):
        return False
    supported = ModelFamily.catalog_supported_parameters(normalized)
    return not supported or "verbosity" in supported


# Gemini Reasoning Helpers

_GEMINI_25_RE = re.compile(r"~?google[./]gemini-2\.5(-|\Z)")
_GEMINI_FAMILY_RE = re.compile(r"~?google[./]gemini-[\d.]")


def _classify_gemini_thinking_family(normalized_model_id: str) -> str | None:
    """Return the Gemini thinking family for the provided normalized model id."""
    if _GEMINI_25_RE.match((normalized_model_id or "").lower()):
        return "gemini-2.5"
    return None


def _is_gemini_thinking_model(normalized_model_id: str) -> bool:
    return bool(_GEMINI_FAMILY_RE.match((normalized_model_id or "").lower()))


def _map_effort_to_gemini_budget(effort: str, base_budget: int) -> int | None:
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


# Pricing Helpers

_PRICING_CATEGORY_KEYS = {
    "prompt",
    "completion",
    "request",
    "image",
    "image_token",
    "image_output",
    "audio",
    "audio_output",
    "input_audio_cache",
    "web_search",
    "internal_reasoning",
    "input_cache_read",
    "input_cache_write",
}
_PRICING_VALUE_KEYS = ("price", "amount", "value", "usd", "cost", "rate")


def sum_pricing_values(node: Any) -> tuple[Decimal, int]:
    """Return (sum, count) of numeric values found in a pricing node.

    Recursively walks pricing structures from OpenRouter model specs,
    summing all numeric values while ignoring discount fields.

    Args:
        node: Pricing data structure (dict, list, or scalar)

    Returns:
        Tuple of (total sum as Decimal, count of numeric values found)
    """
    total = Decimal(0)
    count = 0

    if isinstance(node, dict):
        keys = set(node.keys())
        if keys & _PRICING_CATEGORY_KEYS:
            for key, value in node.items():
                if key == "discount":
                    continue
                child_total, child_count = sum_pricing_values(value)
                total += child_total
                count += child_count
            return total, count

        found_value_key = False
        for key in _PRICING_VALUE_KEYS:
            if key in node:
                found_value_key = True
                child_total, child_count = sum_pricing_values(node[key])
                total += child_total
                count += child_count
        if found_value_key:
            return total, count

        for key, value in node.items():
            if key == "discount":
                continue
            child_total, child_count = sum_pricing_values(value)
            total += child_total
            count += child_count
        return total, count

    if isinstance(node, (list, tuple, set)):
        for value in node:
            child_total, child_count = sum_pricing_values(value)
            total += child_total
            count += child_count
        return total, count

    if isinstance(node, (int, float, Decimal)):
        try:
            return Decimal(str(node)), 1
        except (InvalidOperation, ValueError):
            return Decimal(0), 0

    if isinstance(node, str):
        raw = node.strip()
        if not raw:
            return Decimal(0), 0
        try:
            return Decimal(raw), 1
        except (InvalidOperation, ValueError):
            return Decimal(0), 0

    return Decimal(0), 0


def spec_derived_flags(pricing: Any, *, spec: Any = None) -> dict[str, bool]:
    if isinstance(spec, dict) and isinstance(spec.get("video_model"), dict):
        return {"free": False}
    total, numeric_count = sum_pricing_values(pricing)
    return {"free": numeric_count > 0 and total == Decimal(0)}


def _base_spec_fields(pricing: Any, *, spec: Any = None) -> dict[str, Any]:
    return {
        "pricing": pricing,
        **spec_derived_flags(pricing, spec=spec),
    }


def _is_free_spec(spec: Any) -> bool:
    pricing = spec.get("pricing") or {} if isinstance(spec, dict) else {}
    return spec_derived_flags(pricing, spec=spec)["free"]


def is_free_model(model_norm_id: str) -> bool:
    """Check if a model has free pricing (all pricing values sum to 0).

    Args:
        model_norm_id: Normalized model ID (e.g., "openai/gpt-4o")

    Returns:
        True if model exists and all pricing values sum to zero
    """
    spec = OpenRouterModelRegistry.spec(model_norm_id)
    derived = spec.get("free")
    if isinstance(derived, bool):
        return derived
    return _is_free_spec(spec)


def supports_tool_calling(model_norm_id: str) -> bool:
    """Check if model supports tool calling by checking supported parameters.

    Args:
        model_norm_id: Normalized model ID (e.g., "openai/gpt-4o")

    Returns:
        True if model supports 'tools' or 'tool_choice' parameters
    """
    return bool({"tools", "tool_choice"} & set(ModelFamily.supported_parameters(model_norm_id)))
