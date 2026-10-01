"""Video model catalog integration."""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from typing import Any

import aiohttp

from ..core.warn_latch import warn_level
from ..models.registry import OpenRouterModelRegistry
from .catalog_client import _build_catalog_client
from .video_client import OpenRouterVideoClient

_warned_video_catalog: set[str] = set()

_MODALITY_FETCH_CONCURRENCY = 6

_VIDEO_SWEEP_BUDGET_SECONDS = 45

_VIDEO_CATALOG_LOCK = asyncio.Lock()
_video_catalog_lock_guard = threading.Lock()
_video_catalog_locks: weakref.WeakKeyDictionary[Any, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)
_video_sweeps_in_flight: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()


def _current_video_catalog_lock() -> asyncio.Lock:
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return _VIDEO_CATALOG_LOCK
    existing = _video_catalog_locks.get(running)
    if existing is not None:
        return existing
    with _video_catalog_lock_guard:
        existing = _video_catalog_locks.get(running)
        if existing is None:
            existing = asyncio.Lock()
            _video_catalog_locks[running] = existing
    return existing


def _video_sweep_in_flight() -> bool:
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return False
    with _video_catalog_lock_guard:
        return bool(_video_sweeps_in_flight.get(running))


def _set_video_sweep_in_flight(running: asyncio.AbstractEventLoop, value: bool) -> None:
    with _video_catalog_lock_guard:
        if value:
            _video_sweeps_in_flight[running] = True
        else:
            _video_sweeps_in_flight.pop(running, None)


def _video_catalog_stale(
    cache_seconds: int, wants_modalities: bool, api_key: str,
) -> tuple[bool, bool]:
    now = time.time()
    last_attempt = OpenRouterModelRegistry.last_video_attempt()
    stale_list = (
        not OpenRouterModelRegistry.video_accounts_match(api_key)
        or not last_attempt
        or (now - last_attempt) >= cache_seconds
    )
    if not wants_modalities:
        return stale_list, False
    last_modalities = OpenRouterModelRegistry.last_video_modality_attempt()
    stale_modalities = (
        not last_modalities or (now - last_modalities) >= cache_seconds
    )
    return stale_list, stale_modalities


async def ensure_video_catalog_loaded(
    session: aiohttp.ClientSession,
    *,
    valves: Any,
    api_key: str,
    logger: Any,
    cache_seconds: int,
    with_modalities: bool = True,
    wait_for_in_flight: bool = True,
) -> None:
    """Fetch video models and register them into the shared model registry."""
    if getattr(valves, "ENABLE_VIDEO_GENERATION", False):
        stale_list, stale_modalities = _video_catalog_stale(
            cache_seconds, with_modalities, api_key
        )
        if not stale_list and not stale_modalities:
            return
        if not wait_for_in_flight and _video_sweep_in_flight():
            logger.debug(
                "Video modality sweep already in flight on this loop; answering from the "
                "catalogue already in the registry."
            )
            return

    async with _current_video_catalog_lock():
        if not getattr(valves, "ENABLE_VIDEO_GENERATION", False):
            if OpenRouterModelRegistry.last_video_fetch() > 0:
                OpenRouterModelRegistry.register_video_models([])
                OpenRouterModelRegistry.reset_video_fetch_timestamp()
                OpenRouterModelRegistry.reset_video_attempt()
                OpenRouterModelRegistry.reset_video_modality_attempt()
                logger.info("Video catalog cleared: ENABLE_VIDEO_GENERATION is False.")
            else:
                logger.debug("Video catalog skipped: ENABLE_VIDEO_GENERATION is False.")
            return

        stale_list, stale_modalities = _video_catalog_stale(
            cache_seconds, with_modalities, api_key
        )
        if not stale_list and not stale_modalities:
            return
        if not wait_for_in_flight and _video_sweep_in_flight():
            logger.debug(
                "Video modality sweep already in flight; the caller queued behind it and "
                "is answering from the catalogue already in the registry."
            )
            return

        _set_video_sweep_in_flight(asyncio.get_running_loop(), True)
        try:
            client = _build_catalog_client(
                OpenRouterVideoClient,
                session,
                valves=valves,
                api_key=api_key,
                logger=logger,
            )

            if stale_modalities and not stale_list:
                await _sweep_declared_input_modalities(client, logger)
                OpenRouterModelRegistry.record_video_modality_attempt()
                return

            try:
                models = await client.list_models()
            except (TimeoutError, aiohttp.ClientError, OSError) as exc:
                OpenRouterModelRegistry.record_video_attempt(api_key)
                logger.log(
                    warn_level(_warned_video_catalog, type(exc).__name__),
                    "Video catalog fetch failed (/videos/models): %s — chat catalog kept, video models will not appear.",
                    exc,
                )
                return

            if not models:
                OpenRouterModelRegistry.record_video_attempt(api_key)
                kept = len(OpenRouterModelRegistry._video_catalog_norms)
                logger.log(
                    warn_level(_warned_video_catalog, "empty"),
                    "Video catalog fetch returned 0 models; keeping the %d video model(s) the "
                    "last refresh published. A catalogue that genuinely holds nothing is not "
                    "reconciled until the worker restarts or ENABLE_VIDEO_GENERATION is "
                    "toggled off and on.",
                    kept,
                )
                return

            if with_modalities:
                await _attach_declared_input_modalities(client, models, logger)
            else:
                _carry_declared_input_modalities(models)

            OpenRouterModelRegistry.register_video_models(models)
            OpenRouterModelRegistry.record_video_attempt(api_key)
            if with_modalities:
                OpenRouterModelRegistry.record_video_modality_attempt()
            logger.info(
                "Registered %d OpenRouter video model(s) into the catalog.", len(models)
            )
        finally:
            _set_video_sweep_in_flight(asyncio.get_running_loop(), False)


async def _sweep_declared_input_modalities(
    client: OpenRouterVideoClient, logger: Any,
) -> None:
    ids = sorted(OpenRouterModelRegistry._video_catalog_norms)
    if not ids:
        return
    known = 0
    for norm_id in ids:
        spec = OpenRouterModelRegistry.spec(norm_id)
        video_model = spec.get("video_model")
        if not isinstance(video_model, dict):
            continue
        wire_id = str(video_model.get("id") or "").strip()
        if not wire_id:
            continue
        try:
            found = await client.model_modalities(wire_id)
        except (TimeoutError, aiohttp.ClientError, OSError):
            continue
        if not found:
            continue
        video_model["input_modalities"] = list(found)
        known += 1
    if known < len(ids):
        logger.log(
            warn_level(_warned_video_catalog, "modalities"),
            "Read the accepted input kinds for %d of %d video model(s); the rest are offered "
            "every reference control until it can be read again.",
            known,
            len(ids),
        )


def _carry_declared_input_modalities(models: list[dict[str, Any]]) -> None:
    for item in models:
        if not isinstance(item, dict) or item.get("input_modalities"):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        previous = OpenRouterModelRegistry.spec(model_id).get("video_model") or {}
        carried = previous.get("input_modalities") if isinstance(previous, dict) else None
        if isinstance(carried, list) and carried:
            item["input_modalities"] = list(carried)


async def _attach_declared_input_modalities(
    client: OpenRouterVideoClient,
    models: list[dict[str, Any]],
    logger: Any,
) -> None:
    wanted = [m for m in models if isinstance(m, dict) and isinstance(m.get("id"), str)]
    if not wanted:
        return
    gate = asyncio.Semaphore(_MODALITY_FETCH_CONCURRENCY)

    async def _one(model: dict[str, Any]) -> None:
        async with gate:
            found = await client.model_modalities(str(model["id"]))
        if found:
            model["input_modalities"] = found

    try:
        async with asyncio.timeout(_VIDEO_SWEEP_BUDGET_SECONDS):
            await asyncio.gather(*(_one(model) for model in wanted), return_exceptions=True)
    except TimeoutError:
        unread = sum(1 for m in wanted if not m.get("input_modalities"))
        logger.warning(
            "The video modality sweep ran past %ds with %d of %d model(s) still unread; "
            "those are offered every reference control until a later refresh reads them.",
            _VIDEO_SWEEP_BUDGET_SECONDS, unread, len(wanted),
        )
    known = sum(1 for m in wanted if m.get("input_modalities"))
    if known < len(wanted):
        logger.log(
            warn_level(_warned_video_catalog, "modalities"),
            "Read the accepted input kinds for %d of %d video model(s); the rest are offered "
            "every reference control until it can be read again.",
            known,
            len(wanted),
        )
