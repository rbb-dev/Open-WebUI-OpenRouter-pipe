"""Video model catalog integration."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp

from ..core.config import _select_openrouter_http_referer
from ..core.warn_latch import warn_level
from ..models.registry import OpenRouterModelRegistry
from .video_client import OpenRouterVideoClient

_warned_video_catalog: set[str] = set()

_MODALITY_FETCH_CONCURRENCY = 6


async def ensure_video_catalog_loaded(
    session: aiohttp.ClientSession,
    *,
    valves: Any,
    api_key: str,
    logger: Any,
    cache_seconds: int,
) -> None:
    """Fetch video models and register them into the shared model registry."""
    if not getattr(valves, "ENABLE_VIDEO_GENERATION", False):
        if OpenRouterModelRegistry.last_video_fetch() > 0:
            OpenRouterModelRegistry.register_video_models([])
            OpenRouterModelRegistry.reset_video_fetch_timestamp()
            logger.info("Video catalog cleared: ENABLE_VIDEO_GENERATION is False.")
        else:
            logger.debug("Video catalog skipped: ENABLE_VIDEO_GENERATION is False.")
        return

    last_attempt = OpenRouterModelRegistry.last_video_attempt()
    if last_attempt and (time.time() - last_attempt) < cache_seconds:
        return

    client = OpenRouterVideoClient(
        session,
        base_url=valves.BASE_URL,
        api_key=api_key,
        logger=logger,
        http_referer=_select_openrouter_http_referer(valves),
    )

    try:
        models = await client.list_models()
    except (TimeoutError, aiohttp.ClientError, OSError) as exc:
        OpenRouterModelRegistry.record_video_attempt()
        logger.log(
            warn_level(_warned_video_catalog, type(exc).__name__),
            "Video catalog fetch failed (/videos/models): %s — chat catalog kept, video models will not appear.",
            exc,
        )
        return

    OpenRouterModelRegistry.record_video_attempt()

    if not models:
        logger.warning("Video catalog fetch returned 0 models; nothing to register.")
        return

    await _attach_declared_input_modalities(client, models, logger)

    OpenRouterModelRegistry.register_video_models(models)
    logger.info("Registered %d OpenRouter video model(s) into the catalog.", len(models))


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

    await asyncio.gather(*(_one(model) for model in wanted), return_exceptions=True)
    known = sum(1 for m in wanted if m.get("input_modalities"))
    if known < len(wanted):
        logger.log(
            warn_level(_warned_video_catalog, "modalities"),
            "Read the accepted input kinds for %d of %d video model(s); the rest are offered "
            "every reference control until it can be read again.",
            known,
            len(wanted),
        )
