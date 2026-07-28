"""Video model catalog integration."""

from __future__ import annotations

import time
from typing import Any

import aiohttp

from ..core.config import _select_openrouter_http_referer
from ..models.registry import OpenRouterModelRegistry
from .video_client import OpenRouterVideoClient


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
        logger.info("Video catalog skipped: ENABLE_VIDEO_GENERATION is False.")
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
        logger.warning(
            "Video catalog fetch failed (/videos/models): %s — chat catalog kept, video models will not appear.",
            exc,
        )
        return

    OpenRouterModelRegistry.record_video_attempt()

    if not models:
        logger.warning("Video catalog fetch returned 0 models; nothing to register.")
        return

    OpenRouterModelRegistry.register_video_models(models)
    logger.info("Registered %d OpenRouter video model(s) into the catalog.", len(models))
