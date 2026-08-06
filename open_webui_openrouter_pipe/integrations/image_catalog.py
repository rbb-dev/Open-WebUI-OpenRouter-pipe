"""Image-output model catalog integration.

Mirror of `video_catalog.py` — TTL-gated fetch of the image-output model list,
registers via `OpenRouterModelRegistry.register_image_models`. Multimodal
text+image models in the response are deduplicated by the registry (they
already live in the chat catalog).
"""

from __future__ import annotations

import time
from typing import Any

import aiohttp

from ..core.config import _select_openrouter_http_referer
from ..core.warn_latch import warn_level
from ..models.registry import OpenRouterModelRegistry
from .image_client import OpenRouterImageClient

_warned_image_catalog: set[str] = set()


async def ensure_image_catalog_loaded(
    session: aiohttp.ClientSession,
    *,
    valves: Any,
    api_key: str,
    logger: Any,
    cache_seconds: int,
) -> None:
    """Fetch image-output models and register them into the shared model registry."""
    if not getattr(valves, "ENABLE_OPENROUTER_IMAGE_GENERATION", False):
        # Master-disable: drop any previously-registered pure-image-only models
        # so they vanish from OWUI's dropdown immediately. `register_image_models([])`
        # triggers stale-norm cleanup without re-registering anything. Reset
        # `_last_image_fetch` so subsequent disabled-state calls are no-ops.
        if OpenRouterModelRegistry.last_image_fetch() > 0:
            OpenRouterModelRegistry.register_image_models([])
            OpenRouterModelRegistry.reset_image_fetch_timestamp()
            logger.info("Image catalog cleared: ENABLE_OPENROUTER_IMAGE_GENERATION is False.")
        else:
            logger.debug("Image catalog skipped: ENABLE_OPENROUTER_IMAGE_GENERATION is False.")
        return

    last_attempt = OpenRouterModelRegistry.last_image_attempt()
    if last_attempt and (time.time() - last_attempt) < cache_seconds:
        return

    client = OpenRouterImageClient(
        session,
        base_url=valves.BASE_URL,
        api_key=api_key,
        logger=logger,
        http_referer=_select_openrouter_http_referer(valves),
    )

    try:
        models = await client.list_models()
    except (TimeoutError, aiohttp.ClientError, OSError) as exc:
        OpenRouterModelRegistry.record_image_attempt()
        logger.log(
            warn_level(_warned_image_catalog, type(exc).__name__),
            "Image catalog fetch failed (/models?output_modalities=image): %s — chat catalog kept, image-only models will not appear.",
            exc,
        )
        return

    OpenRouterModelRegistry.record_image_attempt()

    if not models:
        logger.warning("Image catalog fetch returned 0 models; nothing to register.")
        return

    OpenRouterModelRegistry.register_image_models(models)
    logger.info("Registered %d OpenRouter image-output model(s) into the catalog.", len(models))
