"""Image-output model catalog integration.

Mirror of `video_catalog.py` — TTL-gated fetch of the image-output model list,
registers via `OpenRouterModelRegistry.register_image_models`. Multimodal
text+image models in the response are deduplicated by the registry (they
already live in the chat catalog).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp

from ..core.config import _select_openrouter_http_referer
from ..core.warn_latch import warn_level
from ..models.registry import OpenRouterModelRegistry
from .image_client import OpenRouterImageClient

_warned_image_catalog: set[str] = set()

_SWEEP_BUDGET_SECONDS = 45
"""How long the whole published-contract sweep may take.

The per-read cap bounds one request; this bounds the wait a user experiences, so the
number of image models does not appear in the time the model picker takes to appear.
"""


async def ensure_image_catalog_loaded(
    session: aiohttp.ClientSession,
    *,
    valves: Any,
    api_key: str,
    logger: Any,
    cache_seconds: int,
    with_contracts: bool = True,
) -> None:
    """Fetch image-output models and register them into the shared model registry.

    ``with_contracts`` reads each model's published knob contract -- one request per
    model. Only the filter installer consumes those, and it runs on the catalog path, so
    the request path passes False rather than making a user's message wait on contract
    reads for every model it will not call.
    """
    if not getattr(valves, "ENABLE_OPENROUTER_IMAGE_GENERATION", False):
        if OpenRouterModelRegistry.last_image_fetch() > 0:
            OpenRouterModelRegistry.register_image_models([])
            OpenRouterModelRegistry.reset_image_fetch_timestamp()
            logger.info("Image catalog cleared: ENABLE_OPENROUTER_IMAGE_GENERATION is False.")
        else:
            logger.debug("Image catalog skipped: ENABLE_OPENROUTER_IMAGE_GENERATION is False.")
        return

    last_attempt = OpenRouterModelRegistry.last_image_attempt()
    stale_models = not last_attempt or (time.time() - last_attempt) >= cache_seconds
    # Gated on whether this call would actually sweep. Without that, a deployment with
    # both filter valves off never stamps the contract clock, so the freshness check can
    # never be satisfied and every model-list build refetches the catalogue.
    wants_filters = with_contracts and bool(
        getattr(valves, "AUTO_INSTALL_IMAGE_FILTERS", False)
        or getattr(valves, "AUTO_ATTACH_IMAGE_FILTERS", False)
    )
    contract_attempt = OpenRouterModelRegistry.last_image_contract_attempt()
    stale_contracts = wants_filters and (
        not contract_attempt or (time.time() - contract_attempt) >= cache_seconds
    )
    if not stale_models and not stale_contracts:
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

    if not models:
        OpenRouterModelRegistry.record_image_attempt()
        logger.warning("Image catalog fetch returned 0 models; nothing to register.")
        return

    # One contract read per model, so only pay for them when something consumes them.
    # The consumer is the per-model filter install, which runs under the same two valves
    # that `catalog_manager` checks before calling it.
    endpoint_records: dict[str, list[dict[str, Any]]] = {}
    if wants_filters:
        OpenRouterModelRegistry.record_image_contract_attempt()
        endpoint_records = await _fetch_endpoint_records(client, models, logger)
        OpenRouterModelRegistry.set_image_endpoints(
            endpoint_records,
            known_ids={
                str(model.get("id")).strip()
                for model in models
                if isinstance(model, dict) and str(model.get("id") or "").strip()
            },
        )
    OpenRouterModelRegistry.register_image_models(models)
    # Stamped only once the models are registered. Stamping earlier meant a cancellation
    # mid-sweep left the models unregistered AND the retry suppressed for a whole TTL
    # window, so the image models simply vanished from the picker for an hour.
    OpenRouterModelRegistry.record_image_attempt()
    logger.info(
        "Registered %d OpenRouter image-output model(s) into the catalog; %d published a "
        "usable knob contract.",
        len(models),
        len(endpoint_records),
    )


async def _fetch_endpoint_records(
    client: Any,
    models: list[dict[str, Any]],
    logger: Any,
    concurrency: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """Read every published knob contract for each model, keyed by model id.

    A model served by more than one provider publishes one record per provider, and they
    do not agree. All of them are kept because the request is routed to a provider chosen
    later; keeping only the first would show knobs the serving provider may reject.

    A model whose contract cannot be read is absent from the result. The registry then
    keeps its last successful record, so only a model never read ends up with no knobs --
    which is the honest outcome, because the alternative is offering a guessed set, and
    that is what left a third of models advertising values they reject.
    """
    ids = [
        str(model.get("id")).strip()
        for model in models
        if isinstance(model, dict) and str(model.get("id") or "").strip()
    ]
    if not ids:
        return {}

    records: dict[str, list[dict[str, Any]]] = {}
    gate = asyncio.Semaphore(max(1, concurrency))
    failures: list[str] = []

    async def _one(model_id: str) -> None:
        async with gate:
            try:
                published = await client.endpoints(model_id)
                # Consuming the response belongs inside the guard: a client returning
                # something that is not a list raises here, and outside the guard that
                # exception is swallowed by the gather below, so the model would vanish
                # from both the records and the failures with no diagnostic at all.
                offered = [item for item in published if isinstance(item, dict)]
            except Exception as exc:
                # Broad on purpose: a model absent from the result must always be in
                # `failures`, so it is always named in the log. A tuple of enumerated
                # classes let anything unlisted -- an AttributeError, a driver error --
                # be dropped by the gather below with no diagnostic at all. The summary
                # below names at most a few models, so the reason lands here per model.
                logger.debug(
                    "Published contract read failed for %r: %s", model_id, exc, exc_info=True
                )
                failures.append(model_id)
                return
        if offered:
            records[model_id] = offered
        else:
            failures.append(model_id)

    # Bounded as a whole, not only per read. This runs inside the call that builds Open
    # WebUI's model list, and one read per model at this concurrency would otherwise put
    # the catalogue's size into the time a user waits for the picker.
    try:
        async with asyncio.timeout(_SWEEP_BUDGET_SECONDS):
            await asyncio.gather(
                *(_one(model_id) for model_id in ids), return_exceptions=True
            )
    except TimeoutError:
        unread = [model_id for model_id in ids if model_id not in records]
        for model_id in unread:
            if model_id not in failures:
                failures.append(model_id)
        logger.warning(
            "The published-contract sweep ran past %ds with %d of %d model(s) still "
            "unread; those keep whatever settings they already had and the next refresh "
            "retries.",
            _SWEEP_BUDGET_SECONDS,
            len(unread),
            len(ids),
        )

    if failures:
        logger.log(
            warn_level(_warned_image_catalog, f"endpoints:{len(failures)}/{len(ids)}"),
            "Could not read the published knob contract for %d of %d image model(s) (%s). "
            "Any that were read before keep those settings; one never read offers none. "
            "The next refresh retries.",
            len(failures),
            len(ids),
            ", ".join(sorted(failures)[:5]),
        )
    return records
