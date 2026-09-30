"""Image-output model catalog integration.

Mirror of `video_catalog.py` — TTL-gated fetch of the image-output model list,
registers via `OpenRouterModelRegistry.register_image_models`. Multimodal
text+image models in the response are deduplicated by the registry (they
already live in the chat catalog).
"""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from typing import Any

import aiohttp

from ..core.warn_latch import warn_level
from ..models.registry import OpenRouterModelRegistry, _contract_target
from .catalog_client import _build_catalog_client
from .image_client import OpenRouterImageClient

_warned_image_catalog: set[str] = set()

_image_catalog_lock = asyncio.Lock()
_image_contract_lock = asyncio.Lock()
_image_catalog_lock_guard = threading.Lock()
_image_catalog_locks: weakref.WeakKeyDictionary[Any, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)
_image_contract_locks: weakref.WeakKeyDictionary[Any, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)
_image_sweeps_in_flight: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()


def _current_image_catalog_lock() -> asyncio.Lock:
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return _image_catalog_lock
    existing = _image_catalog_locks.get(running)
    if existing is not None:
        return existing
    with _image_catalog_lock_guard:
        existing = _image_catalog_locks.get(running)
        if existing is None:
            existing = asyncio.Lock()
            _image_catalog_locks[running] = existing
    return existing


def _current_image_contract_lock() -> asyncio.Lock:
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return _image_contract_lock
    existing = _image_contract_locks.get(running)
    if existing is not None:
        return existing
    with _image_catalog_lock_guard:
        existing = _image_contract_locks.get(running)
        if existing is None:
            existing = asyncio.Lock()
            _image_contract_locks[running] = existing
    return existing


def _image_sweep_in_flight() -> bool:
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return False
    with _image_catalog_lock_guard:
        return bool(_image_sweeps_in_flight.get(running))


def _set_image_sweep_in_flight(running: asyncio.AbstractEventLoop, value: bool) -> None:
    with _image_catalog_lock_guard:
        if value:
            _image_sweeps_in_flight[running] = True
        else:
            _image_sweeps_in_flight.pop(running, None)


_SWEEP_BUDGET_SECONDS = 45
"""How long the whole published-contract sweep may take.

The per-read cap bounds one request; this bounds the wait a user experiences, so the
number of image models does not appear in the time the model picker takes to appear.
"""

_CONTRACT_VALVES = (
    "AUTO_INSTALL_IMAGE_FILTERS",
    "AUTO_ATTACH_IMAGE_FILTERS",
    "AUTO_INSTALL_IMAGE_GEN_FILTER",
    "AUTO_ATTACH_IMAGE_GEN_FILTER",
)


def _wants_contracts(valves: Any) -> bool:
    return any(getattr(valves, name, False) for name in _CONTRACT_VALVES)


def _stale_contracts(wants_filters: bool, cache_seconds: int, valves: Any) -> bool:
    contract_attempt = OpenRouterModelRegistry.last_image_contract_attempt()
    return wants_filters and (
        OpenRouterModelRegistry.image_contract_retry_pending()
        or not contract_attempt
        or (time.time() - contract_attempt) >= cache_seconds
        or OpenRouterModelRegistry.image_contract_target() != _contract_target(valves)
    )


def _image_contract_sweep_in_progress() -> bool:
    return _current_image_contract_lock().locked()


async def ensure_image_catalog_loaded(
    session: aiohttp.ClientSession,
    *,
    valves: Any,
    api_key: str,
    logger: Any,
    cache_seconds: int,
    with_contracts: bool = True,
    wait_for_in_flight: bool = True,
) -> list[dict[str, Any]]:
    """Fetch image-output models and register them into the shared model registry.

    ``with_contracts`` reads each model's published knob contract -- one request per
    model. Only the filter installer consumes those, and it runs on the catalog path, so
    the request path passes False rather than making a user's message wait on contract
    reads for every model it will not call.
    """
    wants_filters = with_contracts and _wants_contracts(valves)

    while True:
        if wants_filters and _image_contract_sweep_in_progress():
            async with _current_image_contract_lock():
                pass
            continue

        if getattr(valves, "ENABLE_OPENROUTER_IMAGE_GENERATION", False):
            last_attempt = OpenRouterModelRegistry.last_image_attempt()
            stale_models = not last_attempt or (time.time() - last_attempt) >= cache_seconds
            if not stale_models and not _stale_contracts(wants_filters, cache_seconds, valves):
                return []
            if not wait_for_in_flight and _image_sweep_in_flight():
                logger.debug(
                    "Image catalog sweep already in flight on this loop; answering from the "
                    "catalogue already in the registry."
                )
                return []

        swept = False
        try:
            async with _current_image_catalog_lock():
                if not getattr(valves, "ENABLE_OPENROUTER_IMAGE_GENERATION", False):
                    if OpenRouterModelRegistry.last_image_fetch() > 0:
                        OpenRouterModelRegistry.register_image_models([])
                        OpenRouterModelRegistry.reset_image_fetch_timestamp()
                        OpenRouterModelRegistry.reset_image_attempt()
                        OpenRouterModelRegistry.clear_image_contract_attempt()
                        logger.info(
                            "Image catalog cleared: ENABLE_OPENROUTER_IMAGE_GENERATION is False."
                        )
                    else:
                        logger.debug(
                            "Image catalog skipped: ENABLE_OPENROUTER_IMAGE_GENERATION is False."
                        )
                    return []

                last_attempt = OpenRouterModelRegistry.last_image_attempt()
                stale_models = not last_attempt or (time.time() - last_attempt) >= cache_seconds
                if not stale_models and not _stale_contracts(wants_filters, cache_seconds, valves):
                    return []

                if not wait_for_in_flight and _image_sweep_in_flight():
                    logger.debug(
                        "Image catalog sweep already in flight; the caller queued behind it "
                        "and is answering from the catalogue already in the registry."
                    )
                    return []

                if wants_filters and _image_contract_sweep_in_progress():
                    continue

                if OpenRouterModelRegistry.adopt_image_contract_target(_contract_target(valves)):
                    logger.info(
                        "Image contract cache dropped: the base URL or the API key changed."
                    )

                repair = OpenRouterModelRegistry.image_contract_retry_pending()
                OpenRouterModelRegistry.clear_image_contract_retry()
                _set_image_sweep_in_flight(asyncio.get_running_loop(), True)
                swept = True
                fetched = await _refresh_image_models(
                    session,
                    valves=valves,
                    api_key=api_key,
                    logger=logger,
                    wants_filters=wants_filters,
                    cache_seconds=cache_seconds,
                )

            async with _current_image_contract_lock():
                if not fetched or not (
                    (wants_filters and repair)
                    or _stale_contracts(wants_filters, cache_seconds, valves)
                ):
                    return []
                await _sweep_image_contracts(
                    session,
                    valves=valves,
                    api_key=api_key,
                    logger=logger,
                    wants_filters=wants_filters,
                    cache_seconds=cache_seconds,
                    models=fetched,
                    repair=repair,
                )
            return fetched
        finally:
            if swept:
                _set_image_sweep_in_flight(asyncio.get_running_loop(), False)


async def _refresh_image_models(
    session: aiohttp.ClientSession,
    *,
    valves: Any,
    api_key: str,
    logger: Any,
    wants_filters: bool,
    cache_seconds: int,
) -> list[dict[str, Any]]:
    client = _build_catalog_client(
        OpenRouterImageClient,
        session,
        valves=valves,
        api_key=api_key,
        logger=logger,
    )

    try:
        models = await client.list_models()
    except (TimeoutError, aiohttp.ClientError, OSError) as exc:
        OpenRouterModelRegistry.record_image_attempt()
        if wants_filters:
            OpenRouterModelRegistry.record_image_contract_attempt()
        logger.log(
            warn_level(_warned_image_catalog, type(exc).__name__),
            "Image catalog fetch failed (/models?output_modalities=image): %s — chat catalog kept, image-only models will not appear.",
            exc,
        )
        return []

    if not models:
        OpenRouterModelRegistry.record_image_attempt()
        if wants_filters:
            OpenRouterModelRegistry.record_image_contract_attempt()
        kept = len(OpenRouterModelRegistry._image_catalog_norms)
        logger.log(
            warn_level(_warned_image_catalog, "empty"),
            "Image catalog fetch returned 0 models; keeping the %d image-only model(s) "
            "the last sweep published. A catalogue that genuinely holds nothing is not "
            "reconciled until the worker restarts or "
            "ENABLE_OPENROUTER_IMAGE_GENERATION is toggled off and on.",
            kept,
        )
        return []

    OpenRouterModelRegistry.register_image_models(models)
    OpenRouterModelRegistry.record_image_attempt()
    logger.info(
        "Registered %d OpenRouter image-output model(s) into the catalog.",
        len(models),
    )
    return models


async def _sweep_image_contracts(
    session: aiohttp.ClientSession,
    *,
    valves: Any,
    api_key: str,
    logger: Any,
    wants_filters: bool,
    cache_seconds: int,
    models: list[dict[str, Any]],
    repair: bool,
) -> None:
    client = _build_catalog_client(
        OpenRouterImageClient,
        session,
        valves=valves,
        api_key=api_key,
        logger=logger,
    )
    endpoint_records: dict[str, list[dict[str, Any]]] = {}
    abandoned: frozenset[str] = frozenset()
    if wants_filters:
        owed = OpenRouterModelRegistry.image_contract_owed()
        endpoint_records, abandoned = await _fetch_endpoint_records(
            client, models, logger, only=owed or None
        )
        OpenRouterModelRegistry.set_image_endpoints(
            endpoint_records,
            known_ids={
                str(model.get("id")).strip()
                for model in models
                if isinstance(model, dict) and str(model.get("id") or "").strip()
            },
        )
        OpenRouterModelRegistry.record_image_contract_attempt()
        if abandoned:
            OpenRouterModelRegistry.set_image_contract_owed(abandoned)
            if not repair:
                OpenRouterModelRegistry.clear_image_contract_attempt()
                OpenRouterModelRegistry.mark_image_contract_retry(cache_seconds)
        else:
            OpenRouterModelRegistry.clear_image_contract_owed()
    logger.info(
        "%d OpenRouter image-output model(s) published a usable knob contract.",
        len(endpoint_records),
    )


async def _fetch_endpoint_records(
    client: Any,
    models: list[dict[str, Any]],
    logger: Any,
    concurrency: int = 8,
    only: frozenset[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], frozenset[str]]:
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
        and (only is None or str(model.get("id")).strip() in only)
    ]
    if not ids:
        return {}, frozenset()

    records: dict[str, list[dict[str, Any]]] = {}
    gate = asyncio.Semaphore(max(1, concurrency))
    failures: list[str] = []
    completed: set[str] = set()

    async def _one(model_id: str) -> None:
        async with gate:
            try:
                published = await client.endpoints(model_id)
                # Consuming the response belongs inside the guard: a client returning
                # something that is not a list raises here, and outside the guard that
                # exception is swallowed by the gather below, so the model would vanish
                # from both the records and the failures with no diagnostic at all.
                offered = [item for item in published if isinstance(item, dict)]
                completed.add(model_id)
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
    abandoned: frozenset[str] = frozenset()
    try:
        async with asyncio.timeout(_SWEEP_BUDGET_SECONDS):
            await asyncio.gather(
                *(_one(model_id) for model_id in ids), return_exceptions=True
            )
    except TimeoutError:
        abandoned = frozenset(mid for mid in ids if mid not in completed)
        unread = list(abandoned)
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
    return records, abandoned
