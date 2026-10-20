from __future__ import annotations

import asyncio
import threading
import time
import weakref
from typing import Any

import aiohttp

from ..core.warn_latch import warn_level
from ..models.registry import OpenRouterModelRegistry, _contract_target, _fingerprint
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

_CONTRACT_VALVES = (
    "AUTO_INSTALL_IMAGE_FILTERS",
    "AUTO_ATTACH_IMAGE_FILTERS",
    "AUTO_INSTALL_IMAGE_GEN_FILTER",
    "AUTO_ATTACH_IMAGE_GEN_FILTER",
)


def _wants_contracts(valves: Any) -> bool:
    return any(getattr(valves, name, False) for name in _CONTRACT_VALVES)


def _image_catalog_stale(
    *,
    cache_seconds: int,
    wants_filters: bool,
    api_key: str,
    valves: Any,
    target: tuple[str, str],
) -> tuple[bool, bool]:
    account_changed = not OpenRouterModelRegistry.image_accounts_match(api_key, target)
    last_attempt = OpenRouterModelRegistry.last_image_attempt()
    window = OpenRouterModelRegistry._media_retry_window(cache_seconds, _fingerprint(api_key))
    stale_models = (
        account_changed or not last_attempt or (time.time() - last_attempt) >= window
    )
    contract_attempt = OpenRouterModelRegistry.last_image_contract_attempt(target)
    stale_contracts = wants_filters and (
        account_changed
        or OpenRouterModelRegistry.image_contract_retry_pending()
        or not contract_attempt
        or (time.time() - contract_attempt) >= cache_seconds
    )
    return stale_models, stale_contracts


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
    wants_filters = with_contracts and _wants_contracts(valves)
    picker = bool(getattr(valves, "ENABLE_OPENROUTER_IMAGE_GENERATION", False))
    tool_only = wants_filters and not picker
    target = _contract_target(valves)

    while True:
        if wants_filters and _image_contract_sweep_in_progress():
            async with _current_image_contract_lock():
                pass
            continue

        if picker:
            if OpenRouterModelRegistry.adopt_image_contract_target(target):
                logger.info(
                    "Image contract cache rebound: the base URL or the API key changed, "
                    "and this identity has published no contracts of its own yet."
                )
            stale_models, stale_contracts = _image_catalog_stale(
                cache_seconds=cache_seconds, wants_filters=wants_filters, api_key=api_key,
                valves=valves, target=target,
            )
            if not stale_models and not stale_contracts:
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
                if not picker:
                    if OpenRouterModelRegistry.last_image_fetch() > 0:
                        OpenRouterModelRegistry.register_image_models([])
                        OpenRouterModelRegistry.set_image_endpoints(
                            {}, known_ids=OpenRouterModelRegistry.listed_model_ids(),
                            target=target,
                        )
                        OpenRouterModelRegistry.reset_image_fetch_timestamp()
                        OpenRouterModelRegistry.reset_image_attempt(target)
                        OpenRouterModelRegistry.clear_image_contract_attempt(target)
                        OpenRouterModelRegistry.clear_image_contract_owed(target)
                        logger.info(
                            "Image catalog cleared: ENABLE_OPENROUTER_IMAGE_GENERATION is False."
                        )
                    else:
                        logger.debug(
                            "Image catalog skipped: ENABLE_OPENROUTER_IMAGE_GENERATION is False."
                        )
                    if not tool_only:
                        return []
                    if not wait_for_in_flight and _image_sweep_in_flight():
                        logger.debug(
                            "Image catalog sweep already in flight; the caller queued behind it "
                            "and is answering from the catalogue already in the registry."
                        )
                        return []
                else:
                    stale_models, stale_contracts = _image_catalog_stale(
                        cache_seconds=cache_seconds, wants_filters=wants_filters, api_key=api_key,
                        valves=valves, target=target,
                    )
                    if not stale_models and not stale_contracts:
                        return []

                if not wait_for_in_flight and _image_sweep_in_flight():
                    logger.debug(
                        "Image catalog sweep already in flight; the caller queued behind it "
                        "and is answering from the catalogue already in the registry."
                    )
                    return []

                if wants_filters and _image_contract_sweep_in_progress():
                    continue

                account_changed = not OpenRouterModelRegistry.image_accounts_match(
                    api_key, target
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
                    register=picker,
                    target=target,
                )

            async with _current_image_contract_lock():
                if not fetched or not (
                    (wants_filters and (repair or account_changed))
                    or _image_catalog_stale(
                        cache_seconds=cache_seconds, wants_filters=wants_filters, api_key=api_key,
                        valves=valves, target=target,
                    )[1]
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
                    target=target,
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
    register: bool = True,
    target: tuple[str, str] | None = None,
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
        OpenRouterModelRegistry.record_media_failure(api_key)
        OpenRouterModelRegistry.record_image_attempt(api_key, target)
        if wants_filters:
            OpenRouterModelRegistry.record_image_contract_attempt(api_key, target)
        logger.log(
            warn_level(_warned_image_catalog, type(exc).__name__),
            "Image catalog fetch failed (/models?output_modalities=image): %s — chat catalog kept, image-only models will not appear.",
            exc,
        )
        return []

    if not models:
        OpenRouterModelRegistry.record_image_attempt(api_key, target)
        if wants_filters:
            OpenRouterModelRegistry.record_image_contract_attempt(api_key, target)
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

    OpenRouterModelRegistry.record_media_success(api_key)
    if register:
        OpenRouterModelRegistry.register_image_models(models)
        OpenRouterModelRegistry.record_image_attempt(api_key, target)
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
    target: tuple[str, str] | None = None,
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
        owed = OpenRouterModelRegistry.image_contract_owed(target)
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
            target=target,
        )
        OpenRouterModelRegistry.record_image_contract_attempt(api_key, target)
        if abandoned:
            OpenRouterModelRegistry.set_image_contract_owed(abandoned, target)
            if not repair:
                OpenRouterModelRegistry.clear_image_contract_attempt(target)
                OpenRouterModelRegistry.mark_image_contract_retry(cache_seconds)
        else:
            OpenRouterModelRegistry.clear_image_contract_owed(target)
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
    no_contract: list[str] = []
    not_found: list[str] = []
    completed: set[str] = set()

    async def _one(model_id: str) -> None:
        async with gate:
            try:
                published = await client.endpoints(model_id)
                offered = [item for item in published if isinstance(item, dict)]
                completed.add(model_id)
            except aiohttp.ClientResponseError as exc:
                if exc.status == 404:
                    completed.add(model_id)
                    not_found.append(model_id)
                    return
                logger.debug(
                    "Published contract read failed for %r: %s", model_id, exc, exc_info=True
                )
                failures.append(model_id)
                return
            except Exception as exc:
                logger.debug(
                    "Published contract read failed for %r: %s", model_id, exc, exc_info=True
                )
                failures.append(model_id)
                return
        if offered:
            records[model_id] = offered
        else:
            no_contract.append(model_id)

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

    if records and not_found:
        no_contract.extend(not_found)
    elif not_found:
        failures.extend(not_found)

    if no_contract:
        logger.debug(
            "No published knob contract for %d of %d image model(s) (%s): the endpoints "
            "route answered and the answer carries none, so their filters offer no knobs "
            "this cycle; the next refresh rechecks.",
            len(no_contract),
            len(ids),
            ", ".join(sorted(no_contract)[:5]),
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
